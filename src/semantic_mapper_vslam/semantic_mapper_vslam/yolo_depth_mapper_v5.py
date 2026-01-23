#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5 + TensorRT Depth Mapper (RTAB-Map 연동)

주요 특징:
- YOLOv5 DetectMultiBackend를 사용해 TensorRT 엔진/pt/onnx 자동 감지
- 커스텀 데이터셋 클래스 이름을 data.yaml에서 읽어 사용
- Depth/TF 없을 때도 2D 디버그 박스 표기
"""

import os
import sys
import time
import math
import pathlib
from typing import List, Tuple, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, Pose
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration
from rclpy.time import Time

import numpy as np
import cv2
import torch
import yaml
import colorsys

# ===== YOLOv5 경로 설정 =====
YOLOV5_PATH = os.environ.get("YOLOV5_PATH", os.path.expanduser("~/learn/yolov5"))
if os.path.isdir(YOLOV5_PATH) and YOLOV5_PATH not in sys.path:
    sys.path.insert(0, YOLOV5_PATH)

# YOLOv5 모듈 임포트 (DetectMultiBackend 기반)
try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    from utils.torch_utils import select_device
except Exception as e:
    raise ImportError(
        f"Failed to import YOLOv5 modules from {YOLOV5_PATH}. "
        f"Set YOLOV5_PATH env or check repo checkout. Error: {e}"
    )

from semantic_mapper_msgs.msg import DetectionArray, Detection


class YoloDepthMapperV5(Node):
    """
    YOLOv5 TensorRT + Depth 융합 노드
    """

    def __init__(self):
        super().__init__('yolo_depth_mapper_v5')
        self.bridge = CvBridge()

        # ===== 파라미터 선언 =====
        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("world_frame", "map")

        self.declare_parameter("model_path", "best.engine")
        self.declare_parameter("data_yaml", "")
        self.declare_parameter("conf_threshold", 0.35)
        self.declare_parameter("iou_threshold", 0.50)
        self.declare_parameter("input_size", 640)
        self.declare_parameter("device", "cuda:0")

        # 허용 클래스(빈 리스트면 전부)
        self.declare_parameter(
            "allow_classes",
            [],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        )
        # 클래스 이름을 명시적으로 전달할 수 있는 옵션
        self.declare_parameter(
            "class_names",
            [],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        )
        self.declare_parameter("names_path", "")

        self.declare_parameter("debug_topic", "/yolo/debug_image")
        self.declare_parameter("marker_topic", "/yolo/marker_array")
        self.declare_parameter("detection_topic", "/semantic_mapper/detections")

        self.declare_parameter("min_box_px", 30)
        self.declare_parameter("core_roi", 0.60)
        # 처리 FPS 상한 (동적 타이핑으로 int/float 모두 허용)
        # D455 카메라는 보통 15-30fps이므로 20fps로 제한하여 안정성 확보
        self.declare_parameter(
            "max_fps",
            20.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, dynamic_typing=True)
        )

        self.declare_parameter("persist_k", 3)
        self.declare_parameter("persist_t", 10)
        self.declare_parameter("ema_alpha", 0.4)  # 0.15 → 0.4 (더 빠른 반응)
        self.declare_parameter("bbox_deadzone_px", 3)  # 5 → 3 (더 민감하게)
        self.declare_parameter("match_dist_m", 0.40)

        self.declare_parameter("tf_use_latest", True)
        self.declare_parameter("tf_timeout", 0.25)

        # ===== 파라미터 읽기 =====
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.world_frame = self.get_parameter("world_frame").value

        self.model_path = self.get_parameter("model_path").value
        self.data_yaml = self.get_parameter("data_yaml").value
        self.conf = float(self.get_parameter("conf_threshold").value)
        self.iou = float(self.get_parameter("iou_threshold").value)
        self.input_size = int(self.get_parameter("input_size").value)
        self.device_id = self.get_parameter("device").value

        allow_param = self.get_parameter("allow_classes").value
        self.allow_classes = list(allow_param) if allow_param else []
        self.names_path = self.get_parameter("names_path").value
        param_class_names = self.get_parameter("class_names").value
        self.param_class_names = [str(x) for x in param_class_names] if param_class_names else []

        self.debug_topic = self.get_parameter("debug_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.detection_topic = self.get_parameter("detection_topic").value

        self.min_box_px = int(self.get_parameter("min_box_px").value)
        self.core_roi = float(self.get_parameter("core_roi").value)
        self.max_fps = float(self.get_parameter("max_fps").value)
        self.max_period = 1.0 / max(self.max_fps, 1.0)

        self.persist_k = int(self.get_parameter("persist_k").value)
        self.persist_t = int(self.get_parameter("persist_t").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.bbox_deadzone_px = int(self.get_parameter("bbox_deadzone_px").value)
        self.match_dist_m = float(self.get_parameter("match_dist_m").value)

        self.tf_use_latest = bool(self.get_parameter("tf_use_latest").value)
        self.tf_timeout = float(self.get_parameter("tf_timeout").value)

        # ===== YOLOv5 모델 로드 =====
        self.get_logger().info(f"Loading YOLOv5 model: {self.model_path}")
        try:
            self.device = select_device(self.device_id)
            self.model = DetectMultiBackend(
                self.model_path,
                device=self.device,
                dnn=False,
                data=self.data_yaml or None,
                fp16=True,  # TRT/ONNX fp16 지원 시 자동 사용
            )
            self.stride = int(self.model.stride)
            self.is_half = self.model.fp16

            # 클래스 이름 결정
            self.class_names = self._resolve_class_names()
            if not self.allow_classes:
                self.allow_classes = self.class_names

            # 색상 맵을 클래스별로 균일하게 부여
            self.color_map = {name: self._label_to_color(name) for name in self.class_names}
            self.default_color = (0, 255, 0)

            self.get_logger().info(
                f"Model loaded ({'FP16' if self.is_half else 'FP32'}) device={self.device}, stride={self.stride}"
            )
            self.get_logger().info(f"Classes ({len(self.class_names)}): {self.class_names}")
            self.get_logger().info(f"Allowed classes: {self.allow_classes if self.allow_classes else 'ALL'}")

            # 워밍업 (1x3xHxW)
            self.model.warmup(imgsz=(1, 3, self.input_size, self.input_size))
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLOv5 model: {e}")
            import traceback
            traceback.print_exc()
            raise

        # ===== ROS I/O 설정 =====
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.on_rgb, qos_sensor)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos_sensor)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_info, qos_sensor)

        self.pub_debug = self.create_publisher(Image, self.debug_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.pub_detections = self.create_publisher(DetectionArray, self.detection_topic, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self._last_transform = None

        self.last_depth: Optional[Image] = None
        self.camera_info: Optional[CameraInfo] = None
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1

        self._last_infer_t = 0.0
        self._last_fps = 0.0
        self.get_logger().info(f"YoloDepthMapperV5 initialized. Publishing to {self.detection_topic}")

    # ===== 유틸 =====
    def _resolve_class_names(self) -> List[str]:
        # 1) class_names 파라미터 우선
        if self.param_class_names:
            return self.param_class_names

        # 2) names_path yaml
        if self.names_path:
            try:
                with open(self.names_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                names_yaml = data.get("names") or data.get("classes")
                if isinstance(names_yaml, dict):
                    names_yaml = [names_yaml[k] for k in sorted(names_yaml.keys())]
                if isinstance(names_yaml, list) and names_yaml:
                    return [str(x) for x in names_yaml]
            except Exception as e:
                self.get_logger().warn(f"Failed to load names from {self.names_path}: {e}", throttle_duration_sec=2.0)

        # 3) 모델 metadata
        names = self.model.names if hasattr(self.model, "names") else []
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        if names:
            return [str(x) for x in names]

        # 4) 기본값
        return [f"class{i}" for i in range(80)]

    @staticmethod
    def _label_to_color(label: str) -> Tuple[int, int, int]:
        """라벨 문자열을 안정적인 BGR 색상으로 매핑"""
        h = hash(label) % 360
        s = 0.6
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
        return (int(b*255), int(g*255), int(r*255))

    def build_detection_geometry(self, det_tensor: torch.Tensor,
                                 image_shape: Tuple[int, int, int],
                                 stamp) -> List[Dict]:
        geom_list: List[Dict] = []
        if det_tensor is None or len(det_tensor) == 0:
            return geom_list

        depth_img, fx, fy, cx, cy = self.get_depth_and_intrinsics()
        if depth_img is None:
            self.get_logger().warn("Depth or camera_info not received yet; skipping depth fusion",
                                   throttle_duration_sec=2.0)
            return geom_list

        # TF lookup을 한 번만 수행 (성능 최적화)
        transform = self._get_transform_cached(stamp)

        img_h, img_w = image_shape[:2]
        det_array = det_tensor.cpu().numpy()

        for det_id, (x1, y1, x2, y2, conf, cls_idx) in enumerate(det_array.tolist()):
            cls_id = int(cls_idx)
            label = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class{cls_id}"
            if self.allow_classes and label not in self.allow_classes:
                continue

            xi1 = max(0, min(img_w - 1, int(round(x1))))
            yi1 = max(0, min(img_h - 1, int(round(y1))))
            xi2 = max(0, min(img_w - 1, int(round(x2))))
            yi2 = max(0, min(img_h - 1, int(round(y2))))
            if xi2 <= xi1 or yi2 <= yi1:
                continue

            if (xi2 - xi1) < self.min_box_px or (yi2 - yi1) < self.min_box_px:
                continue

            color = self.color_map.get(label, self.default_color)
            geom = {
                'id': det_id,
                'label': label,
                'confidence': float(conf),
                'bbox_px': (xi1, yi1, xi2, yi2),
                'color': color,
                'has_depth': False,
                'center_cam': None,
                'center_world': None,
                'size_m': None,
            }

            frac = float(np.clip(self.core_roi, 0.05, 1.0))
            h = yi2 - yi1
            w = xi2 - xi1
            mx = int(round((1.0 - frac) * w * 0.5))
            my = int(round((1.0 - frac) * h * 0.5))
            cx1 = xi1 + mx
            cx2 = xi2 - mx
            cy1 = yi1 + my
            cy2 = yi2 - my
            if cx2 <= cx1 or cy2 <= cy1:
                cx1, cy1, cx2, cy2 = xi1, yi1, xi2, yi2

            roi_core = depth_img[cy1:cy2, cx1:cx2]
            valid = roi_core[np.isfinite(roi_core) & (roi_core > 0)]

            if valid.size > 0:
                z = float(np.median(valid))
                center_u = (xi1 + xi2) / 2.0
                center_v = (yi1 + yi2) / 2.0

                x_cam = (center_u - cx) * z / fx
                y_cam = (center_v - cy) * z / fy

                width_m = abs((xi2 - xi1) * z / fx)
                height_m = abs((yi2 - yi1) * z / fy)
                depth_m = max(width_m, height_m) * 0.2

                geom['has_depth'] = True
                geom['center_cam'] = (x_cam, y_cam, z)
                geom['size_m'] = (width_m, height_m, depth_m)

                # Calculate orientation from point cloud using PCA
                orientation_quat = self._calculate_orientation_pca(
                    depth_img, xi1, yi1, xi2, yi2, fx, fy, cx, cy, z
                )
                geom['orientation_cam'] = orientation_quat

                # 미리 가져온 transform 사용 (반복 lookup 방지)
                if transform is not None:
                    center_world = self._apply_transform(geom['center_cam'], transform)
                    if center_world is not None:
                        geom['center_world'] = center_world
                    geom['orientation_world'] = None
                    try:
                        if orientation_quat is not None:
                            tf_rot = transform.transform.rotation
                            geom['orientation_world'] = self._quat_multiply(tf_rot, orientation_quat)
                    except Exception:
                        pass

            geom_list.append(geom)

        return geom_list

    def build_2d_geoms(self, det_tensor: torch.Tensor, image_shape: Tuple[int, int, int]) -> List[Dict]:
        geoms: List[Dict] = []
        if det_tensor is None or len(det_tensor) == 0:
            return geoms
        img_h, img_w = image_shape[:2]
        det_array = det_tensor.cpu().numpy()
        for det_id, (x1, y1, x2, y2, conf, cls_idx) in enumerate(det_array.tolist()):
            cls_id = int(cls_idx)
            label = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class{cls_id}"
            if self.allow_classes and label not in self.allow_classes:
                continue
            xi1 = max(0, min(img_w - 1, int(round(x1))))
            yi1 = max(0, min(img_h - 1, int(round(y1))))
            xi2 = max(0, min(img_w - 1, int(round(x2))))
            yi2 = max(0, min(img_h - 1, int(round(y2))))
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            geoms.append({
                'id': det_id,
                'label': label,
                'confidence': float(conf),
                'bbox_px': (xi1, yi1, xi2, yi2),
                'color': self.color_map.get(label, self.default_color),
            })
        return geoms

    def get_depth_and_intrinsics(self):
        if self.last_depth is None or self.camera_info is None:
            return None, None, None, None, None

        depth_img = self.bridge.imgmsg_to_cv2(self.last_depth, desired_encoding='passthrough')
        if depth_img.dtype == np.uint16 or depth_img.dtype == np.uint32:
            depth_img = depth_img.astype(np.float32) * 0.001
        elif depth_img.dtype != np.float32:
            depth_img = depth_img.astype(np.float32)

        cam_info = self.camera_info
        fx = cam_info.k[0]
        fy = cam_info.k[4]
        cx = cam_info.k[2]
        cy = cam_info.k[5]
        return depth_img, fx, fy, cx, cy

    def _calculate_orientation_pca(self, depth_img, x1, y1, x2, y2, fx, fy, cx, cy, median_z):
        """
        Calculate object orientation using PCA on depth point cloud (optimized)

        Returns:
            (qx, qy, qz, qw) quaternion representing orientation in camera frame
        """
        try:
            # Skip PCA for very small boxes (< 20x20 pixels) - not enough data
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                return (0.0, 0.0, 0.0, 1.0)

            # Extract ROI depth values with downsampling (step=4 for 16x speedup)
            step = 4
            roi = depth_img[y1:y2:step, x1:x2:step].astype(np.float32)
            valid_mask = np.isfinite(roi) & (roi > 0)

            if np.sum(valid_mask) < 10:  # Need at least 10 points
                return (0.0, 0.0, 0.0, 1.0)  # Default: no rotation

            # Build 3D point cloud from downsampled ROI (vectorized)
            v_indices, u_indices = np.where(valid_mask)
            v_indices = v_indices * step + y1
            u_indices = u_indices * step + x1

            z_values = depth_img[v_indices, u_indices].astype(np.float32)

            # Filter outliers (keep points within 30% of median depth)
            z_diff = np.abs(z_values - median_z)
            valid_points = z_diff < median_z * 0.3

            if np.sum(valid_points) < 10:
                return (0.0, 0.0, 0.0, 1.0)

            # Select valid points
            u_valid = u_indices[valid_points]
            v_valid = v_indices[valid_points]
            z_valid = z_values[valid_points]

            # Compute 3D coordinates (vectorized)
            x_coords = (u_valid - cx) * z_valid / fx
            y_coords = (v_valid - cy) * z_valid / fy
            points = np.column_stack([x_coords, y_coords, z_valid]).astype(np.float32)

            # Center the points
            centroid = np.mean(points, axis=0, dtype=np.float32)
            centered = points - centroid

            # Compute covariance matrix
            cov = np.cov(centered.T, dtype=np.float32)

            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Primary direction is the eigenvector with largest eigenvalue (last column)
            primary_dir = eigenvectors[:, -1]

            # Ensure primary direction points forward (positive Z in camera frame)
            if primary_dir[2] < 0:
                primary_dir = -primary_dir

            # Calculate rotation from camera Z-axis to primary direction
            # Camera Z-axis is [0, 0, 1]
            z_axis = np.array([0, 0, 1], dtype=np.float32)

            # Rotation axis (cross product)
            axis = np.cross(z_axis, primary_dir)
            axis_norm = np.linalg.norm(axis)

            if axis_norm < 1e-6:  # Vectors are parallel
                return (0.0, 0.0, 0.0, 1.0)

            axis = axis / axis_norm

            # Rotation angle
            angle = np.arccos(np.clip(np.dot(z_axis, primary_dir), -1.0, 1.0))

            # Convert axis-angle to quaternion
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)

            qx = float(axis[0] * sin_half)
            qy = float(axis[1] * sin_half)
            qz = float(axis[2] * sin_half)
            qw = float(np.cos(half_angle))

            return (qx, qy, qz, qw)

        except Exception:
            # Silent failure - no logging to avoid performance impact
            return (0.0, 0.0, 0.0, 1.0)

    def _get_transform_cached(self, stamp):
        """TF lookup을 한 번만 수행하고 캐시 (성능 최적화)"""
        if not self.world_frame:
            return None

        try:
            query_time = Time() if self.tf_use_latest else Time.from_msg(stamp)
        except Exception:
            query_time = Time()

        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.frame_id,
                query_time,
                timeout=Duration(seconds=self.tf_timeout)
            )
            self._last_transform = transform
            return transform
        except TransformException:
            return self._last_transform

    def _apply_transform(self, center_cam: Tuple[float, float, float], transform) -> Optional[Tuple[float, float, float]]:
        """캐시된 transform으로 좌표 변환"""
        if transform is None:
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot = self.quaternion_to_matrix(rotation)
        trans = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
        cam_vec = np.array(center_cam, dtype=np.float64)
        world_vec = rot @ cam_vec + trans
        return float(world_vec[0]), float(world_vec[1]), float(world_vec[2])

    def transform_point_to_world(self, center_cam: Tuple[float, float, float],
                                 stamp) -> Optional[Tuple[float, float, float]]:
        """단일 포인트 변환 (레거시 호환용)"""
        transform = self._get_transform_cached(stamp)
        return self._apply_transform(center_cam, transform)

    @staticmethod
    def quaternion_to_matrix(q) -> np.ndarray:
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-9:
            return np.identity(3, dtype=np.float64)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        return np.array([
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)],
        ], dtype=np.float64)

    @staticmethod
    def _quat_multiply(q1, q2) -> Tuple[float, float, float, float]:
        """q = q1 * q2 (geometry_msgs/tuple 모두 지원)"""
        def to_tuple(q):
            if hasattr(q, "x"):
                return float(q.x), float(q.y), float(q.z), float(q.w)
            return float(q[0]), float(q[1]), float(q[2]), float(q[3])

        x1, y1, z1, w1 = to_tuple(q1)
        x2, y2, z2, w2 = to_tuple(q2)
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            return (0.0, 0.0, 0.0, 1.0)
        return (x / norm, y / norm, z / norm, w / norm)

    # ===== Filter helpers =====
    def _bbox_changed(self, prev_bbox, new_bbox) -> bool:
        if prev_bbox is None or new_bbox is None:
            return True
        dz = max(0, float(self.bbox_deadzone_px))
        return any(abs(nb - pb) >= dz for pb, nb in zip(prev_bbox, new_bbox))

    def _smooth_bbox(self, prev_bbox, new_bbox) -> Tuple[float, float, float, float]:
        if prev_bbox is None or new_bbox is None:
            return new_bbox
        a = float(np.clip(self.ema_alpha, 0.0, 1.0))
        dz = max(0, float(self.bbox_deadzone_px))
        smoothed = []
        for pb, nb in zip(prev_bbox, new_bbox):
            if abs(nb - pb) < dz:
                smoothed.append(pb)
            else:
                smoothed.append(a * nb + (1 - a) * pb)
        return tuple(smoothed)

    def _smooth_tuple(self, prev_val, new_val, deadzone: float = 0.0):
        if new_val is None:
            return prev_val
        if prev_val is None:
            return new_val
        a = float(np.clip(self.ema_alpha, 0.0, 1.0))
        dz = max(0.0, float(deadzone))
        smoothed = []
        for pv, nv in zip(prev_val, new_val):
            if dz > 0.0 and abs(nv - pv) < dz:
                smoothed.append(pv)
            else:
                smoothed.append(a * nv + (1 - a) * pv)
        return tuple(smoothed)

    def apply_persistence(self, detections: List[Dict]) -> List[Dict]:
        by_label: Dict[str, List[int]] = {}
        for tid, t in self.tracks.items():
            by_label.setdefault(t['label'], []).append(tid)

        matched_tids = set()
        for g in detections:
            if g.get('center_world') is None:
                continue
            label = g['label']
            tid = self._match_track(by_label.get(label, []), g['center_world'])
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'label': label,
                    'center_cam': g['center_cam'],
                    'center_world': g['center_world'],
                    'bbox_px': g['bbox_px'],
                    'size_m': g.get('size_m'),
                    'color': g['color'],
                    'confidence': g['confidence'],
                    'hits': 1,
                    'miss': 0,
                }
                by_label.setdefault(label, []).append(tid)
            else:
                t = self.tracks[tid]
                bbox_prev = t.get('bbox_px')
                bbox_changed = self._bbox_changed(bbox_prev, g['bbox_px'])
                if bbox_changed and t.get('center_cam') and g.get('center_cam'):
                    a = float(np.clip(self.ema_alpha, 0.0, 1.0))
                    tx, ty, tz = t['center_cam']
                    nx, ny, nz = g['center_cam']
                    t['center_cam'] = (a*nx + (1-a)*tx, a*ny + (1-a)*ty, a*nz + (1-a)*tz)
                    twx, twy, twz = t['center_world']
                    nwx, nwy, nwz = g['center_world']
                    t['center_world'] = (a*nwx + (1-a)*twx, a*nwy + (1-a)*twy, a*nwz + (1-a)*twz)
                if g.get('size_m') is not None:
                    t['size_m'] = self._smooth_tuple(t.get('size_m'), g.get('size_m'))
                t['bbox_px'] = self._smooth_bbox(bbox_prev, g['bbox_px'])
                t['confidence'] = max(t['confidence'], g['confidence'])
                t['hits'] += 1
                t['miss'] = 0
            matched_tids.add(tid)

        to_delete = []
        for tid, t in self.tracks.items():
            if tid not in matched_tids:
                t['miss'] += 1
            if t['miss'] > self.persist_t:
                to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)

        stable: List[Dict] = []
        for tid, t in self.tracks.items():
            if t['hits'] >= self.persist_k and t.get('center_world') is not None:
                stable.append({
                    'id': tid,
                    'label': t['label'],
                    'confidence': t['confidence'],
                    'bbox_px': t['bbox_px'],
                    'color': t['color'],
                    'has_depth': True,
                    'center_cam': t['center_cam'],
                    'center_world': t['center_world'],
                    'size_m': t.get('size_m'),
                })
        return stable

    def _match_track(self, tracks_for_label: List[int],
                     center_world: Tuple[float, float, float]) -> Optional[int]:
        if not tracks_for_label:
            return None
        best_id = None
        best_dist = 1e9
        cx, cy, cz = center_world
        for tid in tracks_for_label:
            t = self.tracks[tid]
            tx, ty, tz = t['center_world']
            dist = np.linalg.norm([cx - tx, cy - ty, cz - tz])
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id if best_dist <= self.match_dist_m else None

    # ===== 시각화 =====
    def draw_detections(self, frame_bgr, geom_list: List[Dict]):
        if not geom_list:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        for geom in geom_list:
            x1, y1, x2, y2 = geom['bbox_px']
            conf = geom.get('confidence', 0.0)
            label = geom['label']
            color = geom.get('color', self.default_color)
            xi1, yi1 = max(0, min(w-1, int(x1))), max(0, min(h-1, int(y1)))
            xi2, yi2 = max(0, min(w-1, int(x2))), max(0, min(h-1, int(y2)))
            cv2.rectangle(frame_bgr, (xi1, yi1), (xi2, yi2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (xi1, yi1 - th - 6), (xi1 + tw + 2, yi1), color, -1)
            cv2.putText(frame_bgr, text, (xi1 + 1, yi1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr

    def publish_markers(self, geom_list: List[Dict], stamp, frame_id: str):
        markers = []
        for idx, geom in enumerate(geom_list):
            color_bgr = geom['color']
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = 'yolo_bbox'
            marker.id = geom.get('id', idx)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            if geom.get('center_cam'):
                cx, cy, cz = geom['center_cam']
                marker.pose.position.x = float(cx)
                marker.pose.position.y = float(cy)
                marker.pose.position.z = float(cz)
            if geom.get('size_m'):
                sx, sy, sz = geom['size_m']
                marker.scale.x = float(sx)
                marker.scale.y = float(sy)
                marker.scale.z = float(sz)
            else:
                marker.scale.x = marker.scale.y = marker.scale.z = 0.1

            # Use calculated orientation from PCA
            if geom.get('orientation_cam'):
                qx, qy, qz, qw = geom['orientation_cam']
                marker.pose.orientation.x = qx
                marker.pose.orientation.y = qy
                marker.pose.orientation.z = qz
                marker.pose.orientation.w = qw
            else:
                marker.pose.orientation.w = 1.0

            marker.color.r = color_bgr[2] / 255.0
            marker.color.g = color_bgr[1] / 255.0
            marker.color.b = color_bgr[0] / 255.0
            marker.color.a = 0.5
            markers.append(marker)
        self.pub_markers.publish(MarkerArray(markers=markers))

    def publish_detections(self, geom_list: List[Dict], header):
        out = DetectionArray()
        out.header.stamp = header.stamp
        out.header.frame_id = self.world_frame if self.world_frame else self.frame_id
        for geom in geom_list:
            if not geom.get('center_world'):
                continue
            det = Detection()
            det.header.stamp = header.stamp
            det.header.frame_id = out.header.frame_id
            det.id = int(geom.get('id', 0))
            det.label = geom['label']
            det.confidence = float(geom.get('confidence', 0.0))
            cx, cy, cz = geom['center_world']
            det.pose.position.x = float(cx)
            det.pose.position.y = float(cy)
            det.pose.position.z = float(cz)

            # Use calculated orientation from PCA
            if geom.get('orientation_world'):
                qx, qy, qz, qw = geom['orientation_world']
                det.pose.orientation.x = qx
                det.pose.orientation.y = qy
                det.pose.orientation.z = qz
                det.pose.orientation.w = qw
            elif geom.get('orientation_cam'):
                qx, qy, qz, qw = geom['orientation_cam']
                det.pose.orientation.x = qx
                det.pose.orientation.y = qy
                det.pose.orientation.z = qz
                det.pose.orientation.w = qw
            else:
                det.pose.orientation.w = 1.0

            if geom.get('size_m'):
                sx, sy, sz = geom['size_m']
                det.size.x = float(sx)
                det.size.y = float(sy)
                det.size.z = float(sz)
            out.detections.append(det)
        self.pub_detections.publish(out)

    # ===== 콜백 =====
    def on_rgb(self, msg: Image):
        now_t = time.monotonic()
        if (now_t - self._last_infer_t) < self.max_period:
            return
        # FPS 계산
        if self._last_infer_t > 0:
            dt = now_t - self._last_infer_t
            if dt > 0:
                self._last_fps = 1.0 / dt
        self._last_infer_t = now_t

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        try:
            img0 = frame
            # TRT 엔진은 고정 입력(640x640)이라 letterbox를 고정크기로 맞춘다
            im = letterbox(img0, self.input_size, stride=self.stride, auto=False)[0]
            im = im.transpose((2, 0, 1))[::-1]  # BGR->RGB, HWC->CHW
            im = np.ascontiguousarray(im)
            im_tensor = torch.from_numpy(im).to(self.device)
            im_tensor = im_tensor.half() if self.is_half else im_tensor.float()
            im_tensor /= 255.0
            if im_tensor.ndimension() == 3:
                im_tensor = im_tensor.unsqueeze(0)

            pred = self.model(im_tensor, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.conf, self.iou, classes=None, agnostic=False, max_det=100)

            det_tensor = torch.empty((0, 6), dtype=torch.float32)
            if len(pred) and len(pred[0]):
                det = pred[0]
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], img0.shape).round()
                det_tensor = torch.cat([det[:, :4], det[:, 4:5], det[:, 5:6]], dim=1).cpu()

            raw_geoms = self.build_detection_geometry(det_tensor, frame.shape, msg.header.stamp)
            stable_geoms = self.apply_persistence(raw_geoms)

            # 디버그 이미지 생성 (RViz에서 항상 사용하므로 항상 발행)
            # frame을 직접 수정하여 불필요한 copy 제거
            if stable_geoms:
                annotated = self.draw_detections(frame, stable_geoms)
            else:
                annotated = self.draw_detections(frame, self.build_2d_geoms(det_tensor, frame.shape))

            # FPS 오버레이
            fps_text = f"FPS: {self._last_fps:.1f}"
            cv2.putText(annotated, fps_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # 마커와 detection 발행
            self.publish_markers(stable_geoms, msg.header.stamp, self.frame_id)
            self.publish_detections(stable_geoms, msg.header)

            # 디버그 이미지 발행
            debug_img = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            debug_img.header.stamp = msg.header.stamp
            debug_img.header.frame_id = self.frame_id
            self.pub_debug.publish(debug_img)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

    def on_depth(self, msg: Image):
        self.last_depth = msg

    def on_info(self, msg: CameraInfo):
        self.camera_info = msg


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthMapperV5()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
