#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Depth Mapper for RTAB-Map VSLAM
RealSense D455 + YOLOv8 TensorRT + RTAB-Map Integration

이 노드는 다음 작업을 수행합니다:
1. RGB 이미지에서 YOLOv8로 객체 감지 (TensorRT 가속)
2. Depth 이미지에서 3D 위치 계산
3. TF를 사용해 카메라 좌표계 -> 월드 좌표계 변환
4. Persistence tracking으로 노이즈 필터링 (3번 연속 검출되어야 확정)
5. DetectionArray 메시지로 발행 -> yolo_rtabmap_fusion으로 전달
"""

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

# Ultralytics YOLO (YOLO.predict()가 모든 전처리/후처리를 자동으로 수행)
try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(f"ultralytics not installed: {e}. Run: pip3 install ultralytics")

from semantic_mapper_msgs.msg import DetectionArray, Detection


class YoloDepthMapper(Node):
    """
    YOLO + Depth 융합 노드

    주요 기능:
    - RGB 이미지에서 YOLOv8로 2D 객체 감지
    - Depth 이미지로 3D 위치 계산
    - TF를 사용해 월드 좌표계로 변환
    - Persistence tracking으로 노이즈 제거
    - RViz 시각화 마커 발행
    """

    def __init__(self):
        super().__init__('yolo_depth_mapper')
        self.bridge = CvBridge()  # ROS 이미지 <-> OpenCV 변환기

        # ===== 파라미터 선언 =====
        # 카메라 토픽 설정
        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        # 좌표계 프레임
        self.declare_parameter("frame_id", "camera_color_optical_frame")  # 카메라 optical frame
        self.declare_parameter("world_frame", "map")  # 월드 좌표계 (RTAB-Map의 map frame)

        # YOLO 모델 설정
        self.declare_parameter("model_path", "best.engine")  # TensorRT 엔진 파일 경로
        self.declare_parameter("conf_threshold", 0.35)  # 신뢰도 임계값 (35% 이상만 사용)
        self.declare_parameter("iou_threshold", 0.50)  # NMS IOU 임계값
        self.declare_parameter("input_size", 640)  # YOLO 입력 해상도
        self.declare_parameter("device", "cuda:0")  # GPU 디바이스

        # 허용할 객체 클래스 (13개만 사용)
        # 기본: 빈 리스트 -> 모델의 클래스 전체 사용
        self.declare_parameter(
            "allow_classes",
            [],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        )
        # 클래스 이름을 외부 yaml로 지정할 수 있는 옵션
        self.declare_parameter("names_path", "")
        # 또는 파라미터로 직접 클래스 배열을 지정
        self.declare_parameter(
            "class_names",
            [],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        )

        # 출력 토픽
        self.declare_parameter("debug_topic", "/yolo/debug_image")  # 시각화 이미지
        self.declare_parameter("marker_topic", "/yolo/marker_array")  # RViz 마커
        self.declare_parameter("detection_topic", "/semantic_mapper/detections")  # Detection 메시지

        # 튜닝 파라미터
        self.declare_parameter("min_box_px", 30)  # 최소 bounding box 크기 (픽셀)
        self.declare_parameter("core_roi", 0.60)  # Depth 추출시 중심부 영역 비율 (60%)
        self.declare_parameter("max_fps", 10.0)  # 최대 처리 FPS (성능 제한)

        # Persistence tracking 파라미터
        # 노이즈 제거를 위한 시간적 필터링
        self.declare_parameter("persist_k", 3)  # 3번 연속 검출되어야 확정
        self.declare_parameter("persist_t", 10)  # 10프레임 miss 후 삭제
        self.declare_parameter("ema_alpha", 0.15)  # EMA 필터 계수 (0.15 = 더 완만한 보간)
        self.declare_parameter("bbox_deadzone_px", 5)  # 바운딩 박스 픽셀 변화 데드존
        self.declare_parameter("match_dist_m", 0.40)  # 매칭 거리 임계값 (0.4m 이내면 같은 객체)

        # TF (좌표 변환) 설정
        self.declare_parameter("tf_use_latest", True)  # 최신 TF 사용 (시간 동기화 안함)
        self.declare_parameter("tf_timeout", 0.25)  # TF lookup 타임아웃 (0.25초)

        # ===== 파라미터 읽기 =====
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.world_frame = self.get_parameter("world_frame").value

        self.model_path = self.get_parameter("model_path").value
        self.conf = float(self.get_parameter("conf_threshold").value)
        self.iou = float(self.get_parameter("iou_threshold").value)
        self.input_size = int(self.get_parameter("input_size").value)
        self.device = self.get_parameter("device").value

        # allow_classes, class_names는 모델 로드 시점에서 파라미터와 결합하므로 여기서는 읽지 않는다
        self.debug_topic = self.get_parameter("debug_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.detection_topic = self.get_parameter("detection_topic").value

        self.min_box_px = int(self.get_parameter("min_box_px").value)
        self.core_roi = float(self.get_parameter("core_roi").value)
        self.max_fps = float(self.get_parameter("max_fps").value)
        self.max_period = 1.0 / max(self.max_fps, 1.0)  # FPS -> 주기(초)

        self.persist_k = int(self.get_parameter("persist_k").value)
        self.persist_t = int(self.get_parameter("persist_t").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.bbox_deadzone_px = int(self.get_parameter("bbox_deadzone_px").value)
        self.match_dist_m = float(self.get_parameter("match_dist_m").value)

        self.tf_use_latest = bool(self.get_parameter("tf_use_latest").value)
        self.tf_timeout = float(self.get_parameter("tf_timeout").value)

        # ===== YOLO 모델 로드 =====
        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        try:
            suffix = pathlib.Path(self.model_path).suffix.lower()

            # YOLO 클래스를 사용하여 모든 형식 로드 (TensorRT, PyTorch 자동 감지)
            self.model = YOLO(self.model_path)

            # ===== 클래스 이름 결정 (엔진에 metadata 없을 때 대비) =====
            class_names: List[str] = []

            # 1) 파라미터 class_names가 지정되면 우선 사용
            param_class_names = self.get_parameter("class_names").value
            if param_class_names:
                class_names = [str(x) for x in param_class_names]
                self.get_logger().info(f"Using class_names from parameter ({len(class_names)})")

            # 2) names_path 파라미터 yaml에서 names 키를 읽는다
            names_path = self.get_parameter("names_path").value
            if not class_names and names_path:
                try:
                    with open(names_path, "r") as f:
                        data = yaml.safe_load(f) or {}
                    names_yaml = data.get("names") or data.get("classes")
                    if isinstance(names_yaml, dict):
                        names_yaml = [names_yaml[k] for k in sorted(names_yaml.keys())]
                    if isinstance(names_yaml, list) and names_yaml:
                        class_names = [str(x) for x in names_yaml]
                        self.get_logger().info(f"Using class names from yaml: {names_path} ({len(class_names)})")
                except Exception as e:
                    self.get_logger().warn(f"Failed to load names from {names_path}: {e}", throttle_duration_sec=2.0)

            # 3) 모델 metadata에서 names 읽기
            if not class_names:
                names = self.model.names
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names.keys())]
                if names:
                    class_names = [str(x) for x in names]
                    if len(class_names) > 200:
                        self.get_logger().warn(
                            f"Model reports {len(class_names)} classes; consider setting class_names/names_path",
                            throttle_duration_sec=2.0
                        )
                else:
                    class_names = [f"class{i}" for i in range(80)]
                    self.get_logger().warn("Model names empty; defaulting to class0-79", throttle_duration_sec=2.0)

            self.class_names = class_names

            # 허용 클래스: 파라미터가 비어있으면 모델 클래스 전체 사용
            allow_param = self.get_parameter("allow_classes").value
            self.allow_classes = list(allow_param) if allow_param else self.class_names

            if suffix == ".engine":
                self.model_kind = "trt"
                self.get_logger().info(f"TensorRT engine loaded: {self.model_path}")
            elif suffix == ".pt":
                self.model_kind = "pt"
                self.get_logger().info(f"PyTorch model loaded: {self.model_path}")
            else:
                self.model_kind = "other"
                self.get_logger().info(f"Model loaded: {self.model_path}")

            self.get_logger().info(f"Model device: {self.device}, input_size: {self.input_size}")
            self.get_logger().info(f"Model classes: {self.class_names}")
            self.get_logger().info(f"Allowed classes: {self.allow_classes if self.allow_classes else 'ALL'}")

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise

        # ===== ROS I/O 설정 =====
        # BEST_EFFORT QoS: 실시간 센서 데이터용 (일부 손실 허용)
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # 구독자 (Subscriber)
        self.sub_rgb = self.create_subscription(
            Image, self.rgb_topic, self.on_rgb, qos_sensor)
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.on_depth, qos_sensor)
        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.on_info, qos_sensor)

        # 발행자 (Publisher)
        self.pub_debug = self.create_publisher(Image, self.debug_topic, 10)  # 디버그 이미지
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)  # RViz 마커
        self.pub_detections = self.create_publisher(DetectionArray, self.detection_topic, 10)  # Detection 메시지

        # TF (좌표 변환) 리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self._last_transform = None  # 마지막 성공한 transform 캐시

        # 데이터 저장
        self.last_depth: Optional[Image] = None  # 마지막 depth 이미지
        self.camera_info: Optional[CameraInfo] = None  # 카메라 intrinsics

        # Persistence tracking 상태
        self.tracks: Dict[int, Dict] = {}  # track_id -> track_info
        self.next_id = 1  # 다음 track ID

        # 시각화 색상 맵 (각 클래스별 BGR 색상)
        self.color_map = {
            'tv': (255, 255, 0),  # 청록색
            'cup': (255, 192, 203),  # 분홍색
            'monitor': (255, 165, 0),  # 주황색
            'laptop': (0, 255, 255),  # 노란색
            'chair': (255, 0, 0),  # 파란색
            'couch': (211, 0, 148),  # 자주색
            'book': (139, 69, 19),  # 갈색
            'keyboard': (0, 128, 128),  # 올리브색
            'mouse': (128, 128, 0),  # 청록색
            'potted plant': (0, 255, 0),  # 녹색
            'bottle': (0, 191, 255),  # 금색
            'cell phone': (255, 128, 0),  # 주황색
            'dining table': (34, 139, 34),  # 초록색
        }
        self.default_color = (0, 255, 0)  # 기본 색상 (녹색)

        self._last_infer_t = 0.0  # 마지막 추론 시간 (FPS 제한용)
        self.get_logger().info(f"YoloDepthMapper initialized. Publishing to {self.detection_topic}")

    # ===== 3D 기하학 추출 =====
    def build_detection_geometry(self, det_tensor: torch.Tensor,
                                 image_shape: Tuple[int, int, int],
                                 stamp) -> List[Dict]:
        """
        2D YOLO detection에서 3D 기하학 정보 추출

        Args:
            det_tensor: YOLO detection 결과 (N x 6) [x1, y1, x2, y2, conf, cls]
            image_shape: 이미지 크기 (H, W, C)
            stamp: 타임스탬프

        Returns:
            geom_list: 3D 기하학 정보 리스트
                - label: 클래스 이름
                - confidence: 신뢰도
                - bbox_px: 2D bounding box (픽셀 좌표)
                - center_cam: 카메라 좌표계 중심점 (x, y, z)
                - center_world: 월드 좌표계 중심점 (x, y, z)
                - size_m: 객체 크기 (width, height, depth) in meters
        """
        geom_list: List[Dict] = []
        if det_tensor is None or len(det_tensor) == 0:
            return geom_list

        # Depth 이미지와 카메라 intrinsics 가져오기
        depth_img, fx, fy, cx, cy = self.get_depth_and_intrinsics()
        if depth_img is None:
            # 깊이/카메라정보 없으면 3D 계산 패스 (디버그 목적)
            self.get_logger().warn(
                "Depth or camera_info not received yet; skipping depth fusion",
                throttle_duration_sec=2.0
            )
            return geom_list

        img_h, img_w = image_shape[:2]
        det_array = det_tensor.cpu().numpy()

        # 각 detection 처리
        for det_id, (x1, y1, x2, y2, conf, cls_idx) in enumerate(det_array.tolist()):
            cls_id = int(cls_idx)
            label = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class{cls_id}"

            # 클래스 필터링 (허용된 13개 클래스만 사용)
            if self.allow_classes and label not in self.allow_classes:
                continue

            # 픽셀 좌표 정규화 (이미지 범위 내로 clamp)
            xi1 = max(0, min(img_w - 1, int(round(x1))))
            yi1 = max(0, min(img_h - 1, int(round(y1))))
            xi2 = max(0, min(img_w - 1, int(round(x2))))
            yi2 = max(0, min(img_h - 1, int(round(y2))))

            # 유효하지 않은 박스 제거
            if xi2 <= xi1 or yi2 <= yi1:
                continue

            # 너무 작은 박스 필터링 (노이즈 제거)
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

            # ===== Depth 정보 추출 =====
            # 중심부 ROI만 사용 (가장자리는 노이즈가 많음)
            # core_roi=0.6이면 중심 60% 영역만 사용
            frac = float(np.clip(self.core_roi, 0.05, 1.0))
            h = yi2 - yi1
            w = xi2 - xi1
            mx = int(round((1.0 - frac) * w * 0.5))  # X 마진
            my = int(round((1.0 - frac) * h * 0.5))  # Y 마진
            cx1 = xi1 + mx
            cx2 = xi2 - mx
            cy1 = yi1 + my
            cy2 = yi2 - my

            # 너무 작아지면 원본 박스 사용
            if cx2 <= cx1 or cy2 <= cy1:
                cx1, cy1, cx2, cy2 = xi1, yi1, xi2, yi2

            # Depth ROI 추출
            roi_core = depth_img[cy1:cy2, cx1:cx2]
            # 유효한 depth 값만 추출 (finite & > 0)
            valid = roi_core[np.isfinite(roi_core) & (roi_core > 0)]

            if valid.size > 0:
                # Median depth 사용 (outlier에 강건)
                z = float(np.median(valid))
                center_u = (xi1 + xi2) / 2.0  # 박스 중심 픽셀 (u)
                center_v = (yi1 + yi2) / 2.0  # 박스 중심 픽셀 (v)

                # ===== 픽셀 -> 카메라 좌표 변환 =====
                # Pinhole camera model:
                # x_cam = (u - cx) * z / fx
                # y_cam = (v - cy) * z / fy
                # z_cam = z
                x_cam = (center_u - cx) * z / fx
                y_cam = (center_v - cy) * z / fy

                # ===== 크기 추정 =====
                # 픽셀 크기를 미터로 변환
                width_m = abs((xi2 - xi1) * z / fx)
                height_m = abs((yi2 - yi1) * z / fy)
                # Depth는 width/height의 20%로 가정 (임시 값)
                depth_m = max(width_m, height_m) * 0.2

                geom['has_depth'] = True
                geom['center_cam'] = (x_cam, y_cam, z)
                geom['size_m'] = (width_m, height_m, depth_m)

                # ===== 카메라 좌표 -> 월드 좌표 변환 =====
                center_world = self.transform_point_to_world(geom['center_cam'], stamp)
                if center_world is not None:
                    geom['center_world'] = center_world

            geom_list.append(geom)

        return geom_list

    def build_2d_geoms(self, det_tensor: torch.Tensor, image_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        Depth 없이 2D 박스만 만드는 경량 헬퍼 (디버그용)
        """
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
        """
        Depth 이미지와 카메라 intrinsics 가져오기

        Returns:
            depth_img: Depth 이미지 (numpy array, meters)
            fx, fy: Focal length (픽셀)
            cx, cy: Principal point (픽셀)
        """
        if self.last_depth is None or self.camera_info is None:
            return None, None, None, None, None

        # ROS Image -> OpenCV numpy array
        depth_img = self.bridge.imgmsg_to_cv2(self.last_depth, desired_encoding='passthrough')

        # uint16 (mm) -> float32 (m) 변환
        if depth_img.dtype == np.uint16:
            depth_img = depth_img.astype(np.float32) * 0.001
        elif depth_img.dtype == np.uint32:
            depth_img = depth_img.astype(np.float32) * 0.001
        elif depth_img.dtype != np.float32:
            depth_img = depth_img.astype(np.float32)

        # Camera intrinsics 추출
        # K = [fx  0 cx]
        #     [ 0 fy cy]
        #     [ 0  0  1]
        cam_info = self.camera_info
        fx = cam_info.k[0]  # K[0,0]
        fy = cam_info.k[4]  # K[1,1]
        cx = cam_info.k[2]  # K[0,2]
        cy = cam_info.k[5]  # K[1,2]

        return depth_img, fx, fy, cx, cy

    def transform_point_to_world(self, center_cam: Tuple[float, float, float],
                                 stamp) -> Optional[Tuple[float, float, float]]:
        """
        카메라 좌표계 -> 월드 좌표계 변환

        TF2를 사용해 camera_color_optical_frame -> map 변환 수행

        Args:
            center_cam: 카메라 좌표계 점 (x, y, z)
            stamp: 타임스탬프

        Returns:
            center_world: 월드 좌표계 점 (x, y, z) 또는 None
        """
        if not self.world_frame:
            return None

        # 타임스탬프 설정 (최신 TF 사용 or 시간 동기화)
        try:
            query_time = Time() if self.tf_use_latest else Time.from_msg(stamp)
        except Exception:
            query_time = Time()

        # TF lookup 시도
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,  # target frame (map)
                self.frame_id,  # source frame (camera_color_optical_frame)
                query_time,
                timeout=Duration(seconds=self.tf_timeout)
            )
            self._last_transform = transform  # 성공한 transform 캐시
        except TransformException:
            # TF를 찾지 못하면 마지막 캐시 사용
            transform = self._last_transform
            if transform is None:
                return None

        # ===== 좌표 변환 수행 =====
        # world_point = R * cam_point + t
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot = self.quaternion_to_matrix(rotation)  # Quaternion -> Rotation matrix
        trans = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
        cam_vec = np.array(center_cam, dtype=np.float64)
        world_vec = rot @ cam_vec + trans

        return float(world_vec[0]), float(world_vec[1]), float(world_vec[2])

    @staticmethod
    def quaternion_to_matrix(q) -> np.ndarray:
        """
        Quaternion -> Rotation matrix 변환

        Args:
            q: geometry_msgs/Quaternion (x, y, z, w)

        Returns:
            R: 3x3 rotation matrix
        """
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)

        # Quaternion 정규화
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-9:
            return np.identity(3, dtype=np.float64)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

        # Rotation matrix 계산
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        return np.array([
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)],
        ], dtype=np.float64)

    # ===== Filter helpers =====
    def _bbox_changed(self, prev_bbox, new_bbox) -> bool:
        """바운딩 박스가 데드존을 넘어섰는지 여부"""
        if prev_bbox is None or new_bbox is None:
            return True
        dz = max(0, float(self.bbox_deadzone_px))
        return any(abs(nb - pb) >= dz for pb, nb in zip(prev_bbox, new_bbox))

    def _smooth_bbox(self, prev_bbox, new_bbox) -> Tuple[float, float, float, float]:
        """데드존+EMA를 적용한 바운딩 박스 스무딩"""
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
        """크기/좌표에 EMA 및 선택적 데드존 적용"""
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

    # ===== Persistence Tracking =====
    def apply_persistence(self, detections: List[Dict]) -> List[Dict]:
        """
        Persistence tracking으로 노이즈 필터링

        목적: 일시적으로 나타나는 false positive 제거
        방법:
        1. 이전 프레임의 track과 현재 detection 매칭
        2. 매칭 성공하면 EMA 필터로 위치 업데이트
        3. 매칭 실패하면 새로운 track 생성
        4. persist_k번 이상 검출된 track만 확정
        5. persist_t번 이상 miss된 track 삭제

        Args:
            detections: 현재 프레임의 detection 리스트

        Returns:
            stable: 안정적인 detection 리스트 (hits >= persist_k)
        """
        # 레이블별로 track 그룹화
        by_label: Dict[str, List[int]] = {}
        for tid, t in self.tracks.items():
            by_label.setdefault(t['label'], []).append(tid)

        matched_tids = set()

        # ===== Detection과 Track 매칭 =====
        for g in detections:
            if g.get('center_world') is None:
                continue

            label = g['label']
            # 같은 레이블의 track들 중에서 가장 가까운 것 찾기
            tid = self._match_track(by_label.get(label, []), g['center_world'])

            if tid is None:
                # ===== 새로운 track 생성 =====
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
                    'hits': 1,  # 검출 횟수
                    'miss': 0,  # 미검출 횟수
                }
                by_label.setdefault(label, []).append(tid)
            else:
                # ===== 기존 track 업데이트 (EMA 필터) =====
                t = self.tracks[tid]
                bbox_prev = t.get('bbox_px')
                bbox_changed = self._bbox_changed(bbox_prev, g['bbox_px'])

                # EMA: new_value = alpha * measurement + (1-alpha) * old_value
                # 바운딩 박스/크기/좌표를 함께 스무딩 (작은 변화는 데드존으로 무시)
                if bbox_changed and t.get('center_cam') and g.get('center_cam'):
                    a = float(np.clip(self.ema_alpha, 0.0, 1.0))  # EMA 계수

                    # 카메라 좌표 EMA 업데이트
                    tx, ty, tz = t['center_cam']
                    nx, ny, nz = g['center_cam']
                    t['center_cam'] = (a*nx + (1-a)*tx, a*ny + (1-a)*ty, a*nz + (1-a)*tz)

                    # 월드 좌표 EMA 업데이트
                    twx, twy, twz = t['center_world']
                    nwx, nwy, nwz = g['center_world']
                    t['center_world'] = (a*nwx + (1-a)*twx, a*nwy + (1-a)*twy, a*nwz + (1-a)*twz)

                # 크기 스무딩 (EMA)
                if g.get('size_m') is not None:
                    t['size_m'] = self._smooth_tuple(t.get('size_m'), g.get('size_m'))

                # 바운딩 박스 스무딩 (데드존 + EMA)
                t['bbox_px'] = self._smooth_bbox(bbox_prev, g['bbox_px'])
                t['confidence'] = max(t['confidence'], g['confidence'])  # 최대 confidence 사용
                t['hits'] += 1
                t['miss'] = 0  # 검출되었으므로 miss 리셋

            matched_tids.add(tid)

        # ===== 미검출된 track의 miss count 증가 =====
        to_delete = []
        for tid, t in self.tracks.items():
            if tid not in matched_tids:
                t['miss'] += 1
            # persist_t번 이상 miss되면 삭제
            if t['miss'] > self.persist_t:
                to_delete.append(tid)

        for tid in to_delete:
            self.tracks.pop(tid, None)

        # ===== 안정적인 track만 반환 =====
        # hits >= persist_k: persist_k번 이상 연속 검출된 것만
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
        """
        가장 가까운 track 찾기

        Args:
            tracks_for_label: 같은 레이블의 track ID 리스트
            center_world: 현재 detection의 월드 좌표

        Returns:
            track_id: 매칭된 track ID 또는 None
        """
        if not tracks_for_label:
            return None

        best_id = None
        best_dist = 1e9
        cx, cy, cz = center_world

        # 유클리드 거리가 가장 가까운 track 찾기
        for tid in tracks_for_label:
            t = self.tracks[tid]
            tx, ty, tz = t['center_world']
            dist = np.linalg.norm([cx - tx, cy - ty, cz - tz])
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        # match_dist_m 이내면 매칭 성공, 아니면 None
        return best_id if best_dist <= self.match_dist_m else None

    # ===== 시각화 =====
    def draw_detections(self, frame_bgr, geom_list: List[Dict]):
        """
        이미지에 bounding box와 레이블 그리기

        Args:
            frame_bgr: BGR 이미지
            geom_list: detection 리스트

        Returns:
            frame_bgr: annotated 이미지
        """
        if not geom_list:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        for geom in geom_list:
            x1, y1, x2, y2 = geom['bbox_px']
            conf = geom['confidence']
            label = geom['label']
            color = geom['color']

            # 좌표 clamp
            xi1, yi1 = max(0, min(w-1, int(x1))), max(0, min(h-1, int(y1)))
            xi2, yi2 = max(0, min(w-1, int(x2))), max(0, min(h-1, int(y2)))

            # Bounding box 그리기
            cv2.rectangle(frame_bgr, (xi1, yi1), (xi2, yi2), color, 2)

            # 레이블 배경 + 텍스트
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (xi1, yi1 - th - 6), (xi1 + tw + 2, yi1), color, -1)
            cv2.putText(frame_bgr, text, (xi1 + 1, yi1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame_bgr

    def publish_markers(self, geom_list: List[Dict], stamp, frame_id: str):
        """
        RViz 시각화 마커 발행

        Args:
            geom_list: detection 리스트
            stamp: 타임스탬프
            frame_id: frame ID
        """
        markers = []

        for idx, geom in enumerate(geom_list):
            color_bgr = geom['color']
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = 'yolo_bbox'
            marker.id = geom.get('id', idx)
            marker.type = Marker.CUBE  # 큐브로 표시
            marker.action = Marker.ADD

            # 위치 (카메라 좌표계)
            if geom.get('center_cam'):
                cx, cy, cz = geom['center_cam']
                marker.pose.position.x = float(cx)
                marker.pose.position.y = float(cy)
                marker.pose.position.z = float(cz)

            # 크기
            if geom.get('size_m'):
                sx, sy, sz = geom['size_m']
                marker.scale.x = float(sx)
                marker.scale.y = float(sy)
                marker.scale.z = float(sz)
            else:
                marker.scale.x = marker.scale.y = marker.scale.z = 0.1

            # 방향 (회전 없음)
            marker.pose.orientation.w = 1.0

            # 색상 (BGR -> RGB 변환)
            marker.color.r = color_bgr[2] / 255.0
            marker.color.g = color_bgr[1] / 255.0
            marker.color.b = color_bgr[0] / 255.0
            marker.color.a = 0.5  # 투명도

            markers.append(marker)

        self.pub_markers.publish(MarkerArray(markers=markers))

    def publish_detections(self, geom_list: List[Dict], header):
        """
        Detection 메시지 발행 (yolo_rtabmap_fusion으로 전달)

        Args:
            geom_list: detection 리스트
            header: 메시지 헤더
        """
        out = DetectionArray()
        out.header.stamp = header.stamp
        out.header.frame_id = self.world_frame if self.world_frame else self.frame_id

        for geom in geom_list:
            # 월드 좌표가 없으면 스킵
            if not geom.get('center_world'):
                continue

            det = Detection()
            det.header.stamp = header.stamp
            det.header.frame_id = out.header.frame_id
            det.id = int(geom.get('id', 0))
            det.label = geom['label']
            det.confidence = float(geom['confidence'])

            # 위치 (월드 좌표계)
            cx, cy, cz = geom['center_world']
            det.pose.position.x = float(cx)
            det.pose.position.y = float(cy)
            det.pose.position.z = float(cz)
            det.pose.orientation.w = 1.0  # 회전 없음

            # 크기
            if geom.get('size_m'):
                sx, sy, sz = geom['size_m']
                det.size.x = float(sx)
                det.size.y = float(sy)
                det.size.z = float(sz)

            out.detections.append(det)

        self.pub_detections.publish(out)

    # ===== 콜백 함수 =====
    def on_rgb(self, msg: Image):
        """
        RGB 이미지 콜백 - YOLO 추론 수행

        이 함수에서 전체 파이프라인 실행:
        1. YOLO 추론
        2. 3D 기하학 추출
        3. Persistence tracking
        4. 시각화
        5. 메시지 발행
        """
        # FPS 제한 (max_fps 이하로 처리)
        now_t = time.monotonic()
        if (now_t - self._last_infer_t) < self.max_period:
            return
        self._last_infer_t = now_t

        # ROS Image -> OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        try:
            # ===== YOLO 추론 =====
            # TensorRT 또는 PyTorch 자동 선택
            results = self.model.predict(
                frame,
                conf=self.conf,  # 신뢰도 임계값
                iou=self.iou,  # NMS IOU 임계값
                imgsz=self.input_size,  # 입력 크기
                verbose=False,  # 로그 출력 안함
                device=self.device  # GPU 디바이스
            )

            # 검출 결과를 tensor로 변환 (N x 6) [x1, y1, x2, y2, conf, cls]
            det_tensor = torch.empty((0, 6), dtype=torch.float32)
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    det_tensor = torch.cat([
                        boxes.xyxy,  # bounding box 좌표
                        boxes.conf[:, None],  # confidence
                        boxes.cls[:, None]  # class index
                    ], dim=1).cpu()

            # ===== 3D 기하학 추출 =====
            raw_geoms = self.build_detection_geometry(det_tensor, frame.shape, msg.header.stamp)

            # ===== Persistence tracking =====
            stable_geoms = self.apply_persistence(raw_geoms)

            # ===== 시각화 =====
            # depth/TF가 없어도 디버그 이미지에 2D 박스는 그려준다
            if stable_geoms:
                annotated = self.draw_detections(frame.copy(), stable_geoms)
            else:
                annotated = self.draw_detections(frame.copy(), self.build_2d_geoms(det_tensor, frame.shape))

            # ===== 발행 =====
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
        """
        Depth 이미지 콜백 - 저장만 수행
        """
        self.last_depth = msg

    def on_info(self, msg: CameraInfo):
        """
        Camera info 콜백 - intrinsics 저장
        """
        self.camera_info = msg


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = YoloDepthMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
