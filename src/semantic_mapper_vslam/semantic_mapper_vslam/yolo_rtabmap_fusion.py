#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + RTAB-Map Fusion Node
Fuses YOLO detections with RTAB-Map point cloud for accurate size/position estimation
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from semantic_mapper_msgs.msg import DetectionArray

# 다운샘플링 매개변수 수정 의견(1 / 2로)
def pointcloud2_to_numpy(pc2: PointCloud2, stride: int = 4) -> np.ndarray:
    """Convert PointCloud2 to numpy array with optional downsampling"""
    pts = []
    for i, p in enumerate(point_cloud2.read_points(pc2, field_names=("x", "y", "z"), skip_nans=True)):
        if stride > 1 and (i % stride) != 0:
            continue
        pts.append([p[0], p[1], p[2]])

    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def sphere_crop(pts: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Crop points within a sphere"""
    if pts.size == 0:
        return pts
    d2 = np.sum((pts - center[None, :]) ** 2, axis=1)
    return pts[d2 <= radius * radius]


def robust_pca_size(local_pts: np.ndarray, percentile: float = 0.95) -> Optional[Tuple[float, float, float]]:
    """
    Compute robust 3D size using PCA
    Returns (length, width, height) along principal axes
    """
    n = local_pts.shape[0]
    if n < 30:
        return None

    # Center points
    center = np.mean(local_pts, axis=0, keepdims=True)
    X = local_pts - center

    # PCA
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # descending order
    R = eigvecs[:, order]

    # Project to principal axes
    Y = X @ R

    # Robust size estimation (percentile-based)
    q = (1.0 + percentile) * 0.5
    mins = np.quantile(Y, 1.0 - q, axis=0)
    maxs = np.quantile(Y, q, axis=0)
    spans = (maxs - mins).astype(float)
    spans = np.clip(spans, 0.05, 10.0)

    return float(spans[0]), float(spans[1]), float(spans[2])


def estimate_yaw_from_pca(local_pts: np.ndarray) -> Optional[float]:
    """
    Estimate yaw angle (rotation around Z-axis) from PCA
    Returns yaw in radians
    """
    n = local_pts.shape[0]
    if n < 30:
        return None

    # PCA on XY plane only
    center_xy = np.mean(local_pts[:, :2], axis=0, keepdims=True)
    X_xy = local_pts[:, :2] - center_xy

    cov_xy = np.cov(X_xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov_xy)

    # Principal direction (largest eigenvalue)
    principal_vec = eigvecs[:, 1] if eigvals[1] > eigvals[0] else eigvecs[:, 0]

    # Compute yaw
    yaw = np.arctan2(principal_vec[1], principal_vec[0])
    return float(yaw)


class YoloRtabmapFusion(Node):
    def __init__(self) -> None:
        super().__init__("yolo_rtabmap_fusion")

        # ===== Parameters =====
        self.declare_parameter("input_dets_topic", "/semantic_mapper/detections")
        self.declare_parameter("output_dets_topic", "/semantic_mapper/detections_fused")
        self.declare_parameter("map_cloud_topic", "/rtabmap/odom_local_map")  # Use local map instead of cloud_map

        # Fusion parameters
        self.declare_parameter("search_radius", 0.7)  # 객체 주변 탐색 반경 (m)
        self.declare_parameter("downsample_stride", 4)  # 포인트클라우드 다운샘플링
        self.declare_parameter("min_points", 50)  # 최소 포인트 개수
        self.declare_parameter("percentile", 0.95)  # 로버스트 크기 추정 백분위
        self.declare_parameter("adjust_position", True)  # 위치 보정 활성화
        self.declare_parameter("max_shift_m", 0.5)  # 최대 위치 이동 거리
        self.declare_parameter("estimate_orientation", True)  # 방향 추정 활성화
        self.declare_parameter("max_fps", 10.0)  # 최대 처리 속도

        # Z-axis filtering (스마트 Z 보정)
        self.declare_parameter("min_z", -0.3)  # 최소 Z 좌표 (바닥 아래 제거)
        self.declare_parameter("max_z", 3.0)   # 최대 Z 좌표 (천장 위 제거)
        self.declare_parameter("ground_z", 0.0)  # 바닥 Z 좌표
        self.declare_parameter("max_distance_for_z_snap", 3.0)  # Z 스냅 적용 최대 거리 (3m 이내만)
        self.declare_parameter("ground_snap_threshold", 0.15)  # 바닥 스냅 임계값 (15cm)

        # Read parameters
        self.input_dets_topic = self.get_parameter("input_dets_topic").value
        self.output_dets_topic = self.get_parameter("output_dets_topic").value
        self.map_cloud_topic = self.get_parameter("map_cloud_topic").value

        self.search_radius = float(self.get_parameter("search_radius").value)
        self.downsample_stride = int(self.get_parameter("downsample_stride").value)
        self.min_points = int(self.get_parameter("min_points").value)
        self.percentile = float(self.get_parameter("percentile").value)
        self.adjust_position = bool(self.get_parameter("adjust_position").value)
        self.max_shift_m = float(self.get_parameter("max_shift_m").value)
        self.estimate_orientation = bool(self.get_parameter("estimate_orientation").value)
        self.max_fps = float(self.get_parameter("max_fps").value)
        self.max_period = 1.0 / max(self.max_fps, 1.0)

        # Z-axis filtering parameters
        self.min_z = float(self.get_parameter("min_z").value)
        self.max_z = float(self.get_parameter("max_z").value)
        self.ground_z = float(self.get_parameter("ground_z").value)
        self.max_distance_for_z_snap = float(self.get_parameter("max_distance_for_z_snap").value)
        self.ground_snap_threshold = float(self.get_parameter("ground_snap_threshold").value)

        # ===== QoS =====
        qos_cloud = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_dets_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_dets_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ===== Data storage =====
        self._map_pts = np.empty((0, 3), dtype=np.float32)
        self._map_lock = threading.Lock()
        self._t_last = 0.0

        # ===== ROS I/O =====
        self.sub_cloud = self.create_subscription(
            PointCloud2, self.map_cloud_topic, self.on_cloud, qos_cloud
        )
        self.sub_dets = self.create_subscription(
            DetectionArray, self.input_dets_topic, self.on_dets, qos_dets_sub
        )
        self.pub_dets = self.create_publisher(
            DetectionArray, self.output_dets_topic, qos_dets_pub
        )

        self.get_logger().info(
            f"Fusion node initialized:\n"
            f"  Input detections: {self.input_dets_topic}\n"
            f"  RTAB-Map cloud: {self.map_cloud_topic}\n"
            f"  Output detections: {self.output_dets_topic}\n"
            f"  Search radius: {self.search_radius}m\n"
            f"  Downsample stride: {self.downsample_stride}\n"
            f"  Min points: {self.min_points}\n"
            f"  Adjust position: {self.adjust_position}\n"
            f"  Estimate orientation: {self.estimate_orientation}"
        )

    def on_cloud(self, msg: PointCloud2) -> None:
        """RTAB-Map point cloud callback"""
        try:
            pts = pointcloud2_to_numpy(msg, stride=self.downsample_stride)
            with self._map_lock:
                self._map_pts = pts

            if not hasattr(self, '_cloud_logged'):
                self._cloud_logged = True
                self.get_logger().info(f"Received RTAB-Map cloud: {pts.shape[0]} points")
        except Exception as e:
            self.get_logger().warn(f"Cloud parsing failed: {e}")

    def on_dets(self, msg: DetectionArray) -> None:
        """Detection callback - fuse with RTAB-Map cloud"""
        now = time.monotonic()
        if (now - self._t_last) < self.max_period:
            return
        self._t_last = now

        with self._map_lock:
            map_pts = self._map_pts

        if map_pts.size == 0:
            # No map available yet, publish original detections
            self.pub_dets.publish(msg)
            if not hasattr(self, '_no_map_warned'):
                self._no_map_warned = True
                self.get_logger().warn(
                    f"No RTAB-Map cloud available yet on {self.map_cloud_topic}. "
                    "Publishing original detections."
                )
            return

        # Create output message
        out = DetectionArray()
        out.header = msg.header
        out.detections = []

        for det in msg.detections:
            p = det.pose.position
            center = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)

            # Crop local point cloud around detection
            local = sphere_crop(map_pts, center, self.search_radius)

            if local.shape[0] >= self.min_points:
                # 1) Update size from local PCA
                scales = robust_pca_size(local, percentile=self.percentile)
                if scales is not None:
                    det.size.x, det.size.y, det.size.z = scales

                # 2) Adjust position toward robust local center
                if self.adjust_position:
                    c_median = np.median(local, axis=0).astype(np.float32)
                    shift = float(np.linalg.norm(c_median - center))

                    if shift > 1e-6:
                        # Clamp shift to avoid jumping to wrong cluster
                        if self.max_shift_m > 0.0 and shift > self.max_shift_m:
                            alpha = self.max_shift_m / shift
                            c_new = center + alpha * (c_median - center)
                        else:
                            c_new = c_median

                        det.pose.position.x = float(c_new[0])
                        det.pose.position.y = float(c_new[1])
                        det.pose.position.z = float(c_new[2])

            # ===== 스마트 Z-axis 보정 =====
            # 원리:
            # 1. 가까운 객체(3m 이내): 바닥 근처면 바닥에 스냅, 높이는 보존
            # 2. 먼 객체(3m 이상): Z값 보존 (멀리서는 depth 부정확하므로 건드리지 않음)
            # 3. 극단적 값만 클램핑 (바닥 아래 -0.3m, 천장 위 3.0m)

            z_pos = det.pose.position.z
            x_pos = det.pose.position.x
            y_pos = det.pose.position.y

            # 로봇으로부터의 거리 계산 (XY 평면)
            distance = np.sqrt(x_pos**2 + y_pos**2)

            # 극단적 값 클램핑 (항상 적용)
            if z_pos < self.min_z:
                # 바닥 아래는 바닥으로
                det.pose.position.z = self.ground_z
            elif z_pos > self.max_z:
                # 너무 높으면 max_z로 클램프
                det.pose.position.z = self.max_z

            # 가까운 거리에서만 바닥 스냅 적용
            elif distance < self.max_distance_for_z_snap:
                # 바닥 근처 임계값 이내면 바닥에 스냅
                if abs(z_pos - self.ground_z) < self.ground_snap_threshold:
                    det.pose.position.z = self.ground_z
                # 그 외에는 Z값 보존 (탁자 위 물체 등)

            # 3) Estimate orientation from PCA
            if self.estimate_orientation and local.shape[0] >= self.min_points:
                yaw = estimate_yaw_from_pca(local)
                if yaw is not None:
                    # Convert yaw to quaternion
                    half_yaw = yaw * 0.5
                    det.pose.orientation.x = 0.0
                    det.pose.orientation.y = 0.0
                    det.pose.orientation.z = float(np.sin(half_yaw))
                    det.pose.orientation.w = float(np.cos(half_yaw))

            out.detections.append(det)

        # Publish fused detections
        self.pub_dets.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloRtabmapFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
