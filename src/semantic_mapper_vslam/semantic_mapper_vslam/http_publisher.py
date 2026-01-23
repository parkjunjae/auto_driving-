#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP Publisher for Semantic Mapping
Sends detection data to remote BIM server via REST API
"""

import json
import time
import math
from typing import Optional, Dict, List
from threading import Lock, Thread
from queue import Queue, Full

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from rclpy.time import Time
from rclpy.duration import Duration

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np

from semantic_mapper_msgs.msg import DetectionArray, Detection


# ===== 클래스 -> GLB 매핑 =====
# YOLOv5 커스텀 클래스를 서버의 GLB 파일명으로 매핑
CLASS_TO_GLB = {
    # 직접 매핑
    'tv': 'tv',
    'monitor': 'tv',  # 모니터도 tv.glb 사용
    'chair': 'chair',
    'sofa': 'couch',  # sofa는 couch.glb 사용
    'table': 'table',

    # 신규 추가된 GLB
    'bag': 'bag',
    'bookcase': 'bookcase',
    'cabinet': 'cabinet',
    'chest of drawers': 'chest_of_drawers',
    'golf bag': 'golf_bag',
    'whiteboard': 'whiteboard',
    'clock': 'clock',
}


def normalize_label_for_glb(label: str) -> str:
    """
    YOLO 클래스 이름을 GLB 파일명으로 변환

    Args:
        label: YOLO detection label

    Returns:
        GLB 파일명 (확장자 제외)
    """
    return CLASS_TO_GLB.get(label, label)


class HTTPPublisher(Node):
    def __init__(self):
        super().__init__('http_publisher')

        # ===== Parameters =====
        self.declare_parameter("server_url", "http://192.168.1.100:8000")
        # 중복 제거된 최종 객체 토픽 사용 (object_deduplicator 출력)
        self.declare_parameter("detection_topic", "/semantic_mapper/objects")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("robot_id", "jetson_orin_nx_01")
        self.declare_parameter("pose_target_frame", "map")  # 로봇 포즈를 변환할 목표 프레임
        self.declare_parameter("pose_source_frame", "base_link")  # 로봇 포즈의 소스 프레임 (TF 조회용)
        self.declare_parameter("tf_timeout", 0.2)  # TF 조회 타임아웃 (초)
        self.declare_parameter("tf_use_latest", True)

        # 전송 설정
        self.declare_parameter("batch_size", 10)  # N개 모아서 전송
        self.declare_parameter("batch_timeout", 2.0)  # N초마다 강제 전송
        self.declare_parameter("max_queue_size", 100)  # 큐 최대 크기
        self.declare_parameter("retry_attempts", 3)
        self.declare_parameter("timeout", 5.0)

        # ===== Read Parameters =====
        self.server_url = self.get_parameter("server_url").value
        self.detection_topic = self.get_parameter("detection_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.robot_id = self.get_parameter("robot_id").value
        self.pose_target_frame = self.get_parameter("pose_target_frame").value
        self.pose_source_frame = self.get_parameter("pose_source_frame").value
        self.tf_timeout = float(self.get_parameter("tf_timeout").value)
        self.tf_use_latest = bool(self.get_parameter("tf_use_latest").value)

        self.batch_size = int(self.get_parameter("batch_size").value)
        self.batch_timeout = float(self.get_parameter("batch_timeout").value)
        self.max_queue_size = int(self.get_parameter("max_queue_size").value)
        self.retry_attempts = int(self.get_parameter("retry_attempts").value)
        self.timeout = float(self.get_parameter("timeout").value)

        # ===== HTTP Session with Retry =====
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # ===== State =====
        self.current_odom: Optional[Odometry] = None
        self.odom_lock = Lock()

        self.detection_queue: Queue = Queue(maxsize=self.max_queue_size)
        self.stats = {
            'total_sent': 0,
            'total_failed': 0,
            'total_detections': 0,
        }
        self.stats_lock = Lock()
        self._last_tf_warn = 0.0

        # ===== ROS I/O =====
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.sub_detections = self.create_subscription(
            DetectionArray, self.detection_topic, self.on_detections, 10)
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.on_odom, qos_sensor)

        # TF 리스너 (로봇 포즈를 detection frame으로 변환)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # ===== Background Thread =====
        self.running = True
        self.sender_thread = Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()

        # ===== Stats Timer =====
        self.create_timer(10.0, self._print_stats)

        self.get_logger().info(f"HTTPPublisher initialized. Server: {self.server_url}")
        self.get_logger().info(f"Robot ID: {self.robot_id}, Batch: {self.batch_size}, Timeout: {self.batch_timeout}s")

    # ===== TF Helpers =====
    @staticmethod
    def _quat_to_matrix(q) -> np.ndarray:
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-9:
            return np.identity(3, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        return np.array([
            [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
        ], dtype=np.float64)

    @staticmethod
    def _quat_multiply(q1, q2):
        """q = q1 * q2 (geometry_msgs Quaternion 둘 다 허용)"""
        x1, y1, z1, w1 = float(q1.x), float(q1.y), float(q1.z), float(q1.w)
        x2, y2, z2, w2 = float(q2.x), float(q2.y), float(q2.z), float(q2.w)
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        )

    def _transform_pose_to_frame(self, pose, target_frame: str, source_frame: str, stamp):
        """odom 포즈를 detection frame(map)으로 변환"""
        if not target_frame or target_frame == source_frame:
            return (
                (pose.position.x, pose.position.y, pose.position.z),
                (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            )

        try:
            query_time = Time() if self.tf_use_latest else Time.from_msg(stamp)
        except Exception:
            query_time = Time()

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                query_time,
                timeout=Duration(seconds=self.tf_timeout)
            )
        except TransformException as e:
            now = time.monotonic()
            if now - self._last_tf_warn > 5.0:
                self.get_logger().warn(f"TF 변환 실패({source_frame}->{target_frame}), 해당 프레임 스킵: {e}")
                self._last_tf_warn = now
            return None, None

        trans = transform.transform.translation
        rot = transform.transform.rotation
        rot_m = self._quat_to_matrix(rot)
        pos_vec = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
        trans_vec = np.array([trans.x, trans.y, trans.z], dtype=np.float64)
        pos_tf = rot_m @ pos_vec + trans_vec
        ori_tf = self._quat_multiply(rot, pose.orientation)

        return (
            (float(pos_tf[0]), float(pos_tf[1]), float(pos_tf[2])),
            (float(ori_tf[0]), float(ori_tf[1]), float(ori_tf[2]), float(ori_tf[3]))
        )

    def on_odom(self, msg: Odometry):
        """Update current odometry"""
        with self.odom_lock:
            self.current_odom = msg

    def on_detections(self, msg: DetectionArray):
        """Queue detections for batch sending"""
        if len(msg.detections) == 0:
            return

        # Get current robot pose
        with self.odom_lock:
            odom = self.current_odom

        if odom is None:
            self.get_logger().warn("No odometry available, skipping detections")
            return

        target_frame = msg.header.frame_id or self.pose_target_frame
        # 기본은 base_link -> target_frame TF를 사용, 비어있으면 odom 헤더 프레임 사용
        source_frame = self.pose_source_frame or odom.header.frame_id or target_frame
        target_frame = target_frame or source_frame

        # 로봇 포즈를 detection frame(기본 map)으로 변환
        # base_link 사용 시에는 단위 포즈(0,0,0 + 단위 쿼터니언)를 TF로 변환
        pose_for_tf = odom.pose.pose if source_frame == (odom.header.frame_id or "") else Pose()
        pos_tf, ori_tf = self._transform_pose_to_frame(
            pose_for_tf,
            target_frame,
            source_frame,
            odom.header.stamp
        )
        if pos_tf is None or ori_tf is None:
            # TF 실패 시 해당 프레임 스킵 (잘못된 좌표 전송 방지)
            return

        # Convert to JSON-serializable format
        data = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'robot_id': self.robot_id,
            'frame_id': target_frame,
            'robot_pose': {
                'position': {
                    'x': pos_tf[0],
                    'y': pos_tf[1],
                    'z': pos_tf[2],
                },
                'orientation': {
                    'x': ori_tf[0],
                    'y': ori_tf[1],
                    'z': ori_tf[2],
                    'w': ori_tf[3],
                }
            },
            'detections': []
        }

        for det in msg.detections:
            # YOLO 클래스를 GLB 파일명으로 매핑
            glb_label = normalize_label_for_glb(det.label)

            detection = {
                'id': det.id,
                'label': glb_label,  # GLB 매핑된 레이블 사용
                'original_label': det.label,  # 원본 레이블도 함께 전송
                'confidence': det.confidence,
                'position': {
                    'x': det.pose.position.x,
                    'y': det.pose.position.y,
                    'z': det.pose.position.z,
                },
                'orientation': {
                    'x': det.pose.orientation.x,
                    'y': det.pose.orientation.y,
                    'z': det.pose.orientation.z,
                    'w': det.pose.orientation.w,
                },
                'size': {
                    'x': det.size.x,
                    'y': det.size.y,
                    'z': det.size.z,
                }
            }
            data['detections'].append(detection)

        # Add to queue
        try:
            self.detection_queue.put_nowait(data)
            with self.stats_lock:
                self.stats['total_detections'] += len(msg.detections)
        except Full:
            self.get_logger().warn("Detection queue full, dropping data")

    def _sender_loop(self):
        """Background thread for batched HTTP sending"""
        batch: List[Dict] = []
        last_send_time = time.monotonic()

        while self.running:
            try:
                # Wait for data with timeout
                timeout_remaining = self.batch_timeout - (time.monotonic() - last_send_time)
                if timeout_remaining < 0.1:
                    timeout_remaining = 0.1

                try:
                    data = self.detection_queue.get(timeout=timeout_remaining)
                    batch.append(data)
                except:
                    pass  # Timeout - check if we should send

                now = time.monotonic()
                should_send = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and (now - last_send_time) >= self.batch_timeout)
                )

                if should_send:
                    self._send_batch(batch)
                    batch = []
                    last_send_time = now

            except Exception as e:
                self.get_logger().error(f"Sender loop error: {e}")
                time.sleep(1.0)

    def _send_batch(self, batch: List[Dict]):
        """Send batch of detections to server"""
        if not batch:
            return

        payload = {
            'robot_id': self.robot_id,
            'batch': batch
        }

        try:
            url = f"{self.server_url}/api/detections"
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            with self.stats_lock:
                self.stats['total_sent'] += len(batch)

            # Debug: Log first robot orientation
            if batch:
                first_orient = batch[0]['robot_pose']['orientation']
                self.get_logger().info(
                    f"Sent batch of {len(batch)} frames "
                    f"({sum(len(b['detections']) for b in batch)} objects) "
                    f"to server [HTTP {response.status_code}] "
                    f"Robot quat: ({first_orient['x']:.4f}, {first_orient['y']:.4f}, "
                    f"{first_orient['z']:.4f}, {first_orient['w']:.4f})"
                )

        except requests.exceptions.RequestException as e:
            with self.stats_lock:
                self.stats['total_failed'] += len(batch)
            self.get_logger().error(f"Failed to send batch: {e}")

    def _print_stats(self):
        """Print statistics"""
        with self.stats_lock:
            self.get_logger().info(
                f"Stats: Sent={self.stats['total_sent']} frames, "
                f"Failed={self.stats['total_failed']} frames, "
                f"Total objects={self.stats['total_detections']}, "
                f"Queue size={self.detection_queue.qsize()}"
            )

    def destroy_node(self):
        """Cleanup"""
        self.running = False
        if self.sender_thread.is_alive():
            self.sender_thread.join(timeout=2.0)
        self.session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HTTPPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
