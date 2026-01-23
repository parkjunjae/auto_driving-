#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Deduplicator for Semantic VSLAM
영구 객체 저장소 + KD-Tree 기반 중복 제거

이 노드는 다음 문제를 해결합니다:
- 로봇이 재방문할 때 동일 객체가 중복 생성되는 문제
- 일시적 tracking이 아닌 영구적 객체 관리

주요 기능:
1. 영구 객체 저장소 (Global Object Database)
2. KD-Tree 기반 빠른 공간 검색
3. 다중 조건 매칭 (라벨 + 거리 + 크기 유사도)
4. 신뢰도 누적 (자주 관측된 객체 = 높은 신뢰도)
5. EMA 기반 점진적 위치/크기 업데이트
6. 최종 확정 객체만 발행
"""

import time
import math
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray

from semantic_mapper_msgs.msg import DetectionArray, Detection

# scipy KD-Tree (없으면 순차 검색으로 fallback)
try:
    from scipy.spatial import KDTree
    HAS_KDTREE = True
except ImportError:
    HAS_KDTREE = False


@dataclass
class GlobalObject:
    """영구 저장되는 객체 정보"""
    obj_id: int
    label: str
    position: np.ndarray  # (x, y, z) in world frame
    size: np.ndarray  # (width, height, depth)
    orientation: np.ndarray  # (x, y, z, w) quaternion
    confidence: float

    # 통계 정보
    observation_count: int = 1  # 관측 횟수
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # 상태
    is_confirmed: bool = False  # min_observations 이상 관측되었는지


class ObjectDeduplicator(Node):
    """
    객체 중복 제거 노드

    입력: /semantic_mapper/detections_fused (fusion 후 정제된 detection)
    출력: /semantic_mapper/objects (중복 제거된 최종 객체)
    """

    def __init__(self):
        super().__init__('object_deduplicator')

        # ===== Parameters =====
        # 토픽 설정
        self.declare_parameter("input_topic", "/semantic_mapper/detections_fused")
        self.declare_parameter("output_topic", "/semantic_mapper/objects")
        self.declare_parameter("marker_topic", "/semantic_mapper/objects_marker")

        # 매칭 파라미터 (SLAM drift 고려하여 넓게 설정)
        self.declare_parameter("match_distance", 0.8)  # 같은 객체로 판단할 최대 거리 (m) - 0.8m로 증가
        self.declare_parameter("match_size_threshold", 0.3)  # 크기 유사도 임계값 (0~1) - 0.3으로 낮춤
        self.declare_parameter("match_label_required", True)  # 라벨이 같아야 매칭

        # 객체 관리 파라미터
        self.declare_parameter("min_observations", 3)  # 확정까지 필요한 최소 관측 횟수
        self.declare_parameter("ema_alpha", 0.3)  # EMA 업데이트 계수 (0.3 = 30% 새 값)
        self.declare_parameter("max_objects", 500)  # 최대 저장 객체 수
        self.declare_parameter("stale_timeout", 300.0)  # 오래된 객체 제거 시간 (초)

        # 출력 설정
        self.declare_parameter("publish_all", False)  # True면 미확정 객체도 발행
        self.declare_parameter("max_fps", 10.0)

        # ===== Read Parameters =====
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value

        self.match_distance = float(self.get_parameter("match_distance").value)
        self.match_size_threshold = float(self.get_parameter("match_size_threshold").value)
        self.match_label_required = bool(self.get_parameter("match_label_required").value)

        self.min_observations = int(self.get_parameter("min_observations").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.max_objects = int(self.get_parameter("max_objects").value)
        self.stale_timeout = float(self.get_parameter("stale_timeout").value)

        self.publish_all = bool(self.get_parameter("publish_all").value)
        self.max_fps = float(self.get_parameter("max_fps").value)
        self.min_period = 1.0 / max(self.max_fps, 1.0)

        # ===== State =====
        self.objects: Dict[int, GlobalObject] = {}  # obj_id -> GlobalObject
        self.next_id = 1
        self.objects_lock = threading.Lock()

        # 라벨별 객체 인덱스 (빠른 검색용)
        self.label_index: Dict[str, List[int]] = {}  # label -> [obj_id, ...]

        # KD-Tree (라벨별로 별도 관리)
        self.kdtrees: Dict[str, Tuple[Optional[KDTree], List[int]]] = {}  # label -> (tree, obj_ids)
        self.kdtree_dirty: Dict[str, bool] = {}  # label -> needs_rebuild

        self._last_process_time = 0.0
        self._last_cleanup_time = time.time()

        # ===== 통계 =====
        self.stats = {
            'total_input': 0,
            'total_matched': 0,
            'total_created': 0,
            'total_published': 0,
        }

        # ===== ROS I/O =====
        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_detections = self.create_subscription(
            DetectionArray, self.input_topic, self.on_detections, qos_sub
        )
        self.pub_objects = self.create_publisher(
            DetectionArray, self.output_topic, qos_pub
        )
        self.pub_markers = self.create_publisher(
            MarkerArray, self.marker_topic, 10
        )

        # 주기적 정리 타이머 (30초마다)
        self.create_timer(30.0, self.cleanup_stale_objects)

        # 통계 출력 타이머 (60초마다)
        self.create_timer(60.0, self.print_stats)

        self.get_logger().info(
            f"ObjectDeduplicator initialized:\n"
            f"  Input: {self.input_topic}\n"
            f"  Output: {self.output_topic}\n"
            f"  Match distance: {self.match_distance}m\n"
            f"  Min observations: {self.min_observations}\n"
            f"  EMA alpha: {self.ema_alpha}\n"
            f"  KD-Tree available: {HAS_KDTREE}"
        )

    # ===== KD-Tree 관리 =====
    def _rebuild_kdtree(self, label: str):
        """특정 라벨의 KD-Tree 재구축"""
        obj_ids = self.label_index.get(label, [])
        if not obj_ids:
            self.kdtrees[label] = (None, [])
            self.kdtree_dirty[label] = False
            return

        positions = []
        valid_ids = []
        for obj_id in obj_ids:
            obj = self.objects.get(obj_id)
            if obj is not None:
                positions.append(obj.position)
                valid_ids.append(obj_id)

        if not positions:
            self.kdtrees[label] = (None, [])
        elif HAS_KDTREE:
            tree = KDTree(np.array(positions))
            self.kdtrees[label] = (tree, valid_ids)
        else:
            self.kdtrees[label] = (None, valid_ids)

        self.kdtree_dirty[label] = False

    def _find_nearest(self, label: str, position: np.ndarray) -> Tuple[Optional[int], float]:
        """
        주어진 위치에서 가장 가까운 객체 찾기

        Returns:
            (obj_id, distance) 또는 (None, inf)
        """
        # KD-Tree 재구축 필요하면 수행
        if self.kdtree_dirty.get(label, True):
            self._rebuild_kdtree(label)

        tree_data = self.kdtrees.get(label)
        if tree_data is None:
            return None, float('inf')

        tree, obj_ids = tree_data
        if not obj_ids:
            return None, float('inf')

        if HAS_KDTREE and tree is not None:
            # KD-Tree 사용
            dist, idx = tree.query(position, k=1)
            if idx < len(obj_ids):
                return obj_ids[idx], float(dist)
        else:
            # 순차 검색 fallback
            best_id = None
            best_dist = float('inf')
            for obj_id in obj_ids:
                obj = self.objects.get(obj_id)
                if obj is not None:
                    dist = np.linalg.norm(obj.position - position)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = obj_id
            return best_id, best_dist

        return None, float('inf')

    # ===== 객체 매칭 =====
    def compute_size_similarity(self, obj: GlobalObject, det_size: np.ndarray) -> float:
        """
        크기 유사도 계산 (0~1, 1이 완전히 같음)

        IoU와 유사한 개념으로 크기 비교
        """
        obj_vol = np.prod(np.maximum(obj.size, 0.01))
        det_vol = np.prod(np.maximum(det_size, 0.01))

        # 각 축별 최소/최대
        min_dims = np.minimum(obj.size, det_size)

        # 겹치는 부피 추정 (각 축의 최소값 곱)
        intersection_vol = np.prod(np.maximum(min_dims, 0.01))

        # Union
        union_vol = obj_vol + det_vol - intersection_vol

        if union_vol < 1e-9:
            return 1.0

        return float(intersection_vol / union_vol)

    def match_detection(self, label: str, position: np.ndarray,
                       size: np.ndarray) -> Optional[int]:
        """
        Detection을 기존 객체와 매칭

        Returns:
            매칭된 obj_id 또는 None
        """
        # 1. 라벨이 같은 객체 중 가장 가까운 것 찾기
        nearest_id, distance = self._find_nearest(label, position)

        if nearest_id is None:
            return None

        # 2. 거리 체크
        if distance > self.match_distance:
            return None

        # 3. 크기 유사도 체크
        obj = self.objects.get(nearest_id)
        if obj is None:
            return None

        size_sim = self.compute_size_similarity(obj, size)
        if size_sim < self.match_size_threshold:
            # 크기가 너무 다르면 다른 객체로 판단
            return None

        return nearest_id

    # ===== 객체 업데이트/생성 =====
    def update_object(self, obj_id: int, detection: Detection):
        """기존 객체 업데이트 (EMA)"""
        obj = self.objects.get(obj_id)
        if obj is None:
            return

        # 새 값 추출
        new_pos = np.array([
            detection.pose.position.x,
            detection.pose.position.y,
            detection.pose.position.z
        ])
        new_size = np.array([
            detection.size.x,
            detection.size.y,
            detection.size.z
        ])
        new_ori = np.array([
            detection.pose.orientation.x,
            detection.pose.orientation.y,
            detection.pose.orientation.z,
            detection.pose.orientation.w
        ])

        # EMA 업데이트
        alpha = self.ema_alpha
        obj.position = alpha * new_pos + (1 - alpha) * obj.position
        obj.size = alpha * new_size + (1 - alpha) * obj.size

        # Quaternion SLERP 대신 간단한 보간 (normalize 필요)
        obj.orientation = alpha * new_ori + (1 - alpha) * obj.orientation
        norm = np.linalg.norm(obj.orientation)
        if norm > 1e-9:
            obj.orientation /= norm

        # Confidence는 최대값 사용
        obj.confidence = max(obj.confidence, detection.confidence)

        # 관측 횟수 증가
        obj.observation_count += 1
        obj.last_seen = time.time()

        # 확정 상태 업데이트
        if obj.observation_count >= self.min_observations:
            obj.is_confirmed = True

        # KD-Tree 재구축 필요 표시 (위치가 변경됨)
        self.kdtree_dirty[obj.label] = True

    def create_object(self, detection: Detection) -> int:
        """새 객체 생성"""
        obj_id = self.next_id
        self.next_id += 1

        now = time.time()
        obj = GlobalObject(
            obj_id=obj_id,
            label=detection.label,
            position=np.array([
                detection.pose.position.x,
                detection.pose.position.y,
                detection.pose.position.z
            ]),
            size=np.array([
                detection.size.x,
                detection.size.y,
                detection.size.z
            ]),
            orientation=np.array([
                detection.pose.orientation.x,
                detection.pose.orientation.y,
                detection.pose.orientation.z,
                detection.pose.orientation.w
            ]),
            confidence=detection.confidence,
            observation_count=1,
            first_seen=now,
            last_seen=now,
            is_confirmed=False
        )

        # 저장
        self.objects[obj_id] = obj

        # 라벨 인덱스 업데이트
        if detection.label not in self.label_index:
            self.label_index[detection.label] = []
        self.label_index[detection.label].append(obj_id)

        # KD-Tree 재구축 필요 표시
        self.kdtree_dirty[detection.label] = True

        return obj_id

    # ===== 메인 처리 =====
    def on_detections(self, msg: DetectionArray):
        """Detection 메시지 콜백"""
        # FPS 제한
        now = time.monotonic()
        if (now - self._last_process_time) < self.min_period:
            return
        self._last_process_time = now

        if len(msg.detections) == 0:
            return

        with self.objects_lock:
            self.stats['total_input'] += len(msg.detections)

            # 각 detection 처리
            for det in msg.detections:
                position = np.array([
                    det.pose.position.x,
                    det.pose.position.y,
                    det.pose.position.z
                ])
                size = np.array([
                    det.size.x,
                    det.size.y,
                    det.size.z
                ])

                # 매칭 시도
                matched_id = self.match_detection(det.label, position, size)

                if matched_id is not None:
                    # 기존 객체 업데이트
                    self.update_object(matched_id, det)
                    self.stats['total_matched'] += 1
                else:
                    # 새 객체 생성
                    self.create_object(det)
                    self.stats['total_created'] += 1

            # 최대 객체 수 제한 체크
            if len(self.objects) > self.max_objects:
                self._remove_oldest_objects(len(self.objects) - self.max_objects)

        # 결과 발행
        self.publish_objects(msg.header)

    def _remove_oldest_objects(self, count: int):
        """가장 오래된 객체 제거"""
        if count <= 0:
            return

        # last_seen 기준 정렬
        sorted_objs = sorted(
            self.objects.values(),
            key=lambda o: o.last_seen
        )

        for obj in sorted_objs[:count]:
            self._remove_object(obj.obj_id)

    def _remove_object(self, obj_id: int):
        """객체 제거"""
        obj = self.objects.pop(obj_id, None)
        if obj is None:
            return

        # 라벨 인덱스에서 제거
        if obj.label in self.label_index:
            try:
                self.label_index[obj.label].remove(obj_id)
            except ValueError:
                pass

        # KD-Tree 재구축 필요 표시
        self.kdtree_dirty[obj.label] = True

    def cleanup_stale_objects(self):
        """오래된 객체 정리 (타이머 콜백)"""
        now = time.time()

        with self.objects_lock:
            stale_ids = []
            for obj_id, obj in self.objects.items():
                age = now - obj.last_seen
                # 미확정 객체는 더 빨리 제거 (60초)
                timeout = self.stale_timeout if obj.is_confirmed else 60.0
                if age > timeout:
                    stale_ids.append(obj_id)

            for obj_id in stale_ids:
                self._remove_object(obj_id)

            if stale_ids:
                self.get_logger().info(f"Cleaned up {len(stale_ids)} stale objects")

    # ===== 발행 =====
    def publish_objects(self, header):
        """확정된 객체들을 발행"""
        out = DetectionArray()
        out.header = header

        markers = []
        marker_id = 0

        with self.objects_lock:
            for obj in self.objects.values():
                # 미확정 객체는 publish_all이 True일 때만 발행
                if not obj.is_confirmed and not self.publish_all:
                    continue

                # Detection 메시지 생성
                det = Detection()
                det.header = header
                det.id = obj.obj_id
                det.label = obj.label
                det.confidence = obj.confidence

                det.pose.position.x = float(obj.position[0])
                det.pose.position.y = float(obj.position[1])
                det.pose.position.z = float(obj.position[2])

                det.pose.orientation.x = float(obj.orientation[0])
                det.pose.orientation.y = float(obj.orientation[1])
                det.pose.orientation.z = float(obj.orientation[2])
                det.pose.orientation.w = float(obj.orientation[3])

                det.size.x = float(obj.size[0])
                det.size.y = float(obj.size[1])
                det.size.z = float(obj.size[2])

                out.detections.append(det)

                # RViz 마커 생성
                marker = self._create_marker(obj, header, marker_id)
                markers.append(marker)

                # 텍스트 마커 (라벨 + 관측 횟수)
                text_marker = self._create_text_marker(obj, header, marker_id + 1000)
                markers.append(text_marker)

                marker_id += 1

        self.pub_objects.publish(out)
        self.pub_markers.publish(MarkerArray(markers=markers))

        self.stats['total_published'] = len(out.detections)

    def _create_marker(self, obj: GlobalObject, header, marker_id: int) -> Marker:
        """객체용 큐브 마커 생성"""
        marker = Marker()
        marker.header = header
        marker.ns = 'objects'
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(obj.position[0])
        marker.pose.position.y = float(obj.position[1])
        marker.pose.position.z = float(obj.position[2])

        marker.pose.orientation.x = float(obj.orientation[0])
        marker.pose.orientation.y = float(obj.orientation[1])
        marker.pose.orientation.z = float(obj.orientation[2])
        marker.pose.orientation.w = float(obj.orientation[3])

        marker.scale.x = float(max(obj.size[0], 0.05))
        marker.scale.y = float(max(obj.size[1], 0.05))
        marker.scale.z = float(max(obj.size[2], 0.05))

        # 확정 객체는 녹색, 미확정은 노란색
        if obj.is_confirmed:
            marker.color.r = 0.0
            marker.color.g = 0.8
            marker.color.b = 0.2
            marker.color.a = 0.6
        else:
            marker.color.r = 1.0
            marker.color.g = 0.8
            marker.color.b = 0.0
            marker.color.a = 0.4

        marker.lifetime.sec = 1

        return marker

    def _create_text_marker(self, obj: GlobalObject, header, marker_id: int) -> Marker:
        """객체 라벨 텍스트 마커 생성"""
        marker = Marker()
        marker.header = header
        marker.ns = 'object_labels'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(obj.position[0])
        marker.pose.position.y = float(obj.position[1])
        marker.pose.position.z = float(obj.position[2] + obj.size[2] / 2 + 0.15)

        marker.scale.z = 0.15  # 텍스트 크기

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # 라벨 + 관측 횟수 표시
        status = "✓" if obj.is_confirmed else "?"
        marker.text = f"{obj.label} [{obj.observation_count}] {status}"

        marker.lifetime.sec = 1

        return marker

    def print_stats(self):
        """통계 출력 (타이머 콜백)"""
        with self.objects_lock:
            confirmed = sum(1 for o in self.objects.values() if o.is_confirmed)
            total = len(self.objects)

        self.get_logger().info(
            f"Stats: Input={self.stats['total_input']}, "
            f"Matched={self.stats['total_matched']}, "
            f"Created={self.stats['total_created']}, "
            f"Objects={confirmed}/{total} confirmed"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDeduplicator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
