#!/usr/bin/env python3
import math
import struct
from typing import Dict, Set, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class DynamicObjectFilterNode(Node):
    """
    Temporal voxel filter:
      - Dynamic objects: seen briefly -> removed
      - Static objects: seen repeatedly -> passed through
    """

    def __init__(self) -> None:
        super().__init__("dynamic_object_filter")

        self.declare_parameter("input_topic", "/livox/lidar/filtered")
        self.declare_parameter("output_topic", "/livox/lidar/static_filtered")
        self.declare_parameter("voxel_size", 0.10)
        self.declare_parameter("min_hits", 3)
        self.declare_parameter("hit_window_sec", 3.0)
        self.declare_parameter("max_stale_sec", 8.0)
        self.declare_parameter("z_min", -2.0)
        self.declare_parameter("z_max", 2.0)
        self.declare_parameter("min_range", 0.2)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.min_hits = int(self.get_parameter("min_hits").value)
        self.hit_window_sec = float(self.get_parameter("hit_window_sec").value)
        self.max_stale_sec = float(self.get_parameter("max_stale_sec").value)
        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)
        self.min_range = float(self.get_parameter("min_range").value)

        self.voxel_hits: Dict[Tuple[int, int, int], int] = {}
        self.voxel_last_seen: Dict[Tuple[int, int, int], float] = {}

        self.sub = self.create_subscription(
            PointCloud2, self.input_topic, self._callback, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(
            PointCloud2, self.output_topic, qos_profile_sensor_data
        )

        self.get_logger().info(
            "DynamicObjectFilter started: "
            f"{self.input_topic} -> {self.output_topic}, "
            f"voxel={self.voxel_size}, min_hits={self.min_hits}, "
            f"hit_window={self.hit_window_sec}s"
        )

    @staticmethod
    def _field_offset(msg: PointCloud2, name: str) -> int:
        for f in msg.fields:
            if f.name == name:
                return int(f.offset)
        return -1

    def _cleanup_stale(self, now_sec: float) -> None:
        stale = [
            k for k, t in self.voxel_last_seen.items()
            if (now_sec - t) > self.max_stale_sec
        ]
        for k in stale:
            self.voxel_last_seen.pop(k, None)
            self.voxel_hits.pop(k, None)

    def _voxel_key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        s = self.voxel_size
        return (int(math.floor(x / s)), int(math.floor(y / s)), int(math.floor(z / s)))

    def _callback(self, msg: PointCloud2) -> None:
        point_step = int(msg.point_step)
        total_points = int(msg.width) * int(msg.height)
        if point_step <= 0 or total_points <= 0:
            self.pub.publish(msg)
            return

        ox = self._field_offset(msg, "x")
        oy = self._field_offset(msg, "y")
        oz = self._field_offset(msg, "z")
        if ox < 0 or oy < 0 or oz < 0:
            self.get_logger().warn(
                "PointCloud2 has no x/y/z fields, bypassing.",
                throttle_duration_sec=5.0,
            )
            self.pub.publish(msg)
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._cleanup_stale(now_sec)

        frame_voxels: Set[Tuple[int, int, int]] = set()
        data = msg.data
        unpack_f = struct.unpack_from

        for i in range(total_points):
            base = i * point_step
            x = unpack_f("<f", data, base + ox)[0]
            y = unpack_f("<f", data, base + oy)[0]
            z = unpack_f("<f", data, base + oz)[0]

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if z < self.z_min or z > self.z_max:
                continue
            if (x * x + y * y) < (self.min_range * self.min_range):
                continue

            frame_voxels.add(self._voxel_key(x, y, z))

        for k in frame_voxels:
            last = self.voxel_last_seen.get(k)
            if last is None or (now_sec - last) > self.hit_window_sec:
                self.voxel_hits[k] = 1
            else:
                self.voxel_hits[k] = self.voxel_hits.get(k, 0) + 1
            self.voxel_last_seen[k] = now_sec

        static_voxels = {k for k in frame_voxels if self.voxel_hits.get(k, 0) >= self.min_hits}

        out_data = bytearray()
        kept = 0
        for i in range(total_points):
            base = i * point_step
            x = unpack_f("<f", data, base + ox)[0]
            y = unpack_f("<f", data, base + oy)[0]
            z = unpack_f("<f", data, base + oz)[0]
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if self._voxel_key(x, y, z) in static_voxels:
                out_data.extend(data[base: base + point_step])
                kept += 1

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = kept
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = kept * point_step
        out.is_dense = False
        out.data = bytes(out_data)
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DynamicObjectFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
