#!/usr/bin/env python3
import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def quat_conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def quat_multiply(
    q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_angle_rad(q: Tuple[float, float, float, float]) -> float:
    # q is relative quaternion
    w = max(-1.0, min(1.0, q[3]))
    return 2.0 * math.acos(abs(w))


@dataclass
class ChainState:
    name: str
    target: str
    source: str
    prev_tf: Optional[TransformStamped] = None
    miss_start: Optional[Time] = None
    samples: int = 0
    jumps: int = 0
    stale: int = 0


class TfJumpMonitor(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("tf_jump_monitor")
        self.args = args
        self.buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.buffer, self)
        self.chains = [
            ChainState("map->odom", "map", "odom"),
            ChainState("odom->base_link", "odom", "base_link"),
        ]

        self.csv_path = self._prepare_csv(args.output)
        self.csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self.csv = csv.writer(self.csv_file)
        if os.path.getsize(self.csv_path) == 0:
            self.csv.writerow(
                [
                    "wall_time",
                    "ros_time",
                    "chain",
                    "event",
                    "trans_delta_m",
                    "rot_delta_deg",
                    "age_sec",
                    "detail",
                ]
            )
            self.csv_file.flush()

        self.timer = self.create_timer(1.0 / args.rate, self.tick)
        self.last_report = self.get_clock().now()
        self.get_logger().info(f"TF jump monitor started. log={self.csv_path}")

    def _prepare_csv(self, out: str) -> str:
        if out:
            return out
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(os.getcwd(), f"tf_jump_log_{ts}.csv")

    def _log_event(
        self,
        chain: ChainState,
        event: str,
        trans_delta: float = 0.0,
        rot_delta_deg: float = 0.0,
        age_sec: float = 0.0,
        detail: str = "",
    ) -> None:
        now = self.get_clock().now()
        self.csv.writerow(
            [
                datetime.now().isoformat(timespec="milliseconds"),
                f"{now.nanoseconds / 1e9:.6f}",
                chain.name,
                event,
                f"{trans_delta:.6f}",
                f"{rot_delta_deg:.6f}",
                f"{age_sec:.6f}",
                detail,
            ]
        )
        self.csv_file.flush()

    def _lookup(self, chain: ChainState) -> Optional[TransformStamped]:
        try:
            return self.buffer.lookup_transform(chain.target, chain.source, Time())
        except TransformException as e:
            now = self.get_clock().now()
            if chain.miss_start is None:
                chain.miss_start = now
            miss_sec = (now - chain.miss_start).nanoseconds / 1e9
            if miss_sec >= self.args.max_miss:
                self._log_event(chain, "MISSING", detail=str(e))
                chain.miss_start = now
            return None

    def _process_chain(self, chain: ChainState) -> None:
        tf = self._lookup(chain)
        if tf is None:
            return

        chain.miss_start = None
        chain.samples += 1

        now_ns = self.get_clock().now().nanoseconds
        tf_ns = int(tf.header.stamp.sec) * 1_000_000_000 + int(tf.header.stamp.nanosec)
        age_sec = (now_ns - tf_ns) / 1e9
        if age_sec > self.args.max_age:
            chain.stale += 1
            self._log_event(chain, "STALE", age_sec=age_sec)

        if chain.prev_tf is not None:
            dx = tf.transform.translation.x - chain.prev_tf.transform.translation.x
            dy = tf.transform.translation.y - chain.prev_tf.transform.translation.y
            dz = tf.transform.translation.z - chain.prev_tf.transform.translation.z
            trans_delta = math.sqrt(dx * dx + dy * dy + dz * dz)

            q_prev = (
                chain.prev_tf.transform.rotation.x,
                chain.prev_tf.transform.rotation.y,
                chain.prev_tf.transform.rotation.z,
                chain.prev_tf.transform.rotation.w,
            )
            q_now = (
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            )
            q_rel = quat_multiply(quat_conjugate(q_prev), q_now)
            rot_delta_deg = math.degrees(quat_angle_rad(q_rel))

            if trans_delta >= self.args.jump_trans or rot_delta_deg >= self.args.jump_rot_deg:
                chain.jumps += 1
                self._log_event(
                    chain,
                    "JUMP",
                    trans_delta=trans_delta,
                    rot_delta_deg=rot_delta_deg,
                    age_sec=age_sec,
                )

        chain.prev_tf = tf

    def tick(self) -> None:
        for c in self.chains:
            self._process_chain(c)

        now = self.get_clock().now()
        if (now - self.last_report).nanoseconds / 1e9 >= self.args.report_sec:
            parts = []
            for c in self.chains:
                parts.append(
                    f"{c.name}: samples={c.samples}, jumps={c.jumps}, stale={c.stale}"
                )
            self.get_logger().info(" | ".join(parts))
            self.last_report = now

    def destroy_node(self) -> bool:
        try:
            self.csv_file.close()
        except Exception:
            pass
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect TF jumps/missing/stale for map->odom and odom->base_link."
    )
    p.add_argument("--rate", type=float, default=30.0, help="Sampling rate (Hz)")
    p.add_argument(
        "--jump-trans", type=float, default=0.15, help="Jump threshold for translation (m)"
    )
    p.add_argument(
        "--jump-rot-deg", type=float, default=8.0, help="Jump threshold for rotation (deg)"
    )
    p.add_argument(
        "--max-age", type=float, default=0.10, help="Stale TF threshold by age (sec)"
    )
    p.add_argument(
        "--max-miss",
        type=float,
        default=0.50,
        help="Log MISSING event every N sec while transform is unavailable",
    )
    p.add_argument(
        "--report-sec", type=float, default=5.0, help="Console summary period (sec)"
    )
    p.add_argument("--output", default="", help="CSV output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = TfJumpMonitor(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
