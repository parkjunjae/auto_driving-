#!/usr/bin/env python3
import argparse
import math
import statistics
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class OdomSample:
    t_wall: float
    x: float
    y: float
    yaw: float
    vx: float
    wz: float


class OdomCollector(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("odom_cov_estimator")
        self._lock = threading.Lock()
        self._latest: Optional[OdomSample] = None
        self._samples: List[OdomSample] = []
        self.create_subscription(Odometry, topic, self._cb, 50)
        self.get_logger().info(f"Subscribing: {topic}")

    def _cb(self, msg: Odometry) -> None:
        s = OdomSample(
            t_wall=time.time(),
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=yaw_from_quat(
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            ),
            vx=float(msg.twist.twist.linear.x),
            wz=float(msg.twist.twist.angular.z),
        )
        with self._lock:
            self._latest = s
            self._samples.append(s)

    def latest(self) -> Optional[OdomSample]:
        with self._lock:
            return self._latest

    def samples_in_last(self, sec: float) -> List[OdomSample]:
        t0 = time.time() - sec
        with self._lock:
            return [s for s in self._samples if s.t_wall >= t0]


def variance_or_zero(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pvariance(values)


def prompt(msg: str) -> None:
    input(f"\n{msg}\n  -> Press Enter")


def wait_for_odom(node: OdomCollector, timeout_sec: float = 10.0) -> None:
    t0 = time.time()
    while node.latest() is None:
        if time.time() - t0 > timeout_sec:
            raise RuntimeError("No /odom received. Check topic name and publishers.")
        time.sleep(0.05)


def measure_static(node: OdomCollector, duration: float) -> tuple[float, float]:
    print(f"\n[1/3] Static phase: keep robot still for {duration:.0f}s...")
    time.sleep(duration)
    samples = node.samples_in_last(duration)
    if len(samples) < 5:
        raise RuntimeError("Not enough samples in static phase.")
    vx_var = variance_or_zero([s.vx for s in samples])
    wz_var = variance_or_zero([s.wz for s in samples])
    print(f"  samples={len(samples)} vx_var={vx_var:.8f} wz_var={wz_var:.8f}")
    return vx_var, wz_var


def measure_roundtrip_xy(node: OdomCollector, cycles: int) -> tuple[float, float]:
    print("\n[2/3] Straight round-trip phase (forward 5m + backward 5m)")
    print("For each cycle: mark start -> drive forward/back -> mark end.")
    ex: List[float] = []
    ey: List[float] = []
    for i in range(1, cycles + 1):
        prompt(f"Cycle {i}/{cycles}: at START point")
        s0 = node.latest()
        prompt(f"Cycle {i}/{cycles}: finished round-trip, at END point")
        s1 = node.latest()
        if s0 is None or s1 is None:
            raise RuntimeError("No odom sample captured in cycle.")
        ex_i = s1.x - s0.x
        ey_i = s1.y - s0.y
        ex.append(ex_i)
        ey.append(ey_i)
        print(f"  cycle {i}: ex={ex_i:.4f} ey={ey_i:.4f}")
    return variance_or_zero(ex), variance_or_zero(ey)


def measure_spin_yaw(node: OdomCollector, cycles: int) -> float:
    print("\n[3/3] In-place spin phase (360deg left/right)")
    print("For each cycle: mark start -> spin 360deg -> mark end.")
    eyaw: List[float] = []
    for i in range(1, cycles + 1):
        prompt(f"Cycle {i}/{cycles}: before 360deg spin")
        s0 = node.latest()
        prompt(f"Cycle {i}/{cycles}: after 360deg spin")
        s1 = node.latest()
        if s0 is None or s1 is None:
            raise RuntimeError("No odom sample captured in cycle.")
        err = wrap_to_pi(s1.yaw - s0.yaw)
        eyaw.append(err)
        print(f"  cycle {i}: yaw_error={math.degrees(err):.3f} deg")
    return variance_or_zero(eyaw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate /odom covariance from repeatability (no external ground truth)."
    )
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--static-sec", type=float, default=120.0)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument(
        "--inflate",
        type=float,
        default=3.0,
        help="Safety factor applied to variances for conservative EKF tuning",
    )
    args = parser.parse_args()

    rclpy.init()
    node = OdomCollector(args.odom_topic)
    ex = SingleThreadedExecutor()
    ex.add_node(node)

    spin_thread = threading.Thread(target=ex.spin, daemon=False)
    spin_thread.start()

    try:
        wait_for_odom(node)
        print("\nConnected to odom. Starting calibration...")
        vx_var, wz_var = measure_static(node, args.static_sec)
        x_var, y_var = measure_roundtrip_xy(node, args.cycles)
        yaw_var = measure_spin_yaw(node, args.cycles)

        k = args.inflate
        print("\n========== Estimated variances ==========")
        print(f"pose x   var = {x_var:.8f}")
        print(f"pose y   var = {y_var:.8f}")
        print(f"pose yaw var = {yaw_var:.8f}")
        print(f"twist vx var = {vx_var:.8f}")
        print(f"twist wz var = {wz_var:.8f}")

        print("\n====== Recommended (inflated) values ======")
        print(f"pose.cov[0]   = {x_var * k:.8f}")
        print(f"pose.cov[7]   = {y_var * k:.8f}")
        print(f"pose.cov[35]  = {yaw_var * k:.8f}")
        print(f"twist.cov[0]  = {vx_var * k:.8f}")
        print(f"twist.cov[35] = {wz_var * k:.8f}")
        print("\nApply these to tracer odom covariance or EKF measurement covariance settings.")
    finally:
        ex.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
