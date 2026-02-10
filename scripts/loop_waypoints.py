#!/usr/bin/env python3
import argparse
import math
import time

import rclpy
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped


# 기본 웨이포인트 (번호 순서: 1 -> 2 -> 3)
# 1) x=2.7492854595184326, y=-7.285099983215332
# 2) x=-0.1040419340133667, y=4.890406131744385
# 3) x=-0.7926440238952637, y=-3.4049925804138184
DEFAULT_POINTS = [
    (2.7492854595184326, -7.285099983215332, 0.0),
    (-0.1040419340133667, 4.890406131744385, 0.0),
    (-0.7926440238952637, -3.4049925804138184, 0.0),
]


def yaw_to_quat(yaw_rad: float):
    """Yaw(rad) -> (x, y, z, w)"""
    return (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


def parse_points(s: str):
    """
    "x1,y1;x2,y2;..." 형식 파싱.
    z는 0.0 고정.
    """
    pts = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid point '{chunk}', expected 'x,y'")
        x = float(parts[0].strip())
        y = float(parts[1].strip())
        pts.append((x, y, 0.0))
    if not pts:
        raise ValueError("No valid points parsed.")
    return pts


def build_pose(frame_id: str, x: float, y: float, z: float, yaw_deg: float):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    qx, qy, qz, qw = yaw_to_quat(math.radians(yaw_deg))
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def main():
    parser = argparse.ArgumentParser(description="Loop waypoints with NavigateToPose.")
    parser.add_argument("--frame", default="map", help="frame_id (default: map)")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="yaw for all points (deg)")
    parser.add_argument("--points", default="", help="override points: 'x1,y1;x2,y2;...'")
    parser.add_argument("--sleep", type=float, default=0.5, help="sleep between goals (sec)")
    parser.add_argument("--loop-sleep", type=float, default=1.0, help="sleep between loops (sec)")
    parser.add_argument("--loops", type=int, default=0, help="0=infinite, N=repeat N times")
    parser.add_argument("--goal-timeout", type=float, default=0.0,
                        help="timeout per goal (sec), 0=no timeout")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="stop if a goal is rejected/failed")
    args = parser.parse_args()

    points = DEFAULT_POINTS if not args.points else parse_points(args.points)

    rclpy.init()
    node = rclpy.create_node("loop_waypoints")
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")

    node.get_logger().info("Waiting for /navigate_to_pose action server...")
    client.wait_for_server()

    loop_count = 0
    try:
        while args.loops == 0 or loop_count < args.loops:
            node.get_logger().info(f"Starting loop {loop_count + 1}")
            for idx, (x, y, z) in enumerate(points, start=1):
                if not rclpy.ok():
                    return
                goal = NavigateToPose.Goal()
                goal.pose = build_pose(args.frame, x, y, z, args.yaw_deg)
                node.get_logger().info(f"[{idx}/{len(points)}] goal: x={x:.3f}, y={y:.3f}")

                send_future = client.send_goal_async(goal)
                rclpy.spin_until_future_complete(node, send_future)
                goal_handle = send_future.result()
                if goal_handle is None or not goal_handle.accepted:
                    node.get_logger().warn("Goal rejected.")
                    if args.stop_on_fail:
                        return
                    time.sleep(args.sleep)
                    continue

                result_future = goal_handle.get_result_async()
                if args.goal_timeout > 0:
                    rclpy.spin_until_future_complete(
                        node, result_future, timeout_sec=args.goal_timeout
                    )
                    if not result_future.done():
                        node.get_logger().warn("Goal timeout, canceling...")
                        goal_handle.cancel_goal_async()
                        if args.stop_on_fail:
                            return
                        time.sleep(args.sleep)
                        continue
                else:
                    rclpy.spin_until_future_complete(node, result_future)

                result = result_future.result()
                if result is None or result.status != 4:  # 4 = SUCCEEDED
                    node.get_logger().warn(f"Goal finished with status: {getattr(result, 'status', None)}")
                    if args.stop_on_fail:
                        return
                time.sleep(args.sleep)

            loop_count += 1
            time.sleep(args.loop_sleep)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
