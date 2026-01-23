#!/usr/bin/env python3
import json
import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def normalize_angle(x):
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class LinearPolicy:
    def __init__(self, weights: List[List[float]], bias: List[float], activation: str):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def _activate(self, x: float) -> float:
        if self.activation == 'tanh':
            return math.tanh(x)
        if self.activation == 'relu':
            return max(0.0, x)
        return x

    def forward(self, obs: List[float]) -> Tuple[float, float]:
        if len(self.weights) != 2:
            raise ValueError('weights must have 2 rows')
        if len(self.bias) != 2:
            raise ValueError('bias must have 2 elements')
        v = 0.0
        w = 0.0
        for i, o in enumerate(obs):
            v += self.weights[0][i] * o
            w += self.weights[1][i] * o
        v += self.bias[0]
        w += self.bias[1]
        return self._activate(v), self._activate(w)


class RlLocalPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_local_policy')

        self.declare_parameter('obs_mode', 'scan')  # scan or costmap
        self.declare_parameter('scan_in', '/scan')
        self.declare_parameter('costmap_in', '/local_costmap/costmap')
        self.declare_parameter('odom_in', '/odometry/filtered')
        self.declare_parameter('goal_in', '/goal_pose')
        self.declare_parameter('cmd_out', '/cmd_vel_raw')
        self.declare_parameter('policy_type', 'rule')  # rule or linear
        self.declare_parameter('policy_path', '')

        self.declare_parameter('range_max', 6.0)
        self.declare_parameter('front_angle', 0.35)
        self.declare_parameter('side_angle', 1.2)
        self.declare_parameter('obstacle_threshold', 80)

        self.declare_parameter('max_lin', 0.5)
        self.declare_parameter('max_ang', 1.0)
        self.declare_parameter('stop_dist', 0.6)
        self.declare_parameter('slow_dist', 1.5)
        self.declare_parameter('turn_gain', 1.0)
        self.declare_parameter('goal_dist_scale', 5.0)

        self.declare_parameter('loop_hz', 20.0)
        self.declare_parameter('sensor_timeout', 0.5)

        self.obs_mode = self.get_parameter('obs_mode').value
        self.scan_in = self.get_parameter('scan_in').value
        self.costmap_in = self.get_parameter('costmap_in').value
        self.odom_in = self.get_parameter('odom_in').value
        self.goal_in = self.get_parameter('goal_in').value
        self.cmd_out = self.get_parameter('cmd_out').value
        self.policy_type = self.get_parameter('policy_type').value
        self.policy_path = self.get_parameter('policy_path').value

        self.range_max = float(self.get_parameter('range_max').value)
        self.front_angle = float(self.get_parameter('front_angle').value)
        self.side_angle = float(self.get_parameter('side_angle').value)
        self.obstacle_threshold = int(self.get_parameter('obstacle_threshold').value)

        self.max_lin = float(self.get_parameter('max_lin').value)
        self.max_ang = float(self.get_parameter('max_ang').value)
        self.stop_dist = float(self.get_parameter('stop_dist').value)
        self.slow_dist = float(self.get_parameter('slow_dist').value)
        self.turn_gain = float(self.get_parameter('turn_gain').value)
        self.goal_dist_scale = float(self.get_parameter('goal_dist_scale').value)

        self.loop_hz = float(self.get_parameter('loop_hz').value)
        self.sensor_timeout = float(self.get_parameter('sensor_timeout').value)

        self.last_sensor_time = self.get_clock().now()
        self.last_odom_time = self.get_clock().now()
        self.last_goal_time = None
        self.last_goal = None

        self.last_scan = None
        self.last_costmap = None
        self.current_v = 0.0
        self.current_w = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.policy = None
        if self.policy_type == 'linear' and self.policy_path:
            self.policy = self._load_policy(self.policy_path)
            if self.policy is None:
                self.get_logger().warn('Policy load failed, falling back to rule.')
                self.policy_type = 'rule'

        if self.obs_mode == 'scan':
            self.scan_sub = self.create_subscription(
                LaserScan, self.scan_in, self.cb_scan, 10
            )
            self.costmap_sub = None
        elif self.obs_mode == 'costmap':
            self.costmap_sub = self.create_subscription(
                OccupancyGrid, self.costmap_in, self.cb_costmap, 10
            )
            self.scan_sub = None
        else:
            self.get_logger().warn('Unknown obs_mode, defaulting to scan.')
            self.obs_mode = 'scan'
            self.scan_sub = self.create_subscription(
                LaserScan, self.scan_in, self.cb_scan, 10
            )
            self.costmap_sub = None

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_in, self.cb_odom, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_in, self.cb_goal, 10
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_out, 10)

        period = 1.0 / self.loop_hz if self.loop_hz > 0.0 else 0.05
        self.timer = self.create_timer(period, self.loop)

        self.get_logger().info(
            f'RL local policy: mode={self.obs_mode} policy={self.policy_type} '
            f'cmd_out={self.cmd_out}'
        )

    def _load_policy(self, path: str) -> Optional[LinearPolicy]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            weights = data.get('weights', [])
            bias = data.get('bias', [])
            activation = data.get('activation', 'tanh')
            if not weights or not bias:
                return None
            return LinearPolicy(weights, bias, activation)
        except Exception as exc:
            self.get_logger().warn(f'Failed to load policy: {exc}')
            return None

    def cb_scan(self, msg: LaserScan):
        self.last_scan = msg
        self.last_sensor_time = self.get_clock().now()

    def cb_costmap(self, msg: OccupancyGrid):
        self.last_costmap = msg
        self.last_sensor_time = self.get_clock().now()

    def cb_odom(self, msg: Odometry):
        self.current_v = msg.twist.twist.linear.x
        self.current_w = msg.twist.twist.angular.z
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.last_odom_time = self.get_clock().now()

    def cb_goal(self, msg: PoseStamped):
        self.last_goal = msg
        self.last_goal_time = self.get_clock().now()

    def _sector_mins_from_scan(self, scan: LaserScan) -> Tuple[float, float, float]:
        front = self.range_max
        left = self.range_max
        right = self.range_max
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r):
                r = self.range_max
            r = clamp(r, 0.0, self.range_max)
            if -self.front_angle <= angle <= self.front_angle:
                front = min(front, r)
            elif self.front_angle < angle <= self.side_angle:
                left = min(left, r)
            elif -self.side_angle <= angle < -self.front_angle:
                right = min(right, r)
            angle += scan.angle_increment
        return front, left, right

    def _sector_mins_from_costmap(self, grid: OccupancyGrid) -> Tuple[float, float, float]:
        front = self.range_max
        left = self.range_max
        right = self.range_max
        w = grid.info.width
        h = grid.info.height
        if w == 0 or h == 0:
            return front, left, right
        res = grid.info.resolution
        cx = w / 2.0
        cy = h / 2.0
        data = grid.data
        for y in range(h):
            for x in range(w):
                occ = data[y * w + x]
                if occ < self.obstacle_threshold:
                    continue
                dx = (x - cx) * res
                dy = (y - cy) * res
                dist = math.hypot(dx, dy)
                if dist <= 0.0 or dist > self.range_max:
                    continue
                angle = math.atan2(dy, dx)
                if -self.front_angle <= angle <= self.front_angle:
                    front = min(front, dist)
                elif self.front_angle < angle <= self.side_angle:
                    left = min(left, dist)
                elif -self.side_angle <= angle < -self.front_angle:
                    right = min(right, dist)
        return front, left, right

    def _goal_features(self) -> Tuple[float, float]:
        if self.last_goal is None:
            return 0.0, 0.0
        gx = self.last_goal.pose.position.x
        gy = self.last_goal.pose.position.y
        dx = gx - self.current_x
        dy = gy - self.current_y
        dist = math.hypot(dx, dy)
        heading = normalize_angle(math.atan2(dy, dx) - self.current_yaw)
        dist_norm = clamp(dist / self.goal_dist_scale, 0.0, 1.0)
        heading_norm = clamp(heading / math.pi, -1.0, 1.0)
        return dist_norm, heading_norm

    def _build_obs(self) -> Optional[List[float]]:
        now = self.get_clock().now()
        age = (now - self.last_sensor_time).nanoseconds * 1e-9
        if age > self.sensor_timeout:
            return None

        if self.obs_mode == 'scan' and self.last_scan is not None:
            front, left, right = self._sector_mins_from_scan(self.last_scan)
        elif self.obs_mode == 'costmap' and self.last_costmap is not None:
            front, left, right = self._sector_mins_from_costmap(self.last_costmap)
        else:
            return None

        front_n = clamp(front / self.range_max, 0.0, 1.0)
        left_n = clamp(left / self.range_max, 0.0, 1.0)
        right_n = clamp(right / self.range_max, 0.0, 1.0)
        v_n = clamp(self.current_v / self.max_lin, -1.0, 1.0) if self.max_lin > 0.0 else 0.0
        w_n = clamp(self.current_w / self.max_ang, -1.0, 1.0) if self.max_ang > 0.0 else 0.0
        goal_dist_n, goal_heading_n = self._goal_features()
        return [front_n, left_n, right_n, v_n, w_n, goal_dist_n, goal_heading_n]

    def _rule_policy(self, obs: List[float]) -> Tuple[float, float]:
        front = obs[0] * self.range_max
        left = obs[1] * self.range_max
        right = obs[2] * self.range_max
        goal_heading = obs[6] * math.pi

        if front < self.stop_dist:
            v = 0.0
            w = self.turn_gain * (1.0 if left < right else -1.0)
        else:
            v = self.max_lin * clamp(front / self.slow_dist, 0.0, 1.0)
            avoid = self.turn_gain * clamp((right - left) / self.range_max, -1.0, 1.0)
            w = clamp(goal_heading, -1.0, 1.0) * self.max_ang + avoid * self.max_ang
        return clamp(v, -self.max_lin, self.max_lin), clamp(w, -self.max_ang, self.max_ang)

    def loop(self):
        obs = self._build_obs()
        if obs is None:
            self.cmd_pub.publish(Twist())
            return

        if self.policy_type == 'linear' and self.policy is not None:
            try:
                v_n, w_n = self.policy.forward(obs)
                v = clamp(v_n, -1.0, 1.0) * self.max_lin
                w = clamp(w_n, -1.0, 1.0) * self.max_ang
            except Exception as exc:
                self.get_logger().warn(f'Policy error: {exc}')
                v, w = self._rule_policy(obs)
        else:
            v, w = self._rule_policy(obs)

        out = Twist()
        out.linear.x = v
        out.angular.z = w
        self.cmd_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RlLocalPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
