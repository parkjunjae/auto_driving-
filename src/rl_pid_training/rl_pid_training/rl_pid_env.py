import time
from dataclasses import dataclass

import rclpy
from rclpy.parameter import Parameter
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("gymnasium is required. Install stable-baselines3 (it pulls gymnasium).") from e


@dataclass
class PidBounds:
    kp_lin_min: float = 0.2
    kp_lin_max: float = 2.0
    ki_lin_min: float = 0.0
    ki_lin_max: float = 0.05
    kd_lin_min: float = 0.0
    kd_lin_max: float = 0.5
    kp_ang_min: float = 0.5
    kp_ang_max: float = 4.0
    ki_ang_min: float = 0.0
    ki_ang_max: float = 0.05
    kd_ang_min: float = 0.0
    kd_ang_max: float = 0.8


class PidGainEnv(gym.Env):
    """PID 게인을 강화학습으로 조정하기 위한 최소 환경(스캐폴딩).

    - 상태: 목표/측정 속도와 오차
    - 행동: PID 게인 (증감 방식)
    - 보상: 속도 오차 최소화
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cmd_ref_topic: str = "/cmd_vel",
        odom_topic: str = "/odometry/filtered",
        controller_node: str = "/controller_server",
        param_prefix: str = "RLController",
        step_dt: float = 0.1,
        episode_seconds: float = 10.0,
    ):
        super().__init__()

        self.node = rclpy.create_node("rl_pid_env")
        self.cmd_pub = self.node.create_publisher(Twist, cmd_ref_topic, 10)
        self.odom_sub = self.node.create_subscription(Odometry, odom_topic, self._cb_odom, 10)
        self.param_client = rclpy.parameter_client.AsyncParametersClient(self.node, controller_node)

        self.param_prefix = param_prefix
        self.step_dt = step_dt
        self.episode_seconds = episode_seconds

        # 상태(관측) = [v_ref, w_ref, v_meas, w_meas, e_v, e_w]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=float)

        # 행동 = PID 게인 증감 (kp_lin, ki_lin, kd_lin, kp_ang, ki_ang, kd_ang)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=float)

        # PID 게인 초기값(현재 설정과 동일하게 맞춤)
        self.kp_lin = 1.0
        self.ki_lin = 0.0
        self.kd_lin = 0.1
        self.kp_ang = 2.0
        self.ki_ang = 0.0
        self.kd_ang = 0.1

        self.bounds = PidBounds()
        self.gain_steps = (0.05, 0.005, 0.02, 0.1, 0.005, 0.02)

        self.v_ref = 0.0
        self.w_ref = 0.0
        self.v_meas = 0.0
        self.w_meas = 0.0
        self.last_w_meas = 0.0

        self.t0 = None

    def _cb_odom(self, msg: Odometry):
        self.v_meas = msg.twist.twist.linear.x
        self.w_meas = msg.twist.twist.angular.z

    def _set_pid_params(self):
        params = [
            Parameter(f"{self.param_prefix}.pid_kp_lin", Parameter.Type.DOUBLE, float(self.kp_lin)),
            Parameter(f"{self.param_prefix}.pid_ki_lin", Parameter.Type.DOUBLE, float(self.ki_lin)),
            Parameter(f"{self.param_prefix}.pid_kd_lin", Parameter.Type.DOUBLE, float(self.kd_lin)),
            Parameter(f"{self.param_prefix}.pid_kp_ang", Parameter.Type.DOUBLE, float(self.kp_ang)),
            Parameter(f"{self.param_prefix}.pid_ki_ang", Parameter.Type.DOUBLE, float(self.ki_ang)),
            Parameter(f"{self.param_prefix}.pid_kd_ang", Parameter.Type.DOUBLE, float(self.kd_ang)),
        ]
        self.param_client.set_parameters(params)

    def _publish_cmd(self):
        cmd = Twist()
        cmd.linear.x = self.v_ref
        cmd.angular.z = self.w_ref
        self.cmd_pub.publish(cmd)

    def _spin_once(self):
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def _clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

    def _apply_action(self, action):
        # 행동을 PID 게인 증감으로 적용
        deltas = [a * s for a, s in zip(action, self.gain_steps)]
        self.kp_lin = self._clamp(self.kp_lin + deltas[0], self.bounds.kp_lin_min, self.bounds.kp_lin_max)
        self.ki_lin = self._clamp(self.ki_lin + deltas[1], self.bounds.ki_lin_min, self.bounds.ki_lin_max)
        self.kd_lin = self._clamp(self.kd_lin + deltas[2], self.bounds.kd_lin_min, self.bounds.kd_lin_max)
        self.kp_ang = self._clamp(self.kp_ang + deltas[3], self.bounds.kp_ang_min, self.bounds.kp_ang_max)
        self.ki_ang = self._clamp(self.ki_ang + deltas[4], self.bounds.ki_ang_min, self.bounds.ki_ang_max)
        self.kd_ang = self._clamp(self.kd_ang + deltas[5], self.bounds.kd_ang_min, self.bounds.kd_ang_max)
        self._set_pid_params()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t0 = time.time()

        # 에피소드마다 목표 속도 변경(임의)
        self.v_ref = 0.3
        self.w_ref = 0.6
        self.last_w_meas = self.w_meas

        self._set_pid_params()
        self._publish_cmd()
        for _ in range(5):
            self._spin_once()
            time.sleep(0.01)

        return self._get_obs(), {}

    def _get_obs(self):
        e_v = self.v_ref - self.v_meas
        e_w = self.w_ref - self.w_meas
        return [self.v_ref, self.w_ref, self.v_meas, self.w_meas, e_v, e_w]

    def step(self, action):
        self._apply_action(action)
        self._publish_cmd()

        # 한 스텝 동안 센서 업데이트 대기
        end = time.time() + self.step_dt
        while time.time() < end:
            self._spin_once()
            time.sleep(0.001)

        # 보상 계산
        e_v = self.v_ref - self.v_meas
        e_w = self.w_ref - self.w_meas
        de_w = self.w_meas - self.last_w_meas
        self.last_w_meas = self.w_meas

        reward = - (abs(e_v) + 0.5 * abs(e_w) + 0.1 * abs(de_w))

        terminated = False
        truncated = (time.time() - self.t0) > self.episode_seconds

        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        self.cmd_pub.publish(Twist())
        self.node.destroy_node()
        rclpy.shutdown()
