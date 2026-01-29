import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import Twist, TwistStamped
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


class RealPidGainEnv(gym.Env):
    """실차 추론용 PID 게인 환경.

    - 학습용 환경과 달리 **자동 경로 생성/FollowPath 전송 없음**
    - 사용자가 찍은 목표를 Nav2가 처리하고,
      여기서는 PID 게인만 실시간으로 갱신
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        odom_topic: str = "/odometry/filtered",
        controller_node: str = "/controller_server",
        param_prefix: str = "RLController",
        desired_cmd_topic: str = "/controller_server/RLController/desired_cmd",
        desired_cmd_type: str = "auto",
        step_dt: float = 0.1,
        param_wait_sec: float = 15.0,
        use_sim_time: bool = False,
    ):
        super().__init__()

        # rclpy 초기화가 안 되어 있으면 먼저 초기화
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("rl_pid_env_real")
        # 실차는 기본적으로 wall time 사용(필요 시 --use-sim-time로 변경)
        if use_sim_time:
            self.node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        # 오도메트리(실제 측정 속도)
        self.odom_sub = self.node.create_subscription(Odometry, odom_topic, self._cb_odom, 10)

        # 목표 속도(컨트롤러가 계산한 desired_cmd)
        # 토픽 타입을 자동 감지하거나, 필요 시 파라미터로 고정
        self.desired_cmd_type = desired_cmd_type
        self.desired_sub = self._subscribe_desired(desired_cmd_topic, self.desired_cmd_type)

        # 컨트롤러 파라미터 서비스
        self.param_srv = f"{controller_node}/set_parameters"
        self.param_cli = self.node.create_client(SetParameters, self.param_srv)
        self.param_ready = self._wait_for_param_service(param_wait_sec)
        if not self.param_ready:
            self.node.get_logger().warn(f"parameter service not ready: {self.param_srv}")

        self.param_prefix = param_prefix
        self.step_dt = step_dt

        # 상태(관측) = [v_ref, w_ref, v_meas, w_meas, e_v, e_w]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=float)

        # 행동 = PID 게인 증감 (kp_lin, ki_lin, kd_lin, kp_ang, ki_ang, kd_ang)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=float)

        # PID 게인 초기값(현재 설정과 동일하게 맞춤)
        self.kp_lin = 1.0
        self.ki_lin = 0.0
        self.kd_lin = 0.1
        self.kp_ang = 1.0
        self.ki_ang = 0.0
        self.kd_ang = 0.1

        self.bounds = PidBounds()
        self.gain_steps = (0.05, 0.005, 0.02, 0.1, 0.005, 0.02)

        self.v_ref = 0.0
        self.w_ref = 0.0
        self.v_meas = 0.0
        self.w_meas = 0.0
        self.last_w_meas = 0.0

    def _wait_for_param_service(self, timeout_sec: float) -> bool:
        """파라미터 서비스가 준비될 때까지 기다림."""
        t_end = time.time() + timeout_sec
        while time.time() < t_end:
            if self.param_cli.wait_for_service(timeout_sec=0.5):
                return True
            self._spin_once()
        return False

    def _cb_odom(self, msg: Odometry):
        self.v_meas = msg.twist.twist.linear.x
        self.w_meas = msg.twist.twist.angular.z

    def _cb_desired_stamped(self, msg: TwistStamped):
        self.v_ref = msg.twist.linear.x
        self.w_ref = msg.twist.angular.z

    def _cb_desired_plain(self, msg: Twist):
        self.v_ref = msg.linear.x
        self.w_ref = msg.angular.z

    def _subscribe_desired(self, topic: str, desired_type: str):
        # desired_type: "auto" | "twist" | "twist_stamped"
        if desired_type == "auto":
            desired_type = self._detect_desired_type(topic)
        if desired_type == "twist":
            return self.node.create_subscription(Twist, topic, self._cb_desired_plain, 10)
        if desired_type == "twist_stamped":
            return self.node.create_subscription(TwistStamped, topic, self._cb_desired_stamped, 10)

        # 알 수 없는 값이면 기본적으로 TwistStamped로 구독
        self.node.get_logger().warn(
            f"unknown desired_cmd_type='{desired_type}', fallback to TwistStamped"
        )
        return self.node.create_subscription(TwistStamped, topic, self._cb_desired_stamped, 10)

    def _detect_desired_type(self, topic: str) -> str:
        # 토픽 타입이 아직 등록되지 않았을 수 있어 잠깐 대기
        t_end = time.time() + 2.0
        while time.time() < t_end:
            topic_types = dict(self.node.get_topic_names_and_types())
            if topic in topic_types:
                if "geometry_msgs/msg/TwistStamped" in topic_types[topic]:
                    return "twist_stamped"
                if "geometry_msgs/msg/Twist" in topic_types[topic]:
                    return "twist"
            self._spin_once()
            time.sleep(0.05)
        # 기본값: TwistStamped (Nav2 기본)
        self.node.get_logger().warn(
            f"could not detect type for {topic}, fallback to TwistStamped"
        )
        return "twist_stamped"

    def _set_pid_params(self):
        params = [
            Parameter(f"{self.param_prefix}.pid_kp_lin", Parameter.Type.DOUBLE, float(self.kp_lin)),
            Parameter(f"{self.param_prefix}.pid_ki_lin", Parameter.Type.DOUBLE, float(self.ki_lin)),
            Parameter(f"{self.param_prefix}.pid_kd_lin", Parameter.Type.DOUBLE, float(self.kd_lin)),
            Parameter(f"{self.param_prefix}.pid_kp_ang", Parameter.Type.DOUBLE, float(self.kp_ang)),
            Parameter(f"{self.param_prefix}.pid_ki_ang", Parameter.Type.DOUBLE, float(self.ki_ang)),
            Parameter(f"{self.param_prefix}.pid_kd_ang", Parameter.Type.DOUBLE, float(self.kd_ang)),
        ]
        if not self.param_cli.service_is_ready():
            # 서비스가 늦게 올라오는 경우 재시도
            if not self._wait_for_param_service(1.0):
                return
        req = SetParameters.Request()
        req.parameters = [p.to_parameter_msg() for p in params]
        self.param_cli.call_async(req)

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
        # 실차 추론에서는 목표 경로를 보내지 않는다
        self.last_w_meas = self.w_meas
        self._set_pid_params()
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

        # 한 스텝 동안 센서 업데이트 대기
        end = time.time() + self.step_dt
        while time.time() < end:
            self._spin_once()
            time.sleep(0.001)

        # 보상 계산(학습용과 동일 기준)
        e_v = self.v_ref - self.v_meas
        e_w = self.w_ref - self.w_meas
        de_w = self.w_meas - self.last_w_meas
        self.last_w_meas = self.w_meas

        reward = - (abs(e_v) + 0.5 * abs(e_w) + 0.1 * abs(de_w))

        terminated = False
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
