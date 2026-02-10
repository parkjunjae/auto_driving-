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
    # 최근 실차 로그 저오차 구간(상위 25%) 기반으로 범위를 좁혀 헌팅을 억제
    kp_lin_min: float = 1.05
    kp_lin_max: float = 1.30
    ki_lin_min: float = 0.0
    ki_lin_max: float = 0.01
    kd_lin_min: float = 0.02
    kd_lin_max: float = 0.08
    kp_ang_min: float = 1.10
    kp_ang_max: float = 1.35
    ki_ang_min: float = 0.0
    ki_ang_max: float = 0.01
    kd_ang_min: float = 0.07
    kd_ang_max: float = 0.14


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
        step_dt: float = 0.4,
        gain_scale: float = 0.2,
        gain_lpf_alpha: float = 0.18,
        action_deadzone: float = 0.25,
        straight_freeze_w_ref: float = 0.12,
        straight_freeze_v_ref: float = 0.08,
        stop_freeze_v_ref: float = 0.03,
        stop_freeze_w_ref: float = 0.05,
        rotate_freeze_w_ref: float = 0.20,
        rotate_freeze_v_ref: float = 0.05,
        # 저속 좌우 떨림 억제용 w_ref 가드(로그 기반 튜닝)
        w_ref_lpf_alpha: float = 0.18,  # 작을수록 회전 지령이 부드럽게 변함
        w_ref_deadband: float = 0.06,  # 작은 w_ref는 0으로 죽여 미세 떨림 제거
        dither_v_ref_thresh: float = 0.10,  # 저속 구간을 더 넓게 잡아 흔들림 억제
        dither_w_ref_thresh: float = 0.15,  # anti-dither 대상 최소 회전 크기
        w_ref_sign_hold_sec: float = 0.50,  # 부호 유지 시간을 늘려 좌우 반전 억제
        w_ref_abs_max_low_speed: float = 0.30,  # 저속 회전 상한을 낮춰 급회전 방지
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
        self.gain_scale = gain_scale
        self.gain_lpf_alpha = gain_lpf_alpha
        self.action_deadzone = action_deadzone
        self.straight_freeze_w_ref = straight_freeze_w_ref
        self.straight_freeze_v_ref = straight_freeze_v_ref
        self.stop_freeze_v_ref = stop_freeze_v_ref
        self.stop_freeze_w_ref = stop_freeze_w_ref
        self.rotate_freeze_w_ref = rotate_freeze_w_ref
        self.rotate_freeze_v_ref = rotate_freeze_v_ref
        # 저속 구간에서 desired_cmd의 좌/우 부호가 빠르게 뒤집히며 헌팅이 발생하는 것을 줄이기 위한 파라미터
        self.w_ref_lpf_alpha = w_ref_lpf_alpha
        self.w_ref_deadband = w_ref_deadband
        self.dither_v_ref_thresh = dither_v_ref_thresh
        self.dither_w_ref_thresh = dither_w_ref_thresh
        self.w_ref_sign_hold_sec = w_ref_sign_hold_sec
        self.w_ref_abs_max_low_speed = w_ref_abs_max_low_speed

        # 상태(관측) = [v_ref, w_ref, v_meas, w_meas, e_v, e_w]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=float)

        # 행동 = PID 게인 증감 (kp_lin, ki_lin, kd_lin, kp_ang, ki_ang, kd_ang)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=float)

        # PID 게인 초기값(현재 설정과 동일하게 맞춤)
        self.kp_lin = 1.20
        self.ki_lin = 0.0
        self.kd_lin = 0.03
        self.kp_ang = 1.23
        self.ki_ang = 0.0
        self.kd_ang = 0.10

        self.bounds = PidBounds()
        self.gain_steps = (0.015, 0.001, 0.005, 0.02, 0.001, 0.005)

        self.v_ref = 0.0
        self.w_ref = 0.0
        self.v_ref_raw = 0.0
        self.w_ref_raw = 0.0
        self._w_ref_lpf = 0.0
        self._turn_sign = 0
        self._last_turn_sign_change_ts = 0.0
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

    def _update_desired(self, v_ref: float, w_ref: float):
        """desired_cmd를 저속 anti-dither 규칙으로 안정화해 내부 참조값으로 사용."""
        now = time.monotonic()
        self.v_ref_raw = float(v_ref)
        self.w_ref_raw = float(w_ref)

        # 1) deadband: 매우 작은 각속도는 0으로 간주해 불필요한 흔들림을 제거
        w_cmd = self.w_ref_raw
        if abs(w_cmd) < self.w_ref_deadband:
            w_cmd = 0.0

        # 2) 저역통과: desired_cmd 자체의 급격한 변화(특히 목표점 근처)를 완화
        a = self.w_ref_lpf_alpha
        self._w_ref_lpf = self._w_ref_lpf + a * (w_cmd - self._w_ref_lpf)
        w_cmd = self._w_ref_lpf

        low_speed = abs(self.v_ref_raw) < self.dither_v_ref_thresh
        turning = abs(w_cmd) > self.dither_w_ref_thresh

        # 3) sign hold: 저속 회전에서 좌/우 부호가 짧은 시간에 뒤집히는 것을 억제
        if low_speed and turning:
            new_sign = 1 if w_cmd > 0.0 else -1
            if (
                self._turn_sign != 0 and
                new_sign != self._turn_sign and
                (now - self._last_turn_sign_change_ts) < self.w_ref_sign_hold_sec
            ):
                # hold 기간 내 반전은 무시하고 기존 부호 유지
                w_cmd = abs(w_cmd) * float(self._turn_sign)
                new_sign = self._turn_sign

            if new_sign != self._turn_sign:
                self._turn_sign = new_sign
                self._last_turn_sign_change_ts = now

            # 4) 저속 구간 과도한 각속도 지령 상한 제한
            lim = self.w_ref_abs_max_low_speed
            w_cmd = max(-lim, min(lim, w_cmd))
        elif abs(w_cmd) <= self.dither_w_ref_thresh:
            # 회전 요구가 사라지면 부호 상태를 초기화
            self._turn_sign = 0

        self.v_ref = self.v_ref_raw
        self.w_ref = w_cmd

    def _cb_desired_stamped(self, msg: TwistStamped):
        self._update_desired(msg.twist.linear.x, msg.twist.angular.z)

    def _cb_desired_plain(self, msg: Twist):
        self._update_desired(msg.linear.x, msg.angular.z)

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
        # 미세 잡음은 deadzone으로 무시
        scaled = []
        for a in action:
            aa = float(a)
            if abs(aa) < self.action_deadzone:
                aa = 0.0
            scaled.append(aa * self.gain_scale)

        # 정지 구간에서는 게인을 갱신하지 않아 대기 중 흔들림을 억제
        if abs(self.v_ref) < self.stop_freeze_v_ref and abs(self.w_ref) < self.stop_freeze_w_ref:
            return

        # 직진 구간에서는 각속도 PID를 고정해 좌우 헌팅을 줄임
        if self.v_ref > self.straight_freeze_v_ref and abs(self.w_ref) < self.straight_freeze_w_ref:
            scaled[3] = 0.0
            scaled[4] = 0.0
            scaled[5] = 0.0
        # 제자리 회전에 가까운 구간에서는 선속도 PID를 고정
        if abs(self.w_ref) > self.rotate_freeze_w_ref and abs(self.v_ref) < self.rotate_freeze_v_ref:
            scaled[0] = 0.0
            scaled[1] = 0.0
            scaled[2] = 0.0

        deltas = [a * s for a, s in zip(scaled, self.gain_steps)]

        tgt_kp_lin = self._clamp(self.kp_lin + deltas[0], self.bounds.kp_lin_min, self.bounds.kp_lin_max)
        tgt_ki_lin = self._clamp(self.ki_lin + deltas[1], self.bounds.ki_lin_min, self.bounds.ki_lin_max)
        tgt_kd_lin = self._clamp(self.kd_lin + deltas[2], self.bounds.kd_lin_min, self.bounds.kd_lin_max)
        tgt_kp_ang = self._clamp(self.kp_ang + deltas[3], self.bounds.kp_ang_min, self.bounds.kp_ang_max)
        tgt_ki_ang = self._clamp(self.ki_ang + deltas[4], self.bounds.ki_ang_min, self.bounds.ki_ang_max)
        tgt_kd_ang = self._clamp(self.kd_ang + deltas[5], self.bounds.kd_ang_min, self.bounds.kd_ang_max)

        # 저역통과로 게인 급변을 완화
        a = self.gain_lpf_alpha
        self.kp_lin = self._clamp(self.kp_lin + a * (tgt_kp_lin - self.kp_lin), self.bounds.kp_lin_min, self.bounds.kp_lin_max)
        self.ki_lin = self._clamp(self.ki_lin + a * (tgt_ki_lin - self.ki_lin), self.bounds.ki_lin_min, self.bounds.ki_lin_max)
        self.kd_lin = self._clamp(self.kd_lin + a * (tgt_kd_lin - self.kd_lin), self.bounds.kd_lin_min, self.bounds.kd_lin_max)
        self.kp_ang = self._clamp(self.kp_ang + a * (tgt_kp_ang - self.kp_ang), self.bounds.kp_ang_min, self.bounds.kp_ang_max)
        self.ki_ang = self._clamp(self.ki_ang + a * (tgt_ki_ang - self.ki_ang), self.bounds.ki_ang_min, self.bounds.ki_ang_max)
        self.kd_ang = self._clamp(self.kd_ang + a * (tgt_kd_ang - self.kd_ang), self.bounds.kd_ang_min, self.bounds.kd_ang_max)
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
        if rclpy.ok():
            rclpy.shutdown()
