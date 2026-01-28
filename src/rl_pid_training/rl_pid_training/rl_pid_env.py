import time
from dataclasses import dataclass

import rclpy
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import FollowPath

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
        odom_topic: str = "/diff_drive_controller/odom",
        controller_node: str = "/controller_server",
        param_prefix: str = "RLController",
        desired_cmd_topic: str = "/controller_server/RLController/desired_cmd",
        follow_path_action: str = "/follow_path",
        step_dt: float = 0.1,
        episode_seconds: float = 10.0,
        param_wait_sec: float = 15.0,
        use_sim_time: bool = True,
    ):
        super().__init__()

        # rclpy 초기화가 안 되어 있으면 먼저 초기화
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("rl_pid_env")
        # 시뮬레이터 시간 사용 여부(가제보 학습 시 True 권장)
        if use_sim_time:
            self.node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.odom_sub = self.node.create_subscription(Odometry, odom_topic, self._cb_odom, 10)
        self.desired_sub = self.node.create_subscription(
            TwistStamped, desired_cmd_topic, self._cb_desired, 10)

        # FollowPath 액션 클라이언트 (Nav2 controller_server)
        self.follow_client = ActionClient(self.node, FollowPath, follow_path_action)
        # 컨트롤러 노드의 파라미터 서비스로 직접 요청(모듈 의존성 최소화)
        self.param_srv = f"{controller_node}/set_parameters"
        self.param_cli = self.node.create_client(SetParameters, self.param_srv)
        # 서비스가 올라올 때까지 잠깐 대기(레이스 방지)
        self.param_ready = self._wait_for_param_service(param_wait_sec)
        if not self.param_ready:
            self.node.get_logger().warn(f"parameter service not ready: {self.param_srv}")

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
        self.last_odom = None

        self.t0 = None

    def _wait_for_param_service(self, timeout_sec: float) -> bool:
        """파라미터 서비스가 준비될 때까지 기다림."""
        t_end = time.time() + timeout_sec
        while time.time() < t_end:
            if self.param_cli.wait_for_service(timeout_sec=0.5):
                return True
            self._spin_once()
        return False

    def _cb_odom(self, msg: Odometry):
        self.last_odom = msg
        self.v_meas = msg.twist.twist.linear.x
        self.w_meas = msg.twist.twist.angular.z

    def _cb_desired(self, msg: TwistStamped):
        self.v_ref = msg.twist.linear.x
        self.w_ref = msg.twist.angular.z

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
            # 학습 중간에 서비스가 늦게 올라오는 경우를 대비해 재시도
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
        self.t0 = time.time()

        # 에피소드 시작 시 FollowPath 목표를 한번 보낸다.
        self._send_follow_path()
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

        # 보상 계산
        e_v = self.v_ref - self.v_meas
        e_w = self.w_ref - self.w_meas
        de_w = self.w_meas - self.last_w_meas
        self.last_w_meas = self.w_meas

        reward = - (abs(e_v) + 0.5 * abs(e_w) + 0.1 * abs(de_w))

        terminated = False
        truncated = (time.time() - self.t0) > self.episode_seconds

        return self._get_obs(), reward, terminated, truncated, {}

    def _send_follow_path(self):
        # FollowPath 서버 준비 대기
        if not self.follow_client.wait_for_server(timeout_sec=2.0):
            return

        # 오도메트리 수신 대기
        t_end = time.time() + 2.0
        while self.last_odom is None and time.time() < t_end:
            self._spin_once()
            time.sleep(0.01)
        if self.last_odom is None:
            self.node.get_logger().warn("odom not received, skip sending FollowPath")
            return

        # 간단한 직진 경로 생성(현재 위치에서 +2m)
        start = PoseStamped()
        start.header.frame_id = self.last_odom.header.frame_id
        start.header.stamp = self.last_odom.header.stamp
        start.pose = self.last_odom.pose.pose

        goal = PoseStamped()
        goal.header.frame_id = start.header.frame_id
        goal.header.stamp = start.header.stamp
        goal.pose = start.pose
        goal.pose.position.x += 2.0

        path = Path()
        path.header.frame_id = start.header.frame_id
        path.header.stamp = start.header.stamp
        path.poses = [start, goal]

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = ""  # 기본 컨트롤러 사용
        goal_msg.goal_checker_id = ""  # 기본 goal checker 사용

        self.follow_client.send_goal_async(goal_msg)

    def close(self):
        # cmd_vel 퍼블리셔를 사용하지 않으므로 안전하게 종료만 수행
        self.node.destroy_node()
        rclpy.shutdown()
