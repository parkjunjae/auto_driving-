#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def clamp(x, lo, hi):
    # 값이 제한 범위를 벗어나지 않도록 클램프.
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class PID:
    def __init__(self, kp, ki, kd, i_min, i_max):
        # PID 게인과 적분 클램프(윈드업 방지) 설정.
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_min = i_min
        self.i_max = i_max
        self.i = 0.0
        self.prev_e = None

    def reset(self):
        # 적분 항과 미분 계산용 이전 오차 초기화.
        self.i = 0.0
        self.prev_e = None

    def step(self, e, dt):
        # dt가 유효하지 않으면 보정값 0으로 처리.
        if dt <= 0.0:
            return 0.0
        # 적분 항 누적 + 클램프.
        self.i = clamp(self.i + e * dt, self.i_min, self.i_max)
        # 미분 항은 오차 기울기 사용.
        de = 0.0 if self.prev_e is None else (e - self.prev_e) / dt
        self.prev_e = e
        # PID 합산 출력.
        return self.kp * e + self.ki * self.i + self.kd * de


class AngularPidCmdVel(Node):
    def __init__(self):
        super().__init__('angular_pid_cmdvel')

        # 토픽 설정: 입력(cmd_in), 출력(cmd_out), 측정(odom/imu).
        self.declare_parameter('cmd_in', '/cmd_vel_raw')
        self.declare_parameter('cmd_out', '/cmd_vel')
        self.declare_parameter('odom_in', '/odometry/filtered')
        self.declare_parameter('imu_in', '/camera/camera/imu_fixed')
        self.declare_parameter('use_imu', False)

        # PID 게인 및 적분 클램프.
        self.declare_parameter('kp', 2.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.1)
        self.declare_parameter('i_min', -0.5)
        self.declare_parameter('i_max', 0.5)

        # 출력 제한/주기/타임아웃 설정.
        self.declare_parameter('w_max', 1.5)
        self.declare_parameter('w_acc_max', 3.0)
        self.declare_parameter('loop_hz', 50.0)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('meas_timeout', 0.5)
        self.declare_parameter('reset_i_on_zero_ref', True)
        self.declare_parameter('stop_threshold', 1e-3)

        self.cmd_in = self.get_parameter('cmd_in').value
        self.cmd_out = self.get_parameter('cmd_out').value
        self.odom_in = self.get_parameter('odom_in').value
        self.imu_in = self.get_parameter('imu_in').value
        self.use_imu = bool(self.get_parameter('use_imu').value)

        kp = float(self.get_parameter('kp').value)
        ki = float(self.get_parameter('ki').value)
        kd = float(self.get_parameter('kd').value)
        i_min = float(self.get_parameter('i_min').value)
        i_max = float(self.get_parameter('i_max').value)

        # PID 컨트롤러 생성.
        self.pid = PID(kp, ki, kd, i_min, i_max)

        self.w_max = float(self.get_parameter('w_max').value)
        self.w_acc_max = float(self.get_parameter('w_acc_max').value)
        self.loop_hz = float(self.get_parameter('loop_hz').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self.meas_timeout = float(self.get_parameter('meas_timeout').value)
        self.reset_i_on_zero_ref = bool(
            self.get_parameter('reset_i_on_zero_ref').value
        )
        self.stop_threshold = float(self.get_parameter('stop_threshold').value)

        if self.loop_hz <= 0.0:
            self.loop_hz = 50.0

        # 최신 명령/측정값 타임스탬프 관리.
        self.last_cmd = Twist()
        now = self.get_clock().now()
        self.last_cmd_time = now
        self.last_meas_time = now
        self.last_control_time = now
        self.prev_w_cmd = 0.0
        self.w_meas = 0.0

        # 입력 명령/측정 토픽 구독.
        self.sub_cmd = self.create_subscription(
            Twist, self.cmd_in, self.cb_cmd, 10
        )
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_in, self.cb_odom, 10
        )
        self.sub_imu = None
        if self.use_imu:
            self.sub_imu = self.create_subscription(
                Imu, self.imu_in, self.cb_imu, qos_profile_sensor_data
            )

        # 보정된 cmd_vel 출력 퍼블리셔.
        self.pub = self.create_publisher(Twist, self.cmd_out, 10)

        # 주기적 제어 루프 타이머.
        period = 1.0 / self.loop_hz
        self.timer = self.create_timer(period, self.loop)

        self.get_logger().info(
            f'Angular PID cmdvel: in={self.cmd_in} out={self.cmd_out} '
            f'meas={"imu" if self.use_imu else "odom"}'
        )

    def cb_cmd(self, msg):
        # 상위 제어기에서 들어온 목표 cmd_vel 저장.
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def cb_odom(self, msg):
        # EKF 오도메트리의 각속도(z)를 측정값으로 사용.
        if not self.use_imu:
            self.w_meas = msg.twist.twist.angular.z
            self.last_meas_time = self.get_clock().now()

    def cb_imu(self, msg):
        # IMU 자이로 z를 측정값으로 사용.
        if self.use_imu:
            self.w_meas = msg.angular_velocity.z
            self.last_meas_time = self.get_clock().now()

    def loop(self):
        # 제어 루프: 각속도 PID만 보정.
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds * 1e-9
        self.last_control_time = now
        if dt <= 0.0 or dt > 1.0:
            # 비정상 dt는 스킵하고 상태 초기화.
            self.pid.reset()
            self.prev_w_cmd = 0.0
            return

        if self.cmd_timeout > 0.0:
            # 명령이 오래되면 안전을 위해 0으로 리셋.
            cmd_age = (now - self.last_cmd_time).nanoseconds * 1e-9
            if cmd_age > self.cmd_timeout:
                self.last_cmd = Twist()
                self.pid.reset()
                self.prev_w_cmd = 0.0

        # 측정값이 최신일 때만 PID 보정 적용.
        use_meas = True
        if self.meas_timeout > 0.0:
            meas_age = (now - self.last_meas_time).nanoseconds * 1e-9
            if meas_age > self.meas_timeout:
                use_meas = False
                self.pid.reset()
                self.prev_w_cmd = 0.0

        # 목표 속도(참조 값).
        v_ref = self.last_cmd.linear.x
        w_ref = self.last_cmd.angular.z

        if self.reset_i_on_zero_ref:
            # 정지 상태에서 적분 항 초기화(드리프트 방지).
            if abs(v_ref) < self.stop_threshold and abs(w_ref) < self.stop_threshold:
                self.pid.reset()
                self.prev_w_cmd = 0.0

        if use_meas:
            # 오차 기반 PID 보정.
            e = w_ref - self.w_meas
            u = self.pid.step(e, dt)
        else:
            # 측정값이 없으면 보정 없이 참조값 그대로.
            u = 0.0

        # 피드포워드(w_ref) + PID 보정(u).
        w_cmd = w_ref + u
        # 각속도 출력 제한.
        w_cmd = clamp(w_cmd, -self.w_max, self.w_max)

        if self.w_acc_max > 0.0:
            # 각가속도 제한(레이트 리밋).
            max_dw = self.w_acc_max * dt
            w_cmd = clamp(w_cmd, self.prev_w_cmd - max_dw, self.prev_w_cmd + max_dw)

        self.prev_w_cmd = w_cmd

        # 선속도는 그대로 통과, 각속도만 보정값 적용.
        out = Twist()
        out.linear.x = self.last_cmd.linear.x
        out.linear.y = self.last_cmd.linear.y
        out.linear.z = self.last_cmd.linear.z
        out.angular.x = self.last_cmd.angular.x
        out.angular.y = self.last_cmd.angular.y
        out.angular.z = w_cmd
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AngularPidCmdVel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
