#!/usr/bin/env python3
"""카메라 IMU 자이로 바이어스 보정 노드.

정지 상태에서 일정 샘플을 수집해 bias(영점 오차)를 계산하고,
각속도에서 bias를 빼서 /imu_fixed로 재발행합니다.
"""
import copy
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class ImuBiasCorrector(Node):
    def __init__(self):
        super().__init__('camera_imu_bias_corrector')

        self.declare_parameter('input_topic', '/camera/camera/imu')
        self.declare_parameter('output_topic', '/camera/camera/imu_fixed')
        # 정지 상태에서 bias 계산에 사용할 샘플 수
        self.declare_parameter('calib_samples', 1000)
        # 정지 판정 기준(각속도 크기)
        self.declare_parameter('stationary_threshold', 0.01)
        # 공분산 값이 클수록 EKF가 덜 신뢰
        self.declare_parameter('gyro_cov', 0.1)
        self.declare_parameter('accel_cov', 0.1)
        # 보정 중에도 메시지를 계속 내보낼지 여부
        self.declare_parameter('publish_during_calib', True)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.calib_samples = int(self.get_parameter('calib_samples').value)
        self.stationary_threshold = float(self.get_parameter('stationary_threshold').value)
        self.gyro_cov = float(self.get_parameter('gyro_cov').value)
        self.accel_cov = float(self.get_parameter('accel_cov').value)
        self.publish_during_calib = bool(self.get_parameter('publish_during_calib').value)

        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_z = 0.0
        self.count = 0
        self.bias_ready = False
        self.bias_x = 0.0
        self.bias_y = 0.0
        self.bias_z = 0.0

        self.sub = self.create_subscription(Imu, self.input_topic, self.cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(Imu, self.output_topic, qos_profile_sensor_data)

        self.get_logger().info(
            f'IMU bias corrector started: input={self.input_topic} output={self.output_topic}'
        )

    def cb(self, msg: Imu):
        # 현재 각속도의 크기(정지 판정에 사용)
        av = msg.angular_velocity
        mag = math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z)

        if not self.bias_ready:
            # 정지 상태로 판단되면 bias 누적
            if mag < self.stationary_threshold:
                self.sum_x += av.x
                self.sum_y += av.y
                self.sum_z += av.z
                self.count += 1
                if self.count == 1:
                    self.get_logger().info('Calibrating IMU bias... keep robot still.')
                if self.count >= self.calib_samples:
                    # 평균값을 bias로 확정
                    self.bias_x = self.sum_x / self.count
                    self.bias_y = self.sum_y / self.count
                    self.bias_z = self.sum_z / self.count
                    self.bias_ready = True
                    self.get_logger().info(
                        f'IMU bias calibrated: x={self.bias_x:.6f} '
                        f'y={self.bias_y:.6f} z={self.bias_z:.6f}'
                    )

        if not self.bias_ready and not self.publish_during_calib:
            # 보정 완료 전에는 발행하지 않음
            return

        out = copy.deepcopy(msg)
        # Orientation은 사용하지 않음을 명시
        out.orientation_covariance[0] = -1.0
        # EKF 신뢰도를 위해 공분산을 설정
        out.angular_velocity_covariance = (
            self.gyro_cov, 0.0, 0.0,
            0.0, self.gyro_cov, 0.0,
            0.0, 0.0, self.gyro_cov,
        )
        out.linear_acceleration_covariance = (
            self.accel_cov, 0.0, 0.0,
            0.0, self.accel_cov, 0.0,
            0.0, 0.0, self.accel_cov,
        )

        if self.bias_ready:
            # bias 제거 후 각속도 재발행
            out.angular_velocity.x = av.x - self.bias_x
            out.angular_velocity.y = av.y - self.bias_y
            out.angular_velocity.z = av.z - self.bias_z

        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(ImuBiasCorrector())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
