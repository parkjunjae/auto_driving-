import argparse
import csv
import os
import time
from datetime import datetime

import rclpy
from stable_baselines3 import PPO

from rl_pid_training.rl_pid_env_real import RealPidGainEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/world/to_ws/rl_pid_model_new",
        help="학습된 모델(.zip) 경로",
    )
    parser.add_argument(
        "--log-dir",
        default="/home/world/to_ws/rl_pid_logs",
        help="CSV 로그 저장 폴더",
    )
    parser.add_argument(
        "--odom-topic",
        default="/odometry/filtered",
        help="오도메트리 토픽",
    )
    parser.add_argument(
        "--desired-cmd-topic",
        default="/controller_server/RLController/desired_cmd",
        help="컨트롤러 목표 속도 토픽",
    )
    parser.add_argument(
        "--desired-cmd-type",
        default="auto",
        help="desired_cmd 타입(auto|twist|twist_stamped)",
    )
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        help="시뮬레이션 시간 사용",
    )
    parser.add_argument(
        "--step-dt",
        type=float,
        default=0.4,
        help="agent PID 갱신 주기(초), 클수록 덜 민감",
    )
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=0.2,
        help="행동(게인 증감) 스케일, 작을수록 덜 흔들림",
    )
    parser.add_argument(
        "--gain-lpf-alpha",
        type=float,
        default=0.18,
        help="게인 저역통과 계수(0~1), 작을수록 더 부드러움",
    )
    parser.add_argument(
        "--action-deadzone",
        type=float,
        default=0.25,
        help="행동 deadzone(절대값 이하면 무시)",
    )
    parser.add_argument(
        "--straight-freeze-w-ref",
        type=float,
        default=0.12,
        help="직진 판정용 |w_ref| 임계값(rad/s)",
    )
    parser.add_argument(
        "--straight-freeze-v-ref",
        type=float,
        default=0.08,
        help="직진 판정용 v_ref 임계값(m/s)",
    )
    parser.add_argument(
        "--stop-freeze-v-ref",
        type=float,
        default=0.03,
        help="정지 판정용 |v_ref| 임계값(m/s)",
    )
    parser.add_argument(
        "--stop-freeze-w-ref",
        type=float,
        default=0.05,
        help="정지 판정용 |w_ref| 임계값(rad/s)",
    )
    parser.add_argument(
        "--rotate-freeze-w-ref",
        type=float,
        default=0.20,
        help="제자리 회전 판정용 |w_ref| 임계값(rad/s)",
    )
    parser.add_argument(
        "--rotate-freeze-v-ref",
        type=float,
        default=0.05,
        help="제자리 회전 판정용 |v_ref| 임계값(m/s)",
    )
    # ros2 launch adds ROS remap args (e.g. --ros-args ...), so ignore unknown CLI args.
    args, _ = parser.parse_known_args()

    # 학습된 모델 로드(학습 종료 시 저장된 파일)
    model = PPO.load(args.model)

    # 환경 생성(시뮬 실행 + controller_server 활성 상태여야 함)
    env = RealPidGainEnv(
        odom_topic=args.odom_topic,
        desired_cmd_topic=args.desired_cmd_topic,
        desired_cmd_type=args.desired_cmd_type,
        step_dt=args.step_dt,
        gain_scale=args.gain_scale,
        gain_lpf_alpha=args.gain_lpf_alpha,
        action_deadzone=args.action_deadzone,
        straight_freeze_w_ref=args.straight_freeze_w_ref,
        straight_freeze_v_ref=args.straight_freeze_v_ref,
        stop_freeze_v_ref=args.stop_freeze_v_ref,
        stop_freeze_w_ref=args.stop_freeze_w_ref,
        rotate_freeze_w_ref=args.rotate_freeze_w_ref,
        rotate_freeze_v_ref=args.rotate_freeze_v_ref,
        use_sim_time=args.use_sim_time,
    )

    # 초기화
    obs, _ = env.reset()

    # CSV 로그 파일 준비
    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"pid_policy_{ts}.csv")

    # 로그 헤더 정의
    header = [
        "t",
        "kp_lin",
        "ki_lin",
        "kd_lin",
        "kp_ang",
        "ki_ang",
        "kd_ang",
        "v_ref",
        "w_ref",
        "v_meas",
        "w_meas",
        "reward",
    ]

    try:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            step = 0
            while rclpy.ok():
                # 정책으로부터 행동 추론
                action, _ = model.predict(obs, deterministic=True)
                # 행동 적용
                obs, reward, terminated, truncated, _ = env.step(action)

                # 현재 게인/상태를 CSV로 기록
                writer.writerow([
                    time.time(),
                    env.kp_lin,
                    env.ki_lin,
                    env.kd_lin,
                    env.kp_ang,
                    env.ki_ang,
                    env.kd_ang,
                    env.v_ref,
                    env.w_ref,
                    env.v_meas,
                    env.w_meas,
                    reward,
                ])

                # 디스크 버퍼 플러시(너무 자주 I/O 안 나오도록 완화)
                step += 1
                if step % 10 == 0:
                    f.flush()

                # 에피소드 종료 시 리셋
                if terminated or truncated:
                    obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

def cli_main():
    # console_scripts / ros2 run entrypoint
    rclpy.init()
    try:
        main()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    cli_main()
