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
    args = parser.parse_args()

    # 학습된 모델 로드(학습 종료 시 저장된 파일)
    model = PPO.load(args.model)

    # 환경 생성(시뮬 실행 + controller_server 활성 상태여야 함)
    env = RealPidGainEnv(
        odom_topic=args.odom_topic,
        desired_cmd_topic=args.desired_cmd_topic,
        desired_cmd_type=args.desired_cmd_type,
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


if __name__ == "__main__":
    rclpy.init()
    main()
