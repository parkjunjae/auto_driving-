import rclpy
from stable_baselines3 import PPO

from rl_pid_training.rl_pid_env import PidGainEnv


def main():
    # 학습된 모델 로드(학습 종료 시 저장된 파일)
    model = PPO.load("/home/world/to_ws/rl_pid_model")

    # 환경 생성(시뮬 실행 + controller_server 활성 상태여야 함)
    env = PidGainEnv()

    # 초기화
    obs, _ = env.reset()

    try:
        while rclpy.ok():
            # 정책으로부터 행동 추론
            action, _ = model.predict(obs, deterministic=True)
            # 행동 적용
            obs, _, _, _, _ = env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    rclpy.init()
    main()
