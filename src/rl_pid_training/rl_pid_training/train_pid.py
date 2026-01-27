import rclpy

from rl_pid_training.rl_pid_env import PidGainEnv


def main():
    # Gymnasium 환경 생성
    env = PidGainEnv()

    # stable-baselines3 학습 예시
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError("stable-baselines3가 필요합니다. 가상환경에서 설치하세요.") from e

    model = PPO("MlpPolicy", env, verbose=1)

    # 학습 스텝 수는 환경에 맞게 조절
    model.learn(total_timesteps=20000)

    # 모델 저장
    model.save("/home/world/to_ws/rl_pid_model")

    env.close()


if __name__ == '__main__':
    rclpy.init()
    main()
