# auto_driving-

## 가제보(ros_gz) 기반 PID 강화학습 절차

> Jetson Orin에서 GUI가 느릴 수 있어 **headless(-s)** 권장

### 1) 설치(arm64 기준)

```bash
sudo apt-get update
sudo apt-get install -y \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-bridge \
  ros-humble-gz-ros2-control \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers
```

### 2) 빌드

```bash
colcon build --packages-select tracer_description rl_pid_training --symlink-install
```

### 3) Gazebo 실행

GUI 버전:

```bash
source /opt/ros/humble/setup.bash
source ~/to_ws/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1   # GPU 이슈 시
ros2 launch tracer_description tracer_gazebo.launch.py
```

Headless(권장):

```bash
source /opt/ros/humble/setup.bash
source ~/to_ws/install/setup.bash
ros2 launch tracer_description tracer_gazebo.launch.py gz_args:="-r -s -v 3"
```

#### 컨트롤러 활성화 확인

```bash
ros2 control list_controllers
ros2 control set_controller_state joint_state_broadcaster active
ros2 control set_controller_state diff_drive_controller active
```

### 4) RL 학습 실행

```bash
source /opt/ros/humble/setup.bash
source ~/to_ws/install/setup.bash
source ~/to_ws/.venv/bin/activate
python3 ~/to_ws/rl_pid_training/train_pid.py
```

#### (필수) controller_server 실행 시 cmd_vel 리맵 + odom 토픽 맞추기

가제보에서는 실제 오도메트리 토픽이 `/diff_drive_controller/odom`이므로 아래처럼 맞춰야
controller_server가 정상적으로 목표 속도를 계산하고, diff_drive_controller가 cmd_vel을 받습니다.

```bash
ros2 run nav2_controller controller_server \
  --ros-args \
  --params-file /home/world/to_ws/src/rtabmap_ros/rtabmap_launch/launch/config/nav2_rtabmap_params.yaml \
  -p use_sim_time:=true \
  -p odom_topic:=/diff_drive_controller/odom \
  -r cmd_vel:=/diff_drive_controller/cmd_vel
```

#### 학습 로그 해석(예시)

```
| rollout/           |
|    ep_len_mean     | 87.2
|    ep_rew_mean     | -52.3
| time/              |
|    fps             | 8
|    total_timesteps | 2048
```

- `ep_len_mean`: 평균 에피소드 길이(스텝 수)
- `ep_rew_mean`: 평균 보상(값이 올라가면 성능 개선)
- `fps`: 초당 스텝 처리량(젯슨+Gazebo는 5~10fps가 흔함)
- `total_timesteps`: 누적 학습 스텝 수

#### PPO 학습 로그 파라미터 의미

- `ep_len_mean`: 에피소드 평균 길이(스텝 수). 일정하면 환경이 안정적임.
- `ep_rew_mean`: 에피소드 평균 보상. 덜 음수로 갈수록 성능 개선.
- `iterations`: PPO 업데이트 반복 횟수.
- `time_elapsed`: 학습 시작 후 경과 시간(초).
- `approx_kl`: 정책 변화량(KL 발산 근사). 너무 크면 불안정해질 수 있음.
- `clip_fraction`: PPO 클리핑 비율. 높을수록 업데이트가 거칠다는 뜻.
- `clip_range`: PPO 클리핑 폭(현재 0.2).
- `entropy_loss`: 탐색(랜덤성) 정도. 더 음수면 탐색이 많음.
- `explained_variance`: 가치함수 예측 성능(0~1). 1에 가까울수록 잘 맞음.
- `learning_rate`: 학습률.
- `loss`: 총 손실 값(추세를 보는 용도).
- `n_updates`: 누적 gradient 업데이트 횟수.
- `policy_gradient_loss`: 정책 업데이트 손실(0 근처면 안정적).
- `std`: 행동 분포 표준편차(탐색 크기).
- `value_loss`: 가치함수 손실(낮아질수록 좋음).

#### PID 파라미터 의미 (RLController 기준)

- `pid_kp_lin`: 선속도 오차에 즉각 반응하는 비례 이득
- `pid_ki_lin`: 선속도 누적 오차 보정(드리프트 보정), 과하면 저속에서 흔들림
- `pid_kd_lin`: 선속도 변화율 억제(오버슈트 완화)
- `pid_kp_ang`: 각속도 오차에 즉각 반응하는 비례 이득
- `pid_ki_ang`: 각속도 누적 오차 보정(저속 회전 보정), 과하면 회전 후 잔진동
- `pid_kd_ang`: 각속도 변화율 억제(회전 오버슈트 완화)
- `pid_i_max_lin`: 선속도 적분항 최대치(윈드업 방지 한계)
- `pid_i_max_ang`: 각속도 적분항 최대치(윈드업 방지 한계)
- `pid_dt_max`: PID 적분/미분에 쓰는 시간 간격 상한(이보다 크면 적분 초기화)

#### 종료/저장

- `train_pid.py`의 `model.learn(total_timesteps=...)`까지 학습하면 자동 종료
- 종료 시 모델 자동 저장: `/home/world/to_ws/<지정한모델 이름>.zip`

### 5) 동작 확인 (옵션)

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear:{x:0.3}, angular:{z:0.0}}" -r 10
```

### 6) 학습된 모델로 추론 실행

```bash
source /opt/ros/humble/setup.bash
source ~/to_ws/install/setup.bash
source ~/to_ws/.venv/bin/activate
python3 ~/to_ws/src/rl_pid_training/rl_pid_training/run_pid_policy.py
```

> 추론 중 PID 파라미터는 `ros2 param get /controller_server RLController.pid_kp_lin` 등으로 변화를 확인할 수 있습니다.

---
