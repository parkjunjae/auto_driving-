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

#### 종료/저장
- `train_pid.py`의 `model.learn(total_timesteps=...)`까지 학습하면 자동 종료
- 종료 시 모델 자동 저장: `/home/world/to_ws/rl_pid_model.zip`

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

## Gazebo에서 로봇이 작게/납작하게 보이는 경우
- 기본 카메라가 멀리 있고 그리드가 크게 보여 **로봇이 작고 낮게** 보일 수 있습니다.
- 카메라 줌/오빗(마우스 휠/우클릭 드래그)으로 가까이 보거나  
  Entity Tree에서 `tracer` 선택 → 카메라 follow 하면 정상적으로 보입니다.
