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

### 7) 실차(Real) 적용 순서

아래 순서대로 올리면 됩니다. **중요:** `/controller_server/RLController/desired_cmd` 토픽은
`controller_server`에 **RLController 플러그인이 로드/활성화**된 경우에만 생성됩니다.

1) ROS 환경 소스

```bash
source /opt/ros/humble/setup.bash
source ~/to_ws/install/setup.bash
```

2) 하드웨어/센서/TF 기동  
   (트레이서 베이스 + 리얼센스 + 리복스 + 카메라/라이다 static TF)

```bash
# 예시: 사용 중인 실제 런치로 교체
# ros2 launch tracer_base tracer_base.launch.py
# ros2 launch realsense2_camera rs_launch.py
# ros2 launch livox_ros_driver2 msg_mid360.launch.py
# ros2 run tf2_ros static_transform_publisher ... (camera_link/livox_frame)
```

3) 로컬라이제이션(ekf) + 맵/내비게이션(RTAB-Map + Nav2)

```bash
# 예시: 실차 통합 런치 사용
ros2 launch rtabmap_launch rtabmap_nav2.launch.py
```

4) controller_server가 RLController로 뜨는지 확인

```bash
ros2 lifecycle get /controller_server
ros2 topic list | grep /controller_server/RLController/desired_cmd
```

5) **실차용 추론 실행**  
   (실차는 시뮬 시간이 아니므로 `--use-sim-time` 없이 실행)

```bash
source ~/to_ws/.venv/bin/activate
python3 ~/to_ws/src/rl_pid_training/rl_pid_training/run_pid_policy.py \
  --model /home/world/to_ws/rl_pid_model_new \
  --odom-topic /odometry/filtered \
  --desired-cmd-topic /controller_server/RLController/desired_cmd
```

6) 목표 주행은 기존 방식 그대로  
   (RViz 2D Nav Goal 또는 기존 목표 전송 로직 사용)

> 요약: **run_pid_policy.py는 PID 게인만 실시간으로 갱신**합니다.  
> 실제 이동/경로 생성은 기존 Nav2/RTAB-Map 흐름 그대로 유지됩니다.

---

## RTAB-Map 맵핑/루프클로저 안정화

### 1) IMU 파이프라인(C++)

IMU 축이 이미 정상이라면 **변환 기능은 끄는 것이 안전**합니다.  
기본값은 변환 OFF로 바꿔두었습니다.

```bash
colcon build --packages-select camera_imu_pipeline_cpp --symlink-install
source ~/to_ws/install/setup.bash

# 변환 OFF(기본값)
ros2 launch camera_imu_pipeline_cpp imu_pipeline_cpp.launch.py

# 필요 시 변환 ON
# ros2 launch camera_imu_pipeline_cpp imu_pipeline_cpp.launch.py imu_target_frame:=camera_imu_frame
```

### 2) Livox deskew (rtabmap.launch.py 사용 시)

`rtabmap_nav2.launch.py`를 쓰지 않으면 deskew 노드를 별도로 올려야 합니다.

```bash
ros2 run rtabmap_util lidar_deskewing --ros-args \
  -p fixed_frame_id:=odom \
  -p wait_for_transform:=0.5 \
  -p slerp:=true \
  -r input_cloud:=/livox/lidar \
  -r output_cloud:=/livox/lidar/deskewed
```

### 3) 루프클로저 강화 실행(인라인)

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
  rtabmap_viz:=true \
  localization:=false \
  delete_db_on_start:=true \
  imu_topic:=/camera/camera/imu_fixed \
  scan_cloud_topic:=/livox/lidar/deskewed \
  odom_sensor_sync:=false \
  RGBD/ProximityBySpace:=true \
  Rtabmap/LoopThr:=0.15 \
  Rtabmap/DetectionRate:=2.0 \
  Mem/STMSize:=50 \
  Mem/RehearsalSimilarity:=0.3 \
  Vis/MinInliers:=10 \
  RGBD/LinearUpdate:=0.15 \
  RGBD/AngularUpdate:=0.10 \
  Reg/Strategy:=1 \
  topic_queue_size:=30 \
  sync_queue_size:=30 \
  approx_sync_max_interval:=0.2
```

### 4) 루프클로저 자동 모니터링

```bash
python3 /home/world/to_ws/rtabmap_loop_status.py
```

출력에서 `accepted`가 0이 아니면 루프클로저가 붙은 것입니다.

---
