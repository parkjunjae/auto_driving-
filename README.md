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
ros2 launch rl_pid_training agent_pid.launch.py \
  python_exec:=/home/world/to_ws/.venv/bin/python3
```

> 추론 중 PID 파라미터는 `ros2 param get /controller_server RLController.pid_kp_lin` 등으로 변화를 확인할 수 있습니다.

### 7) 실차(Real) 적용 순서

아래 순서대로 올리면 됩니다. **중요:** `/controller_server/RLController/desired_cmd` 토픽은
`controller_server`에 **RLController 플러그인이 로드/활성화**된 경우에만 생성됩니다.

1. ROS 환경 소스

```bash
source /opt/ros/humble/setup.bash
source ~/to_ws/install/setup.bash
```

2. 하드웨어/센서/TF 기동  
   (트레이서 베이스 + 리얼센스 + 리복스 + 카메라/라이다 static TF)

```bash
# 예시: 사용 중인 실제 런치로 교체
# ros2 launch tracer_base tracer_base.launch.py
# ros2 launch realsense2_camera rs_launch.py
# ros2 launch livox_ros_driver2 msg_mid360.launch.py
# ros2 run tf2_ros static_transform_publisher ... (camera_link/livox_frame)
```

3. EKF + 맵/내비게이션(RTAB-Map + Nav2)

```bash
# 예시: 실차 통합 런치 사용
ros2 launch rtabmap_launch rtabmap_nav2.launch.py
```

4. controller_server가 RLController로 뜨는지 확인

```bash
ros2 lifecycle get /controller_server
ros2 topic list | grep /controller_server/RLController/desired_cmd
```

5. **실차용 추론 실행**  
   (실차는 시뮬 시간이 아니므로 `--use-sim-time` 없이 실행)

```bash
source ~/to_ws/.venv/bin/activate
ros2 launch rl_pid_training agent_pid.launch.py \
  python_exec:=/home/world/to_ws/.venv/bin/python3 \
  model:=/home/world/to_ws/rl_pid_model_new \
  odom_topic:=/odometry/filtered \
  desired_cmd_topic:=/controller_server/RLController/desired_cmd
```

6. 목표 주행은 기존 방식 그대로  
   (RViz 2D Nav Goal 또는 기존 목표 전송 로직 사용)

> 요약: **run_pid_policy.py는 PID 게인만 실시간으로 갱신**합니다.  
> 실제 이동/경로 생성은 기존 Nav2/RTAB-Map 흐름 그대로 유지됩니다.

### 8) Agent PID 실차 안정화 (최신 기본값 + 적용 이유)

- **왜 수정했는가**
  - 기존에는 agent가 게인을 너무 자주/크게 바꿔 `kp/kd`가 경계값에 붙는 헌팅이 발생했고,
    전진 중 좌우 떨림(`w_meas` 부호 전환)으로 이어짐.
  - 목표는 **학습 모델은 유지**하면서, 실차에서 게인 변경을 보수화해 진동을 줄이는 것.

- **수정 파일**
  - `src/rl_pid_training/rl_pid_training/rl_pid_env_real.py`
  - `src/rl_pid_training/rl_pid_training/run_pid_policy.py`
  - `src/rl_pid_training/launch/agent_pid.launch.py`

- **무엇을 어떻게 바꿨는가**
  - 게인 범위 축소(`PidBounds`): 모델이 과격한 영역으로 튀지 않도록 제한
    - `kp_lin: 1.05~1.30`, `ki_lin: 0.00~0.01`, `kd_lin: 0.02~0.08`
    - `kp_ang: 1.10~1.35`, `ki_ang: 0.00~0.01`, `kd_ang: 0.07~0.14`
  - 초기값 고정: 실차 저오차 구간 중앙값 근처에서 시작
    - `kp_lin=1.20`, `ki_lin=0.0`, `kd_lin=0.03`
    - `kp_ang=1.23`, `ki_ang=0.0`, `kd_ang=0.10`
  - 업데이트 완화: `step_dt=0.4`, `gain_scale=0.2`, `gain_steps` 축소
  - 미세 잡음 제거: `action_deadzone=0.25` (작은 action 무시)
  - 상황별 freeze:
    - 정지: 게인 업데이트 중지 (`stop_freeze_*`)
    - 직진: yaw PID 업데이트 중지 (`straight_freeze_*`)
    - 제자리 회전: lin PID 업데이트 중지 (`rotate_freeze_*`)

- **LPF 적용 방식 (핵심)**
  - 목적: 모델 action이 순간적으로 커져도, 게인이 급점프하지 않게 완만하게 반영
  - 처리 순서:
    1. `action_raw` 수신
    2. deadzone 적용 (`|action| < action_deadzone -> 0`)
    3. `gain_scale` 곱해 증감량 축소
    4. `gain_steps` 곱해 파라미터별 delta 생성
    5. `target_gain = clamp(current_gain + delta, min, max)`
    6. `applied_gain = current_gain + alpha * (target_gain - current_gain)` (`alpha=gain_lpf_alpha`)
  - 해석:
    - `alpha`가 작을수록 더 부드럽고 안정적(대신 반응 느림)
    - 범위 제한 + LPF를 같이 써서 "급격한 게인 진동"을 억제

- **런치 인자 확장 (`agent_pid.launch.py`)**
  - 런타임에서 바로 조정 가능:
    - `step_dt`, `gain_scale`, `gain_lpf_alpha`, `action_deadzone`
    - `straight_freeze_*`, `stop_freeze_*`, `rotate_freeze_*`
  - `python_exec` 인자로 가상환경 Python 강제 실행 가능

- **Nav2 기본 PID 동기화**
  - agent 시작 전 기본 PID도 같은 기준으로 맞춤:
    - `src/rtabmap_ros/rtabmap_launch/launch/config/nav2_rtabmap_params.yaml`
    - `src/rtabmap_ros/rtabmap_launch/launch/config/nav2_rtabmap_params_train.yaml`
    - `pid_kp_lin=1.2`, `pid_kd_lin=0.03`, `pid_kp_ang=1.23`, `pid_kd_ang=0.10`

- **재빌드**

```bash
cd ~/to_ws
colcon build --packages-select rl_pid_training rtabmap_launch
source ~/to_ws/install/setup.bash
```

---

## RTAB-Map 맵핑/TF 안정화 (현재 반영 상태)

### 1. 시간동기/deskew/TF 보정

- **목표**
  - LiDAR/IMU/odom 시간축 불일치와 TF 초기 불안정을 줄여 맵 뒤틀림과 고스팅을 완화.

- **수정한 내용**
  - `sensor_sync.launch.py`에서 Livox 타임스탬프 오프셋 적용:
    - `/livox/lidar -> /livox/lidar/synced`
    - 운영값: `lidar_offset_sec:=0.036`
  - deskew/필터/RTAB-Map 입력 토픽 경로 통일:
    - deskew 입력 기준: `/livox/lidar/synced/deskewed`
    - 필터 출력: `/livox/lidar/filtered`
  - `rtabmap_nav2.launch.py`에서 TF 체인 고정:
    - `odom_topic=/odometry/filtered`
    - `odom_frame_id=odom`
    - `map_frame_id=map`
    - `publish_tf_map=true`
    - 내부 odom 중복 방지: `visual_odometry=false`, `icp_odometry=false`
  - `rtabmap.launch.py`:
    - `tf_delay: 0.05`로 조정
    - `Grid/MaxGroundHeight: 0.07`로 경고 제거
    - 정합 보수화 반영:
      - `Reg/Strategy=2 (Vis+ICP)`
      - `Reg/Force3DoF=true`
      - `RGBD/ProximityBySpace=false`
      - `Vis/MinInliers=20`
      - `Rtabmap/LoopThr=0.25`
      - `RGBD/OptimizeMaxError=0.3`
      - `RGBD/AngularUpdate=0.20`

- **검증 명령**

```bash
ros2 topic delay /livox/lidar/synced
ros2 topic delay /livox/lidar/synced/deskewed
ros2 topic delay /odometry/filtered

sleep 5
python3 ~/to_ws/tf_jump_monitor.py --jump-trans 0.03 --jump-rot-deg 2 --output ~/to_ws/tf_jump_final_sensitive.csv
```

- **검증 결과 요약**
  - `/livox/lidar/synced` delay 평균 `~0.001~0.002s`
  - `/odometry/filtered` delay 평균 `~0.003s` (max `~0.010s`)
  - TF `MISSING` 1회는 시작 시점(`0.00s`) transient로 확인됨

### 2. EKF 튜닝 (odom->base_link 안정화)

- **문제**
  - 회전 구간에서 `odom->base_link` 점프가 많고, RViz에서 전진 시 대각선 드리프트가 발생.

- **원인 분석**
  - wheel `/odom`의 `pose/yaw` 공분산이 너무 작아 EKF가 wheel pose를 과신.
  - 실측 반복 오차에서 회전(360도) yaw 오차가 크게 확인됨.

- **수정한 내용**
  - `src/robot_localization/params/ekf.yaml`:
    - `predict_to_current_time=false`
    - `odom0_config`: `vx`만 사용 (`x,y,yaw,vyaw` 비활성)
    - `imu0_config`: `yaw`, `vyaw` 사용
  - `src/tracer_ros2/tracer_base/include/tracer_base/tracer_messenger.hpp`:
    - wheel pose/yaw 비신뢰 처리:
      - `pose.covariance[x,y,yaw] = 1e6`
    - wheel 속도는 `vx`만 제한적으로 사용:
      - `twist.covariance[vx] = 0.01`
      - `twist.covariance[vyaw] = 1e6`

- **검증 결과 요약**
  - 튜닝 후 민감 기준에서:
    - `odom->base_link JUMP`가 크게 감소 (`53 -> 11`)
    - `MISSING`은 시작 transient 1회 수준
  - 해석:
    - 회전 시 관측 충돌이 줄었고, TF/odom 체인이 실사용 수준으로 안정화됨

- **추가 미세조정 포인트**
  - 회전 시 아직 흔들리면 `twist.covariance[vx]`를 `0.01 -> 0.02`로 상향
  - 반응이 둔하면 `0.01 -> 0.005`로 하향
