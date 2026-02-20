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

### 1-1. 센서 높이 변경(12cm 하향) 후 Grid Ground 재설정

- **배경**
  - LiDAR 높이를 12cm 낮춘 뒤, 소파가 `/rtabmap/map`에서 사라짐.
  - `/rtabmap/cloud_obstacles`의 `frame_id=map` 이므로 **Grid/Min/MaxGroundHeight는 map 좌표계 기준**이어야 함.
  - `map -> livox_frame`의 z가 `0.83` → **바닥은 map z=0** 기준으로 잡아야 정상.

- **계산 방법**
  - `map -> livox_frame` 변환 확인:
    ```bash
    ros2 run tf2_ros tf2_echo map livox_frame
    ```
  - `z=0.83`이면 바닥은 map `z≈0`

- **적용값(권장 시작값)**

  ```yaml
  Grid/MinGroundHeight: -0.03
  Grid/MaxGroundHeight: 0.03
  Grid/MinObstacleHeight: 0.08
  Grid/MaxObstacleHeight: 2.0
  ```

- **런타임 적용**

  ```bash
  ros2 param set /rtabmap/rtabmap Grid/MinGroundHeight "-0.05"
  ros2 param set /rtabmap/rtabmap Grid/MaxGroundHeight "0.05"
  ros2 param set /rtabmap/rtabmap Grid/MinObstacleHeight "0.10"
  ros2 param set /rtabmap/rtabmap Grid/MaxObstacleHeight "2.0"
  ```

- **검증**
  - 바닥은 `/rtabmap/cloud_ground`에만, 소파는 `/rtabmap/cloud_obstacles`로 분리되는지 확인:
    ```bash
    ros2 topic echo /rtabmap/cloud_ground --once | grep frame_id
    ros2 topic echo /rtabmap/cloud_obstacles --once | grep frame_id
    ```

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

### 3. 동적 장애물 제거(글로벌용 정적 후보 필터)

- **목표**
  - 사람/카트처럼 짧게 지나가는 동적 물체가 글로벌 맵에 궤적으로 누적되는 문제를 줄임.
  - 정적 장애물(벽/가구)은 유지하고, 동적은 글로벌 반영을 약화.

- **핵심 아이디어**
  - LiDAR 포인트를 시간-보셀(temporal voxel) 기준으로 누적해
    `반복 관측된 보셀만` 정적으로 인정.
  - 결과적으로 동적 물체는 히트 수가 부족해 필터 출력에서 제외됨.

- **수정 파일**
  - `src/livox_pointcloud_filter/src/dynamic_object_filter_node.cpp`
  - `src/livox_pointcloud_filter/CMakeLists.txt`
  - `src/livox_pointcloud_filter/launch/dynamic_object_filter.launch.py`
  - `src/rtabmap_ros/rtabmap_launch/launch/rtabmap_nav2.launch.py`
  - `src/rtabmap_ros/rtabmap_launch/launch/config/nav2_rtabmap_params.yaml`

- **노드 코드 변경 요약**
  - `min_static_sec` 파라미터 추가:
    - **연속 관측 시간이 `min_static_sec` 이상**일 때만 정적으로 인정
    - 사람이 잠깐 멈춰도 global에 남는 문제를 줄이기 위한 핵심 조건
  - 적용 위치:
    - `dynamic_object_filter_node.cpp`에서 `VoxelState`에 `first_seen_sec` 추가
    - `hits + (now - first_seen_sec) >= min_static_sec` 조건으로 정적 판정

- **토픽 체인(최종)**
  - `/livox/lidar/synced/deskewed` (deskew 입력)
  - -> `/livox/lidar/filtered` (기존 livox 필터 출력)
  - -> `/livox/lidar/static_filtered` (동적 필터 출력, 정적 위주)
  - 글로벌 코스트맵 `lidar_mark.topic`은 `/livox/lidar/static_filtered` 사용.
  - **RTAB-Map 입력은 `/livox/lidar/filtered` 유지**
    - 이유: `static_filtered`는 동적 제거용이라 디테일이 손실될 수 있음.
    - 정적맵 품질(루프클로저/정합) 유지를 위해 **RTAB-Map은 원본 필터링 결과를 사용**.
  - **`deskewed → filtered` 변경 이유**
    - `deskewed`는 보정은 됐지만 노이즈/외란이 많아 정합 불안정(고스팅/번짐)을 유발.
    - `filtered`는 `deskewed`를 입력으로 **보정 효과는 유지**하면서 Voxel+ROR로 노이즈 제거.
    - 결과적으로 **RTAB-Map의 루프클로저/정합 안정성**이 더 좋아짐.

- **런치 통합**
  - 동적 필터 노드는 `rtabmap_nav2.launch.py`에 통합됨.
  - `sensor_sync.launch.py`에는 의존성 꼬임 방지를 위해 포함하지 않음.

- **동적 필터 파라미터 의미/튜닝**
  - `voxel_size`:
    - 보셀 크기[m]. 클수록 빠르지만 디테일 손실.
  - `min_hits`:
    - 정적 인정 최소 관측 횟수. 클수록 동적 제거 강함.
  - `hit_window_sec`:
    - 히트 누적 시간창[s]. 길수록 정적 판단이 보수적.
  - `max_stale_sec`:
    - 오래 미관측된 보셀 상태 제거 시간[s].
  - `min_static_sec`:
    - 정적으로 인정되기까지 필요한 **연속 유지 시간**[s]. (사람 잔상 억제 핵심)
  - `z_min`, `z_max`:
    - 필터 대상 높이 범위[m].
  - `min_range`:
    - 근거리 노이즈 제거 거리[m].

- **현재 적용값(사람 잔상 억제용)**
  - `voxel_size=0.10`
  - `min_hits=6`
  - `hit_window_sec=0.7`
  - `max_stale_sec=1.0`
  - `min_static_sec=1.0`
  - `z_min=0.05`, `z_max=1.2`
  - `min_range=0.8`

- **튜닝 가이드**
  - 사람 잔상이 남으면: `min_hits↑` 또는 `min_static_sec↑`
  - 소파가 끊기면: `min_static_sec↓`, `hit_window_sec↑`
  - 근거리 링(원형) 남으면: `min_range↑`

- **검증 명령**

```bash
ros2 topic info /livox/lidar/static_filtered -v
ros2 topic hz /livox/lidar/static_filtered
```

- **재빌드**

```bash
cd ~/to_ws
colcon build --packages-select livox_pointcloud_filter
source ~/to_ws/install/setup.bash
```

---

## RLController 근거리 목표 접근 로직 개선 (제자리 회전 과다/원형 궤적 완화)

- **배경 문제**
  - 짧은 거리 목표에서 시작 직후 좌/우 회전이 먼저 나오거나, 목표 주변을 원형으로 도는 현상 발생.
  - 원인: heading 오차가 큰데 전진 속도(`v_des`)가 충분히 유지되면 회전 반경이 커져 원형 궤적이 만들어짐.

- **코드 수정 파일**
  - `src/rl_local_controller/src/rl_local_controller.cpp`
  - `src/rl_local_controller/include/rl_local_controller/rl_local_controller.hpp`
  - `src/rtabmap_ros/rtabmap_launch/launch/config/nav2_rtabmap_params.yaml`

- **핵심 로직 변경**
  1. 제자리 회전 조건을 목표 근처로 제한
     - 기존: `abs(heading_error) > in_place_heading`이면 제자리 회전
     - 변경: `dist_to_goal <= in_place_dist`일 때만 제자리 회전 허용
  2. heading 오차 기반 전진 감속 추가
     - `heading_slow_angle` 이상이면 `v_des`를 단계적으로 낮춤
     - `heading_slow_min_scale` 아래로는 떨어지지 않게 하여 완전 정지는 방지

- **추가/적용 파라미터**
  - `in_place_dist=0.5`
  - `in_place_heading=0.8`
  - `min_turn_rate=0.15`
  - `heading_slow_angle=0.8`
  - `heading_slow_min_scale=0.2`
  - `yaw_goal_tolerance=0.7`
  - `lookahead_dist=1.5`

- **변수 의미(코드 기준)**
  - `heading_error`: 현재 로봇 heading과 로컬 타겟 heading의 각도 오차(rad)
  - `dist_to_goal`: 현재 위치와 최종 goal pose 사이 거리(m)
  - `v_des`: RLController가 계산한 목표 선속도(m/s)
  - `w_des`: RLController가 계산한 목표 각속도(rad/s)
  - `in_place_heading`: 제자리 회전을 고려하기 시작하는 heading 오차 임계값(rad)
  - `in_place_dist`: 제자리 회전을 허용하는 goal 근접 거리 임계값(m)
  - `min_turn_rate`: 제자리 회전 시 보장할 최소 각속도(rad/s)
  - `heading_slow_angle`: 전진 감속을 시작하는 heading 오차 임계값(rad)
  - `heading_slow_min_scale`: heading 오차가 커도 유지할 최소 전진 비율(0~1)
  - `heading_abs`: `abs(heading_error)`로 계산한 절대 오차(rad)
  - `over`: `heading_slow_angle` 초과분을 0~1로 정규화한 값
  - `heading_scale`: heading 오차에 따라 `v_des`에 곱하는 감속 계수
  - `yaw_goal_tolerance`: goal 도달로 판정할 yaw 오차 허용치(rad)
  - `lookahead_dist`: 경로 추종 시 앞쪽 타겟을 잡는 거리(m)

- **기대 효과**
  - 멀리 있는 목표: 전진하면서 부드럽게 heading 정렬
  - 가까운 목표: 필요 시 제자리 회전으로 빠른 정렬
  - 짧은 거리 목표에서 원형 궤적/불필요 회전 감소

- **재빌드**

```bash
cd ~/to_ws
colcon build --packages-select rl_local_controller rtabmap_launch
source ~/to_ws/install/setup.bash
```

## 원복 메모 (1/2/4/5 항목)

- 아래 항목은 실주행 품질 저하(도착 지점 제자리 회전 증가, 글로벌맵 가시성 저하)로 **원복 완료**:
  - 1) `last_turn_dir_` 스코프 변경(클래스 멤버화)
  - 2) `controller_server.odom_topic=/odometry/filtered` 강제
  - 4) dynamic filter `min_range`를 센서 프레임에서 선적용
  - 5) global costmap 부하 완화(축소 크기/partial update 중심 설정)
- 현재 워크스페이스 기준으로는 위 4개는 적용하지 않고, 기존 설정으로 복귀한 상태에서 튜닝 진행.

## Agent PID 저속 anti-dither 안전가드 (추가 반영)

- **문제**
  - 목표점 근처 저속 구간에서 `w_ref` 부호가 자주 반전되어 좌우 떨림(헌팅) 발생.
  - 로그에서 마지막 구간에 `|w_ref|`는 크고 실제 `w_meas` 추종이 약한 패턴 확인.

- **핵심 전략**
  - RL이 PID 게인을 자동 조정하는 구조는 유지.
  - 대신 저속 회전 구간에서 `desired_cmd(w_ref)`를 안정화하는 안전가드 추가.
  - 즉, RL 대체가 아니라 RL 입력(참조 신호) 품질 개선 레이어.

- **수정 파일**
  - `src/rl_pid_training/rl_pid_training/rl_pid_env_real.py`
  - `src/rl_pid_training/rl_pid_training/run_pid_policy.py`
  - `src/rl_pid_training/launch/agent_pid.launch.py`

- **추가된 가드 로직**
  1. `w_ref` deadband:
     - 매우 작은 각속도 지령을 0으로 처리해 미세 떨림 제거.
  2. `w_ref` LPF:
     - 목표 각속도 급변 완화.
  3. 저속 회전 sign hold:
     - 짧은 시간 내 좌/우 반전 시 기존 부호 유지.
  4. 저속 각속도 상한:
     - 저속 구간 과도한 회전 지령 제한.

- **신규 파라미터**
  - `w_ref_lpf_alpha` (기본 `0.25`)
  - `w_ref_deadband` (기본 `0.03`)
  - `dither_v_ref_thresh` (기본 `0.06`)
  - `dither_w_ref_thresh` (기본 `0.15`)
  - `w_ref_sign_hold_sec` (기본 `0.35`)
  - `w_ref_abs_max_low_speed` (기본 `0.45`)

- **로그 확장**
  - CSV에 `w_ref_raw` 컬럼 추가.
  - `w_ref_raw`(원본) vs `w_ref`(가드 후)를 비교해 반전/떨림 억제 효과를 확인 가능.

- **실행 예시**

```bash
ros2 launch rl_pid_training agent_pid.launch.py \
  python_exec:=/home/world/to_ws/.venv/bin/python3 \
  w_ref_lpf_alpha:=0.25 \
  w_ref_deadband:=0.03 \
  dither_v_ref_thresh:=0.06 \
  dither_w_ref_thresh:=0.15 \
  w_ref_sign_hold_sec:=0.35 \
  w_ref_abs_max_low_speed:=0.45
```

---
