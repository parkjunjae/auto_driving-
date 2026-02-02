# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROS 2 Humble-based autonomous mobile robot project combining:
- **RL PID Controller**: PPO-based adaptive PID gain tuning (training in Gazebo, inference on real robot)
- **Visual-Inertial SLAM**: RTAB-Map with Livox LiDAR + RealSense camera
- **Platform**: Tracer UGV with differential drive

**Languages**: Python 3.10 (RL/utilities), C++ (ROS 2 nodes)

## Build Commands

```bash
# Build all packages
colcon build --symlink-install

# Build specific packages (common pattern)
colcon build --packages-select tracer_description rl_pid_training --symlink-install
colcon build --packages-select camera_imu_pipeline_cpp --symlink-install

# Source environment (always needed after build)
source /opt/ros/humble/setup.bash
source ~/to_ws/install/setup.bash

# Activate Python venv for RL scripts
source ~/to_ws/.venv/bin/activate
```

## Key Workflows

### RL Training (Gazebo Simulation)

```bash
# Terminal 1: Launch Gazebo (headless recommended for Jetson)
ros2 launch tracer_description tracer_gazebo.launch.py gz_args:="-r -s -v 3"

# Terminal 2: Activate controllers
ros2 control set_controller_state joint_state_broadcaster active
ros2 control set_controller_state diff_drive_controller active

# Terminal 3: Run training
source ~/to_ws/.venv/bin/activate
python3 ~/to_ws/src/rl_pid_training/rl_pid_training/train_pid.py
```

### RL Inference (Real Robot)

```bash
python3 ~/to_ws/src/rl_pid_training/rl_pid_training/run_pid_policy.py \
  --model /home/world/to_ws/rl_pid_model_new \
  --odom-topic /odometry/filtered
```

### RTAB-Map with Livox LiDAR

Livox requires timestamp offset correction before deskewing:
```bash
# 1. Timestamp offset
ros2 launch livox_timestamp_offset livox_timestamp_offset.launch.py \
  input_topic:=/livox/lidar output_topic:=/livox/lidar/offset offset_sec:=-0.42

# 2. ICP Odometry
ros2 run rtabmap_odom icp_odometry --ros-args -r scan_cloud:=/livox/lidar/offset ...

# 3. Deskew
ros2 run rtabmap_util lidar_deskewing --ros-args \
  -p fixed_frame_id:=icp_odom -r input_cloud:=/livox/lidar/offset \
  -r output_cloud:=/livox/lidar/offset/deskewed

# 4. RTAB-Map
ros2 launch rtabmap_launch rtabmap.launch.py scan_cloud_topic:=/livox/lidar/offset/deskewed ...
```

## Architecture

```
src/
├── rl_pid_training/          # PPO training (train_pid.py) & inference (run_pid_policy.py)
│   └── rl_pid_env.py         # Gymnasium env for simulation
│   └── rl_pid_env_real.py    # Gymnasium env for real robot
├── camera_imu_pipeline_cpp/  # IMU bias correction + frame transformation (C++)
├── livox_timestamp_offset/   # Livox timestamp alignment
├── semantic_mapper_vslam/    # Visual SLAM with semantics
├── temp_goal_bt/             # Behavior tree for navigation goals (C++)
├── traversability_layer/     # Costmap layer for terrain analysis
├── tracer_ros2/              # Tracer UGV ROS 2 integration
└── [external packages]       # rtabmap_ros, realsense-ros, livox_ros_driver2, robot_localization
```

### Data Flow

```
Sensors → Processing → Localization → Control
  │           │            │            │
  ├─ RealSense RGB-D       │            │
  ├─ Livox LiDAR ─→ timestamp_offset ─→ deskew ─→ RTAB-Map SLAM
  └─ IMU ─→ imu_pipeline_cpp (bias correction) ─→ EKF fusion
                                                      │
                                              RL PID Controller
                                              (adjusts gains in real-time)
```

## Key Files

- `src/rl_pid_training/rl_pid_training/train_pid.py` - PPO training entry point
- `src/rl_pid_training/rl_pid_training/run_pid_policy.py` - Real robot inference
- `src/rl_pid_training/rl_pid_training/rl_pid_env.py` - Simulation environment with PID bounds
- `rtabmap_loop_status.py` - Loop closure monitoring CLI
- `rl_pid_model_new.zip` - Latest trained PPO model

## PID Parameter Bounds (rl_pid_env.py)

```python
kp_lin: 0.2 - 2.0    # Linear proportional
ki_lin: 0.0 - 0.05   # Linear integral
kd_lin: 0.0 - 0.5    # Linear derivative
kp_ang: 0.5 - 4.0    # Angular proportional
ki_ang: 0.0 - 0.05   # Angular integral
kd_ang: 0.0 - 0.8    # Angular derivative
```

## Important Notes

- **Gazebo on Jetson**: Use headless mode (`gz_args:="-r -s -v 3"`) for 2-3x speedup
- **Livox LiDAR**: Always apply timestamp offset before deskewing (check `now - stamp` to tune `offset_sec`)
- **TF verification**: `tf2_echo icp_odom livox_frame` must output continuously for deskew to work
- **Odom topics**: Gazebo uses `/diff_drive_controller/odom`, real robot uses `/odometry/filtered`
- **Documentation language**: Comments and docs are in Korean (한국어)
