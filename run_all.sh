#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT}/logs/run_$(date +%Y%m%d_%H%M%S)"
SLEEP_SEC=5
mkdir -p "$LOG_DIR"

pids=()

run_cmd() {
  local name="$1"; shift
  echo "[RUN] $name"
  (
    "$@"
  ) > "${LOG_DIR}/${name}.log" 2>&1 &
  pids+=($!)
}

run_ros() {
  local name="$1"; shift
  run_cmd "$name" bash -lc "source /opt/ros/humble/setup.bash && source ${ROOT}/install/setup.bash && export RCUTILS_LOGGING_SEVERITY_THRESHOLD=30 && $*"
}

run_ros_with_nav2_env() {
  local name="$1"; shift
  run_cmd "$name" bash -lc "source /opt/ros/humble/setup.bash && source ${ROOT}/install/setup.bash && source ${ROOT}/install/rl_local_controller/share/rl_local_controller/local_setup.bash && source ${ROOT}/install/temp_goal_bt/share/temp_goal_bt/local_setup.bash && export RCUTILS_LOGGING_SEVERITY_THRESHOLD=30 && $*"
}

run_realsense() {
  local name="$1"; shift
  run_cmd "$name" bash -lc "source ${ROOT}/install/setup.bash && export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH && export PATH=/usr/local/bin:\$PATH && export RCUTILS_LOGGING_SEVERITY_THRESHOLD=30 && $*"
}

run_agent_pid() {
  local name="$1"; shift
  run_cmd "$name" bash -lc "source /opt/ros/humble/setup.bash && source ${ROOT}/install/setup.bash && source ${ROOT}/install/rl_local_controller/share/rl_local_controller/local_setup.bash && source ${ROOT}/install/temp_goal_bt/share/temp_goal_bt/local_setup.bash && source ${ROOT}/.venv/bin/activate && export RCUTILS_LOGGING_SEVERITY_THRESHOLD=30 && $*"
}

# 토픽이 준비될 때까지 대기(네비게이션 초기 실패 방지)
wait_for_topic() {
  local topic="$1"
  local timeout_sec="$2"
  echo "[WAIT] topic ${topic} (timeout ${timeout_sec}s)"
  if ! timeout "${timeout_sec}" bash -lc "source /opt/ros/humble/setup.bash && source ${ROOT}/install/setup.bash && ros2 topic echo ${topic} --once >/dev/null 2>&1"; then
    echo "[WARN] timeout waiting for ${topic}"
  else
    echo "[OK] ${topic} ready"
  fi
}

# TF가 준비될 때까지 대기(플래너/코스트맵 TF 오류 방지)
wait_for_tf() {
  local target="$1"
  local source="$2"
  local timeout_sec="$3"
  echo "[WAIT] TF ${target} -> ${source} (timeout ${timeout_sec}s)"
  if ! timeout "${timeout_sec}" bash -lc "source /opt/ros/humble/setup.bash && source ${ROOT}/install/setup.bash && python3 - <<'PY'
import sys, time
import rclpy
from tf2_ros import Buffer, TransformListener

target = sys.argv[1]
source = sys.argv[2]
timeout = float(sys.argv[3])

rclpy.init()
node = rclpy.create_node('tf_wait')
buf = Buffer()
listener = TransformListener(buf, node)

start = time.time()
ok = False
while time.time() - start < timeout:
    rclpy.spin_once(node, timeout_sec=0.1)
    if buf.can_transform(target, source, rclpy.time.Time()):
        ok = True
        break

node.destroy_node()
rclpy.shutdown()
if not ok:
    sys.exit(1)
PY ${target} ${source} ${timeout_sec}"; then
    echo "[WARN] timeout waiting for TF ${target} -> ${source}"
  else
    echo "[OK] TF ${target} -> ${source} ready"
  fi
}

cleanup() {
  echo "[STOP] stopping child processes..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup INT TERM

echo "[INFO] logs: ${LOG_DIR}"

# 1) CAN bring-up (ignore busy/failed)
if ! sudo ip link set can1 up type can bitrate 500000; then
  echo "[WARN] can1 up failed or busy, continuing..."
fi

# 2) Sensors
run_ros "livox" "ros2 launch livox_ros_driver2 msg_MID360_launch.py"
sleep "$SLEEP_SEC"
run_realsense "realsense" "ros2 launch realsense2_camera rs_launch.py"
sleep "$SLEEP_SEC"

# 3) Base + sync + EKF
run_ros "tracer_base" "ros2 launch tracer_base tracer_base.launch.py"
sleep "$SLEEP_SEC"
run_ros "sensor_sync" "ros2 launch rtabmap_launch sensor_sync.launch.py"
sleep "$SLEEP_SEC"
run_ros "ekf" "ros2 launch robot_localization ekf.launch.py"
sleep "$SLEEP_SEC"

# 4) RTAB-Map + Nav2
run_ros_with_nav2_env "rtabmap_nav2" "ros2 launch rtabmap_launch rtabmap_nav2.launch.py"
sleep "$SLEEP_SEC"

# 4.5) 핵심 토픽/TF가 준비된 뒤 agent PID 시작
wait_for_topic "/odometry/filtered" 20
wait_for_tf "odom" "base_link" 20
wait_for_topic "/rtabmap/map" 60
wait_for_tf "map" "odom" 30
wait_for_topic "/global_costmap/costmap" 30

# 5) Agent PID (venv)
run_agent_pid "agent_pid" "ros2 launch rl_pid_training agent_pid.launch.py"

echo "[INFO] all processes started. Ctrl+C to stop."
wait
