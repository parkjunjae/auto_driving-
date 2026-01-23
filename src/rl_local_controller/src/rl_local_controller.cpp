#include "rl_local_controller/rl_local_controller.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace rl_local_controller
{

void RlLocalController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;

  // 기본 파라미터를 선언하고 읽는다.
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".max_lin", rclcpp::ParameterValue(max_lin_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".max_ang", rclcpp::ParameterValue(max_ang_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".stop_dist", rclcpp::ParameterValue(stop_dist_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".hard_stop_dist", rclcpp::ParameterValue(hard_stop_dist_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".creep_speed", rclcpp::ParameterValue(creep_speed_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".in_place_heading", rclcpp::ParameterValue(in_place_heading_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".min_turn_rate", rclcpp::ParameterValue(min_turn_rate_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".slow_dist", rclcpp::ParameterValue(slow_dist_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".heading_gain", rclcpp::ParameterValue(heading_gain_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".avoid_gain", rclcpp::ParameterValue(avoid_gain_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".turn_gain", rclcpp::ParameterValue(turn_gain_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".turn_deadband", rclcpp::ParameterValue(turn_deadband_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".range_max", rclcpp::ParameterValue(range_max_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".lookahead_dist", rclcpp::ParameterValue(lookahead_dist_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".front_angle", rclcpp::ParameterValue(front_angle_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".side_angle", rclcpp::ParameterValue(side_angle_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".angle_step", rclcpp::ParameterValue(angle_step_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".raycast_step", rclcpp::ParameterValue(raycast_step_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".cost_threshold", rclcpp::ParameterValue(cost_threshold_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".treat_unknown_as_obstacle",
    rclcpp::ParameterValue(treat_unknown_as_obstacle_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".use_pid", rclcpp::ParameterValue(use_pid_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".pid_kp_lin", rclcpp::ParameterValue(pid_kp_lin_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".pid_ki_lin", rclcpp::ParameterValue(pid_ki_lin_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".pid_kd_lin", rclcpp::ParameterValue(pid_kd_lin_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".pid_kp_ang", rclcpp::ParameterValue(pid_kp_ang_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".pid_ki_ang", rclcpp::ParameterValue(pid_ki_ang_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".pid_kd_ang", rclcpp::ParameterValue(pid_kd_ang_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".pid_i_max_lin", rclcpp::ParameterValue(pid_i_max_lin_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".pid_i_max_ang", rclcpp::ParameterValue(pid_i_max_ang_));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".pid_dt_max", rclcpp::ParameterValue(pid_dt_max_));

  node_->get_parameter(name_ + ".max_lin", max_lin_);
  node_->get_parameter(name_ + ".max_ang", max_ang_);
  node_->get_parameter(name_ + ".stop_dist", stop_dist_);
  node_->get_parameter(name_ + ".hard_stop_dist", hard_stop_dist_);
  node_->get_parameter(name_ + ".creep_speed", creep_speed_);
  node_->get_parameter(name_ + ".slow_dist", slow_dist_);
  node_->get_parameter(name_ + ".heading_gain", heading_gain_);
  node_->get_parameter(name_ + ".avoid_gain", avoid_gain_);
  node_->get_parameter(name_ + ".turn_gain", turn_gain_);
  node_->get_parameter(name_ + ".turn_deadband", turn_deadband_);
  node_->get_parameter(name_ + ".range_max", range_max_);
  node_->get_parameter(name_ + ".lookahead_dist", lookahead_dist_);
  node_->get_parameter(name_ + ".front_angle", front_angle_);
  node_->get_parameter(name_ + ".side_angle", side_angle_);
  node_->get_parameter(name_ + ".angle_step", angle_step_);
  node_->get_parameter(name_ + ".raycast_step", raycast_step_);
  node_->get_parameter(name_ + ".cost_threshold", cost_threshold_);
  node_->get_parameter(name_ + ".treat_unknown_as_obstacle", treat_unknown_as_obstacle_);
  node_->get_parameter(name_ + ".use_pid", use_pid_);
  node_->get_parameter(name_ + ".pid_kp_lin", pid_kp_lin_);
  node_->get_parameter(name_ + ".pid_ki_lin", pid_ki_lin_);
  node_->get_parameter(name_ + ".pid_kd_lin", pid_kd_lin_);
  node_->get_parameter(name_ + ".pid_kp_ang", pid_kp_ang_);
  node_->get_parameter(name_ + ".pid_ki_ang", pid_ki_ang_);
  node_->get_parameter(name_ + ".pid_kd_ang", pid_kd_ang_);
  node_->get_parameter(name_ + ".pid_i_max_lin", pid_i_max_lin_);
  node_->get_parameter(name_ + ".pid_i_max_ang", pid_i_max_ang_);
  node_->get_parameter(name_ + ".pid_dt_max", pid_dt_max_);

  base_max_lin_ = max_lin_;
  base_max_ang_ = max_ang_;
  pid_last_time_ = rclcpp::Time(0, 0, node_->get_clock()->get_clock_type());
}

void RlLocalController::cleanup()
{
  // 별도 자원 정리가 필요하지 않다.
}

void RlLocalController::activate()
{
  // 활성화 단계에서 추가 동작은 없다.
}

void RlLocalController::deactivate()
{
  // 비활성화 단계에서 추가 동작은 없다.
}

void RlLocalController::setPlan(const nav_msgs::msg::Path & path)
{
  // 최신 전역 경로를 저장한다.
  plan_ = path;
}

geometry_msgs::msg::TwistStamped RlLocalController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker *)
{
  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.stamp = node_->now();
  cmd.header.frame_id = costmap_ros_->getBaseFrameID();

  if (plan_.poses.empty()) {
    // 경로가 없으면 정지한다.
    return cmd;
  }

  double target_x = 0.0;
  double target_y = 0.0;
  if (!getLookaheadTarget(pose, target_x, target_y)) {
    return cmd;
  }

  const double yaw = getYaw(pose);
  const double dx = target_x - pose.pose.position.x;
  const double dy = target_y - pose.pose.position.y;
  const double heading_error = normalizeAngle(std::atan2(dy, dx) - yaw);

  double front = range_max_;
  double left = range_max_;
  double right = range_max_;
  computeSectorDistances(pose, front, left, right);

  double v_des = 0.0;
  double w_des = 0.0;

  // 좌우 차이 데드밴드로 진동을 줄인다.
  const double lr_diff = left - right;

  // 목표 방향 오차가 크면 전진하지 않고 제자리 회전으로 정렬한다.
  if (std::abs(heading_error) > in_place_heading_ && front > hard_stop_dist_) {
    v_des = 0.0;
    const double turn_dir = heading_error >= 0.0 ? 1.0 : -1.0;
    w_des = turn_dir * std::max(min_turn_rate_, turn_gain_ * max_ang_ * 0.5);
  } else   if (front < stop_dist_) {
    // 전방이 막혀도 아주 천천히 전진해 재탐색을 유도한다.
    const double hard_stop = std::min(hard_stop_dist_, stop_dist_ * 0.95);
    const double denom = std::max(1e-3, stop_dist_ - hard_stop);
    double creep_scale = 0.0;
    if (front > hard_stop) {
      creep_scale = clamp((front - hard_stop) / denom, 0.0, 1.0);
    }
    v_des = creep_speed_ * creep_scale;

    // 여유가 큰 쪽을 우선해 회전한다.
    double turn_dir = 0.0;

    if (std::abs(lr_diff) < turn_deadband_) {
      // 좌우 차이가 작으면 목표 방향을 기준으로 회전한다.
      if (std::abs(heading_error) > 0.2) {
        turn_dir = heading_error >= 0.0 ? 1.0 : -1.0;
      } else if (last_turn_dir_ != 0) {
        turn_dir = static_cast<double>(last_turn_dir_);
      } else {
        turn_dir = 1.0;
      }
    } else {
      turn_dir = (lr_diff > 0.0) ? 1.0 : -1.0;
    }

    last_turn_dir_ = (turn_dir >= 0.0) ? 1 : -1;
    w_des = turn_dir * turn_gain_ * max_ang_;
  } else {
    // 전방 여유에 따라 전진 속도를 조절한다.
    const double speed_scale = clamp(front / slow_dist_, 0.0, 1.0);
    v_des = max_lin_ * speed_scale;

    // 목표 방향과 장애물 회피를 합쳐 회전 속도를 만든다.
    const double heading_term = heading_gain_ * clamp(heading_error / M_PI, -1.0, 1.0);
    double avoid_term = 0.0;
    if (std::abs(lr_diff) >= turn_deadband_) {
      avoid_term = avoid_gain_ * clamp(lr_diff / range_max_, -1.0, 1.0);
    }
    w_des = clamp((heading_term + avoid_term) * max_ang_, -max_ang_, max_ang_);
    if (std::abs(w_des) > 1e-3) {
      last_turn_dir_ = (w_des >= 0.0) ? 1 : -1;
    }
  }

  // PID를 쓰지 않으면 바로 목표 속도를 출력한다.
  if (!use_pid_) {
    cmd.twist.linear.x = v_des;
    cmd.twist.angular.z = w_des;
    return cmd;
  }

  // PID 기반으로 현재 속도와 목표 속도 차이를 보정한다.
  const auto now = node_->now();
  if (pid_last_time_.nanoseconds() == 0) {
    pid_last_time_ = now;
    pid_prev_err_lin_ = v_des - velocity.linear.x;
    pid_prev_err_ang_ = w_des - velocity.angular.z;
    pid_i_lin_ = 0.0;
    pid_i_ang_ = 0.0;
    cmd.twist.linear.x = v_des;
    cmd.twist.angular.z = w_des;
    return cmd;
  }

  const double dt = (now - pid_last_time_).seconds();
  if (dt <= 0.0 || dt > pid_dt_max_) {
    // 시간 간격이 비정상이면 적분을 초기화하고 목표 속도를 바로 사용한다.
    pid_last_time_ = now;
    pid_prev_err_lin_ = v_des - velocity.linear.x;
    pid_prev_err_ang_ = w_des - velocity.angular.z;
    pid_i_lin_ = 0.0;
    pid_i_ang_ = 0.0;
    cmd.twist.linear.x = v_des;
    cmd.twist.angular.z = w_des;
    return cmd;
  }

  const double err_lin = v_des - velocity.linear.x;
  const double err_ang = w_des - velocity.angular.z;
  pid_i_lin_ = clamp(pid_i_lin_ + err_lin * dt, -pid_i_max_lin_, pid_i_max_lin_);
  pid_i_ang_ = clamp(pid_i_ang_ + err_ang * dt, -pid_i_max_ang_, pid_i_max_ang_);
  const double d_lin = (err_lin - pid_prev_err_lin_) / dt;
  const double d_ang = (err_ang - pid_prev_err_ang_) / dt;

  const double corr_lin =
    pid_kp_lin_ * err_lin + pid_ki_lin_ * pid_i_lin_ + pid_kd_lin_ * d_lin;
  const double corr_ang =
    pid_kp_ang_ * err_ang + pid_ki_ang_ * pid_i_ang_ + pid_kd_ang_ * d_ang;

  double v_out = velocity.linear.x + corr_lin;
  double w_out = velocity.angular.z + corr_ang;

  // 최종 출력은 최대 속도 제한을 따른다.
  v_out = clamp(v_out, -max_lin_, max_lin_);
  w_out = clamp(w_out, -max_ang_, max_ang_);

  pid_prev_err_lin_ = err_lin;
  pid_prev_err_ang_ = err_ang;
  pid_last_time_ = now;

  cmd.twist.linear.x = v_out;
  cmd.twist.angular.z = w_out;
  return cmd;
}

void RlLocalController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  // 속도 제한을 선형/각속도에 동시에 반영한다.
  if (percentage) {
    const double scale = clamp(speed_limit / 100.0, 0.0, 1.0);
    max_lin_ = base_max_lin_ * scale;
    max_ang_ = base_max_ang_ * scale;
    return;
  }

  max_lin_ = std::max(0.0, std::min(speed_limit, base_max_lin_));
}

double RlLocalController::clamp(double x, double lo, double hi) const
{
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

double RlLocalController::normalizeAngle(double a) const
{
  while (a > M_PI) {
    a -= 2.0 * M_PI;
  }
  while (a < -M_PI) {
    a += 2.0 * M_PI;
  }
  return a;
}

double RlLocalController::getYaw(const geometry_msgs::msg::PoseStamped & pose) const
{
  tf2::Quaternion q;
  tf2::fromMsg(pose.pose.orientation, q);
  return tf2::getYaw(q);
}

bool RlLocalController::getLookaheadTarget(
  const geometry_msgs::msg::PoseStamped & pose,
  double & target_x,
  double & target_y) const
{
  const double px = pose.pose.position.x;
  const double py = pose.pose.position.y;

  for (const auto & p : plan_.poses) {
    const double dx = p.pose.position.x - px;
    const double dy = p.pose.position.y - py;
    const double dist = std::hypot(dx, dy);
    if (dist >= lookahead_dist_) {
      target_x = p.pose.position.x;
      target_y = p.pose.position.y;
      return true;
    }
  }

  // 룩어헤드 거리가 부족하면 마지막 목표를 사용한다.
  const auto & last = plan_.poses.back();
  target_x = last.pose.position.x;
  target_y = last.pose.position.y;
  return true;
}

double RlLocalController::raycastDistance(
  const geometry_msgs::msg::PoseStamped & pose,
  double angle) const
{
  const auto * costmap = costmap_ros_->getCostmap();
  const double yaw = getYaw(pose);
  const double world_angle = yaw + angle;
  const double start_x = pose.pose.position.x;
  const double start_y = pose.pose.position.y;

  for (double dist = 0.0; dist <= range_max_; dist += raycast_step_) {
    const double wx = start_x + std::cos(world_angle) * dist;
    const double wy = start_y + std::sin(world_angle) * dist;

    unsigned int mx = 0;
    unsigned int my = 0;
    if (!costmap->worldToMap(wx, wy, mx, my)) {
      // 맵 바깥은 장애물로 간주한다.
      return dist;
    }

    const unsigned char cost = costmap->getCost(mx, my);
    if (cost == nav2_costmap_2d::NO_INFORMATION) {
      if (treat_unknown_as_obstacle_) {
        return dist;
      }
      continue;
    }
    if (static_cast<int>(cost) >= cost_threshold_) {
      return dist;
    }
  }

  return range_max_;
}

void RlLocalController::computeSectorDistances(
  const geometry_msgs::msg::PoseStamped & pose,
  double & front,
  double & left,
  double & right) const
{
  front = range_max_;
  left = range_max_;
  right = range_max_;

  for (double a = -side_angle_; a <= side_angle_; a += angle_step_) {
    const double dist = raycastDistance(pose, a);
    if (a >= -front_angle_ && a <= front_angle_) {
      front = std::min(front, dist);
    } else if (a > front_angle_) {
      left = std::min(left, dist);
    } else {
      right = std::min(right, dist);
    }
  }
}

}  // namespace rl_local_controller

PLUGINLIB_EXPORT_CLASS(rl_local_controller::RlLocalController, nav2_core::Controller)
