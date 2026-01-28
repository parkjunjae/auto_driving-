#pragma once

#include <memory>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/logger.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp/time.hpp"
#include "tf2_ros/buffer.h"

namespace rl_local_controller
{

class RlLocalController : public nav2_core::Controller
{
public:
  RlLocalController() = default;
  ~RlLocalController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  // 값 범위를 제한한다.
  double clamp(double x, double lo, double hi) const;

  // 각도를 -pi~pi로 정규화한다.
  double normalizeAngle(double a) const;

  // 로봇 자세에서 요 각도를 계산한다.
  double getYaw(const geometry_msgs::msg::PoseStamped & pose) const;

  // 전역 경로에서 룩어헤드 목표점을 찾는다.
  bool getLookaheadTarget(
    const geometry_msgs::msg::PoseStamped & pose,
    double & target_x,
    double & target_y) const;

  // 지정한 각도로 레이캐스팅해 장애물까지의 거리를 구한다.
  double raycastDistance(
    const geometry_msgs::msg::PoseStamped & pose,
    double angle) const;

  // 전방/좌/우 섹터의 최소 장애물 거리를 계산한다.
  void computeSectorDistances(
    const geometry_msgs::msg::PoseStamped & pose,
    double & front,
    double & left,
    double & right) const;

  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  rclcpp::Logger logger_{rclcpp::get_logger("rl_local_controller")};
  std::string name_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav_msgs::msg::Path plan_;

  // 파라미터(속도/회피/레이캐스팅 설정)
  double max_lin_{0.3};
  double max_ang_{1.0};
  double base_max_lin_{0.3};
  double base_max_ang_{1.0};
  double stop_dist_{0.5};
  // 전방이 너무 가까우면 완전 정지하는 거리
  double hard_stop_dist_{0.25};
  // 막힘 상태에서 아주 천천히 전진하는 속도
  double creep_speed_{0.05};
  // 제자리 회전을 시작하는 heading 오차(rad)
  double in_place_heading_{0.7};
  // 제자리 회전 최소 각속도(rad/s)
  double min_turn_rate_{0.4};
  double slow_dist_{1.2};
  double heading_gain_{1.0};
  double avoid_gain_{0.8};
  double turn_gain_{1.0};
  // 좌우 차이 데드밴드(회전 방향 흔들림 방지)
  double turn_deadband_{0.2};
  double range_max_{6.0};
  double lookahead_dist_{0.6};
  double front_angle_{0.35};
  double side_angle_{1.2};
  double angle_step_{0.05};
  double raycast_step_{0.05};
  int cost_threshold_{253};
  bool treat_unknown_as_obstacle_{false};

  // PID 기반 속도 보정(저수준 속도 추종 보완)
  bool use_pid_{true};
  double pid_kp_lin_{1.0};
  double pid_ki_lin_{0.0};
  double pid_kd_lin_{0.1};
  double pid_kp_ang_{2.0};
  double pid_ki_ang_{0.0};
  double pid_kd_ang_{0.1};
  double pid_i_max_lin_{0.5};
  double pid_i_max_ang_{1.0};
  double pid_dt_max_{0.5};

  // PID 내부 상태
  double pid_i_lin_{0.0};
  double pid_i_ang_{0.0};
  double pid_prev_err_lin_{0.0};
  double pid_prev_err_ang_{0.0};
  rclcpp::Time pid_last_time_;

  // PID 파라미터 동적 업데이트 콜백
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // PID 목표 속도 퍼블리셔(학습/디버깅용)
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr desired_cmd_pub_;
};

}  // namespace rl_local_controller

  // 직전 회전 방향(진동 억제용 히스테리시스)
  int last_turn_dir_{0};
