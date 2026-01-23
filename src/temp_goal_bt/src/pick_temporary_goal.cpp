#include "temp_goal_bt/pick_temporary_goal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "behaviortree_cpp_v3/bt_factory.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace
{
constexpr int kUnknownCost = 255;

static double normalizeAngle(double a)
{
  while (a > M_PI) {
    a -= 2.0 * M_PI;
  }
  while (a < -M_PI) {
    a += 2.0 * M_PI;
  }
  return a;
}
}

namespace temp_goal_bt
{

PickTemporaryGoal::PickTemporaryGoal(const std::string & name, const BT::NodeConfiguration & config)
: BT::SyncActionNode(name, config)
{
  // BT 블랙보드에서 nav2 노드를 받아서 같은 실행 컨텍스트를 사용한다.
  node_ = config.blackboard->get<rclcpp::Node::SharedPtr>("node");
  tf_buffer_ = config.blackboard->get<std::shared_ptr<tf2_ros::Buffer>>("tf_buffer");

  costmap_topic_ = "local_costmap/costmap_raw";
  getInput("costmap_topic", costmap_topic_);

  callback_group_ = node_->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive, false);
  callback_group_executor_.add_callback_group(
    callback_group_, node_->get_node_base_interface());

  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_;
  costmap_sub_ = node_->create_subscription<nav2_msgs::msg::Costmap>(
    costmap_topic_, rclcpp::QoS(1),
    std::bind(&PickTemporaryGoal::costmapCallback, this, std::placeholders::_1),
    options);
}

BT::PortsList PickTemporaryGoal::providedPorts()
{
  return {
    BT::OutputPort<geometry_msgs::msg::PoseStamped>("temp_goal", "임시 목표 Pose"),
    BT::InputPort<geometry_msgs::msg::PoseStamped>("goal", "원래 목표 Pose"),
    BT::InputPort<double>("goal_heading_weight", 0.6, "목표 방향 선호 가중치"),
    BT::InputPort<double>("min_goal_cos", 0.0, "목표 방향과의 최소 코사인"),
    BT::InputPort<bool>("debug_log", false, "goal_valid 로그 출력 여부"),
    BT::InputPort<std::string>("costmap_topic", "local_costmap/costmap_raw", "코스트맵 토픽"),
    BT::InputPort<std::string>("base_frame", "base_link", "로봇 기준 프레임"),
    BT::InputPort<double>("max_goal_dist", 1.0, "임시 목표 최대 거리"),
    BT::InputPort<double>("min_goal_dist", 0.3, "임시 목표 최소 거리"),
    BT::InputPort<double>("min_clearance", 0.4, "최소 통과 여유 거리"),
    BT::InputPort<double>("safety_margin", 0.2, "장애물과의 안전 여유"),
    BT::InputPort<double>("angle_min", -1.2, "탐색 최소 각도(rad)"),
    BT::InputPort<double>("angle_max", 1.2, "탐색 최대 각도(rad)"),
    BT::InputPort<double>("angle_step", 0.2, "각도 샘플 간격(rad)"),
    BT::InputPort<double>("raycast_step", 0.05, "레이캐스트 간격(m)"),
    BT::InputPort<int>("cost_threshold", 253, "장애물로 간주할 비용 임계값"),
    BT::InputPort<bool>("treat_unknown_as_obstacle", false, "UNKNOWN 셀을 장애물로 볼지")
  };
}

void PickTemporaryGoal::costmapCallback(const nav2_msgs::msg::Costmap::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(costmap_mutex_);
  latest_costmap_ = msg;
}

static bool worldToMap(
  const nav2_msgs::msg::Costmap & costmap,
  const double wx, const double wy,
  int & mx, int & my)
{
  // 코스트맵은 로컬 좌표에서 축과 정렬되어 있다고 가정한다.
  const auto & meta = costmap.metadata;
  const double ox = meta.origin.position.x;
  const double oy = meta.origin.position.y;
  const double resolution = meta.resolution;

  mx = static_cast<int>(std::floor((wx - ox) / resolution));
  my = static_cast<int>(std::floor((wy - oy) / resolution));

  if (mx < 0 || my < 0) {
    return false;
  }
  if (mx >= static_cast<int>(meta.size_x) || my >= static_cast<int>(meta.size_y)) {
    return false;
  }
  return true;
}

static bool isBlocked(
  const nav2_msgs::msg::Costmap & costmap,
  const int mx, const int my,
  const int cost_threshold,
  const bool treat_unknown_as_obstacle)
{
  const auto & meta = costmap.metadata;
  const size_t index = static_cast<size_t>(my) * meta.size_x + static_cast<size_t>(mx);
  const uint8_t cost = costmap.data[index];

  if (cost == kUnknownCost) {
    return treat_unknown_as_obstacle;
  }
  return static_cast<int>(cost) >= cost_threshold;
}

static double raycastFreeDistance(
  const nav2_msgs::msg::Costmap & costmap,
  const double base_x, const double base_y,
  const double heading,
  const double max_dist,
  const double raycast_step,
  const int cost_threshold,
  const bool treat_unknown_as_obstacle)
{
  double distance = 0.0;
  const double cos_h = std::cos(heading);
  const double sin_h = std::sin(heading);

  for (; distance <= max_dist; distance += raycast_step) {
    const double wx = base_x + distance * cos_h;
    const double wy = base_y + distance * sin_h;

    int mx = 0;
    int my = 0;
    if (!worldToMap(costmap, wx, wy, mx, my)) {
      // 코스트맵 밖으로 나가면 현재 거리까지는 안전하다고 본다.
      return distance;
    }

    if (isBlocked(costmap, mx, my, cost_threshold, treat_unknown_as_obstacle)) {
      return distance;
    }
  }

  return max_dist;
}

BT::NodeStatus PickTemporaryGoal::tick()
{
  callback_group_executor_.spin_some();

  nav2_msgs::msg::Costmap::SharedPtr costmap;
  {
    std::lock_guard<std::mutex> lock(costmap_mutex_);
    costmap = latest_costmap_;
  }

  if (!costmap) {
    RCLCPP_WARN(node_->get_logger(), "임시 목표 생성: 코스트맵이 아직 없습니다.");
    return BT::NodeStatus::FAILURE;
  }

  std::string base_frame = "base_link";
  getInput("base_frame", base_frame);

  double max_goal_dist = 1.0;
  double min_goal_dist = 0.3;
  double min_clearance = 0.4;
  double safety_margin = 0.2;
  double angle_min = -1.2;
  double angle_max = 1.2;
  double angle_step = 0.2;
  double raycast_step = 0.05;
  int cost_threshold = 253;
  bool treat_unknown_as_obstacle = false;
  double goal_heading_weight = 0.6;
  double min_goal_cos = 0.0;
  bool debug_log = false;
  geometry_msgs::msg::PoseStamped goal;
  // BT의 Expected 결과로 goal 포트 유무를 판단한다.
  const auto goal_result = getInput("goal", goal);
  const bool has_goal = static_cast<bool>(goal_result);

  getInput("max_goal_dist", max_goal_dist);
  getInput("min_goal_dist", min_goal_dist);
  getInput("min_clearance", min_clearance);
  getInput("safety_margin", safety_margin);
  getInput("angle_min", angle_min);
  getInput("angle_max", angle_max);
  getInput("angle_step", angle_step);
  getInput("raycast_step", raycast_step);
  getInput("cost_threshold", cost_threshold);
  getInput("treat_unknown_as_obstacle", treat_unknown_as_obstacle);
  getInput("goal_heading_weight", goal_heading_weight);
  getInput("min_goal_cos", min_goal_cos);
  getInput("debug_log", debug_log);

  if (debug_log && !has_goal) {
    RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
      "임시 목표 생성: goal 포트가 비어 있어 방향 가중치를 건너뜁니다.");
  }

  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_->lookupTransform(
      costmap->header.frame_id, base_frame, rclcpp::Time(0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(node_->get_logger(), "임시 목표 생성: TF 실패: %s", ex.what());
    return BT::NodeStatus::FAILURE;
  }

  const double base_x = tf.transform.translation.x;
  const double base_y = tf.transform.translation.y;
  const double base_yaw = tf2::getYaw(tf.transform.rotation);

  // 원래 목표 방향을 추정해 임시 목표를 전진 방향으로 유도한다.
  bool goal_valid = false;
  double goal_dir = 0.0;
  if (has_goal) {
    geometry_msgs::msg::PoseStamped goal_in_costmap = goal;
    if (!goal.header.frame_id.empty() && goal.header.frame_id != costmap->header.frame_id) {
      try {
        const auto tf_goal = tf_buffer_->lookupTransform(
          costmap->header.frame_id, goal.header.frame_id, rclcpp::Time(0));
        tf2::doTransform(goal, goal_in_costmap, tf_goal);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(node_->get_logger(), "임시 목표 생성: goal TF 실패: %s", ex.what());
      }
    }

    if (goal_in_costmap.header.frame_id == costmap->header.frame_id) {
      const double gx = goal_in_costmap.pose.position.x;
      const double gy = goal_in_costmap.pose.position.y;
      const double gdist = std::hypot(gx - base_x, gy - base_y);
      if (gdist > 0.05) {
        goal_dir = std::atan2(gy - base_y, gx - base_x);
        goal_valid = true;
      }
    }
  }

  if (debug_log && has_goal && !goal_valid) {
    RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
      "임시 목표 생성: goal_valid=false (좌표 변환 실패 또는 너무 근접).");
  }

  const double max_check_dist = std::max(max_goal_dist, min_clearance);
  // 목표 방향과 여유 공간을 함께 고려해 각도 점수를 계산한다.
  double best_clearance_any = -std::numeric_limits<double>::infinity();
  double best_angle_any = 0.0;
  double best_score = -std::numeric_limits<double>::infinity();
  double best_angle_score = 0.0;
  double best_clearance_score = -std::numeric_limits<double>::infinity();
  bool found_scored = false;

  for (double angle = angle_min; angle <= angle_max; angle += angle_step) {
    const double heading = base_yaw + angle;
    const double clear_dist = raycastFreeDistance(
      *costmap, base_x, base_y, heading, max_check_dist, raycast_step,
      cost_threshold, treat_unknown_as_obstacle);

    if (clear_dist > best_clearance_any) {
      best_clearance_any = clear_dist;
      best_angle_any = angle;
    }

    if (clear_dist < min_clearance || !goal_valid) {
      continue;
    }

    const double goal_diff = normalizeAngle(heading - goal_dir);
    const double goal_cos = std::cos(goal_diff);
    if (goal_cos < min_goal_cos) {
      continue;
    }

    const double clearance_score = clear_dist / max_check_dist;
    const double heading_score = (goal_cos + 1.0) * 0.5;
    const double score =
      (1.0 - goal_heading_weight) * clearance_score + goal_heading_weight * heading_score;

    if (score > best_score) {
      best_score = score;
      best_angle_score = angle;
      best_clearance_score = clear_dist;
      found_scored = true;
    }
  }

  // 목표 방향 점수가 없으면 여유 공간만으로 선택한다.
  double best_clearance = best_clearance_any;
  double best_angle = best_angle_any;
  if (goal_valid && found_scored) {
    best_clearance = best_clearance_score;
    best_angle = best_angle_score;
  }
  if (debug_log && goal_valid && !found_scored) {
    RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
      "임시 목표 생성: 목표 방향 조건을 만족하는 각도가 없어 여유 공간만 사용합니다.");
  }

  if (best_clearance < min_clearance) {
    RCLCPP_WARN(node_->get_logger(), "임시 목표 생성: 충분한 여유 공간이 없습니다.");
    return BT::NodeStatus::FAILURE;
  }

  double goal_dist = std::min(max_goal_dist, best_clearance - safety_margin);
  if (goal_dist < min_goal_dist) {
    goal_dist = std::min(max_goal_dist, best_clearance * 0.8);
  }
  if (goal_dist < min_goal_dist) {
    RCLCPP_WARN(node_->get_logger(), "임시 목표 생성: 목표 거리가 너무 짧습니다.");
    return BT::NodeStatus::FAILURE;
  }

  const double goal_heading = base_yaw + best_angle;

  geometry_msgs::msg::PoseStamped temp_goal;
  temp_goal.header.stamp = node_->now();
  temp_goal.header.frame_id = costmap->header.frame_id;
  temp_goal.pose.position.x = base_x + goal_dist * std::cos(goal_heading);
  temp_goal.pose.position.y = base_y + goal_dist * std::sin(goal_heading);
  temp_goal.pose.position.z = 0.0;

  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, goal_heading);
  temp_goal.pose.orientation = tf2::toMsg(q);

  setOutput("temp_goal", temp_goal);
  return BT::NodeStatus::SUCCESS;
}

}  // namespace temp_goal_bt

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<temp_goal_bt::PickTemporaryGoal>("PickTemporaryGoal");
}
