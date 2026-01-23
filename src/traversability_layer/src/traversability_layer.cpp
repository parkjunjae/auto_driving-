#include "traversability_layer/traversability_layer.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

PLUGINLIB_EXPORT_CLASS(traversability_layer::TraversabilityLayer, nav2_costmap_2d::Layer)

using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::FREE_SPACE;
using nav2_costmap_2d::NO_INFORMATION;

namespace traversability_layer
{

TraversabilityLayer::TraversabilityLayer()
: robot_height_(0.5),
  min_clearance_(0.1),
  check_radius_(0.25)
{
}

TraversabilityLayer::~TraversabilityLayer()
{
}

void TraversabilityLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("Unable to lock node!");
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("robot_height", rclcpp::ParameterValue(0.5));
  declareParameter("min_clearance", rclcpp::ParameterValue(0.1));
  declareParameter("check_radius", rclcpp::ParameterValue(0.25));
  declareParameter("cloud_topic", rclcpp::ParameterValue("/rtabmap/cloud_map"));

  node->get_parameter(name_ + "." + "enabled", enabled_);
  node->get_parameter(name_ + "." + "robot_height", robot_height_);
  node->get_parameter(name_ + "." + "min_clearance", min_clearance_);
  node->get_parameter(name_ + "." + "check_radius", check_radius_);
  node->get_parameter(name_ + "." + "cloud_topic", cloud_topic_);

  RCLCPP_INFO(
    node->get_logger(),
    "TraversabilityLayer: robot_height=%.2f, min_clearance=%.2f, cloud_topic=%s",
    robot_height_, min_clearance_, cloud_topic_.c_str());

  // 포인트클라우드 구독 - RELIABLE QoS로 변경!
  cloud_sub_ = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    cloud_topic_,
    rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::Reliable),
    std::bind(&TraversabilityLayer::pointCloudCallback, this, std::placeholders::_1));

  current_ = true;
}

void TraversabilityLayer::pointCloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  latest_cloud_ = msg;

  auto node = node_.lock();
  if (node) {
    RCLCPP_INFO_THROTTLE(
      node->get_logger(),
      *node->get_clock(),
      5000,  // 5초마다 출력
      "TraversabilityLayer: ✓ Received cloud with %d points",
      msg->width * msg->height);
  }
}

bool TraversabilityLayer::isTraversable(
  double x, double y,
  const sensor_msgs::msg::PointCloud2::SharedPtr& cloud)
{
  if (!cloud) {
    return false;
  }

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud, "z");

  int points_in_robot_zone = 0;  // ★ bool → int 변경
  int points_above = 0;           // ★ bool → int 변경
  int points_below = 0;           // ★ 바닥면 포인트

  int points_in_radius = 0;
  int points_checked = 0;
  const int max_points = 5000;

  double min_z = 999.0, max_z = -999.0;

  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    if (++points_checked > max_points) {
      break;
    }

    double dx = *iter_x - x;
    double dy = *iter_y - y;
    double dist = std::sqrt(dx * dx + dy * dy);

    if (dist > check_radius_) {
      continue;
    }

    points_in_radius++;
    double z = *iter_z;

    if (z < min_z) min_z = z;
    if (z > max_z) max_z = z;

    // ★ 포인트 개수로 카운트
    if (z < 0.15) {
      points_below++;  // 바닥면
    } else if (z >= 0.15 && z <= robot_height_) {
      points_in_robot_zone++;  // 로봇 충돌 영역
    } else if (z > (robot_height_ + min_clearance_) && z <= 2.0) {
      points_above++;  // 로봇 위쪽
    }
  }

  // // ★ 디버그 로그 (샘플링)
  // static int debug_counter = 0;
  // if (++debug_counter % 100 == 0 && points_in_radius > 0) {
  //   auto node = node_.lock();
  //   if (node) {
  //     RCLCPP_INFO(node->get_logger(),
  //       "isTraversable(%.2f, %.2f): total=%d, z=[%.2f,%.2f], below=%d, robot=%d, above=%d",
  //       x, y, points_in_radius, min_z, max_z, points_below, points_in_robot_zone, points_above);
  //   }
  //}

  // ★ 비율 기반 판단 로직:

  // 1. 포인트가 너무 적으면 판단 불가
  if (points_in_radius < 10) {
    return false;
  }

  // 2. 위쪽에 아무것도 없으면 → 벽 (LETHAL 유지)
  if (points_above == 0) {
    return false;
  }

  if (points_above > points_in_robot_zone * 1.5) {
    return true;  // 책상 아래 통과 가능!
  }

  // 4. 그 외 → LETHAL 유지
  return false;
}

void TraversabilityLayer::updateBounds(
  double robot_x, double robot_y, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }

  // ★ 로봇 주변 5m만 업데이트 (성능 향상)
  double update_range = 5.0;
  *min_x = std::min(*min_x, robot_x - update_range);
  *min_y = std::min(*min_y, robot_y - update_range);
  *max_x = std::max(*max_x, robot_x + update_range);
  *max_y = std::max(*max_y, robot_y + update_range);
}

void TraversabilityLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_ || !latest_cloud_) {
    return;
  }

  // 각 셀에 대해 통과 가능 여부 체크
  for (int j = min_j; j < max_j; j++) {
    for (int i = min_i; i < max_i; i++) {
      unsigned char current_cost = master_grid.getCost(i, j);  // ← 현재 값 읽기
      // ★ LETHAL 장애물만 체크 (성능 대폭 향상!)
      if (current_cost == LETHAL_OBSTACLE) {
        double wx, wy;
        master_grid.mapToWorld(i, j, wx, wy);

        if (isTraversable(wx, wy, latest_cloud_)) {
          // 책상 아래 → FREE로 변경
          master_grid.setCost(i, j, FREE_SPACE);
        }
        // else: 로봇 높이에 장애물 → LETHAL 유지
      }
      // FREE, NO_INFORMATION은 그대로 유지
    
    }
  }
}

void TraversabilityLayer::onFootprintChanged()
{
  // footprint 변경 시 처리 (필요시 구현)
}

}  // namespace traversability_layer
