#ifndef TEMP_GOAL_BT__PICK_TEMPORARY_GOAL_HPP_
#define TEMP_GOAL_BT__PICK_TEMPORARY_GOAL_HPP_

#include <memory>
#include <mutex>
#include <string>

#include "behaviortree_cpp_v3/action_node.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/msg/costmap.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"

namespace temp_goal_bt
{

class PickTemporaryGoal : public BT::SyncActionNode
{
public:
  PickTemporaryGoal(const std::string & name, const BT::NodeConfiguration & config);

  static BT::PortsList providedPorts();

  BT::NodeStatus tick() override;

private:
  void costmapCallback(const nav2_msgs::msg::Costmap::SharedPtr msg);

  rclcpp::Node::SharedPtr node_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::executors::SingleThreadedExecutor callback_group_executor_;
  rclcpp::Subscription<nav2_msgs::msg::Costmap>::SharedPtr costmap_sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  std::mutex costmap_mutex_;
  nav2_msgs::msg::Costmap::SharedPtr latest_costmap_;
  std::string costmap_topic_;
};

}  // namespace temp_goal_bt

#endif  // TEMP_GOAL_BT__PICK_TEMPORARY_GOAL_HPP_
