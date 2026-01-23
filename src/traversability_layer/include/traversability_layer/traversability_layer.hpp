#ifndef TRAVERSABILITY_LAYER_HPP_
#define TRAVERSABILITY_LAYER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav2_costmap_2d/costmap_layer.hpp"
#include "nav2_costmap_2d/voxel_layer.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace traversability_layer
{

class TraversabilityLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  TraversabilityLayer();
  virtual ~TraversabilityLayer();

  virtual void onInitialize();
  virtual void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y);
  virtual void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j);

  virtual void reset()
  {
    return;
  }

  virtual void onFootprintChanged();

  virtual bool isClearable() {return false;}

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  bool isTraversable(double x, double y, const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  sensor_msgs::msg::PointCloud2::SharedPtr latest_cloud_;
  
  // 파라미터
  double robot_height_;      // 로봇 전체 높이 (예: 0.3m)
  double min_clearance_;     // 최소 통과 여유 (예: 0.05m)
  double check_radius_;      // 체크 반경 (예: 0.3m)
  std::string cloud_topic_;  // 포인트클라우드 토픽
};

}  // namespace traversability_layer

#endif  // TRAVERSABILITY_LAYER_HPP_