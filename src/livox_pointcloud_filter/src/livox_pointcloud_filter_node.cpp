#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/filters/radius_outlier_removal.h"
#include "pcl/point_types.h"

class LivoxPointCloudFilter : public rclcpp::Node
{
public:
  LivoxPointCloudFilter()
  : Node("livox_pointcloud_filter")
  {
    // 파라미터 선언 및 기본값 설정
    // 입력 포인트클라우드 토픽
    declare_parameter<std::string>("input_topic", "/livox/lidar/deskewed");
    // 출력 포인트클라우드 토픽
    declare_parameter<std::string>("output_topic", "/livox/lidar/filtered");
    // 다운샘플 해상도(m)
    declare_parameter<double>("leaf_size", 0.05);
    // ROR 반경(m)
    declare_parameter<double>("ror_radius", 0.10);
    // ROR 최소 이웃 수
    declare_parameter<int>("ror_min_neighbors", 4);
    // 다운샘플 사용 여부
    declare_parameter<bool>("use_voxel", true);
    // ROR 사용 여부
    declare_parameter<bool>("use_ror", true);
    // 출력 프레임 고정(비어 있으면 입력 프레임 유지)
    declare_parameter<std::string>("frame_id", "");

    input_topic_ = get_parameter("input_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    leaf_size_ = get_parameter("leaf_size").as_double();
    ror_radius_ = get_parameter("ror_radius").as_double();
    ror_min_neighbors_ = get_parameter("ror_min_neighbors").as_int();
    use_voxel_ = get_parameter("use_voxel").as_bool();
    use_ror_ = get_parameter("use_ror").as_bool();
    frame_id_ = get_parameter("frame_id").as_string();

    // 센서 QoS로 구독(실시간 센서 메시지에 적합)
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&LivoxPointCloudFilter::callback, this, std::placeholders::_1));

    // 필터 결과를 퍼블리셔로 송출
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::SensorDataQoS());
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // 입력을 PCL로 변환
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud_in);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);

    // 1) 다운샘플링(VoxelGrid)으로 포인트 수를 줄여 연산량을 감소
    if (use_voxel_ && leaf_size_ > 1e-6) {
      pcl::VoxelGrid<pcl::PointXYZI> voxel;
      voxel.setInputCloud(cloud_in);
      voxel.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
      voxel.filter(*cloud_tmp);
    } else {
      *cloud_tmp = *cloud_in;
    }

    // 2) ROR 필터링으로 주변 이웃이 적은 노이즈를 제거
    if (use_ror_ && ror_radius_ > 1e-6 && ror_min_neighbors_ > 0) {
      pcl::RadiusOutlierRemoval<pcl::PointXYZI> ror;
      ror.setInputCloud(cloud_tmp);
      ror.setRadiusSearch(ror_radius_);
      ror.setMinNeighborsInRadius(ror_min_neighbors_);
      ror.filter(*cloud_filtered);
    } else {
      *cloud_filtered = *cloud_tmp;
    }

    // 출력 PCL을 ROS 메시지로 변환
    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*cloud_filtered, out_msg);
    out_msg.header.stamp = msg->header.stamp;
    out_msg.header.frame_id = frame_id_.empty() ? msg->header.frame_id : frame_id_;

    // 최종 메시지 퍼블리시
    pub_->publish(out_msg);
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string frame_id_;
  double leaf_size_{0.05};
  double ror_radius_{0.10};
  int ror_min_neighbors_{4};
  bool use_voxel_{true};
  bool use_ror_{true};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  // 노드 실행
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LivoxPointCloudFilter>());
  rclcpp::shutdown();
  return 0;
}
