#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

struct VoxelKey
{
  int x{0};
  int y{0};
  int z{0};

  bool operator==(const VoxelKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & k) const
  {
    const std::uint64_t h1 = static_cast<std::uint64_t>(k.x) * 73856093u;
    const std::uint64_t h2 = static_cast<std::uint64_t>(k.y) * 19349663u;
    const std::uint64_t h3 = static_cast<std::uint64_t>(k.z) * 83492791u;
    return static_cast<std::size_t>(h1 ^ h2 ^ h3);
  }
};

struct VoxelState
{
  int hits{0};
  double first_seen_sec{0.0};
  double last_seen_sec{0.0};
};

class DynamicObjectFilterNode : public rclcpp::Node
{
public:
  DynamicObjectFilterNode()
  : Node("dynamic_object_filter")
  {
    // 입력/출력 토픽
    declare_parameter<std::string>("input_topic", "/livox/lidar/filtered");
    declare_parameter<std::string>("output_topic", "/livox/lidar/static_filtered");
    // 보셀 크기: 작을수록 디테일↑, 잡음/연산량↑
    declare_parameter<double>("voxel_size", 0.10);
    // 정적으로 인정할 최소 관측 횟수(보셀 히트 수)
    declare_parameter<int>("min_hits", 3);
    // 히트 누적 시간 창: 이 시간 안에 반복 관측되어야 hits가 누적됨
    declare_parameter<double>("hit_window_sec", 3.0);
    // 오래 안 보인 보셀 상태는 제거(메모리/유령 제거)
    declare_parameter<double>("max_stale_sec", 8.0);
    // 정적으로 인정하기 위한 최소 유지 시간(연속 관측 유지 시간)
    declare_parameter<double>("min_static_sec", 2.0);
    // 높이 필터(지면/천장/노이즈 제거)
    declare_parameter<double>("z_min", -2.0);
    declare_parameter<double>("z_max", 2.0);
    // 센서 근접 링 제거(로봇 주변 원형 잔상 방지)
    declare_parameter<double>("min_range", 0.2);
    // 누적 기준 프레임(회전 시 도넛/휘어짐 방지용)
    declare_parameter<std::string>("target_frame", "odom");
    // TF lookup 타임아웃(지연 시 드롭)
    declare_parameter<double>("tf_timeout_sec", 0.05);

    input_topic_ = get_parameter("input_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    voxel_size_ = get_parameter("voxel_size").as_double();
    min_hits_ = get_parameter("min_hits").as_int();
    hit_window_sec_ = get_parameter("hit_window_sec").as_double();
    max_stale_sec_ = get_parameter("max_stale_sec").as_double();
    min_static_sec_ = get_parameter("min_static_sec").as_double();
    z_min_ = get_parameter("z_min").as_double();
    z_max_ = get_parameter("z_max").as_double();
    min_range_ = get_parameter("min_range").as_double();
    min_range_sq_ = min_range_ * min_range_;
    target_frame_ = get_parameter("target_frame").as_string();
    tf_timeout_sec_ = get_parameter("tf_timeout_sec").as_double();

    // 고정 프레임(odom/map) 기준으로 누적해야 회전 시 도넛 아티팩트가 줄어든다.
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&DynamicObjectFilterNode::callback, this, std::placeholders::_1));
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::SensorDataQoS());

    RCLCPP_INFO(
      get_logger(),
      "Dynamic filter: %s -> %s, voxel=%.3f, min_hits=%d, hit_window=%.2fs, min_static=%.2fs",
      input_topic_.c_str(), output_topic_.c_str(), voxel_size_, min_hits_, hit_window_sec_, min_static_sec_);
  }

private:
  VoxelKey to_voxel(const pcl::PointXYZ & p) const
  {
    return VoxelKey{
      static_cast<int>(std::floor(p.x / voxel_size_)),
      static_cast<int>(std::floor(p.y / voxel_size_)),
      static_cast<int>(std::floor(p.z / voxel_size_))
    };
  }

  void cleanup_stale(double now_sec)
  {
    std::vector<VoxelKey> to_erase;
    to_erase.reserve(voxel_map_.size());
    for (const auto & kv : voxel_map_) {
      if (now_sec - kv.second.last_seen_sec > max_stale_sec_) {
        to_erase.push_back(kv.first);
      }
    }
    for (const auto & k : to_erase) {
      voxel_map_.erase(k);
    }
  }

  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    sensor_msgs::msg::PointCloud2 cloud_in_frame;
    if (msg->header.frame_id != target_frame_) {
      try {
        auto tf = tf_buffer_->lookupTransform(
          target_frame_, msg->header.frame_id, msg->header.stamp,
          rclcpp::Duration::from_seconds(tf_timeout_sec_));
        tf2::doTransform(*msg, cloud_in_frame, tf);
      } catch (const std::exception & e) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "TF transform failed (%s -> %s): %s",
          msg->header.frame_id.c_str(), target_frame_.c_str(), e.what());
        return;
      }
    } else {
      cloud_in_frame = *msg;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(cloud_in_frame, *in);
    if (in->empty()) {
      pub_->publish(cloud_in_frame);
      return;
    }

    const double now_sec = this->get_clock()->now().seconds();
    cleanup_stale(now_sec);

    std::unordered_set<VoxelKey, VoxelKeyHash> frame_voxels;
    frame_voxels.reserve(in->size());

    for (const auto & p : in->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        continue;
      }
      if (p.z < z_min_ || p.z > z_max_) {
        continue;
      }
      if ((p.x * p.x + p.y * p.y) < min_range_sq_) {
        continue;
      }
      frame_voxels.insert(to_voxel(p));
    }

    // 이번 프레임에서 관측된 보셀에 대해 히트 누적
    for (const auto & v : frame_voxels) {
      auto it = voxel_map_.find(v);
      if (it == voxel_map_.end()) {
        voxel_map_[v] = VoxelState{1, now_sec, now_sec};
      } else {
        if (now_sec - it->second.last_seen_sec > hit_window_sec_) {
          it->second.hits = 1;
          it->second.first_seen_sec = now_sec;
        } else {
          it->second.hits += 1;
        }
        it->second.last_seen_sec = now_sec;
      }
    }

    std::unordered_set<VoxelKey, VoxelKeyHash> static_voxels;
    static_voxels.reserve(frame_voxels.size());
    // 정적 판정: hits + 연속 유지시간(min_static_sec) 조건 모두 만족
    for (const auto & v : frame_voxels) {
      auto it = voxel_map_.find(v);
      if (it != voxel_map_.end() &&
          it->second.hits >= min_hits_ &&
          (now_sec - it->second.first_seen_sec) >= min_static_sec_) {
        static_voxels.insert(v);
      }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    out->header = in->header;
    out->points.reserve(in->size());
    for (const auto & p : in->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        continue;
      }
      if (static_voxels.find(to_voxel(p)) != static_voxels.end()) {
        out->points.push_back(p);
      }
    }
    out->width = static_cast<std::uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = false;

    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*out, out_msg);
    out_msg.header = cloud_in_frame.header;
    out_msg.header.frame_id = target_frame_;
    pub_->publish(out_msg);
  }

  std::string input_topic_;
  std::string output_topic_;
  double voxel_size_{0.10};
  int min_hits_{3};
  double hit_window_sec_{3.0};
  double max_stale_sec_{8.0};
  double min_static_sec_{2.0};
  double z_min_{-2.0};
  double z_max_{2.0};
  double min_range_{0.2};
  double min_range_sq_{0.04};
  std::string target_frame_{"odom"};
  double tf_timeout_sec_{0.05};

  std::unordered_map<VoxelKey, VoxelState, VoxelKeyHash> voxel_map_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamicObjectFilterNode>());
  rclcpp::shutdown();
  return 0;
}
