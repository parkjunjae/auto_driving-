cmake_minimum_required(VERSION 3.8)
project(livox_pointcloud_filter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS filters)

add_executable(livox_pointcloud_filter_node
  src/livox_pointcloud_filter_node.cpp
)

target_include_directories(livox_pointcloud_filter_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(livox_pointcloud_filter_node
  ${PCL_LIBRARIES}
)

ament_target_dependencies(livox_pointcloud_filter_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
)

install(TARGETS livox_pointcloud_filter_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
