cmake_minimum_required(VERSION 3.8)
project(camera_imu_pipeline_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(camera_imu_bias_corrector_cpp src/camera_imu_bias_corrector.cpp)
ament_target_dependencies(camera_imu_bias_corrector_cpp rclcpp sensor_msgs tf2 tf2_ros)

install(TARGETS camera_imu_bias_corrector_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
