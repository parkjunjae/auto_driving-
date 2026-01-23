cmake_minimum_required(VERSION 3.8)
project(temp_goal_bt)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_library(temp_goal_bt_node SHARED
  src/pick_temporary_goal.cpp
)

target_include_directories(temp_goal_bt_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(temp_goal_bt_node
  rclcpp
  behaviortree_cpp_v3
  nav2_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
)

target_compile_definitions(temp_goal_bt_node PRIVATE "BT_PLUGIN_EXPORT")

install(TARGETS temp_goal_bt_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
