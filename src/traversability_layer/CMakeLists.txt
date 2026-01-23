cmake_minimum_required(VERSION 3.8)
project(traversability_layer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Include directories
include_directories(
  include
)

set(dependencies
  rclcpp
  nav2_costmap_2d
  pluginlib
  tf2_ros
  sensor_msgs
  geometry_msgs
)

# Library
add_library(${PROJECT_NAME} SHARED
  src/traversability_layer.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  ${dependencies}
)

# Install
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

install(FILES traversability_layer_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

# Export
pluginlib_export_plugin_description_file(nav2_costmap_2d traversability_layer_plugin.xml)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})

ament_package()