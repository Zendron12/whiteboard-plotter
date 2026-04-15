cmake_minimum_required(VERSION 3.8)
project(wall_climber_draw_body)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(wall_climber_interfaces REQUIRED)

add_executable(cable_draw_executor src/cable_draw_executor.cpp)
target_compile_features(cable_draw_executor PUBLIC cxx_std_17)
ament_target_dependencies(cable_draw_executor
  geometry_msgs
  rclcpp
  std_msgs
  wall_climber_interfaces
)

install(TARGETS cable_draw_executor
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
