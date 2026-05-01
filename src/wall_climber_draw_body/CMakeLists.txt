cmake_minimum_required(VERSION 3.8)
project(wall_climber_draw_body)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest QUIET)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(wall_climber_interfaces REQUIRED)

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
_ament_cmake_python_register_environment_hook()

add_library(wall_climber_geometry
  src/geometry_eval.cpp
  src/geometry_sampling.cpp
)
target_compile_features(wall_climber_geometry PUBLIC cxx_std_17)
target_include_directories(wall_climber_geometry
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_library(wall_climber_transport
  src/transport_conversions.cpp
)
target_compile_features(wall_climber_transport PUBLIC cxx_std_17)
target_include_directories(wall_climber_transport
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(wall_climber_transport wall_climber_geometry)
ament_target_dependencies(wall_climber_transport
  geometry_msgs
  wall_climber_interfaces
)

add_library(wall_climber_geometry_cpp MODULE src/python_bindings.cpp)
target_compile_features(wall_climber_geometry_cpp PUBLIC cxx_std_17)
target_include_directories(wall_climber_geometry_cpp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(wall_climber_geometry_cpp
  PRIVATE
    Python3::Module
    wall_climber_geometry
)
set_target_properties(wall_climber_geometry_cpp PROPERTIES PREFIX "")

add_executable(cable_draw_executor src/cable_draw_executor.cpp)
target_compile_features(cable_draw_executor PUBLIC cxx_std_17)
target_include_directories(cable_draw_executor
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cable_draw_executor wall_climber_geometry wall_climber_transport)
ament_target_dependencies(cable_draw_executor
  geometry_msgs
  rclcpp
  std_msgs
  wall_climber_interfaces
)

install(TARGETS
  cable_draw_executor
  wall_climber_geometry
  wall_climber_transport
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS wall_climber_geometry_cpp
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
)
install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING AND ament_cmake_gtest_FOUND)
  ament_add_gtest(test_geometry_kernel test/test_geometry_kernel.cpp)
  if(TARGET test_geometry_kernel)
    target_link_libraries(test_geometry_kernel wall_climber_geometry)
    target_include_directories(test_geometry_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
  ament_add_gtest(test_transport_conversions test/test_transport_conversions.cpp)
  if(TARGET test_transport_conversions)
    target_link_libraries(test_transport_conversions wall_climber_transport)
    ament_target_dependencies(test_transport_conversions geometry_msgs wall_climber_interfaces)
    target_include_directories(test_transport_conversions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
endif()

ament_package()
