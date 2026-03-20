cmake_minimum_required(VERSION 3.8)
project(los_guidance)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ai_service REQUIRED)
find_package(std_srvs REQUIRED)
# find_package(message_filters REQUIRED)


include_directories(
  include
  ${Eigen3_INCLUDE_DIRS}
)

set(project_LIBRARIES
  ${rclcpp_LIBRARIES}
  #other_libs
)

set(project_DEPENDENCIES
  rclcpp
  eigen3_cmake_module
  Eigen3
  px4_msgs
  vision_msgs
  std_msgs
  sensor_msgs
  ai_service
  std_srvs
  # message_filters
)

set(project_SOURCE_FILES
  src/node.cpp
  src/target_tracker.cpp
  src/guidance_controller.cpp
  src/los_estimator.cpp
  src/vehicle_controller.cpp    
  # src/sensor_sync.cpp
)

# Executable target
set(executable_name ${PROJECT_NAME}_node)
add_executable(${executable_name} src/main.cpp ${project_SOURCE_FILES})
target_link_libraries(${executable_name} ${project_LIBRARIES} "${cpp_typesupport_target}")
ament_target_dependencies(${executable_name} ${project_DEPENDENCIES})
install(TARGETS
  ${executable_name}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}/
)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
