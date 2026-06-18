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

# ── ONNX Runtime ────────────────────────────────────────────────────────
# Find onnxruntime. On Ubuntu with pip-installed onnxruntime:
#   find_library(ONNX_RUNTIME_LIB onnxruntime)
# Or use the system path if installed via apt.
find_library(ONNX_RUNTIME_LIB
  NAMES onnxruntime
  PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)
if(ONNX_RUNTIME_LIB)
  message(STATUS "✅ Found onnxruntime: ${ONNX_RUNTIME_LIB}")
  # Find onnxruntime include path (headers are typically in /usr/local/include)
  find_path(ONNX_RUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS /usr/local/include /usr/include /usr/lib/x86_64-linux-gnu
  )
  if(ONNX_RUNTIME_INCLUDE_DIR)
    message(STATUS "✅ Found onnxruntime headers: ${ONNX_RUNTIME_INCLUDE_DIR}")
    include_directories(SYSTEM ${ONNX_RUNTIME_INCLUDE_DIR})
  else()
    message(WARNING "⚠️  onnxruntime headers not found — neural guidance will be disabled")
  endif()
else()
  message(WARNING "⚠️  onnxruntime not found — neural guidance will be disabled")
endif()

include_directories(SYSTEM
  include
  ${Eigen3_INCLUDE_DIRS}
)

set(project_LIBRARIES
  ${rclcpp_LIBRARIES}
  ${ONNX_RUNTIME_LIB}
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
  src/sensor_sync.cpp
  src/commander.cpp
  src/gps_parser.cpp
  src/synthetic_data_generator.cpp
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
