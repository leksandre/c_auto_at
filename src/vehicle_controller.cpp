/**
 * @file vehicle_controller.cpp
 * @brief Реализация класса VehicleController
 * @note Вся логика управления PX4 вынесена из node.cpp для модульности
 *
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#include "los_guidance/vehicle_controller.hpp"
#include <cmath>
#include <algorithm>

namespace los_guidance
{

  // ============================================================================
  // Конструктор
  // ============================================================================

  VehicleController::VehicleController(
      rclcpp::Node::SharedPtr node,
      const VehicleControllerConfig &config)
      : node_(node), config_(config)
  {
    if (!node_)
    {
      throw std::runtime_error("VehicleController: node pointer is null");
    }

    // Создание publishers
    createPublishers_();

    // Инициализация состояния
    reset();

    if (config_.debug_logging)
    {
      RCLCPP_INFO(node_->get_logger(), "VehicleController initialized");
    }
  }

  // ============================================================================
  // Публичные методы
  // ============================================================================

  void VehicleController::updateStatus(
      const px4_msgs::msg::VehicleStatus::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!msg)
    {
      return;
    }

    status_.nav_state = msg->nav_state;
    status_.flight_mode = navStateToFlightMode_(msg->nav_state);
    status_.is_armed = (msg->arming_state == px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED);
    status_.last_status_timestamp_us = msg->timestamp;
    status_.is_valid = true;

    // Offboard доступен только в режиме OFFBOARD
    status_.is_offboard_available =
        (status_.flight_mode == FlightMode::OFFBOARD) && status_.is_armed;

    if (config_.debug_logging)
    {
      RCLCPP_DEBUG(
          node_->get_logger(),
          "Vehicle status: mode=%s, armed=%s, offboard=%s",
          flightModeToString(status_.flight_mode).c_str(),
          status_.is_armed ? "true" : "false",
          status_.is_offboard_available ? "true" : "false");
    }
  }

  VehicleStatusInfo VehicleController::getStatus() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
  }

  FlightMode VehicleController::getCurrentMode() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_.flight_mode;
  }

  bool VehicleController::isReadyForAttack() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_.is_valid && status_.is_armed;
  }

  bool VehicleController::isOffboardAvailable() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_.is_offboard_available;
  }

  bool VehicleController::isArmed() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_.is_armed;
  }

  bool VehicleController::setOffboardMode()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (status_.flight_mode == FlightMode::OFFBOARD)
    {
      if (config_.debug_logging)
      {
        RCLCPP_DEBUG(node_->get_logger(), "Already in OFFBOARD mode");
      }
      return true;
    }

    bool success = publishVehicleCommand_(
        VEHICLE_CMD_DO_SET_MODE,
        1.0, // Base mode
        static_cast<float>(PX4_CUSTOM_MAIN_MODE_OFFBOARD));

    if (success)
    {
      offboard_mode_sent_ = true;
      last_command_timestamp_us_ = getCurrentTimestamp_();

      if (config_.debug_logging)
      {
        RCLCPP_INFO(node_->get_logger(), "OFFBOARD mode command sent");
      }
    }

    return success;
  }

  bool VehicleController::setManualMode()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (status_.flight_mode == FlightMode::MANUAL)
    {
      return true;
    }

    bool success = publishVehicleCommand_(
        VEHICLE_CMD_DO_SET_MODE,
        1.0,
        static_cast<float>(PX4_CUSTOM_MAIN_MODE_MANUAL));

    if (success && config_.debug_logging)
    {
      RCLCPP_INFO(node_->get_logger(), "MANUAL mode command sent");
    }

    return success;
  }

  bool VehicleController::setAltitudeMode()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (status_.flight_mode == FlightMode::ALTCTL)
    {
      return true;
    }

    bool success = publishVehicleCommand_(
        VEHICLE_CMD_DO_SET_MODE,
        1.0,
        static_cast<float>(PX4_CUSTOM_MAIN_MODE_ALTCTL));

    if (success && config_.debug_logging)
    {
      RCLCPP_INFO(node_->get_logger(), "ALTCTL mode command sent");
    }

    return success;
  }

  bool VehicleController::setPositionControlMode()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (status_.flight_mode == FlightMode::POSCTL)
    {
      return true;
    }

    bool success = publishVehicleCommand_(
        VEHICLE_CMD_DO_SET_MODE,
        1.0,
        static_cast<float>(PX4_CUSTOM_MAIN_MODE_POSCTL));

    if (success && config_.debug_logging)
    {
      RCLCPP_INFO(node_->get_logger(), "POSCTL mode command sent");
    }

    return success;
  }

  bool VehicleController::publishAccelerationSetpoint(
      const Eigen::Vector3d &acc_setpoint)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!status_.is_valid)
    {
      RCLCPP_WARN(node_->get_logger(), "Cannot publish: status not valid");
      return false;
    }

    // Сглаживание ускорения
    Eigen::Vector3d acc_smooth = smoothAcceleration_(acc_setpoint);

    // Ограничение ускорения
    acc_smooth = clampAcceleration_(acc_smooth);

    // Публикация offboard control mode (должна публиковаться регулярно)
    // publishOffboardControlMode();

    // Публикация trajectory setpoint
    px4_msgs::msg::TrajectorySetpoint msg;
    msg.position = {NAN, NAN, NAN};
    msg.velocity = {NAN, NAN, NAN};
    msg.acceleration = {
        static_cast<float>(acc_smooth.x()),
        static_cast<float>(acc_smooth.y()),
        static_cast<float>(acc_smooth.z())};
    msg.yaw = NAN;
    msg.timestamp = getCurrentTimestamp_();

    traj_pub_->publish(msg);

    // Сохранение последней уставки
    last_setpoint_.acceleration = acc_smooth;
    last_setpoint_.timestamp_us = msg.timestamp;
    last_setpoint_.is_valid = true;

    if (config_.debug_logging)
    {
      RCLCPP_DEBUG(
          node_->get_logger(),
          "Acc setpoint published: [%.3f, %.3f, %.3f]",
          acc_smooth.x(), acc_smooth.y(), acc_smooth.z());
    }

    return true;
  }

  bool VehicleController::publishVelocitySetpoint(
      const Eigen::Vector3d &vel_setpoint)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!status_.is_valid)
    {
      return false;
    }

    // publishOffboardControlMode();

    px4_msgs::msg::TrajectorySetpoint msg;
    msg.position = {NAN, NAN, NAN};
    msg.velocity = {
        static_cast<float>(vel_setpoint.x()),
        static_cast<float>(vel_setpoint.y()),
        static_cast<float>(vel_setpoint.z())};
    msg.acceleration = {NAN, NAN, NAN};
    msg.yaw = NAN;
    msg.timestamp = getCurrentTimestamp_();

    traj_pub_->publish(msg);

    last_setpoint_.velocity = vel_setpoint;
    last_setpoint_.timestamp_us = msg.timestamp;
    last_setpoint_.is_valid = true;

    return true;
  }

  // bool VehicleController::publishPositionSetpoint(
  //   const Eigen::Vector3d& pos_setpoint
  // ) {
  //   std::lock_guard<std::mutex> lock(mutex_);

  //   if (!status_.is_valid) {
  //     return false;
  //   }

  //   // publishOffboardControlMode();

  //   px4_msgs::msg::TrajectorySetpoint msg;
  //   msg.position = {
  //     static_cast<float>(pos_setpoint.x()),
  //     static_cast<float>(pos_setpoint.y()),
  //     static_cast<float>(pos_setpoint.z())
  //   };
  //   msg.velocity = {NAN, NAN, NAN};
  //   msg.acceleration = {NAN, NAN, NAN};
  //   msg.yaw = NAN;
  //   msg.timestamp = getCurrentTimestamp_();

  //   traj_pub_->publish(msg);

  //   last_setpoint_.position = pos_setpoint;
  //   last_setpoint_.timestamp_us = msg.timestamp;
  //   last_setpoint_.is_valid = true;

  //   return true;
  // }

  /**
   * @brief Публикует setpoint позиции с опциональной скоростью и ускорением
   * @param pos_setpoint Желаемая позиция [м] в локальной системе координат (NED)
   * @param velocity_opt Опциональная желаемая скорость [м/с] (по умолчанию = NaN → игнорируется контроллером)
   * @param acceleration_opt Опциональное желаемое ускорение [м/с²] (по умолчанию = NaN → игнорируется)
   * @return true если сообщение успешно опубликовано
   */
  bool VehicleController::publishPositionSetpoint(
      const Eigen::Vector3d &pos_setpoint,
      const Eigen::Vector3d &velocity,
      const Eigen::Vector3d &acceleration)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!status_.is_valid)
    {
      return false;
    }

    px4_msgs::msg::TrajectorySetpoint msg;

    // === 1. Позиция (всегда валидна) ===
    msg.position[0] = static_cast<float>(pos_setpoint.x());
    msg.position[1] = static_cast<float>(pos_setpoint.y());
    msg.position[2] = static_cast<float>(pos_setpoint.z());

    // === 2. Скорость (проверяем на NaN) ===
    // Если хоть один компонент = NaN, считаем весь вектор "не заданным"
    if (std::isnan(velocity.x()) || std::isnan(velocity.y()) || std::isnan(velocity.z()))
    {
      msg.velocity[0] = NAN;
      msg.velocity[1] = NAN;
      msg.velocity[2] = NAN;
    }
    else
    {
      //  Правильное присваивание в std::array для C++11/14/17
      msg.velocity[0] = static_cast<float>(velocity.x());
      msg.velocity[1] = static_cast<float>(velocity.y());
      msg.velocity[2] = static_cast<float>(velocity.z());
    }

    // === 3. Ускорение (аналогично) ===
    if (std::isnan(acceleration.x()) || std::isnan(acceleration.y()) || std::isnan(acceleration.z()))
    {
      msg.acceleration[0] = NAN;
      msg.acceleration[1] = NAN;
      msg.acceleration[2] = NAN;
    }
    else
    {
      msg.acceleration[0] = static_cast<float>(acceleration.x());
      msg.acceleration[1] = static_cast<float>(acceleration.y());
      msg.acceleration[2] = static_cast<float>(acceleration.z());
    }

    // === 4. Yaw и время ===
    msg.yaw = NAN;
    msg.timestamp = getCurrentTimestamp_();

    traj_pub_->publish(msg);

    // === Сохраняем стейт ===
    last_setpoint_.position = pos_setpoint;
    last_setpoint_.timestamp_us = msg.timestamp;
    last_setpoint_.is_valid = true;

    return true;
  }

  bool VehicleController::publishOffboardControlMode()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    px4_msgs::msg::OffboardControlMode msg;
    msg.position = config_.enable_position_control;
    msg.velocity = config_.enable_velocity_control;
    msg.acceleration = config_.enable_acceleration_control;
    msg.attitude = false;
    msg.body_rate = false;
    msg.timestamp = getCurrentTimestamp_();

    offboard_pub_->publish(msg);
    offboard_mode_sent_ = true;

    return true;
  }

  bool VehicleController::publishTrajectorySetpoint(
      const ControlSetpoint &setpoint)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!setpoint.is_valid)
    {
      return false;
    }

    // publishOffboardControlMode();

    px4_msgs::msg::TrajectorySetpoint msg;
    msg.position = {
        static_cast<float>(setpoint.position.x()),
        static_cast<float>(setpoint.position.y()),
        static_cast<float>(setpoint.position.z())};
    msg.velocity = {
        static_cast<float>(setpoint.velocity.x()),
        static_cast<float>(setpoint.velocity.y()),
        static_cast<float>(setpoint.velocity.z())};
    msg.acceleration = {
        static_cast<float>(setpoint.acceleration.x()),
        static_cast<float>(setpoint.acceleration.y()),
        static_cast<float>(setpoint.acceleration.z())};
    msg.yaw = std::isnan(setpoint.yaw) ? NAN : static_cast<float>(setpoint.yaw);
    msg.timestamp = getCurrentTimestamp_();

    traj_pub_->publish(msg);

    last_setpoint_ = setpoint;

    return true;
  }

  void VehicleController::reset()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    status_ = VehicleStatusInfo();
    last_setpoint_ = ControlSetpoint();
    offboard_mode_sent_ = false;
    last_command_timestamp_us_ = 0;
    command_retry_counter_ = 0;
    acc_smoothed_.setZero();
    acc_initialized_ = false;

    if (config_.debug_logging)
    {
      RCLCPP_INFO(node_->get_logger(), "VehicleController reset");
    }
  }

  void VehicleController::updateConfig(const VehicleControllerConfig &config)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
  }

  ControlSetpoint VehicleController::getLastSetpoint() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_setpoint_;
  }

  void VehicleController::setDebugLogging(bool enable)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.debug_logging = enable;
  }

  std::string VehicleController::flightModeToString(FlightMode mode)
  {
    switch (mode)
    {
    case FlightMode::UNKNOWN:
      return "UNKNOWN";
    case FlightMode::OFFBOARD:
      return "OFFBOARD";
    case FlightMode::MANUAL:
      return "MANUAL";
    case FlightMode::ALTCTL:
      return "ALTCTL";
    case FlightMode::POSCTL:
      return "POSCTL";
    case FlightMode::AUTO:
      return "AUTO";
    case FlightMode::ACRO:
      return "ACRO";
    case FlightMode::STABILIZED:
      return "STABILIZED";
    case FlightMode::OTHERS:
      return "OTHERS";
    default:
      return "INVALID";
    }
  }

  // ============================================================================
  // Приватные методы
  // ============================================================================

  void VehicleController::createPublishers_()
  {
    cmd_pub_ = node_->create_publisher<px4_msgs::msg::VehicleCommand>(
        "fmu/in/vehicle_command", 10);

    offboard_pub_ = node_->create_publisher<px4_msgs::msg::OffboardControlMode>(
        "fmu/in/offboard_control_mode", 10);

    traj_pub_ = node_->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
        "fmu/in/trajectory_setpoint", 10);

    if (config_.debug_logging)
    {
      RCLCPP_INFO(node_->get_logger(), "VehicleController publishers created");
    }
  }

  bool VehicleController::publishVehicleCommand_(
      uint8_t command,
      float param1,
      float param2)
  {
    px4_msgs::msg::VehicleCommand msg;
    msg.param1 = param1;
    msg.param2 = param2;
    msg.param3 = 0.0;
    msg.param4 = 0.0;
    msg.param5 = 0.0;
    msg.param6 = 0.0;
    msg.param7 = 0.0;
    msg.command = command;
    msg.target_system = config_.target_system;
    msg.target_component = config_.target_component;
    msg.source_system = config_.source_system;
    msg.source_component = config_.source_component;
    msg.from_external = true;
    msg.timestamp = getCurrentTimestamp_();

    cmd_pub_->publish(msg);

    return true;
  }

  // Eigen::Vector3d VehicleController::smoothAcceleration_(
  //   const Eigen::Vector3d& acc_raw
  // ) {
  //   if (!acc_initialized_) {
  //     acc_smoothed_ = acc_raw;
  //     acc_initialized_ = true;
  //     return acc_raw;
  //   }

  //   // Exponential moving average
  //   acc_smoothed_ = acc_smoothed_ +
  //     config_.acc_smooth_alpha * (acc_raw - acc_smoothed_);

  //   return acc_smoothed_;
  // }

  // Eigen::Vector3d VehicleController::clampAcceleration_(
  //   const Eigen::Vector3d& acc
  // ) {
  //   Eigen::Vector3d acc_clamped;

  //   // Горизонтальные оси (X, Y)
  //   acc_clamped.x() = std::clamp(
  //     acc.x(),
  //     -config_.max_hor_acc_limit,
  //     config_.max_hor_acc_limit
  //   );
  //   acc_clamped.y() = std::clamp(
  //     acc.y(),
  //     -config_.max_hor_acc_limit,
  //     config_.max_hor_acc_limit
  //   );

  //   // Вертикальная ось (Z) - NED frame
  //   acc_clamped.z() = std::clamp(
  //     acc.z(),
  //     config_.max_up_acc,
  //     config_.max_down_acc
  //   );

  //   return acc_clamped;
  // }

  Eigen::Vector3d VehicleController::smoothAcceleration_(
      const Eigen::Vector3d &acc_raw)
  {
    if (!acc_initialized_)
    {
      acc_smoothed_ = acc_raw;
      acc_initialized_ = true;
      return acc_raw;
    }

    // 🆕 Адаптивное сглаживание для тяжёлого дрона (2519г)
    double acc_change = (acc_raw - acc_smoothed_).norm();
    double adaptive_alpha = config_.acc_smooth_alpha;

    // Более консервативно для тяжёлого
    if (acc_change > 1.0)
    {                        // ↓ Меньше порог
      adaptive_alpha = 0.15; // ↓ Меньше сглаживание при резких изменениях
    }
    else if (acc_change < 0.2)
    {
      adaptive_alpha = 0.04; // ↓ Больше сглаживание при плавных
    }

    acc_smoothed_ = acc_smoothed_ + adaptive_alpha * (acc_raw - acc_smoothed_);
    return acc_smoothed_;
  }

  Eigen::Vector3d VehicleController::clampAcceleration_(
      const Eigen::Vector3d &acc)
  {
    Eigen::Vector3d acc_clamped;

    const double hf = config_.heavy_drone_factor;

    acc_clamped.x() = std::clamp(
        acc.x(),
        -config_.max_hor_acc_limit * hf,
        config_.max_hor_acc_limit * hf);
    acc_clamped.y() = std::clamp(
        acc.y(),
        -config_.max_hor_acc_limit * hf,
        config_.max_hor_acc_limit * hf);
    acc_clamped.z() = std::clamp(
        acc.z(),
        config_.max_up_acc * hf,
        config_.max_down_acc * hf);

    return acc_clamped;
  }

  uint64_t VehicleController::getCurrentTimestamp_() const
  {
    auto now = node_->get_clock()->now();
    return static_cast<uint64_t>(now.nanoseconds() / 1000); // ns → us
  }

  FlightMode VehicleController::navStateToFlightMode_(uint8_t nav_state) const
  {
    using NavState = px4_msgs::msg::VehicleStatus;

    switch (nav_state)
    {
    case NavState::NAVIGATION_STATE_OFFBOARD:
      return FlightMode::OFFBOARD;

    case NavState::NAVIGATION_STATE_MANUAL:
    case NavState::NAVIGATION_STATE_STAB:
      return FlightMode::MANUAL;

    case NavState::NAVIGATION_STATE_ALTCTL:
      return FlightMode::ALTCTL;

    case NavState::NAVIGATION_STATE_POSCTL:
      return FlightMode::POSCTL;

    case NavState::NAVIGATION_STATE_AUTO_MISSION:
    case NavState::NAVIGATION_STATE_AUTO_LOITER:
    case NavState::NAVIGATION_STATE_AUTO_RTL:
    case NavState::NAVIGATION_STATE_AUTO_TAKEOFF:
    case NavState::NAVIGATION_STATE_AUTO_LAND:
    case NavState::NAVIGATION_STATE_AUTO_FOLLOW_TARGET:
    case NavState::NAVIGATION_STATE_AUTO_PRECLAND:
      return FlightMode::AUTO;

    case NavState::NAVIGATION_STATE_ACRO:
      return FlightMode::ACRO;

    default:
      return FlightMode::OTHERS;
    }
  }

} // namespace los_guidance