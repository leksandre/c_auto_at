/**
 * @file vehicle_controller.hpp
 * @brief Контроллер управления транспортным средством PX4
 * @note Отвечает за режимы полёта, публикацию команд и мониторинг статуса
 *
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#ifndef LOS_GUIDANCE_VEHICLE_CONTROLLER_HPP_
#define LOS_GUIDANCE_VEHICLE_CONTROLLER_HPP_

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <memory>
#include <string>

#include <optional>

namespace los_guidance
{

  /**
   * @enum FlightMode
   * @brief Режимы полёта PX4
   *
   * Соответствуют NAV_STATE из px4_msgs/msg/VehicleStatus
   */
  enum class FlightMode
  {
    UNKNOWN = 0,    ///< Неизвестный режим
    OFFBOARD = 1,   ///< Offboard режим (внешнее управление)
    MANUAL = 2,     ///< Ручной режим
    ALTCTL = 3,     ///< Контроль высоты
    POSCTL = 4,     ///< Контроль позиции
    AUTO = 5,       ///< Автономный режим
    ACRO = 6,       ///< Акробатический режим
    STABILIZED = 7, ///< Стабилизированный режим
    OTHERS = 99     ///< Другие режимы
  };

  /**
   * @struct VehicleControllerConfig
   * @brief Конфигурация контроллера транспортного средства
   *
   * Все параметры вынесены из node.cpp в этот struct для удобства
   * загрузки из YAML и тестирования.
   */
  struct VehicleControllerConfig
  {
    // === Системные ID ===
    uint8_t target_system{1};    ///< Target system ID
    uint8_t target_component{1}; ///< Target component ID
    uint8_t source_system{1};    ///< Source system ID
    uint8_t source_component{1}; ///< Source component ID

    // === Тайминги ===
    int offboard_timeout_ms{500}; ///< Таймаут offboard режима (ms)
    int command_retry_count{3};   ///< Количество повторных попыток команды

    // === Ограничения ускорений ===
    double max_hor_acc_limit{12.0};  ///< Максимальное горизонтальное ускорение (m/s²)
    double max_vert_acc_limit{12.0}; ///< Максимальное вертикальное ускорение (m/s²)
    double max_up_acc{-5.0};         ///< Максимальное ускорение вверх (NED: negative)
    double max_down_acc{5.0};        ///< Максимальное ускорение вниз (NED: positive)

    // === Фильтрация ===
    double acc_smooth_alpha{0.1}; ///< Коэффициент сглаживания ускорения

    // === Режимы ===
    bool enable_yaw_control{false};         ///< Включить контроль yaw
    bool enable_position_control{false};    //< Включить контроль позиции
    bool enable_velocity_control{false};    //< Включить контроль скорости
    bool enable_acceleration_control{true}; //< Включить контроль ускорения (по умолчанию)

    // === Масштабирование для тяжёлого дрона ===
    double heavy_drone_factor{1.0}; ///< Множитель ограничений ускорения (0.62 для 2519г)

    // === Отладка ===
    bool debug_logging{false}; ///< Включить отладочное логирование
  };

  /**
   * @struct VehicleStatusInfo
   * @brief Текущий статус транспортного средства
   */
  struct VehicleStatusInfo
  {
    FlightMode flight_mode{FlightMode::UNKNOWN}; ///< Текущий режим полёта
    uint8_t nav_state{0};                        ///< NAV_STATE от PX4
    bool is_armed{false};                        ///< Дрон вооружён
    bool is_offboard_available{false};           ///< Offboard режим доступен
    uint64_t last_status_timestamp_us{0};        ///< Timestamp последнего статуса
    bool is_valid{false};                        ///< Валидность статуса
  };

  /**
   * @struct ControlSetpoint
   * @brief Уставка управления для публикации
   */
  struct ControlSetpoint
  {
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()}; ///< Ускорение (nav frame)
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};     ///< Скорость (nav frame)
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};     ///< Позиция (nav frame)
    double yaw{NAN};                                       ///< Yaw угол (rad)
    double yaw_rate{NAN};                                  ///< Yaw скорость (rad/s)
    uint64_t timestamp_us{0};                              ///< Timestamp
    bool is_valid{false};                                  ///< Валидность уставки
  };

  /**
   * @class VehicleController
   * @brief Контроллер управления транспортным средством PX4
   *
   * Этот класс инкапсулирует всю логику взаимодействия с PX4:
   * - Мониторинг статуса дрона
   * - Переключение режимов полёта (Offboard, Manual, etc.)
   * - Публикация команд управления (ускорение, скорость, позиция)
   * - Контроль таймаутов offboard режима
   * - Сглаживание команд управления
   *
   * @thread_safe Да (все публичные методы защищены mutex)
   *
   * @example
   * ```cpp
   * VehicleControllerConfig config;
   * config.max_hor_acc_limit = 12.0;
   * config.acc_smooth_alpha = 0.1;
   *
   * auto controller = std::make_unique<VehicleController>(node, config);
   *
   * // В callback статуса
   * controller->updateStatus(status_msg);
   *
   * // В цикле управления
   * if (controller->isReadyForAttack()) {
   *   controller->publishAccelerationSetpoint(acc_cmd);
   * }
   * ```
   */
  class VehicleController
  {
  public:
    /**
     * @brief Конструктор
     * @param node Shared pointer на ROS2 node для создания publishers
     * @param config Конфигурация контроллера
     */
    explicit VehicleController(
        rclcpp::Node::SharedPtr node,
        const VehicleControllerConfig &config = VehicleControllerConfig());

    /**
     * @brief Деструктор
     */
    ~VehicleController() = default;

    /**
     * @brief Обновление статуса дрона
     * @param msg Сообщение VehicleStatus от PX4
     */
    void updateStatus(const px4_msgs::msg::VehicleStatus::SharedPtr msg);

    /**
     * @brief Получение текущего статуса
     * @return VehicleStatusInfo с текущим статусом
     */
    VehicleStatusInfo getStatus() const;

    /**
     * @brief Получение текущего режима полёта
     * @return Текущий режим полёта
     */
    FlightMode getCurrentMode() const;

    /**
     * @brief Проверка готовности к атаке
     * @return true если дрон готов к выполнению команд
     */
    bool isReadyForAttack() const;

    /**
     * @brief Проверка availability offboard режима
     * @return true если offboard режим доступен
     */
    bool isOffboardAvailable() const;

    /**
     * @brief Проверка вооружения дрона
     * @return true если дрон вооружён
     */
    bool isArmed() const;

    /**
     * @brief Установка режима Offboard
     * @return true если команда успешно отправлена
     */
    bool setOffboardMode();

    /**
     * @brief Установка режима Manual
     * @return true если команда успешно отправлена
     */
    bool setManualMode();

    /**
     * @brief Установка режима Altitude Control
     * @return true если команда успешно отправлена
     */
    bool setAltitudeMode();

    /**
     * @brief Установка режима Position Control
     * @return true если команда успешно отправлена
     */
    bool setPositionControlMode();

    /**
     * @brief Публикация команды ускорения
     * @param acc_setpoint Команда ускорения (nav frame, m/s²)
     * @return true если команда успешно опубликована
     */
    bool publishAccelerationSetpoint(const Eigen::Vector3d &acc_setpoint);

    /**
     * @brief Публикация команды скорости
     * @param vel_setpoint Команда скорости (nav frame, m/s)
     * @return true если команда успешно опубликована
     */
    bool publishVelocitySetpoint(const Eigen::Vector3d &vel_setpoint);

    /**
     * @brief Публикация команды позиции
     * @param pos_setpoint Команда позиции (nav frame, m)
     * @return true если команда успешно опубликована
     */

    bool publishPositionSetpoint(
        const Eigen::Vector3d &pos_setpoint,
        const Eigen::Vector3d &velocity = Eigen::Vector3d(NAN, NAN, NAN),
        const Eigen::Vector3d &acceleration = Eigen::Vector3d(NAN, NAN, NAN));

    /**
     * @brief Публикация режима управления Offboard
     * @return true если режим успешно опубликован
     */
    bool publishOffboardControlMode();

    /**
     * @brief Публикация полной уставки (trajectory setpoint)
     * @param setpoint Уставка управления
     * @return true если уставка успешно опубликована
     */
    bool publishTrajectorySetpoint(const ControlSetpoint &setpoint);

    /**
     * @brief Сброс состояния контроллера
     */
    void reset();

    /**
     * @brief Обновление конфигурации на лету
     * @param config Новая конфигурация
     */
    void updateConfig(const VehicleControllerConfig &config);

    /**
     * @brief Получение последней опубликованной уставки
     * @return ControlSetpoint с последней уставкой
     */
    ControlSetpoint getLastSetpoint() const;

    /**
     * @brief Включение/выключение отладочного логирования
     * @param enable true = включить логирование
     */
    void setDebugLogging(bool enable);

    /**
     * @brief Конвертация FlightMode в строку
     * @param mode Режим полёта
     * @return Строковое представление режима
     */
    static std::string flightModeToString(FlightMode mode);

  private:
    rclcpp::Node::SharedPtr node_;
    VehicleControllerConfig config_;
    mutable std::mutex mutex_; // Защита состояния

    // === Publishers ===
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr cmd_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr traj_pub_;

    // === Состояние ===
    VehicleStatusInfo status_;
    ControlSetpoint last_setpoint_;
    bool offboard_mode_sent_{false};
    uint64_t last_command_timestamp_us_{0};
    int command_retry_counter_{0};

    // === Сглаживание ускорения ===
    Eigen::Vector3d acc_smoothed_{Eigen::Vector3d::Zero()};
    bool acc_initialized_{false};

    // === Константы PX4 ===
    static constexpr uint8_t PX4_CUSTOM_MAIN_MODE_MANUAL = 1;
    static constexpr uint8_t PX4_CUSTOM_MAIN_MODE_ALTCTL = 2;
    static constexpr uint8_t PX4_CUSTOM_MAIN_MODE_POSCTL = 3;
    static constexpr uint8_t PX4_CUSTOM_MAIN_MODE_AUTO = 4;
    static constexpr uint8_t PX4_CUSTOM_MAIN_MODE_ACRO = 5;
    static constexpr uint8_t PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6;
    static constexpr uint8_t PX4_CUSTOM_MAIN_MODE_STABILIZED = 7;

    static constexpr uint8_t VEHICLE_CMD_DO_SET_MODE = 176;
    //   static constexpr uint8_t VEHICLE_CMD_COMPONENT_ARM_DISARM = 400;
    static constexpr uint16_t VEHICLE_CMD_COMPONENT_ARM_DISARM = 400;
    // === Внутренние методы ===

    /**
     * @brief Создание publishers
     */
    void createPublishers_();

    /**
     * @brief Публикация команды VehicleCommand
     * @param command ID команды PX4
     * @param param1 Параметр 1
     * @param param2 Параметр 2
     * @return true если команда успешно опубликована
     */
    bool publishVehicleCommand_(uint8_t command, float param1, float param2);

    /**
     * @brief Сглаживание ускорения (exponential moving average)
     * @param acc_raw_raw Сырое ускорение
     * @return Сглаженное ускорение
     */
    Eigen::Vector3d smoothAcceleration_(const Eigen::Vector3d &acc_raw);

    /**
     * @brief Ограничение ускорения
     * @param acc Ускорение для ограничения
     * @return Ограниченное ускорение
     */
    Eigen::Vector3d clampAcceleration_(const Eigen::Vector3d &acc);

    /**
     * @brief Получение текущего timestamp
     * @return Timestamp в микросекундах
     */
    uint64_t getCurrentTimestamp_() const;

    /**
     * @brief Конвертация NAV_STATE в FlightMode
     * @param nav_state NAV_STATE от PX4
     * @return FlightMode
     */
    FlightMode navStateToFlightMode_(uint8_t nav_state) const;
  };

} // namespace los_guidance

#endif // LOS_GUIDANCE_VEHICLE_CONTROLLER_HPP_