/**
 * @file node.hpp
 * @brief ROS2 Node - главный оркестратор системы наведения LOS
 * @note Этот класс только координирует компоненты, вся логика делегирована контроллерам
 *
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#ifndef LOS_GUIDANCE_NODE_HPP_
#define LOS_GUIDANCE_NODE_HPP_

#pragma once
#include <memory>
#include <string>
#include <chrono>
#include <functional>
#include <algorithm>
#include <cmath>
#include <vector>
#include <mutex>
#include <limits>
#include <atomic>
#include <optional>

#include <random>
// === Сетевые заголовки (для TCP клиента/сервера) ===
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <px4_msgs/msg/vehicle_odometry.hpp>

#include <px4_msgs/msg/vehicle_global_position.hpp>
#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_angular_velocity.hpp>
#include <px4_msgs/msg/vehicle_los_status.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <ai_service/msg/tracker_result.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <Eigen/Dense>
#include "los_guidance/target_tracker.hpp"
#include "los_guidance/guidance_controller.hpp"
#include "los_guidance/los_estimator.hpp"
#include "los_guidance/vehicle_controller.hpp"
#include "los_guidance/sensor_sync.hpp"
#include "los_guidance/adrc.hpp"
#include "los_guidance/commander.hpp"


#include "los_guidance/gps_parser.hpp"


#include <deque>
#include <chrono>

namespace los_guidance
{

  /**
   * @struct NodeConfig
   * @brief Конфигурация главного узла (параметры из YAML)
   */
  struct NodeConfig
  {
    // Таймеры
    int update_ms{20};

    // Камера
    double camera_fx{410.0};
    double camera_fy{410.0};
    double camera_cx{640.0};
    double camera_cy{360.0};
    double vertical_fov_deg{60.0};

    // Установка камеры
    double cam_install_roll_deg{0.0};
    double cam_install_pitch_deg{5.0};
    double cam_install_yaw_deg{0.0};

    // Режимы
    bool enable_PI_control{false};
    bool top_attack_enable{false};
    bool enable_acc_parallel{false};
    bool use_gyro_comp{false};
    bool debug{false};

    // Фильтрация трекера
    double pixel_smooth_alpha{0.7};
    double los_smooth_alpha{0.3};
    double min_confidence{0.6};

    // Логирование
    int log_throttle_interval_ms{12000};

    // Ограничения
    double max_tilt_deg{45.0};
    double min_tilt_deg{5.0};
    double max_lat_acc{3.0};
    double max_up_acc{-5.0};
    double max_down_acc{5.0};
    double max_hor_acc_limit{12.0};

    // Скорость
    double min_vc{1.0};
    double max_vc{20.0};
    double vc_alpha{0.1};

    // PN параметры
    double N_gain{3.0};
    double apn_alpha{0.1};
    double ax_rate_limit{1.0};

    // Parallel acceleration
    double acc_parallel_gain{0.1};
    double acc_parallel_alpha{0.1};
    double max_acc_parallel{2.0};

    // PI LOS параметры (Y axis)
    bool los_y_ff_enable{false};
    double los_y_ff{7.0};
    double los_y_ff_limit{3.0};
    double los_y_err_ff_limit{0.1};
    double los_y_kp{3.0};
    double los_y_ki{0.01};
    double los_y_int_limit{3.0};

    // PI LOS параметры (Z axis)
    bool los_z_ff_enable{false};
    double los_z_ff{7.0};
    double los_z_ff_limit{3.0};
    double los_z_err_ff_limit{0.1};
    double los_z_kp{3.0};
    double los_z_ki{0.01};
    double los_z_int_limit{5.0};

    // Intercept параметры
    double intercept_kp{3.0};
    double intercept_ki{0.01};
    double intercept_kd{0.01};
    double intercept_int_limit{1.0};
    double intercept_z_kp{5.0};
    double intercept_acc_alpha{0.1};

    // Kalman Filter параметры
    double KF_los_vec_Q{0.01};
    double KF_los_vec_R{0.1};

    // ADRC параметры
    double adrc_r{1000.0};
    double adrc_h{0.01};
    double adrc_n{2};

    // Sync параметры
    size_t sync_buffer_size{100};
    int sync_retention_us{200000};

    // Takeoff параметры
    int takeoff_timeout_s{20};  ///< Таймаут взлёта (секунды)
    double kp_yaw_track{0.5};  // 🆕
    double max_approach_distance{200.0};  ///< Максимальное расстояние для перехода в визуальную атаку (м)

    // Waypoint tracking параметры
    double waypoint_arrival_tolerance_m{30.0};  ///< Допуск прибытия в точку маршрута (м)
    double waypoint_gps_loss_timeout_s{1800.0}; ///< Таймаут потери GPS перед посадкой (с), 1800 = 30 мин
    double waypoint_max_radius_from_home_m{20000.0}; ///< Максимальное расстояние от стартовой точки (м), 20 км

    // Параметры ручного управления
    int manual_control_update_rate_hz{20};
    double manual_control_deadzone{0.05};
    double manual_control_smoothing_alpha_base{0.1};
    double manual_control_smoothing_alpha_max{0.3};
    double manual_control_max_rate_per_sec{2.0};
    double manual_control_max_accel_mps2{1.86};
    double manual_control_max_yaw_rate_dps{45.0};
    double manual_control_max_vert_speed_mps{2.0};


    

        // === Параметры масштабирования ускорения (clamp) ===
    double clamp_modifications_scale_x{1.0};           // Множитель масштабирования для X оси
    double clamp_modifications_scale_y{1.0};           // Множитель масштабирования для Y оси
    double clamp_modifications_scale_z{1.0};           // Множитель масштабирования для Z оси
    





    


  };

  /**
   * @class Node
   * @brief Главный ROS2 узел системы автоматической атаки
   *
   * Ответственность:
   * - Загрузка параметров из YAML конфига
   * - Создание и инициализация контроллеров
   * - Подписка/публикация ROS2 топиков
   * - Координация между компонентами (Tracker → Estimator → Guidance → Vehicle)
   * - Обработка сервисов и таймеров
   *
   * @note Вся бизнес-логика делегирована отдельным контроллерам
   * @thread_safe Частично (использует mutex для защиты состояния)
   *
   * @see TargetTracker
   * @see LosEstimator
   * @see GuidanceController
   * @see VehicleController
   * @see SensorSync
   */
  class Node : public rclcpp::Node
  {
  public:
    /**
     * @brief Конструктор узла
     */
    Node();

    void postConstruct();

    
    void declareCommonParameters(); 
    /**
     * @brief Деструктор
     */
    ~Node() = default;

    /**
     * @brief Мгновенное чтение текущих координат из кэша (без блокировки надолго)
     * @param[out] lat Широта (градусы)
     * @param[out] lon Долгота (градусы)
     * @param[out] alt Высота (метры, MSL)
     * @return true если данные валидны
     * @note Читает из кэша, работает в любом режиме полёта
     */
    bool getDroneGps(double &lat, double &lon, double &alt) const;

  private:
    // === Инициализация ===

    void softReset();

    void uartGpsCallback();

    bool use_uart_gps_{false}; ///< Флаг использования UART GPS вместо PX4
    

    /**
     * @brief Загрузка всех параметров из YAML
     */
    void loadParameters();

    /**
     * @brief Создание всех подписчиков
     */
    void createSubscribers();

    /**
     * @brief Создание всех издателей
     */
    void createPublishers();

    /**
     * @brief Создание сервисов
     */
    void createServices();

    /**
     * @brief Инициализация всех контроллеров
     */
    void initializeControllers();

    // === Callbacks (подписчики) ===
    /**
     * @brief Обработка результата трекинга от AI сервиса
     * @param msg Сообщение TrackerResult
     */
    void trackerResultCallback(ai_service::msg::TrackerResult::SharedPtr msg);

    /**
     * @brief Обработка bounding box от vision_msgs
     * @param msg Сообщение Detection2D
     */
    void trackBBox2DCallback(vision_msgs::msg::Detection2D::SharedPtr msg);

    /**
     * @brief Общая обработка данных трекера (KF + guidance + lock-free store)
     * Вызывается из trackerResultCallback и trackBBox2DCallback после получения TargetInfo
     * @param target Информация о цели от трекера
     * @param timestamp_us Временная метка в микросекундах
     * @param omega_b Угловая скорость (gyro) для KF predict
     * @param R_nb Матрица вращения nav→body для KF predict
     * @param roll Текущий крен (rad) для computeAcceleration
     * @param pitch Текущий тангаж (rad) для computeAcceleration
     * @param yaw Текущий рыскание (rad) для computeAcceleration
     */
    void processTargetData_(
        const TargetInfo &target,
        uint64_t timestamp_us,
        const Eigen::Vector3d &omega_b,
        const Eigen::Matrix3d &R_nb,
        double roll,
        double pitch,
        double yaw);

    /**
     * @brief Обработка статуса дрона PX4
     * @param msg Сообщение VehicleStatus
     */
    void vehicleStatusCallback(px4_msgs::msg::VehicleStatus::SharedPtr msg);

    void odomCallback(px4_msgs::msg::VehicleOdometry::SharedPtr msg);

    /**
     * @brief Выполняет последовательность взлета (Takeoff)
     * @param skip_takeoff Если true, пропускать физический взлет (для симуляции или если уже летит)
     * @return void
     */
    void executeTakeoffSequence(bool skip_takeoff);
    
    /**
     * @brief Обработка ориентации дрона
     * @param msg Сообщение VehicleAttitude
     */
    void vehicleAttitudeCallback(px4_msgs::msg::VehicleAttitude::SharedPtr msg);

    /**
     * @brief Обработка угловой скорости дрона
     * @param msg Сообщение VehicleAngularVelocity
     */
    void vehicleAngularVelocityCallback(px4_msgs::msg::VehicleAngularVelocity::SharedPtr msg);

    /**
     * @brief Обработка сигнала начала атаки
     * @param msg Сообщение Bool
     */
    void attackSignalCallback(std_msgs::msg::Bool::SharedPtr msg);

    // === Service Callbacks ===
    /**
     * @brief Сервис включения/выключения авто-атаки
     * @param request Запрос (enable/disable)
     * @param response Ответ (success/message)
     */
    void autoAttackService(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response);

    /**
     * @brief Сервис сброса аварийной остановки
     * @param request Запрос
     * @param response Ответ
     */
    void emergencyResetService(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    /**
     * @brief Initialize Commander TCP server (port 11472)
     * @note Extracted for reusability/restart capability
     */
    void initCommander();

    /**
     * @brief Send ARM/DISARM command (10x loop for reliability)
     * @param arm true=ARM, false=DISARM
     */
    void sendArmCommand(bool arm);


    // === Timer Callbacks ===
    /**
     * @brief Основной цикл управления (вызывается по таймеру)
     */
    void controlLoopTimerCallback();

    // === Вспомогательные методы ===
    /**
     * @brief Публикация статуса LOS для отладки
     * @param target_info Информация о цели от трекера
     * @param acc_command Команда ускорения от guidance контроллера
     */
    void publishLosStatus(
        const TargetInfo &target_info,
        const Eigen::Vector3d &acc_command);

    /**
     * @brief Публикация отладочной информации Intercept
     * @param debug_data Массив отладочных данных
     */
    void publishInterceptDebug(const std::vector<float> &debug_data);
 

    // === Компоненты (контроллеры) ===
    std::unique_ptr<TargetTracker> target_tracker_;     ///< Трекинг цели
    std::unique_ptr<LosEstimator> los_estimator_;       ///< Kalman Filter LOS
    std::unique_ptr<GuidanceController> guidance_ctrl_; ///< Алгоритмы наведения
    std::unique_ptr<VehicleController> vehicle_ctrl_;   ///< Управление PX4
    std::unique_ptr<SensorSync> sensor_sync_;           ///< Синхронизация данных

    // === Конфигурация ===
    NodeConfig config_;              ///< Параметры узла
    TrackerConfig tracker_config_;   ///< Параметры трекера
    LosEstimatorConfig los_config_;  ///< Параметры KF
    GuidanceConfig guidance_config_; ///< Параметры наведения
    SyncConfig sync_config_;         ///< Параметры синхронизации

    // === ROS2 Subscribers ===
    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr vehicle_status_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr attack_signal_sub_;
    rclcpp::Subscription<ai_service::msg::TrackerResult>::SharedPtr tracker_result_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr track_bbox_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr vehicle_attitude_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAngularVelocity>::SharedPtr vehicle_angular_velocity_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr subscription_odom_;
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr heartbeat_sub_;  ///< AI heartbeat

    // === ROS2 Publishers ===
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicle_cmd_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_ctrl_mode_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_setpoint_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleLosStatus>::SharedPtr los_status_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr los_intercept_debug_pub_;

    // === ROS2 Services ===
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr auto_attack_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr emergency_reset_srv_;

    // === ROS2 Timer ===
    rclcpp::TimerBase::SharedPtr timer_;

    // === Callback Groups (для разделения AI и control flow) ===
    rclcpp::CallbackGroup::SharedPtr ai_callback_group_;      ///< Группа для AI подписок (отдельный поток)
    rclcpp::CallbackGroup::SharedPtr control_callback_group_; ///< Группа для PX4 telemetry + timer
    rclcpp::CallbackGroup::SharedPtr manual_control_callback_group_; ///< Группа для ручного управления (отдельный поток)

    double start_lat_{0.0}, start_lon_{0.0}, start_alt_{0.0};
    bool start_gps_valid_{false};

    los_guidance::GpsParser uart_gps_parser_;
    rclcpp::TimerBase::SharedPtr uart_gps_timer_; ///< Таймер для опроса UART GPS и публикации/синхронизации


    // === Состояние (защищено mutex) ===
    mutable std::mutex state_mutex_;  ///< Защита состояния
    bool attack_enabled_{false};      ///< Авто-атака включена
    bool start_attack_{false};        ///< Сигнал начала атаки
    std::atomic<bool> attack_ready_{false}; ///< Система готова к атаке (lock-free)
// std::atomic<double> last_los_cam_y_{0.0};  // 🆕
std::atomic<double> last_los_cam_y_{0.0};  // 🆕
bool update_acc_setpoint_{false}; 




    // === Ключи синхронизации ===
    std::string att_sync_key_{"att"};       ///< Ключ attitude
    std::string ang_rate_sync_key_{"gyro"}; ///< Ключ gyro

    // === Константы ===
    static constexpr double CONSTANTS_ONE_G = 9.80665; ///< Ускорение свободного падения
    static constexpr double RAD2DEG = 180.0 / M_PI;    ///< Радианы → Градусы
    static constexpr double DEG2RAD = M_PI / 180.0;    ///< Градусы → Радианы

    std::vector<std::string> commander_targets_;
    int target_query_port_{11472};

    // === преследование цели ===
    enum class AutoPhase
    {
      IDLE,
      GPS_APPROACH,
      VISUAL_ATTACK,
      VISUAL_SEARCH,
      GPS_WAYPOINT_TRACKING
    };
    AutoPhase auto_phase_ = AutoPhase::IDLE;

    // === Manual control input (from TCP Commander) ===
    struct ManualInput {
        std::atomic<float> roll{0.0f};
        std::atomic<float> pitch{0.0f};
        std::atomic<float> yaw{0.0f};
        std::atomic<float> throttle{0.5f};
        std::atomic<bool> updated{false};
    };
    ManualInput manual_input_;

    // === Manual control state for filtering ===
    struct FilteredManualInput {
        float roll_filtered{0.0f};
        float pitch_filtered{0.0f};
        float yaw_filtered{0.0f};
        float throttle_filtered{0.5f};
        std::chrono::steady_clock::time_point last_update;
        std::deque<std::pair<uint64_t, float>> throttle_spikes_history; // для детекции всплесков
    };
    FilteredManualInput filtered_manual_input_;

    // === Manual control limits ===
    double max_speed_{3.0};                    // м/с
    double max_vert_speed_{2.0};               // м/с  
    double max_yaw_rate_{45.0 * M_PI / 180.0}; // рад/с

    // 🆕 Активная команда (ручное управление через TCP)
    enum class ActiveCommand
    {
      NONE,
      LAND,              // Приземление
      SEARCH_ROUND,      // Круговой поиск цели
      UP_TO,             // Подъём на высоту
      DOWN_TO,           // Снижение на высоту
      PATROL_ROUND,      // Круговое патрулирование
      HOLD_TO_SQUARE,     // Удержание позиции в квадрате
      VISION_ATTACK,     // Режим визуальной атаки
      AUTO_LAND          // Автопосадка
    };
    ActiveCommand active_command_ = ActiveCommand::NONE;
    double active_command_param_{0.0};  // Параметр активной команды

    struct GpsState
    {
      double lat, lon, alt;
      uint64_t timestamp_us;
      bool is_valid;
    };

    GpsState target_gps_{0, 0, 0, 0, false};

    std::string closest_target_ip_;

    GpsState ref_gps_{0, 0, 0, 0, false};
    bool ref_gps_valid_ = false;
    Eigen::Vector3d target_ned_ = Eigen::Vector3d::Zero();

    // 🆕 Последняя валидная NED-позиция дрона (для fallback при потере GPS)
    Eigen::Vector3d last_hover_ned_ = Eigen::Vector3d::Zero();
    bool last_hover_ned_valid_{false};

    // 🆕 Таймер входа в VISUAL_ATTACK (для fallback в GPS_APPROACH через 2с при потере цели)
    std::chrono::steady_clock::time_point visual_attack_entry_time_;
    bool visual_attack_timer_active_{false};

    GpsState drone_gps_;

    // === GPS Waypoint Tracking ===
    struct Waypoint
    {
      double lat, lon;       // координаты точки
      std::string action;    // "wait", "land", "up"
      double param;          // параметр: секунды ожидания, метры подъёма
    };
    std::vector<Waypoint> gps_waypoints_;         ///< Загруженные waypoint из params.yaml
    size_t current_waypoint_idx_{0};              ///< Индекс текущего waypoint
    std::chrono::steady_clock::time_point waypoint_wait_start_;  ///< Время начала ожидания в waypoint
    bool waypoint_waiting_{false};                ///< Флаг: дрон ожидает в waypoint
    double waypoint_wait_duration_{0.0};          ///< Длительность ожидания (секунды)
    Eigen::Vector3d waypoint_wait_target_ned_;    ///< 🆕 Целевая NED позиция для удержания во время ожидания
    bool waypoint_climbing_{false};               ///< Флаг: дрон выполняет подъём
    double waypoint_climb_target_{0.0};           ///< Целевая высота подъёма (метры)
    std::chrono::steady_clock::time_point waypoint_climb_start_;  ///< Время начала подъёма
    double waypoint_speed_{7.0};                ///< 🆕 Скорость движения между waypoint (м/с)

    // GPS loss handling
    std::chrono::steady_clock::time_point gps_loss_start_;  ///< Время начала потери GPS
    bool gps_loss_active_{false};                ///< Флаг: GPS потерян
    bool gps_loss_landing_triggered_{false};     ///< Флаг: посадка по таймауту GPS уже запущена

    static Eigen::Vector3d gpsToNed(double lat, double lon, double alt,
                                    double ref_lat, double ref_lon, double ref_alt);

    std::unique_ptr<Commander> commander_;
    bool commander_enabled_{false};
    int commander_port_{11472};

    std::atomic<bool> takeoff_running_{false};   // 🆕 Флаг: взлёт активен
    std::atomic<bool> takeoff_abort_{false};      // 🆕 Флаг: прервать взлёт
    std::atomic<bool> emergency_stop_{false};     // 🆕 Флаг: аварийная остановка (блокирует ВСЕ команды)

    // === Lock-free очередь команд от AI к control loop ===
    // 🎯 Позволяет AI callback работать в отдельном потоке без блокировки mutex
    struct AccelerationCommand
    {
      Eigen::Vector3d acc_cmd = Eigen::Vector3d::Zero();
      double confidence = 0.0;
      uint64_t timestamp_us = 0;
      bool is_valid = false;
    };
    std::atomic<bool> new_acc_cmd_available_{false};  // Флаг: новая команда готова
    AccelerationCommand latest_acc_cmd_;              // Последняя команда (защищена флагом)

    rclcpp::Subscription<px4_msgs::msg::VehicleGlobalPosition>::SharedPtr gps_pos_sub_;
    void vehicleGlobalPositionCallback(const px4_msgs::msg::VehicleGlobalPosition::SharedPtr msg);

    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr local_pos_sub_;
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

    // 🆕 Хелпер: синхронизация attitude/gyro + конвертация quaternion → euler
    bool syncAndGetOrientation_(
        uint64_t ref_us,
        Eigen::Quaterniond& quat,
        Eigen::Vector3d& omega_b,
        double& roll,
        double& pitch,
        double& yaw,
        Eigen::Matrix3d& R_nb);

    // 🆕 Хелпер: публикация setpoint атаки (OFFBOARD + acc_cmd)
    void executeAttackSetpoint_(const Eigen::Vector3d& acc_cmd);

    void executeAutoModeSequence();
    void executeSearchMode();                          // 🆕
    void executeWaypointTrackingSequence();            // 🆕 GPS_WAYPOINT_TRACKING
    void waypointDeleteByIndex(int index);             // 🆕 waypoint_delete
    void waypointAddAtIndex(int index, const std::string& waypoint_str); // 🆕 waypoint_add
    void waypointSkipCurrent();                        // 🆕 waypoint_skip
    std::string waypointListJson();                    // 🆕 waypoint_list — returns JSON
    static double haversineDist(double lat1, double lon1, double lat2, double lon2);
    bool queryTargetGPS(const std::string &ip, int port, double &out_lat, double &out_lon, double &out_alt, int &out_sats);

    // 🆕 Обработка прерывания взлёта
    void handleTakeoffAbort();
  };

} // namespace los_guidance

#endif // LOS_GUIDANCE_NODE_HPP_
