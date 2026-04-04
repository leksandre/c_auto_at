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

// === Сетевые заголовки (для TCP клиента/сервера) ===
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

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

#include <Eigen/Dense>
#include "los_guidance/target_tracker.hpp"
#include "los_guidance/guidance_controller.hpp"
#include "los_guidance/los_estimator.hpp"
#include "los_guidance/vehicle_controller.hpp"
#include "los_guidance/sensor_sync.hpp"
#include "los_guidance/adrc.hpp"
#include "los_guidance/commander.hpp"



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

    // Ограничения
    double max_tilt_deg{45.0};
    double min_tilt_deg{5.0};
    double max_lat_acc{3.0};
    double max_up_acc{-5.0};
    double max_down_acc{5.0};
    double max_vert_acc_limit{12.0};
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
    double los_y_ff_enable{false};
    double los_y_ff{7.0};
    double los_y_ff_limit{3.0};
    double los_y_err_ff_limit{0.1};
    double los_y_kp{3.0};
    double los_y_ki{0.01};
    double los_y_int_limit{3.0};

    // PI LOS параметры (Z axis)
    double los_z_ff_enable{false};
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

    /**
     * @brief Деструктор
     */
    ~Node() = default;

  private:
    // === Инициализация ===
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
     * @brief Обработка статуса дрона PX4
     * @param msg Сообщение VehicleStatus
     */
    void vehicleStatusCallback(px4_msgs::msg::VehicleStatus::SharedPtr msg);

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

    // === ROS2 Publishers ===
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicle_cmd_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_ctrl_mode_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_setpoint_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleLosStatus>::SharedPtr los_status_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr los_intercept_debug_pub_;

    // === ROS2 Services ===
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr auto_attack_srv_;

    // === ROS2 Timer ===
    rclcpp::TimerBase::SharedPtr timer_;

    // === Состояние (защищено mutex) ===
    mutable std::mutex state_mutex_;  ///< Защита состояния
    bool attack_enabled_{false};      ///< Авто-атака включена
    bool start_attack_{false};        ///< Сигнал начала атаки
    bool attack_ready_{false};        ///< Система готова к атаке
    bool update_acc_setpoint_{false}; ///< Флаг обновления ускорения

    // === Ключи синхронизации ===
    std::string att_sync_key_{"att"};       ///< Ключ attitude
    std::string ang_rate_sync_key_{"gyro"}; ///< Ключ gyro

    // === Константы ===
    static constexpr double CONSTANTS_ONE_G = 9.80665; ///< Ускорение свободного падения
    static constexpr double RAD2DEG = 180.0 / M_PI;    ///< Радианы → Градусы
    static constexpr double DEG2RAD = M_PI / 180.0;    ///< Градусы → Радианы

    std::vector<std::string> commander_targets_;
    int target_query_port_{11471};

    // === преследование цели ===
    enum class AutoPhase
    {
      IDLE,
      GPS_APPROACH,
      VISUAL_ATTACK
    };
    AutoPhase auto_phase_ = AutoPhase::IDLE;

    struct
    {
      double lat, lon, alt;
    } target_gps_ = {0, 0, 0};
    struct
    {
      double lat, lon, alt;
    } ref_gps_ = {0, 0, 0};
    bool ref_gps_valid_ = false;
    Eigen::Vector3d target_ned_ = Eigen::Vector3d::Zero();

    static Eigen::Vector3d gpsToNed(double lat, double lon, double alt,
                                    double ref_lat, double ref_lon, double ref_alt);

    struct GpsState
    {
      double lat, lon, alt;
      uint64_t timestamp_us;
      bool is_valid;
    } drone_gps_;

    std::unique_ptr<Commander> commander_;
    bool commander_enabled_{false};
    int commander_port_{11471};

    rclcpp::Subscription<px4_msgs::msg::VehicleGlobalPosition>::SharedPtr gps_pos_sub_;
    void vehicleGlobalPositionCallback(const px4_msgs::msg::VehicleGlobalPosition::SharedPtr msg);


    void executeAutoModeSequence();
    static double haversineDist(double lat1, double lon1, double lat2, double lon2);
    static bool queryTargetGPS(const std::string &ip, int port, double &out_lat, double &out_lon, double &out_alt, int &out_sats);
  };

  // === Математика расстояний (Haversine) ===
 

  static double haversineDist(double lat1, double lon1, double lat2, double lon2);
  static bool queryTargetGPS(const std::string &ip, int port, double &out_lat, double &out_lon, double &out_alt, int &out_sats);
  void executeAutoModeSequence();

} // namespace los_guidance

#endif // LOS_GUIDANCE_NODE_HPP_