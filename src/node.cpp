/**
 * @file node.cpp
 * @brief Реализация главного ROS2 узла системы наведения LOS
 * @note Этот файл содержит только оркестрацию, вся логика в контроллерах
 *
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#include <los_guidance/node.hpp>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>

namespace los_guidance
{

  // ============================================================================
  // Конструктор
  // ============================================================================

  Node::Node() : rclcpp::Node("los_guidance_node")
  {
    RCLCPP_INFO(this->get_logger(), "Initializing LOS Guidance Node...");

    // 1. Загрузка параметров из YAML
    loadParameters();

    // 2. Инициализация контроллеров
    // initializeControllers(); // вызовем в другом месте

    // 3. Создание publishers/subscribers/services
    createPublishers();
    createSubscribers();
    createServices();

    // 4. Создание таймера основного цикла
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(config_.update_ms),
        std::bind(&Node::controlLoopTimerCallback, this));

    RCLCPP_INFO(this->get_logger(), "LOS Guidance Node initialized successfully");
  }

  void Node::postConstruct()
  {
    RCLCPP_INFO(this->get_logger(), "Post-constructing controllers...");
    initializeControllers(); // ✅ Теперь shared_from_this() работает
  }

  // ============================================================================
  // Инициализация
  // ============================================================================

  void Node::loadParameters()
  {
    RCLCPP_INFO(this->get_logger(), "Loading parameters...");

    // === Таймеры ===
    this->declare_parameter("timer.update_ms", 20);
    config_.update_ms = this->get_parameter("timer.update_ms").as_int();

    // === Камера ===
    this->declare_parameter("camera.fx", 410.0);
    this->declare_parameter("camera.fy", 410.0);
    this->declare_parameter("camera.cx", 640.0);
    this->declare_parameter("camera.cy", 360.0);
    this->declare_parameter("camera.vertical_fov_deg", 60.0);

    config_.camera_fx = this->get_parameter("camera.fx").as_double();
    config_.camera_fy = this->get_parameter("camera.fy").as_double();
    config_.camera_cx = this->get_parameter("camera.cx").as_double();
    config_.camera_cy = this->get_parameter("camera.cy").as_double();
    config_.vertical_fov_deg = this->get_parameter("camera.vertical_fov_deg").as_double();

    // === Установка камеры ===
    this->declare_parameter("camera.install.roll_deg", 0.0);
    this->declare_parameter("camera.install.pitch_deg", 5.0);
    this->declare_parameter("camera.install.yaw_deg", 0.0);

    config_.cam_install_roll_deg = this->get_parameter("camera.install.roll_deg").as_double();
    config_.cam_install_pitch_deg = this->get_parameter("camera.install.pitch_deg").as_double();
    config_.cam_install_yaw_deg = this->get_parameter("camera.install.yaw_deg").as_double();

    // === Режимы ===
    this->declare_parameter("guidance.enable_PI_control", false);
    this->declare_parameter("guidance.top_attack_mode", false);
    this->declare_parameter("guidance.enable_acceleration_parallel", false);
    this->declare_parameter("kalman.use_gyro_compensation", false);
    this->declare_parameter("sync.debug", false);

    config_.enable_PI_control = this->get_parameter("guidance.enable_PI_control").as_bool();
    config_.top_attack_enable = this->get_parameter("guidance.top_attack_mode").as_bool();
    config_.enable_acc_parallel = this->get_parameter("guidance.enable_acceleration_parallel").as_bool();
    config_.use_gyro_comp = this->get_parameter("kalman.use_gyro_compensation").as_bool();
    config_.debug = this->get_parameter("sync.debug").as_bool();

    // === Ограничения ===
    this->declare_parameter("guidance.max_tilt_deg", 45.0);
    this->declare_parameter("guidance.min_tilt_deg", 5.0);
    this->declare_parameter("acceleration.max_lateral_mps2", 3.0);
    this->declare_parameter("acceleration.max_vertical_up_mps2", -5.0);
    this->declare_parameter("acceleration.max_vertical_down_mps2", 5.0);
    this->declare_parameter("acceleration.max_horizontal_mps2", 12.0);

    config_.max_tilt_deg = this->get_parameter("guidance.max_tilt_deg").as_double();
    config_.min_tilt_deg = this->get_parameter("guidance.min_tilt_deg").as_double();
    config_.max_lat_acc = this->get_parameter("acceleration.max_lateral_mps2").as_double();
    config_.max_up_acc = this->get_parameter("acceleration.max_vertical_up_mps2").as_double();
    config_.max_down_acc = this->get_parameter("acceleration.max_vertical_down_mps2").as_double();
    config_.max_vert_acc_limit = this->get_parameter("acceleration.max_horizontal_mps2").as_double();
    config_.max_hor_acc_limit = this->get_parameter("acceleration.max_horizontal_mps2").as_double();

    // === Скорость ===
    this->declare_parameter("velocity.min_vc_mps", 1.0);
    this->declare_parameter("velocity.max_vc_mps", 20.0);
    this->declare_parameter("velocity.vc_alpha_smooth", 0.1);

    config_.min_vc = this->get_parameter("velocity.min_vc_mps").as_double();
    config_.max_vc = this->get_parameter("velocity.max_vc_mps").as_double();
    config_.vc_alpha = this->get_parameter("velocity.vc_alpha_smooth").as_double();

    // === PN параметры ===
    this->declare_parameter("guidance.pn_gain_N", 3.0);
    this->declare_parameter("acceleration.apn_alpha", 0.1);
    this->declare_parameter("acceleration.ax_rate_limit_mps3", 1.0);

    config_.N_gain = this->get_parameter("guidance.pn_gain_N").as_double();
    config_.apn_alpha = this->get_parameter("acceleration.apn_alpha").as_double();
    config_.ax_rate_limit = this->get_parameter("acceleration.ax_rate_limit_mps3").as_double();

    // === Parallel acceleration ===
    this->declare_parameter("acceleration.parallel.enable", false);
    this->declare_parameter("acceleration.parallel.gain", 0.1);
    this->declare_parameter("acceleration.parallel.alpha", 0.1);
    this->declare_parameter("acceleration.parallel.max_mps2", 2.0);

    config_.enable_acc_parallel = this->get_parameter("acceleration.parallel.enable").as_bool();
    config_.acc_parallel_gain = this->get_parameter("acceleration.parallel.gain").as_double();
    config_.acc_parallel_alpha = this->get_parameter("acceleration.parallel.alpha").as_double();
    config_.max_acc_parallel = this->get_parameter("acceleration.parallel.max_mps2").as_double();

    // === PI LOS параметры (Y) ===
    this->declare_parameter("los_pi.y.kp", 3.0);
    this->declare_parameter("los_pi.y.ki", 0.01);
    this->declare_parameter("los_pi.y.ff_enable", false);
    this->declare_parameter("los_pi.y.ff_gain", 7.0);
    this->declare_parameter("los_pi.y.ff_limit", 3.0);
    this->declare_parameter("los_pi.y.err_ff_limit", 0.1);
    this->declare_parameter("los_pi.y.int_limit", 3.0);

    config_.los_y_kp = this->get_parameter("los_pi.y.kp").as_double();
    config_.los_y_ki = this->get_parameter("los_pi.y.ki").as_double();
    config_.los_y_ff_enable = this->get_parameter("los_pi.y.ff_enable").as_bool();
    config_.los_y_ff = this->get_parameter("los_pi.y.ff_gain").as_double();
    config_.los_y_ff_limit = this->get_parameter("los_pi.y.ff_limit").as_double();
    config_.los_y_err_ff_limit = this->get_parameter("los_pi.y.err_ff_limit").as_double();
    config_.los_y_int_limit = this->get_parameter("los_pi.y.int_limit").as_double();

    // === PI LOS параметры (Z) ===
    this->declare_parameter("los_pi.z.kp", 3.0);
    this->declare_parameter("los_pi.z.ki", 0.01);
    this->declare_parameter("los_pi.z.ff_enable", false);
    this->declare_parameter("los_pi.z.ff_gain", 7.0);
    this->declare_parameter("los_pi.z.ff_limit", 3.0);
    this->declare_parameter("los_pi.z.err_ff_limit", 0.1);
    this->declare_parameter("los_pi.z.int_limit", 5.0);

    config_.los_z_kp = this->get_parameter("los_pi.z.kp").as_double();
    config_.los_z_ki = this->get_parameter("los_pi.z.ki").as_double();
    config_.los_z_ff_enable = this->get_parameter("los_pi.z.ff_enable").as_bool();
    config_.los_z_ff = this->get_parameter("los_pi.z.ff_gain").as_double();
    config_.los_z_ff_limit = this->get_parameter("los_pi.z.ff_limit").as_double();
    config_.los_z_err_ff_limit = this->get_parameter("los_pi.z.err_ff_limit").as_double();
    config_.los_z_int_limit = this->get_parameter("los_pi.z.int_limit").as_double();

    // === Intercept параметры ===
    this->declare_parameter("intercept.kp", 3.0);
    this->declare_parameter("intercept.ki", 0.01);
    this->declare_parameter("intercept.kd", 0.01);
    this->declare_parameter("intercept.int_limit", 1.0);
    this->declare_parameter("intercept.z_kp", 5.0);
    this->declare_parameter("intercept.acc_alpha", 0.1);

    config_.intercept_kp = this->get_parameter("intercept.kp").as_double();
    config_.intercept_ki = this->get_parameter("intercept.ki").as_double();
    config_.intercept_kd = this->get_parameter("intercept.kd").as_double();
    config_.intercept_int_limit = this->get_parameter("intercept.int_limit").as_double();
    config_.intercept_z_kp = this->get_parameter("intercept.z_kp").as_double();
    config_.intercept_acc_alpha = this->get_parameter("intercept.acc_alpha").as_double();

    // === Kalman Filter параметры ===
    this->declare_parameter("kalman.los_process_noise_Q", 0.01);
    this->declare_parameter("kalman.los_measurement_noise_R", 0.1);

    config_.KF_los_vec_Q = this->get_parameter("kalman.los_process_noise_Q").as_double();
    config_.KF_los_vec_R = this->get_parameter("kalman.los_measurement_noise_R").as_double();

    // === ADRC параметры ===
    this->declare_parameter("adrc.td_r", 1000.0);
    this->declare_parameter("adrc.td_h", 0.01);
    this->declare_parameter("adrc.td_n", 2);

    config_.adrc_r = this->get_parameter("adrc.td_r").as_double();
    config_.adrc_h = this->get_parameter("adrc.td_h").as_double();
    config_.adrc_n = static_cast<double>(this->get_parameter("adrc.td_n").as_int());

    // === Commander параметры ===
    this->declare_parameter("commander.enable", false);
    this->declare_parameter("commander.port", 11472);
    this->declare_parameter("commander.targets", std::vector<std::string>{}); // 🆕
    this->declare_parameter("commander.target_port", 11471);                  // 🆕

    commander_enabled_ = this->get_parameter("commander.enable").as_bool();
    commander_port_ = static_cast<int>(this->get_parameter("commander.port").as_int());
    commander_targets_ = this->get_parameter("commander.targets").as_string_array();
    target_query_port_ = static_cast<int>(this->get_parameter("commander.target_port").as_int());

    // === Sync параметры ===
    // this->declare_parameter("sync.buffer_size", 10А0);
    this->declare_parameter("sync.buffer_size", 1000);
    this->declare_parameter("sync.retention_us", 200000);

    config_.sync_buffer_size = this->get_parameter("sync.buffer_size").as_int();
    config_.sync_retention_us = this->get_parameter("sync.retention_us").as_int();

    RCLCPP_INFO(this->get_logger(), "Parameters loaded successfully");
    commander_enabled_ = this->get_parameter("commander.enable").as_bool();
RCLCPP_INFO(this->get_logger(), "🔧 Commander enable: %s | Port: %d", 
            commander_enabled_ ? "TRUE" : "FALSE", commander_port_);
            
  }

  void Node::initializeControllers()
  {
    RCLCPP_INFO(this->get_logger(), "Initializing controllers...");

    // === 1. Tracker Config ===
    tracker_config_.fx = config_.camera_fx;
    tracker_config_.fy = config_.camera_fy;
    tracker_config_.cx = config_.camera_cx;
    tracker_config_.cy = config_.camera_cy;
    tracker_config_.vertical_fov_deg = config_.vertical_fov_deg;
    tracker_config_.install_roll_deg = config_.cam_install_roll_deg;
    tracker_config_.install_pitch_deg = config_.cam_install_pitch_deg;
    tracker_config_.install_yaw_deg = config_.cam_install_yaw_deg;
    tracker_config_.pixel_smooth_alpha = 0.3; // Можно вынести в конфиг
    tracker_config_.los_smooth_alpha = 0.1;   // Можно вынести в конфиг
    tracker_config_.min_confidence = 0.5;     // Можно вынести в конфиг

    target_tracker_ = std::make_unique<TargetTracker>(tracker_config_);

    // === 2. Los Estimator Config ===
    los_config_.process_noise_Q = config_.KF_los_vec_Q;
    los_config_.measurement_noise_R = config_.KF_los_vec_R;
    los_config_.use_gyro_compensation = config_.use_gyro_comp;

    los_estimator_ = std::make_unique<LosEstimator>(los_config_);

    // === 3. Guidance Config ===
    guidance_config_.pn_gain_N = config_.N_gain;
    guidance_config_.los_y_kp = config_.los_y_kp;
    guidance_config_.los_y_ki = config_.los_y_ki;
    guidance_config_.los_y_ff_enable = config_.los_y_ff_enable;
    guidance_config_.los_y_ff = config_.los_y_ff;
    guidance_config_.los_y_ff_limit = config_.los_y_ff_limit;
    guidance_config_.los_y_int_limit = config_.los_y_int_limit;
    guidance_config_.los_z_kp = config_.los_z_kp;
    guidance_config_.los_z_ki = config_.los_z_ki;
    guidance_config_.los_z_ff_enable = config_.los_z_ff_enable;
    guidance_config_.los_z_ff = config_.los_z_ff;
    guidance_config_.los_z_ff_limit = config_.los_z_ff_limit;
    guidance_config_.los_z_int_limit = config_.los_z_int_limit;
    guidance_config_.intercept_kp = config_.intercept_kp;
    guidance_config_.intercept_ki = config_.intercept_ki;
    guidance_config_.intercept_kd = config_.intercept_kd;
    guidance_config_.intercept_int_limit = config_.intercept_int_limit;
    guidance_config_.intercept_z_kp = config_.intercept_z_kp;
    guidance_config_.max_lat_acc = config_.max_lat_acc;
    guidance_config_.max_up_acc = config_.max_up_acc;
    guidance_config_.max_down_acc = config_.max_down_acc;
    guidance_config_.enable_PI_control = config_.enable_PI_control;
    guidance_config_.top_attack_mode = config_.top_attack_enable;
    guidance_config_.enable_acc_parallel = config_.enable_acc_parallel;
    guidance_config_.max_hor_acc = config_.max_hor_acc_limit;

    guidance_ctrl_ = std::make_unique<GuidanceController>(
        guidance_config_,
        config_.update_ms / 1000.0 // dt в секундах
    );

    // === 4. Vehicle Controller ===
    vehicle_ctrl_ = std::make_unique<VehicleController>(shared_from_this());

    // === 5. Sensor Sync ===
    sync_config_.buffer_size = config_.sync_buffer_size;
    sync_config_.retention_us = config_.sync_retention_us;
    sync_config_.debug = config_.debug;

    sensor_sync_ = std::make_unique<SensorSync>(sync_config_);
    sensor_sync_->registerSource(att_sync_key_);
    sensor_sync_->registerSource(ang_rate_sync_key_);

    // === 6. Commander (сетевые команды атаки) ===

    if (commander_enabled_)
    {
      commander_ = std::make_unique<Commander>(
          commander_port_,
          [this](bool start)
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            start_attack_ = start;
            RCLCPP_INFO(this->get_logger(), "🌐 Commander: start_attack_ set to %s", start ? "true" : "false");
          },
          [this]()
          {
            std::thread(&Node::executeAutoModeSequence, this).detach();
          },
          [this](double &lat, double &lon, double &alt) -> bool
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!drone_gps_.is_valid)
              return false;
            lat = drone_gps_.lat;
            lon = drone_gps_.lon;
            alt = drone_gps_.alt;
            return true;
          },
          // 🆕 Колбэк для команды HOME
          [this]() -> Commander::HomeData
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            Commander::HomeData hd;
            hd.lat = start_lat_;
            hd.lon = start_lon_;
            hd.alt = start_alt_;
            hd.valid = start_gps_valid_;

            // Сбрасываем текущую миссию
            auto_phase_ = AutoPhase::IDLE;
            start_attack_ = false;
            attack_enabled_ = false;

            if (hd.valid)
            {
              RCLCPP_INFO(this->get_logger(), "🏠 RTL triggered. Home: %.6f, %.6f, %.1f", start_lat_, start_lon_, start_alt_);
              px4_msgs::msg::VehicleCommand cmd;
              cmd.command = 20; // MAV_CMD_NAV_RETURN_TO_LAUNCH
              cmd.param1 = 0.0;
              cmd.target_system = 1;
              cmd.target_component = 1;
              cmd.from_external = true;
              cmd.timestamp = this->get_clock()->now().nanoseconds() / 1000;
              vehicle_cmd_pub_->publish(cmd);
            }
            else
            {
              RCLCPP_WARN(this->get_logger(), "🏠 No home coordinates! Triggering emergency LAND.");
              px4_msgs::msg::VehicleCommand cmd;
              cmd.command = 21; // MAV_CMD_NAV_LAND
              cmd.param1 = 0.0;
              cmd.target_system = 1;
              cmd.target_component = 1;
              cmd.from_external = true;
              cmd.timestamp = this->get_clock()->now().nanoseconds() / 1000;
              vehicle_cmd_pub_->publish(cmd);
            }
            return hd;
          },
          this->get_logger());
      commander_->start();
    }

    if (config_.debug)
    {
      sensor_sync_->setDebug(true);
    }

    RCLCPP_INFO(this->get_logger(), "Controllers initialized successfully");
  }

  void Node::createPublishers()
  {
    RCLCPP_INFO(this->get_logger(), "Creating publishers...");

    // 🆕 QoS для команд управления - RELIABLE для надёжности
    rclcpp::QoS command_qos(10);
    command_qos.reliable();
    command_qos.transient_local();

    vehicle_cmd_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
        "fmu/in/vehicle_command", command_qos);

    offboard_ctrl_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
        "fmu/in/offboard_control_mode", command_qos);

    trajectory_setpoint_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
        "fmu/in/trajectory_setpoint", command_qos);

    los_status_pub_ = this->create_publisher<px4_msgs::msg::VehicleLosStatus>(
        "fmu/in/vehicle_los_status", 10);

    los_intercept_debug_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "los_intercept_debug", 10);

    RCLCPP_INFO(this->get_logger(), "Publishers created successfully");
  }

  void Node::createSubscribers()
  {
    RCLCPP_INFO(this->get_logger(), "Creating subscribers...");

    // QoS для PX4 сообщений (sensor_data)
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 5), qos_profile);

    vehicle_status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
        "fmu/out/vehicle_status",
        qos,
        std::bind(&Node::vehicleStatusCallback, this, std::placeholders::_1));

    attack_signal_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/attack_signal",
        10,
        std::bind(&Node::attackSignalCallback, this, std::placeholders::_1));

    tracker_result_sub_ = this->create_subscription<ai_service::msg::TrackerResult>(
        "tracker_result",
        10,
        std::bind(&Node::trackerResultCallback, this, std::placeholders::_1));

    track_bbox_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
        "track_bbox",
        10,
        std::bind(&Node::trackBBox2DCallback, this, std::placeholders::_1));

    vehicle_attitude_sub_ = this->create_subscription<px4_msgs::msg::VehicleAttitude>(
        "fmu/out/vehicle_attitude",
        qos,
        std::bind(&Node::vehicleAttitudeCallback, this, std::placeholders::_1));

    vehicle_angular_velocity_sub_ = this->create_subscription<px4_msgs::msg::VehicleAngularVelocity>(
        "fmu/out/vehicle_angular_velocity",
        qos,
        std::bind(&Node::vehicleAngularVelocityCallback, this, std::placeholders::_1));

    gps_pos_sub_ = this->create_subscription<px4_msgs::msg::VehicleGlobalPosition>(
        "/fmu/out/vehicle_global_position",
        rclcpp::SensorDataQoS().keep_last(1), // DDS QoS для сенсоров
        std::bind(&Node::vehicleGlobalPositionCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Subscribers created successfully");
  }

  void Node::createServices()
  {
    RCLCPP_INFO(this->get_logger(), "Creating services...");

    auto_attack_srv_ = this->create_service<std_srvs::srv::SetBool>(
        "auto_attack",
        std::bind(&Node::autoAttackService, this,
                  std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Services created successfully");
  }

  // ============================================================================
  // Callbacks
  // ============================================================================

  void Node::vehicleGlobalPositionCallback(const px4_msgs::msg::VehicleGlobalPosition::SharedPtr msg)
  {
    if (!msg)
      return;

    // 1. Сохраняем в state_mutex_ (потокобезопасно)
    std::lock_guard<std::mutex> lock(state_mutex_);
    drone_gps_.lat = msg->lat;
    drone_gps_.lon = msg->lon;
    drone_gps_.alt = msg->alt;
    drone_gps_.timestamp_us = msg->timestamp_sample;

    // 🛠 FIX: valid_global_pos удалён в новых px4_msgs.
    // Считаем GPS валидным, если пришло сообщение с ненулевым временем.
    drone_gps_.is_valid = (msg->timestamp_sample > 0);

    // 2. Синхронизируем с IMU/attitude через SensorSync
    // 🛠 FIX: msg уже является std::shared_ptr, не нужно оборачивать в make_shared
    sensor_sync_->addMessage("drone_gps", msg->timestamp_sample, msg);

    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "GPS: %.7f, %.7f, %.2f (valid: %s)",
                          msg->lat, msg->lon, msg->alt,
                          drone_gps_.is_valid ? "Y" : "N");
  }

  void Node::trackerResultCallback(ai_service::msg::TrackerResult::SharedPtr msg)
  {

    // === 🪵 ОТЛАДОЧНЫЕ ЛОГИ (добавить в начало) ===
    if (config_.debug)
    {
      RCLCPP_DEBUG(this->get_logger(),
                   "🎯 trackerResultCallback 00: ts=%lu, conf=%.3f",
                   msg->timestamp,
                   msg->target_info.confidence);
    }

    uint64_t ref_us = msg->timestamp / 1000; // ns → us

    // RCLCPP_INFO(this->get_logger(), "🔍 trackerResultCallback 11: ref_us=%lu", ref_us);
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000, "🔍 trackerResultCallback 11: ref_us=%lu", ref_us);

    // Синхронизация с attitude и gyro
    auto sync_msg = sensor_sync_->sync(ref_us);

    Eigen::Quaterniond quat;
    Eigen::Vector3d omega_b;
    bool quat_ready = false;
    bool gyro_ready = false;

    // === Синхронизация с attitude и gyro ===

    if (sync_msg[att_sync_key_])
    {
      quat = *std::static_pointer_cast<Eigen::Quaterniond>(sync_msg[att_sync_key_]);
      quat_ready = true;
    }

    if (sync_msg[ang_rate_sync_key_])
    {
      omega_b = *std::static_pointer_cast<Eigen::Vector3d>(sync_msg[ang_rate_sync_key_]);
      gyro_ready = true;
    }

    // insertion for test
    quat_ready = true;
    gyro_ready = true;

    // RCLCPP_INFO(this->get_logger(), "   quat_ready=%d, gyro_ready=%d", quat_ready, gyro_ready);
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000, "   quat_ready=%d, gyro_ready=%d", quat_ready, gyro_ready);

    // === 🪵 Лог синхронизации ===
    if (config_.debug)
    {
      // RCLCPP_DEBUG(this->get_logger(),
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                           "🔗 Sync: trackerResultCallback 22:  quat=%d, gyro=%d, ref_us=%lu",
                           quat_ready ? 1 : 0,
                           gyro_ready ? 1 : 0,
                           ref_us);
    }

    if (quat_ready && gyro_ready)
    {
      // Конвертация quaternion → euler
      const Eigen::Matrix3d R = quat.toRotationMatrix();
      const double roll = std::atan2(R(2, 1), R(2, 2));
      const double pitch = std::asin(-R(2, 0));
      const double yaw = std::atan2(R(1, 0), R(0, 0));

      // Матрица вращения nav → body
      Eigen::Matrix3d R_nb = quat.toRotationMatrix();

      // 🎯 Обработка через TargetTracker
      TargetInfo target = target_tracker_->processTrackerResult(
          msg,
          roll,
          pitch,
          R_nb);

      // === 🪵 Лог результата трекинга ===
      if (config_.debug)
      {
        // RCLCPP_DEBUG(this->get_logger(),
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                             "🎯 Target: trackerResultCallback 33:  pixel=(%.1f,%.1f)→(%.1f,%.1f), conf=%.3f, valid=%d",
                             target.pixel_u, target.pixel_v,
                             target.pixel_u_smooth, target.pixel_v_smooth,
                             target.confidence,
                             target.is_valid ? 1 : 0);
      }

      if (target.is_valid)
      {
        // RCLCPP_INFO(this->get_logger(),
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                             " trackerResultCallback 44:   ✅ target.valid=true, confidence=%.2f", target.confidence);
        // RCLCPP_INFO(this->get_logger(),
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                             "   🎯 Setting attack_ready_=true");
      }
      else
      {
        // RCLCPP_WARN(this->get_logger(),
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                             "  trackerResultCallback 44:  ❌ target.valid=false");
      }

      if (target.is_valid)
      {
        // 🎯 Обновление Kalman Filter
        los_estimator_->predict(config_.update_ms / 1000.0, omega_b, R_nb);
        los_estimator_->update(target.los_nav);

        // 🎯 Получение оценённых LOS и lambda
        Eigen::Vector3d u_los = los_estimator_->getLosVector();
        Eigen::Vector3d lambda = los_estimator_->getLosAngularVelocity();

        // === 🪵 Лог LOS вектора ===
        if (config_.debug)
        {
          // RCLCPP_DEBUG(this->get_logger(),
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                               "📐 LOS: trackerResultCallback 55:  nav=[%.3f,%.3f,%.3f], lambda=[%.3f,%.3f,%.3f]",
                               u_los.x(), u_los.y(), u_los.z(),
                               lambda.x(), lambda.y(), lambda.z());
        }

        // 🎯 Расчёт ускорения через GuidanceController
        // double Vc = guidance_ctrl_->estimateVc(pitch, config_.max_tilt_deg * DEG2RAD);

        GuidanceOutput output = guidance_ctrl_->computeAcceleration(
            u_los,
            lambda,
            pitch, // ← 3-й параметр pitch_rad, не Vc!
            yaw);
        Eigen::Vector3d acc_cmd = output.acceleration_cmd;

        // === 🪵 Лог команды ускорения ===
        if (config_.debug)
        {
          // RCLCPP_DEBUG(this->get_logger(),
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                               "🚀 Acc cmd: trackerResultCallback 66:  [%.3f, %.3f, %.3f] m/s²",
                               acc_cmd.x(), acc_cmd.y(), acc_cmd.z());
        }

        // 🎯 Публикация статуса для отладки
        publishLosStatus(target, acc_cmd);

        // Сохранение команды для timer callback
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          // guidance_ctrl_->setLastAccelerationCommand(acc_cmd);
          attack_ready_ = true;
          // RCLCPP_INFO(this->get_logger(),
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                               "attack_ready_  trackerResultCallback");

          // === 🪵 Лог готовности к атаке ===
          // RCLCPP_INFO(this->get_logger(),
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                               "✅ trackerResultCallback 77:  attack_ready_=true | conf=%.2f | acc=[%.2f,%.2f,%.2f]",
                               target.confidence,
                               acc_cmd.x(), acc_cmd.y(), acc_cmd.z());
        }
      }
      else
      {
        // === 🪵 Лог если цель не валидна ===
        if (config_.debug)
        {
          // RCLCPP_WARN(this->get_logger(),
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 12000,
                               "❌ trackerResultCallback 77:  Target invalid: conf=%.3f < min_confidence=%.3f",
                               target.confidence,
                               tracker_config_.min_confidence);
        }
      }
    }
  }

  void Node::trackBBox2DCallback(vision_msgs::msg::Detection2D::SharedPtr msg)
  {
    uint64_t ref_us = static_cast<uint64_t>(msg->header.stamp.sec * 1e6) +
                      static_cast<uint64_t>(msg->header.stamp.nanosec * 1e-3);
    RCLCPP_INFO(this->get_logger(), "🔍 trackBBox2DCallback: ref_us=%lu", ref_us);

    auto sync_msg = sensor_sync_->sync(ref_us);

    Eigen::Quaterniond quat;
    Eigen::Vector3d omega_b;
    bool quat_ready = false;
    bool gyro_ready = false;

    if (sync_msg[att_sync_key_])
    {
      quat = *std::static_pointer_cast<Eigen::Quaterniond>(sync_msg[att_sync_key_]);
      quat_ready = true;
    }

    if (sync_msg[ang_rate_sync_key_])
    {
      omega_b = *std::static_pointer_cast<Eigen::Vector3d>(sync_msg[ang_rate_sync_key_]);
      gyro_ready = true;
    }

    if (quat_ready && gyro_ready)
    {

      // Конвертация quaternion → euler
      const Eigen::Matrix3d R = quat.toRotationMatrix();
      const double roll = std::atan2(R(2, 1), R(2, 2));
      const double pitch = std::asin(-R(2, 0));
      const double yaw = std::atan2(R(1, 0), R(0, 0));

      // Матрица вращения nav → body
      Eigen::Matrix3d R_nb = quat.toRotationMatrix();

      TargetInfo target = target_tracker_->processDetection2D(
          msg,
          roll,
          pitch,
          R_nb);

      // if (target.is_valid) {
      //   // los_estimator_->predict(config_.update_ms / 1000.0, omega_b);
      //   los_estimator_->predict(config_.update_ms / 1000.0, omega_b, R_nb);
      //   los_estimator_->update(target.los_nav);

      //   Eigen::Vector3d u_los = los_estimator_->getLosVector();
      //   Eigen::Vector3d lambda = los_estimator_->getLosAngularVelocity();

      //   // double Vc = guidance_ctrl_->estimateVc(pitch, config_.max_tilt_deg * DEG2RAD);
      //   double Vc = guidance_ctrl_->estimateVcFromPitch(pitch, config_.max_tilt_deg * DEG2RAD);

      //   publishLosStatus(target, acc_cmd);

      //   {
      //     std::lock_guard<std::mutex> lock(state_mutex_);
      //     // guidance_ctrl_->setLastAccelerationCommand(acc_cmd);
      //     attack_ready_ = true;
      //   }
      // }

      if (target.is_valid)
      {
        los_estimator_->predict(config_.update_ms / 1000.0, omega_b, R_nb);
        los_estimator_->update(target.los_nav);

        Eigen::Vector3d u_los = los_estimator_->getLosVector();
        Eigen::Vector3d lambda = los_estimator_->getLosAngularVelocity();

        // 🎯 Расчёт ускорения через GuidanceController
        GuidanceOutput output = guidance_ctrl_->computeAcceleration(
            u_los,
            lambda,
            pitch,
            yaw);
        Eigen::Vector3d acc_cmd = output.acceleration_cmd; // ← Получить из output

        publishLosStatus(target, acc_cmd);

        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          attack_ready_ = true;
          RCLCPP_INFO(this->get_logger(), "attack_ready_  trackBBox2DCallback");
        }
      }
    }
  }

  void Node::vehicleStatusCallback(px4_msgs::msg::VehicleStatus::SharedPtr msg)
  {
    vehicle_ctrl_->updateStatus(msg);

    // // 🆕 Авто-триггер атаки при переходе в OFFBOARD
    // FlightMode new_mode = navStateToFlightMode_(msg->nav_state);

    // if (new_mode == FlightMode::OFFBOARD && state_ != FlightMode::OFFBOARD) {
    //   // Только если система готова к атаке
    //   if (attack_ready_) {
    //     std::lock_guard<std::mutex> lock(state_mutex_);
    //     start_attack_ = true;  // ← Триггер атаки!
    //     RCLCPP_INFO(this->get_logger(), "🎯 OFFBOARD → ATTACK START");
    //   }
    // }

    // // 🆕 Авто-стоп при выходе из OFFBOARD
    // if (new_mode != FlightMode::OFFBOARD && state_ == FlightMode::OFFBOARD) {
    //   std::lock_guard<std::mutex> lock(state_mutex_);
    //   start_attack_ = false;  // ← Стоп атаки
    //   RCLCPP_INFO(this->get_logger(), "🛑 Left OFFBOARD → ATTACK STOP");
    // }

    // state_ = new_mode;
  }

  void Node::vehicleAttitudeCallback(px4_msgs::msg::VehicleAttitude::SharedPtr msg)
  {
    uint64_t timestamp = msg->timestamp;
    Eigen::Quaterniond q(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
    std::shared_ptr<Eigen::Quaterniond> q_ptr(new Eigen::Quaterniond(q));
    sensor_sync_->addMessage(att_sync_key_, timestamp, q_ptr);
  }

  void Node::vehicleAngularVelocityCallback(
      px4_msgs::msg::VehicleAngularVelocity::SharedPtr msg)
  {
    uint64_t timestamp = msg->timestamp;
    Eigen::Vector3d omega_b(msg->xyz[0], msg->xyz[1], msg->xyz[2]);
    std::shared_ptr<Eigen::Vector3d> omega_b_ptr(new Eigen::Vector3d(omega_b));
    sensor_sync_->addMessage(ang_rate_sync_key_, timestamp, omega_b_ptr);
  }

  void Node::attackSignalCallback(std_msgs::msg::Bool::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    start_attack_ = msg->data;
    RCLCPP_INFO(this->get_logger(), "Attack signal: %s", start_attack_ ? "START" : "STOP");
  }

  void Node::autoAttackService(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (attack_ready_)
    {
      attack_enabled_ = request->data;
      response->success = true;
      response->message = attack_enabled_ ? "Auto attack START" : "Auto attack STOP";
      RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
    }
    else
    {
      attack_enabled_ = false;
      response->success = false;
      response->message = "Auto attack NOT READY";
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
    }
  }

  // ============================================================================
  // Timer Callback (основной цикл)
  // ============================================================================

  void Node::controlLoopTimerCallback()
  {
    std::unique_lock<std::mutex> lock(state_mutex_);// блять блочит весь цикл...

    // === ФАЗА 1: СБЛИЖЕНИЕ ПО GPS ===
    if (auto_phase_ == AutoPhase::GPS_APPROACH)
    {
      // 🆕 Инициализируем ref_gps_ при первом входе (если ещё не установлен)
      if (!ref_gps_valid_ && drone_gps_.is_valid)
      {
        ref_gps_.lat = drone_gps_.lat;
        ref_gps_.lon = drone_gps_.lon;
        ref_gps_.alt = drone_gps_.alt;
        ref_gps_valid_ = true;
        RCLCPP_INFO(this->get_logger(), "📍 Reference GPS set: %.6f, %.6f, %.2f", ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);
      }

      if (ref_gps_valid_)
      {
        // Конвертируем цель в локальные NED-координаты относительно точки взлёта
        target_ned_ = gpsToNed(target_gps_.lat, target_gps_.lon, target_gps_.alt,
                               ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);
      }

      bool camera_locked = attack_ready_;
      double current_dist = 9999.0;
      if (drone_gps_.is_valid)
      {
        current_dist = haversineDist(drone_gps_.lat, drone_gps_.lon,
                                     target_gps_.lat, target_gps_.lon);
      }

      // 🆕 Логирование фаз
      std::string phase_name = (auto_phase_ == AutoPhase::GPS_APPROACH) ? "GPS" : (auto_phase_ == AutoPhase::VISUAL_ATTACK) ? "VISUAL"
                                                                                                                            : "IDLE";
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "🔄 Phase: %s | Dist: %.1fm | Camera: %s",
                           phase_name.c_str(), current_dist, camera_locked ? "LOCKED" : "SEARCHING");

      static int poll_counter = 0;
      if (++poll_counter >= 250)
      {
        poll_counter = 0;

        // 🆕 Если цели нет (hover mode) — опрашиваем ВСЕ цели в поисках рабочей
        if (closest_target_ip_.empty() && !commander_targets_.empty())
        {
          RCLCPP_INFO(this->get_logger(), "🔍 Scanning all targets for available target...");
          std::string found_ip;
          double found_lat = 0, found_lon = 0, found_alt = 0;
          double my_lat = 0; //, my_lon = 0;
          {
            std::lock_guard<std::mutex> gps_lock(state_mutex_);
            if (drone_gps_.is_valid)
            {
              my_lat = drone_gps_.lat;
              my_lon = drone_gps_.lon;
            }
          }

          for (const auto &ip : commander_targets_)
          {
            double t_lat, t_lon, t_alt;
            int t_sats;
            if (queryTargetGPS(ip, target_query_port_, t_lat, t_lon, t_alt, t_sats))
            {
              RCLCPP_INFO(this->get_logger(), "✅ Found available target: %s", ip.c_str());
              found_ip = ip;
              found_lat = t_lat;
              found_lon = t_lon;
              found_alt = t_alt;
              break;
            }
          }

          lock.lock();
          if (!found_ip.empty() && std::abs(my_lat) > 1e-3)
          {
            target_gps_ = {found_lat, found_lon, found_alt};
            closest_target_ip_ = found_ip;
            RCLCPP_INFO(this->get_logger(), "🎯 Target acquired: %s. Starting approach!", found_ip.c_str());
            // ref_gps будет обновлён при следующем входе в callback
          }
          lock.unlock();
          lock.lock();
        }
        else if (!closest_target_ip_.empty())
        {
          // 1. Копируем IP и отпускаем мьютекс перед TCP
          std::string ip_to_poll = closest_target_ip_;
          lock.unlock(); // 🔓 Временно освобождаем

          // 2. TCP вызов (не блокирует мьютекс)
          double t_lat, t_lon, t_alt;
          int t_sats;
          if (queryTargetGPS(ip_to_poll, target_query_port_, t_lat, t_lon, t_alt, t_sats))
          {
            lock.lock(); // 🔒 Захватываем обратно для записи
            target_gps_ = {t_lat, t_lon, t_alt};
            RCLCPP_INFO(this->get_logger(), "📡 Target GPS updated: %.6f, %.6f", t_lat, t_lon);
            lock.unlock(); // 🔓 Сразу отпускаем, чтобы не тормозить публикацию
            lock.lock();   // 🔒 Возвращаем для продолжения работы функции
          }
        }
      }

      // Условие перехода в атаку: камера видит цель И расстояние < 200м
      if (camera_locked && current_dist < 200.0)
      {
        auto_phase_ = AutoPhase::VISUAL_ATTACK;
        start_attack_ = true;
        attack_enabled_ = true;
        RCLCPP_INFO(this->get_logger(), "🎯 VISUAL ATTACK ENGAGED (dist=%.1fm < 200m)", current_dist);
        // Проваливаемся в логику атаки в этом же тике
      }
      else
      {
        // Продолжаем лететь по GPS
        vehicle_ctrl_->publishOffboardControlMode();
        vehicle_ctrl_->publishPositionSetpoint(target_ned_);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "🚁 Approaching... Dist: %.1fm | Camera: %s",
                             current_dist, camera_locked ? "LOCKED" : "SEARCHING");
        // без!!! return; здесь. Пусть функция завершится естественно.
      }
    }

    // === ФАЗА 2: ВИЗУАЛЬНАЯ АТАКА (LOS Guidance) ===

    if (auto_phase_ == AutoPhase::VISUAL_ATTACK)
    {
      if (!attack_ready_)
      {
        // 🔄 Цель потеряна — возвращаемся к подлёту по GPS (не останавливаемся!)
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "⚠️ Target lost! Returning to GPS_APPROACH (target_gps_ preserved)");
        auto_phase_ = AutoPhase::GPS_APPROACH;
        start_attack_ = false;
        attack_enabled_ = false;
        // ❗ Не делаем return — проваливаемся в логику GPS_APPROACH ниже
      }
      else
      {
        // Цель видима — выполняем визуальную атаку
        bool should_attack = (start_attack_ || attack_enabled_);
        if (should_attack)
        {
          if (vehicle_ctrl_->getCurrentMode() == FlightMode::OFFBOARD)
          {
            update_acc_setpoint_ = true;
            vehicle_ctrl_->publishOffboardControlMode();
            vehicle_ctrl_->publishAccelerationSetpoint(
                guidance_ctrl_->getLastAccelerationCommand());
          }
          else
          {
            guidance_ctrl_->reset();
            vehicle_ctrl_->publishOffboardControlMode();
            vehicle_ctrl_->publishAccelerationSetpoint(
                guidance_ctrl_->getLastAccelerationCommand());
            vehicle_ctrl_->setOffboardMode();
          }
        }
        else
        {
          if (vehicle_ctrl_->getCurrentMode() == FlightMode::OFFBOARD && update_acc_setpoint_)
          {
            guidance_ctrl_->reset();
            vehicle_ctrl_->setManualMode();
            update_acc_setpoint_ = false;
            auto_phase_ = AutoPhase::IDLE;
            RCLCPP_INFO(this->get_logger(), "Attack stopped, switched to MANUAL");
          }
        }
        return; // ✅ Выходим только если цель видима или атака остановлена вручную
      }
      // Если мы здесь → цель потеряна, фаза сменилась на GPS_APPROACH
      // ↓↓↓ Проваливаемся в код фазы ниже (не прерываем выполнение)
    }

    // === ФАЗА 3: ЛЕГАСИ / РУЧНОЙ ТРИГГЕР (TCP start_attack / RC) ===
    if (!attack_ready_)
      return;
    bool should_attack = (start_attack_ || attack_enabled_);
    if (should_attack)
    {
      if (vehicle_ctrl_->getCurrentMode() == FlightMode::OFFBOARD)
      {
        update_acc_setpoint_ = true;
        vehicle_ctrl_->publishOffboardControlMode();
        vehicle_ctrl_->publishAccelerationSetpoint(guidance_ctrl_->getLastAccelerationCommand());
      }
      else
      {
        guidance_ctrl_->reset();
        vehicle_ctrl_->publishOffboardControlMode();
        vehicle_ctrl_->publishAccelerationSetpoint(guidance_ctrl_->getLastAccelerationCommand());
        vehicle_ctrl_->setOffboardMode();
      }
    }
    else
    {
      if (vehicle_ctrl_->getCurrentMode() == FlightMode::OFFBOARD && update_acc_setpoint_)
      {
        guidance_ctrl_->reset();
        vehicle_ctrl_->setManualMode();
        update_acc_setpoint_ = false;
        RCLCPP_INFO(this->get_logger(), "Attack stopped, switched to MANUAL");
      }
    }
  }

  // ============================================================================
  // Вспомогательные методы
  // ============================================================================

  void Node::publishLosStatus(
      const TargetInfo &target_info,
      const Eigen::Vector3d &acc_command)
  {
    px4_msgs::msg::VehicleLosStatus msg;

    msg.detect_cu = static_cast<float>(target_info.pixel_u);
    msg.detect_cv = static_cast<float>(target_info.pixel_v);
    msg.correct_cu = static_cast<float>(target_info.pixel_u_smooth);
    msg.correct_cv = static_cast<float>(target_info.pixel_v_smooth);
    msg.los_vc_x = static_cast<float>(target_info.los_cam.x());
    msg.los_vc_y = static_cast<float>(target_info.los_cam.y());
    msg.los_vn_x_flt = static_cast<float>(target_info.los_nav.x());
    msg.los_vn_y_flt = static_cast<float>(target_info.los_nav.y());
    msg.los_vn_z_flt = static_cast<float>(target_info.los_nav.z());
    msg.a_pn_x = static_cast<float>(acc_command.x());
    msg.a_pn_y = static_cast<float>(acc_command.y());
    msg.a_pn_z = static_cast<float>(acc_command.z());
    msg.timestamp = target_info.timestamp_us;

    los_status_pub_->publish(msg);
  }

  void Node::publishInterceptDebug(const std::vector<float> &debug_data)
  {
    std_msgs::msg::Float32MultiArray msg;
    msg.data = debug_data;
    los_intercept_debug_pub_->publish(msg);
  }

  Eigen::Vector3d Node::gpsToNed(double lat, double lon, double alt,
                                 double ref_lat, double ref_lon, double ref_alt)
  {
    double dlat = (lat - ref_lat) * M_PI / 180.0;
    double dlon = (lon - ref_lon) * M_PI / 180.0;

    double dist = haversineDist(ref_lat, ref_lon, lat, lon);
    if (dist > 3000.0)
    {
      // RCLCPP_WARN(logger_, "⚠️ Large distance %.1fm may cause NED projection errors", dist);
      std::cerr << "[WARN] Large distance " << dist << "m may cause NED projection errors\n";
    }

    // Простая проекция на плоскость (валидна для дистанций < 2-3 км)

    // double x = dlon * 6371000.0 * std::cos(ref_lat * M_PI / 180.0); // East
    // double y = dlat * 6371000.0;                                    // North
    // 🛠 FIX: PX4 NED frame: X=North, Y=East, Z=Down
    // Раньше было наоборот (X=East), из-за чего дрон летел боком
    double x = dlat * 6371000.0;                                    // North
    double y = dlon * 6371000.0 * std::cos(ref_lat * M_PI / 180.0); // East

    double z = -(alt - ref_alt); // Down (NED)

    return {x, y, z};
  }

  double Node::haversineDist(double lat1, double lon1, double lat2, double lon2)
  {
    const double R = 6371000.0; // Радиус Земли в метрах
    double dLat = (lat2 - lat1) * M_PI / 180.0;
    double dLon = (lon2 - lon1) * M_PI / 180.0;
    double a = std::sin(dLat / 2) * std::sin(dLat / 2) +
               std::cos(lat1 * M_PI / 180.0) * std::cos(lat2 * M_PI / 180.0) *
                   std::sin(dLon / 2) * std::sin(dLon / 2);
    return R * 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
  }

  // === TCP-клиент для опроса целей ===
  bool Node::queryTargetGPS(const std::string &ip, int port, double &out_lat, double &out_lon, double &out_alt, int &out_sats)
  {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
      return false;

    struct sockaddr_in addr
    {
    };
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0)
    {
      close(sock);
      return false;
    }

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200000; // 200 мс вместо 2 с
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
      close(sock);
      return false;
    }

    std::string req = "{\"showgps\": true}\n";
    send(sock, req.c_str(), req.size(), 0);
    char buf[512] = {0};
    int n = recv(sock, buf, sizeof(buf) - 1, 0);
    close(sock);
    if (n <= 0)
      return false;
    buf[n] = '\0';

    // Лёгкий парсер без зависимостей
    auto findVal = [&](const char *key) -> double
    {
      std::string s(buf);
      size_t pos = s.find(key);
      if (pos == std::string::npos)
        return 0.0;
      pos = s.find(':', pos);
      if (pos == std::string::npos)
        return 0.0;
      try
      {
        return std::stod(s.c_str() + pos + 1);
      }
      // catch (...)
      // {
      //   return 0.0;
      // }
      catch (const std::exception &e)
      {
        // RCLCPP_WARN(logger_, "⚠️ Parse error for key '%s': %s", key, e.what());
        std::cerr << "[WARN] Parse error for key in json " << key << "  !!errors\n";
        return 0.0;
      }
    };

    // if (std::string(buf).find("\"status\":\"success\"") == std::string::npos)
    //   return false;

    std::string data(buf);
    if (data.find("\"status\"") == std::string::npos || data.find("success") == std::string::npos)
    {
      return false;
    }

    out_lat = findVal("\"lat\"");
    out_lon = findVal("\"lon\"");
    out_alt = findVal("\"alt\"");
    out_sats = (int)findVal("\"sats\"");
    return (out_sats > 0 && std::abs(out_lat) > 1e-3);
  }

  // === Основная авто-последовательность ===
  void Node::executeAutoModeSequence()
  {
    RCLCPP_INFO(this->get_logger(), "🚀 Starting AUTO mode sequence...");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Сохраняем точку взлёта как NED-начало координат
    // 🆕 Сохраняем полную стартовую позицию (lat/lon/alt)
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (drone_gps_.is_valid)
      {
        start_lat_ = drone_gps_.lat;
        start_lon_ = drone_gps_.lon;
        start_alt_ = drone_gps_.alt;
        start_gps_valid_ = true;
      }
    }
    double start_alt = NAN;
    if (start_gps_valid_)
      start_alt = start_alt_;

    // 1. Arm (отправляем многократно для надёжности)
    RCLCPP_INFO(this->get_logger(), "🔑 Sending ARM command...");
    for (int i = 0; i < 10; i++)
    {
      px4_msgs::msg::VehicleCommand arm_cmd;
      arm_cmd.command = 400; // MAV_CMD_COMPONENT_ARM_DISARM
      arm_cmd.param1 = 1.0f;
      arm_cmd.target_system = 1;
      arm_cmd.target_component = 1;
      arm_cmd.from_external = true;
      arm_cmd.timestamp = this->get_clock()->now().nanoseconds() / 1000;
      vehicle_cmd_pub_->publish(arm_cmd);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 2. Offboard (отправляем многократно для надёжности)
    RCLCPP_INFO(this->get_logger(), "📻 Switching to OFFBOARD...");
    for (int i = 0; i < 10; i++)
    {
      vehicle_ctrl_->setOffboardMode();
      vehicle_ctrl_->publishOffboardControlMode();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 3. Takeoff to 5m @ 7m/s
    RCLCPP_INFO(this->get_logger(), "🛫 Taking off to 5m altitude...");
    // 🛠 FIX: Используем float, чтобы убрать warning narrowing conversion
    float target_z = -5.0f;
    float vz = -7.0f;

    int cycles = 0;

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (drone_gps_.is_valid)
        start_alt = drone_gps_.alt;
    }

    while (cycles < 1200)
    {
      px4_msgs::msg::TrajectorySetpoint ts;
      ts.position = {NAN, NAN, target_z};
      ts.velocity = {NAN, NAN, vz};
      ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
      trajectory_setpoint_pub_->publish(ts);
      vehicle_ctrl_->publishOffboardControlMode();

      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (drone_gps_.is_valid && !std::isnan(start_alt))
        {
          // 🛠 FIX: При взлёте высота (MSL) увеличивается. Меняем <= на >=
          if (drone_gps_.alt >= start_alt + 4.5)
          {
            RCLCPP_INFO(this->get_logger(), "✅ Altitude reached.");
            break;
          }
          if (drone_gps_.alt >= start_alt + 2.0 && cycles > 20)
          {
            RCLCPP_INFO(this->get_logger(), "✅ Already flying.");
            break;
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      cycles++;
    }

    // 4. Poll targets
    if (commander_targets_.empty())
    {
      RCLCPP_WARN(this->get_logger(), "⚠️ No targets. Aborting.");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "📡 Polling %zu targets...", commander_targets_.size());
    std::string closest_ip;
    double min_dist = std::numeric_limits<double>::max();
    double drone_lat = 0, drone_lon = 0, closest_lat = 0, closest_lon = 0, closest_alt = 0;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (drone_gps_.is_valid)
      {
        drone_lat = drone_gps_.lat;
        drone_lon = drone_gps_.lon;
      }
    }

    for (const auto &ip : commander_targets_)
    {
      double t_lat, t_lon, t_alt;
      int t_sats;
      if (queryTargetGPS(ip, target_query_port_, t_lat, t_lon, t_alt, t_sats))
      {
        if (std::abs(drone_lat) > 1e-3)
        {
          double d = haversineDist(drone_lat, drone_lon, t_lat, t_lon);
          if (d < min_dist)
          {
            min_dist = d;
            closest_ip = ip;
            closest_lat = t_lat;
            closest_lon = t_lon;
            closest_alt = t_alt;
          }
        }
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "❌ Timeout/invalid response from %s", ip.c_str());
      }
    }

    // 5. Переключаемся в режим GPS-сближения
    if (!closest_ip.empty())
    {
      RCLCPP_INFO(this->get_logger(), "🎯 Closest target: %s (%.1f m). INITIATING GPS APPROACH!", closest_ip.c_str(), min_dist);
      std::lock_guard<std::mutex> lock(state_mutex_);
      target_gps_ = {closest_lat, closest_lon, closest_alt};
      closest_target_ip_ = closest_ip; // 🆕 Сохраняем IP для периодического опроса
      auto_phase_ = AutoPhase::GPS_APPROACH;
      attack_ready_ = false;
    }
    else
    {
      // 🆕 Цели не найдены — переводим дрон в режим зависания на текущей позиции
      // и продолжаем периодически опрашивать цели в основном цикле
      RCLCPP_WARN(this->get_logger(), "🛑 No valid targets found. Entering HOVER mode and will keep polling...");
      std::lock_guard<std::mutex> lock(state_mutex_);
      // Устанавливаем цель = текущая позиция дрона (дрон будет зависать на месте)
      if (drone_gps_.is_valid)
      {
        target_gps_ = {drone_gps_.lat, drone_gps_.lon, drone_gps_.alt};
      }
      else
      {
        // Если GPS дрона недоступен, используем точку взлёта
        target_gps_ = {start_lat_, start_lon_, start_alt_ + 5.0};
      }
      closest_target_ip_ = ""; // Пока нет цели
      auto_phase_ = AutoPhase::GPS_APPROACH;
      attack_ready_ = false;
    }
  }

  // === Публичный метод для мгновенного чтения GPS ===
  bool Node::getDroneGps(double &lat, double &lon, double &alt) const
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!drone_gps_.is_valid)
      return false;
    lat = drone_gps_.lat;
    lon = drone_gps_.lon;
    alt = drone_gps_.alt;
    return true;
  }

} // namespace los_guidance