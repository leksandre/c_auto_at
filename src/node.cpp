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

    declareCommonParameters();

    // 1. Загрузка параметров из YAML
    loadParameters();

    // 2. Инициализация контроллеров
    // initializeControllers(); // вызовем в другом месте

    // 3. Создание publishers/subscribers/services
    createPublishers();
    createSubscribers();
    createServices();

    // 4. Создание таймера основного цикла (в control callback group)
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(config_.update_ms),
        std::bind(&Node::controlLoopTimerCallback, this),
        control_callback_group_);

    RCLCPP_INFO(this->get_logger(), "LOS Guidance Node initialized successfully");
  }

  void Node::postConstruct()
  {
    RCLCPP_INFO(this->get_logger(), "Post-constructing controllers...");
    initializeControllers(); // ✅ Теперь shared_from_this() работает

    // В функции Node::postConstruct() или в конце конструктора Node():

    // 1. Загрузка параметров UART GPS
    this->declare_parameter("uart_gps.port", "/dev/ttyUSB0");
    this->declare_parameter("uart_gps.baudrate", 9600);
    this->declare_parameter("uart_gps.enable", false); // По умолчанию выключено, так как может конфликтовать с PX4 GPS

    std::string uart_port = this->get_parameter("uart_gps.port").as_string();
    int uart_baudrate = this->get_parameter("uart_gps.baudrate").as_int();
    bool enable_uart_gps = this->get_parameter("uart_gps.enable").as_bool();

    if (enable_uart_gps)
    {
      RCLCPP_INFO(this->get_logger(), "🛰️ Starting UART GPS parser on %s", uart_port.c_str());
      uart_gps_parser_.start(uart_port, uart_baudrate);

      // Создаем таймер для периодического считывания данных из UART GPS
      // Частота 10 Гц достаточно для навигации, если PX4 GPS недоступен
      uart_gps_timer_ = this->create_wall_timer(
          std::chrono::milliseconds(100),
          std::bind(&Node::uartGpsCallback, this));
    }
    else
    {
      RCLCPP_INFO(this->get_logger(), "🛰️ UART GPS disabled");
    }
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
    // Дистанция для перехода в визуальную атаку
    this->declare_parameter("attack.max_approach_distance_m", 200.0);
    config_.max_approach_distance = this->get_parameter("attack.max_approach_distance_m").as_double();

    config_.enable_PI_control = this->get_parameter("guidance.enable_PI_control").as_bool();
    config_.top_attack_enable = this->get_parameter("guidance.top_attack_mode").as_bool();
    config_.enable_acc_parallel = this->get_parameter("guidance.enable_acceleration_parallel").as_bool();
    config_.use_gyro_comp = this->get_parameter("kalman.use_gyro_compensation").as_bool();
    config_.debug = this->get_parameter("sync.debug").as_bool();

    // === Логирование ===
    this->declare_parameter("logging.throttle_interval_ms", 12000);
    config_.log_throttle_interval_ms = this->get_parameter("logging.throttle_interval_ms").as_int();

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

    // Commander params moved to initCommander()

    // === Sync параметры ===
    // this->declare_parameter("sync.buffer_size", 10А0);
    this->declare_parameter("sync.buffer_size", 1000);
    this->declare_parameter("sync.retention_us", 200000);

    config_.sync_buffer_size = this->get_parameter("sync.buffer_size").as_int();
    config_.sync_retention_us = this->get_parameter("sync.retention_us").as_int();

    // === Search mode params ===

    this->declare_parameter("search.kp_yaw_track", 0.5);

    config_.kp_yaw_track = this->get_parameter("search.kp_yaw_track").as_double();

    // === Search mode params ===

    // === Commander параметры (moved from loadParameters) ===
    this->declare_parameter("commander.enable", false);
    this->declare_parameter("commander.port", 11472);
    this->declare_parameter("commander.targets", std::vector<std::string>{});
    this->declare_parameter("commander.target_port", 11472); // 11471 было

    commander_enabled_ = this->get_parameter("commander.enable").as_bool();
    commander_port_ = static_cast<int>(this->get_parameter("commander.port").as_int());
    commander_targets_ = this->get_parameter("commander.targets").as_string_array();
    target_query_port_ = static_cast<int>(this->get_parameter("commander.target_port").as_int());

    RCLCPP_INFO(this->get_logger(), "🔧 Commander enable: %s | Port: %d | Targets: %zu",
                commander_enabled_ ? "TRUE" : "FALSE", commander_port_, commander_targets_.size());

    config_.kp_yaw_track = this->get_parameter("search.kp_yaw_track").as_double();

    this->declare_parameter("gps.source", "uart"); // Возможные значения: "px4", "uart"
    std::string gps_source = this->get_parameter("gps.source").as_string();

    if (gps_source == "uart")
    {
      RCLCPP_INFO(this->get_logger(), "🛰️ GPS Source set to: UART");
      use_uart_gps_ = true; // Предположим, у вас есть такой флаг, см. ниже
    }
    else
    {
      RCLCPP_INFO(this->get_logger(), "🛰️ GPS Source set to: PX4 (Default)");
      use_uart_gps_ = false;
    }

    RCLCPP_INFO(this->get_logger(), "Parameters loaded successfully");
    commander_enabled_ = this->get_parameter("commander.enable").as_bool();
    RCLCPP_INFO(this->get_logger(), "🔧 Commander enable: %s | Port: %d",
                commander_enabled_ ? "TRUE" : "FALSE", commander_port_);

    RCLCPP_INFO(this->get_logger(), "🔧 takeoff.target_altitude_m: %.1f m",
                this->get_parameter("takeoff.target_altitude_m").as_double());

    auto targets = this->get_parameter("commander.targets").as_string_array();
    std::string targets_str = "[]";
    if (!targets.empty())
    {
      std::ostringstream oss;
      std::copy(targets.begin(), targets.end() - 1,
                std::ostream_iterator<std::string>(oss, ", "));
      oss << targets.back();
      targets_str = "[" + oss.str() + "]";
    }
    RCLCPP_INFO(this->get_logger(), "🔧 commander.targets: %s", targets_str.c_str());

    RCLCPP_INFO(this->get_logger(), "🔧 commander.target_port: %ld",
                this->get_parameter("commander.target_port").as_int());
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

    // === 6. Commander (extracted to initCommander()) ===
    initCommander();

    if (config_.debug)
    {
      sensor_sync_->setDebug(true);
    }

    RCLCPP_INFO(this->get_logger(), "Controllers initialized successfully");
  }

  void Node::initCommander()
  {
    RCLCPP_INFO(this->get_logger(), "Initializing Commander...");

    // === Commander instantiation (handles {"set_mode": "auto"} | nc 11472) ===
    if (commander_enabled_)
    {
      commander_ = std::make_unique<Commander>(
          commander_port_,
          // [this](bool start)
          // {
          //   std::lock_guard<std::mutex> lock(state_mutex_);
          //   start_attack_ = start;
          //   RCLCPP_INFO(this->get_logger(), "🌐 Commander: start_attack_ set to %s", start ? "true" : "false");
          // },

          [this](bool start)
          {
            std::lock_guard<std::mutex> lock(state_mutex_);

            if (start)
            {
              start_attack_ = true;
              attack_enabled_ = true;
              guidance_ctrl_->setAttackMode(true); // 🆕 Явный старт режима в контроллере
              RCLCPP_INFO(this->get_logger(), "🎯 Attack STARTED via TCP command");

              if (auto_phase_ == AutoPhase::VISUAL_SEARCH && attack_ready_.load(std::memory_order_acquire))
              {
                auto_phase_ = AutoPhase::VISUAL_ATTACK;
              }
            }
            else
            {
              start_attack_ = false;
              attack_enabled_ = false;
              guidance_ctrl_->setAttackMode(false); // 🆕 Явный стоп и сброс в контроллере
              guidance_ctrl_->reset();              // Общий сброс

              if (auto_phase_ == AutoPhase::VISUAL_ATTACK)
              {
                auto_phase_ = AutoPhase::VISUAL_SEARCH;
                RCLCPP_INFO(this->get_logger(), "⛔ Attack STOPPED. Returned to VISUAL SEARCH mode");
              }
            }
          },

          // [this](bool start)
          // {
          //   std::lock_guard<std::mutex> lock(state_mutex_);

          //   if (start)
          //   {
          //     // Команда {"start_attack": true}
          //     start_attack_ = true;
          //     attack_enabled_ = true; // Также включаем флаг разрешения атаки
          //     RCLCPP_INFO(this->get_logger(), "🎯 Attack STARTED via TCP command");

          //     // Если мы были в поиске и цель видна, можно сразу перейти в атаку
          //     if (auto_phase_ == AutoPhase::VISUAL_SEARCH && attack_ready_.load(std::memory_order_acquire))
          //     {
          //       auto_phase_ = AutoPhase::VISUAL_ATTACK;
          //       RCLCPP_INFO(this->get_logger(), "🚀 Switched to VISUAL ATTACK immediately");
          //     }
          //   }
          //   else
          //   {
          //     // Команда {"start_attack": false}
          //     start_attack_ = false;
          //     attack_enabled_ = false;
          //     guidance_ctrl_->reset(); // Сброс контроллера наведения

          //     // Переводим в режим поиска, как вы просили
          //     if (auto_phase_ == AutoPhase::VISUAL_ATTACK)
          //     {
          //       auto_phase_ = AutoPhase::VISUAL_SEARCH;
          //       RCLCPP_INFO(this->get_logger(), "⛔ Attack STOPPED. Returned to VISUAL SEARCH mode");
          //     }
          //     else
          //     {
          //       RCLCPP_INFO(this->get_logger(), "⛔ Attack flag cleared. Mode: %s",
          //                   (auto_phase_ == AutoPhase::VISUAL_SEARCH) ? "SEARCH" : "OTHER");
          //     }
          //   }
          // },

          [this]() // ← "set_mode": "auto" calls this → executeAutoModeSequence()
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
          // 🆕 Колбэк для команды HOME/RTL
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
              // 🔽 ДОМАШНЯЯ ТОЧКА НЕ ЗАПИСАНА → выполняем поведение команды LAND
              RCLCPP_WARN(this->get_logger(), "🏠 No home coordinates! Executing LAND maneuver instead.");

              // === Повторяем логику из land_command (с speed) ===
              double land_speed = 1.0; // Значение по умолчанию
              // Если хотите — можно вынести в параметр: this->get_parameter("land.default_speed", land_speed);

              RCLCPP_INFO(this->get_logger(), "🛬 Executing emergency LAND with speed: %.2f m/s", land_speed);

              px4_msgs::msg::VehicleCommand cmd;
              cmd.command = 21;                            // MAV_CMD_NAV_LAND
              cmd.param7 = static_cast<float>(land_speed); // Вертикальная скорость снижения, м/с
              cmd.target_system = 1;
              cmd.target_component = 1;
              cmd.from_external = true;
              cmd.timestamp = this->get_clock()->now().nanoseconds() / 1000;
              vehicle_cmd_pub_->publish(cmd);

              // Также сбрасываем флаги и режим
              active_command_ = ActiveCommand::LAND;
              active_command_param_ = land_speed;
            }
            return hd;
          },
          this->get_logger(),
          // 🆕 Колбэк для команд ARM/DISARM (с прерыванием взлёта)
          [this](bool arm)
          {
            this->sendArmCommand(arm);    // 10x reliable loop
            if (!arm && takeoff_running_) // DISARM abort logic
            {
              RCLCPP_WARN(this->get_logger(), "⚠️ Aborting takeoff due to DISARM");
              takeoff_abort_ = true;
              this->softReset(); //   сброс всех параметров!!!!!!!!
            }
          },
          // 🆕 Колбэк для приземления
          [this](double speed)
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            RCLCPP_INFO(this->get_logger(), "🛬 LAND command received, speed: %.2f m/s", speed);
            active_command_ = ActiveCommand::LAND;
            active_command_param_ = speed;
            // Прерываем атаку и авто режим
            start_attack_ = false;
            attack_enabled_ = false;
            auto_phase_ = AutoPhase::IDLE;
          },
          // 🆕 Колбэк для кругового поиска
          [this](double rotation_time)
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            RCLCPP_INFO(this->get_logger(), "🔍 SEARCH_ROUND command, rotation time: %.2f s", rotation_time);
            active_command_ = ActiveCommand::SEARCH_ROUND;
            active_command_param_ = rotation_time;
            start_attack_ = false;
            attack_enabled_ = false;
            auto_phase_ = AutoPhase::IDLE;
          },
          // 🆕 Колбэк для подъёма
          [this](double altitude)
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            RCLCPP_INFO(this->get_logger(), "⬆️ UP_TO command, altitude: %.2f m", altitude);
            active_command_ = ActiveCommand::UP_TO;
            active_command_param_ = altitude;
            start_attack_ = false;
            attack_enabled_ = false;
            auto_phase_ = AutoPhase::IDLE;
          },
          // 🆕 Колбэк для снижения
          [this](double altitude)
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            RCLCPP_INFO(this->get_logger(), "⬇️ DOWN_TO command, altitude: %.2f m", altitude);
            active_command_ = ActiveCommand::DOWN_TO;
            active_command_param_ = altitude;
            start_attack_ = false;
            attack_enabled_ = false;
            auto_phase_ = AutoPhase::IDLE;
          },
          // 🆕 Колбэк для кругового патрулирования
          [this](double diameter)
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            RCLCPP_INFO(this->get_logger(), "🔄 PATROL_ROUND command, diameter: %.2f m", diameter);
            active_command_ = ActiveCommand::PATROL_ROUND;
            active_command_param_ = diameter;
            start_attack_ = false;
            attack_enabled_ = false;
            auto_phase_ = AutoPhase::IDLE;
          },
          // 🆕 Колбэк для удержания позиции в квадрате
          [this](double square_size)
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            RCLCPP_INFO(this->get_logger(), "⏸️ HOLD_TO_SQUARE command, size: %.2f m", square_size);
            active_command_ = ActiveCommand::HOLD_TO_SQUARE;
            active_command_param_ = square_size;
            start_attack_ = false;
            attack_enabled_ = false;
            auto_phase_ = AutoPhase::IDLE;
          },
          // 🆕 Колбэк для set_mode: search
          [this]()
          {
            std::thread(&Node::executeSearchMode, this).detach();
          },
          // 🎮 Колбэк для ручного управления
          [this](float roll, float pitch, float yaw, float throttle)
          {
            // 🔒 Безопасное сохранение (в отдельном struct или atomic)
            manual_input_.roll = roll;
            manual_input_.pitch = pitch;
            manual_input_.yaw = yaw;
            manual_input_.throttle = throttle;
            manual_input_.updated.store(true, std::memory_order_release);
            uint64_t now_us;
            // Лог с троттлингом
            static uint64_t log_ts = 0;
            now_us = this->get_clock()->now().nanoseconds() / 1000;
            if (now_us - log_ts > 1500000) {  // 500 мс = 500'000 мкс
                log_ts = now_us;
                RCLCPP_INFO(this->get_logger(), "📥 MANUAL CB@@@!!!!!!!: r=%.2f p=%.2f y=%.2f t=%.2f | Phase=%d",
                    manual_input_.roll.load(), manual_input_.pitch.load(),
                    manual_input_.yaw.load(), manual_input_.throttle.load(),
                    static_cast<int>(auto_phase_));
            }

            // === 1. Применение фильтрации и ограничений к ручному управлению ===
            //const auto& cfg = guidance_config_;
            
            const auto& cfg = guidance_config_;
            auto now = std::chrono::steady_clock::now();
            auto dt = std::chrono::duration<float>(now - filtered_manual_input_.last_update).count();
            filtered_manual_input_.last_update = now;

            // Функция для фильтрации с зоной нечувствительности, ограничением скорости и фильтром низких частот
            auto filterAxis = [&](float raw_input, float& filtered_state, float deadzone, float max_slew_rate, float filter_time_constant) -> float {
                // Устранение мертвых зон
                float corrected = std::abs(raw_input) > deadzone ? raw_input : 0.0f;

                // Пропуск через ФНЧ первого порядка
                float alpha = 1.0f;
                if (filter_time_constant > 0.0f && dt > 0.0f) {
                    float tau = filter_time_constant / cfg.ship_mode_response_factor;
                    alpha = dt / (tau + dt);
                }
                float new_filtered = filtered_state + alpha * (corrected - filtered_state);

                // Ограничение скорости изменения (slew rate limiting)
                float max_delta = max_slew_rate * dt;
                float delta = new_filtered - filtered_state;
                if (std::abs(delta) > max_delta) {
                    new_filtered = filtered_state + std::copysign(max_delta, delta);
                }

                return new_filtered;
            };

            // Применение фильтрации к каждой оси
            float filtered_roll = filterAxis(roll, filtered_manual_input_.roll_filtered, 
                                            cfg.manual_roll_deadzone, cfg.max_roll_slew_rate, cfg.manual_filter_time_constant);
            float filtered_pitch = filterAxis(pitch, filtered_manual_input_.pitch_filtered, 
                                             cfg.manual_pitch_deadzone, cfg.max_pitch_slew_rate, cfg.manual_filter_time_constant);
            float filtered_yaw = filterAxis(yaw, filtered_manual_input_.yaw_filtered, 
                                           cfg.manual_yaw_deadzone, cfg.max_yaw_slew_rate, cfg.manual_filter_time_constant);
            
            // Специальная фильтрация для дроссельной заслонки с поддержкой истории для обнаружения всплесков
            float corrected_throttle = std::abs(throttle - 0.5f) > cfg.manual_throttle_deadzone ? throttle : 0.5f;
            float alpha_throttle = 1.0f;
            if (cfg.manual_filter_time_constant > 0.0f && dt > 0.0f) {
                float tau = cfg.manual_filter_time_constant / cfg.ship_mode_response_factor;
                alpha_throttle = dt / (tau + dt);
            }
            float new_throttle_filtered = filtered_manual_input_.throttle_filtered + 
                                          alpha_throttle * (corrected_throttle - filtered_manual_input_.throttle_filtered);

            // Ограничение скорости изменения для газа
            float max_throttle_delta = cfg.max_throttle_slew_rate * dt;
            float throttle_delta = new_throttle_filtered - filtered_manual_input_.throttle_filtered;
            if (std::abs(throttle_delta) > max_throttle_delta) {
                new_throttle_filtered = filtered_manual_input_.throttle_filtered + 
                                        std::copysign(max_throttle_delta, throttle_delta);
            }

            // // 🔒 Блокировка сброса в ноль и резкой перегазовки
            // if (manual_input_.throttle < 0.1f || manual_input_.throttle > 0.9f) {
            //   if ( manual_input_.throttle > 0.9f){
            //     RCLCPP_WARN(this->get_logger(), "🛑 ОПАСНЫЙ ВВОД: Слишком высокий газ, команда игнорируется");
            //   }
            //     new_throttle_filtered = filtered_manual_input_.throttle_filtered; // Сохраняем предыдущее значение
            // } else {
            //     filtered_manual_input_.throttle_filtered = new_throttle_filtered;
            // }

            // Обнаружение опасных всплесков дросселя
            uint64_t now_us_timestamp = now_us;
            filtered_manual_input_.throttle_spikes_history.push_back({now_us_timestamp, new_throttle_filtered});
            // Удаление устаревших записей за пределами окна
            while (!filtered_manual_input_.throttle_spikes_history.empty() && 
                   (now_us_timestamp - filtered_manual_input_.throttle_spikes_history.front().first) > 
                   static_cast<uint64_t>(cfg.spike_filter_time_window * 1e6)) {
                filtered_manual_input_.throttle_spikes_history.pop_front();
            }

            // Проверка на опасные команды (резкие всплески или отключение)
            bool is_throttle_spike_dangerous = cfg.enable_emergency_lockout && 
                !filtered_manual_input_.throttle_spikes_history.empty() && 
                filtered_manual_input_.throttle_spikes_history.back().second > cfg.throttle_spikes_threshold;
            bool is_throttle_cutoff_dangerous = cfg.enable_emergency_lockout && 
                new_throttle_filtered < cfg.throttle_cutoff_threshold;

            if (is_throttle_spike_dangerous || is_throttle_cutoff_dangerous) {
                RCLCPP_WARN(this->get_logger(), "🛑 ОПАСНЫЙ ВВОД: Слишком высокий всплеск или отключение дросселя, команда игнорируется");
                return; // Блокировка опасных команд
            }

            // === 2. Переключение режимов по команде ===
            if (active_command_ == ActiveCommand::VISION_ATTACK && cfg.allow_vision_attack) {
                // Логика переключения в режим визуальной атаки
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (auto_phase_ == AutoPhase::IDLE) {
                    auto_phase_ = AutoPhase::VISUAL_ATTACK;
                    RCLCPP_INFO(this->get_logger(), "🎯 Переключение в режим визуальной атаки по команде");
                }
            }
            else if (active_command_ == ActiveCommand::AUTO_LAND && cfg.allow_autoland) {
                // Логика автопосадки
                RCLCPP_INFO(this->get_logger(), "🛬 Инициирована автопосадка по команде");
                // Здесь должна быть вызвана соответствующая функция посадки
                // executeAutoLandSequence();
            }

            // === 3. Конвертация в команды PX4 ===
            // Roll/Pitch → горизонтальное ускорение (м/с²)
            // Yaw → угловая скорость (рад/с)
            // Throttle → вертикальная скорость (м/с)

            // double max_lat_acc = guidance_config_.max_lat_acc;
            double max_lat_acc = config_.max_lat_acc; // ✅ Берём из NodeConfig
            double max_yaw_rate = 45.0 * M_PI / 180.0; // 45°/с

            Eigen::Vector3d acc_cmd;
            acc_cmd.x() = filtered_pitch * max_lat_acc;                           // pitch → X (North)
            acc_cmd.y() = filtered_roll * max_lat_acc;                            // roll → Y (East)
            //acc_cmd.z() = (0.5f - filtered_manual_input_.throttle_filtered) * guidance_config_.max_down_acc; // throttle → Z (Down)

            acc_cmd.z() = (0.5f - filtered_manual_input_.throttle_filtered) * config_.max_down_acc; // ✅

            float yaw_rate_cmd = filtered_yaw * static_cast<float>(max_yaw_rate);

            // Обновление отфильтрованных значений
            filtered_manual_input_.roll_filtered = filtered_roll;
            filtered_manual_input_.pitch_filtered = filtered_pitch;
            filtered_manual_input_.yaw_filtered = filtered_yaw;


                // RCLCPP_INFO(this->get_logger(), "-----------2: r=%.2f p=%.2f y=%.2f t=%.2f | Phase=%d",
                //     manual_input_.roll.load(), manual_input_.pitch.load(),
                //     manual_input_.yaw.load(), manual_input_.throttle.load(),
                //     static_cast<int>(auto_phase_));
                    
            // === 4. Публикация (только если не в авто-режиме) ===
            // if (auto_phase_ == AutoPhase::IDLE && active_command_ == ActiveCommand::NONE) {}
            
              // Отправляем комбинированный setpoint
              px4_msgs::msg::TrajectorySetpoint ts;
              ts.position = {NAN, NAN, NAN};
              ts.velocity = {NAN, NAN, NAN};
              ts.acceleration = {
                  static_cast<float>(acc_cmd.x()),
                  static_cast<float>(acc_cmd.y()),
                  static_cast<float>(acc_cmd.z())};
              ts.yaw = NAN;
              ts.yawspeed = yaw_rate_cmd;
              ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;

              trajectory_setpoint_pub_->publish(ts);
              vehicle_ctrl_->publishOffboardControlMode();

              //  RCLCPP_INFO(this->get_logger(), "-----------3: r=%.2f p=%.2f y=%.2f t=%.2f | Phase=%d",
              //       manual_input_.roll.load(), manual_input_.pitch.load(),
              //       manual_input_.yaw.load(), manual_input_.throttle.load(),
              //       static_cast<int>(auto_phase_));

              // RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 100,
              //                       "🎮 Manual: r=%.2f p=%.2f y=%.2f t=%.2f → acc=[%.2f,%.2f,%.2f]",
              //                       roll, pitch, yaw, throttle,
              //                       acc_cmd.x(), acc_cmd.y(), acc_cmd.z());
            
          
          }
        
        ); // ← Закрывающая скобка конструктора Commander
        
      commander_->start();
      RCLCPP_INFO(this->get_logger(), "✅ Commander TCP server started on port %d", commander_port_);
    
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "⚠️ Commander disabled (commander.enable=false)");
    }
  }

  void Node::sendArmCommand(bool arm)
  {
    const char *action = arm ? "ARM" : "DISARM";
    RCLCPP_INFO(this->get_logger(), "🔑 Sending %s command (10x loop)...", action);
 
    emergency_stop_ = false;
    // emergency_reset_srv_ = this->create_service<std_srvs::srv::Trigger>(
    // "emergency_reset",
    // std::bind(&Node::emergencyResetService, this,
    //           std::placeholders::_1, std::placeholders::_2));

              
    for (int i = 0; i < 10; i++)
    {
      px4_msgs::msg::VehicleCommand cmd;
      cmd.command = 400; // MAV_CMD_COMPONENT_ARM_DISARM
      cmd.param1 = arm ? 1.0f : 0.0f;
      cmd.target_system = 1;
      cmd.target_component = 1;
      cmd.from_external = true;
      cmd.timestamp = this->get_clock()->now().nanoseconds() / 1000;
      vehicle_cmd_pub_->publish(cmd);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    RCLCPP_INFO(this->get_logger(), "✅ %s sequence complete", action);
  }

  void Node::createPublishers()
  {
    RCLCPP_INFO(this->get_logger(), "Creating publishers...");

    vehicle_cmd_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
        "fmu/in/vehicle_command", 10);

    offboard_ctrl_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
        "fmu/in/offboard_control_mode", 10);

    trajectory_setpoint_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
        "fmu/in/trajectory_setpoint", 10);

    los_status_pub_ = this->create_publisher<px4_msgs::msg::VehicleLosStatus>(
        "fmu/in/vehicle_los_status", 10);

    los_intercept_debug_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "los_intercept_debug", 10);

    RCLCPP_INFO(this->get_logger(), "Publishers created successfully");
  }

  void Node::createSubscribers()
  {
    RCLCPP_INFO(this->get_logger(), "Creating subscribers...");

    // ========================================================================
    // 1. Создаём callback groups
    // ========================================================================

    // 🎯 AI callback group — для трекинга (отдельный поток в MultiThreadedExecutor)
    ai_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    // 🚀 Control callback group — для PX4 telemetry + timer
    control_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    // 🎮 Manual Control callback group — для ручного управления (отдельный поток)
    manual_control_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    // ========================================================================
    // 2. PX4 Subscribers (control group — поток управления)
    // ========================================================================

    // QoS для PX4 сообщений (sensor_data)
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 5), qos_profile);

    auto control_sub_opts = rclcpp::SubscriptionOptions();
    control_sub_opts.callback_group = control_callback_group_;

    vehicle_status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
        "fmu/out/vehicle_status",
        qos,
        std::bind(&Node::vehicleStatusCallback, this, std::placeholders::_1),
        control_sub_opts);

    vehicle_attitude_sub_ = this->create_subscription<px4_msgs::msg::VehicleAttitude>(
        "fmu/out/vehicle_attitude",
        qos,
        std::bind(&Node::vehicleAttitudeCallback, this, std::placeholders::_1),
        control_sub_opts);

    vehicle_angular_velocity_sub_ = this->create_subscription<px4_msgs::msg::VehicleAngularVelocity>(
        "fmu/out/vehicle_angular_velocity",
        qos,
        std::bind(&Node::vehicleAngularVelocityCallback, this, std::placeholders::_1),
        control_sub_opts);

    gps_pos_sub_ = this->create_subscription<px4_msgs::msg::VehicleGlobalPosition>(
        "/fmu/out/vehicle_global_position",
        rclcpp::SensorDataQoS().keep_last(1),
        std::bind(&Node::vehicleGlobalPositionCallback, this, std::placeholders::_1),
        control_sub_opts);

    // ========================================================================
    // 3. AI Subscribers (ai group — отдельный поток)
    // ========================================================================

    auto ai_sub_opts = rclcpp::SubscriptionOptions();
    ai_sub_opts.callback_group = ai_callback_group_;

    tracker_result_sub_ = this->create_subscription<ai_service::msg::TrackerResult>(
        "tracker_result",
        rclcpp::SensorDataQoS().keep_last(3),
        std::bind(&Node::trackerResultCallback, this, std::placeholders::_1),
        ai_sub_opts);

    track_bbox_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
        "track_bbox",
        10,
        std::bind(&Node::trackBBox2DCallback, this, std::placeholders::_1),
        ai_sub_opts);

    // AI Heartbeat (тоже в AI группе)
    heartbeat_sub_ = this->create_subscription<std_msgs::msg::Header>(
        "/ai_heartbeat", 10,
        [this](std_msgs::msg::Header::SharedPtr msg)
        {
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                               "💓 AI heartbeat: %u", msg->stamp.nanosec);
        },
        ai_sub_opts);

    // ========================================================================
    // 4. Signal Subscribers (control group — управляющие сигналы)
    // ========================================================================

    attack_signal_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/attack_signal",
        10,
        std::bind(&Node::attackSignalCallback, this, std::placeholders::_1),
        control_sub_opts);

    RCLCPP_INFO(this->get_logger(), "✅ Subscribers created:");
    RCLCPP_INFO(this->get_logger(), "   🎯 AI group: tracker_result, track_bbox, ai_heartbeat");
    RCLCPP_INFO(this->get_logger(), "   🚀 Control group: PX4 telemetry, attack_signal");
    RCLCPP_INFO(this->get_logger(), "Subscribers created successfully");
  }

  void Node::createServices()
  {
    RCLCPP_INFO(this->get_logger(), "Creating services...");

    auto_attack_srv_ = this->create_service<std_srvs::srv::SetBool>(
        "auto_attack",
        std::bind(&Node::autoAttackService, this,
                  std::placeholders::_1, std::placeholders::_2));

    // 🆕 Сервис сброса аварийной остановки
    emergency_reset_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "emergency_reset",
        std::bind(&Node::emergencyResetService, this,
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

    std::lock_guard<std::mutex> lock(state_mutex_);

    if (use_uart_gps_)
    {
      // Не обновляем lat/lon/alt из PX4
      // drone_gps_.lat = ...
      // drone_gps_.lon = ...
      // drone_gps_.alt = ...

      // Можно логировать, что мы игнорируем PX4 GPS
      // RCLCPP_DEBUG_THROTTLE(..., "Ignoring PX4 GPS position, using UART");
    }
    else
    {
      // 1. Сохраняем в state_mutex_ (потокобезопасно)

      // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 9000, "GPS messsage Subscribers vehicleGlobalPositionCallback");
      drone_gps_.lat = msg->lat;
      drone_gps_.lon = msg->lon;
      drone_gps_.alt = msg->alt;
      drone_gps_.timestamp_us = msg->timestamp_sample;

      // 🛠 FIX: valid_global_pos удалён в новых px4_msgs.
      // Считаем GPS валидным, если пришло сообщение с ненулевым временем.
      drone_gps_.is_valid = (msg->timestamp_sample > 0);

      // 2. Синхронизируем с IMU/attitude через SensorSync
      // 🛠 FIX: msg уже является std::shared_ptr, не нужно оборачивать в make_shared
      if (drone_gps_.is_valid)
      {
        sensor_sync_->addMessage("drone_gps", msg->timestamp_sample, msg);

        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                              "GPS: %.7f, %.7f, %.2f (valid: %s)",
                              msg->lat, msg->lon, msg->alt,
                              drone_gps_.is_valid ? "Y" : "N");
      }
    }
  }

  void Node::uartGpsCallback()
  {
    GpsData uart_data = uart_gps_parser_.getData();

    if (uart_data.is_valid)
    {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "📡 UART GPS: %.6f, %.6f, %.2f (Sats: %d)",
                            uart_data.latitude, uart_data.longitude, uart_data.altitude,
                            uart_data.satellites);
      {

        // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 9000, "GPS messsage Subscribers uartGpsCallback");

        std::lock_guard<std::mutex> lock(state_mutex_);

        // Записываем данные UART в общее состояние дрона
        drone_gps_.lat = uart_data.latitude;
        drone_gps_.lon = uart_data.longitude;
        drone_gps_.alt = uart_data.altitude;
        drone_gps_.is_valid = true;
        drone_gps_.timestamp_us = uart_data.timestamp_us;

        // Опционально: логирование раз в N секунд
        static int log_counter = 0;
        if (++log_counter > 50)
        { // ~1 секунда при 50 Гц
          RCLCPP_DEBUG(this->get_logger(), "📡 UART GPS Updated: %.6f, %.6f, %.2f",
                       drone_gps_.lat, drone_gps_.lon, drone_gps_.alt);
          log_counter = 0;
        }
      }
    }
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
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms, "🔍 trackerResultCallback 11: ref_us=%lu", ref_us);

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
    // quat_ready = true;
    // gyro_ready = true;

    // RCLCPP_INFO(this->get_logger(), "   quat_ready=%d, gyro_ready=%d", quat_ready, gyro_ready);
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms, "   quat_ready=%d, gyro_ready=%d", quat_ready, gyro_ready);

    // === 🪵 Лог синхронизации ===
    if (config_.debug)
    {
      // RCLCPP_DEBUG(this->get_logger(),
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
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
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
                             "🎯 Target: trackerResultCallback 33:  pixel=(%.1f,%.1f)→(%.1f,%.1f), conf=%.3f, valid=%d",
                             target.pixel_u, target.pixel_v,
                             target.pixel_u_smooth, target.pixel_v_smooth,
                             target.confidence,
                             target.is_valid ? 1 : 0);
      }

      last_los_cam_y_.store(target.los_cam.y(), std::memory_order_relaxed); // 🆕
      if (target.is_valid)
      {
        // RCLCPP_INFO(this->get_logger(),
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
                             " trackerResultCallback 44:   ✅ target.valid=true, confidence=%.2f", target.confidence);
        // RCLCPP_INFO(this->get_logger(),
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
                             "   🎯 Setting attack_ready_=true");
      }
      else
      {
        // RCLCPP_WARN(this->get_logger(),
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
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
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
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
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
                               "🚀 Acc cmd: trackerResultCallback 66:  [%.3f, %.3f, %.3f] m/s²",
                               acc_cmd.x(), acc_cmd.y(), acc_cmd.z());
        }

        // 🎯 Публикация статуса для отладки
        publishLosStatus(target, acc_cmd);

        // 🎯 Lock-free сохранение команды для timer callback (без mutex!)
        // ✅ AI callback работает полностью независимо — никакого блокирования
        latest_acc_cmd_.acc_cmd = acc_cmd;
        latest_acc_cmd_.confidence = target.confidence;
        latest_acc_cmd_.timestamp_us = msg->timestamp / 1000; // ns → us
        latest_acc_cmd_.is_valid = true;
        new_acc_cmd_available_.store(true, std::memory_order_release);

        // Атомарно устанавливаем attack_ready (без mutex)
        attack_ready_.store(true, std::memory_order_release);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
                             "✅ AI Thread: attack_ready_=true | conf=%.2f | acc=[%.2f,%.2f,%.2f]",
                             target.confidence,
                             acc_cmd.x(), acc_cmd.y(), acc_cmd.z());
      }
      else
      {
        // === 🪵 Лог если цель не валидна ===
        if (config_.debug)
        {
          // RCLCPP_WARN(this->get_logger(),
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
                               "❌ trackerResultCallback 77:  Target invalid: conf=%.3f < min_confidence=%.3f",
                               target.confidence,
                               tracker_config_.min_confidence);
        }
      }
    }
  }

  // void Node::uartGpsCallback()
  // {
  //   // Определяем, используем ли мы UART GPS как основной источник
  //   // Логика: Если PX4 GPS (drone_gps_) невалиден или старый, пробуем взять данные из UART
  //   // Или можно всегда перезаписывать drone_gps_, если это внешний приемник

  //   // ВАЖНО: Убедитесь, что вы понимаете разницу между GPS на борту PX4 и внешним GPS.
  //   // Обычно PX4 использует свой GPS. Этот код полезен, если:
  //   // 1. PX4 GPS сломан.
  //   // 2. Вы хотите дублировать данные.
  //   // 3. Вы тестируете без PX4 GPS.

  //   GpsData uart_data = uart_gps_parser_.getData();

  //   if (uart_data.is_valid)
  //   {
  //     RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
  //                           "📡 UART GPS: %.6f, %.6f, %.2f (Sats: %d)",
  //                           uart_data.latitude, uart_data.longitude, uart_data.altitude,
  //                           uart_data.satellites);

  //     // Синхронизируем данные с внутренним состоянием дрона
  //     // Используем mutex для безопасности потоков
  //     {
  //       std::lock_guard<std::mutex> lock(state_mutex_);

  //       // Обновляем drone_gps_ только если он невалиден или мы явно хотим использовать UART
  //       // Здесь простая логика: если PX4 GPS не пришел давно, берем UART.
  //       // Для простоты, просто обновляем, если UART валиден.
  //       // ⚠️ ВНИМАНИЕ: Это может конфликтовать с реальным GPS PX4.
  //       // Рекомендуется использовать этот GPS только если PX4 GPS недоступен.

  //       // Пример: Если drone_gps_.is_valid == false, то берем UART
  //       if (!drone_gps_.is_valid)
  //       {
  //         drone_gps_.lat = uart_data.latitude;
  //         drone_gps_.lon = uart_data.longitude;
  //         drone_gps_.alt = uart_data.altitude;
  //         drone_gps_.is_valid = true;
  //         drone_gps_.timestamp_us = uart_data.timestamp_us;

  //         RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
  //                              "🔄 Using UART GPS as fallback");
  //       }
  //     }
  //   }
  // }

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

        // 🎯 Lock-free сохранение (как в trackerResultCallback)
        latest_acc_cmd_.acc_cmd = acc_cmd;
        latest_acc_cmd_.confidence = target.confidence;
        latest_acc_cmd_.timestamp_us = static_cast<uint64_t>(msg->header.stamp.sec * 1e6) +
                                       static_cast<uint64_t>(msg->header.stamp.nanosec * 1e-3);
        latest_acc_cmd_.is_valid = true;
        new_acc_cmd_available_.store(true, std::memory_order_release);
        attack_ready_.store(true, std::memory_order_release);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), config_.log_throttle_interval_ms,
                             "✅ AI Thread (BBox): attack_ready_=true");
      }
    }
  }

  void Node::vehicleStatusCallback(px4_msgs::msg::VehicleStatus::SharedPtr msg)
  {
    static bool first_log = true;
    if (first_log)
    {
      RCLCPP_INFO(this->get_logger(), "✅ First vehicle status received! nav_state=%d, arming_state=%d",
                  msg->nav_state, msg->arming_state);
      first_log = false;
    }

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

    if (start_attack_){
      auto_phase_ = AutoPhase::VISUAL_ATTACK;
    } else {
      auto_phase_ = AutoPhase::IDLE;
    }

    RCLCPP_INFO(this->get_logger(), "Attack signal: %s", start_attack_ ? "START" : "STOP");
  }

  void Node::autoAttackService(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (attack_ready_.load(std::memory_order_acquire))
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

  // 🆕 Сервис сброса аварийной остановки
  void Node::emergencyResetService(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> /* request */,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (emergency_stop_)
    {
      emergency_stop_ = false;
      takeoff_abort_ = false;
      takeoff_running_ = false;
      auto_phase_ = AutoPhase::IDLE;
      attack_enabled_ = false;
      attack_ready_.store(false, std::memory_order_release);
      start_attack_ = false;

      response->success = true;
      response->message = "Emergency stop RESET. System IDLE.";
      RCLCPP_INFO(this->get_logger(), "✅ %s", response->message.c_str());
    }
    else
    {
      response->success = true;
      response->message = "No emergency stop active. System is normal.";
      RCLCPP_INFO(this->get_logger(), "ℹ️ %s", response->message.c_str());
    }
  }

  // ============================================================================
  // Timer Callback (основной цикл)
  // ============================================================================

  void Node::controlLoopTimerCallback()
  {
    // 🚦 Проверка аварийной остановки: блокируем все команды
    if (emergency_stop_)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "🛑 EMERGENCY STOP active — all commands blocked");
      return;
    }


    if (manual_input_.updated.load(std::memory_order_acquire)) {
        RCLCPP_DEBUG(this->get_logger(), "🔄 Manual input consumed: r=%.2f p=%.2f", 
                    manual_input_.roll.load(), manual_input_.pitch.load());
        manual_input_.updated.store(false, std::memory_order_release);
    }


    // 🎮 Обработка ручного управления от джойстика (если включен)
    if (manual_input_.updated.load(std::memory_order_acquire))
    {
      // Конвертация значений джойстика в команды NED
      float vel_x = manual_input_.pitch * max_speed_;  // Вперёд/назад
      float vel_y = manual_input_.roll  * max_speed_;  // Влево/вправо
      float vel_z = (manual_input_.throttle - 0.5f) * max_vert_speed_; // Вверх/вниз (0.5f — центр)
      float yaw_rate = manual_input_.yaw * max_yaw_rate_; // Поворот

      // Логирование команд управления
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 200,
          "🚀 MANUAL SETPOINT: vel=[%.2f,%.2f,%.2f] yaw_rate=%.2f | armed=%s mode=%s",
          vel_x, vel_y, vel_z, yaw_rate,
          vehicle_ctrl_->isArmed() ? "Y" : "N",
          vehicle_ctrl_->flightModeToString(vehicle_ctrl_->getCurrentMode()).c_str());

      // Формирование сообщения для публикации
      px4_msgs::msg::TrajectorySetpoint msg;
      msg.position = {NAN, NAN, NAN}; // Используем velocity control
      msg.velocity = {vel_x, vel_y, vel_z};
      msg.acceleration = {NAN, NAN, NAN};
      msg.yawspeed = yaw_rate;
      msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;

      // Публикация команды управления
      trajectory_setpoint_pub_->publish(msg);
      vehicle_ctrl_->publishOffboardControlMode(); // Обязательно публикуем режим управления

      // Логирование факта публикации
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
          "📡 MANUAL PUBLISHED TrajectorySetpoint: vel=[%.2f,%.2f,%.2f] yawspeed=%.2f",
          msg.velocity[0], msg.velocity[1], msg.velocity[2],
          msg.yawspeed);

      // Сброс флага обновления
      manual_input_.updated.store(false, std::memory_order_release);
    }









    
    // 🆕 АВАРИЙНАЯ ОСТАНОВКА: блокируем ВСЕ команды немедленно
    if (emergency_stop_)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "🛑 EMERGENCY STOP active — all commands blocked");
      return;
    }

    // vehicle_ctrl_->publishOffboardControlMode();
    std::unique_lock<std::mutex> lock(state_mutex_); // блять блочит весь цикл...

    // === ФАЗА 1: СБЛИЖЕНИЕ ПО GPS ===
    if (auto_phase_ == AutoPhase::GPS_APPROACH)
    {

      if (takeoff_abort_)
      {
        handleTakeoffAbort();
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "🛑 EMERGENCY STOP active — all commands blocked");
        for (int i = 0; i < 20; i++) // 20 * 1ms = 20ms
        {

          handleTakeoffAbort();

          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return;
      }

      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 9000, "GPS messsage Subscribers controlLoopTimerCallback");
      // 🆕 Инициализируем ref_gps_ при первом входе (если ещё не установлен)
      if (!ref_gps_valid_ && drone_gps_.is_valid)
      {
        ref_gps_.lat = drone_gps_.lat;
        ref_gps_.lon = drone_gps_.lon;
        ref_gps_.alt = drone_gps_.alt;
        ref_gps_valid_ = true;
        // RCLCPP_INFO(this->get_logger(), "📍 Reference GPS set: %.6f, %.6f, %.2f", ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);

        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                              "📡 UART GPS: %.6f, %.6f, %.2f ",
                              drone_gps_.lat, drone_gps_.lon, drone_gps_.alt);
      }

      if (ref_gps_valid_)
      {
        // Конвертируем цель в локальные NED-координаты относительно точки взлёта
        target_ned_ = gpsToNed(target_gps_.lat, target_gps_.lon, target_gps_.alt,
                               ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);

        // RCLCPP_INFO(this->get_logger(), "📍 Reference GPS get to ned: %.6f, %.6f, %.2f", ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);

        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                              "📡 Reference GPS: %.6f, %.6f, %.2f",
                              drone_gps_.lat, drone_gps_.lon, drone_gps_.alt);
      }

      bool camera_locked = attack_ready_.load(std::memory_order_acquire);
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
          // RCLCPP_INFO(this->get_logger(), "🔍 Scanning all targets for available target...");

          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                               "🔍 Scanning all targets... | count=%zu | port=%d",
                               commander_targets_.size(), target_query_port_);

          std::string found_ip;
          double found_lat = 0, found_lon = 0, found_alt = 0;
          double my_lat = 0;
          {
            std::lock_guard<std::mutex> gps_lock(state_mutex_);
            if (drone_gps_.is_valid)
            {
              my_lat = drone_gps_.lat;
            }
          }

          for (const auto &ip : commander_targets_)
          {
            double t_lat, t_lon, t_alt;
            int t_sats;

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "🔍 Scanning target: %s", ip.c_str());

            if (queryTargetGPS(ip, target_query_port_, t_lat, t_lon, t_alt, t_sats))
            {
              RCLCPP_INFO(this->get_logger(), "✅ Target controlLoopTimerCallback %s responded: lat=%.6f, lon=%.6f, alt=%.2f",
                          ip.c_str(), t_lat, t_lon, t_alt);
              RCLCPP_INFO(this->get_logger(), "✅ Found available target: %s", ip.c_str());
              found_ip = ip;
              found_lat = t_lat;
              found_lon = t_lon;
              found_alt = t_alt;
              break;
            }
          }

          // lock.lock();
          if (!found_ip.empty() && std::abs(my_lat) > 1e-3)
          {
            target_gps_ = {found_lat, found_lon, found_alt};
            closest_target_ip_ = found_ip;
            RCLCPP_INFO(this->get_logger(), "🎯 Target acquired: %s. Starting approach!", found_ip.c_str());
            // ref_gps будет обновлён при следующем входе в callback
          }
          else
          {

            // visual_searching = this->get_parameter("takeoff.visual_searching").as_bool();
            // visual_search_timeout_s = this->get_parameter("takeoff.visual_search_timeout_s").as_int();
          }
          // lock.unlock();
          // lock.lock();
        }
        else if (!closest_target_ip_.empty())
        {
          // 1. Копируем IP и отпускаем мьютекс перед TCP
          std::string ip_to_poll = closest_target_ip_;
          // lock.unlock(); // 🔓 Временно освобождаем

          // 2. TCP вызов (не блокирует мьютекс)
          double t_lat, t_lon, t_alt;
          int t_sats;
          if (queryTargetGPS(ip_to_poll, target_query_port_, t_lat, t_lon, t_alt, t_sats))
          {
            // lock.lock(); // 🔒 Захватываем обратно для записи
            target_gps_ = {t_lat, t_lon, t_alt};
            RCLCPP_INFO(this->get_logger(), "📡 Target GPS updated: %.6f, %.6f", t_lat, t_lon);
            // lock.unlock(); // 🔓 Сразу отпускаем, чтобы не тормозить публикацию
            // lock.lock();   // 🔒 Возвращаем для продолжения работы функции
          }
        }
      }

      // 🔍 Отладочный вывод: текущая цель (IP и координаты)
      if (!closest_target_ip_.empty())
      {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "📍 Current target: IP=%s | GPS: %.6f, %.6f, %.2f m",
                             closest_target_ip_.c_str(),
                             target_gps_.lat,
                             target_gps_.lon,
                             target_gps_.alt);
      }
      else
      {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "📍 No active target selected (closest_target_ip_ is empty)");
      }

      // Условие перехода в атаку: камера видит цель И расстояние < 200м
        if (camera_locked && current_dist < config_.max_approach_distance) // && current_dist < 200.0 // потом добавил условие на дистанцию
        {
          RCLCPP_INFO(this->get_logger(), "🎯 Target acquired! Switching to VISUAL_ATTACK");
          auto_phase_ = AutoPhase::VISUAL_ATTACK;
        start_attack_ = true;
        attack_enabled_ = true;
        RCLCPP_INFO(this->get_logger(), "🎯 VISUAL ATTACK ENGAGED (dist=%.1fm < 200m)", current_dist);
        // Проваливаемся в логику атаки в этом же тике
      }
      else
      {
        // Продолжаем лететь по GPS
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "🎯 go to Target acquired: still approach!");
        vehicle_ctrl_->publishOffboardControlMode();
        // vehicle_ctrl_->publishPositionSetpoint(target_ned_);
        vehicle_ctrl_->publishPositionSetpoint(target_ned_,
                                               Eigen::Vector3d{7.0, 0.0, 0.0}, // скорость
                                               Eigen::Vector3d{2.5, 0.0, 0.0}  // ускорение);
        );

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "🚁 Approaching... Dist: %.1fm | Camera: %s",
                             current_dist, camera_locked ? "LOCKED" : "SEARCHING");
        // без!!! return; здесь. Пусть функция завершится естественно.
      }

      if (closest_target_ip_.empty() &&
          this->get_parameter("takeoff.visual_searching").as_bool())
      {

        RCLCPP_INFO(this->get_logger(),
                    "🔍 No TCP targets available, switching to VISUAL_SEARCH (camera-based detection)");

        auto_phase_ = AutoPhase::VISUAL_SEARCH;

        // Сбрасываем флаги атаки
        start_attack_ = false;
        attack_enabled_ = false;
        attack_ready_.store(false, std::memory_order_release);

        // Сбрасываем контроллер наведения
        guidance_ctrl_->reset();

        return; // Выходим, чтобы в следующем тике сработала логика VISUAL_SEARCH
      }
    }

    // === ФАЗА 2.5: ВИЗУАЛЬНЫЙ ПОИСК (Yaw Tracking + Hover) ===

    // === ФАЗА 2.5: ВИЗУАЛЬНЫЙ ПОИСК (Yaw Tracking + Hover) ===
    if (auto_phase_ == AutoPhase::VISUAL_SEARCH)
    {
      vehicle_ctrl_->publishOffboardControlMode();

      // === 1. Обновляем состояние: видим ли цель? ===
      bool target_seen = attack_ready_.load(std::memory_order_acquire);
      // std::lock_guard<std::mutex> lock(state_mutex_); // пока уберемю. ьтого что заголовке - достаточно

      // === 2. Публикуем setpoint: зависание на месте + yaw control ===
      if (drone_gps_.is_valid && ref_gps_valid_)
      {
        // Конвертируем текущую позицию в NED (для зависания)
        Eigen::Vector3d hover_ned = gpsToNed(
            drone_gps_.lat, drone_gps_.lon, drone_gps_.alt,
            ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);

        // Yaw rate: вращаемся, чтобы держать цель в центре
        double yaw_rate_cmd = 0.0;
        if (target_seen)
        {
          yaw_rate_cmd = -config_.kp_yaw_track * last_los_cam_y_.load(std::memory_order_relaxed);
          yaw_rate_cmd = std::clamp(yaw_rate_cmd, -30.0 * DEG2RAD, 30.0 * DEG2RAD);
        }

        // Публикуем setpoint
        px4_msgs::msg::TrajectorySetpoint msg;
        msg.position = {static_cast<float>(hover_ned.x()), static_cast<float>(hover_ned.y()), static_cast<float>(hover_ned.z())};
        msg.velocity = {NAN, NAN, NAN};
        msg.acceleration = {NAN, NAN, NAN};
        msg.yaw = NAN;
        msg.yawspeed = static_cast<float>(yaw_rate_cmd);
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;

        trajectory_setpoint_pub_->publish(msg);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "🔍 SEARCH: hover at (%.2f, %.2f, %.2f) | yaw_rate=%.1f°/s | target=%s",
                             hover_ned.x(), hover_ned.y(), hover_ned.z(),
                             yaw_rate_cmd * 180.0 / M_PI,
                             target_seen ? "TRACKING" : "SCANNING");
      }
      else
      {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "⚠️ GPS not valid in SEARCH mode — cannot hover");
      }

      // === 3. 🔥 КЛЮЧЕВОЕ: Переход в атаку ТОЛЬКО по команде '{"start_attack": true}' ===
      if (target_seen && start_attack_) // теперь start_attack_ управляет переходом
      {
        RCLCPP_INFO(this->get_logger(), "🎯 'start_attack' received and target seen → switching to VISUAL_ATTACK");
        auto_phase_ = AutoPhase::VISUAL_ATTACK;
        attack_enabled_ = true;
        // ❗ Не вызываем guidance_ctrl_->setAttackMode(true) здесь — пусть будет в timer callback
      }
      else if (target_seen && !start_attack_)
      {
        // Цель видна, но атака не разрешена → продолжаем слежение
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "👀 Target detected but 'start_attack' not received — continuing to track");
      }

      return; // ❗️ Выходим — больше ничего не делаем
    }

    // === ФАЗА 2: ВИЗУАЛЬНАЯ АТАКА (LOS Guidance) ===
    if (auto_phase_ == AutoPhase::VISUAL_ATTACK)
    {
      if (!attack_ready_.load(std::memory_order_acquire))
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

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                             "🔍 TIMER: phase=%d, ready=%d, should_attack=%d",
                             (int)auto_phase_,
                             attack_ready_.load() ? 1 : 0,
                             (start_attack_ || attack_enabled_) ? 1 : 0);

        // 🎯 Читаем команду ускорения из lock-free очереди
        Eigen::Vector3d acc_cmd = guidance_ctrl_->getLastAccelerationCommand(); // default
        if (new_acc_cmd_available_.load(std::memory_order_acquire))
        {
          acc_cmd = latest_acc_cmd_.acc_cmd;
          new_acc_cmd_available_.store(false, std::memory_order_release);
        }

        // Цель видима — выполняем визуальную атаку
        bool should_attack = (start_attack_ || attack_enabled_);
        if (should_attack)
        {

          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                               "----------should_attack");

          if (vehicle_ctrl_->getCurrentMode() == FlightMode::OFFBOARD)
          {

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                                 "----------FlightMode::OFFBOARD");

            update_acc_setpoint_ = true;
            vehicle_ctrl_->publishOffboardControlMode();
            vehicle_ctrl_->publishAccelerationSetpoint(acc_cmd);
          }
          else
          {

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                                 "----------reset");

            guidance_ctrl_->reset();
            vehicle_ctrl_->publishOffboardControlMode();
            vehicle_ctrl_->publishAccelerationSetpoint(acc_cmd);
            vehicle_ctrl_->setOffboardMode();
          }
        }
        else
        {
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                               "----------NOT should_attack");
          if (vehicle_ctrl_->getCurrentMode() == FlightMode::OFFBOARD && update_acc_setpoint_)
          {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                                 "---------- var 2 FlightMode::OFFBOARD");
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
    // if (!attack_ready_)
    //   return;
    bool should_attack = (start_attack_ || attack_enabled_);

    // 🆕 ПРОВЕРКА: Если есть активная команда - выполняем её (приоритет над атакой!)
    if (active_command_ != ActiveCommand::NONE)
    {
      // 🆕 Активная команда прерывает даже атаку!
      should_attack = false; // Принудительно отключаем атаку

      switch (active_command_)
      {
      case ActiveCommand::LAND:
      {
        // Приземление со скоростью N м/с
        double land_speed = active_command_param_;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "🛬 LANDING with speed: %.2f m/s", land_speed);

        // Отправляем команду посадки через PX4
        px4_msgs::msg::VehicleCommand cmd;
        cmd.command = 21;                            // MAV_CMD_NAV_LAND
        cmd.param7 = static_cast<float>(land_speed); // Скорость снижения
        cmd.target_system = 1;
        cmd.target_component = 1;
        cmd.from_external = true;
        cmd.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        vehicle_cmd_pub_->publish(cmd);

        // Если высота < 5м, плавно снижаем скорость
        if (drone_gps_.is_valid)
        {
          double current_alt = drone_gps_.alt;
          if (current_alt < 5.0)
          {
            // Плавно снижаем скорость от land_speed до 0.1 м/с
            double smooth_speed = land_speed * (current_alt / 5.0);
            smooth_speed = std::max(0.1, smooth_speed);
            cmd.param7 = static_cast<float>(smooth_speed);
            vehicle_cmd_pub_->publish(cmd);
          }
        }
        break;
      }

      case ActiveCommand::SEARCH_ROUND:
      {
        // Круговой поиск цели - вращение вокруг оси Z
        double rotation_time = active_command_param_;   // время на 360°
        double yaw_rate = (2.0 * M_PI) / rotation_time; // rad/s

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "🔍 SEARCH_ROUND with yaw_rate: %.2f deg/s", yaw_rate * 180.0 / M_PI);

        // Устанавливаем offboard режим и публикуем yaw_rate
        if (vehicle_ctrl_->getCurrentMode() != FlightMode::OFFBOARD)
        {
          vehicle_ctrl_->setOffboardMode();
        }

        px4_msgs::msg::TrajectorySetpoint msg;
        msg.position = {NAN, NAN, NAN};
        msg.velocity = {NAN, NAN, NAN};
        msg.acceleration = {NAN, NAN, NAN};
        msg.yaw = NAN;
        msg.yawspeed = static_cast<float>(yaw_rate);
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        trajectory_setpoint_pub_->publish(msg);
        break;
      }

      case ActiveCommand::UP_TO:
      {
        // Подъём на N метров
        double target_altitude = active_command_param_;

        if (drone_gps_.is_valid)
        {
          double current_alt = drone_gps_.alt;
          double altitude_diff = target_altitude - current_alt;

          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                               "⬆️ UP_TO: current=%.1f, target=%.1f, diff=%.1f",
                               current_alt, target_altitude, altitude_diff);

          // Если ещё нужно подняться
          if (altitude_diff > 0.5)
          {
            // Устанавливаем вертикальную скорость вверх
            px4_msgs::msg::TrajectorySetpoint msg;
            msg.position = {NAN, NAN, NAN};
            msg.velocity = {0.0f, 0.0f, -2.0f}; // NED: negative = up
            msg.acceleration = {NAN, NAN, NAN};
            msg.yaw = NAN;
            msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
            trajectory_setpoint_pub_->publish(msg);
          }
          else
          {
            // Достигли цели - стабилизируем позицию
            RCLCPP_INFO(this->get_logger(), "✅ UP_TO target reached");
            active_command_ = ActiveCommand::NONE;
          }
        }
        break;
      }

      case ActiveCommand::DOWN_TO:
      {
        // Снижение на N метров
        double target_altitude = active_command_param_;

        if (drone_gps_.is_valid)
        {
          double current_alt = drone_gps_.alt;
          double altitude_diff = current_alt - target_altitude;

          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                               "⬇️ DOWN_TO: current=%.1f, target=%.1f, diff=%.1f",
                               current_alt, target_altitude, altitude_diff);

          // Если ещё нужно снизиться
          if (altitude_diff > 0.5)
          {
            // Устанавливаем вертикальную скорость вниз
            px4_msgs::msg::TrajectorySetpoint msg;
            msg.position = {NAN, NAN, NAN};
            msg.velocity = {0.0f, 0.0f, 1.0f}; // NED: positive = down
            msg.acceleration = {NAN, NAN, NAN};
            msg.yaw = NAN;
            msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
            trajectory_setpoint_pub_->publish(msg);
          }
          else
          {
            // Достигли цели - стабилизируем позицию
            RCLCPP_INFO(this->get_logger(), "✅ DOWN_TO target reached");
            active_command_ = ActiveCommand::NONE;
          }
        }
        break;
      }

      case ActiveCommand::PATROL_ROUND:
      {
        // Движение по окружности диаметром N метров
        double diameter = active_command_param_;
        double radius = diameter / 2.0;

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "🔄 PATROL_ROUND with diameter: %.1f m", diameter);

        // Вычисляем позицию на окружности на основе времени
        auto now = this->get_clock()->now();
        double time_sec = now.seconds();
        double angular_speed = 0.5; // rad/s (можно настроить)
        double angle = fmod(time_sec * angular_speed, 2.0 * M_PI);

        if (drone_gps_.is_valid)
        {
          // Текущая позиция - центр окружности
          double center_lat = drone_gps_.lat;
          double center_lon = drone_gps_.lon;

          // Позиция на окружности
          double target_lat = center_lat + (radius * std::cos(angle)) / 111320.0; // Convert m to deg
          double target_lon = center_lon + (radius * std::sin(angle)) / (111320.0 * std::cos(center_lat * M_PI / 180.0));

          // Конвертируем в NED
          Eigen::Vector3d target_pos = gpsToNed(target_lat, target_lon, drone_gps_.alt,
                                                ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);

          // Публикуем позицию
          if (!ref_gps_valid_)
          {
            ref_gps_.lat = drone_gps_.lat;
            ref_gps_.lon = drone_gps_.lon;
            ref_gps_.alt = drone_gps_.alt;
            ref_gps_valid_ = true;
          }

          vehicle_ctrl_->publishPositionSetpoint(target_pos,
                                                 Eigen::Vector3d{2.5, 0.0, 0.0}, Eigen::Vector3d{0.1, 0.0, 0.0});
        }
        break;
      }

      case ActiveCommand::HOLD_TO_SQUARE:
      {
        double square_size = active_command_param_; // напр. 20 м
        double half_size = square_size / 2.0;       // ±10 м

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "⏸️ HOLD_TO_SQUARE: drifting within %.1f m", half_size);

        if (drone_gps_.is_valid)
        {
          if (!ref_gps_valid_)
          {
            ref_gps_.lat = drone_gps_.lat;
            ref_gps_.lon = drone_gps_.lon;
            ref_gps_.alt = drone_gps_.alt;
            ref_gps_valid_ = true;
          }

          // Генерируем случайные смещения в пределах квадрата
          static std::random_device rd;
          static std::mt19937 gen(rd());
          static std::uniform_real_distribution<double> dis(-half_size, half_size);

          double offset_x = dis(gen);
          double offset_y = dis(gen);

          // Базовая позиция — текущая в NED
          Eigen::Vector3d center_ned = gpsToNed(
              drone_gps_.lat, drone_gps_.lon, drone_gps_.alt,
              ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);

          Eigen::Vector3d target_ned = center_ned;
          target_ned.x() += offset_x;
          target_ned.y() += offset_y;

          vehicle_ctrl_->publishPositionSetpoint(target_ned,
                                                 Eigen::Vector3d{2.5, 0.0, 0.0}, Eigen::Vector3d{0.1, 0.0, 0.0});

          RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                "🎯 Drift target: offset=(%.1f, %.1f), NED=(%.2f, %.2f, %.2f)",
                                offset_x, offset_y, target_ned.x(), target_ned.y(), target_ned.z());
        }
        break;
      }

      case ActiveCommand::NONE:
      default:
        break;
      }

      // После выполнения команды НЕ выходим из callback,
      // чтобы ниже могла сработать логика атаки если нужно
    }

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
    {
      RCLCPP_INFO(this->get_logger(), "🚫 Failed to create socket for GPS query to %s:%d", ip.c_str(), port);
      return false;
    }

    // ✅ Правильная инициализация структуры
    struct sockaddr_in addr
    {
    };
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0)
    {
      RCLCPP_INFO(this->get_logger(), "🚫 Invalid IP address for GPS query: %s", ip.c_str());
      close(sock);
      return false;
    }

    // ✅ Увеличил таймаут до 500 мс (200 мс мало для сети)
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 500000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
      RCLCPP_INFO(this->get_logger(), "🚫 Failed to connect to target for GPS query: %s:%d", ip.c_str(), port);
      close(sock);
      return false;
    }

    // ✅ Отправка с проверкой
    std::string req = "{\"showgps\": true}\n";
    ssize_t sent = send(sock, req.c_str(), req.size(), 0);
    if (sent < 0 || sent != static_cast<ssize_t>(req.size()))
    {
      close(sock);
      return false;
    }

    // ✅ Чтение ответа
    char buf[512] = {0};
    int n = recv(sock, buf, sizeof(buf) - 1, 0);
    close(sock);
    if (n <= 0)
    {
      RCLCPP_INFO(this->get_logger(), "🚫 No response received from target during GPS query: %s:%d", ip.c_str(), port);
      return false;
    }
    buf[n] = '\0';

    // ✅ Парсер с пропуском пробелов
    auto findVal = [&](const char *key) -> double
    {
      std::string s(buf);
      size_t pos = s.find(key);
      if (pos == std::string::npos)
        return 0.0;
      pos = s.find(':', pos);
      if (pos == std::string::npos)
        return 0.0;
      pos++;
      while (pos < s.size() && std::isspace(s[pos]))
        pos++;
      try
      {
        return std::stod(s.substr(pos));
      }
      catch (...)
      {
        return 0.0;
      }
    };

    std::string data(buf);
    if (data.find("\"status\"") == std::string::npos || data.find("success") == std::string::npos)
    {
      RCLCPP_INFO(this->get_logger(), "🚫 Target responded with failure status during GPS query: %s:%d", ip.c_str(), port);
      return false;
    }

    out_lat = findVal("\"lat\"");
    out_lon = findVal("\"lon\"");
    out_alt = findVal("\"alt\"");
    out_sats = static_cast<int>(findVal("\"sats\""));

    // ✅ Валидация: sats опционален, проверяем только координаты
    bool has_valid_coords = (std::abs(out_lat) > 1e-3 && std::abs(out_lon) > 1e-3);
    if (!has_valid_coords)
    {
      RCLCPP_INFO(this->get_logger(), "🚫 Target returned invalid GPS coordinates: lat=%.6f, lon=%.6f, alt=%.2f, sats=%d",
                  out_lat, out_lon, out_alt, out_sats);
    }
    return has_valid_coords;
  }

  // // === TCP-клиент для опроса целей ===
  // bool Node::queryTargetGPS(const std::string &ip, int port, double &out_lat, double &out_lon, double &out_alt, int &out_sats)
  // {
  //   int sock = socket(AF_INET, SOCK_STREAM, 0);
  //   if (sock < 0)
  //     return false;

  //   struct sockaddr_in addr
  //   {
  //   };
  //   addr.sin_family = AF_INET;
  //   addr.sin_port = htons(port);
  //   if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0)
  //   {
  //     close(sock);
  //     return false;
  //   }

  //   struct timeval tv;
  //   tv.tv_sec = 0;
  //   tv.tv_usec = 200000; // 200 мс вместо 2 с
  //   setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  //   setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  //   if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)
  //   {
  //     close(sock);
  //     return false;
  //   }

  //   std::string req = "{\"showgps\": true}\n";
  //   send(sock, req.c_str(), req.size(), 0);
  //   char buf[512] = {0};
  //   int n = recv(sock, buf, sizeof(buf) - 1, 0);
  //   close(sock);
  //   if (n <= 0)
  //     return false;
  //   buf[n] = '\0';

  //   // Лёгкий парсер без зависимостей
  //   auto findVal = [&](const char *key) -> double
  //   {
  //     std::string s(buf);
  //     size_t pos = s.find(key);
  //     if (pos == std::string::npos)
  //       return 0.0;
  //     pos = s.find(':', pos);
  //     if (pos == std::string::npos)
  //       return 0.0;
  //     try
  //     {
  //       return std::stod(s.c_str() + pos + 1);
  //     }
  //     // catch (...)
  //     // {
  //     //   return 0.0;
  //     // }
  //     catch (const std::exception &e)
  //     {
  //       // RCLCPP_WARN(logger_, "⚠️ Parse error for key '%s': %s", key, e.what());
  //       std::cerr << "[WARN] Parse error for key in json " << key << "  !!errors\n";
  //       return 0.0;
  //     }
  //   };

  //   // if (std::string(buf).find("\"status\":\"success\"") == std::string::npos)
  //   //   return false;

  //   std::string data(buf);
  //   if (data.find("\"status\"") == std::string::npos || data.find("success") == std::string::npos)
  //   {
  //     return false;
  //   }

  //   out_lat = findVal("\"lat\"");
  //   out_lon = findVal("\"lon\"");
  //   out_alt = findVal("\"alt\"");
  //   out_sats = (int)findVal("\"sats\"");

  //   //v1 with sats
  //   //return (out_sats > 0 && std::abs(out_lat) > 1e-3);

  //   //v2
  //   // std::string response(buf);
  //   // bool has_sats = response.find("\"sats\"") != std::string::npos;
  //   // if (has_sats && out_sats <= 0) return false;
  //   // return std::abs(out_lat) > 1e-3 && std::abs(out_lon) > 1e-3;

  //   //v3
  //   bool has_valid_coords = (std::abs(out_lat) > 1e-3 && std::abs(out_lon) > 1e-3);
  //   return has_valid_coords;
  // }

  // === Основная авто-последовательность ===

  // 🆕 Обработка прерывания взлёта: логирование, переключение в ручной режим, DISARM
  void Node::handleTakeoffAbort()
  {
    RCLCPP_INFO(this->get_logger(), "🛑 Takeoff aborted by command");
    takeoff_running_ = false;

    // 🆕 БЛОКИРУЕМ основной цикл НЕМЕДЛЕННО
    emergency_stop_ = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Даём основному циклу заметить флаг

    // 🆕 Переводим в ручной режим (однократно, не в цикле)
    RCLCPP_INFO(this->get_logger(), "📻 Switching to MANUAL mode...");
    for (int i = 0; i < 20; i++)
    {
      vehicle_ctrl_->setManualMode();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // 🆕 Отправляем DISARM
    RCLCPP_INFO(this->get_logger(), "🔓 Sending DISARM command...");
    for (int i = 0; i < 20; i++)
    {
      px4_msgs::msg::VehicleCommand disarm_cmd;
      disarm_cmd.command = 400; // MAV_CMD_COMPONENT_ARM_DISARM
      disarm_cmd.param1 = 0.0f; // 0 = disarm
      disarm_cmd.target_system = 1;
      disarm_cmd.target_component = 1;
      disarm_cmd.from_external = true;
      disarm_cmd.timestamp = this->get_clock()->now().nanoseconds() / 1000;
      vehicle_cmd_pub_->publish(disarm_cmd);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // 🆕 Публикуем NAN setpoint чтобы PX4 остановился
    RCLCPP_INFO(this->get_logger(), "🚫 Publishing NAN setpoint to stop PX4...");
    for (int i = 0; i < 20; i++)
    {
      px4_msgs::msg::TrajectorySetpoint ts;
      ts.position = {NAN, NAN, NAN};
      ts.velocity = {NAN, NAN, NAN};
      ts.acceleration = {NAN, NAN, NAN};
      ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
      trajectory_setpoint_pub_->publish(ts);
      vehicle_ctrl_->publishOffboardControlMode();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    RCLCPP_INFO(this->get_logger(), "✅ Takeoff abort complete. Emergency stop ACTIVE.");
    RCLCPP_INFO(this->get_logger(), "🔄 Call /emergency_reset service to resume operations");
  }

  void Node::executeAutoModeSequence()
  {
    RCLCPP_INFO(this->get_logger(), "🚀 Starting AUTO mode sequence...");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    bool visual_searching = false;
    int visual_search_timeout_s = 600;

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
    sendArmCommand(true);
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

    bool skip_takeoff = this->get_parameter("takeoff.skip").as_bool();
    int takeoff_timeout_s = this->get_parameter("takeoff.timeout_s").as_int();

    RCLCPP_INFO(this->get_logger(), "🔧 !!!!!!!!!!!!! skip_takeoff = : %s ",
                skip_takeoff ? "TRUE" : "FALSE");
    RCLCPP_INFO(this->get_logger(), "🔧 !!!!!!!!!!!!! debug = : %s ",
                config_.debug ? "TRUE" : "FALSE");

    if (skip_takeoff)
    {
      RCLCPP_INFO(this->get_logger(), "⏩ Skipping takeoff phase, going straight to target search");

      // 🆕 Устанавливаем ref_gps_ на текущую позицию
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (drone_gps_.is_valid)
        {
          ref_gps_.lat = drone_gps_.lat;
          ref_gps_.lon = drone_gps_.lon;
          ref_gps_.alt = drone_gps_.alt;
          ref_gps_valid_ = true;
          target_gps_ = {drone_gps_.lat, drone_gps_.lon, drone_gps_.alt}; // hover на месте
          RCLCPP_INFO(this->get_logger(), "📍 Hover GPS:  \n  %.6f, \n %.6f, \n %.2f",
                      drone_gps_.lat, drone_gps_.lon, drone_gps_.alt);
        }
        else
        {
          RCLCPP_WARN(this->get_logger(), "⚠️ GPS invalid at takeoff skip, using NAN for hover");
          ref_gps_valid_ = false;
          target_gps_ = {0, 0, 0};
        }
        auto_phase_ = AutoPhase::GPS_APPROACH;
        start_attack_ = false;
        attack_enabled_ = false;
        attack_ready_.store(false, std::memory_order_release);
      }

      // 🆕 Публикуем hover setpoint многократно
      for (int i = 0; i < 10; i++)
      {
        px4_msgs::msg::TrajectorySetpoint hover_ts;
        hover_ts.position = {NAN, NAN, NAN};
        hover_ts.velocity = {NAN, NAN, NAN};
        hover_ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        trajectory_setpoint_pub_->publish(hover_ts);
        vehicle_ctrl_->publishOffboardControlMode();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }

      // 🆕 Переходим к опросу целей (poll targets) — как после взлёта
      // Проваливаемся в код ниже
    }
    else
    {

      float target_z = -static_cast<float>(this->get_parameter("takeoff.target_altitude_m").as_double());
      float vz = static_cast<float>(this->get_parameter("takeoff.climb_rate_mps").as_double());
      float altitude_threshold = static_cast<float>(this->get_parameter("takeoff.altitude_threshold_m").as_double());
      float min_alt_already_flying = static_cast<float>(this->get_parameter("takeoff.min_altitude_already_flying_m").as_double());
      int min_cycles_already_flying = static_cast<int>(this->get_parameter("takeoff.min_cycles_for_already_flying").as_int());

      RCLCPP_INFO(this->get_logger(), "🛫 Takeoff: target=%.1fm, climb=%.1fm/s, threshold=%.1fm, timeout=%ds",
                  -target_z, vz, altitude_threshold, takeoff_timeout_s);

      // 🆕 Флаги для прерывания взлёта командами
      takeoff_running_ = true;
      takeoff_abort_ = false;

      int cycles = 0;
      auto takeoff_start_time = std::chrono::steady_clock::now();

      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (drone_gps_.is_valid)
          start_alt = drone_gps_.alt;
      }

      while (cycles < 1200) // 1200
      {
        // 🆕 Лог статуса на каждой 500-й итерации
        if (cycles > 0 && cycles % 500 == 0)
        {
          RCLCPP_INFO(this->get_logger(), "🔄 -------- Takeoff cycle %d: takeoff_abort_=%s, takeoff_running_=%s",
                      cycles,
                      takeoff_abort_ ? "true" : "false",
                      takeoff_running_ ? "true" : "false");
        }

        if (takeoff_abort_)
        {
          handleTakeoffAbort();
          return;
        }

        // 🆕 Проверка флага прерывания (команда извне)
        if (takeoff_abort_)
        {
          return;
          RCLCPP_INFO(this->get_logger(), "1!!!!!!!!!!!!!!----------🛑 Takeoff aborted by command");
          takeoff_running_ = false;

          // 🆕 Переводим дрон в режим зависания на текущей позиции
          // и немедленно публикуем setpoint чтобы PX4 перестал набирать высоту
          double cur_lat = 0, cur_lon = 0, cur_alt = 0;
          bool gps_valid = false;
          {
            std::lock_guard<std::mutex> lock(state_mutex_);
            RCLCPP_INFO(this->get_logger(), "🔍 Abort check: drone_gps_.is_valid=%d, start_alt=%.2f",
                        drone_gps_.is_valid ? 1 : 0, start_alt);
            if (drone_gps_.is_valid && !std::isnan(start_alt))
            {
              target_gps_ = {drone_gps_.lat, drone_gps_.lon, drone_gps_.alt};
              cur_lat = drone_gps_.lat;
              cur_lon = drone_gps_.lon;
              cur_alt = drone_gps_.alt;
              gps_valid = true;

              // 🆕 Устанавливаем ref_gps_ чтобы controlLoopTimerCallback мог публиковать setpoint
              ref_gps_.lat = drone_gps_.lat;
              ref_gps_.lon = drone_gps_.lon;
              ref_gps_.alt = drone_gps_.alt;
              ref_gps_valid_ = true;
            }
            auto_phase_ = AutoPhase::GPS_APPROACH;
            start_attack_ = false;
            attack_enabled_ = false;
            attack_ready_.store(false, std::memory_order_release);
          }

          if (gps_valid && !std::isnan(start_alt))
          {
            double current_ned_z = -(cur_alt - start_alt);
            RCLCPP_INFO(this->get_logger(), "🛑 Publishing hover setpoint at z=%.2f (NED), GPS: %.6f, %.6f, %.2f",
                        current_ned_z, cur_lat, cur_lon, cur_alt);

            // 🆕 Отправляем setpoint многократно чтобы PX4 точно принял
            for (int i = 0; i < 10; i++)
            {
              px4_msgs::msg::TrajectorySetpoint hover_ts;
              hover_ts.position = {NAN, NAN, static_cast<float>(current_ned_z)};
              hover_ts.velocity = {NAN, NAN, 0.0f};
              hover_ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
              trajectory_setpoint_pub_->publish(hover_ts);
              vehicle_ctrl_->publishOffboardControlMode();
              std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
          }
          else
          {
            RCLCPP_WARN(this->get_logger(), "⚠️ GPS invalid during abort (gps_valid=%d, start_alt=%.2f), publishing NAN hover",
                        gps_valid ? 1 : 0, start_alt);
            // 🆕 Отправляем NAN setpoint многократно
            for (int i = 0; i < 10; i++)
            {
              px4_msgs::msg::TrajectorySetpoint hover_ts;
              hover_ts.position = {NAN, NAN, NAN};
              hover_ts.velocity = {NAN, NAN, NAN};
              hover_ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
              trajectory_setpoint_pub_->publish(hover_ts);
              vehicle_ctrl_->publishOffboardControlMode();
              std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
          }
          return;
        }

        // 🆕 Проверка таймаута взлёта
        auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - takeoff_start_time)
                             .count();
        if (elapsed_s >= takeoff_timeout_s)
        {
          RCLCPP_WARN(this->get_logger(), "⏱️ Takeoff timeout: %ds elapsed, target_alt=%.1fm, current_alt=%.1fm. Proceeding anyway.",
                      static_cast<int>(elapsed_s), -target_z,
                      drone_gps_.is_valid ? drone_gps_.alt : NAN);
          break;
        }

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
            if (drone_gps_.alt >= start_alt + altitude_threshold)
            {
              RCLCPP_INFO(this->get_logger(), "✅ Altitude reached.");
              break;
            }
            if (drone_gps_.alt >= start_alt + min_alt_already_flying && cycles > min_cycles_already_flying)
            {
              RCLCPP_INFO(this->get_logger(), "✅ Already flying.");
              break;
            }
          }
        }

        // 🆕 Короткий сон с проверкой прерывания (вместо одного длинного sleep)
        for (int i = 0; i < 20; i++) // 20 * 1ms = 20ms
        {
          if (takeoff_abort_)
          {
            handleTakeoffAbort();
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cycles++;
      }

      takeoff_running_ = false;

      // 🆕 Если высота не достигнута — всё равно продолжаем к poll targets
      // и переходим в режим детекции/удержания цели
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (drone_gps_.is_valid && !std::isnan(start_alt))
        {
          double current_alt = drone_gps_.alt;
          if (current_alt < start_alt + altitude_threshold)
          {
            RCLCPP_WARN(this->get_logger(),
                        "⚠️ Target altitude not reached (alt=%.1f, target=%.1f). Proceeding to detection mode anyway.",
                        current_alt, -target_z);
          }
        }
      }
    } // end of else (skip_takeoff == false)

    RCLCPP_WARN(this->get_logger(), "step 1");

    // 🆕 Если ref_gps_ не установлен (skip_takeoff=true), но GPS валиден — устанавливаем
    {
              
        // 🆕 Если ref_gps_ не установлен, но GPS валиден — устанавливаем
        if (!ref_gps_valid_ && drone_gps_.is_valid) {
            // Пробуем захватить без блокировки
            std::unique_lock<std::mutex> lock(state_mutex_, std::try_to_lock);
            if (lock.owns_lock()) {
                // Double-checked locking: повторная проверка после захвата
                if (!ref_gps_valid_ && drone_gps_.is_valid) {
                    ref_gps_.lat = drone_gps_.lat;
                    ref_gps_.lon = drone_gps_.lon;
                    ref_gps_.alt = drone_gps_.alt;
                    ref_gps_valid_ = true;
                    RCLCPP_INFO(this->get_logger(), "📍 Reference GPS set: %.6f, %.6f, %.2f",
                              ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);
                }
            }
            // Если не удалось захватить — пропустим, в следующий раз попробуем снова
            // (это безопасно, т.к. ref_gps_valid_ всё ещё false)
        }
    }

    RCLCPP_WARN(this->get_logger(), "step 2");

    // 4. Poll targets
    if (commander_targets_.empty())
    {
      RCLCPP_WARN(this->get_logger(), "⚠️ No targets. Aborting.");
      // visual_searching = true; // может так??
      // return;
    }

    RCLCPP_INFO(this->get_logger(), "📡 Polling %zu targets...", commander_targets_.size());
    std::string closest_ip;
    RCLCPP_INFO(this->get_logger(), "🔍 check: drone_gps_.lat =%d, start_alt=%.2f",
                drone_gps_.is_valid ? 1 : 0, drone_gps_.lat);
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
        RCLCPP_INFO(this->get_logger(), "✅ Target AutomodeSec %s responded: lat=%.6f, lon=%.6f, alt=%.2f",
                    ip.c_str(), t_lat, t_lon, t_alt);
        RCLCPP_INFO(this->get_logger(), "✅ Target AutomodeSec %s responded: drone_lat=%.6f, drone_lon=%.6f",
                    ip.c_str(), drone_lat, drone_lon);
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

            RCLCPP_INFO(this->get_logger(), "✅ Closest target AutomodeSec %s responded: lat=%.6f, lon=%.6f, alt=%.2f",
                        ip.c_str(), t_lat, t_lon, t_alt);
          }
        }
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "❌ Timeout/invalid response from %s", ip.c_str());
      }
    }

    if (closest_ip.empty())
    {

      visual_searching = this->get_parameter("takeoff.visual_searching").as_bool();
      visual_search_timeout_s = this->get_parameter("takeoff.visual_search_timeout_s").as_int();

      // 🆕 Визуальный поиск цели после взлёта (висим на месте, ждём обнаружения камерой)

      RCLCPP_INFO(this->get_logger(), " --- visual_searching %s", visual_searching ? "TRUE" : "FALSE");
    }
    else
    {

      Eigen::Vector3d approach_velocity{5.0, 0.0, 0.0}; // пример значений
      Eigen::Vector3d approach_acceleration{1.2, 0.0, 0.0};
      constexpr double BOOST_SPEED_MAX = 15.0; // константа // Макс. скорость при рывке [м/с]
      // constexpr double turbo_start_dist = 100.0;
      // constexpr double turbo_end_dist = 30.0;

      for (int i = 0; i < 100; i++)
      {
        // Вычисление желаемого yaw (угол от дрона к цели)
        double target_yaw = NAN;
        double dx = target_ned_.x(); // North
        double dy = target_ned_.y(); // East
        if (std::abs(dx) > 0.5 || std::abs(dy) > 0.5)
        {
          target_yaw = std::atan2(dy, dx); // Радианы, диапазон [-π, π]
        }

        // Если нужно передать yaw — модифицируйте publishPositionSetpoint
        // или публикуйте полный TrajectorySetpoint вручную:
        px4_msgs::msg::TrajectorySetpoint msg;
        msg.position = {static_cast<float>(target_ned_.x()),
                        static_cast<float>(target_ned_.y()),
                        static_cast<float>(target_ned_.z())};
        msg.velocity[0] = static_cast<float>(approach_velocity.x());
        msg.velocity[1] = static_cast<float>(approach_velocity.y());
        msg.velocity[2] = static_cast<float>(approach_velocity.z());
        msg.acceleration[0] = static_cast<float>(approach_acceleration.x());
        msg.acceleration[1] = static_cast<float>(approach_acceleration.y());
        msg.acceleration[2] = static_cast<float>(approach_acceleration.z());
        msg.yaw = std::isnan(target_yaw) ? NAN : static_cast<float>(target_yaw);
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        trajectory_setpoint_pub_->publish(msg);
      }

      // === Плавный турбо-режим (ramp-up acceleration) ===
      double turbo_start_distF = 30.0; // Начать ускорение с 30м
      double turbo_end_distF = 5.0;    // Закончить на 5м

      if (min_dist <= turbo_start_distF && min_dist >= turbo_end_distF)
      {
        // Коэффициент усиления: 1.0 → 3.0 по мере сближения
        double turbo_factor = 1.0 + 2.0 * (turbo_start_distF - min_dist) /
                                        (turbo_start_distF - turbo_end_distF);

        Eigen::Vector3d dir_to_target(target_ned_.x(), target_ned_.y(), 0.0);
        dir_to_target.normalize();

        // Плавно наращиваем ускорение
        Eigen::Vector3d turbo_acceleration = dir_to_target *
                                             (guidance_config_.max_lat_acc * turbo_factor);

        // Ограничение по скорости
        double turbo_speed = std::min(
            (guidance_config_.max_vc * 0.5) * turbo_factor,
            BOOST_SPEED_MAX);
        Eigen::Vector3d turbo_velocity = dir_to_target * turbo_speed;

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "⚡ Turbo mode: factor=%.1f | acc=%.2f | speed=%.1f",
                             turbo_factor, turbo_acceleration.norm(), turbo_speed);

        // Публикация (в основном цикле, не в отдельном while)
        px4_msgs::msg::TrajectorySetpoint msg;
        msg.position = {static_cast<float>(target_ned_.x()),
                        static_cast<float>(target_ned_.y()),
                        static_cast<float>(target_ned_.z())};
        msg.velocity = {static_cast<float>(turbo_velocity.x()),
                        static_cast<float>(turbo_velocity.y()),
                        static_cast<float>(turbo_velocity.z())};
        msg.acceleration = {static_cast<float>(turbo_acceleration.x()),
                            static_cast<float>(turbo_acceleration.y()),
                            static_cast<float>(turbo_acceleration.z())};
        msg.yaw = std::atan2(target_ned_.y(), target_ned_.x());
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        trajectory_setpoint_pub_->publish(msg);
      }

      // === ФАЗА: МОЩНЫЙ РЫВОК К ЦЕЛИ (Sprint Boost) ===
      // Параметры рывка (можно вынести в params.yaml)
      constexpr double BOOST_DURATION_SEC = 3.0; // Длительность рывка [с]
      constexpr double BOOST_ACCEL_FACTOR = 3.0; // Множитель ускорения (×3 от обычного)

      constexpr double MIN_BOOST_DIST = 10.0; // Мин. дистанция для запуска рывка [м]
      constexpr double MAX_BOOST_DIST = 50.0; // Макс. дистанция для запуска рывка [м]

      // Проверяем условия для рывка
      double current_dist = std::sqrt(
          target_ned_.x() * target_ned_.x() +
          target_ned_.y() * target_ned_.y());

      if (current_dist >= MIN_BOOST_DIST && current_dist <= MAX_BOOST_DIST)
      {
        // 🚀 Вычисляем единичный вектор направления к цели
        Eigen::Vector3d dir_to_target(target_ned_.x(), target_ned_.y(), 0.0);
        dir_to_target.normalize(); // Горизонтальный рывок (без изменения высоты)

        // 🚀 Усиленное ускорение в направлении цели
        Eigen::Vector3d boost_acceleration = dir_to_target *
                                             (guidance_config_.max_lat_acc * BOOST_ACCEL_FACTOR);

        // 🚀 Высокая скорость сближения
        Eigen::Vector3d boost_velocity = dir_to_target * BOOST_SPEED_MAX;

        RCLCPP_INFO(this->get_logger(),
                    "🚀 BOOST ENGAGED! Dist: %.1fm | Acc: %.2f m/s² | Speed: %.1f m/s",
                    current_dist, boost_acceleration.norm(), boost_velocity.norm());

        // 🚀 Публикуем setpoint с рывком (ограничено по времени)
        auto boost_start = this->get_clock()->now();
        while ((this->get_clock()->now() - boost_start).seconds() < BOOST_DURATION_SEC)
        {
          // Проверка: не достигли ли цель
          double new_dist = std::sqrt(
              target_ned_.x() * target_ned_.x() +
              target_ned_.y() * target_ned_.y());
          if (new_dist < 2.0)
            break; // Остановка при сближении < 2м

          // Вычисление yaw для ориентации носом к цели
          double target_yaw = std::atan2(target_ned_.y(), target_ned_.x());

          px4_msgs::msg::TrajectorySetpoint msg;
          msg.position = {
              static_cast<float>(target_ned_.x()),
              static_cast<float>(target_ned_.y()),
              static_cast<float>(target_ned_.z())};
          msg.velocity = {
              static_cast<float>(boost_velocity.x()),
              static_cast<float>(boost_velocity.y()),
              static_cast<float>(boost_velocity.z())};
          msg.acceleration = {
              static_cast<float>(boost_acceleration.x()),
              static_cast<float>(boost_acceleration.y()),
              static_cast<float>(boost_acceleration.z())};
          msg.yaw = static_cast<float>(target_yaw);
          msg.yawspeed = NAN;
          msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;

          trajectory_setpoint_pub_->publish(msg);
          vehicle_ctrl_->publishOffboardControlMode();

          // Короткий сон для стабильности цикла
          rclcpp::sleep_for(std::chrono::milliseconds(20));
        }

        RCLCPP_INFO(this->get_logger(), "✅ Boost phase complete");
      }
    }

    if (visual_searching)
    {
      RCLCPP_INFO(this->get_logger(), "🔍 Visual search mode: hovering for %d seconds waiting for target detection...",
                  visual_search_timeout_s);

      auto_phase_ = AutoPhase::VISUAL_SEARCH;

      // Устанавливаем hover на текущей позиции
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (drone_gps_.is_valid)
        {
          target_gps_ = {drone_gps_.lat, drone_gps_.lon, drone_gps_.alt};
        }
        auto_phase_ = AutoPhase::GPS_APPROACH;
        start_attack_ = false;
        attack_enabled_ = false;
        // attack_ready_.store(false, std::memory_order_release);
        RCLCPP_INFO(this->get_logger(), "🔍 Visual search init: attack_ready_ preserved as %s",
                    attack_ready_.load() ? "true" : "false");
      }

      auto search_start = std::chrono::steady_clock::now();
      bool target_found = false;

      while (true)
      {

        if (takeoff_abort_)
        {
          handleTakeoffAbort();
          return;
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - search_start)
                           .count();

        if (elapsed >= visual_search_timeout_s)
        {
          RCLCPP_WARN(this->get_logger(), "⏰ Visual search timeout (%ds) — no target detected",
                      visual_search_timeout_s);
          break;
        }

        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          if (attack_ready_.load(std::memory_order_acquire))
          {
            RCLCPP_INFO(this->get_logger(), "🎯 Target detected by camera after %ld seconds!",
                        elapsed);
            target_found = true;
            break;
          }
        }

        // Публикуем hover setpoint
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          if (drone_gps_.is_valid && ref_gps_valid_)
          {
            Eigen::Vector3d hover_ned = gpsToNed(
                drone_gps_.lat, drone_gps_.lon, drone_gps_.alt,
                ref_gps_.lat, ref_gps_.lon, ref_gps_.alt);
            vehicle_ctrl_->publishPositionSetpoint(hover_ned,
                                                   Eigen::Vector3d{2.5, 0.0, 0.0}, Eigen::Vector3d{0.1, 0.0, 0.0});
          }
        }
        vehicle_ctrl_->publishOffboardControlMode();

        // Короткий сон с проверкой abort
        for (int i = 0; i < 20; i++)
        {
          if (takeoff_abort_)
          {
            handleTakeoffAbort();
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }

      if (target_found)
      {
        // Цель найдена камерой — переходим в атаку
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto_phase_ = AutoPhase::VISUAL_ATTACK;
        start_attack_ = true; // если будешь включать по команде - закомменти
        attack_enabled_ = true;
        RCLCPP_INFO(this->get_logger(), "🚀 Switching to VISUAL ATTACK mode!");
        return;
      }
      else
      {
        // Таймаут — переходим к GPS сближению если цель есть
        RCLCPP_INFO(this->get_logger(), "🔄 Proceeding to GPS approach phase...");
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
      attack_ready_.store(false, std::memory_order_release);
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
      attack_ready_.store(false, std::memory_order_release);
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

  void Node::executeTakeoffSequence(bool skip_takeoff)
  {
    RCLCPP_INFO(this->get_logger(), "🛫 Starting Takeoff Sequence...");

    if (skip_takeoff)
    {
      RCLCPP_INFO(this->get_logger(), "⏩ Skipping takeoff phase, going straight to target search");

      std::lock_guard<std::mutex> lock(state_mutex_);
      if (drone_gps_.is_valid)
      {
        ref_gps_.lat = drone_gps_.lat;
        ref_gps_.lon = drone_gps_.lon;
        ref_gps_.alt = drone_gps_.alt;
        ref_gps_valid_ = true;
        target_gps_ = {drone_gps_.lat, drone_gps_.lon, drone_gps_.alt}; // hover на месте
        RCLCPP_INFO(this->get_logger(), "📍 Hover GPS set: %.6f, %.6f, %.2f",
                    drone_gps_.lat, drone_gps_.lon, drone_gps_.alt);
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "⚠️ GPS invalid at takeoff skip, using NAN for hover");
        ref_gps_valid_ = false;
        target_gps_ = {0, 0, 0};
      }

      auto_phase_ = AutoPhase::GPS_APPROACH;
      start_attack_ = false;
      attack_enabled_ = false;
      attack_ready_.store(false, std::memory_order_release);

      // Публикуем hover setpoint многократно для стабильности
      for (int i = 0; i < 10; i++)
      {
        px4_msgs::msg::TrajectorySetpoint hover_ts;
        hover_ts.position = {NAN, NAN, NAN};
        hover_ts.velocity = {NAN, NAN, NAN};
        hover_ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        trajectory_setpoint_pub_->publish(hover_ts);
        vehicle_ctrl_->publishOffboardControlMode();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      return;
    }

    float target_z = -static_cast<float>(this->get_parameter("takeoff.target_altitude_m").as_double());
    float vz = static_cast<float>(this->get_parameter("takeoff.climb_rate_mps").as_double());
    float altitude_threshold = static_cast<float>(this->get_parameter("takeoff.altitude_threshold_m").as_double());
    float min_alt_already_flying = static_cast<float>(this->get_parameter("takeoff.min_altitude_already_flying_m").as_double());
    int min_cycles_already_flying = static_cast<int>(this->get_parameter("takeoff.min_cycles_for_already_flying").as_int());
    int takeoff_timeout_s = this->get_parameter("takeoff.timeout_s").as_int();

    RCLCPP_INFO(this->get_logger(), "🛫 Takeoff params: target=%.1fm, climb=%.1fm/s, threshold=%.1fm, timeout=%ds",
                -target_z, vz, altitude_threshold, takeoff_timeout_s);

    // Флаги для прерывания
    takeoff_running_ = true;
    takeoff_abort_ = false;

    double start_alt = NAN;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (drone_gps_.is_valid)
        start_alt = drone_gps_.alt;
    }

    int cycles = 0;
    auto takeoff_start_time = std::chrono::steady_clock::now();
    const int MAX_CYCLES = 1200; // Ограничитель безопасности

    while (cycles < MAX_CYCLES)
    {
      // Лог статуса каждые 500 итераций
      if (cycles > 0 && cycles % 500 == 0)
      {
        RCLCPP_INFO(this->get_logger(), "🔄 Takeoff cycle %d: abort=%s, running=%s",
                    cycles, takeoff_abort_ ? "true" : "false", takeoff_running_ ? "true" : "false");
      }

      // Проверка прерывания
      if (takeoff_abort_)
      {
        handleTakeoffAbort();
        return;
      }

      // Проверка таймаута
      auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - takeoff_start_time)
                           .count();
      if (elapsed_s >= takeoff_timeout_s)
      {
        RCLCPP_WARN(this->get_logger(), "⏱️ Takeoff timeout: %ds elapsed. Proceeding anyway.",
                    static_cast<int>(elapsed_s));
        break;
      }

      // Публикация команды взлета
      px4_msgs::msg::TrajectorySetpoint ts;
      ts.position = {NAN, NAN, target_z};
      ts.velocity = {NAN, NAN, vz};
      ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
      trajectory_setpoint_pub_->publish(ts);
      vehicle_ctrl_->publishOffboardControlMode();

      // Проверка достижения высоты
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (drone_gps_.is_valid && !std::isnan(start_alt))
        {
          // Высота достигнута
          if (drone_gps_.alt >= start_alt + altitude_threshold)
          {
            RCLCPP_INFO(this->get_logger(), "✅ Altitude reached.");
            break;
          }
          // Уже летит (короткая проверка)
          if (drone_gps_.alt >= start_alt + min_alt_already_flying && cycles > min_cycles_already_flying)
          {
            RCLCPP_INFO(this->get_logger(), "✅ Already flying.");
            break;
          }
        }
      }

      // Короткий сон с проверкой прерывания
      for (int i = 0; i < 20; i++)
      {
        if (takeoff_abort_)
        {
          handleTakeoffAbort();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      cycles++;
    }

    takeoff_running_ = false;

    // Финальная проверка высоты (логирование предупреждения, если не достигнута)
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (drone_gps_.is_valid && !std::isnan(start_alt))
      {
        double current_alt = drone_gps_.alt;
        if (current_alt < start_alt + altitude_threshold)
        {
          RCLCPP_WARN(this->get_logger(),
                      "⚠️ Target altitude not reached (alt=%.1f, target=%.1f). Proceeding to detection mode anyway.",
                      current_alt, -target_z);
        }
      }
    }
  }

  void Node::executeSearchMode()
  {
    RCLCPP_INFO(this->get_logger(), "🔍 Starting SEARCH mode sequence: hover + yaw tracking (awaiting 'start_attack')");

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // === 1. Сохраняем точку взлёта ===
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (drone_gps_.is_valid)
      {
        start_lat_ = drone_gps_.lat;
        start_lon_ = drone_gps_.lon;
        start_alt_ = drone_gps_.alt;
        start_gps_valid_ = true;
        RCLCPP_INFO(this->get_logger(), "🏠 Start position saved: %.6f, %.6f, %.2f",
                    start_lat_, start_lon_, start_alt_);
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "⚠️ GPS not valid — will wait for fix");
      }
    }

    // === 2. Arm ===
    sendArmCommand(true);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // === 3. OFFBOARD режим ===
    RCLCPP_INFO(this->get_logger(), "📡 Switching to OFFBOARD mode...");
    for (int i = 0; i < 10; ++i)
    {
      vehicle_ctrl_->setOffboardMode();
      vehicle_ctrl_->publishOffboardControlMode();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // === 4. Пропуск взлёта? ===
    bool skip_takeoff = this->get_parameter("takeoff.skip").as_bool();
    RCLCPP_INFO(this->get_logger(), "🔧 Takeoff skip: %s", skip_takeoff ? "YES" : "NO");

    // === 5. Выполняем взлёт (синхронно) ===
    executeTakeoffSequence(skip_takeoff);

    // === 6. Ждём валидности ref_gps_ для NED ===
    RCLCPP_INFO(this->get_logger(), "⏳ Waiting for ref_gps_ to be valid...");
    auto wait_start = std::chrono::steady_clock::now();
    while (!ref_gps_valid_ && (std::chrono::steady_clock::now() - wait_start) < std::chrono::seconds(5))
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (!ref_gps_valid_)
    {
      RCLCPP_ERROR(this->get_logger(), "❌ ref_gps_ not valid — cannot start SEARCH");
      return;
    }

    // === 7. Устанавливаем позицию зависания: текущая GPS ===
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!drone_gps_.is_valid)
      {
        RCLCPP_ERROR(this->get_logger(), "❌ Drone GPS invalid — cannot hover");
        return;
      }

      target_gps_ = {drone_gps_.lat, drone_gps_.lon, drone_gps_.alt};
      RCLCPP_INFO(this->get_logger(), "📍 Hover target set: %.6f, %.6f, %.2f",
                  target_gps_.lat, target_gps_.lon, target_gps_.alt);
    }

    // === 8. Включаем режим VISUAL_SEARCH — НЕ АТАКУЕТ, только слежение ===
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      auto_phase_ = AutoPhase::VISUAL_SEARCH;
      start_attack_ = false;   // ❌ Не включаем атаку
      attack_enabled_ = false; // ❌
      attack_ready_.store(false, std::memory_order_release);
      guidance_ctrl_->reset();
    }

    RCLCPP_INFO(this->get_logger(), "✅ SEARCH mode ACTIVE: hovering and yaw-tracking. Awaiting 'start_attack' command.");
    RCLCPP_INFO(this->get_logger(), "💡 Tip: Send {\"start_attack\": true} to begin attack.");
  }

  void Node::declareCommonParameters()
  {

    // auto safe_declare = [this](const std::string& name, auto&& default_val) {
    //     try {
    //         this->declare_parameter(name, std::forward<decltype(default_val)>(default_val));
    //     }
    //     catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException&) {
    //         // Параметр уже загружен из YAML, пропускаем объявление
    //     }
    // };

    /*=== Параметры логирования ===*/
    // this->declare_parameter<int>("logging.throttle_interval_ms", 1000);

    /*=== Параметры логирования ===*/
    // safe_declare("logging.throttle_interval_ms", 136000);

    /*=== Параметры командования (ДО ЗАГРУЗКИ!) ===*/
    // this->declare_parameter<bool>("commander.enable", false);
    // this->declare_parameter<int>("commander.port", 11472);
    // this->declare_parameter<std::vector<std::string>>("commander.targets", std::vector<std::string>{});
    // this->declare_parameter<int>("commander.target_port", 11471);

    /*=== Параметры поиска/search режим ===*/
    // this->declare_parameter<double>("search.kp_yaw_track", 0.5);

    // === Параметры взлета из YAML ===
    // this->declare_parameter("takeoff.target_altitude_m", 91.0);
    // this->declare_parameter("takeoff.climb_rate_mps", -3.0);
    // this->declare_parameter("takeoff.altitude_threshold_m", 1.5);
    // this->declare_parameter("takeoff.min_altitude_already_flying_m", 2.0);
    // this->declare_parameter("takeoff.min_cycles_for_already_flying", 20);
    // this->declare_parameter("takeoff.timeout_s", 20);

    // из части 3. Takeoff — читаем параметры из YAML
    this->declare_parameter("takeoff.target_altitude_m", 91.0);
    this->declare_parameter("takeoff.climb_rate_mps", -3.0);
    this->declare_parameter("takeoff.altitude_threshold_m", 1.5);
    this->declare_parameter("takeoff.min_altitude_already_flying_m", 2.0);
    this->declare_parameter("takeoff.min_cycles_for_already_flying", 20);
    this->declare_parameter("takeoff.skip", false);   // 🆕 Пропуск этапа взлёта
    this->declare_parameter("takeoff.timeout_s", 20); // 🆕 Таймаут взлёта (секунды)

    this->declare_parameter("takeoff.visual_searching", false);
    this->declare_parameter("takeoff.visual_search_timeout_s", 300); // 5 минут по умолчанию

    RCLCPP_DEBUG(this->get_logger(), "✓ All parameters declared successfully");
  }



void Node::softReset()
{
    RCLCPP_WARN(this->get_logger(), "🔄 SOFT RESET: Clearing all states...");
    
    // 1. Блокируем всё
    emergency_stop_ = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // 2. Сброс контроллеров (безопасно, т.к. они используют свои mutex)
    if (guidance_ctrl_) guidance_ctrl_->reset();
    if (los_estimator_) los_estimator_->reset();
    if (target_tracker_) target_tracker_->reset();

    // 3. Очистка состояния
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto_phase_ = AutoPhase::IDLE;
    start_attack_ = false;
    attack_enabled_ = false;
    attack_ready_.store(false, std::memory_order_release);
    takeoff_running_ = false;
    takeoff_abort_ = false;
    active_command_ = ActiveCommand::NONE;
    active_command_param_ = 0.0;
    closest_target_ip_.clear();
    ref_gps_valid_ = false;
    
    // 4. Сброс ручного управления
    manual_input_.roll = 0.0f;
    manual_input_.pitch = 0.0f;
    manual_input_.yaw = 0.0f;
    manual_input_.throttle = 0.5f;
    manual_input_.updated.store(false, std::memory_order_release);
    filtered_manual_input_ = FilteredManualInput();

    // 5. Публикация безопасного setpoint (PX4 не будет падать по offboard timeout)
    px4_msgs::msg::TrajectorySetpoint safe_ts;
    safe_ts.position = {NAN, NAN, NAN};
    safe_ts.velocity = {NAN, NAN, NAN};
    safe_ts.acceleration = {NAN, NAN, NAN};
    safe_ts.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    trajectory_setpoint_pub_->publish(safe_ts);
    vehicle_ctrl_->publishOffboardControlMode();

    // 6. Снимаем аварийную остановку через 100 мс
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    emergency_stop_ = false;
    RCLCPP_INFO(this->get_logger(), "✅ SOFT RESET complete. System IDLE.");
}


} // namespace los_guidance

