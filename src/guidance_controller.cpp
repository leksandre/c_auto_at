/**
 * @file guidance_controller.cpp
 * @brief Реализация класса GuidanceController
 * @note Вся логика наведения вынесена из node.cpp для модульности
 *
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#include "los_guidance/guidance_controller.hpp"
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp> // Здесь определены RCLCPP_INFO, RCLCPP_WARN, RCLCPP_ERROR и RCLCPP_DEBUG

// ── ONNX Runtime ────────────────────────────────────────────────────────
#ifdef HAS_ONNX_RUNTIME
#  include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#endif

#include <fstream>
#include <cstring>
#include <numeric>

namespace los_guidance
{

  // ── Static ONNX Runtime environment ───────────────────────────────────
#ifdef HAS_ONNX_RUNTIME
  Ort::Env GuidanceController::onnx_env_(ORT_LOGGING_LEVEL_WARNING, "guidance_onnx");
#endif

  // ============================================================================
  // Конструктор
  // ============================================================================

  GuidanceController::GuidanceController(const GuidanceConfig &config, double dt)
      : config_(config), dt_(dt),
        onnx_memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
    // Вычисление линейных пределов для оценки скорости
    vc_linear_max_ = std::sqrt(std::tan(config_.max_lat_acc / CONSTANTS_ONE_G));
    vc_linear_min_ = std::sqrt(std::tan(config_.min_vc / CONSTANTS_ONE_G));
    vc_linear_k_ = (config_.max_vc - config_.min_vc) / (vc_linear_max_ - vc_linear_min_);

    // Инициализация ADRC
    initADRC_();

    // Сброс состояний
    reset();

    // 🧠 Автозагрузка ONNX модели, если указан путь
    if (config_.use_neural_network && !config_.nn_model_path.empty())
    {
      loadOnnxModel(config_.nn_model_path, config_.nn_scaler_path);
    }
  }

  // ============================================================================
  // Публичные методы
  // ============================================================================

  GuidanceOutput GuidanceController::computeAcceleration(
      const Eigen::Vector3d &u_los,
      const Eigen::Vector3d &lambda,
      double pitch_rad,
      double yaw_rad,
      double roll_rad)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Проверка валидности входных данных
    if (u_los.norm() < EPSILON)
    {
      GuidanceOutput output;
      output.is_valid = false;
      return output;
    }

    // Оценка скорости сближения (Vc) на основе тангажа
    double Vc = estimateVcFromPitch(pitch_rad, config_.max_lat_acc / CONSTANTS_ONE_G);
    
    // Fallback: если оценка по тангажу слишком мала или невалидна — используем GPS скорость
    if (Vc < config_.min_vc || std::isnan(Vc))
    {
      Vc = gps_velocity_;
    }
    
    // Ограничение Vc в допустимых пределах
    Vc = std::clamp(Vc, config_.min_vc, config_.max_vc);

    return computeAccelerationImpl_(u_los, lambda, Vc, roll_rad, pitch_rad, yaw_rad);
  }

  GuidanceOutput GuidanceController::computeAccelerationWithVc(
      const Eigen::Vector3d &u_los,
      const Eigen::Vector3d &lambda,
      double Vc,
      double yaw_rad,
      double roll_rad)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (u_los.norm() < EPSILON)
    {
      GuidanceOutput output;
      output.is_valid = false;
      return output;
    }

    return computeAccelerationImpl_(u_los, lambda, Vc, roll_rad, 0.0, yaw_rad);
  }

  GuidanceOutput GuidanceController::computeAccelerationImpl_(
      const Eigen::Vector3d &u_los,
      const Eigen::Vector3d &lambda,
      double Vc,
      double roll_rad,
      double pitch_rad,
      double yaw_rad)
  {
    GuidanceOutput output;
    output.Vc_estimate = Vc;

    // ── Нейросетевой режим (ONNX) ──────────────────────────────────────
    if (use_nn_ && onnx_session_)
    {
      GuidanceOutput nn_out = computeNN_(u_los, lambda, roll_rad, pitch_rad, yaw_rad);
      if (!nn_out.is_valid) return nn_out;

      // Clamp acceleration (same as classical path)
      Eigen::Vector3d acc = clampAcceleration_(nn_out.acceleration_frd);
      nn_out.acceleration_frd = acc;
      nn_out.acceleration_cmd = transformAccToNav_(acc, roll_rad, pitch_rad, yaw_rad);
      last_acc_cmd_ = nn_out.acceleration_cmd;

      // Altitude protection: if too low, override Z to upward
      if (altitude_valid_ && current_alt_m_ < min_attack_alt_m_)
      {
        double up_override = config_.max_up_acc * config_.accel_scale * 0.8;
        if (nn_out.acceleration_frd.z() > up_override)
        {
          nn_out.acceleration_frd.z() = up_override;
          nn_out.acceleration_cmd = transformAccToNav_(nn_out.acceleration_frd, roll_rad, pitch_rad, yaw_rad);
          last_acc_cmd_ = nn_out.acceleration_cmd;
        }
      }

      return nn_out;
    }

    // ── Классический алгоритм (PN/PI/Intercept) ────────────────────────
    Eigen::Vector3d acc_cmd;
    if (config_.enable_PI_control)
    {
      if (config_.top_attack_mode)
      {
        acc_cmd = computeInterceptPI_(u_los, dt_);
      }
      else
      {
        acc_cmd = computePnPI_(u_los, lambda, Vc, dt_);
      }
    }
    else
    {
      // Базовый PN: a = N * Vc * (lambda × u_los)
      acc_cmd = config_.pn_gain_N * Vc * lambda.cross(u_los);
    }

    // Добавление parallel acceleration (для увеличения Vc)
    if (config_.enable_acc_parallel)
    {
      acc_cmd = addParallelAcceleration_(acc_cmd, u_los, Vc, dt_);
    }

    // Ограничение ускорений
    acc_cmd = clampAcceleration_(acc_cmd);

    // Трансформация в навигационную систему (полная: roll + pitch + yaw)
    output.acceleration_cmd = transformAccToNav_(acc_cmd, roll_rad, pitch_rad, yaw_rad);
    output.acceleration_frd = acc_cmd;
    output.is_valid = true;

    // Сохранение для последующего использования
    last_acc_cmd_ = output.acceleration_cmd;

    return output;
  }

  // double GuidanceController::estimateVcFromPitch(double pitch_rad, double max_tilt_rad) {
  //   // v = K * sqrt(tan(-pitch))
  //   double vc = 0.0;

  //   if (-pitch_rad > max_tilt_rad) {
  //     vc = config_.max_vc;
  //   } else if (-pitch_rad < config_.min_vc / CONSTANTS_ONE_G) {
  //     vc = config_.min_vc;
  //   } else {
  //     double forward_acc = std::sqrt(std::tan(-pitch_rad));
  //     vc = (forward_acc - vc_linear_min_) * vc_linear_k_ + config_.min_vc;
  //   }

  //   // Сглаживание
  //   vc = vc_pre_ + config_.vc_alpha * (vc - vc_pre_);
  //   vc_pre_ = vc;

  //   debug_info_.vc_estimate = vc;

  //   return vc;
  // }

  double GuidanceController::estimateVcFromPitch(double pitch_rad, double max_tilt_rad)
  {
    // 🆕 Компенсация ЦТ
    double pitch_compensated = pitch_rad;
    if (config_.enable_cg_compensation)
    {
      double cg_trim_rad = config_.cg_trim_pitch_deg * M_PI / 180.0;
      pitch_compensated = pitch_rad - cg_trim_rad;
    }

    double vc = 0.0;

    // 🆕 Масштабирование для тяжёлого дрона
    double effective_max_tilt = max_tilt_rad * config_.accel_scale;
    double effective_min_tilt = (config_.min_vc / CONSTANTS_ONE_G) * config_.accel_scale;

    if (-pitch_compensated > effective_max_tilt)
    {
      vc = config_.max_vc * config_.accel_scale; // 🆕 Применяем масштаб
    }
    else if (-pitch_compensated < effective_min_tilt)
    {
      vc = config_.min_vc * config_.accel_scale;
    }
    else
    {
      double forward_acc = std::sqrt(std::tan(-pitch_compensated));
      vc = (forward_acc - vc_linear_min_) * vc_linear_k_ + config_.min_vc;
      vc *= config_.accel_scale; // 🆕 Применяем масштаб
    }

    // 🆕 Больше сглаживание для тяжёлого
    vc = vc_pre_ + (config_.vc_alpha * config_.mass_ratio) * (vc - vc_pre_);
    vc_pre_ = vc;

    debug_info_.vc_estimate = vc;
    return vc;
  }

  double GuidanceController::estimateVcFromAcc(const Eigen::Vector3d &acc_sp)
  {
    // v = K * sqrt(acc)
    double acc = acc_sp.norm();
    double vc = 0.0;

    if (acc > EPSILON)
    {
      double x = std::sqrt(acc);
      vc = (x - vc_linear_min_) * vc_linear_k_ + config_.min_vc;
    }
    else
    {
      vc = config_.min_vc;
    }

    vc = vc_pre_ + config_.vc_alpha * (vc - vc_pre_);
    vc_pre_ = vc;

    debug_info_.vc_estimate = vc;

    return vc;
  }

  void GuidanceController::reset()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Сброс интегралов PI
    los_err_int_y_ = 0.0;
    los_err_int_z_ = 0.0;

    // Сброс интегралов Intercept
    intercept_x_int_ = 0.0;
    intercept_y_int_ = 0.0;
    intercept_los_err_x_prev_ = 0.0;
    intercept_los_err_y_prev_ = 0.0;
    intercept_x_d_flt_ = 0.0;
    intercept_y_d_flt_ = 0.0;

    // Сброс ADRC
    adrc_initialized_ = false;

    // Сброс состояний ускорения
    a_front_sp_ = 0.0;
    ay_sp_ = 0.0;
    acc_parallel_pre_ = 0.0;
    vc_pre_ = 0.0;

    // Сброс команд
    last_acc_cmd_.setZero();
    acc_frd_sp_.setZero();

    // Сброс отладки
    debug_info_ = GuidanceDebug();
  }

  void GuidanceController::updateConfig(const GuidanceConfig &config)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    initADRC_();
  }

  Eigen::Vector3d GuidanceController::getLastAccelerationCommand() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_acc_cmd_;
  }

  GuidanceDebug GuidanceController::getDebugInfo() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return debug_info_;
  }

  void GuidanceController::setPIControlEnabled(bool enable)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.enable_PI_control = enable;
  }

  void GuidanceController::setTopAttackMode(bool enable)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.top_attack_mode = enable;
  }

  void GuidanceController::setParallelAccEnabled(bool enable)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.enable_acc_parallel = enable;
  }

  // ============================================================================
  // Приватные методы
  // ============================================================================

  Eigen::Vector3d GuidanceController::computePnPI_(
      const Eigen::Vector3d &u_los,
      const Eigen::Vector3d &lambda,
      double Vc,
      double dt)
  {
    // Ошибка LOS (lambda × u_los)
    Eigen::Vector3d los_err = lambda.cross(u_los);
    double los_err_y = los_err.y();
    double los_err_z = los_err.z();

    // Интеграл с anti-windup (Y axis)
    if (!((los_err_int_y_ >= config_.los_y_int_limit && los_err_y > 0.0) ||
          (los_err_int_y_ <= -config_.los_y_int_limit && los_err_y < 0.0)))
    {
      los_err_int_y_ += config_.los_y_ki * los_err_y * dt;
    }

    // Интеграл с anti-windup (Z axis)
    if (!((los_err_int_z_ >= config_.los_z_int_limit && los_err_z > 0.0) ||
          (los_err_int_z_ <= -config_.los_z_int_limit && los_err_z < 0.0)))
    {
      los_err_int_z_ += config_.los_z_ki * los_err_z * dt;
    }

    // Feedforward (Y axis) ✅ ИСПРАВЛЕНО
    double ay_pn_ff = 0.0;
    if (config_.los_y_ff_enable)
    { // ← было los_y_ffenable
      if (u_los.y() > config_.los_y_err_ff_limit)
      {
        ay_pn_ff = config_.los_y_ff * (u_los.y() - config_.los_y_err_ff_limit);
      }
      else if (u_los.y() < -config_.los_y_err_ff_limit)
      {
        ay_pn_ff = config_.los_y_ff * (u_los.y() + config_.los_y_err_ff_limit);
      }
      ay_pn_ff = clamp_(ay_pn_ff, -config_.los_y_ff_limit, config_.los_y_ff_limit); // ← было los_y_fflimit
    }

    // Feedforward (Z axis) ✅ ИСПРАВЛЕНО
    double az_pn_ff = 0.0;
    if (config_.los_z_ff_enable)
    {
      if (u_los.z() > config_.los_z_err_ff_limit)
      {
        az_pn_ff = config_.los_z_ff * (u_los.z() - config_.los_z_err_ff_limit);
      }
      else if (u_los.z() < -config_.los_z_err_ff_limit)
      {
        az_pn_ff = config_.los_z_ff * (u_los.z() + config_.los_z_err_ff_limit);
      }
      az_pn_ff = clamp_(az_pn_ff, -config_.los_z_ff_limit, config_.los_z_ff_limit); // ← было los_z_fflimit
    }

    // PI контроль
    double a_pn_y = ay_pn_ff + config_.los_y_kp * Vc * los_err_y + los_err_int_y_;
    double a_pn_z = az_pn_ff + config_.los_z_kp * Vc * los_err_z + los_err_int_z_;

    // Ограничение
    a_pn_y = clamp_(a_pn_y, -config_.max_lat_acc, config_.max_lat_acc);
    a_pn_z = clamp_(a_pn_z, config_.max_up_acc, config_.max_down_acc);

    // Сохранение отладки
    debug_info_.los_y_ff = ay_pn_ff;
    debug_info_.los_z_ff = az_pn_ff;
    debug_info_.los_y_kp = config_.los_y_kp * Vc * los_err_y;
    debug_info_.los_z_kp = config_.los_z_kp * Vc * los_err_z;
    debug_info_.los_y_int = los_err_int_y_;
    debug_info_.los_z_int = los_err_int_z_;
    debug_info_.los_err_y = los_err_y;
    debug_info_.los_err_z = los_err_z;

    Eigen::Vector3d a_pn{0.0, a_pn_y, a_pn_z};
    return a_pn;
  }

  Eigen::Vector3d GuidanceController::computeInterceptPI_(
      const Eigen::Vector3d &u_los,
      const Eigen::Vector3d &lambda,
      double Vc,
      double dt)
  {
    // Ошибка LOS
    Eigen::Vector3d los_err = lambda.cross(u_los);
    double los_err_x = los_err.x();
    double los_err_y = los_err.y();

    // Интеграл с anti-windup (X axis)
    if (!((intercept_x_int_ >= config_.intercept_int_limit && los_err_x > 0.0) ||
          (intercept_x_int_ <= -config_.intercept_int_limit && los_err_x < 0.0)))
    {
      intercept_x_int_ += config_.intercept_ki * los_err_x * dt;
    }

    // Интеграл с anti-windup (Y axis)
    if (!((intercept_y_int_ >= config_.intercept_int_limit && los_err_y > 0.0) ||
          (intercept_y_int_ <= -config_.intercept_int_limit && los_err_y < 0.0)))
    {
      intercept_y_int_ += config_.intercept_ki * los_err_y * dt;
    }

    // PI контроль
    double a_pn_x = config_.intercept_kp * Vc * los_err_x + intercept_x_int_;
    double a_pn_y = config_.intercept_kp * Vc * los_err_y + intercept_y_int_;
    double a_pn_z = config_.intercept_z_kp * u_los.z();

    Eigen::Vector3d a_pn{a_pn_x, a_pn_y, a_pn_z};
    return a_pn;
  }

  Eigen::Vector3d GuidanceController::computeInterceptPI_(
      const Eigen::Vector3d &u_los,
      double dt)
  {
    // Ошибка LOS (без lambda)
    double los_err_x = u_los.x();
    double los_err_y = u_los.y();

    // Интеграл с anti-windup
    if (!((intercept_x_int_ >= config_.intercept_int_limit && los_err_x > 0.0) ||
          (intercept_x_int_ <= -config_.intercept_int_limit && los_err_x < 0.0)))
    {
      intercept_x_int_ += config_.intercept_ki * los_err_x * dt;
    }

    if (!((intercept_y_int_ >= config_.intercept_int_limit && los_err_y > 0.0) ||
          (intercept_y_int_ <= -config_.intercept_int_limit && los_err_y < 0.0)))
    {
      intercept_y_int_ += config_.intercept_ki * los_err_y * dt;
    }

    // Производная
    double intercept_x_d = (los_err_x - intercept_los_err_x_prev_) / dt;
    double intercept_y_d = (los_err_y - intercept_los_err_y_prev_) / dt;
    intercept_los_err_x_prev_ = los_err_x;
    intercept_los_err_y_prev_ = los_err_y;

    // Фильтрация производной
    intercept_x_d_flt_ = (1.0 - config_.intercept_acc_alpha) * intercept_x_d_flt_ +
                         config_.intercept_acc_alpha * intercept_x_d;
    intercept_y_d_flt_ = (1.0 - config_.intercept_acc_alpha) * intercept_y_d_flt_ +
                         config_.intercept_acc_alpha * intercept_y_d;

    // PID контроль
    double pn_vx = config_.intercept_kp * los_err_x + intercept_x_int_ +
                   config_.intercept_kd * intercept_x_d_flt_;
    double pn_vy = config_.intercept_kp * los_err_y + intercept_y_int_ +
                   config_.intercept_kd * intercept_y_d_flt_;
    double a_pn_z = config_.intercept_z_kp * u_los.z();

    // ADRC для сглаживания
    double a_pn_x = 0.0;
    double a_pn_y = 0.0;

    if (!adrc_initialized_)
    {
      initADRC_();
    }
    else
    {
      adrc_intercept_ax_td_.fhan_td(pn_vx);
      adrc_intercept_ay_td_.fhan_td(pn_vy);
      a_pn_x = adrc_intercept_ax_td_.get_td_x2();
      a_pn_y = adrc_intercept_ay_td_.get_td_x2();
    }

    // Сохранение отладки
    acc_frd_sp_ = {a_pn_x, a_pn_y, a_pn_z};

    Eigen::Vector3d a_pn{a_pn_x, a_pn_y, a_pn_z};
    return a_pn;
  }

  // Eigen::Vector3d GuidanceController::clampAcceleration_(const Eigen::Vector3d& acc) {
  //   Eigen::Vector3d acc_clamped;
  //   acc_clamped.x() = clamp_(acc.x(), -config_.max_hor_acc, config_.max_hor_acc);
  //   acc_clamped.y() = clamp_(acc.y(), -config_.max_hor_acc, config_.max_hor_acc);
  //   acc_clamped.z() = clamp_(acc.z(), config_.max_up_acc, config_.max_down_acc);
  //   return acc_clamped;
  // }

  Eigen::Vector3d GuidanceController::clampAcceleration_(const Eigen::Vector3d &acc)
  {
    Eigen::Vector3d acc_clamped;

    // 🆕 Динамическое ограничение с учётом массы
    double mass_factor = config_.accel_scale; // = 0.62 для 2519г

    // Симметричные пределы для X и Y (можно двигаться в обе стороны)
    double hor_limit = config_.max_hor_acc * mass_factor;

    double acc_x_scaled = acc.x() * config_.clamp_modifications_scale_x;
    double acc_y_scaled = acc.y() * config_.clamp_modifications_scale_y;
    double acc_z_scaled = acc.z() * config_.clamp_modifications_scale_z;

    acc_clamped.x() = clamp_(acc_x_scaled, 0, hor_limit);
    acc_clamped.y() = clamp_(acc_y_scaled, -hor_limit, hor_limit);
    
    double up_limit   = config_.max_up_acc * mass_factor;
    double down_limit = config_.max_down_acc * mass_factor;
    // Z: вертикальные пределы (NED: up = negative, down = positive)
    acc_clamped.z() = clamp_(acc_z_scaled, up_limit, down_limit);

    {
      static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
      RCLCPP_INFO_THROTTLE(rclcpp::get_logger("guidance_ctrl"), *clock, 2500,
        "🔧 clampAcceleration_:"
        "  in=[%.2f, %.2f, %.2f]"
        "  mass_factor=%.2f  scale=[%.2f, %.2f, %.2f]"
        "  scaled=[%.2f, %.2f, %.2f]"
        "  limits: hor=±%.1f  up=%.1f  down=%.1f"
        "  out=[%.2f, %.2f, %.2f]",
        acc.x(), acc.y(), acc.z(),
        mass_factor,
        config_.clamp_modifications_scale_x,
        config_.clamp_modifications_scale_y,
        config_.clamp_modifications_scale_z,
        acc_x_scaled, acc_y_scaled, acc_z_scaled,
        hor_limit, up_limit, down_limit,
        acc_clamped.x(), acc_clamped.y(), acc_clamped.z());
    }

    return acc_clamped;
  }

  Eigen::Vector3d GuidanceController::addParallelAcceleration_(
      const Eigen::Vector3d &acc,
      const Eigen::Vector3d &u_los,
      double Vc,
      double dt)
  {
    (void)dt;
    // Parallel acceleration для увеличения Vc
    double acc_parallel = config_.acc_parallel_gain * (config_.max_vc - Vc);
    acc_parallel = acc_parallel_pre_ + config_.acc_parallel_alpha * 2 * (acc_parallel - acc_parallel_pre_); // здесь константа   
    acc_parallel = clamp_(acc_parallel, 0.0, config_.max_acc_parallel);
    acc_parallel_pre_ = acc_parallel;

    debug_info_.acc_parallel = acc_parallel;

    return acc + acc_parallel * u_los;
  }

  Eigen::Vector3d GuidanceController::transformAccToNav_(
      const Eigen::Vector3d &acc_frd,
      double roll_rad,
      double pitch_rad,
      double yaw_rad)
  {
    // Полная трансформация из body (FRD) в nav (NED) frame
    // Используем полную матрицу вращения: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    // В FRD: X-вперёд, Y-вправо, Z-вниз
    // В NED: X-север, Y-восток, Z-вниз
    // Трансформация: acc_nav = R * acc_body (где R — rotation body→nav)
    
    Eigen::Quaterniond q_local =
        Eigen::AngleAxisd(yaw_rad,   Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll_rad,  Eigen::Vector3d::UnitX());
    
    Eigen::Affine3d transformation(q_local);
    return transformation * acc_frd;
  }

  void GuidanceController::initADRC_()
  {
    adrc_intercept_ax_td_.init_td_params(config_.adrc_r, config_.adrc_h, config_.adrc_n);
    adrc_intercept_ay_td_.init_td_params(config_.adrc_r, config_.adrc_h, config_.adrc_n);
    adrc_initialized_ = true;
  }

  Eigen::Matrix3d GuidanceController::skewSymmetric_(const Eigen::Vector3d &v)
  {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
        v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return m;
  }

  double GuidanceController::clamp_(double value, double min_val, double max_val)
  {
    return std::max(min_val, std::min(max_val, value));
  }

  void GuidanceController::setAttackMode(bool enable)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Можно хранить состояние, если нужно
    // config_.enable_attack_mode = enable;


    if (enable)
    {
      // При включении атаки РЕЗКО увеличиваем целевое ускорение вперед
      // для мощного рывка к цели. config_.max_acc_parallel теперь 5.0
      // acc_parallel_pre_ = config_.max_acc_parallel; // Резкий старт
      // RCLCPP_INFO(rclcpp::get_logger("guidance_ctrl"), "Attack mode enabled! POWERFUL thrust initiated.");
    }
    else
    {
      // При отключении атаки сбрасываем интеграторы и ошибки
      los_err_int_y_ = 0.0;
      // los_err_fnt_z_ = 0.0;
      intercept_x_int_ = 0.0;
      intercept_y_int_ = 0.0;
      // ... другие сбросы ...

      RCLCPP_DEBUG(rclcpp::get_logger("guidance_ctrl"), "Attack mode disabled, states reset.");

    }
  }

  // ============================================================================
  // ONNX Neural Network Inference
  // ============================================================================

  bool GuidanceController::loadOnnxModel(
      const std::string &model_path,
      const std::string &scaler_path)
  {
#ifdef HAS_ONNX_RUNTIME
    try
    {
      Ort::SessionOptions session_options;
      session_options.SetIntraOpNumThreads(1);
      session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

      onnx_session_ = std::make_unique<Ort::Session>(
          onnx_env_, model_path.c_str(), session_options);

      // Получаем имена входов/выходов
      Ort::AllocatorWithDefaultOptions allocator;
      onnx_input_names_.clear();
      onnx_output_names_.clear();

      size_t num_inputs = onnx_session_->GetInputCount();
      for (size_t i = 0; i < num_inputs; i++)
      {
        auto name = onnx_session_->GetInputNameAllocated(i, allocator);
        onnx_input_names_.push_back(name.get());
      }

      size_t num_outputs = onnx_session_->GetOutputCount();
      for (size_t i = 0; i < num_outputs; i++)
      {
        auto name = onnx_session_->GetOutputNameAllocated(i, allocator);
        onnx_output_names_.push_back(name.get());
      }

      RCLCPP_INFO(rclcpp::get_logger("guidance_ctrl"),
                  "✅ ONNX model loaded: %s (inputs=%zu, outputs=%zu)",
                  model_path.c_str(), num_inputs, num_outputs);

      // Загрузка scaler параметров
      if (!scaler_path.empty())
      {
        loadScalerParams_(scaler_path);
      }

      use_nn_ = config_.use_neural_network;
      return true;
    }
    catch (const Ort::Exception &e)
    {
      RCLCPP_ERROR(rclcpp::get_logger("guidance_ctrl"),
                   "❌ Failed to load ONNX model: %s", e.what());
      onnx_session_.reset();
      use_nn_ = false;
      return false;
    }
#else
    (void)model_path;
    (void)scaler_path;
    RCLCPP_ERROR(rclcpp::get_logger("guidance_ctrl"),
                 "❌ ONNX Runtime not available. Recompile with onnxruntime.");
    return false;
#endif
  }

  void GuidanceController::setUseNeuralNetwork(bool enable)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enable && !onnx_session_)
    {
      RCLCPP_WARN(rclcpp::get_logger("guidance_ctrl"),
                  "Cannot enable NN mode: no ONNX model loaded");
      use_nn_ = false;
      return;
    }
    use_nn_ = enable;
    RCLCPP_INFO(rclcpp::get_logger("guidance_ctrl"),
                "🧠 Neural network mode: %s", enable ? "ENABLED" : "DISABLED");
  }

  bool GuidanceController::isNeuralNetworkLoaded() const
  {
    return onnx_session_ != nullptr;
  }

  GuidanceOutput GuidanceController::computeNN_(
      const Eigen::Vector3d &u_los,
      const Eigen::Vector3d &lambda,
      double roll_rad,
      double pitch_rad,
      double yaw_rad)
  {
    GuidanceOutput output;
    output.is_valid = false;

#ifdef HAS_ONNX_RUNTIME
    try
    {
      // ── Сбор входных данных (9 floats) ───────────────────────────────
      // Порядок: [u_los_x, u_los_y, u_los_z, lambda_x, lambda_y, lambda_z, roll, pitch, yaw]
      std::array<float, 9> input = {
          static_cast<float>(u_los.x()), static_cast<float>(u_los.y()), static_cast<float>(u_los.z()),
          static_cast<float>(lambda.x()), static_cast<float>(lambda.y()), static_cast<float>(lambda.z()),
          static_cast<float>(roll_rad), static_cast<float>(pitch_rad), static_cast<float>(yaw_rad)
      };

      // ── Нормализация через X_mean / X_scale ──────────────────────────
      if (scaler_loaded_)
      {
        for (int i = 0; i < 9; i++)
        {
          input[i] = (input[i] - nn_mean_[i]) / nn_scale_[i];
        }
      }

      // ── Создание тензора ─────────────────────────────────────────────
      std::array<int64_t, 2> input_shape = {1, 9};
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          onnx_memory_info_,
          input.data(), input.size(),
          input_shape.data(), input_shape.size());

      // ── Инференс ─────────────────────────────────────────────────────
      auto output_tensors = onnx_session_->Run(
          Ort::RunOptions{},
          onnx_input_names_.data(), &input_tensor, onnx_input_names_.size(),
          onnx_output_names_.data(), onnx_output_names_.size());

      if (output_tensors.empty())
      {
        RCLCPP_WARN(rclcpp::get_logger("guidance_ctrl"),
                    "NN inference returned empty output");
        return output;
      }

      // ── Извлечение результата ────────────────────────────────────────
      float *output_data = output_tensors[0].GetTensorMutableData<float>();

      // ── Денормализация через y_max ───────────────────────────────────
      Eigen::Vector3d acc_frd;
      if (scaler_loaded_)
      {
        acc_frd.x() = static_cast<double>(output_data[0] * nn_y_max_[0]);
        acc_frd.y() = static_cast<double>(output_data[1] * nn_y_max_[1]);
        acc_frd.z() = static_cast<double>(output_data[2] * nn_y_max_[2]);
      }
      else
      {
        acc_frd.x() = static_cast<double>(output_data[0]);
        acc_frd.y() = static_cast<double>(output_data[1]);
        acc_frd.z() = static_cast<double>(output_data[2]);
      }

      // ── Заполнение output (через clamp + transform как в алгоритме) ──
      output.acceleration_frd = acc_frd;
      output.acceleration_cmd = transformAccToNav_(acc_frd, roll_rad, pitch_rad, yaw_rad);
      output.is_valid = true;
      last_acc_cmd_ = output.acceleration_cmd;

      RCLCPP_DEBUG(rclcpp::get_logger("guidance_ctrl"),
                   "🧠 NN acc: [%.3f, %.3f, %.3f] m/s²",
                   acc_frd.x(), acc_frd.y(), acc_frd.z());

      return output;
    }
    catch (const Ort::Exception &e)
    {
      RCLCPP_ERROR(rclcpp::get_logger("guidance_ctrl"),
                   "❌ NN inference failed: %s", e.what());
      output.is_valid = false;
      return output;
    }
#else
    (void)u_los;
    (void)lambda;
    (void)roll_rad;
    (void)pitch_rad;
    (void)yaw_rad;
    RCLCPP_ERROR(rclcpp::get_logger("guidance_ctrl"),
                 "❌ ONNX Runtime not available");
    return output;
#endif
  }

  bool GuidanceController::loadScalerParams_(const std::string &scaler_path)
  {
#ifdef HAS_ONNX_RUNTIME
    try
    {
      // Загружаем .npz файл вручную (без numpy)
      // Формат .npz: ZIP архив с файлами .npy
      // Мы ожидаем ключи: X_mean (9,), X_scale (9,), y_max (3,)

      FILE *fp = fopen(scaler_path.c_str(), "rb");
      if (!fp)
      {
        RCLCPP_WARN(rclcpp::get_logger("guidance_ctrl"),
                    "Scaler file not found: %s", scaler_path.c_str());
        return false;
      }

      // Читаем весь файл в память
      fseek(fp, 0, SEEK_END);
      long fsize = ftell(fp);
      rewind(fp);
      std::vector<uint8_t> buffer(fsize);
      fread(buffer.data(), 1, fsize, fp);
      fclose(fp);

      uint8_t *data_ptr = buffer.data();
      size_t data_len = buffer.size();

      // Поиск .npy данных по ключу
      auto find_npy_data = [&](const std::string &key, float *out, int expected_count) -> bool
      {
        // Ищем ключ в буфере (как строку)
        auto it = std::search(data_ptr, data_ptr + data_len,
                              key.data(), key.data() + key.size());
        if (it == data_ptr + data_len)
          return false;

        // Ищем magic .npy после ключа
        const uint8_t npy_magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
        auto npy_it = std::search(it + key.size(), data_ptr + data_len,
                                  npy_magic, npy_magic + 6);
        if (npy_it == data_ptr + data_len)
          return false;

        // Вычисляем смещение от начала буфера
        size_t npy_offset = npy_it - data_ptr;

        // Парсим заголовок .npy
        // Формат: magic(6) + major(1) + minor(1) + header_len(4) + header_dict + padding
        size_t header_start = npy_offset + 6; // после magic
        uint32_t header_len = *reinterpret_cast<uint32_t *>(data_ptr + header_start + 2);

        // Данные начинаются после заголовка с выравниванием до 64 байт
        size_t data_offset = npy_offset + 6 + 2 + 4 + header_len; // magic + ver + hdr_len + hdr
        // Выравнивание до 64 байт от начала .npy
        size_t padding = (64 - (data_offset - npy_offset) % 64) % 64;
        data_offset += padding;

        // Проверяем, что данные влезают
        if (data_offset + expected_count * sizeof(float) > data_len)
          return false;

        float *npy_data = reinterpret_cast<float *>(data_ptr + data_offset);
        for (int i = 0; i < expected_count; i++)
        {
          out[i] = npy_data[i];
        }
        return true;
      };

      bool found_mean = find_npy_data("X_mean", nn_mean_.data(), 9);
      bool found_scale = find_npy_data("X_scale", nn_scale_.data(), 9);
      bool found_ymax = find_npy_data("y_max", nn_y_max_.data(), 3);

      if (found_mean && found_scale && found_ymax)
      {
        scaler_loaded_ = true;
        RCLCPP_INFO(rclcpp::get_logger("guidance_ctrl"),
                    "✅ Scaler params loaded: X_mean, X_scale, y_max");
        return true;
      }
      else
      {
        RCLCPP_WARN(rclcpp::get_logger("guidance_ctrl"),
                    "⚠️  Scaler file incomplete: mean=%d scale=%d ymax=%d",
                    found_mean, found_scale, found_ymax);
        return false;
      }
    }
    catch (const std::exception &e)
    {
      RCLCPP_WARN(rclcpp::get_logger("guidance_ctrl"),
                  "⚠️  Failed to load scaler: %s", e.what());
      return false;
    }
#else
    (void)scaler_path;
    RCLCPP_WARN(rclcpp::get_logger("guidance_ctrl"),
                "⚠️  ONNX Runtime not available — scaler not loaded");
    return false;
#endif
  }

} // namespace los_guidance