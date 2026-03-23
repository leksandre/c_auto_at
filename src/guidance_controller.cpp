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

namespace los_guidance {

// ============================================================================
// Конструктор
// ============================================================================

GuidanceController::GuidanceController(const GuidanceConfig& config, double dt)
  : config_(config)
  , dt_(dt)
{
  // Вычисление линейных пределов для оценки скорости
  vc_linear_max_ = std::sqrt(std::tan(config_.max_lat_acc / CONSTANTS_ONE_G));
  vc_linear_min_ = std::sqrt(std::tan(config_.min_vc / CONSTANTS_ONE_G));
  vc_linear_k_ = (config_.max_vc - config_.min_vc) / (vc_linear_max_ - vc_linear_min_);
  
  // Инициализация ADRC
  initADRC_();
  
  // Сброс состояний
  reset();
}

// ============================================================================
// Публичные методы
// ============================================================================

GuidanceOutput GuidanceController::computeAcceleration(
  const Eigen::Vector3d& u_los,
  const Eigen::Vector3d& lambda,
  double pitch_rad,
  double yaw_rad
) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  GuidanceOutput output;
  
  // Проверка валидности входных данных
  if (u_los.norm() < EPSILON) {
    output.is_valid = false;
    return output;
  }
  
  // Оценка скорости на основе тангажа
  double Vc = estimateVcFromPitch(pitch_rad, config_.max_lat_acc / CONSTANTS_ONE_G);
  output.Vc_estimate = Vc;
  
  // Выбор алгоритма наведения
  Eigen::Vector3d acc_cmd;
  if (config_.enable_PI_control) {
    if (config_.top_attack_mode) {
      acc_cmd = computeInterceptPI_(u_los, dt_);
    } else {
      acc_cmd = computePnPI_(u_los, lambda, Vc, dt_);
    }
  } else {
    // Базовый PN: a = N * Vc * (lambda × u_los)
    acc_cmd = config_.pn_gain_N * Vc * lambda.cross(u_los);
  }
  
  // Добавление parallel acceleration (для увеличения Vc)
  if (config_.enable_acc_parallel) {
    acc_cmd = addParallelAcceleration_(acc_cmd, u_los, Vc, dt_);
  }
  
  // Ограничение ускорений
  acc_cmd = clampAcceleration_(acc_cmd);
  
  // Трансформация в навигационную систему
  output.acceleration_cmd = transformAccToNav_(acc_cmd, yaw_rad);
  output.acceleration_frd = acc_cmd;
  output.is_valid = true;
  
  // Сохранение для последующего использования
  last_acc_cmd_ = output.acceleration_cmd;
  
  return output;
}

GuidanceOutput GuidanceController::computeAccelerationWithVc(
  const Eigen::Vector3d& u_los,
  const Eigen::Vector3d& lambda,
  double Vc,
  double yaw_rad
) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  GuidanceOutput output;
  
  if (u_los.norm() < EPSILON) {
    output.is_valid = false;
    return output;
  }
  
  output.Vc_estimate = Vc;
  
  Eigen::Vector3d acc_cmd;
  if (config_.enable_PI_control) {
    if (config_.top_attack_mode) {
      acc_cmd = computeInterceptPI_(u_los, dt_);
    } else {
      acc_cmd = computePnPI_(u_los, lambda, Vc, dt_);
    }
  } else {
    acc_cmd = config_.pn_gain_N * Vc * lambda.cross(u_los);
  }
  
  if (config_.enable_acc_parallel) {
    acc_cmd = addParallelAcceleration_(acc_cmd, u_los, Vc, dt_);
  }
  
  acc_cmd = clampAcceleration_(acc_cmd);
  output.acceleration_cmd = transformAccToNav_(acc_cmd, yaw_rad);
  output.acceleration_frd = acc_cmd;
  output.is_valid = true;
  
  last_acc_cmd_ = output.acceleration_cmd;
  
  return output;
}

double GuidanceController::estimateVcFromPitch(double pitch_rad, double max_tilt_rad) {
  // v = K * sqrt(tan(-pitch))
  double vc = 0.0;
  
  if (-pitch_rad > max_tilt_rad) {
    vc = config_.max_vc;
  } else if (-pitch_rad < config_.min_vc / CONSTANTS_ONE_G) {
    vc = config_.min_vc;
  } else {
    double forward_acc = std::sqrt(std::tan(-pitch_rad));
    vc = (forward_acc - vc_linear_min_) * vc_linear_k_ + config_.min_vc;
  }
  
  // Сглаживание
  vc = vc_pre_ + config_.vc_alpha * (vc - vc_pre_);
  vc_pre_ = vc;
  
  debug_info_.vc_estimate = vc;
  
  return vc;
}

double GuidanceController::estimateVcFromAcc(const Eigen::Vector3d& acc_sp) {
  // v = K * sqrt(acc)
  double acc = acc_sp.norm();
  double vc = 0.0;
  
  if (acc > EPSILON) {
    double x = std::sqrt(acc);
    vc = (x - vc_linear_min_) * vc_linear_k_ + config_.min_vc;
  } else {
    vc = config_.min_vc;
  }
  
  vc = vc_pre_ + config_.vc_alpha * (vc - vc_pre_);
  vc_pre_ = vc;
  
  debug_info_.vc_estimate = vc;
  
  return vc;
}

void GuidanceController::reset() {
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

void GuidanceController::updateConfig(const GuidanceConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  initADRC_();
}

Eigen::Vector3d GuidanceController::getLastAccelerationCommand() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_acc_cmd_;
}

GuidanceDebug GuidanceController::getDebugInfo() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return debug_info_;
}

void GuidanceController::setPIControlEnabled(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.enable_PI_control = enable;
}

void GuidanceController::setTopAttackMode(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.top_attack_mode = enable;
}

void GuidanceController::setParallelAccEnabled(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.enable_acc_parallel = enable;
}

// ============================================================================
// Приватные методы
// ============================================================================



Eigen::Vector3d GuidanceController::computePnPI_(
  const Eigen::Vector3d& u_los,
  const Eigen::Vector3d& lambda,
  double Vc,
  double dt
) {
  // Ошибка LOS (lambda × u_los)
  Eigen::Vector3d los_err = lambda.cross(u_los);
  double los_err_y = los_err.y();
  double los_err_z = los_err.z();
  
  // Интеграл с anti-windup (Y axis)
  if (!((los_err_int_y_ >= config_.los_y_int_limit && los_err_y > 0.0) ||
        (los_err_int_y_ <= -config_.los_y_int_limit && los_err_y < 0.0))) {
    los_err_int_y_ += config_.los_y_ki * los_err_y * dt;
  }
  
  // Интеграл с anti-windup (Z axis)
  if (!((los_err_int_z_ >= config_.los_z_int_limit && los_err_z > 0.0) ||
        (los_err_int_z_ <= -config_.los_z_int_limit && los_err_z < 0.0))) {
    los_err_int_z_ += config_.los_z_ki * los_err_z * dt;
  }
  
  // Feedforward (Y axis) ✅ ИСПРАВЛЕНО
  double ay_pn_ff = 0.0;
  if (config_.los_y_ff_enable) {  // ← было los_y_ffenable
    if (u_los.y() > config_.los_y_err_ff_limit) {
      ay_pn_ff = config_.los_y_ff * (u_los.y() - config_.los_y_err_ff_limit);
    } else if (u_los.y() < -config_.los_y_err_ff_limit) {
      ay_pn_ff = config_.los_y_ff * (u_los.y() + config_.los_y_err_ff_limit);
    }
    ay_pn_ff = clamp_(ay_pn_ff, -config_.los_y_ff_limit, config_.los_y_ff_limit);  // ← было los_y_fflimit
  }
  
  // Feedforward (Z axis) ✅ ИСПРАВЛЕНО
  double az_pn_ff = 0.0;
  if (config_.los_z_ff_enable) {
    if (u_los.z() > config_.los_z_err_ff_limit) {
      az_pn_ff = config_.los_z_ff * (u_los.z() - config_.los_z_err_ff_limit);
    } else if (u_los.z() < -config_.los_z_err_ff_limit) {
      az_pn_ff = config_.los_z_ff * (u_los.z() + config_.los_z_err_ff_limit);
    }
    az_pn_ff = clamp_(az_pn_ff, -config_.los_z_ff_limit, config_.los_z_ff_limit);  // ← было los_z_fflimit
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
  const Eigen::Vector3d& u_los,
  const Eigen::Vector3d& lambda,
  double Vc,
  double dt
) {
  // Ошибка LOS
  Eigen::Vector3d los_err = lambda.cross(u_los);
  double los_err_x = los_err.x();
  double los_err_y = los_err.y();
  
  // Интеграл с anti-windup (X axis)
  if (!((intercept_x_int_ >= config_.intercept_int_limit && los_err_x > 0.0) ||
        (intercept_x_int_ <= -config_.intercept_int_limit && los_err_x < 0.0))) {
    intercept_x_int_ += config_.intercept_ki * los_err_x * dt;
  }
  
  // Интеграл с anti-windup (Y axis)
  if (!((intercept_y_int_ >= config_.intercept_int_limit && los_err_y > 0.0) ||
        (intercept_y_int_ <= -config_.intercept_int_limit && los_err_y < 0.0))) {
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
  const Eigen::Vector3d& u_los,
  double dt
) {
  // Ошибка LOS (без lambda)
  double los_err_x = u_los.x();
  double los_err_y = u_los.y();
  
  // Интеграл с anti-windup
  if (!((intercept_x_int_ >= config_.intercept_int_limit && los_err_x > 0.0) ||
        (intercept_x_int_ <= -config_.intercept_int_limit && los_err_x < 0.0))) {
    intercept_x_int_ += config_.intercept_ki * los_err_x * dt;
  }
  
  if (!((intercept_y_int_ >= config_.intercept_int_limit && los_err_y > 0.0) ||
        (intercept_y_int_ <= -config_.intercept_int_limit && los_err_y < 0.0))) {
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
  
  if (!adrc_initialized_) {
    initADRC_();
  } else {
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

Eigen::Vector3d GuidanceController::clampAcceleration_(const Eigen::Vector3d& acc) {
  Eigen::Vector3d acc_clamped;
  acc_clamped.x() = clamp_(acc.x(), -config_.max_hor_acc, config_.max_hor_acc);
  acc_clamped.y() = clamp_(acc.y(), -config_.max_hor_acc, config_.max_hor_acc);
  acc_clamped.z() = clamp_(acc.z(), config_.max_up_acc, config_.max_down_acc);
  return acc_clamped;
}

Eigen::Vector3d GuidanceController::addParallelAcceleration_(
  const Eigen::Vector3d& acc,
  const Eigen::Vector3d& u_los,
  double Vc,
  double dt
) {
    (void)dt; 
  // Parallel acceleration для увеличения Vc
  double acc_parallel = config_.acc_parallel_gain * (config_.max_vc - Vc);
  acc_parallel = acc_parallel_pre_ + config_.acc_parallel_alpha * (acc_parallel - acc_parallel_pre_);
  acc_parallel = clamp_(acc_parallel, 0.0, config_.max_acc_parallel);
  acc_parallel_pre_ = acc_parallel;
  
  debug_info_.acc_parallel = acc_parallel;
  
  return acc + acc_parallel * u_los;
}

Eigen::Vector3d GuidanceController::transformAccToNav_(
  const Eigen::Vector3d& acc_frd,
  double yaw_rad
) {
  // Трансформация из body (FRD) в nav frame через yaw
  Eigen::Vector3d euler_local{0.0, 0.0, yaw_rad};
  Eigen::Quaterniond q_local = Eigen::AngleAxisd(euler_local.z(), Eigen::Vector3d::UnitZ()) *
                                Eigen::AngleAxisd(euler_local.y(), Eigen::Vector3d::UnitY()) *
                                Eigen::AngleAxisd(euler_local.x(), Eigen::Vector3d::UnitX());
  Eigen::Affine3d transformation(q_local);
  return transformation * acc_frd;
}

void GuidanceController::initADRC_() {
  adrc_intercept_ax_td_.init_td_params(config_.adrc_r, config_.adrc_h, config_.adrc_n);
  adrc_intercept_ay_td_.init_td_params(config_.adrc_r, config_.adrc_h, config_.adrc_n);
  adrc_initialized_ = true;
}

Eigen::Matrix3d GuidanceController::skewSymmetric_(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m <<     0, -v.z(),  v.y(),
         v.z(),     0, -v.x(),
        -v.y(),  v.x(),     0;
  return m;
}

double GuidanceController::clamp_(double value, double min_val, double max_val) {
  return std::max(min_val, std::min(max_val, value));
}

} // namespace los_guidance