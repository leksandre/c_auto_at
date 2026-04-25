/**
 * @file los_estimator.cpp
 * @brief Реализация класса LosEstimator (Kalman Filter для LOS)
 * @note Вся логика KF вынесена из node.cpp для модульности
 * 
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#include "los_guidance/los_estimator.hpp"
#include <Eigen/Geometry>

namespace los_guidance {

// ============================================================================
// Конструктор
// ============================================================================

LosEstimator::LosEstimator(const LosEstimatorConfig& config)
  : config_(config)
{
  // Инициализация матриц
  x_.setZero(STATE_DIM);
  P_.setIdentity(STATE_DIM, STATE_DIM);
  F_.setIdentity(STATE_DIM, STATE_DIM);
  H_.setZero(MEASUREMENT_DIM, STATE_DIM);
  
  initMatrices_();
  
  // Начальная ковариация
  P_.block<3, 3>(0, 0) *= config_.init_position_variance;
  P_.block<3, 3>(3, 3) *= config_.init_velocity_variance;
  
  initialized_ = true;
}

// ============================================================================
// Публичные методы
// ============================================================================

void LosEstimator::predict(
  double dt,
  const Eigen::Vector3d& omega_b,
  const Eigen::Matrix3d& R_nb
) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (!initialized_) {
    return;
  }
  
  if (dt <= 0.0) {
    return;
  }
  
  debug_info_.gyro_compensation_applied = false;
  
  if (config_.use_gyro_compensation) {
    // === С компенсацией гироскопа ===
    // Конвертация угловой скорости из body frame в nav frame
    Eigen::Vector3d omega_n = R_nb.transpose() * omega_b;
    
    // Построение кососимметричной матрицы
    Eigen::Matrix3d omega_sym = skewSymmetric_(omega_n);
    
    // Построение матрицы перехода F с учётом вращения
    // Модель: u_dot = u_dot - ω × u
    F_.setIdentity();
    F_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() - dt * omega_sym;
    F_.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    F_.block<3, 3>(3, 0) = -omega_sym;
    F_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    
    // Предсказание состояния
    Eigen::Vector3d u = x_.segment<3>(0);
    Eigen::Vector3d u_dot = x_.segment<3>(3);
    
    // Компенсированная скорость: u_dot_comp = u_dot - ω × u
    Eigen::Vector3d u_dot_comp = u_dot - omega_n.cross(u);
    
    // Следующее состояние: u_next = u + dt * u_dot_comp
    Eigen::Vector3d u_next = u + dt * u_dot_comp;
    Eigen::Vector3d u_dot_next = u_dot_comp;
    
    // Нормализация LOS вектора
    double norm = u_next.norm();
    if (norm > EPSILON) {
      u_next.normalize();
    }
    
    // Обновление состояния
    x_.segment<3>(0) = u_next;
    x_.segment<3>(3) = u_dot_next;
    
    debug_info_.gyro_compensation_applied = true;
    
  } else {
    // === Без компенсации гироскопа (простая модель постоянной скорости) ===
    F_.setIdentity();
    F_.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;  // Позиция обновляется скоростью
    
    // Предсказание состояния: x = F * x
    x_ = F_ * x_;
    
    // Нормализация LOS вектора
    normalizeLosState_();
  }
  
  // Предсказание ковариации: P = F * P * F^T + Q
  P_ = F_ * P_ * F_.transpose() + Q_;
  
  // Обеспечение симметричности и положительной определённости P
  P_ = (P_ + P_.transpose()) * 0.5;
}

void LosEstimator::update(const Eigen::Vector3d& z_meas) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (!initialized_) {
    return;
  }
  
  // Проверка валидности измерения
  if (!isValidVector_(z_meas)) {
    return;
  }
  
  // Нормализация измерения
  Eigen::Vector3d z_normalized = z_meas.normalized();
  
  // Сохранение измерения для отладки
  debug_info_.measurement = z_normalized;
  
  // Вычисление остатка (innovation): y = z - H * x
  Eigen::VectorXd y = z_normalized - H_ * x_;
  debug_info_.residual = y;
  debug_info_.residual_norm = y.norm();
  
  // Вычисление ковариации остатка: S = H * P * H^T + R
  Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
  
  // Вычисление коэффициента Калмана: K = P * H^T * S^-1
  Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
  
  // Сохранение диагонали K для отладки
  debug_info_.kalman_gain << K(0, 0), K(1, 1), K(2, 2);
  
  // Обновление состояния: x = x + K * y
  x_ = x_ + K * y;
  
  // Обновление ковариации: P = (I - K * H) * P
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
  P_ = (I - K * H_) * P_;
  
  // Обеспечение симметричности P
  P_ = (P_ + P_.transpose()) * 0.5;
  
  // Нормализация LOS вектора в состоянии
  normalizeLosState_();
  
  // Адаптация шумов (если включена)
  if (config_.enable_adaptive_noise) {
    adaptNoise_(y);
  }
  
  // Обновление статистики
  update_count_++;
  if (update_count_ > CONVERGENCE_THRESHOLD) {
    converged_ = true;
  }
  
  // Сохранение отладочной информации
  debug_info_.trace_covariance = P_.trace();
}

LosEstimate LosEstimator::getEstimate() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  LosEstimate estimate;
  
  if (!initialized_) {
    estimate.is_valid = false;
    return estimate;
  }
  
  estimate.los_vector = x_.segment<3>(0).normalized();
  estimate.los_velocity = x_.segment<3>(3);
  estimate.los_angular_velocity = computeLosLambda_();
  estimate.covariance = P_;
  estimate.is_valid = true;
  estimate.is_converged = converged_;
  
  return estimate;
}

Eigen::Vector3d LosEstimator::getLosVector() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (!initialized_) {
    return Eigen::Vector3d::Zero();
  }
  
  return x_.segment<3>(0).normalized();
}

Eigen::Vector3d LosEstimator::getLosAngularVelocity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (!initialized_) {
    return Eigen::Vector3d::Zero();
  }
  
  return computeLosLambda_();
}

Eigen::Matrix<double, 6, 6> LosEstimator::getCovariance() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return P_;
}

bool LosEstimator::isConverged() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return converged_;
}

bool LosEstimator::isValid() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return initialized_ && isValidVector_(x_);
}

void LosEstimator::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  x_.setZero(STATE_DIM);
  P_.setIdentity(STATE_DIM, STATE_DIM);
  P_.block<3, 3>(0, 0) *= config_.init_position_variance;
  P_.block<3, 3>(3, 3) *= config_.init_velocity_variance;
  
  F_.setIdentity();
  H_.setZero(MEASUREMENT_DIM, STATE_DIM);
  H_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  
  initialized_ = true;
  converged_ = false;
  update_count_ = 0;
  
  debug_info_ = LosEstimatorDebug();
}

void LosEstimator::updateConfig(const LosEstimatorConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  initMatrices_();
}

LosEstimatorDebug LosEstimator::getDebugInfo() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return debug_info_;
}

void LosEstimator::setGyroCompensationEnabled(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.use_gyro_compensation = enable;
}

void LosEstimator::setAdaptiveNoiseEnabled(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.enable_adaptive_noise = enable;
}

// ============================================================================
// Приватные методы
// ============================================================================

void LosEstimator::initMatrices_() {
  // Матрица измерения H: измеряем только позицию LOS (первые 3 состояния)
  H_.setZero(MEASUREMENT_DIM, STATE_DIM);
  H_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  
  // Матрица шума процесса Q (диагональная)
  Q_ = config_.process_noise_Q * Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
  
  // Матрица шума измерения R (диагональная)
  R_ = config_.measurement_noise_R * Eigen::MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM);
  
  // Матрица перехода F (инициализация, будет обновляться в predict)
  F_.setIdentity(STATE_DIM, STATE_DIM);
  F_.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();  // Constant velocity model
}

void LosEstimator::adaptNoise_(const Eigen::VectorXd& residual) {
  double res_norm = residual.norm();
  
  // Адаптивный масштаб: больший остаток → больше доверия к модели
  double scale = std::clamp(
    1.0 + res_norm,
    config_.adaptive_min_scale,
    config_.adaptive_max_scale
  );
  
  // Увеличиваем Q (меньше доверия к модели при большом остатке)
  Q_ *= scale;
  
  // Уменьшаем R (больше доверия к измерению при большом остатке)
  R_ /= scale;
  
  // Ограничение Q и R для стабильности
  Q_ = Q_.cwiseMax(EPSILON * Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM));
  R_ = R_.cwiseMax(EPSILON * Eigen::MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM));
}

Eigen::Matrix3d LosEstimator::skewSymmetric_(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m <<     0, -v.z(),  v.y(),
         v.z(),     0, -v.x(),
        -v.y(),  v.x(),     0;
  return m;
}

void LosEstimator::normalizeLosState_() {
  Eigen::Vector3d u = x_.segment<3>(0);
  double norm = u.norm();
  
  if (norm > EPSILON) {
    u.normalize();
    x_.segment<3>(0) = u;
  } else {
    // Если вектор слишком мал, инициализируем единичным
    x_.segment<3>(0) << 0.0, 0.0, 1.0;
  }
}

Eigen::Vector3d LosEstimator::computeLosLambda_() const {
  // lambda = u × u_dot (угловая скорость LOS)
  Eigen::Vector3d u = x_.segment<3>(0).normalized();
  Eigen::Vector3d u_dot = x_.segment<3>(3);
  return u.cross(u_dot);
}

bool LosEstimator::isValidVector_(const Eigen::VectorXd& v) const {
  if (v.size() < MEASUREMENT_DIM) {
    return false;
  }
  
  for (int i = 0; i < v.size(); ++i) {
    if (std::isnan(v(i)) || std::isinf(v(i))) {
      return false;
    }
  }
  
  return true;
}

} // namespace los_guidance