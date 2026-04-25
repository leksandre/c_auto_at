/**
 * @file los_estimator.hpp
 * @brief Kalman Filter для оценки LOS вектора и угловой скорости
 * @note Состояние: [u_n, v_n, w_n, u_dot_n, v_dot_n, w_dot_n]
 * 
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#ifndef LOS_GUIDANCE_LOS_ESTIMATOR_HPP_
#define LOS_GUIDANCE_LOS_ESTIMATOR_HPP_

#pragma once

#include <Eigen/Dense>
#include <mutex>
#include <memory>
#include <cmath>
#include <algorithm>

namespace los_guidance {

/**
 * @struct LosEstimatorConfig
 * @brief Параметры Kalman Filter для LOS оценки
 * 
 * Все параметры вынесены из node.cpp в этот struct для удобства
 * загрузки из YAML и тестирования.
 */
struct LosEstimatorConfig {
  // === Шумы процесса и измерения ===
  double process_noise_Q{0.01};       ///< Шум процесса (диагональный коэффициент)
  double measurement_noise_R{0.1};    ///< Шум измерения (диагональный коэффициент)
  
  // === Режимы ===
  bool use_gyro_compensation{false};  ///< Использовать компенсацию гироскопа
  
  // === Адаптация ===
  bool enable_adaptive_noise{false};  ///< Включить адаптивную настройку Q/R
  double adaptive_min_scale{1.0};     ///< Минимальный масштаб адаптации
  double adaptive_max_scale{10.0};    ///< Максимальный масштаб адаптации
  
  // === Инициализация ===
  double init_position_variance{1.0}; ///< Начальная дисперсия позиции
  double init_velocity_variance{1.0}; ///< Начальная дисперсия скорости
};

/**
 * @struct LosEstimate
 * @brief Результат оценки LOS
 */
struct LosEstimate {
  Eigen::Vector3d los_vector{Eigen::Vector3d::Zero()};      ///< LOS вектор (nav frame)
  Eigen::Vector3d los_velocity{Eigen::Vector3d::Zero()};    ///< Скорость изменения LOS
  Eigen::Vector3d los_angular_velocity{Eigen::Vector3d::Zero()}; ///< Угловая скорость LOS (lambda)
  Eigen::Matrix<double, 6, 6> covariance{
    Eigen::Matrix<double, 6, 6>::Identity()};  ///< Ковариация состояния
  uint64_t timestamp_us{0};                     ///< Timestamp последней оценки
  bool is_valid{false};                         ///< Валидность оценки
  bool is_converged{false};                     ///< Сходимость фильтра
};

/**
 * @struct LosEstimatorDebug
 * @brief Отладочная информация для логирования
 */
struct LosEstimatorDebug {
  Eigen::Vector3d measurement{Eigen::Vector3d::Zero()};  ///< Измерение
  Eigen::Vector3d residual{Eigen::Vector3d::Zero()};     ///< Остаток (innovation)
  Eigen::Vector3d kalman_gain{Eigen::Vector3d::Zero()};  ///< Коэффициент Калмана (diag)
  double residual_norm{0.0};                              ///< Норма остатка
  double trace_covariance{0.0};                           ///< След ковариации
  bool gyro_compensation_applied{false};                  ///< Применена ли компенсация гироскопа
};

/**
 * @class LosEstimator
 * @brief Оценщик LOS вектора на основе Kalman Filter
 * 
 * Модель состояния (6D):
 * - Позиция LOS (единичный вектор в навигационной системе): [u_n, v_n, w_n]
 * - Скорость изменения LOS: [u_dot_n, v_dot_n, w_dot_n]
 * 
 * Модель измерения (3D):
 * - Измеренный LOS вектор из камеры: [u_meas, v_meas, w_meas]
 * 
 * @thread_safe Да (все публичные методы защищены mutex)
 * 
 * @example
 * ```cpp
 * LosEstimatorConfig config;
 * config.process_noise_Q = 0.01;
 * config.measurement_noise_R = 0.1;
 * config.use_gyro_compensation = true;
 * 
 * LosEstimator estimator(config);
 * 
 * // В цикле обновления
 * estimator.predict(dt, omega_b);
 * estimator.update(z_meas);
 * LosEstimate estimate = estimator.getEstimate();
 * ```
 */
class LosEstimator {
public:
  /**
   * @brief Конструктор
   * @param config Конфигурация Kalman Filter
   */
  explicit LosEstimator(const LosEstimatorConfig& config = LosEstimatorConfig());
  
  /**
   * @brief Деструктор
   */
  ~LosEstimator() = default;
  
  /**
   * @brief Предсказание состояния KF
   * @param dt Шаг времени (сек)
   * @param omega_b Угловая скорость тела (body frame, rad/s)
   * @param R_nb Матрица вращения nav → body
   */
  void predict(double dt, const Eigen::Vector3d& omega_b, const Eigen::Matrix3d& R_nb);
  
  /**
   * @brief Обновление состояния KF по измерению
   * @param z_meas Измеренный LOS вектор (nav frame, нормированный)
   */
  void update(const Eigen::Vector3d& z_meas);
  
  /**
   * @brief Получение текущей оценки LOS
   * @return LosEstimate с оценёнными значениями
   */
  LosEstimate getEstimate() const;
  
  /**
   * @brief Получение оценённого LOS вектора
   * @return LOS вектор (nav frame)
   */
  Eigen::Vector3d getLosVector() const;
  
  /**
   * @brief Получение оценённой угловой скорости LOS (lambda)
   * @return Угловая скорость LOS (rad/s)
   */
  Eigen::Vector3d getLosAngularVelocity() const;
  
  /**
   * @brief Получение ковариации состояния
   * @return Матрица ковариации 6x6
   */
  Eigen::Matrix<double, 6, 6> getCovariance() const;
  
  /**
   * @brief Проверка сходимости фильтра
   * @return true если фильтр сошёлся
   */
  bool isConverged() const;
  
  /**
   * @brief Проверка валидности оценки
   * @return true если оценка валидна
   */
  bool isValid() const;
  
  /**
   * @brief Сброс фильтра к начальным условиям
   */
  void reset();
  
  /**
   * @brief Обновление конфигурации на лету
   * @param config Новая конфигурация
   */
  void updateConfig(const LosEstimatorConfig& config);
  
  /**
   * @brief Получение отладочной информации
   * @return LosEstimatorDebug с отладочными данными
   */
  LosEstimatorDebug getDebugInfo() const;
  
  /**
   * @brief Включение/выключение компенсации гироскопа
   * @param enable true = включить компенсацию
   */
  void setGyroCompensationEnabled(bool enable);
  
  /**
   * @brief Включение/выключение адаптивного шума
   * @param enable true = включить адаптацию
   */
  void setAdaptiveNoiseEnabled(bool enable);

private:
  LosEstimatorConfig config_;
  mutable std::mutex mutex_;  // Защита состояния
  
  // === Состояние KF ===
  Eigen::VectorXd x_;  // Вектор состояния (6x1): [u, v, w, u_dot, v_dot, w_dot]
  Eigen::MatrixXd P_;  // Ковариация (6x6)
  Eigen::MatrixXd Q_;  // Шум процесса (6x6)
  Eigen::MatrixXd R_;  // Шум измерения (3x3)
  Eigen::MatrixXd F_;  // Перехода состояния (6x6)
  Eigen::MatrixXd H_;  // Измерений (3x6)
  
  // === Флаги ===
  bool initialized_{false};
  bool converged_{false};
  uint64_t update_count_{0};
  
  // === Отладка ===
  LosEstimatorDebug debug_info_;
  
  // === Константы ===
  static constexpr double EPSILON = 1e-6;
  static constexpr int STATE_DIM = 6;
  static constexpr int MEASUREMENT_DIM = 3;
  static constexpr int CONVERGENCE_THRESHOLD = 50;  // Обновлений для сходимости
  
  // === Внутренние методы ===
  
  /**
   * @brief Инициализация матриц KF
   */
  void initMatrices_();
  
  /**
   * @brief Адаптация шумов Q/R на основе остатка
   * @param residual Вектор остатка (innovation)
   */
  void adaptNoise_(const Eigen::VectorXd& residual);
  
  /**
   * @brief Матрица косого симметричного вектора
   * @param v Вектор 3D
   * @return Кососимметричная матрица 3x3
   */
  Eigen::Matrix3d skewSymmetric_(const Eigen::Vector3d& v);
  
  /**
   * @brief Нормализация LOS вектора в состоянии
   */
  void normalizeLosState_();
  
  /**
   * @brief Вычисление угловой скорости LOS (lambda = u × u_dot)
   * @return Угловая скорость LOS
   */
  Eigen::Vector3d computeLosLambda_() const;
  
  /**
   * @brief Проверка валидности вектора
   * @param v Вектор
   * @return true если вектор валиден
   */
  bool isValidVector_(const Eigen::VectorXd& v) const;
};

} // namespace los_guidance

#endif // LOS_GUIDANCE_LOS_ESTIMATOR_HPP_