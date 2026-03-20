/**
 * @file target_tracker.hpp
 * @brief Класс для трекинга цели и расчёта LOS вектора
 * @note Преобразует пиксельные координаты → камера → тело → навигация
 */
#ifndef LOS_GUIDANCE_TARGET_TRACKER_HPP_
#define LOS_GUIDANCE_TARGET_TRACKER_HPP_

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <mutex>
#include "vision_msgs/msg/detection2_d.hpp"
#include "ai_service/msg/tracker_result.hpp"

namespace los_guidance {

/**
 * @struct TrackerConfig
 * @brief Конфигурация трекера цели
 */
struct TrackerConfig {
  // Параметры камеры
  double fx{410.0};
  double fy{410.0};
  double cx{640.0};
  double cy{360.0};
  double vertical_fov_deg{60.0};
  
  // Установка камеры (body → camera)
  double install_roll_deg{0.0};
  double install_pitch_deg{5.0};
  double install_yaw_deg{0.0};
  
  // Фильтрация
  double pixel_smooth_alpha{0.3};
  double los_smooth_alpha{0.1};
  double min_confidence{0.5};
  
  // Frame IDs
  std::string frame_id{"camera"};
  std::string nav_frame_id{"ned"};
};

/**
 * @struct TargetInfo
 * @brief Информация о цели после обработки
 */
struct TargetInfo {
  uint64_t timestamp_us{0};
  
  // Пиксельные координаты (raw)
  double pixel_u{0.0};
  double pixel_v{0.0};
  
  // Пиксельные координаты (сглаженные)
  double pixel_u_smooth{0.0};
  double pixel_v_smooth{0.0};
  
  // LOS вектор в навигационной системе
  Eigen::Vector3d los_nav{Eigen::Vector3d::Zero()};
  
  // LOS вектор в системе камеры
  Eigen::Vector3d los_cam{Eigen::Vector3d::Zero()};
  
  // Угловая скорость LOS (lambda)
  Eigen::Vector3d los_lambda{Eigen::Vector3d::Zero()};
  
  // Доверие детекции
  double confidence{0.0};
  
  // Валидность данных
  bool is_valid{false};
  
  // Размер цели (для оценки дистанции)
  double bbox_width{0.0};
  double bbox_height{0.0};
};

/**
 * @class TargetTracker
 * @brief Обработка данных трекинга и расчёт LOS вектора
 * 
 * Этот класс отвечает за:
 * - Получение данных от детектора (TrackerResult / Detection2D)
 * - Сглаживание пиксельных координат
 * - Коррекцию на roll/pitch дрона
 * - Преобразование координат: pixel → camera → body → nav
 * - Расчёт угловой скорости LOS
 * 
 * @thread_safe Да (использует mutex для защиты состояния)
 */
class TargetTracker {
public:
  /**
   * @brief Конструктор
   * @param config Конфигурация трекера
   */
  explicit TargetTracker(const TrackerConfig& config);
  
  /**
   * @brief Обработка сообщения TrackerResult
   * @param msg Сообщение от AI сервиса
   * @param roll_rad Крен дрона (rad)
   * @param pitch_rad Тангаж дрона (rad)
   * @param R_nb Матрица вращения nav → body
   * @return Информация о цели
   */
  TargetInfo processTrackerResult(
    const ai_service::msg::TrackerResult::SharedPtr& msg,
    double roll_rad,
    double pitch_rad,
    const Eigen::Matrix3d& R_nb
  );
  
  /**
   * @brief Обработка сообщения Detection2D
   * @param msg Сообщение bounding box
   * @param roll_rad Крен дрона (rad)
   * @param pitch_rad Тангаж дрона (rad)
   * @param R_nb Матрица вращения nav → body
   * @return Информация о цели
   */
  TargetInfo processDetection2D(
    const vision_msgs::msg::Detection2D::SharedPtr& msg,
    double roll_rad,
    double pitch_rad,
    const Eigen::Matrix3d& R_nb
  );
  
  /**
   * @brief Получение последнего LOS вектора
   */
  Eigen::Vector3d getLosVector() const;
  
  /**
   * @brief Получение угловой скорости LOS
   */
  Eigen::Vector3d getLosAngularVelocity() const;
  
  /**
   * @brief Проверка валидности трекинга
   */
  bool isValid() const;
  
  /**
   * @brief Сброс состояния трекера
   */
  void reset();
  
  /**
   * @brief Обновление конфигурации на лету
   */
  void updateConfig(const TrackerConfig& config);
  
  /**
   * @brief Получение статистики (для дебага)
   */
  struct TrackerStats {
    double avg_latency_ms{0.0};
    uint64_t frames_processed{0};
    uint64_t frames_dropped{0};
    double last_confidence{0.0};
  };
  TrackerStats getStats() const;

private:
  TrackerConfig config_;
  mutable std::mutex mutex_;  // Защита состояния
  
  // Сглаживание пикселей (exponential moving average)
  double pixel_u_prev_{0.0};
  double pixel_v_prev_{0.0};
  bool pixel_initialized_{false};
  
  // Сглаживание LOS вектора
  Eigen::Vector3d los_nav_prev_{Eigen::Vector3d::Zero()};
  bool los_initialized_{false};
  
  // Для расчёта lambda (угловой скорости LOS)
  Eigen::Vector3d los_nav_history_{Eigen::Vector3d::Zero()};
  uint64_t last_timestamp_us_{0};
  
  // Статистика
  TrackerStats stats_;
  
  // Матрицы вращения
  Eigen::Matrix3d R_bc_;  // body → camera
  Eigen::Matrix3d R_cb_;  // camera → body (transpose)
  
  // === Внутренние методы ===
  
  /**
   * @brief Инициализация матриц вращения камеры
   */
  void initCameraMatrices_();
  
  /**
   * @brief Преобразование пикселей в вектор камеры
   * @param u Координата U (pixels)
   * @param v Координата V (pixels)
   * @return Нормированный вектор в системе камеры
   */
  Eigen::Vector3d pixelToCamVec_(double u, double v);
  
  /**
   * @brief Коррекция пикселей на roll/pitch дрона
   * @param u Координата U
   * @param v Координата V
   * @param roll_rad Крен
   * @param pitch_rad Тангаж
   * @return Скорректированные координаты
   */
  std::pair<double, double> correctPixelRollPitch_(
    double u, double v,
    double roll_rad, double pitch_rad
  );
  
  /**
   * @brief Преобразование вектора камеры в навигационную систему
   * @param u_c Вектор в системе камеры
   * @param R_bc Матрица body → camera
   * @param R_nb Матрица nav → body
   * @return Вектор в навигационной системе
   */
  Eigen::Vector3d camToNav_(
    const Eigen::Vector3d& u_c,
    const Eigen::Matrix3d& R_bc,
    const Eigen::Matrix3d& R_nb
  );
  
  /**
   * @brief Сглаживание значения (exponential moving average)
   */
  double smooth_(double prev, double current, double alpha);
  
  /**
   * @brief Сглаживание вектора
   */
  Eigen::Vector3d smooth_(
    const Eigen::Vector3d& prev,
    const Eigen::Vector3d& current,
    double alpha
  );
  
  /**
   * @brief Расчёт угловой скорости LOS
   */
  Eigen::Vector3d computeLosLambda_(
    const Eigen::Vector3d& los_current,
    const Eigen::Vector3d& los_prev,
    double dt
  );
  
  /**
   * @brief Матрица косого симметричного вектора
   */
  Eigen::Matrix3d skewSymmetric_(const Eigen::Vector3d& v);
};

} // namespace los_guidance
#endif