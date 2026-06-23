/**
 * @file guidance_controller.hpp
 * @brief Контроллер наведения - реализует алгоритмы PN, PI, Intercept + ONNX нейросеть
 * @note Все параметры загружаются из YAML конфига, класс тестирован независимо от ROS2
 *
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#ifndef LOS_GUIDANCE_GUIDANCE_CONTROLLER_HPP_
#define LOS_GUIDANCE_GUIDANCE_CONTROLLER_HPP_

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <array>
#include <string>
#include <vector>
#include "los_guidance/adrc.hpp"

// ── ONNX Runtime (опционально) ──────────────────────────────────────────
// Note: onnxruntime headers may be installed flat in /usr/local/include/
// or in a subdirectory. Try both paths.
#ifdef __has_include
#  if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#    include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#    define HAS_ONNX_RUNTIME
#  elif __has_include(<onnxruntime_cxx_api.h>)
#    include <onnxruntime_cxx_api.h>
#    define HAS_ONNX_RUNTIME
#  endif
#endif

namespace los_guidance
{

  /**
   * @struct GuidanceConfig
   * @brief Конфигурация всех параметров наведения
   *
   * Все параметры вынесены из node.cpp в этот struct для удобства
   * загрузки из YAML и тестирования.
   */
  struct GuidanceConfig
  {
    // === PN параметры ===
    double pn_gain_N{3.0};
    double apn_alpha{0.1};
    double ax_rate_limit{1.0};

    // === PI LOS параметры (Y axis) ===
    bool los_y_ff_enable{false};
    double los_y_ff{7.0};
    double los_y_ff_limit{3.0};
    double los_y_err_ff_limit{0.1};
    double los_y_kp{3.0};
    double los_y_ki{0.01};
    double los_y_int_limit{3.0};

    // === PI LOS параметры (Z axis) ===
    bool los_z_ff_enable{false};
    double los_z_ff{7.0};
    double los_z_ff_limit{3.0};
    double los_z_err_ff_limit{0.1};
    double los_z_kp{3.0};
    double los_z_ki{0.01};
    double los_z_int_limit{5.0};

    // === Intercept параметры ===
    double intercept_kp{3.0};
    double intercept_ki{0.01};
    double intercept_kd{0.01};
    double intercept_int_limit{1.0};
    double intercept_z_kp{5.0};
    double intercept_acc_alpha{0.1};

    // === Ограничения ускорений ===
    double max_lat_acc{3.0};
    double max_up_acc{-5.0};
    double max_down_acc{5.0};
    double max_hor_acc{12.0};

    // === Скорость ===
    double min_vc{1.0};
    double max_vc{20.0};
    double vc_alpha{0.1};
    double vc_linear_k{0.0}; // Вычисляется в node.cpp

    // === Режимы ===
    bool enable_PI_control{false};
    bool top_attack_mode{false};
    bool enable_acc_parallel{false};

    // === Parallel acceleration ===
    double acc_parallel_gain{0.1};
    double acc_parallel_alpha{0.1};
    double max_acc_parallel{2.0};

    // === ADRC параметры ===
    double adrc_r{1000.0};
    double adrc_h{0.01};
    double adrc_n{2};

    // 🆕 Для тяжёлого дрона
    double mass_old_g{1562.0}; // Старая масса (граммы)
    double mass_new_g{2519.0}; // Новая масса (граммы)
    double mass_ratio{1.61};   // = mass_new / mass_old
    double accel_scale{0.62};  // = 1/mass_ratio // ПОКА ПРИМЕНЯЕМ ДЛЯ ЗЭТ ВЕЛИЧИНЫ ТОЛЬКО

    // 🆕 Компенсация ЦТ
    double cg_trim_pitch_deg{2.5};
    bool enable_cg_compensation{true};

    // 🧠 Нейросетевое наведение (ONNX)
    bool use_neural_network{false};     ///< Использовать нейросеть вместо алгоритма
    std::string nn_model_path;          ///< Путь к ONNX модели
    std::string nn_scaler_path;         ///< Путь к scaler_params.npz

    double ship_mode_response_factor{0.3};
    double manual_roll_deadzone{0.1};
    double max_roll_slew_rate{2.0};
    double manual_filter_time_constant{0.2};
    double manual_pitch_deadzone{0.1};
    double max_pitch_slew_rate{2.0};
    double manual_yaw_deadzone{0.1};
    double max_yaw_slew_rate{60.0 * M_PI / 180.0}; // rad/s
    double manual_throttle_deadzone{0.05};
    double max_throttle_slew_rate{1.0};
    double spike_filter_time_window{0.1};
    bool enable_emergency_lockout{true};
    double throttle_spikes_threshold{0.9};
    double throttle_cutoff_threshold{0.01};
    bool allow_vision_attack{true};
    bool allow_autoland{true};

        // === Параметры масштабирования ускорения (clamp) ===
    double clamp_modifications_scale_x{1.0};           // Множитель масштабирования для X оси
    double clamp_modifications_scale_y{1.0};           // Множитель масштабирования для Y оси
    double clamp_modifications_scale_z{1.0};           // Множитель масштабирования для Z оси
  };

  /**
   * @struct GuidanceOutput
   * @brief Результат вычисления команды наведения
   */
  struct GuidanceOutput
  {
    Eigen::Vector3d acceleration_cmd{Eigen::Vector3d::Zero()}; ///< Команда ускорения (nav frame)
    Eigen::Vector3d acceleration_frd{Eigen::Vector3d::Zero()}; ///< Команда ускорения (body frame)
    double Vc_estimate{0.0};                                   ///< Оценка скорости
    double los_err_y{0.0};                                     ///< Ошибка LOS Y
    double los_err_z{0.0};                                     ///< Ошибка LOS Z
    bool is_valid{false};                                      ///< Валидность вычислений
  };

  /**
   * @struct GuidanceDebug
   * @brief Отладочная информация для логирования
   */
  struct GuidanceDebug
  {
    double los_y_ff{0.0};
    double los_z_ff{0.0};
    double los_y_kp{0.0};
    double los_z_kp{0.0};
    double los_y_int{0.0};
    double los_z_int{0.0};
    double los_err_y{0.0};
    double los_err_z{0.0};
    double vc_estimate{0.0};
    double acc_parallel{0.0};
  };

  /**
   * @class GuidanceController
   * @brief Основной контроллер расчёта команд ускорения
   *
   * Этот класс инкапсулирует всю логику наведения:
   * - Proportional Navigation (PN)
   * - PI контроль с feedforward
   * - Intercept алгоритм с ADRC
   * - Оценка скорости (Vc)
   * - Ограничение ускорений
   *
   * @thread_safe Частично (требует внешней синхронизации для shared state)
   *
   * @example
   * ```cpp
   * GuidanceConfig config;
   * config.pn_gain_N = 3.0;
   * config.max_lat_acc = 3.0;
   *
   * GuidanceController controller(config, 0.02);  // dt = 20ms
   *
   * GuidanceOutput output = controller.computeAcceleration(
   *   u_los, lambda, Vc, dt
   * );
   * ```
   */
  class GuidanceController
  {
  public:
    /**
     * @brief Конструктор
     * @param config Конфигурация из YAML
     * @param dt Шаг времени (сек), обычно 0.02 для 50Hz
     */
    explicit GuidanceController(const GuidanceConfig &config, double dt = 0.02);

    void setAttackMode(bool enable);

    /**
     * @brief Деструктор
     */
    ~GuidanceController() = default;

    /**
     * @brief Расчёт команд ускорения
     * @param u_los Нормированный LOS вектор (nav frame)
     * @param lambda Угловая скорость LOS (nav frame)
     * @param pitch_rad Тангаж дрона (rad) для оценки Vc
     * @param yaw_rad Рыскание дрона (rad) для трансформации
     * @param roll_rad Крен дрона (rad) для полной трансформации body→nav
     * @return GuidanceOutput с командой ускорения
     */
    GuidanceOutput computeAcceleration(
        const Eigen::Vector3d &u_los,
        const Eigen::Vector3d &lambda,
        double pitch_rad,
        double yaw_rad,
        double roll_rad = 0.0);

    /**
     * @brief Расчёт команд ускорения (альтернативный метод с Vc)
     * @param u_los Нормированный LOS вектор
     * @param lambda Угловая скорость LOS
     * @param Vc Оценка скорости (m/s)
     * @param yaw_rad Рыскание дрона (rad)
     * @param roll_rad Крен дрона (rad) для полной трансформации body→nav
     * @return GuidanceOutput с командой ускорения
     */
    GuidanceOutput computeAccelerationWithVc(
        const Eigen::Vector3d &u_los,
        const Eigen::Vector3d &lambda,
        double Vc,
        double yaw_rad,
        double roll_rad = 0.0);

    /**
     * @brief Оценка скорости на основе тангажа
     * @param pitch_rad Тангаж дрона (rad)
     * @param max_tilt_rad Максимальный тангаж (rad)
     * @return Оценка скорости (m/s)
     */
    double estimateVcFromPitch(double pitch_rad, double max_tilt_rad);

    /**
     * @brief Оценка скорости на основе ускорения
     * @param acc_sp Команда ускорения (m/s²)
     * @return Оценка скорости (m/s)
     */
    double estimateVcFromAcc(const Eigen::Vector3d &acc_sp);

    /**
     * @brief Сброс внутренних состояний (интегралы, фильтры)
     */
    void reset();

    /**
     * @brief Обновление конфигурации на лету
     * @param config Новая конфигурация
     */
    void updateConfig(const GuidanceConfig &config);

    /**
     * @brief Получение последней команды ускорения
     * @return Команда ускорения (nav frame)
     */
    Eigen::Vector3d getLastAccelerationCommand() const;

    /**
     * @brief Получение отладочной информации
     * @return GuidanceDebug с отладочными данными
     */
    GuidanceDebug getDebugInfo() const;

    /**
     * @brief Включение/выключение режима PI контроля
     * @param enable true = включить PI
     */
    void setPIControlEnabled(bool enable);

    /**
     * @brief Включение/выключение режима top attack
     * @param enable true = включить top attack
     */
    void setTopAttackMode(bool enable);

    /**
     * @brief Включение/выключение parallel acceleration
     * @param enable true = включить parallel acc
     */
    void setParallelAccEnabled(bool enable);

    /**
     * @brief Установка GPS скорости для fallback оценки Vc
     * @param velocity_mps Скорость дрона (м/с)
     */
    void setGpsVelocity(double velocity_mps) { gps_velocity_ = velocity_mps; }

    void setAltitude(double alt_m) { current_alt_m_ = alt_m; altitude_valid_ = true; }
    void setMinAttackAltitude(double alt_m) { min_attack_alt_m_ = alt_m; }

    // 🧠 === ONNX Neural Network Inference ===

    /**
     * @brief Загрузка ONNX модели для нейросетевого наведения
     * @param model_path Путь к .onnx файлу
     * @param scaler_path Путь к scaler_params.npz (опционально)
     * @return true если модель загружена успешно
     */
    bool loadOnnxModel(
        const std::string &model_path,
        const std::string &scaler_path = "");

    /**
     * @brief Включение/выключение нейросетевого режима
     * @param enable true = нейросеть, false = алгоритм
     */
    void setUseNeuralNetwork(bool enable);

    /**
     * @brief Проверка, загружена ли ONNX модель
     * @return true если модель загружена
     */
    bool isNeuralNetworkLoaded() const;

  private:
    /**
     * @brief Внутренняя реализация расчёта ускорения (общая для обоих публичных методов)
     * @param u_los Нормированный LOS вектор (nav frame)
     * @param lambda Угловая скорость LOS (nav frame)
     * @param Vc Скорость сближения (m/s)
     * @param roll_rad Крен (rad)
     * @param pitch_rad Тангаж (rad)
     * @param yaw_rad Рыскание (rad)
     * @return GuidanceOutput с командой ускорения
     */
    GuidanceOutput computeAccelerationImpl_(
        const Eigen::Vector3d &u_los,
        const Eigen::Vector3d &lambda,
        double Vc,
        double roll_rad,
        double pitch_rad,
        double yaw_rad);

    GuidanceConfig config_;
    double dt_;
    mutable std::mutex mutex_; // Защита состояния

    // === Состояния PI контроллера ===
    double los_err_int_y_{0.0};
    double los_err_int_z_{0.0};

    // === Состояния Intercept ===
    double intercept_x_int_{0.0};
    double intercept_y_int_{0.0};
    double intercept_los_err_x_prev_{0.0};
    double intercept_los_err_y_prev_{0.0};
    double intercept_x_d_flt_{0.0};
    double intercept_y_d_flt_{0.0};

    // === ADRC экземпляры ===
    ADRC adrc_intercept_ax_td_;
    ADRC adrc_intercept_ay_td_;
    bool adrc_initialized_{false};

    // === Состояния скорости ===
    double vc_pre_{0.0};
    double acc_parallel_pre_{0.0};
    double a_front_sp_{0.0};
    double ay_sp_{0.0};
    double vc_linear_min_{0.0};
    double vc_linear_max_{0.0};
    double vc_linear_k_{0.0};

    // === Последняя команда ===
    Eigen::Vector3d last_acc_cmd_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acc_frd_sp_{Eigen::Vector3d::Zero()};

    // === GPS скорость для fallback Vc ===
    double gps_velocity_{0.0};

    // === Altitude protection ===
    double current_alt_m_{0.0};
    bool altitude_valid_{false};
    double min_attack_alt_m_{3.0};

    // === Отладка ===
    GuidanceDebug debug_info_;

    // 🧠 === ONNX Neural Network ===
    bool use_nn_{false};                                    ///< Флаг: использовать нейросеть
    bool scaler_loaded_{false};                             ///< Scaler параметры загружены
    std::array<float, 9> nn_mean_{};                        ///< X_mean (9,) для нормализации
    std::array<float, 9> nn_scale_{};                       ///< X_scale (9,) для нормализации
    std::array<float, 3> nn_y_max_{};                       ///< y_max (3,) для денормализации
    std::unique_ptr<Ort::Session> onnx_session_;            ///< ONNX Runtime сессия
    Ort::MemoryInfo onnx_memory_info_{nullptr};             ///< Memory info для тензоров
    std::vector<const char *> onnx_input_names_;            ///< Имена входов
    std::vector<const char *> onnx_output_names_;           ///< Имена выходов
    static Ort::Env onnx_env_;                              ///< ONNX Runtime окружение (static — один на все экземпляры)

    /**
     * @brief Вычисление ускорения через нейросеть
     * @param u_los Нормированный LOS вектор
     * @param lambda Угловая скорость LOS
     * @param roll_rad Крен (rad)
     * @param pitch_rad Тангаж (rad)
     * @param yaw_rad Рыскание (rad)
     * @return GuidanceOutput с командой ускорения
     */
    GuidanceOutput computeNN_(
        const Eigen::Vector3d &u_los,
        const Eigen::Vector3d &lambda,
        double roll_rad,
        double pitch_rad,
        double yaw_rad);

    /**
     * @brief Загрузка scaler параметров из .npz файла
     * @param scaler_path Путь к scaler_params.npz
     * @return true если загружено успешно
     */
    bool loadScalerParams_(const std::string &scaler_path);

    // === Константы ===
    static constexpr double CONSTANTS_ONE_G = 9.80665;
    static constexpr double EPSILON = 1e-6;

    // === Внутренние методы ===

    /**
     * @brief Вычисление PN ускорения с PI контролем
     */
    Eigen::Vector3d computePnPI_(
        const Eigen::Vector3d &u_los,
        const Eigen::Vector3d &lambda,
        double Vc,
        double dt);

    /**
     * @brief Вычисление Intercept ускорения
     */
    Eigen::Vector3d computeInterceptPI_(
        const Eigen::Vector3d &u_los,
        const Eigen::Vector3d &lambda,
        double Vc,
        double dt);

    /**
     * @brief Вычисление Intercept ускорения (без lambda)
     */
    Eigen::Vector3d computeInterceptPI_(
        const Eigen::Vector3d &u_los,
        double dt);

    /**
     * @brief Ограничение ускорения
     */
    Eigen::Vector3d clampAcceleration_(const Eigen::Vector3d &acc);

    /**
     * @brief Добавление parallel acceleration
     */
    Eigen::Vector3d addParallelAcceleration_(
        const Eigen::Vector3d &acc,
        const Eigen::Vector3d &u_los,
        double Vc,
        double dt);

    /**
     * @brief Трансформация ускорения из body (FRD) в nav (NED) frame
     * @param acc_frd Ускорение в FRD (body) frame
     * @param roll_rad Крен (rad)
     * @param pitch_rad Тангаж (rad)
     * @param yaw_rad Рыскание (rad)
     * @return Ускорение в NED (nav) frame
     */
    Eigen::Vector3d transformAccToNav_(
        const Eigen::Vector3d &acc_frd,
        double roll_rad,
        double pitch_rad,
        double yaw_rad);

    /**
     * @brief Инициализация ADRC
     */
    void initADRC_();

    /**
     * @brief Матрица косого симметричного вектора
     */
    Eigen::Matrix3d skewSymmetric_(const Eigen::Vector3d &v);

    /**
     * @brief Clamp значения
     */
    double clamp_(double value, double min_val, double max_val);
  };

} // namespace los_guidance

#endif // LOS_GUIDANCE_GUIDANCE_CONTROLLER_HPP_