#pragma once

#include <rclcpp/rclcpp.hpp>

namespace los_guidance {

struct GuidanceConfig
{
  // Базовые параметры управления
  double max_speed_{3.0};                    // м/с
  double max_vert_speed_{2.0};               // м/с  
  double max_yaw_rate_{45.0 * M_PI / 180.0}; // рад/с

  // Параметры фильтрации ручного управления
  double manual_roll_deadzone{0.1};           // Нулевая зона по крену
  double manual_pitch_deadzone{0.1};          // Нулевая зона по тангажу
  double manual_yaw_deadzone{0.1};            // Нулевая зона по рысканью
  double manual_throttle_deadzone{0.05};      // Нулевая зона по газу

  // Ограничения скорости изменения (для плавности)
  double max_roll_slew_rate{2.0};             // Максимальная скорость изменения roll (ед/с)
  double max_pitch_slew_rate{2.0};            // Максимальная скорость изменения pitch (ед/с)
  double max_yaw_slew_rate{60.0 * M_PI / 180.0}; // Максимальная скорость изменения yaw (рад/с)
  double max_throttle_slew_rate{1.0};         // Максимальная скорость изменения throttle (ед/с)

  // Фильтр низких частот (первый порядок)
  double manual_filter_time_constant{0.2};    // Время нарастания (сек), 0 = выкл

  // Аварийные блокировки
  bool enable_emergency_lockout{true};         // Включить режим блокировки при опасных сигналах
  double throttle_cutoff_threshold{0.01};     // Порог отключения двигателя (низкий газ)
  double throttle_spikes_threshold{0.9};      // Порог блокировки резких всплесков газа
  double spike_filter_time_window{0.1};       // Окно фильтрации всплесков (с)

  // Режимы плавности (как у корабля)
  bool enable_ship_mode{true};                 // Включить медленный отклик
  double ship_mode_response_factor{0.3};       // Фактор замедления (0.1 = очень медленно, 1.0 = нормально)

  // Параметры переключения режимов
  bool allow_vision_attack{true};             // Разрешить режим визуальной атаки
  bool allow_autoland{true};                   // Разрешить автопосадку




      // === Параметры навигации ===
    double kp_yaw{0.5};           // Пропорциональный коэффициент yaw
    double kp_thrust{0.1};        // Пропорциональный коэффициент тяги
    double kp_pitch{0.1};         // Пропорциональный коэффициент pitch
    double kp_roll{0.3};          // Пропорциональный коэффициент roll
    
    // === Параметры камеры ===
    double camera_fx{410.0};      // Фокусное расстояние X (пиксели)
    double camera_fy{410.0};      // Фокусное расстояние Y (пиксели)
    double camera_cx{640.0};      // Оптический центр X (пиксели)
    double camera_cy{360.0};      // Оптический центр Y (пиксели)
    double vertical_fov{60.0};    // Вертикальное поле зрения (градусы)
    double camera_pitch{5.0};     // Угол установки камеры (градусы)
    double camera_yaw{0.0};       // Угол установки камеры по yaw (градусы)
    double camera_roll{0.0};      // Угол установки камеры по roll (градусы)
    
    // === Ограничения управления ===
    double max_compensation_angle{5.0};   // Макс угол компенсации (градусы)
    double max_pitch_angle{8.0};          // Макс угол тангажа (градусы)
    double max_xy_accel{5.0};             // Макс ускорение XY (м/с²)
    double max_z_accel{5.0};              // Макс ускорение Z (м/с²)
    double yawrate_max{20.0};             // Макс угловая скорость yaw (град/с)
    
    // === Параметры фильтрации ===
    double thrust_alpha{0.5};     // Коэффициент сглаживания тяги
    double roll_alpha{0.4};       // Коэффициент сглаживания roll
    double alpha{0.1};            // Общий коэффициент фильтрации
    double alpha_boost{0.1};      // Коэффициент усиления
    
    // === Параметры наведения ===
    double N_gain{3.0};           // Коэффициент усиления PN (Proportional Navigation)
    double yaw_max_err{0.15};     // Макс ошибка yaw (радианы)
    double dy_threshold{240.0};   // Порог ошибки по Y (пиксели)
    
    // === Параметры ускорения ===
    double forward_boost{1.0};    // Усиление движения вперёд
    double min_forward_boost{0.0}; // Мин усиление
    double forward_acc_step{2.0}; // Шаг ускорения (м/с²)
    double acc_max_z{2.0};        // Макс ускорение по Z
    
    // === Флаги режимов ===
    bool enable_estimate_close_vel{true};   // Включить оценку скорости сближения
    bool enable_yaw_ctr{false};             // Включить контроль yaw
    bool enable_use_target_est_vel{false};  // Использовать оценку скорости цели
    bool enable_linear{true};               // Включить линейную интерполяцию
    bool enable_attack_init{true};          // Включить инициализацию атаки
    
    // === Параметры целеуказания ===
    double boundbox_y_err_center{0.0};      // Центр ошибки bounding box
    double boundbox_y_err_max{-0.4};        // Макс ошибка bounding box
    
    // === Скорости ===
    double forward_speed{3.0};      // Скорость движения вперёд (м/с)
    double z_still_eps{0.03};       // Порог неподвижности по Z (м)
    
    // === Новый параметр для отложенного ускорения ===
    double timeout_for_centr_target{2.0};   // Время (сек) для центрирования цели перед подачей ускорения вперед


    // // === Параметры ручного управления (добавить в конец структуры) ===
    // // Deadzones
    // float manual_roll_deadzone{0.05f};
    // float manual_pitch_deadzone{0.05f};
    // float manual_yaw_deadzone{0.05f};
    // float manual_throttle_deadzone{0.05f};

    // // Slew rate limits (ед/сек)
    // float max_roll_slew_rate{2.0f};
    // float max_pitch_slew_rate{2.0f};
    // float max_yaw_slew_rate{2.0f};
    // float max_throttle_slew_rate{2.0f};

    // // Filter constants
    // float manual_filter_time_constant{0.1f};  // seconds
    // float ship_mode_response_factor{1.0f};     // response multiplier

    // // Emergency safety
    // float spike_filter_time_window{0.5f};      // seconds
    // bool enable_emergency_lockout{true};
    // float throttle_spikes_threshold{0.9f};
    // float throttle_cutoff_threshold{0.1f};

    // // Feature flags
    // bool allow_vision_attack{true};
    // bool allow_autoland{true};

    
  // Загрузка параметров из ноды
  void loadFromNode(const rclcpp::Node::SharedPtr& node);
};

} // namespace los_guidance