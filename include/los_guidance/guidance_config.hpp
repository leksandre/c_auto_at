#ifndef LOS_GUIDANCE_GUIDANCE_CONFIG_HPP_
#define LOS_GUIDANCE_GUIDANCE_CONFIG_HPP_

#include <rclcpp/rclcpp.hpp>
#include <string>

namespace los_guidance {

/**
 * @brief Конфигурация системы наведения LOS
 * 
 * Все параметры вынесены в config/params.yaml
 * Этот класс управляет загрузкой и валидацией параметров
 */
struct GuidanceConfig {
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
    
    /**
     * @brief Загрузка параметров из ROS2 параметров
     * @param node ROS2 node для получения параметров
     */
    void loadFromROS(rclcpp::Node* node);
    
    /**
     * @brief Валидация параметров
     * @return true если все параметры корректны
     */
    bool validate() const;
    
    /**
     * @brief Логирование текущей конфигурации
     * @param logger ROS2 logger
     */
    void logConfig(rclcpp::Logger logger) const;
};

}  // namespace los_guidance

#endif  // LOS_GUIDANCE_GUIDANCE_CONFIG_HPP_