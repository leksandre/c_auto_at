/**
 * @file main.cpp
 * @brief Точка входа ROS2 узла LOS Guidance
 * @note Использует MultiThreadedExecutor для параллельной обработки callbacks
 * 
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */


#include <rclcpp/rclcpp.hpp>
#include <los_guidance/node.hpp>

/**
 * @brief Главная функция ROS2 узла
 *
 * @threading_model MultiThreadedExecutor + CallbackGroups
 *
 * Архитектура потоков:
 * ┌─────────────────────────────────────────────────────────────┐
 * │  Поток 1 (Control):                                        │
 * │  ├── PX4 telemetry (attitude, gyro, GPS, vehicle_status)   │
 * │  ├── Control loop timer (20ms)                             │
 * │  └── Attack signal, services                               │
 * │                                                             │
 * │  Поток 2 (AI):                                             │
 * │  ├── tracker_result callback                               │
 * │  ├── track_bbox callback                                   │
 * │  └── AI heartbeat                                          │
 * │                                                             │
 * │  Lock-free queue: AI → Control (acceleration commands)     │
 * └─────────────────────────────────────────────────────────────┘
 *
 * @param argc Количество аргументов командной строки
 * @param argv Аргументы командной строки
 * @return int Код возврата (0 = успех)
 */


int main(int argc, char **argv) {
  // === 1. Инициализация ROS2 ===
  rclcpp::init(argc, argv);
  
  // === 2. Создание узла ===
  // Используем shared_ptr для совместимости с Executor
  auto los_node = std::make_shared<los_guidance::Node>();
  
  los_node->postConstruct();

  // === 3. Логирование старта ===
  RCLCPP_INFO(los_node->get_logger(), "===========================================");
    
  RCLCPP_INFO(los_node->get_logger(), "Threading Model: MultiThreadedExecutor");
  RCLCPP_INFO(los_node->get_logger(), "===========================================");
  
  // === 4. Создание MultiThreadedExecutor ===
  // Callback groups автоматически распределяются по потокам:
  // - ai_callback_group_ → Поток AI
  // - control_callback_group_ → Поток управления
  //
  // Lock-free queue (new_acc_cmd_available_) передаёт команды AI → Control
  // без блокировки мьютекса — AI поток никогда не ждёт control loop
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions{},
    2
  );


  // === 5. Добавление узла в executor ===
  executor.add_node(los_node);
  
  // === 6. Запуск цикла обработки ===
  // spin() будет работать до тех пор, пока:
  // - Не будет вызван rclcpp::shutdown()
  // - Не будет нажат Ctrl+C
  // - Не произойдёт ошибка
  try {
    executor.spin();
  } catch (const std::exception &e) {
    RCLCPP_FATAL(los_node->get_logger(), "Executor exception: %s", e.what());
  }
  
  // === 7. Очистка ===
  RCLCPP_INFO(los_node->get_logger(), "LOS Guidance Node Shutting Down...");
  
  // === 8. Shutdown ROS2 ===
  rclcpp::shutdown();
  
  return 0;
}