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
 * @threading_model MultiThreadedExecutor
 * - Callbacks обрабатываются в параллельных потоках
 * - Timer callback выполняется в отдельном потоке
 * - Shared state защищён mutex в контроллерах
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
  // Преимущества перед rclcpp::spin():
  // - Параллельная обработка callbacks из разных топиков
  // - Timer callback не блокирует обработку сообщений
  // - Лучшая отзывчивость при высокой нагрузке
  // 
  // ⚠️ ВАЖНО: Все shared state в контроллерах должны быть защищены mutex!
  rclcpp::executors::MultiThreadedExecutor executor;
  



//   // Вариант 1: Явное указание количества потоков
// rclcpp::executors::MultiThreadedExecutor executor(
//   rclcpp::ExecutorOptions(), 
//   4  // 4 потока
// );

// // Вариант 2: Использование количества CPU cores
// unsigned int num_threads = std::thread::hardware_concurrency();
// rclcpp::executors::MultiThreadedExecutor executor(
//   rclcpp::ExecutorOptions(), 
//   num_threads
// );

// // Вариант 3: По умолчанию (рекомендуется)
// // Executor сам выберет оптимальное количество
// rclcpp::executors::MultiThreadedExecutor executor;


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