/**
 * @file sensor_sync.hpp
 * @brief Синхронизация сообщений по времени
 * @note Поддерживает PX4 и ROS2 сообщения с разными timestamp
 */
#ifndef LOS_GUIDANCE_SENSOR_SYNC_HPP_
#define LOS_GUIDANCE_SENSOR_SYNC_HPP_

#pragma once
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace los_guidance {

/**
 * @struct SyncConfig
 * @brief Конфигурация синхронизации
 */
struct SyncConfig {
  size_t buffer_size{100};
  int retention_us{200000};  // 200ms
  bool debug{false};
};

/**
 * @class SensorSync
 * @brief Политика синхронизации по ближайшему сообщению
 * 
 * @thread_safe Да (все методы потокобезопасны)
 */
class SensorSync {
public:
  explicit SensorSync(const SyncConfig& config = SyncConfig());
  
  /**
   * @brief Регистрация источника данных
   * @param name Имя источника (например "att", "gyro")
   */
  void registerSource(const std::string& name);
  
  /**
   * @brief Добавление сообщения в буфер
   * @param source Имя источника
   * @param ts_us Timestamp в микросекундах
   * @param msg Указатель на сообщение
   */
  void addMessage(
    const std::string& source,
    uint64_t ts_us,
    std::shared_ptr<void> msg
  );
  
  /**
   * @brief Синхронизация всех источников по опорному времени
   * @param ref_time Опорный timestamp
   * @return Map с синхронизированными сообщениями
   */
  std::map<std::string, std::shared_ptr<void>> sync(uint64_t ref_time);
  
  /**
   * @brief Включение/выключение отладки
   */
  void setDebug(bool enable);

private:
  SyncConfig config_;
  std::map<std::string, std::deque<std::pair<uint64_t, std::shared_ptr<void>>>> buffers_;
  mutable std::mutex mutex_;  // Защита буферов
  size_t sync_count_{0};      // Счётчик синхронизаций
  size_t add_count_{0};       // Счётчик добавлений сообщений

  std::shared_ptr<void> getNearest_(const std::string& source, uint64_t ref);
  void dropUntil_(const std::string& source, uint64_t ref_time);
};

} // namespace los_guidance
#endif