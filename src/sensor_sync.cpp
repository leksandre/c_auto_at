/**
 * @file sensor_sync.cpp
 * @brief Реализация класса SensorSync (синхронизация сообщений)
 * @note Рефакторинг SyncApproxPolicy для модульности
 * 
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#include "los_guidance/sensor_sync.hpp"
#include <algorithm>
#include <cmath>
#include <iostream> 

namespace los_guidance {

// ============================================================================
// Конструктор
// ============================================================================

SensorSync::SensorSync(const SyncConfig& config)
  : config_(config), sync_count_(0), add_count_(0)
{
}

// ============================================================================
// Публичные методы
// ============================================================================

void SensorSync::registerSource(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (buffers_.find(name) == buffers_.end()) {
    buffers_[name] = std::deque<std::pair<uint64_t, std::shared_ptr<void>>>();
    
    if (config_.debug) {
      std::cout << "[SensorSync] Registered source: " << name << std::endl;
    }
  }
}

void SensorSync::addMessage(
  const std::string& source,
  uint64_t ts_us,
  std::shared_ptr<void> msg
) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = buffers_.find(source);
  if (it == buffers_.end()) {
    if (config_.debug) {
      std::cerr << "[SensorSync] Warning: Source '" << source 
                << "' not registered" << std::endl;
    }
    return;
  }
  
  auto& buf = it->second;
  buf.push_back({ts_us, msg});
  
  // Ограничение размера буфера
  while (buf.size() > config_.buffer_size) {
    buf.pop_front();
  }
  
  // Очистка старых сообщений
  while (!buf.empty() && buf.front().first < ts_us - config_.retention_us) {
    buf.pop_front();
  }
  
  if (config_.debug) {
    add_count_++;
    if (add_count_ % 500 == 0) {
      std::cout << "[SensorSync] Added message to '" << source
                << "' at " << ts_us << " us, buffer size: "
                << buf.size() << std::endl;
    }
  }
}

std::map<std::string, std::shared_ptr<void>> SensorSync::sync(uint64_t ref_time) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::map<std::string, std::shared_ptr<void>> result;

  for (auto& [name, buf] : buffers_) {
    result[name] = getNearest_(name, ref_time);

    if (result[name]) {
      dropUntil_(name, ref_time);
    }
  }

  if (config_.debug) {
    sync_count_++;
    if (sync_count_ % 500 == 0) {
      std::cout << "[SensorSync] Sync completed at " << ref_time 
                << " us, sources: " << result.size() << std::endl;
    }
  }

  return result;
}

void SensorSync::setDebug(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.debug = enable;
}

// ============================================================================
// Приватные методы
// ============================================================================

std::shared_ptr<void> SensorSync::getNearest_(
  const std::string& source,
  uint64_t ref
) {
  auto it = buffers_.find(source);
  if (it == buffers_.end() || it->second.empty()) {
    return nullptr;
  }
  
  const auto& buf = it->second;
  std::shared_ptr<void> nearest = nullptr;
  uint64_t min_diff = std::numeric_limits<uint64_t>::max();
  uint64_t nearest_ts = 0;
  
  for (const auto& [ts, msg] : buf) {
    uint64_t diff = std::llabs(static_cast<int64_t>(ts - ref));
    if (diff < min_diff) {
      min_diff = diff;
      nearest = msg;
      nearest_ts = ts;
    }
  }
  
  if (config_.debug && nearest) {
    std::cout << "[SensorSync] Source: " << source
              << " | Ref: " << ref
              << " | Nearest: " << nearest_ts
              << " | Diff: " << min_diff << " us" << std::endl;
  }
  
  return nearest;
}

void SensorSync::dropUntil_(
  const std::string& source,
  uint64_t ref_time
) {
  auto it = buffers_.find(source);
  if (it == buffers_.end()) {
    return;
  }
  
  auto& buf = it->second;
  while (!buf.empty() && buf.front().first < ref_time) {
    buf.pop_front();
  }
}

} // namespace los_guidance