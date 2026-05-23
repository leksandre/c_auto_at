# Улучшение алгоритма поиска и удержания цели при потере видимости

## Задача
Улучшить алгоритм так, чтобы:

1. В момент начала атаки данные цели (класс, размер, направление движения) фиксировались в памяти.
2. В случае если цель резко теряется за пределами видимой области:
   - Сохранялась точка, обозначающая край, за которым пропала цель.
   - Сохранялись данные цели (класс, размер, направление).
   - Система пыталась довернуть в сторону, куда пропала цель, чтобы снова её отыскать.

## Анализ текущей архитектуры

### Компоненты системы
1. **`TargetTracker`** - отвечает за обработку данных от детектора (TrackerResult / Detection2D), сглаживание координат, коррекцию на roll/pitch дрона и преобразование координат (пиксель → камера → тело → навигация).
2. **`Node`** - главный ROS2 узел, координирует работу всех компонентов, управляет фазами полета (GPS_APPROACH, VISUAL_ATTACK, VISUAL_SEARCH).
3. **`GuidanceController`** - реализует алгоритмы наведения (PN, PI, Intercept).

### Текущее поведение при потере цели
Когда цель теряется, в `node.cpp` происходит следующее:

```cpp
if (!attack_ready_.load(std::memory_order_acquire)) {
  // 🔄 Цель потеряна — возвращаемся к подлёту по GPS (не останавливаемся!)
  RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                       "⚠️ Target lost! Returning to GPS_APPROACH (target_gps_ preserved)");
  auto_phase_ = AutoPhase::GPS_APPROACH;
  start_attack_ = false;
  attack_enabled_ = false;
}
```

Текущая реализация просто возвращается к GPS-сближению, но не сохраняет информацию о том, в каком направлении была потеряна цель.

## Предлагаемое решение

### 1. Сохранение данных цели при начале атаки

Добавить структуру для хранения данных цели в `Node`:

```cpp
struct TargetData {
  std::string class_name;
  double width;
  double height;
  Eigen::Vector3d direction; // направление движения
  uint64_t timestamp_us;
  bool valid;
};

TargetData last_known_target_;
```

Заполнять её в `trackerResultCallback` при начале атаки:

```cpp
if (config_.enable_attack_init && target.is_valid) {
  last_known_target_.class_name = msg->target_info.class_name;
  last_known_target_.width = info.bbox_width;
  last_known_target_.height = info.bbox_height;
  last_known_target_.direction = los_estimator_->getLosVector();
  last_known_target_.timestamp_us = info.timestamp_us;
  last_known_target_.valid = true;
}
```

### 2. Сохранение точки потери цели

Добавить переменную для хранения точки потери цели:

```cpp
struct LostTargetInfo {
  Eigen::Vector3d position; // позиция дрона при потере цели
  Eigen::Vector3d los_vector; // вектор LOS при потере
  uint64_t timestamp_us;
  bool valid;
};

LostTargetInfo lost_target_info_;
```

Обновлять её в `trackerResultCallback` при получении валидной цели:

```cpp
if (target.is_valid) {
  lost_target_info_.position = current_position_;
  lost_target_info_.los_vector = target.los_nav;
  lost_target_info_.timestamp_us = msg->timestamp / 1000;
  lost_target_info_.valid = true;
}
```

### 3. Поиск цели после потери

Модифицировать логику перехода в `GPS_APPROACH` при потере цели:

```cpp
if (!attack_ready_.load(std::memory_order_acquire)) {
  // Сохраняем информацию о потере цели
  if (lost_target_info_.valid) {
    RCLCPP_WARN(this->get_logger(), "⚠️ Target lost at position (NED): [%.2f, %.2f, %.2f]", 
                lost_target_info_.position.x(), lost_target_info_.position.y(), lost_target_info_.position.z());
    
    // Вычисляем направление, куда нужно повернуть для поиска
    Eigen::Vector3d search_direction = lost_target_info_.los_vector;
    
    // Нормализуем и проецируем на горизонтальную плоскость
    search_direction.z() = 0;
    search_direction.normalize();
    
    // Запоминаем направление для поиска
    search_direction_ = search_direction;
    search_mode_active_ = true;
    
    RCLCPP_INFO(this->get_logger(), "🧭 Starting search in direction: [%.2f, %.2f]", 
                search_direction.x(), search_direction.y());
  }
  
  auto_phase_ = AutoPhase::GPS_APPROACH;
  start_attack_ = false;
  attack_enabled_ = false;
}
```

Добавить фазу поиска `SEARCH_MODE`:

```cpp
enum class AutoPhase {
  IDLE,
  GPS_APPROACH,
  VISUAL_ATTACK,
  VISUAL_SEARCH,
  SEARCH_MODE  // Новая фаза для активного поиска
};

Eigen::Vector3d search_direction_;
bool search_mode_active_ = false;
```

В `controlLoopTimerCallback` добавить обработку фазы поиска:

```cpp
if (auto_phase_ == AutoPhase::SEARCH_MODE) {
  if (search_mode_active_) {
    // Поворачиваемся в сторону, куда пропала цель
    double yaw_rate_cmd = -config_.kp_yaw_track * 
      std::atan2(search_direction_.y(), search_direction_.x());
    
    // Ограничиваем скорость поворота
    yaw_rate_cmd = std::clamp(yaw_rate_cmd, -30.0 * DEG2RAD, 30.0 * DEG2RAD);
    
    // Публикуем команду поворота
    px4_msgs::msg::TrajectorySetpoint msg;
    msg.position = {NAN, NAN, NAN};
    msg.velocity = {NAN, NAN, NAN};
    msg.acceleration = {NAN, NAN, NAN};
    msg.yawspeed = static_cast<float>(yaw_rate_cmd);
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    
    trajectory_setpoint_pub_->publish(msg);
    vehicle_ctrl_->publishOffboardControlMode();
    
    // Проверяем, видна ли цель снова
    if (attack_ready_.load(std::memory_order_acquire)) {
      RCLCPP_INFO(this->get_logger(), "🎯 Target reacquired! Switching back to VISUAL_ATTACK");
      auto_phase_ = AutoPhase::VISUAL_ATTACK;
      attack_enabled_ = true;
      search_mode_active_ = false;
    }
  }
  return;
}
```

### 4. Интеграция с существующей логикой

Модифицировать `executeSearchMode` для использования сохраненных данных цели:

```cpp
void Node::executeSearchMode() {
  // ... существующая логика ...
  
  // Если у нас есть данные о последней цели, ищем именно её
  if (last_known_target_.valid) {
    RCLCPP_INFO(this->get_logger(), "🔍 Searching for target class: %s, size: %.2fx%.2f", 
                last_known_target_.class_name.c_str(), 
                last_known_target_.width, last_known_target_.height);
  }
  
  // ... остальная логика ...
}
```

## Требуемые изменения

1. **Добавить новые структуры данных в `Node`**:
   - `TargetData last_known_target_`
   - `LostTargetInfo lost_target_info_`
   - `Eigen::Vector3d search_direction_`
   - `bool search_mode_active_`

2. **Модифицировать `NodeConfig`** для добавления параметров поиска:
   - `double search_yaw_rate` - максимальная скорость поворота при поиске
   - `double search_timeout` - таймаут поиска

3. **Обновить `trackerResultCallback`** для сохранения данных цели.

4. **Модифицировать логику потери цели в `controlLoopTimerCallback`**.

5. **Добавить новую фазу `SEARCH_MODE`** и её обработку.

6. **Обновить `executeSearchMode`** для использования сохраненных данных.

## Потенциальные проблемы и решения

1. **Проблема**: Цель может двигаться, и её направление может измениться.
   **Решение**: Добавить таймер устаревания данных цели. Если прошло более N секунд, считать данные недействительными.

2. **Проблема**: Дрон может быть слишком далеко от точки потери цели.
   **Решение**: Перед началом поиска сначала вернуться к приблизительной позиции потери цели.

3. **Проблема**: Множественные цели одного класса.
   **Решение**: Использовать размер и движение как дополнительные признаки для идентификации.

## Дополнительные улучшения

1. **Прогнозирование траектории**: Использовать сохраненное направление движения для прогнозирования положения цели.
2. **Паттерны поиска**: Реализовать спиральный или зигзагообразный паттерн поиска.
3. **Мультимодальный поиск**: Использовать данные от других сенсоров (например, радиолокационные) для уточнения положения.
4. **Обучение на лету**: На основе успешных поисков корректировать параметры поиска.

## Заключение

Предложенное решение позволит системе более эффективно справляться с потерей цели, сохраняя контекст и активно ища цель в направлении её исчезновения, что значительно повысит надежность и эффективность атаки.