#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${ROOT_DIR}/backup_$(date +%Y%m%d_%H%M%S)"

echo "📦 Создание бэкапа в $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR/include/los_guidance" "$BACKUP_DIR/src"
cp "$ROOT_DIR/include/los_guidance/commander.hpp" "$BACKUP_DIR/include/los_guidance/"
cp "$ROOT_DIR/include/los_guidance/node.hpp"       "$BACKUP_DIR/include/los_guidance/"
cp "$ROOT_DIR/src/commander.cpp" "$BACKUP_DIR/src/"
cp "$ROOT_DIR/src/node.cpp"      "$BACKUP_DIR/src/"

echo "🔧 Патч 1: Добавляем парсер search/searsh в commander.cpp..."
sed -i '/\/\/ 2\. Обработка set_mode: auto/,/return;/{
  /return;/a\
  // 🆕 2.1 Обработка set_mode: search / searsh\
  if (data.find("\\\"set_mode\\\"") != std::string::npos &&\
      (data.find("search") != std::string::npos || data.find("searsh") != std::string::npos))\
  {\
      RCLCPP_INFO(logger_, "🔍 Received command: set_mode search");\
      if (search_mode_callback_) search_mode_callback_();\
      sendResponse(client_fd, "{\\"status\\":\\"success\\",\\"command\\":\\"SET_MODE\\",\\"mode\\":\\"search\\"}\\n");\
      return;\
  }
}' "$ROOT_DIR/src/commander.cpp"

echo "🔧 Патч 2: Обновляем callback атаки и добавляем search-лямбду в node.cpp..."
python3 << 'PYEOF'
import re

file = "src/node.cpp"
with open(file, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Исправляем callback start_attack: false -> возврат в VISUAL_SEARCH
old_attack_cb = '''[this](bool start)
{
std::lock_guard<std::mutex> lock(state_mutex_);
start_attack_ = start;
RCLCPP_INFO(this->get_logger(), "🌐 Commander: start_attack_ set to %s", start ? "true" : "false");
},'''

new_attack_cb = '''[this](bool arm_or_attack)
{
std::lock_guard<std::mutex> lock(state_mutex_);
if (!arm_or_attack && auto_phase_ == AutoPhase::VISUAL_ATTACK) {
auto_phase_ = AutoPhase::VISUAL_SEARCH;
guidance_ctrl_->reset();
RCLCPP_INFO(this->get_logger(), "⛔ Attack cancelled, returning to SEARCH mode");
}
start_attack_ = arm_or_attack;
RCLCPP_INFO(this->get_logger(), "🌐 Commander: start_attack_ set to %s", arm_or_attack ? "true" : "false");
},'''

code = code.replace(old_attack_cb, new_attack_cb)

# 2. Добавляем 14-й аргумент (search callback) в конструктор Commander
old_cmd_end = '''[this](double square_size)
{
std::lock_guard<std::mutex> lock(state_mutex_);
RCLCPP_INFO(this->get_logger(), "⏸️ HOLD_TO_SQUARE command, size: %.2f m", square_size);
active_command_ = ActiveCommand::HOLD_TO_SQUARE;
active_command_param_ = square_size;
start_attack_ = false;
attack_enabled_ = false;
auto_phase_ = AutoPhase::IDLE;
});'''

new_cmd_end = '''[this](double square_size)
{
std::lock_guard<std::mutex> lock(state_mutex_);
RCLCPP_INFO(this->get_logger(), "⏸️ HOLD_TO_SQUARE command, size: %.2f m", square_size);
active_command_ = ActiveCommand::HOLD_TO_SQUARE;
active_command_param_ = square_size;
start_attack_ = false;
attack_enabled_ = false;
auto_phase_ = AutoPhase::IDLE;
},
// 🆕 Колбэк для set_mode: search
[this]()
{
std::thread(&Node::executeSearchMode, this).detach();
});'''

code = code.replace(old_cmd_end, new_cmd_end)

# 3. Добавляем сохранение last_los_cam_y_ в trackerResultCallback
old_tracker_proc = '''TargetInfo target = target_tracker_->processTrackerResult(
msg,
roll,
pitch,
R_nb);'''

new_tracker_proc = '''TargetInfo target = target_tracker_->processTrackerResult(
msg,
roll,
pitch,
R_nb);
last_los_cam_y_.store(target.los_cam.y(), std::memory_order_relaxed); // 🆕'''

code = code.replace(old_tracker_proc, new_tracker_proc)

with open(file, "w", encoding="utf-8") as f:
    f.write(code)
PYEOF

echo "✅ Все патчи применены успешно!"
echo "📂 Бэкап сохранён в: $BACKUP_DIR"
echo ""
echo "🚀 Пересоберите пакет:"
echo "  cd ~/planer_motion/lastc && colcon build --packages-select los_guidance"
echo "  source install/setup.bash"
echo ""
echo "📡 Тест команды:"
echo "  echo '{\"set_mode\": \"search\"}' | nc 192.168.14.175 11472"