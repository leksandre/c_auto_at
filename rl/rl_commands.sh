#!/bin/bash
# ============================================================================
# RL Training Pipeline — Команды запуска/остановки
# ============================================================================
# Все команды выполняются ВНУТРИ контейнера sim_dev_container
# Подключение: docker exec -it sim_dev_container bash
# ============================================================================

# ────────────────────────────────────────────────────────────────────────────
# 1. ОСТАНОВКА ВСЕХ ПРОЦЕССОВ
# ────────────────────────────────────────────────────────────────────────────
# Убивает все процессы пайплайна: PX4, Gazebo, LightTrack, DDS, RL env, training
# Осторожно: это убьёт ВСЁ!

docker exec sim_dev_container bash -c "\
kill -9 \$(pgrep -f 'train_rl') 2>/dev/null; \
kill -9 \$(pgrep -f 'px4_bringup') 2>/dev/null; \
kill -9 \$(pgrep -f 'gz sim') 2>/dev/null; \
kill -9 \$(pgrep -f 'ros2 launch') 2>/dev/null; \
kill -9 \$(pgrep -f 'parameter_bridge') 2>/dev/null; \
kill -9 \$(pgrep -f 'MicroXRCE') 2>/dev/null; \
kill -9 \$(pgrep -f 'lighttrack') 2>/dev/null; \
kill -9 \$(pgrep -f 'LightTrackNode') 2>/dev/null; \
kill -9 \$(pgrep -f 'gazebo_reset') 2>/dev/null; \
kill -9 \$(pgrep -f 'moving_target') 2>/dev/null; \
kill -9 \$(pgrep -f 'px4 ') 2>/dev/null; \
sleep 2; \
echo 'All processes killed'"

# ────────────────────────────────────────────────────────────────────────────
# 2. ЗАПУСК PX4 SITL + GAZEBO (Terminal 1)
# ────────────────────────────────────────────────────────────────────────────
# Запускает PX4 SITL + Gazebo с миром car_track
# real_time_factor=5 в car_track.sdf (5x быстрее реального времени)
# --headless: PX4_GZ_HEADLESS=1 добавлен в launch (сервер без GUI)
#
# Параметры (меняются в px4_sim_attack_car.launch.py):
#   world_name: car_track (мир с Prius и роками)
#   model_pose: -1,0,0,0,0,0 (стартовая позиция дрона)
#   drone_id: 0

docker exec -d sim_dev_container bash -c "\
source /opt/ros/jazzy/setup.bash && \
source /simulation_ws/install/setup.bash && \
ros2 launch px4_bringup px4_sim_attack_car.launch.py > /tmp/px4.log 2>&1"

# Ждём пока PX4 будет готов (проверяем "Ready for takeoff!"):
#   tail -f /tmp/px4.log

# ────────────────────────────────────────────────────────────────────────────
# 3. DDS BRIDGE — MicroXRCEAgent (Terminal 3)
# ────────────────────────────────────────────────────────────────────────────
# Мост между PX4 (DDS/FastRTPS) и ROS2
# порт 8888 — UDP4 соединение с PX4
# Запускается ПОСЛЕ PX4 (через 15-20 сек после старта)

docker exec -d sim_dev_container bash -c "\
source /opt/ros/jazzy/setup.bash && \
source /simulation_ws/install/setup.bash && \
MicroXRCEAgent udp4 -p 8888 > /tmp/microxrce.log 2>&1"

# ────────────────────────────────────────────────────────────────────────────
# 4. LIGHTTRACK — Трекер цели (Terminal 2)
# ────────────────────────────────────────────────────────────────────────────
# Siamese tracker для автоматического обнаружения и трекинга Prius
# Подписки: /image_raw (камера), /car/navsat (GPS цели)
# Публикации: /tracker_result (bbox + confidence)
#
# Параметры (меняются в lighttrack_launch.py):
#   image_topic: /image_raw (topic камеры из Gazebo)
#   init_model: lighttrack_init.onnx (модель инициализации)
#   update_model: lighttrack_update.onnx (модель обновления)
#   auto_init: true (автоматический init через GPS)

docker exec -d sim_dev_container bash -c "\
source /opt/ros/jazzy/setup.bash && \
source /simulation_ws/install/setup.bash && \
ros2 launch lighttrack_ros lighttrack_launch.py > /tmp/lighttrack.log 2>&1"

# ────────────────────────────────────────────────────────────────────────────
# 5. RL PIPELINE — Reset service + Target node (Terminal 4)
# ────────────────────────────────────────────────────────────────────────────
# Запускает:
#   - gazebo_reset_service: сброс симуляции (teleport drone + target)
#   - moving_target_node: управление движением Prius
#   - target_pose_bridge: мост /target/pose из Gazebo в ROS2
#
# Параметры (меняются в rl_pipeline_launch.py):
#   trajectory: static|linear|circle|random|evasive (траектория цели)
#   speed: 3.0 (скорость цели м/с)
#   turn_rate: 0.3 (угловая скорость для circle)
#   world_name: car_track

docker exec -d sim_dev_container bash -c "\
source /opt/ros/jazzy/setup.bash && \
source /simulation_ws/install/setup.bash && \
ros2 launch los_guidance rl_pipeline_launch.py > /tmp/rl_pipeline.log 2>&1"

# ────────────────────────────────────────────────────────────────────────────
# 6. RL ТРЕНИРОВКА (Terminal 4)
# ────────────────────────────────────────────────────────────────────────────
# PPO тренировка с Behavior Cloning весами
#
# Основные параметры:
#   --bc-weights: путь к .pth файлу с весами BC модели (PyTorch)
#   --bc-model: путь к .onnx файлу BC модели (для C++ деплоя)
#
# Дополнительные параметры (в train_rl.py):
#   --num-envs N: количество параллельных окружений (по умолчанию 1)
#   --quick: быстрый режим (меньше таймстепов, для тестов)
#   --resume PATH: продолжить тренировку с чекпоинта
#
# Окружение:
#   State dim: 9 (u_los_x, u_los_y, u_los_z, lambda_x, lambda_y, lambda_z, roll, pitch, yaw)
#   Action dim: 3 (acc_x, acc_y, acc_z в body FRD frame)
#   Control rate: 50Hz
#   Max episode: 30s

docker exec -d sim_dev_container bash -c "\
source /opt/ros/jazzy/setup.bash && \
source /simulation_ws/install/setup.bash && \
source /simulation_ws/.venv/bin/activate && \
cd /simulation_ws/src/los_guidance/rl && \
python3 train_rl.py \
  --bc-weights /simulation_ws/onnx_model/best_guidance_mlp.pth \
  --bc-model /simulation_ws/onnx_model/guidance_mlp.onnx \
  > /tmp/rl_train.log 2>&1"

# ────────────────────────────────────────────────────────────────────────────
# 7. МОНИТОРИНГ
# ────────────────────────────────────────────────────────────────────────────

# Смотреть лог тренировки (live):
#   docker exec sim_dev_container tail -f /tmp/rl_train.log

# TensorBoard:
#   docker exec -it sim_dev_container bash -c "
#     source /opt/ros/jazzy/setup.bash && \
#     source /simulation_ws/.venv/bin/activate && \
#     tensorboard --logdir /simulation_ws/src/los_guidance/rl/logs/tb --port 6006 --bind_all"
#   Открыть: http://<docker_ip>:6006

# Проверить все процессы:
#   docker exec sim_dev_container ps aux | grep -E 'px4|gz|lighttrack|MicroXRCE|train_rl|ros2'

# Проверить топики:
#   docker exec sim_dev_container bash -c "
#     source /opt/ros/jazzy/setup.bash && source /simulation_ws/install/setup.bash && \
#     ros2 topic hz /fmu/out/vehicle_odometry && \
#     ros2 topic hz /target/pose"

# ────────────────────────────────────────────────────────────────────────────
# 8. ОСТАНОВКА ТОЛЬКО ТРЕНИРОВКИ
# ────────────────────────────────────────────────────────────────────────────
# Останавливает только RL training, оставляя PX4/Gazebo/LightTrack работать

docker exec sim_dev_container bash -c "kill -9 \$(pgrep -f 'train_rl') 2>/dev/null && echo 'Training stopped'"

# ────────────────────────────────────────────────────────────────────────────
# 9. ПЕРЕЗАПУСК ТОЛЬКО ТРЕНИРОВКИ (с другим параметром)
# ────────────────────────────────────────────────────────────────────────────

# Остановить старую тренировку:
docker exec sim_dev_container bash -c "kill -9 \$(pgrep -f 'train_rl') 2>/dev/null"

# Запустить с другими параметрами:
docker exec -d sim_dev_container bash -c "\
source /opt/ros/jazzy/setup.bash && \
source /simulation_ws/install/setup.bash && \
source /simulation_ws/.venv/bin/activate && \
cd /simulation_ws/src/los_guidance/rl && \
python3 train_rl.py \
  --bc-weights /simulation_ws/onnx_model/best_guidance_mlp.pth \
  --bc-model /simulation_ws/onnx_model/guidance_mlp.onnx \
  > /tmp/rl_train.log 2>&1"

# ────────────────────────────────────────────────────────────────────────────
# 10. ЭКСПОРТ ОБУЧЕННОЙ МОДЕЛИ В ONNX
# ────────────────────────────────────────────────────────────────────────────
# Конвертирует SB3 PPO модель в ONNX для деплоя в C++ guidance_controller

docker exec sim_dev_container bash -c "\
source /opt/ros/jazzy/setup.bash && \
source /simulation_ws/install/setup.bash && \
source /simulation_ws/.venv/bin/activate && \
cd /simulation_ws/src/los_guidance/rl && \
python3 export_rl_model.py \
  --model ./logs/best_model/best_model.zip \
  --output /simulation_ws/onnx_model/rl_guidance.onnx"

# ────────────────────────────────────────────────────────────────────────────
# ПОЛЕЗНЫЕ ПУТИ
# ────────────────────────────────────────────────────────────────────────────
# Логи тренировки:     /tmp/rl_train.log
# TensorBoard:         /simulation_ws/src/los_guidance/rl/logs/tb/PPO_*/
# Чекпоинты:           /simulation_ws/src/los_guidance/logs/rl_checkpoint_*.zip
# Best model:          /simulation_ws/src/los_guidance/rl/models/best/best_model.zip
# BC модель (PyTorch): /simulation_ws/onnx_model/best_guidance_mlp.pth
# BC модель (ONNX):    /simulation_ws/onnx_model/guidance_mlp.onnx
# Gazebo world:        /simulation_ws/src/PX4-Autopilot/Tools/simulation/gz/worlds/car_track.sdf
# PX4 launch:          /simulation_ws/src/autonomy/modules/px4_bringup/launch/px4_sim_attack_car.launch.py
# RL env:              /simulation_ws/src/los_guidance/rl/drone_attack_env.py
# RL trainer:          /simulation_ws/src/los_guidance/rl/train_rl.py
