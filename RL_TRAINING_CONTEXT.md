# RL Training — Full Context for Future Sessions

_Этот файл содержит ВСЮ информацию для продолжения работы. При запуске новой сессии — покажи этот файл агенту._

---

## Быстрый старт

```bash
# Контейнер
docker start sim_dev_container

# Запуск training
docker exec -d sim_dev_container bash -c "bash /simulation_ws/src/los_guidance/rl/stable_rl_launcher.sh > /tmp/stable_launcher.log 2>&1"

# Мониторинг
docker exec sim_dev_container tail -20 /tmp/stable_launcher.log
docker exec sim_dev_container bash -c 'cat /tmp/rl_train.log | strings | grep -E "HIT|Episode|POS:|CRASH|Tilt" | tail -20'

# Остановка
docker exec sim_dev_container bash -c 'pkill -9 -f "train_rl\|stable_rl"'
```

---

## Все команды запуска

### Полный пайплайн (рекомендуется)
```bash
docker exec -d sim_dev_container bash -c "bash /simulation_ws/src/los_guidance/rl/stable_rl_launcher.sh > /tmp/stable_launcher.log 2>&1"
```

### Пошаговый запуск (если launcher не работает)
```bash
# 1. PX4 SITL + Gazebo
docker exec -d sim_dev_container bash -c "source /opt/ros/jazzy/setup.bash && source /simulation_ws/install/setup.bash && ros2 launch px4_bringup px4_sim_attack_car.launch.py > /tmp/px4.log 2>&1"

# 2. DDS Bridge (ждать 20s после PX4)
docker exec -d sim_dev_container bash -c "MicroXRCEAgent udp4 -p 8888 > /tmp/microxrce.log 2>&1"

# 3. LightTrack
docker exec -d sim_dev_container bash -c "source /opt/ros/jazzy/setup.bash && source /simulation_ws/install/setup.bash && ros2 launch lighttrack_ros lighttrack_launch.py > /tmp/lighttrack.log 2>&1"

# 4. RL Pipeline (reset service + moving target)
docker exec -d sim_dev_container bash -c "source /opt/ros/jazzy/setup.bash && source /simulation_ws/install/setup.bash && ros2 launch los_guidance rl_pipeline_launch.py > /tmp/rl_pipeline.log 2>&1"

# 5. Training
docker exec -d sim_dev_container bash -c "source /opt/ros/jazzy/setup.bash && source /simulation_ws/install/setup.bash && source /simulation_ws/.venv/bin/activate && cd /simulation_ws/src/los_guidance/rl && python3 train_rl.py --bc-weights /simulation_ws/onnx_model/best_guidance_mlp.pth --bc-model /simulation_ws/onnx_model/guidance_mlp.onnx > /tmp/rl_train.log 2>&1"
```

### Остановка всех процессов
```bash
docker exec sim_dev_container bash -c 'pkill -9 -f "train_rl\|px4\|gz sim\|MicroXRCE\|lighttrack\|ros2 launch\|parameter_bridge\|moving_target\|gazebo_reset\|stable_rl"'
```

### Остановка контейнера
```bash
docker stop sim_dev_container
```

### Копирование изменений в контейнер
```bash
docker cp /home/kopty/sim/docker_ws/src/los_guidance/rl/drone_attack_env.py sim_dev_container:/simulation_ws/src/los_guidance/rl/drone_attack_env.py
docker cp /home/kopty/sim/docker_ws/src/los_guidance/rl/train_rl.py sim_dev_container:/simulation_ws/src/los_guidance/rl/train_rl.py
docker cp /home/kopty/sim/docker_ws/src/los_guidance/rl/stable_rl_launcher.sh sim_dev_container:/simulation_ws/src/los_guidance/rl/stable_rl_launcher.sh
docker cp /home/kopty/sim/docker_ws/src/los_guidance/launch/rl_pipeline_launch.py sim_dev_container:/simulation_ws/src/los_guidance/launch/rl_pipeline_launch.py
```

### Пересборка пакетов (после изменений в launch файлах)
```bash
docker exec sim_dev_container bash -c 'cd /simulation_ws && source /opt/ros/jazzy/setup.bash && source install/setup.bash && rm -rf build/los_guidance/ && colcon build --packages-select los_guidance'
docker exec sim_dev_container bash -c 'cd /simulation_ws && source /opt/ros/jazzy/setup.bash && source install/setup.bash && rm -rf build/px4_bringup/ && colcon build --packages-select px4_bringup'
```

### Мониторинг логов
```bash
# Launcher
docker exec sim_dev_container tail -20 /tmp/stable_launcher.log

# Training (HIT, episodes, POS)
docker exec sim_dev_container bash -c 'cat /tmp/rl_train.log | strings | grep -E "HIT|Episode|POS:|CRASH|Tilt" | tail -20'

# PX4
docker exec sim_dev_container bash -c 'cat /tmp/px4.log | strings | tail -20'

# LightTrack
docker exec sim_dev_container bash -c 'tail -10 /tmp/lighttrack.log'

# GPU
docker exec sim_dev_container bash -c 'nvidia-smi | head -12'

# Процессы
docker exec sim_dev_container bash -c 'ps aux | grep -E "train_rl|px4|gz|lighttrack|MicroXRCE" | grep -v grep'

# Checkpoints
docker exec sim_dev_container bash -c 'ls -lt /simulation_ws/src/los_guidance/logs/rl_checkpoint_*.zip 2>/dev/null | head -5'
```

---

## Скрипты и их назначение

### `stable_rl_launcher.sh` — Автоматическое управление пайплайном
**Что делает:**
- Запускает полный пайплайн: PX4 → DDS → LightTrack → RL Pipeline → Training
- Мониторит health: training alive, PX4 alive, DDS alive
- Auto-restart при crash: training died, PX4 died, EKF divergence, arm failures, stuck
- Shared memory cleanup при kill
- PX4 wait timeout: 120s
- Random drone spawn (30-40m от car)

**Ключевые функции:**
- `full_pipeline_start()` — kill_all → start PX4 → wait → start DDS → LightTrack → RL Pipeline
- `start_training()` — запуск train_rl.py с BC weights
- `find_latest_checkpoint()` — поиск последнего checkpoint для resume
- Health checks: `check_training_alive()`, `check_px4_alive()`, `check_dds_alive()`

### `train_rl.py` — PPO Training
**Что делает:**
- Создаёт Gymnasium environment (DroneAttackEnv)
- Настраивает PPO с гиперпараметрами
- Загружает BC weights для warmstart
- CheckpointCallback каждые 50k steps
- Auto-restart при ODOM_DEAD / PipelineRestartNeeded

**Ключевые функции:**
- `train_once()` — один запуск training (может crash)
- `train()` — outer loop с auto-resume из checkpoints
- `load_bc_weights_into_policy()` — загрузка BC weights в PPO policy

### `drone_attack_env.py` — Gymnasium Environment
**Что делает:**
- Wraps PX4 SITL через ROS2
- Observation: 9-dim (visual LOS from LightTrack)
- Action: 3-dim (body frame accelerations)
- Reward shaping для attack behavior
- Soft reset (disarm→15s→rearm) при crash
- Recovery mode при потере трека

**Ключевые методы:**
- `reset()` — сброс эпизода, взлёт, позиционирование
- `step()` — один шаг RL: send action → wait → reward → check done
- `_send_action()` — body→nav rotation, gravity comp, tilt safety
- `_compute_reward()` — reward shaping
- `_check_done()` — HIT/CRASH/timeout detection
- `_fly_to_altitude()` — velocity-based takeoff
- `_hover_and_face_target()` — hover phase перед атакой
- `_fly_to_starting_position()` — random spawn 30-40m от car
- `_soft_reset()` — disarm→15s→rearm при crash

---

## Точки корректировки

### 1. Reward Function (`_compute_reward`)
**Где:** `drone_attack_env.py` lines 1047-1110
**Что регулирует:** Как NN учится атаковать
**Ключевые параметры:**
- `reward -= 10.0 * d_distance` — closing velocity signal
- `reward -= 3.0 * abs(action[0])` — deceleration penalty
- `reward -= 5.0 * abs(vel_forward)` — negative velocity penalty
- Proximity bonuses: +0.5/+1/+2/+5/+10/+20
- HIT reward: +200, CRASH penalty: -100

### 2. Action Space (`_send_action`)
**Где:** `drone_attack_env.py` lines 1340-1415
**Что регулирует:** Как NN управляет drone
**Ключевые параметры:**
- Body→nav rotation matrix
- Gravity compensation: `acc_nav[2] -= 9.81`
- Tilt safety: 30° → position hold
- **NO CLIPS** — NN outputs unclamped

### 3. Observation Space (`_get_obs`)
**Где:** `drone_attack_env.py` lines 1010-1045
**Что регулирует:** Что NN видит
**Текущий:** 9-dim [u_los(3), lambda(3), rpy(3)]
**Возможные добавления:** altitude, velocity, distance to target

### 4. Takeoff (`_fly_to_altitude`)
**Где:** `drone_attack_env.py` lines 879-921
**Что регулирует:** Взлёт drone
**Ключевые параметры:**
- `climb_speed = 4.0` m/s
- `timeout = 30.0` s
- Velocity control: `[0.0, 0.0, -climb_speed]`

### 5. Reset Flow (`reset`)
**Где:** `drone_attack_env.py` lines 561-736
**Что регулирует:** Полный цикл reset
**Flow:** crash detect → soft reset → takeoff → hover → attack start

### 6. PPO Config (`get_default_config`)
**Где:** `train_rl.py` lines 370-404
**Что регулирует:** PPO hyperparameters
**Ключевые параметры:**
- `n_steps: 2048`, `batch_size: 64`, `n_epochs: 10`
- `learning_rate: 3e-4`, `gamma: 0.99`
- `total_timesteps: 1,000,000`

---

## Ключевые параметры для регулировки

### Reward Parameters
| Параметр | Текущее значение | Где | Эффект |
|----------|------------------|-----|--------|
| Closing velocity coeff | 10.0 | reward line 1067 | Сила signal за приближение |
| Time penalty | 0.05 | reward line 1070 | Скорость атаки |
| Deceleration penalty | 3.0 | reward line 1094 | Штраф за торможение |
| Negative velocity penalty | 5.0 | reward line 1104 | Штраф за полёт назад |
| Sideways drift penalty | 0.02 | reward line 1108 | Штраф за боковое дрейфование |
| HIT reward | 200 | reward line 1165 | Мотивация для HIT |
| CRASH penalty | 100 | reward line 1168 | Штраф за crash |

### Environment Parameters
| Параметр | Текущее значение | Где | Эффект |
|----------|------------------|-----|--------|
| Hit threshold | 1.0m | init line 96 | Дистанция для HIT |
| Hit altitude min | 1.0m | reward line 1182 | Минимальная altitude для HIT |
| Hit altitude max | 3.0m | reward line 1182 | Максимальная altitude для HIT |
| Crash altitude | 0.5m | check_done line 1198 | Altitude для CRASH |
| Max episode time | 30s | init line 92 | Длительность эпизода |
| Stuck threshold | 500 steps | init line 222 | Время до stuck detection |
| Control rate | 50Hz | init line 91 | Частота управления |

### PX4 Parameters
| Параметр | Текущее значение | Где | Эффект |
|----------|------------------|-----|--------|
| RTF | 5x | car_track.sdf | Скорость симуляции |
| Climb speed | 4.0 m/s | _fly_to_altitude line 889 | Скорость взлёта |
| Takeoff timeout | 30s | _fly_to_altitude callers | Время на взлёт |
| EKF settle time | 15s | _soft_reset line 1569 | Время на восстановление EKF |
| Tilt safety limit | 30° | _send_action line 1366 | Порог для position hold |

### Training Parameters
| Параметр | Текущее значение | Где | Эффект |
|----------|------------------|-----|--------|
| Total timesteps | 1,000,000 | train_rl.py | Общее количество шагов |
| Checkpoint freq | 50,000 | train_rl.py | Частота сохранения |
| Learning rate | 3e-4 | PPO config | Скорость обучения |
| Gamma | 0.99 | PPO config | Discount factor |
| GAE lambda | 0.95 | PPO config | Advantage estimation |
| Entropy coeff | 0.0 | PPO config | Exploration vs exploitation |

---

## Схема запуска пайплайна

```
┌─────────────────────────────────────────────────────────────┐
│                    stable_rl_launcher.sh                     │
├─────────────────────────────────────────────────────────────┤
│  1. kill_all (rm shared memory)                             │
│  2. start_px4 (PX4 SITL + Gazebo, RTF=5x)                 │
│  3. wait_for_px4 (120s timeout)                             │
│  4. start_dds (MicroXRCEAgent udp4 -p 8888)                │
│  5. start_lighttrack (LightTrack tracker)                   │
│  6. start_rl_pipeline (reset service + moving target)       │
│  7. start_training (PPO + BC warmstart)                     │
├─────────────────────────────────────────────────────────────┤
│  MONITORING LOOP (15s interval):                            │
│    - check_training_alive (pgrep train_rl)                  │
│    - check_px4_alive (pgrep px4 + gz sim)                  │
│    - check_dds_alive (pgrep MicroXRCE)                      │
│    - get_ekf_altitude (grep Alt: from log)                  │
│    - get_arm_failures (grep "Failed to arm" from log)       │
│    - get_episode_count (grep "Episode reset complete")       │
├─────────────────────────────────────────────────────────────┤
│  RESTART TRIGGERS:                                          │
│    - Training died → full restart                            │
│    - PX4 died → full restart                                 │
│    - EKF altitude > 50m or < -2m → full restart             │
│    - Arm failures > 2 → full restart                        │
│    - No episodes for 3min → full restart                     │
│    - Max restarts: 100                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Схема RL Environment

```
┌─────────────────────────────────────────────────────────────┐
│                      reset()                                │
├─────────────────────────────────────────────────────────────┤
│  1. Detect crash (alt < 1m or roll/pitch > 45°)            │
│  2. If crash: soft_reset (disarm→15s→rearm)                 │
│  3. Fly to random altitude (5-20m)                          │
│  4. Fly to starting position (30-40m from car)              │
│  5. Hover + face target (5s)                                │
│  6. Ensure min altitude (2m)                                │
│  7. Return initial observation                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      step(action)                           │
├─────────────────────────────────────────────────────────────┤
│  1. Get observation (9-dim: u_los + lambda + rpy)           │
│  2. Send action (_send_action):                             │
│     - Body→nav rotation                                     │
│     - Gravity compensation (-9.81)                          │
│     - Tilt safety (30° → position hold)                     │
│     - Publish to PX4                                        │
│  3. Wait (sim sleep at 50Hz)                                │
│  4. Compute reward                                          │
│  5. Check done (HIT/CRASH/timeout)                          │
│  6. Return (obs, reward, terminated, truncated, info)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Текущая конфигурация

### NN Architecture
- Input: 9 dim [u_los(3), lambda(3), rpy(3)]
- Output: 3 dim [acc_x, acc_y, acc_z] (body frame)
- PPO policy: [256, 128, 64] with ReLU
- BC warmstart: `best_guidance_mlp.pth`

### Observation (NO GPS!)
| Index | Source | Description |
|-------|--------|-------------|
| 0-2 | LightTrack | u_los — unit LOS vector |
| 3-5 | LightTrack | lambda — LOS angular velocity |
| 6-8 | PX4 odom | rpy — roll, pitch, yaw |

### Action
- Body frame FRD: [acc_x, acc_y, acc_z]
- PX4 receives NED (body→nav rotation in `_send_action`)
- **NO CLIPS** — NN outputs unclamped

### Reward
- Closing velocity: `-10.0 * d_distance`
- Time penalty: `-0.05`
- Proximity: +0.5/+1/+2/+5/+10/+20 at 50/30/20/10/5/2m
- Action reg: `-0.005 * sum(action²)`
- Deceleration: `-3.0 * |acc_x|` при acc_x < 0
- Negative velocity: `-5.0 * |vel_forward|` при полёте назад
- Sideways drift: `-0.02 * perpendicular_velocity`
- Altitude shaping: +1/+0.3/-0.5 at <3/<10/>10m error
- **HIT**: +200 (alt > 1m AND alt < 3m)
- **CRASH**: -100 (alt < 0.5m)
- LOST TARGET: -50

### Environment
- Control rate: 50Hz
- Max episode: 30s
- Hit threshold: 1.0m horizontal distance
- Hit altitude: 1.0m < alt < 3.0m
- Crash altitude: < 0.5m
- Stuck threshold: 500 steps (10s)

---

## Текущие проблемы и Known Issues

| Проблема | Статус | Решение |
|----------|--------|---------|
| Velocity takeoff unreliable | ⚠️ | PX4 velocity controller, sometimes 5/5 attempts |
| No clips → drone flip risk | ⚠️ | Tilt safety (30°) + reward penalties |
| Training at 0.4% (4k/1M steps) | ✅ | Early stage, needs time |
| EKF divergence через crash cycles | ✅ | CONSECUTIVE_CRASHES detection |

---

## Сценарий обучения

```
1. Взлёт (velocity 4.0 m/s, 30s timeout)
2. Recovery position (30-100m от car)
3. Атака (NN outputs accelerations at 50Hz)
   → HIT (h_dist < 1.0m, 1.0m < alt < 3.0m)
   → Timeout (30s)
   → CRASH (alt < 0.5m)
4. После HIT/timeout → Recovery → Шаг 2
5. После CRASH → Soft reset (disarm→15s→rearm) → Шаг 1
6. После 2 consecutive crashes → Full pipeline restart
```

---

## Git

- Branch: `forGazebo`
- HEAD: `d82dbc9`
- Remote: `https://github.com/leksandre/c_auto_at.git`

---

## Что делали сегодня (2026-06-21)

1. Fixed `_attack_started` bug → HIT detection works
2. Tilt safety (30°) + position hold
3. Removed all acceleration clips
4. Added negative velocity penalty
5. HIT altitude: 1-3m (no ground-level hits)
6. RTF 10x → 5x
7. Random drone spawn 30-40m от car
8. Hover phase altitude fix
9. OFFBOARD re-switch after ARM
10. Consecutive crash detection
11. Enhanced telemetry logging
12. Launcher trap for graceful shutdown

---

## Следующие шаги

- [ ] Дождаться 50k+ steps для первого checkpoint
- [ ] Оценить hit rate с unclipped actions
- [ ] Если flip — вернуть clip или усилить tilt safety
- [ ] Оптимизировать reward за approach distance
- [ ] Тест на longer episodes (60s+)
- [ ] Добавить altitude/velocity в observation space
- [ ] Увеличить hit rate до >90%
