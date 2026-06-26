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
docker exec sim_dev_container bash -c 'pkill -9 -f "train_rl\|px4\|gz sim\|MicroXRCE\|lighttrack\|ros2 launch\|parameter_bridge\|target_pose\|stable_rl"'
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
docker exec sim_dev_container bash -c 'pkill -9 -f "train_rl\|px4\|gz sim\|MicroXRCE\|lighttrack\|ros2 launch\|parameter_bridge\|target_pose\|stable_rl"'
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

## Текущая конфигурация (a2729c9)

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

### Reward (e117adf)
- Closing velocity: `-10.0 * d_distance`
- Time penalty: `-0.05`
- Proximity: +0.3/+0.5/+0.8/+1.5/+2.5/+5/+10/+20 at 100/80/60/50/30/20/10/5/2m
- Action reg: `-0.005 * sum(action²)`
- Deceleration: `-3.0 * |acc_x|` при acc_x < 0
- Negative velocity: `-5.0 * |vel_forward|` при полёте назад
- Sideways drift: `-0.02 * perpendicular_velocity`
- Altitude shaping: +1.5 at 5-15m, -4.0*(3-alt) при alt<3m, -1.0*(alt-20) при alt>20m
- Vertical velocity penalty: `-3.0 * vel_z` при descending
- **HIT**: +200 (h_dist < 1.0m AND 1.0m < alt < 3.0m)
- **CRASH**: -100 (alt < 0.5m)
- LOST TARGET: -50

### Environment
- Control rate: 50Hz
- Max episode: **30s**
- Hit threshold: 1.0m horizontal distance
- Hit altitude: 1.0m < alt < 3.0m
- Crash altitude: < 0.5m
- Stuck threshold: 500 steps (10s)

### PPO Config
- n_steps: 2048, batch_size: 64, n_epochs: 10
- learning_rate: 3e-4, gamma: 0.99, GAE lambda: 0.95
- Total timesteps: 1,000,000

### Key Functions
| Function | What it does |
|----------|-------------|
| `_fly_to_altitude()` | Velocity-based climb (4 m/s) + position hold, timeout 30s |
| `_fly_to_starting_position()` | Velocity XY + altitude hold (Z=0.0 + position hold_alt) |
| `_hover_and_face_target()` | Hover + yaw toward car, 5s duration |
| `_soft_reset()` | Disarm → **minimum 5s** EKF settle → rearm |
| `_send_action()` | Body→nav rotation, tilt safety (30° → velocity hold) |
| `kill_all()` | Double-pass kill with target_pose_bridge |

---

## Схема запуска пайплайна

```
┌─────────────────────────────────────────────────────────────┐
│                    stable_rl_launcher.sh                     │
├─────────────────────────────────────────────────────────────┤
│  1. kill_all (rm shared memory, double pass)                │
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
│    - Max restarts: 500                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Схема RL Environment

```
┌─────────────────────────────────────────────────────────────┐
│                      reset()                                │
├─────────────────────────────────────────────────────────────┤
│  1. Detect crash (alt < 1m or roll/pitch > 45°)            │
│  2. If crash: soft_reset (disarm→5-15s EKF→rearm)          │
│  3. Fly to random altitude (5-20m)                          │
│  4. Fly to starting position (10-20m from car, alt hold)    │
│  5. Hover + face target (5s)                                │
│  6. LightTrack lock wait (5s timeout)                       │
│  7. Ensure min altitude (2m)                                │
│  8. Return initial observation                              │
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
│     - Tilt safety (30° → velocity hold)                     │
│     - Publish to PX4                                        │
│  3. Wait (sim sleep at 50Hz)                                │
│  4. Compute reward                                          │
│  5. Check done (HIT/CRASH/timeout)                          │
│  6. Return (obs, reward, terminated, truncated, info)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Точки корректировки

### 1. Reward Function (`_compute_reward`)
**Где:** `drone_attack_env.py` ~lines 1099-1200
**Что регулирует:** Как NN учится атаковать
**Ключевые параметры:**
- `reward -= 10.0 * d_distance` — closing velocity signal
- `reward -= 3.0 * abs(action[0])` — deceleration penalty
- `reward -= 5.0 * abs(vel_forward)` — negative velocity penalty
- Proximity bonuses: +0.3/+0.5/+0.8/+1.5/+2.5/+5/+10/+20 at 100/80/60/50/30/20/10/5/2m
- HIT reward: +200, CRASH penalty: -100

### 2. Action Space (`_send_action`)
**Где:** `drone_attack_env.py` ~lines 1340-1415
**Что регулирует:** Как NN управляет drone
**Ключевые параметры:**
- Body→nav rotation matrix
- Gravity compensation: `acc_nav[2] -= 9.81`
- Tilt safety: 30° → velocity hold
- **NO CLIPS** — NN outputs unclamped

### 3. Observation Space (`_get_obs`)
**Где:** `drone_attack_env.py` ~lines 1010-1045
**Что регулирует:** Что NN видит
**Текущий:** 9-dim [u_los(3), lambda(3), rpy(3)]

### 4. Takeoff (`_fly_to_altitude`)
**Где:** `drone_attack_env.py` ~line 896
**Что регулирует:** Взлёт drone
**Ключевые параметры:**
- `climb_speed = 4.0` m/s
- `timeout = 30.0` s
- Velocity control below target, position hold at target

### 5. Reset Flow (`reset`)
**Где:** `drone_attack_env.py` ~lines 561-736
**Что регулирует:** Полный цикл reset
**Flow:** crash detect → soft reset → takeoff → hover → attack start

### 6. PPO Config (`get_default_config`)
**Где:** `train_rl.py` ~lines 370-404
**Что регулирует:** PPO hyperparameters
**Ключевые параметры:**
- `n_steps: 2048`, `batch_size: 64`, `n_epochs: 10`
- `learning_rate: 3e-4`, `gamma: 0.99`
- `total_timesteps: 1,000,000`

---

## Что сделали今天 (2026-06-26)

1. **Откат к e117adf** — база с лучшим hit rate
2. **A1 fix**: `_fly_to_starting_position` altitude hold (velocity Z=0 + position hold_alt)
3. **A3 fix**: `_fly_to_altitude` velocity control + timeout 30s
4. **A4 fix**: `_soft_reset` minimum 5s EKF settle
5. **A5 fix**: `kill_all` double-pass + target_pose_bridge
6. **QoS fix**: RELIABLE QoS для target cmd_vel publisher
7. **max_episode_time**: 90→30s

### Результаты тестирования
- Hit rate: ~50% на первых эпизодах
- QoS warning: ✅ убран
- EKF settle: ✅ 5.0s вместо 0.0s
- Altitude climb: ✅ до 20m (было 4-5m)
- Altitude hold в start pos: ✅ drone держит altitude
- Zombie processes: ✅ double-pass kill

### Оставшиеся проблемы
- `attack starting with zeros` — LightTrack не залочился
- Segfault train_rl.py (1 раз)
- CRASH rate ~30%

---

## Известные проблемы и Known Issues

| Проблема | Статус | Решение |
|----------|--------|---------|
| `attack starting with zeros` | ⚠️ | Добавить wait loop для LightTrack lock |
| Segfault train_rl.py | ⚠️ | Investigate GPU memory |
| ODOM_DEAD | ⚠️ | Auto-restart через launcher |
| CRASH rate ~30% | ⚠️ | Reward shaping, longer episodes |

---

## Git

- Branch: `forGazebo`
- HEAD: `a2729c9` (2026-06-26)
- Remote: `https://github.com/leksandre/c_auto_at.git`

---

## Следующие шаги

- [ ] Исправить `attack starting with zeros` — wait loop для LightTrack lock
- [ ] Проверить hit rate на 10+ эпизодах стабильно
- [ ] Investigate segfault в train_rl.py
- [ ] Рассмотреть увеличение max_episode_time до 60s
- [ ] Добавить altitude/velocity в observation space
- [ ] Увеличить hit rate до >90%
