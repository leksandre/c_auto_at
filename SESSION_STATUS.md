# Session Status — 2026-06-20

## Текущая цель
Обучить NN (PPO + BC warmstart) атаковать машину через визуальный bbox от LightTrack. NN получает LOS vector из LightTrack bbox и выдаёт ускорения в body frame.

---

## История коммитов

```
372e0f9 feat(rl): visual LOS from LightTrack + recovery mode + deceleration penalty
d21a219 feat(rl): soft-reset on crash — disarm→gazebo_reset→rearm
d2540b4 fix(tracker): bbox filter + divergence guard for edge-of-frame and runaway tracking
2343429 step2_createRlSim
8baa1b7 step1-create_rl_sim
6a05210 fix_true_forward_acc
26c2763 adding_Data_generator_for_Reinfocement_ML
f0a0fe0 addingLogs_into_simulations
eefd892 versionForGazebo
```

### Описание коммитов

| Коммит | Описание |
|--------|----------|
| `372e0f9` | **ТЕКУЩИЙ.** Visual LOS из LightTrack bbox, recovery mode при потере трека, deceleration penalty для acc_x, hit detection ужесточён, NVIDIA GPU env vars |
| `d21a219` | Soft reset при crash: disarm → wait 5s → rearm. EKF settle. OFFBOARD heartbeat. |
| `d2540b4` | Bbox proximity gate, max_bbox_fraction=0.6, divergence guard |
| `2343429` | Полная реализация RL simulation (Gymnasium env + PX4 SITL) |
| `8baa1b7` | Базовый scaffold RL симуляции |

---

## Что сделано за сессию (2026-06-20)

### Ключевые изменения

1. **GPS УБРАН из observation space** — `_get_obs()` использует ТОЛЬКО visual LOS из LightTrack bbox. GPS (`/target/pose`) только для reward (distance, hit check) и hover yaw.

2. **LightTrack bbox → LOS конвертация** — полная реализация в Python: bbox → pixel → pinhole model (fx=fy=410, cx=640, cy=360) → camera → body → nav. Matches C++ TargetTracker.

3. **Deceleration penalty** — `reward -= 3.0 * |acc_x|` при `acc_x < 0`. Штрафует торможение.

4. **Hit detection ужесточён** — `horiz_dist < 1.0m AND altitude < 2.0m AND odom_fresh AND _attack_started`. Hit только во время фазы атаки (после hover phase).

5. **Recovery mode при потере трека** — random altitude (2-15m) + random distance (10-120m от машины) + yaw к машине.

6. **NVIDIA GPU env vars** — `__NV_PRIME_RENDER_OFFLOAD=1` + `__GLX_VENDOR_LIBRARY_NAME=nvidia` в launch файле PX4.

7. **Shared memory cleanup** — `rm -f /dev/shm/*` в launcher при kill_all.

8. **PX4 wait timeout** — увеличен до 120s (было 60s).

### Что работает
- ✅ Visual LOS из LightTrack стабильно поступает в NN
- ✅ Recovery mode при потере трека (alt=9.3m, dist=91.4m)
- ✅ Deceleration penalty снижает отрицательный acc_x
- ✅ Hit detection: 2/3 (67%) с visual LOS (до ODOM_DEAD)
- ✅ Hover при потере трека → recovery
- ✅ PX4 startup на GPU (14s вместо 60s+)

### Что НЕ работает
- ❌ **Velocity takeoff нестабилен** — drone застряёт на 0.3m, не взлетает выше
- ❌ **OgreNext segfault** — intermittent, PX4 иногда падает при restart
- ❌ **Shared memory corruption** — после kill_all, PX4 не может создать shared memory
- ❌ **RCLError после restarts** — ROS2 node creation fails после многих restarts
- ❌ **EKF degradation** — через 3-5 crash cycles

---

## Команды запуска

### Полный запуск (через launcher)
```bash
docker exec -d sim_dev_container bash -c "bash /simulation_ws/src/los_guidance/rl/stable_rl_launcher.sh > /tmp/stable_launcher.log 2>&1"
```

### Проверка статуса
```bash
# Launcher лог
docker exec sim_dev_container tail -20 /tmp/stable_launcher.log

# Training лог (acc_nn, hits, episodes)
docker exec sim_dev_container bash -c 'cat /tmp/rl_train.log | strings | grep -E "HIT|Episode|acc_nn" | tail -20'

# PX4 лог (errors, crashes)
docker exec sim_dev_container bash -c 'cat /tmp/px4.log | strings | tail -20'

# LightTrack трекинг
docker exec sim_dev_container bash -c 'tail -10 /tmp/lighttrack.log'

# GPU использование
docker exec sim_dev_container bash -c 'nvidia-smi | head -12'

# Процессы
docker exec sim_dev_container bash -c 'ps aux | grep -E "train_rl|px4|gz|lighttrack" | grep -v grep'
```

### Остановка
```bash
docker exec sim_dev_container bash -c 'pkill -f "train_rl\|px4\|gz sim\|MicroXRCE\|lighttrack\|rl_pipeline"'
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
```

---

## Структура контейнера

```
simulation_ws/
├── install/
│   ├── px4_bringup/share/px4_bringup/launch/
│   │   ├── px4_sim.launch.py              ← PX4 SITL launch (NVIDIA GPU env vars)
│   │   ├── px4_sim_attack_car.launch.py   ← RL training world (car_track)
│   │   └── px4_sim_attack.launch.py
│   └── lighttrack_ros/                    ← LightTrack ROS2 package
├── src/
│   ├── los_guidance/
│   │   ├── rl/
│   │   │   ├── drone_attack_env.py        ← ГЛАВНЫЙ: Gymnasium env + LightTrack LOS
│   │   │   ├── train_rl.py               ← PPO training script
│   │   │   ├── stable_rl_launcher.sh      ← Auto-recovery launcher
│   │   │   ├── train_mlp.py              ← BC model training
│   │   │   ├── export_rl_model.py        ← ONNX export
│   │   │   └── rl_commands.sh            ← Manual command reference
│   │   ├── src/
│   │   │   ├── target_tracker.cpp         ← C++ bbox → LOS conversion
│   │   │   ├── los_estimator.cpp          ← Kalman Filter для LOS
│   │   │   ├── guidance_controller.cpp    ← PN/PI/Intercept + ONNX NN
│   │   │   ├── vehicle_controller.cpp     ← PX4 OFFBOARD interface
│   │   │   └── node.cpp                   ← Main ROS2 node
│   │   ├── include/los_guidance/
│   │   ├── launch/
│   │   │   └── rl_pipeline_launch.py      ← Gazebo reset + moving target + bridge
│   │   └── logs/                          ← PPO checkpoints
│   ├── PX4-Autopilot/                     ← PX4 firmware
│   │   └── Tools/simulation/gz/worlds/
│   │       └── car_track.sdf              ← Gazebo world (RTF=10x)
│   └── LightTrack-ncnn_ros2/             ← LightTrack tracker
├── onnx_model/
│   ├── guidance_mlp.onnx                  ← BC ONNX model
│   ├── best_guidance_mlp.pth             ← BC PyTorch weights
│   └── scaler_params.npz                 ← Scaler (X_mean, X_scale, y_max)
└── .venv/                                ← Python venv (stable-baselines3, onnxruntime)
```

---

## Описание скриптов

### `drone_attack_env.py` — ГЛАВНЫЙ ФАЙЛ
Gymnasium RL environment wrapping PX4 SITL via ROS2.

**Observation (9 dim):**
| Index | Источник | Описание |
|-------|----------|----------|
| 0-2 | LightTrack bbox | `u_los` — unit LOS vector в nav frame (bbox → pinhole → camera → body → nav) |
| 3-5 | LightTrack bbox | `lambda` — LOS angular velocity `cross(u_los, du_los/dt)` |
| 6-8 | PX4 odom | `rpy` — roll, pitch, yaw (body orientation) |

**Action (3 dim):**
- `acc_x, acc_y, acc_z` — body frame acceleration (FRD), range [-20, +20] m/s²
- Rotated to NED frame via `_rpy_to_rot()` before sending to PX4

**Key methods:**
- `_tracker_cb(msg)` — LightTrack /tracker_result callback, converts bbox → visual LOS
- `_get_obs()` — builds 9-dim observation from visual LOS (NO GPS)
- `_send_action(action)` — body→NED rotation, gravity compensation, recovery mode
- `_check_done()` — hit/crash/timeout detection with attack_started guard
- `_compute_reward(action)` — closing velocity + proximity + deceleration penalty
- `_init_camera_matrices()` — precompute body→camera rotation matrices
- `_correct_pixel_roll_pitch(u, v)` — undo drone tilt from pixel coordinates
- `_soft_reset()` — disarm → 5s wait → rearm (no Gazebo reset)

### `train_rl.py` — PPO Training
- BC weight loading into PPO policy (9→256→128→64→3)
- DummyVecEnv (ROS2 can't fork)
- CheckpointCallback every 50k steps
- Auto-restart on ODOM_DEAD / PipelineRestartNeeded

### `stable_rl_launcher.sh` — Auto-Recovery
- Full pipeline start order: PX4 → DDS → LightTrack → RL Pipeline → Training
- Health checks: training dead, PX4 dead, DDS dead, EKF divergence, arm failures, stuck
- Shared memory cleanup on kill_all
- PX4 wait timeout: 120s

### `rl_pipeline_launch.py` — RL Infrastructure
Launches 3 nodes:
1. `gazebo_reset_service` — episode reset service
2. `moving_target_node` — target vehicle movement
3. `target_pose_bridge` — Gazebo→ROS2 bridge for /target/pose

---

## Параметры

### NN Architecture
- Input: 9 dim [u_los_x, u_los_y, u_los_z, lambda_x, lambda_y, lambda_z, roll, pitch, yaw]
- Output: 3 dim [acc_x, acc_y, acc_z] (body frame, tanh → denorm)
- PPO policy: [256, 128, 64] with ReLU
- BC warmstart from `best_guidance_mlp.pth`

### PPO Config
- n_steps: 2048, batch_size: 64, n_epochs: 10
- learning_rate: 3e-4, gamma: 0.99, gae_lambda: 0.95
- Total: 1,000,000 timesteps

### Reward
- Closing velocity: `-10.0 * d_distance`
- Time penalty: `-0.05`
- Proximity: +0.5/+1/+2/+5/+10/+20 at 50/30/20/10/5/2m
- Action reg: `-0.005 * sum(action²)`
- Deceleration penalty: `-3.0 * |acc_x|` при `acc_x < 0`
- Sideways drift: `-0.02 * perpendicular_velocity`
- Altitude shaping: +1/+0.3/-0.5 at <3/<10/>10m error
- HIT: +200 (только во время атаки)
- CRASH: -100 (после grace period)
- LOST TARGET: -50

### Environment
- Action space: [-20, +20] m/s² body frame
- Observation: 9 dim (visual LOS ONLY)
- hit_threshold: 1.0m + altitude < 2.0m
- max_distance: 100m
- min_altitude: 0.5m
- crash_grace_steps: 100 (2s)
- stuck_threshold: 250 (5s)
- control_rate: 50Hz
- max_episode: 30s

### Camera (match C++ TargetTracker)
- fx=fy=410, cx=640, cy=360
- install_pitch_deg=-10, roll=0, yaw=0
- pixel_smooth_alpha=0.3
- los_smooth_alpha=0.1
- lambda_smooth_alpha=0.3

---

## Проблемы и Known Issues

### 1. Velocity Takeoff Failure (КРИТИЧЕСКАЯ)
**Симптом:** Drone застряёт на 0.3m, не взлетает выше. `_fly_to_altitude` timeout.
**Причина:** PX4 velocity controller не выполняет climb command после crash cycle.
**Решение:** НЕ найдено. Это основная проблема стабильности.

### 2. OgreNext Segfault (ИНТЕРМИТЕНТНАЯ)
**Симптом:** `Ogre::Hlms::createDatablock` segfault → PX4 exit 255.
**Причина:** Gazebo OgreNext renderer crash при инициализации.
**Workaround:** NVIDIA GPU env vars (`__NV_PRIME_RENDER_OFFLOAD=1`) снижают частоту.

### 3. Shared Memory Corruption
**Симптом:** `boost::interprocess_exception::library_error` при restart.
**Решение:** `rm -f /dev/shm/*` в kill_all (уже добавлено).

### 4. ODOM DEAD
**Симптом:** Odom stale >15s → RuntimeError → launcher restart.
**Причина:** PX4 EKF degradation через 3-5 crash cycles.
**Workaround:** Launcher auto-restart с shared memory cleanup.

### 5. RCLError After Many Restarts
**Симптом:** `rclpy.RCLError: error creating node` — ROS2 DDS state corruption.
**Решение:** `rm -rf /tmp/cyclone_* /tmp/fastrtps_*` перед restart.

---

## Точки сохранения (Checkpoints)

### PPO Checkpoints
- `/simulation_ws/src/los_guidance/logs/GPS_checkpoint_BACKUP.zip` — GPS-trained (25k steps, backup)
- Новые checkpoints создаются каждые 50k steps в `logs/rl_checkpoint_*.zip`

### BC Model
- `/simulation_ws/onnx_model/best_guidance_mlp.pth` — PyTorch weights
- `/simulation_ws/onnx_model/guidance_mlp.onnx` — ONNX model
- `/simulation_ws/onnx_model/scaler_params.npz` — Scaler (y_max=[15.4, 0.68, 0.29])

### Git Commits
- `372e0f9` — текущий HEAD (visual LOS + recovery + deceleration)
- Ветка: `forGazebo`

---

## Следующие шаги

### Приоритет 1: Velocity Takeoff
- [ ] Исправить `_fly_to_altitude` — drone не взлетает после crash cycles
- [ ] Возможные причины: stale odom, PX4 velocity controller timeout, OFFBOARD mode loss

### Приоритет 2: Training Stability
- [ ] Убрать OgreNext segfault (другой render engine? minimal world?)
- [ ] Увеличить retry tolerance в launcher
- [ ] Добавить health check для shared memory state

### Приоритет 3: Training Quality
- [ ] Увеличить hit rate с 67% до >90%
- [ ] Ужесточить hit_threshold до 0.5m для precision
- [ ] Запустить training на 1 час и собрать статистику

---

## Контейнер

**Имя:** `sim_dev_container`
**Образ:** `sim_dev:gz_harmonic_with_agent_RLsim_onnx`
**Статус:** Запущен (нужно остановить)
**GPU:** NVIDIA GeForce RTX 4050 (passthrough)
**RAM:** 15.7GB total, ~5-10GB used during training

### Остановка контейнера
```bash
docker stop sim_dev_container
```
