# Session Status — 2026-06-26

## Текущее состояние

| Метрика | Значение |
|---------|----------|
| HEAD | `a2729c9` (2026-06-26) |
| Branch | `forGazebo` |
| Remote | `https://github.com/leksandre/c_auto_at.git` |
| Code base | `e117adf` + fixes from today |
| Action space | ±15 (body frame accel) |
| Tilt safety | Velocity hold (zero vel) |
| Hit rate | ~50% on first episodes |
| Crash rate | ~30% (consecutive crashes trigger restart) |
| Restarts before stop | ~5 |
| Container | Остановлен |

## Изменения今天 (2026-06-26)

### 1. Откат к e117adf
Пользователь выбрал `e117adf` как базу — лучший hit rate. Откатили:
- `drone_attack_env.py`
- `train_rl.py`
- `stable_rl_launcher.sh`
- `launch/rl_pipeline_launch.py`

### 2. Исправления (a2729c9)

| Фикс | Проблема | Решение | Эффект |
|------|----------|---------|--------|
| **A1: _fly_to_starting_position** | Drone терял altitude при горизонтальном полёте (NaN Z velocity) | `velocity Z=0.0` + `position Z=hold_alt` | Altitude держится |
| **A3: _fly_to_altitude** | PX4 position control не взлетает с земли; timeout 10s слишком короткий | Velocity control (4 m/s climb) + timeout 30s | Drone взлетает до 20m |
| **A4: _soft_reset EKF settle** | EKF settled after 0.0s — rearm без стабилизации | Minimum 5s settle, max 15s | EKF стабилизируется |
| **A5: kill_all** | Zombie процессы накапливаются | `target_pose_bridge` + second pass kill | Чистое убийство |
| **QoS: target_cmd_vel** | Incompatible QoS warning на /cmd_vel | RELIABLE QoS для target publisher | Warning исчез |
| **max_episode_time** | 90s — слишком длинные эпизоды | 30s | Быстрее итерации |

### 3. Оставшиеся проблемы

| Проблема | Статус | Влияние |
|----------|--------|---------|
| `attack starting with zeros` | ⚠️ Не исправлено | NN летит вслепую без LightTrack lock |
| Segfault train_rl.py | ⚠️ 1 раз за сессию | Launcher auto-recovers |
| ODOM_DEAD | ⚠️ 1 раз | Полный рестарт |
| Altitude collapse при полёте к start pos | ✅ Исправлено (A1) | — |
| EKF settle 0.0s | ✅ Исправлено (A4) | — |
| Zombie processes | ✅ Исправлено (A5) | — |
| QoS warning | ✅ Исправлено | — |

## Git

```
a2729c9 fix(rl): altitude hold, EKF settle, velocity takeoff, QoS fix, kill_all
2abf7cd fix(rl): hybrid takeoff + position control for altitude hold
328f497 Replace position control with velocity control in _fly_to_altitude
e117adf feat: add ML pipeline files + all Gazebo world files
a671460 feat(rl): update env + guidance controller + add car_track world
451441e feat(rl): velocity hold tilt safety + ±15 action space + LightTrack lock fix
```

## Команды запуска

### Быстрый старт
```bash
docker start sim_dev_container
docker exec -d sim_dev_container bash -c "bash /simulation_ws/src/los_guidance/rl/stable_rl_launcher.sh > /tmp/stable_launcher.log 2>&1"
```

### Мониторинг
```bash
docker exec sim_dev_container tail -20 /tmp/stable_launcher.log
docker exec sim_dev_container bash -c 'cat /tmp/rl_train.log | strings | grep -E "HIT|Episode|CRASH|timeout|zeros|attack start" | tail -20'
```

### Остановка
```bash
docker exec sim_dev_container bash -c 'pkill -9 -f "train_rl\|px4\|gz sim\|MicroXRCE\|lighttrack\|ros2 launch\|parameter_bridge\|target_pose\|stable_rl"'
docker stop sim_dev_container
```

### Копирование изменений в контейнер
```bash
docker cp /home/kopty/sim/docker_ws/src/los_guidance/rl/drone_attack_env.py sim_dev_container:/simulation_ws/src/los_guidance/rl/drone_attack_env.py
docker cp /home/kopty/sim/docker_ws/src/los_guidance/rl/train_rl.py sim_dev_container:/simulation_ws/src/los_guidance/rl/train_rl.py
docker cp /home/kopty/sim/docker_ws/src/los_guidance/rl/stable_rl_launcher.sh sim_dev_container:/simulation_ws/src/los_guidance/rl/stable_rl_launcher.sh
docker cp /home/kopty/sim/docker_ws/src/los_guidance/launch/rl_pipeline_launch.py sim_dev_container:/simulation_ws/src/los_guidance/launch/rl_pipeline_launch.py
```

## Сценарий обучения (ожидаемый)

```
1. Взлёт (velocity 4.0 m/s, 30s timeout)
2. Recovery position (10-20m от car, altitude hold)
3. Hover + face target (5s)
4. LightTrack lock wait (5s timeout)
5. Атака (NN outputs accelerations at 50Hz)
   → HIT (h_dist < 1.0m, alt 1-3m) → Recovery → Шаг 2
   → Timeout (30s) → Recovery → Шаг 2
   → CRASH (alt < 0.5m) → Soft reset (disarm → 5-15s EKF → rearm) → Шаг 1
6. 2 consecutive crashes → Full pipeline restart
```

## Следующие шаги

- [ ] Исправить `attack starting with zeros` — добавить wait loop для LightTrack lock перед attack
- [ ] Проверить hit rate на 10+ эпизодах
- [ ] Рассмотреть увеличение max_episode_time до 60s если 30s мало
- [ ] Investigate segfault в train_rl.py
- [ ] Добавить altitude/velocity в observation space
