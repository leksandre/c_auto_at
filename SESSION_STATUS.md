# Session Status — 2026-06-21

## Текущий прогресс

| Метрика | Значение |
|---------|----------|
| Hit rate | 0-60% (зависит от сессии) |
| Hit precision | 0.50-1.00m |
| Flip events | 0 (XY clip работает) |
| Crash rate | 3-5 на сессию |
| Timesteps | ~4,096 / 1,000,000 (0.4%) |
| Checkpoints | 0 (freq=50k) |
| Restart count | 18 (auto-recovery) |

## Команды запуска

### 1. Полный запуск (через launcher)
```bash
docker exec -d sim_dev_container bash -c "bash /simulation_ws/src/los_guidance/rl/stable_rl_launcher.sh > /tmp/stable_launcher.log 2>&1"
```

### 2. Проверка статуса
```bash
# Launcher лог
docker exec sim_dev_container tail -20 /tmp/stable_launcher.log

# Training лог (HIT, episodes, POS)
docker exec sim_dev_container bash -c 'cat /tmp/rl_train.log | strings | grep -E "HIT|Episode|POS:|CRASH|Tilt" | tail -20'

# Текущие процессы
docker exec sim_dev_container bash -c 'ps aux | grep -E "train_rl|px4|gz|lighttrack" | grep -v grep'

# GPU
docker exec sim_dev_container bash -c 'nvidia-smi | head -12'
```

### 3. Остановка
```bash
docker exec sim_dev_container bash -c 'pkill -9 -f "train_rl\|stable_rl"'
```

### 4. Остановка контейнера
```bash
docker stop sim_dev_container
```

### 5. Копирование изменений в контейнер
```bash
docker cp /home/kopty/sim/docker_ws/src/los_guidance/rl/drone_attack_env.py sim_dev_container:/simulation_ws/src/los_guidance/rl/drone_attack_env.py
docker cp /home/kopty/sim/docker_ws/src/los_guidance/rl/train_rl.py sim_dev_container:/simulation_ws/src/los_guidance/rl/train_rl.py
docker cp /home/kopty/sim/docker_ws/src/los_guidance/rl/stable_rl_launcher.sh sim_dev_container:/simulation_ws/src/los_guidance/rl/stable_rl_launcher.sh
```

## Сценарий обучения

```
1. Взлёт (velocity commands, 4.0 m/s, 30s timeout)
2. Занятие позиции (recovery: 30-100m от car)
3. Атака (NN outputs accelerations, 50Hz)
   → HIT (h_dist < 1.0m, alt < 2.0m)
   → Timeout (30s)
   → CRASH (alt < 0.5m)
4. После HIT/timeout → Recovery → Шаг 2
5. После CRASH → Soft reset (disarm→5s→rearm) → Шаг 1
6. После 2 consecutive crashes → Full pipeline restart
```

## Исправления (сессия 2026-06-21)

| # | Исправление | Эффект |
|---|------------|--------|
| 1 | `_attack_started` bug fix | HIT detection работает |
| 2 | Tilt safety (30° + position hold) | Предотвращает flip |
| 3 | XY clip (±5 m/s²) | **Полностью предотвращает flip** |
| 4 | Consecutive crash counter (2→restart) | Auto-recovery |
| 5 | RTF 10x → 5x | PX4 velocity controller работает лучше |
| 6 | Hover phase altitude fix (NaN→actual) | Drone не падает во время hover |
| 7 | OFFBOARD re-switch after ARM | PX4 не теряет OFFBOARD mode |
| 8 | EKF settle 5s→15s | Больше времени на восстановление |
| 9 | Random spawn 30-40m от car | NN учится приближаться |
| 10 | Enhanced telemetry logging | Видно altitude, distance, velocity |

## Known Issues

| Проблема | Статус | Причина |
|----------|--------|---------|
| Velocity takeoff unreliable | ⚠️ | PX4 velocity controller |
| 0% hit rate в начале | ✅ | NN尚未 обучился (0.4% steps) |
| EKF divergence через crash cycles | ✅ | Auto-recovery (consecutive crashes) |

## Git

- Branch: `forGazebo`
- HEAD: `99eb84c` (stability fixes)
- Remote: `https://github.com/leksandre/c_auto_at.git`
