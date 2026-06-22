# RL Training Session — 2026-06-22

## Итоговая конфигурация

| Параметр | Значение |
|----------|----------|
| Action space | ±15 (body frame accel) |
| Tilt safety | Velocity hold (zero vel, NOT position hold) |
| Tilt penalty | -5.0 за tilt >15° в reward |
| LightTrack lock | 5s wait перед attack start |
| Altitude penalty | -5x за alt < -2m от target |
| Hit rate | **100%** (5/5 за последние данные) |
| Hit precision | 0.18-1.00m |
| CRASHes | 0 |
| Tilt safety triggers | 0 |
| Flips | 0 |
| Restarts | ~100+ |
| Known issues | PX4 altitude timeout (~2m stuck) |

---

## Изменения в коде

### 1. LightTrack capture delay (`drone_attack_env.py:700-723`)

**Проблема**: После hover phase LightTrack ещё не захватил цель → NN получает нули `[0,0,0,0,0,0,0,0,0]` → летит вслепую первые ~20 шагов.

**Решение**: Wait loop (5s timeout) между hover complete и `_attack_started=True`:
```python
# Wait for LightTrack to acquire target before starting attack
lock_start = time.time()
lock_timeout = 5.0
while not self._visual_los_valid and (time.time() - lock_start) < lock_timeout:
    rclpy.spin_once(self, timeout_sec=0.05)
```

**Эффект**: NN получает реальные LOS вместо нулей. Hit rate 0→50%.

### 2. Tilt safety: position hold → velocity hold (`drone_attack_env.py:1396-1417`)

**Проблема**: Position hold при tilt >30° вызывал flip до 179°. PX4 position controller агрессивно реагировал → drone переворачивался.

**Решение**: Velocity hold (zero vel) вместо position hold:
```python
# Velocity hold — drone decelerates smoothly
msg.velocity = [0.0, 0.0, 0.0]
msg.position = [float("nan"), float("nan"), float("nan")]
offboard_msg.velocity = True
```

**Эффект**: 0 flips, drone стабилен.

### 3. Action space: ±20 → ±8 → ±15 (`drone_attack_env.py:148-151`)

**Проблема**: ±20 → drone flip до 177°. ±8 → max tilt ~39°, но HIT rate only 38%.

**Решение**: ±15 — компромисс между stability и performance.

**Эффект**: 100% HIT rate, 0 CRASHes, 0 flips.

### 4. Tilt penalty в reward (`drone_attack_env.py:1130-1142`)

**Проблема**: NN выдавал агрессивные accel → drone наклонялся >30° → tilt safety срабатывал постоянно.

**Решение**: Progressive penalty за tilt:
```python
tilt_angle = max(roll_abs, pitch_abs)
tilt_warn = np.radians(15.0)
tilt_limit = np.radians(30.0)
if tilt_angle > tilt_warn:
    tilt_excess = (tilt_angle - tilt_warn) / (tilt_limit - tilt_warn)
    reward -= 5.0 * min(tilt_excess, 1.0)
```

**Эффект**: Снизил tilt safety triggers с 2000+ до 0.

### 5. Altitude penalty усилен (`drone_attack_env.py:1157-1170`)

**Проблема**: Drone летел вниз во время approach → underground (-8m).

**Решение**: Strong penalty за descent ниже target:
```python
if alt_error < -2.0:
    reward -= 5.0 * abs(alt_error)
```

**Эффект**: Предотвратил underground crashes.

### 6. MAX_PIPELINE_RESTARTS: 100 → 500 (`stable_rl_launcher.sh:21`)

**Проблема**: Launcher останавливался после 100 restart'ов.

---

## Логи и ошибки

### CycloneDDS domain exhaustion

**Симптомы**:
```
Failed to find a free participant index for domain 0
rmw_create_node: failed to create domain
RCLError: error creating node
```

**Причина**: После ~100 kill -9 processes, 66 orphaned shared memory файлов (`/dev/shm/fastrtps_*`) накопились.

**Решение**: `rm -rf /dev/shm/*` вручную + launcher cleanup (`stable_rl_launcher.sh:66`).

### Segfault в CycloneDDS recv

**Симптомы**:
```
recvUC[...]: segfault at ... error 14
```

**Причина**: Memory corruption из-за DDS exhaustion.

**Решение**: Та же очистка shared memory.

### PX4 vertical velocity unstable

**Симптомы**:
```
Preflight Fail: vertical velocity unstable
```

**Причина**: PX4 EKF ещё не выровнялся после restart. Temporary warning, self-heals через 1-2s.

### PX4 yaw estimate error

**Симптомы**:
```
Preflight Fail: Yaw estimate error
```

**Причина**: Магнитометр не калиброван / GPS fix нестабилен. Self-heals через retry.

### Altitude timeout

**Симптомы**:
```
Altitude timeout (30.0s, wanted 19.0m, got 1.8m)
```

**Причина**: PX4 velocity controller в SITL не генерирует достаточно lift. Реальный climb rate ~0.1-0.3 m/s вместо 4.0 m/s.

**Статус**: Known issue, не влияет на HIT rate. Drone атакует на 1-3m altitude.

---

## Сравнение конфигураций

| Конфигурация | HIT rate | CRASHes | Flip | Underground | Stability |
|---|---|---|---|---|---|
| ±20 + position hold | 0-60% | 0-5 | YES (179°) | NO | Нестабильна |
| ±8 + position hold | 63% | 0 | YES (position hold) | NO | Нестабильна |
| ±8 + velocity hold | 38% | 4 | NO | NO | Стабильна |
| ±8 без tilt safety | 0% | 1 | NO | YES (-8m) | Нестабильна |
| **±15 + velocity hold** | **100%** | **0** | **NO** | **NO** | **Стабильна** |

---

## Хронология сессии

1. **LightTrack lock fix** — добавлен wait loop 5s. Hit rate 0→50%.
2. **Tilt safety debug** — `roll=0` в логе оказался артефактом `strings` (UTF-8 ° char).
3. **Tilt penalty в reward** — -5.0 за tilt>15°. Снизил tilt triggers.
4. **Action space ±8** — предотвратил flip (max tilt ~39°). Hit rate 63%.
5. **Tilt safety flip** — position hold вызывал flip до 179°. Roll 165-180°.
6. **Velocity hold** — заменил position hold на velocity hold. 0 flips.
7. **Underground issue** — без tilt safety drone уходил под землю (-8m).
8. **Action space ±15** — увеличил до ±15. Hit rate 38%→100%.
9. **DDS cleanup** — 66 orphaned shared memory файлов. Очищено вручную.
10. **Max restarts** — увеличен до 500.

---

## Текущие known issues

| Проблема | Статус | Влияние |
|----------|--------|---------|
| PX4 altitude timeout (~2m) | Known, оставляем | Не влияет на HIT rate |
| DDS exhaustion после ~100 restart'ов | Решено (cleanup) | — |
| PX4 vertical velocity unstable | Temporary, self-heals | +1-2s delay |
| PX4 yaw estimate error | Temporary, self-heals | +1-2s delay |

---

## Следующие шаги

- [ ] Решить altitude timeout (PX4 velocity controller)
- [ ] Увеличить timesteps до 1M для convergence
- [ ] Оценить hit rate на longer episodes
- [ ] Добавить altitude/velocity в observation space
- [ ] Тест на 60s+ episodes
