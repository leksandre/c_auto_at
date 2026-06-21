# Session Status — 2026-06-21 (обновление)

## Исправления (7 штук)

| # | Исправление | Файл | Статус |
|---|------------|------|--------|
| 1 | `_attack_started = False` bug | drone_attack_env.py:707 | ✅ Удалено |
| 2 | Stuck threshold 250→500 (10s) | drone_attack_env.py:222 | ✅ |
| 3 | Climb speed 2.0→4.0 m/s | drone_attack_env.py:889 | ✅ |
| 4 | Timeouts 15s/20s→30s | drone_attack_env.py:655,686 | ✅ |
| 5 | EKF settle 5s→15s, disarm 3s→5s | drone_attack_env.py:1563,1569 | ✅ |
| 6 | Starting distance 10-20m | drone_attack_env.py:791-835 | ✅ (но PX4 velocity медленный) |
| 7 | OFFBOARD re-switch after ARM | drone_attack_env.py:1772-1795 | ✅ Работает! |
| 8 | Consecutive crash counter (2→restart) | drone_attack_env.py:228-229,587-598 | ✅ |
| 9 | EKF altitude < -2m check | stable_rl_launcher.sh:280 | ✅ |
| 10 | Launcher trap (SIGTERM/SIGINT) | stable_rl_launcher.sh:71-77 | ✅ |
| 11 | CONSECUTIVE_CRASHES + EKF_DIVERGENCE в handler | train_rl.py:436 | ✅ |

## Известная проблема

**PX4 velocity commands работают крайне медленно.** Drone перемещается ~0.2 m/s вместо заявленных 4-5 m/s. Причины:
- PX4 velocity controller может иметь внутренние лимиты
- Sim time factor (RTF=10x) может влиять на обработку команд
- OFFBOARD mode может ограничивать максимальную скорость

**Результат**: взлёт до 3m занимает >2 минут, перемещение на 17m — невозможно за 30s.

## Команды

```bash
# Мониторинг
docker exec sim_dev_container tail -10 /tmp/stable_launcher.log
docker exec sim_dev_container bash -c 'cat /tmp/rl_train.log | strings | grep -E "Episode|HIT|CRASH|timeout" | tail -15'

# Остановка
docker exec sim_dev_container bash -c 'pkill -9 -f "train_rl\|stable_rl"'
```
