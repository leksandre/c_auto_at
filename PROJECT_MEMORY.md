# Project memory
_Durable project-level knowledge. Persists across all sessions in this project. Edit only content under italic instructions._

## Project context
_What is this project? What's its goal? High-level identity._

ROS2-based LOS (Line-of-Sight) guidance system for a PX4 drone (x500_0 model) in Gazebo simulation. Integrates visual tracking (camera-based target tracking with Kalman filter), guidance algorithms (PN, PI, Intercept), and optional neural network guidance via ONNX. Includes an RL training pipeline using PPO (Stable-Baselines3) wrapped in a Gymnasium environment (`DroneAttackEnv`) that communicates with PX4 via ROS2 topics.

## Rules
_Hard constraints from user that every session must respect._

- **Критическое сопровождение**: Задавать вопросы по всем непонятным местам. Всегда выявлять места с несколькими вариантами решения и спрашивать у пользователя какой выбрать. Подвергать критике требования — указывать на узкие места, ошибки в логике, возможные подводные камни. Указывать на архитектурные риски и предлагать альтернативы. Не молча выполнять, если видишь проблему. При любом нетривиальном решении — перечислить варианты с trade-offs, не принимать решение за пользователя.
- **Мыслительные стратегии (применять автоматически, выбирать наиболее эффективную)**:
  - 🟠 [ATAKA] — атакуй идею как оппонент: укажи, где сломается, что я не учёл, как это провалится на практике
  - 🟠 [KOMPROMISS] — сначала усилить позицию пользователя (лучшие аргументы «за»), затем разбить её и собрать компромисс
  - 🟠 [PROROK] — представь, что проект провалился через 90 дней. Что пошло не так? Какие сигналы раннего предупреждения? Что делаем сейчас?
  - 🟠 [POTESTIM] — придумай способ опровергнуть гипотезу быстрее и дешевле (какой один тест даст максимально информативный результат?)
  - 🟠 [TUPITSA] — разметь возможные когнитивные искажения в логике и как их обойти
  - 🟠 [RESHENIE] — перечисли издержки и обмены: что выигрываем/теряем при каждом варианте, пороги при которых решение меняется
  - 🟠 [CALC] — потребуй и покажи расчёты по шагам (включая единицы измерения и диапазоны)
  - 🟠 [ISTOCHNIKI] — запроси источники «уровня доказательств» и отсечь слабые
- **Не соглашаться со всем** — если пользователь прав, подтвердить. Если нет — аргументированно возразить. Не быть «yes-man».
- **RL training simulation launch sequence**:
  1. `ros2 launch px4_bringup px4_sim_attack_car.launch.py` (headless via `PX4_GZ_HEADLESS=1`)
  2. `ros2 launch lighttrack_ros lighttrack_launch.py`
  3. `MicroXRCEAgent udp4 -p 8888`
  4. `python3 train_rl.py --quick` (or with BC weights)
- `ros2 launch los_guidance rl_pipeline_launch.py` provides Gazebo reset service (env no longer calls it — reset removed to avoid EKF breakage)
- **NO GPS IN RL OBSERVATION**: `_get_obs()` uses ONLY visual LOS from LightTrack bbox. Zero GPS data in neural network input. GPS ground truth used only for reward computation (distance, hit check).

## Architecture decisions
_Major design choices with rationale. The "why" matters more than the "what" for future sessions._

- **Three-layer RL training architecture**:
  1. **Turning algorithm** (separate from NN) — turns drone toward car for training convenience
  2. **LightTrack** — automatically detects car, provides bbox. Creates conditions for simulation.
  3. **NN** — receives bbox/LOS data, calculates accelerations, gets HIT. THIS is the actual training.
  Attack start conditions: drone facing car + valid LightTrack tracking + altitude ≥ 2m.
- **4-algorithm guidance system**: PN (baseline), PI-PN (feedforward), Intercept+ADRC (top attack), NN (RL policy). Runtime-selectable via params. NN completely replaces classical when enabled — no blending.
- **RL training pipeline**: PPO (SB3) in Gymnasium env wrapping PX4 SITL. State=9 floats matching BC model interface. Supports BC weight initialization for warm-starting. Launch sequence documented in `rl_commands.sh`.
- **Kalman filter for LOS estimation**: 6-state linear KF with optional gyro compensation and adaptive noise. Provides smoothed LOS vector and angular rate to guidance algorithms.

## Discovered durable knowledge
_Cross-task facts that survive across sessions. Promoted from session checkpoints' §7 when proven durable._

- **PX4 OFFBOARD mode entry requirement**: PX4 requires OffboardControlMode heartbeat at ≥2 Hz for ~1 second BEFORE accepting OFFBOARD mode switch. Without this, all subsequent TrajectorySetpoint commands are silently rejected. ARM must come AFTER OFFBOARD is confirmed.
- **Simulation stack**: Gazebo + ROS-Gazebo bridge. RL pipeline: Gazebo reset service + moving target node + target pose bridge. Training uses PPO (Stable-Baselines3) with DummyVecEnv (single-process sequential for ROS2).
- **PX4 ARM is fire-and-forget — must verify arming_state**: PX4 silently rejects ARM when preflight checks fail. `vehicle_status.arming_state == 2` confirms armed. Always verify after sending ARM command.
- **Parameter wiring pattern in los_guidance**: New parameters follow 3 steps: (1) `declare_parameter()` + store in `config_` in `loadParameters()`, (2) pass to component config struct in `initializeControllers()`, (3) optionally add to `params.yaml` for runtime override. All three steps required for a parameter to work.
- **Docker build pattern**: `docker exec sim_dev_container bash -c "cd /simulation_ws && source install/setup.bash && rm -rf build/<pkg>/ && colcon build --packages-select <pkg>"`. No `-it` flags. Path is `/simulation_ws` not `/home/kopty/sim/docker_ws`.
- **Odom glitch filter must accept post-disarm large jumps**: `_odom_received=True` + strict >10m threshold causes permanent ODOM DEAD. After any disarm, `_odom_received` is reset to False, and when it becomes True again, the filter accepts jumps up to 100m. Only >100m is rejected. This is critical for crash recovery without Gazebo reset.
- **set_pose causes PX4 EKF divergence under ALL conditions**: Even with odom state reset, `gz service set_pose` makes PX4's internal estimator diverge (tested: 143m altitude instead of 1m). EKF state is internal to PX4 and doesn't sync with Gazebo model position. **set_pose is NEVER viable for RL episode recovery.**
- **Disarm→5s wait→rearm is the only safe crash recovery**: PX4 EKF settles on ground because IMU+GPS are consistent. No teleport needed. If drone is flipped → can't climb → stuck detection → episode end → consecutive failures → full pipeline restart.
- **ODOM DEAD now triggers RuntimeError for clean exit**: `_check_done` raises `RuntimeError("ODOM_DEAD:...")` instead of `return True, False`. `train_once()` catches it → saves checkpoint → `sys.exit(0)` → launcher restarts pipeline.
- **PX4 OFFBOARD heartbeat dead zone causes silent mode loss**: 1.0s sleep without heartbeats between OFFBOARD mode command and heartbeat continuation triggers PX4's COM_OF_LOSS_T (500ms default). PX4 reverts to POSCTL/AUTO_LOITER. ARM succeeds but in wrong mode. Position setpoints silently ignored. **Always publish heartbeats continuously during mode switch.**
- **nav_state must be verified after OFFBOARD mode switch**: `arming_state == 2` (ARMED) does NOT confirm OFFBOARD mode. Must check `nav_state == 14` (NAVIGATION_STATE_OFFBOARD). Without this, code silently operates in wrong flight mode.
- **PX4 VehicleStatus doesn't have `flight_mode` field**: Only `nav_state`, `arming_state`, `armation_reason`. Attempting `.flight_mode` → AttributeError.
- **forward_bias=3.0 m/s² masks NN learning in RL training**: NN outputs accel in [-1,+1] range, but +3.0 bias dominates. NN contributes nothing. Loss decreasing (203→3.4) but drone flies AWAY from target. Must remove/reduce bias to test actual NN learning.
- **Action space must accommodate gravity compensation**: PX4 OFFBOARD accel mode doesn't compensate gravity. NN needs +9.81 in body Z to hover. With [-5,+5], NN physically cannot hover (max +5 < +9.81). BC model y_max=[15.4, 0.68, 0.29] also requires wider range. Minimum viable: [-10,+10], used [-20,+20].
- **Sideways drift penalty must be LOS-relative**: Penalizing ALL horizontal velocity (`reward -= 0.01 * horiz_vel`) blocks movement toward target. Must project velocity onto LOS direction and only penalize perpendicular component.
- **Stable RL launcher needs EKF degradation detection**: PX4 EKF degrades after 2-5 crash cycles (arm failures increase). Detecting ≥3 consecutive arm failures → full pipeline restart. Altitude > 50m is another EKF divergence signal.
- **BC model uses LOS data, not GPS**: `synthetic_guidance_data.csv` columns: u_los (unit LOS vector), lambda (angular velocity), rpy → acc. These are LOS-derived quantities, not raw GPS. Training script at `/home/kopty/Загрузки/onnx_model/train_mlp.py`.
- **Bbox proximity gate prevents tracking flip**: When car is at frame edge, LightTrack bbox is large AND off-center → old filter rejected it → LightTrack locked elsewhere → tracking flipped to wrong side of frame. Fix: `bbox_proximity_threshold=0.15` stores `last_valid_bbox_cx_/cy_` and accepts large bboxes near last known position. [ses_12643fec3ffea4DW4RHubKZZ1x]
- **LightTrack false positive pattern**: Tracker locks onto entire frame (bbox ≈ full image) with high confidence when no real target visible. Example: `target_sz: [1183, 720]` on `[1280, 720]` = 88%×100% coverage. C++ filter `max_bbox_fraction=0.6` in `target_tracker.cpp` rejects these. [ses_13ab7e317ffetziUR2p5i97hEN]
- **PX4 position controller cannot climb from ground level**: Position setpoints with NaN XY only work when drone is already airborne. From ground level (0-1m), PX4's position controller does not generate enough lift. Must use velocity commands (2 m/s upward) for initial climb, then switch to position control.
- **synthetic_guidance_data.csv has 50% zero acc_x values**: 50,008 of 100,000 samples had acc_x=0. Replaced with random [1,10] to force BC model to learn positive forward acceleration. MSE improved from 0.0849 to 0.0573, MAE from 2.07 to 1.47 m/s².
- **scikit-learn not in container venv by default**: Must `pip install scikit-learn` in container for `train_mlp.py` to work.
- **Altitude safety floors mask NN learning**: Altitude safety floors in `_send_action` that override NN's Z acceleration (forced acc_z, suppressed horizontal) prevent the NN from learning altitude control. Must be removed for training.
- **PX4 OFFBOARD acceleration mode ignores yaw field**: When `OffboardControlMode.acceleration=True`, PX4 does NOT respect `TrajectorySetpoint.yaw`. Drone heading drifts during attack. Fix needed: either combine acceleration + velocity setpoints, or use separate yaw control.
- **Velocity takeoff requires XY=0.0, not NaN**: PX4 velocity controller rejects NaN XY velocity fields. Must set `ts.velocity = [0.0, 0.0, -climb_speed]` for reliable climb from ground level.
- **Session status file**: Full state saved at `SESSION_STATUS.md` with all commands, schemas, files, and next steps.
- **RL env uses visual LOS only, NO GPS in observation**: `_get_obs()` returns only visual LOS from LightTrack bbox. No GPS ground truth, no fallback. `_visual_los_valid=False` → returns zeros. GPS data used ONLY for reward computation (distance, hit check), never seen by NN. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **README is stale**: Describes 16-dim state and 4-dim action space; actual code uses 9-dim state and 3-dim action. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **Deceleration penalty reduces negative acc_x but doesn't eliminate it**: Coefficient 3.0 in `reward -= 3.0 * abs(action[0])` when acc_x < 0. At acc_x=-5, penalty=-15.0/step dominates closing velocity (+10.0). Improved hit rate from 0 to 2/6 (33%). PPO exploration noise still produces small negative acc_x values (up to -0.7). [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **LightTrack visual LOS integration ported from C++ to Python**: `_tracker_cb()` in drone_attack_env.py converts normalized bbox → pixel → roll/pitch correction → pinhole model (fx=fy=410, cx=640, cy=360) → body→nav rotation → EMA smoothed LOS + lambda. `_get_obs()` uses visual LOS when `_visual_los_valid=True`, returns zeros otherwise (no GPS fallback). Camera install angles: pitch=-10°, roll=0°, yaw=0°. Axis swap FRD→RDF. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **C++ TargetTracker pixel formula**: `pixel_u = (left + right) * cx` where cx=640 for normalized [0,1] bbox coords. Image is 1280×720. Python port replicates this exactly. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **Hit detection uses horiz_dist + altitude**: `horiz_dist < 1.0m AND altitude < 2.0m AND odom_fresh` for HIT. User plans progressive tightening for precision. Was 3D Euclidean < 1.5m — caused false positives. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **BC warmstart from GPS checkpoint is counterproductive with visual LOS**: Visual LOS distribution differs from GPS-based LOS. Training from scratch with BC weights (not GPS PPO checkpoint) gives 50% hit rate vs 0% with GPS checkpoint. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **Body→NED rotation in _send_action is interface conversion**: NN outputs body-frame accelerations, PX4 requires NED frame. The rotation converts between coordinate systems for PX4's TrajectorySetpoint interface — not redundant with NN's observation. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **OgreNext segfault caused by Mesa software rendering**: Container has NVIDIA GPU passthrough (`/dev/nvidia0`) but Gazebo defaults to Mesa Intel software rendering. Fix: `__NV_PRIME_RENDER_OFFLOAD=1` + `__GLX_VENDOR_LIBRARY_NAME=nvidia` env vars. Without these, every Gazebo restart segfaults in `Ogre::Hlms::createDatablock` → infinite PX4 crash loop. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **`nohup env VAR=val` does NOT propagate to child processes**: Must use `nohup bash -c 'export VAR=val; exec cmd'` wrapper for env vars to reach PX4/Gazebo process tree. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **sim_sleep_factor floor is 0.1**: Values below 0.1 (tried 0.05, 0.02) cause rclpy RCLError and PX4 instability. Cannot be reduced for speed. Gazebo RTF=10x is the other speed dimension (set in car_track.sdf). [ses_11bb87a1dffeHcPxtWliK10U4Q]
- **Hover on tracking loss**: `_send_action()` checks `np.linalg.norm(obs[:3]) > 0.1` — if zeros (LightTrack lost tracking), drone hovers in place + yaws toward car via GPS ground truth `_compute_yaw_to_target()`. [ses_11bb87a1dffeHcPxtWliK10U4Q]
- Spillover reference files (historical, less-relevant entries): MEMORY-rl-tracker-onnx.md (30 entries) + MEMORY-rl-training-historical.md (63 entries) — combined 93 entries covering Gazebo reset attempts, ONNX deployment, detailed training constraints, tracking fixes, C++ code details
