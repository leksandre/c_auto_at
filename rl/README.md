# RL Pipeline — Reinforcement Learning for Drone Attack

## Overview

This directory contains the Reinforcement Learning (RL) pipeline for training a drone to autonomously attack (intercept) moving targets in the Gazebo simulation environment.

The pipeline uses **PPO** (Proximal Policy Optimization) from Stable-Baselines3, with the option to initialise from a pre-trained Behaviour Cloning (BC) model.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python (RL Agent)                                      │
│  ┌─────────────────┐    ┌──────────────────────┐       │
│  │ PPO / SAC       │◄──►│ DroneAttackEnv      │       │
│  │ (Stable-Baselines3)  │ (gymnasium.Env)     │       │
│  └─────────────────┘    └──────────┬───────────┘       │
│                                    │ ROS2 (DDS)        │
└────────────────────────────────────┼────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────┐
│  Docker Container                  │                    │
│  ┌─────────┐   ┌─────────┐   ┌────▼─────┐             │
│  │ PX4 SITL│◄──│MicroXRCE│◄──│ ROS2     │             │
│  │         │   │ Agent   │   │ topics   │             │
│  └────┬────┘   └─────────┘   └──────────┘             │
│       │                                                │
│  ┌────▼──────────────┐                                 │
│  │ Gazebo Harmonic   │                                 │
│  │ (drone + target)  │                                 │
│  └───────────────────┘                                 │
└─────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `drone_attack_env.py` | Gymnasium environment wrapping ROS2 + Gazebo + PX4 |
| `train_rl.py` | PPO training script with callbacks and evaluation |
| `export_rl_model.py` | Export trained model to ONNX for C++ deployment |
| `requirements.txt` | Python dependencies |

## State Space (16 floats)

| Index | Description | Range |
|-------|-------------|-------|
| 0:3 | Relative target position (body frame) | metres |
| 3:6 | Drone velocity (body frame) | m/s |
| 6:9 | Target velocity estimate | m/s |
| 9 | Distance to target | metres |
| 10:13 | Roll, pitch, yaw | radians |
| 13 | Altitude above target | metres |
| 14 | LOS rate (normalised) | [-1, 1] |
| 15 | Normalised episode time | [0, 1] |

## Action Space (4 floats)

| Index | Description | Range |
|-------|-------------|-------|
| 0:3 | Acceleration command (body frame) | [-5, +5] m/s² |
| 3 | Yaw rate command | [-1, +1] rad/s |

## Reward Structure

- **Closing velocity**: Primary signal — positive reward when distance decreases
- **Proximity bonuses**: Progressive rewards at 20m, 10m, 5m, 2m
- **Time penalty**: -0.05 per step (encourages fast attacks)
- **Action regularisation**: Small penalty for large/jerky actions
- **Terminal rewards**: +200 (hit), -100 (crash), -50 (lost target)

## C++ Nodes (in `src/`)

| Node | Description |
|------|-------------|
| `gazebo_reset_service.cpp` | ROS2 service for Gazebo simulation reset |
| `moving_target_node.cpp` | Target vehicle trajectory controller |

## Usage

### 1. Install Python dependencies

```bash
pip install -r rl/requirements.txt
```

### 2. Start the simulation

```bash
# Terminal 1: Start Gazebo + PX4
docker exec -it sim_dev_container bash
ros2 launch px4_bringup px4_sim_attack_car.launch.py

# Terminal 2: Start RL pipeline services
ros2 launch los_guidance rl_pipeline_launch.py trajectory:=random speed:=3.0
```

### 3. Train the RL agent

```bash
# Inside the container, from /simulation_ws/src/los_guidance
python3 rl/train_rl.py

# Quick test (100k timesteps, 2 envs)
python3 rl/train_rl.py --quick

# With BC model initialisation
python3 rl/train_rl.py --bc-weights /path/to/bc_weights.zip

# Resume from checkpoint
python3 rl/train_rl.py --resume ./logs/checkpoints/rl_drone_25000_steps.zip
```

### 4. Monitor training

```bash
tensorboard --logdir=./logs/tensorboard/ --port=6006
```

### 5. Export to ONNX

```bash
python3 rl/export_rl_model.py --model ./models/rl_drone_final.zip
```

## Training Phases

| Phase | Timesteps | Expected Hit Rate |
|-------|-----------|-------------------|
| BC baseline | 0 | 30-50% |
| RL fine-tuning | 100k | 50-70% |
| RL stabilisation | 500k | 70-85% |
| RL + Domain Randomisation | 1M+ | 85%+ |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_episode_time` | 30s | Maximum episode duration |
| `hit_threshold` | 1.5m | Distance for a successful hit |
| `control_rate_hz` | 50Hz | Control loop frequency |
| `num_envs` | 4 | Parallel environments |
| `learning_rate` | 3e-4 | PPO learning rate |
| `total_timesteps` | 1,000,000 | Total training steps |

## Domain Randomisation

The environment supports domain randomisation for robust policy learning:

- Random drone starting position (angle + radius around target)
- Random target movement patterns (static, linear, circle, random, evasive)
- Random target speed (0-5 m/s)
- Random wind conditions (via PX4 params)

## Troubleshooting

**Q: Gazebo reset service not available**
A: Ensure `gz-transport` is installed and the simulation is running.

**Q: PX4 exits OFFBOARD mode**
A: The environment publishes `OffboardControlMode` at 2Hz minimum. Check the heartbeat.

**Q: Training is slow**
A: Increase `num_envs` (parallel environments) or reduce `control_rate_hz`.

**Q: Agent doesn't learn**
A: Check reward shaping. The closing velocity reward is the most important signal.