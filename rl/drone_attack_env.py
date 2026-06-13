#!/usr/bin/env python3
"""
drone_attack_env.py — Gymnasium RL environment for drone attack simulation.

This environment wraps a ROS2 + Gazebo + PX4 SITL simulation into a standard
Gymnasium interface for Reinforcement Learning with Stable-Baselines3.

State space (9 floats) — matches BC model (guidance_mlp.onnx) interface:
  [0:3]   — unit LOS vector (u_los_x, u_los_y, u_los_z)
  [3:6]   — LOS angular velocity (lambda_x, lambda_y, lambda_z) (rad/s)
  [6:9]   — drone attitude (roll, pitch, yaw) (rad)

Action space (3 floats) — acceleration in body frame:
  [0:3] — acceleration command in body frame (m/s²), range [-5, +5]

Reward structure:
  - Proximity reward: bonus when close to target
  - Closing velocity reward: positive when distance decreases
  - Time penalty: encourages fast attacks
  - Action regularisation: penalises large/jerky actions
  - Terminal rewards: +200 for hit, -100 for crash, -50 for losing target

Author: Generated for los_guidance RL pipeline
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from px4_msgs.msg import (
    VehicleOdometry, VehicleStatus, TrajectorySetpoint,
    OffboardControlMode, VehicleCommand
)
from geometry_msgs.msg import Pose, Twist, Point
from std_srvs.srv import Trigger, SetBool

import threading
import time
import os
import json


# ── Module-level flag to ensure rclpy.init() is called only once ────────
_rclpy_initialised = False


class DroneAttackEnv(gym.Env, Node):
    """
    RL environment for training a drone to attack a moving/static target.

    Uses ROS2 topics to communicate with PX4 SITL and Gazebo simulation.
    Supports domain randomisation, BC model prior, and parallel environments.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_time: float = 30.0,
        hit_threshold: float = 1.5,
        max_distance: float = 200.0,
        min_altitude: float = 0.5,
        control_rate_hz: int = 50,
        randomize_on_reset: bool = True,
        use_bc_model_as_prior: bool = False,
        bc_model_path: str = None,
        target_trajectory: str = "static",
        drone_start_pose: list = None,
        target_start_pose: list = None,
        namespace: str = "",
        render_mode: str = None,
    ):
        # Initialise ROS2 (only once per process)
        global _rclpy_initialised
        if not _rclpy_initialised:
            rclpy.init()
            _rclpy_initialised = True

        # Initialise Gymnasium Env first
        gym.Env.__init__(self)

        # Then initialise ROS2 Node
        node_name = f"rl_drone_attack_env_{namespace}" if namespace else "rl_drone_attack_env"
        Node.__init__(self, node_name)

        # ── Parameters ──────────────────────────────────────────────────────
        self.max_episode_time = max_episode_time
        self.hit_threshold = hit_threshold
        self.max_distance = max_distance
        self.min_altitude = min_altitude
        self.dt = 1.0 / control_rate_hz
        self.randomize_on_reset = randomize_on_reset
        self.target_trajectory = target_trajectory
        self.namespace = namespace
        self.render_mode = render_mode

        # Default poses
        self.drone_start_pose = drone_start_pose or [-1.0, 0.0, 0.0]
        self.target_start_pose = target_start_pose or [50.0, 0.0, 0.0]

        # ── BC model as prior (optional) ────────────────────────────────────
        self.use_bc_prior = use_bc_model_as_prior
        self.bc_session = None
        self.bc_scaler_mean = None
        self.bc_scaler_scale = None
        self.bc_y_max = None
        if use_bc_model_as_prior and bc_model_path:
            try:
                import onnxruntime as ort
                self.bc_session = ort.InferenceSession(bc_model_path)
                # Try to load scaler from same directory
                scaler_dir = os.path.dirname(bc_model_path)
                scaler_path = os.path.join(scaler_dir, "scaler_params.npz")
                if os.path.exists(scaler_path):
                    scaler_data = np.load(scaler_path)
                    self.bc_scaler_mean = scaler_data["X_mean"]
                    self.bc_scaler_scale = scaler_data["X_scale"]
                    self.bc_y_max = scaler_data["y_max"]
                    self.get_logger().info("✅ BC model + scaler loaded as prior")
                else:
                    self.get_logger().info("✅ BC model loaded as prior (no scaler)")
            except Exception as e:
                self.get_logger().warn(f"⚠️ Failed to load BC model: {e}")

        # ── Action & Observation Spaces ─────────────────────────────────────
        # Action: body-frame acceleration [acc_x, acc_y, acc_z] (m/s²)
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        )

        # Observation: [u_los_xyz, lambda_xyz, roll, pitch, yaw] — 9 floats
        # Matches BC model (guidance_mlp.onnx) input interface
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        # ── Internal state cache ────────────────────────────────────────────
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_rpy = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.prev_distance = np.inf
        self.vehicle_status = None
        self._prev_rel_pos = None

        # Episode tracking
        self.episode_start_time = 0.0
        self.step_count = 0
        self.episode_reward = 0.0

        # ── ROS2 setup ──────────────────────────────────────────────────────
        self._setup_ros2()

        # ── Spin thread ─────────────────────────────────────────────────────
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        self.get_logger().info("🎮 DroneAttackEnv initialised")
        self.get_logger().info(
            f"   State dim: {self.observation_space.shape[0]}, "
            f"Action dim: {self.action_space.shape[0]}"
        )
        self.get_logger().info(
            f"   hit_threshold={hit_threshold}m, "
            f"max_episode_time={max_episode_time}s, "
            f"control_rate={control_rate_hz}Hz"
        )

    # ──────────────────────────────────────────────────────────────────────
    # ROS2 Setup
    # ──────────────────────────────────────────────────────────────────────

    def _setup_ros2(self):
        """Create all ROS2 publishers, subscribers, and service clients."""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscribers ─────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            "fmu/out/vehicle_odometry",
            self._odom_cb,
            qos,
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            "fmu/out/vehicle_status",
            self._status_cb,
            qos,
        )

        # Target pose from Gazebo (via ROS2-GZ bridge)
        self.target_sub = self.create_subscription(
            Pose,
            "/target/pose",
            self._target_cb,
            qos,
        )

        # ── Publishers ──────────────────────────────────────────────────
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            "fmu/in/trajectory_setpoint",
            qos,
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            "fmu/in/offboard_control_mode",
            qos,
        )

        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            "fmu/in/vehicle_command",
            qos,
        )

        # ── Service Clients ─────────────────────────────────────────────
        self.reset_sim_client = self.create_client(
            Trigger, "/gazebo/reset_simulation"
        )
        self.reset_world_client = self.create_client(
            Trigger, "/gazebo/reset_world"
        )
        self.pause_client = self.create_client(
            SetBool, "/gazebo/pause_physics"
        )

        # ── Target control publisher ────────────────────────────────────
        self.target_cmd_vel_pub = self.create_publisher(
            Twist, "/cmd_vel", qos
        )

    def _spin_loop(self):
        """Background thread for ROS2 spinning."""
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(self)
        try:
            executor.spin()
        except rclpy.executors.ExternalShutdownException:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # ROS2 Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        """Cache vehicle odometry."""
        self.drone_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        self.drone_vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])
        # Quaternion → RPY
        q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
        self.drone_rpy = self._quat_to_rpy(q)

    def _status_cb(self, msg):
        """Cache vehicle status."""
        self.vehicle_status = msg

    def _target_cb(self, msg):
        """Cache target pose from Gazebo."""
        self.target_pos = np.array([msg.position.x, msg.position.y, msg.position.z])

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset the simulation and return initial observation."""
        super().reset(seed=seed)

        self.get_logger().info("🔄 Resetting simulation...")

        # 1. Reset Gazebo simulation
        self._reset_simulation()

        # 2. Wait for simulation to stabilise
        time.sleep(2.0)

        # 3. Randomise initial conditions
        if self.randomize_on_reset:
            self._randomize_episode()

        # 4. Arm, takeoff, switch to offboard
        self._arm_and_takeoff()

        # 5. Wait for drone to actually reach altitude (with timeout)
        takeoff_altitude = 30.0  # same as in _arm_and_takeoff
        min_start_altitude = 5.0  # minimum altitude to start the episode
        takeoff_timeout = 15.0    # max wait for takeoff
        takeoff_start = time.time()

        self.get_logger().info(f"⏳ Waiting for takeoff (target={takeoff_altitude}m, min={min_start_altitude}m, timeout={takeoff_timeout}s)...")
        while time.time() - takeoff_start < takeoff_timeout:
            # Check current altitude
            current_alt = -self.drone_pos[2]  # NED: z-down, so altitude = -z
            if current_alt >= min_start_altitude:
                self.get_logger().info(f"✅ Takeoff confirmed at {current_alt:.1f}m")
                break
            time.sleep(0.5)
        else:
            self.get_logger().warn(f"⚠️ Takeoff timeout! Altitude={-self.drone_pos[2]:.1f}m, starting anyway")

        # 6. Start target movement
        self._start_target_movement()

        # 7. Reset episode tracking
        self.episode_start_time = time.time()
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_distance = self._get_distance()
        self._prev_rel_pos = None

        # 8. Get initial observation
        obs = self._get_obs()

        self.get_logger().info(
            f"✅ Episode reset complete. "
            f"Altitude: {-self.drone_pos[2]:.1f}m, "
            f"Distance to target: {self._get_distance():.1f}m"
        )

        return obs, {}

    def step(self, action):
        """Execute one RL step: send action, wait, observe, reward."""
        self.step_count += 1

        # 1. Send action to PX4
        self._send_action(action)

        # 2. Wait for control cycle
        time.sleep(self.dt)

        # 3. Get new observation
        obs = self._get_obs()

        # 4. Compute reward
        reward = self._compute_reward(action)

        # 5. Check termination conditions
        terminated, truncated = self._check_done()

        # 6. Update tracking
        self.prev_distance = self._get_distance()
        self.episode_reward += reward

        # 7. Info dict
        info = {
            "distance": self._get_distance(),
            "hit": terminated and self._get_distance() < self.hit_threshold,
            "step": self.step_count,
            "time": time.time() - self.episode_start_time,
            "episode_reward": self.episode_reward,
            "drone_pos": self.drone_pos.tolist(),
            "target_pos": self.target_pos.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render is not supported (headless simulation)."""
        pass

    def close(self):
        """Clean up ROS2 resources."""
        try:
            self.get_logger().info("Closing DroneAttackEnv...")
        except Exception:
            pass
        try:
            self.destroy_node()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Observation & Reward
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        """
        Build 9-dim observation vector matching BC model interface.

        Returns:
            [u_los_x, u_los_y, u_los_z,     # 0:3 — unit LOS vector (nav frame)
             lambda_x, lambda_y, lambda_z,   # 3:6 — LOS angular velocity (rad/s)
             roll, pitch, yaw]               # 6:9 — drone attitude (rad)
        """
        # Relative position in nav frame
        rel_pos_nav = self.target_pos - self.drone_pos
        distance = np.linalg.norm(rel_pos_nav)

        # Unit LOS vector (u_los)
        u_los = rel_pos_nav / max(distance, 1e-6)

        # LOS angular velocity (lambda) — finite difference of rel_pos
        if self._prev_rel_pos is not None:
            lambda_vec = (rel_pos_nav - self._prev_rel_pos) / max(self.dt, 1e-6)
            # Normalise by distance to get angular rate (rad/s)
            lambda_vec = lambda_vec / max(distance, 1e-6)
        else:
            lambda_vec = np.zeros(3)
        self._prev_rel_pos = rel_pos_nav.copy()

        # Clamp lambda to reasonable range
        lambda_vec = np.clip(lambda_vec, -10.0, 10.0)

        obs = np.concatenate(
            [
                u_los,           # 0:3
                lambda_vec,      # 3:6
                self.drone_rpy,  # 6:9
            ]
        ).astype(np.float32)

        return obs

    def _compute_reward(self, action):
        """
        Compute reward for the current step.

        Reward shaping principles:
        - Closing velocity is the primary signal
        - Progressive bonuses for getting close
        - Time penalty encourages fast attacks
        - Action regularisation prevents jerky behaviour
        """
        distance = self._get_distance()
        d_distance = distance - self.prev_distance

        reward = 0.0

        # 1. Closing velocity reward (primary signal)
        #    Negative d_distance = getting closer = positive reward
        reward -= 3.0 * d_distance

        # 2. Time penalty (encourages fast attacks)
        reward -= 0.05

        # 3. Progressive proximity bonuses
        if distance < 20.0:
            reward += 0.5
        if distance < 10.0:
            reward += 1.0
        if distance < 5.0:
            reward += 2.0
        if distance < 2.0:
            reward += 5.0

        # 4. Action regularisation (penalise large/jerky actions)
        reward -= 0.005 * np.sum(action ** 2)

        # 5. Terminal rewards (applied when episode ends)
        if distance < self.hit_threshold:
            reward += 200.0  # HIT!
        elif self.drone_pos[2] < self.min_altitude:
            reward -= 100.0  # CRASH
        elif distance > self.max_distance:
            reward -= 50.0  # LOST TARGET

        return float(reward)

    def _check_done(self):
        """Check if episode should terminate or truncate."""
        distance = self._get_distance()
        elapsed = time.time() - self.episode_start_time

        # Terminal conditions
        if distance < self.hit_threshold:
            self.get_logger().info(f"🎯 HIT! Distance: {distance:.2f}m")
            return True, False

        if self.drone_pos[2] < self.min_altitude:
            self.get_logger().warn(f"💥 CRASH! Altitude: {self.drone_pos[2]:.2f}m")
            return True, False

        if distance > self.max_distance:
            self.get_logger().warn(f"📡 LOST TARGET! Distance: {distance:.1f}m")
            return True, False

        # Truncation (timeout)
        if elapsed > self.max_episode_time:
            return False, True

        return False, False

    # ──────────────────────────────────────────────────────────────────────
    # Action Execution
    # ──────────────────────────────────────────────────────────────────────

    def _send_action(self, action):
        """Send acceleration command to PX4 via OFFBOARD mode (body frame)."""
        msg = TrajectorySetpoint()
        msg.timestamp = int(time.time() * 1e6)

        # Acceleration command (body frame) — 3-dim action
        msg.acceleration = [float(action[0]), float(action[1]), float(action[2])]

        # NaN for unused fields (position/velocity control disabled)
        msg.position = [float("nan")] * 3
        msg.velocity = [float("nan")] * 3
        msg.yaw = float("nan")
        msg.yawspeed = float("nan")

        self.setpoint_pub.publish(msg)

        # Offboard mode heartbeat (required at ≥2 Hz, send every 25 steps)
        if self.step_count % 25 == 0:
            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.acceleration = True
            self.offboard_pub.publish(offboard_msg)

    # ──────────────────────────────────────────────────────────────────────
    # Simulation Control
    # ──────────────────────────────────────────────────────────────────────

    def _reset_simulation(self):
        """Reset Gazebo simulation via service calls."""
        try:
            # Try reset_simulation first
            if self.reset_sim_client.wait_for_service(timeout_sec=2.0):
                req = Trigger.Request()
                future = self.reset_sim_client.call_async(req)
                # Don't wait — fire and forget
            elif self.reset_world_client.wait_for_service(timeout_sec=2.0):
                req = Trigger.Request()
                future = self.reset_world_client.call_async(req)
            else:
                self.get_logger().warn(
                    "⚠️ No Gazebo reset service available. "
                    "Simulation may need manual reset."
                )
        except Exception as e:
            self.get_logger().warn(f"⚠️ Reset service call failed: {e}")

        # Small delay for simulation to process reset
        time.sleep(1.0)

    def _randomize_episode(self):
        """
        Domain randomisation for robust policy learning.

        Randomises:
        - Drone starting position (angle + radius around target)
        - Target starting position
        - Wind conditions (via PX4 params)
        - Target movement pattern
        """
        # 1. Random drone start position around target
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(30.0, 80.0)
        height = np.random.uniform(20.0, 60.0)

        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)

        self.get_logger().info(
            f"🎲 Randomising: drone at ({dx:.0f}, {dy:.0f}, {height:.0f})m "
            f"from target, radius={radius:.0f}m"
        )

        # Note: Actual position setting requires Gazebo service or model spawning
        # This is a placeholder — implement via Gazebo API or model SDF modification

        # 2. Random target movement pattern
        patterns = ["static", "linear", "circle", "random"]
        self.target_trajectory = np.random.choice(patterns)

        # 3. Random target speed
        self.target_speed = np.random.uniform(0.0, 5.0)

        self.get_logger().info(
            f"🎲 Target: pattern={self.target_trajectory}, "
            f"speed={self.target_speed:.1f}m/s"
        )

    def _arm_and_takeoff(self):
        """Arm the drone, take off, and switch to OFFBOARD mode."""
        self.get_logger().info("🚁 Arming and taking off...")

        # ARM
        arm_cmd = VehicleCommand()
        arm_cmd.command = 400  # MAV_CMD_COMPONENT_ARM_DISARM
        arm_cmd.param1 = 1.0  # ARM
        arm_cmd.target_system = 1
        arm_cmd.target_component = 1
        arm_cmd.source_system = 101
        arm_cmd.source_component = 1
        arm_cmd.from_external = True
        arm_cmd.timestamp = int(time.time() * 1e6)
        self.cmd_pub.publish(arm_cmd)
        time.sleep(0.5)

        # TAKEOFF
        takeoff_cmd = VehicleCommand()
        takeoff_cmd.command = 22  # MAV_CMD_NAV_TAKEOFF
        takeoff_cmd.param7 = 30.0  # Takeoff altitude (m)
        takeoff_cmd.target_system = 1
        takeoff_cmd.target_component = 1
        takeoff_cmd.source_system = 101
        takeoff_cmd.source_component = 1
        takeoff_cmd.from_external = True
        takeoff_cmd.timestamp = int(time.time() * 1e6)
        self.cmd_pub.publish(takeoff_cmd)
        time.sleep(1.0)

        # OFFBOARD mode
        offboard_cmd = VehicleCommand()
        offboard_cmd.command = 176  # MAV_CMD_DO_SET_MODE
        offboard_cmd.param1 = 1.0  # Custom mode
        offboard_cmd.param2 = 6.0  # PX4_MODE_OFFBOARD
        offboard_cmd.target_system = 1
        offboard_cmd.target_component = 1
        offboard_cmd.source_system = 101
        offboard_cmd.source_component = 1
        offboard_cmd.from_external = True
        offboard_cmd.timestamp = int(time.time() * 1e6)
        self.cmd_pub.publish(offboard_cmd)

        self.get_logger().info("✅ Armed, takeoff initiated, OFFBOARD mode set")

    def _start_target_movement(self):
        """Start target movement based on current trajectory pattern."""
        if self.target_trajectory == "static":
            # No movement needed
            return

        # Publish initial cmd_vel for the target
        twist = Twist()
        if self.target_trajectory == "linear":
            twist.linear.x = self.target_speed
        elif self.target_trajectory == "circle":
            twist.linear.x = self.target_speed
            twist.angular.z = 0.3  # Turn rate for circle
        elif self.target_trajectory == "random":
            twist.linear.x = self.target_speed * np.random.uniform(-1, 1)
            twist.linear.y = self.target_speed * np.random.uniform(-1, 1)

        self.target_cmd_vel_pub.publish(twist)

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def _get_distance(self):
        """Euclidean distance from drone to target."""
        return float(np.linalg.norm(self.target_pos - self.drone_pos))

    @staticmethod
    def _quat_to_rpy(q):
        """Convert quaternion [w, x, y, z] to roll, pitch, yaw."""
        w, x, y, z = q
        # Roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch
        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        # Yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    @staticmethod
    def _rpy_to_rot(rpy):
        """Convert roll, pitch, yaw to rotation matrix (body → nav)."""
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )
        return R

    def get_bc_action(self, obs):
        """
        Get action from BC model (used as prior/initialisation).

        The BC model (guidance_mlp.onnx) uses:
          Input:  9 floats [u_los_xyz, lambda_xyz, roll, pitch, yaw]
          Output: 3 floats [acc_x, acc_y, acc_z] (body frame, Tanh → [-1,1])
          Denormalisation: action * y_max (where y_max = [15.4, 0.68, 0.29])

        Args:
            obs: Observation vector (9,) matching BC model input

        Returns:
            Action vector (3,) or None if BC model not available
        """
        if self.bc_session is None:
            return None

        try:
            # Preprocess observation using BC scaler (X_mean, X_scale)
            input_obs = obs.copy()
            if self.bc_scaler_mean is not None and self.bc_scaler_scale is not None:
                input_obs = (input_obs - self.bc_scaler_mean) / self.bc_scaler_scale

            # Run inference
            input_name = self.bc_session.get_inputs()[0].name
            output_name = self.bc_session.get_outputs()[0].name
            result = self.bc_session.run(
                [output_name], {input_name: input_obs.reshape(1, -1).astype(np.float32)}
            )

            # BC model outputs Tanh → [-1, 1]; denormalise with y_max
            action = result[0].flatten()
            if self.bc_y_max is not None:
                action = action * self.bc_y_max

            # Clip to action space
            action = np.clip(action, self.action_space.low, self.action_space.high)
            return action.astype(np.float32)

        except Exception as e:
            self.get_logger().warn(f"⚠️ BC inference failed: {e}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Factory function for vectorised environments
# ──────────────────────────────────────────────────────────────────────────────

def make_env(
    rank: int = 0,
    max_episode_time: float = 30.0,
    hit_threshold: float = 1.5,
    control_rate_hz: int = 50,
    randomize_on_reset: bool = True,
    use_bc_model_as_prior: bool = False,
    bc_model_path: str = None,
    target_trajectory: str = "static",
):
    """
    Create a DroneAttackEnv instance (factory for SubprocVecEnv).

    Args:
        rank: Environment index (used for seeding)
        max_episode_time: Maximum episode duration (seconds)
        hit_threshold: Distance threshold for a hit (metres)
        control_rate_hz: Control loop frequency (Hz)
        randomize_on_reset: Enable domain randomisation
        use_bc_model_as_prior: Use BC model for action prior
        bc_model_path: Path to BC ONNX model
        target_trajectory: Target movement pattern

    Returns:
        Callable that creates a DroneAttackEnv
    """

    def _init():
        env = DroneAttackEnv(
            max_episode_time=max_episode_time,
            hit_threshold=hit_threshold,
            control_rate_hz=control_rate_hz,
            randomize_on_reset=randomize_on_reset,
            use_bc_model_as_prior=use_bc_model_as_prior,
            bc_model_path=bc_model_path,
            target_trajectory=target_trajectory,
            namespace=f"env_{rank}",
        )
        env.reset(seed=42 + rank)
        return env

    return _init


# ──────────────────────────────────────────────────────────────────────────────
# Main (standalone test)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rclpy.init(args=sys.argv)

    env = DroneAttackEnv(
        max_episode_time=10.0,
        hit_threshold=1.5,
        control_rate_hz=50,
        randomize_on_reset=False,
    )

    try:
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"  Expected: (9,) — [u_los_xyz, lambda_xyz, rpy]")
        print(f"  Got:      {obs.shape}")
        print(f"  Obs:      {obs}")
        print(f"Initial distance: {env._get_distance():.1f}m")

        total_reward = 0.0
        for i in range(100):
            action = env.action_space.sample()  # 3-dim action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if i % 10 == 0:
                print(
                    f"Step {i}: distance={info['distance']:.1f}m, "
                    f"reward={reward:.2f}, total={total_reward:.2f}"
                )

            if terminated or truncated:
                print(f"Episode finished at step {i}: {info}")
                break

    finally:
        env.close()