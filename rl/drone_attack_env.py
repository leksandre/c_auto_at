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
    VehicleOdometry, VehicleLocalPosition, VehicleStatus, TrajectorySetpoint,
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
        max_distance: float = 100.0,
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
        rpy_smooth_alpha: float = 0.3,
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
        self.target_speed = 0.0
        self.namespace = namespace
        self.render_mode = render_mode
        # Speed-up: use short real-time sleep instead of full dt
        # When Gazebo runs at N× real-time, sim produces data N× faster.
        # We only need a small real-time pause to let sim advance.
        self._use_sim_sleep = True
        self._sim_sleep_factor = 0.1  # sleep dt * factor (0.1 = 2ms at 50Hz)
        # Delay ATTACK START log by 4s after model creation to avoid
        # log spam during SB3 model initialization.
        self._env_created_time = time.time()

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

        # RPY smoothing (EMA filter) — reduces noise from odom quaternion
        self._rpy_smooth_alpha = rpy_smooth_alpha  # EMA coefficient (0-1, smaller = smoother)
        self._rpy_prev = np.zeros(3)
        self.prev_distance = np.inf
        self.vehicle_status = None
        self._prev_rel_pos = None
        self._odom_received = False
        self._target_received = False
        self._last_odom_time = 0.0
        # Last observation — used by _send_action to check target visibility
        self._last_obs = np.zeros(9, dtype=np.float32)

        # Episode tracking
        self.episode_start_time = 0.0
        self.step_count = 0
        self.episode_reward = 0.0
        # FIX: Grace period prevents false CRASH detection during takeoff.
        # After reset/Gazebo teleport, drone is at ground level (altitude=0).
        # RL agent sends random actions for ~2s before stabilising.
        # Without this, every episode immediately terminates with CRASH penalty.
        self._crash_grace_steps = 100  # 100 steps x 20ms = 2 seconds

        # Hit tracking — counts successful attacks across episodes
        self._total_hits = 0
        self._total_episodes = 0
        self._reset_alt = 0.0
        # First episode flag — arm + takeoff only once at startup
        self._first_episode = True

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

        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            "fmu/out/vehicle_local_position",
            self._local_pos_cb,
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
        """Cache vehicle odometry with timesync glitch protection."""
        pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])

        # Timesync glitch filter: reject odom with NaN or large jumps
        if np.any(np.isnan(pos)) or np.any(np.isnan(vel)):
            return
        if self._odom_received:
            jump = np.linalg.norm(pos - self.drone_pos)
            if jump > 10.0:  # >10m in one step = timesync glitch
                return

        self.drone_pos = pos
        self.drone_vel = vel
        # Quaternion → RPY with EMA smoothing
        q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
        rpy_raw = self._quat_to_rpy(q)
        # EMA: smoothed = prev + alpha * (raw - prev)
        self.drone_rpy = self._rpy_prev + self._rpy_smooth_alpha * (rpy_raw - self._rpy_prev)
        self._rpy_prev = self.drone_rpy.copy()
        self._last_odom_time = time.time()
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info(f"First odom received: pos={self.drone_pos}")

    def _local_pos_cb(self, msg):
        """Fallback: use vehicle_local_position when odom is not published.

        FIX: PX4 in this config does not publish vehicle_odometry topic,
        only vehicle_local_position. Without this callback, odom never
        arrives and drone_pos stays at [0,0,0] forever.
        """
        if self._odom_received:
            return  # odom is working, don't override

        pos = np.array([float(msg.x), float(msg.y), float(msg.z)])
        vel = np.array([float(msg.vx), float(msg.vy), float(msg.vz)])

        # Timesync glitch filter
        if np.any(np.isnan(pos)) or np.any(np.isnan(vel)):
            return

        self.drone_pos = pos
        self.drone_vel = vel
        # Yaw from heading with EMA smoothing
        yaw_raw = float(msg.heading)
        self.drone_rpy[2] = self._rpy_prev[2] + self._rpy_smooth_alpha * (yaw_raw - self._rpy_prev[2])
        self._rpy_prev[2] = self.drone_rpy[2]
        self._last_odom_time = time.time()
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info(f"Local position received (fallback): pos={self.drone_pos}")

    def _status_cb(self, msg):
        """Cache vehicle status."""
        self.vehicle_status = msg

    def _target_cb(self, msg):
        """Cache target pose from Gazebo."""
        self.target_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        if not self._target_received:
            self._target_received = True
            self.get_logger().info(f"🎯 First target pose received: {self.target_pos}")

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset episode — recover drone and start fresh.

        Reset flow:
        1. If crashed/flipped on ground → soft reset (disarm→wait→rearm)
        2. If airborne → reposition only
        3. Climb to random altitude
        4. Face camera toward car
        5. Hold position for stabilisation
        6. Return initial observation
        """
        super().reset(seed=seed)

        self.get_logger().info("Resetting...")

        # Detect crash state: drone on ground (alt < 1m) or flipped (large roll/pitch)
        altitude = -self.drone_pos[2] if self._odom_received else 0.0
        roll_abs = abs(self.drone_rpy[0])
        pitch_abs = abs(self.drone_rpy[1])
        is_crashed = (altitude < 1.0) or (roll_abs > 0.8 or pitch_abs > 0.8)  # ~45deg

        if is_crashed or self._first_episode:
            if self._first_episode:
                self.get_logger().info("First episode — arming and entering OFFBOARD...")
                self._arm_and_takeoff()
                self._first_episode = False
            else:
                self.get_logger().warn(
                    f"🔄 Crash detected (alt={altitude:.1f}m, roll={np.degrees(roll_abs):.0f}°, "
                    f"pitch={np.degrees(pitch_abs):.0f}°) — soft reset")
                self._soft_reset()
                # Wait for odom to be fresh after rearm
                for _ in range(50):
                    rclpy.spin_once(self, timeout_sec=0.01)
                    if self._odom_received and (time.time() - self._last_odom_time < 1.0):
                        break
                    self._sim_sleep(0.1)

            self._reset_alt = -self.drone_pos[2]
        else:
            # Subsequent episodes: reposition only
            odom_wait = time.time()
            while time.time() - odom_wait < 5.0:
                if time.time() - self._last_odom_time < 1.0:
                    break
                self._sim_sleep(0.1)

            distance = self._get_distance()
            if distance > 60.0:
                self.get_logger().info(f"Too far ({distance:.0f}m) — flying back...")
                self._fly_toward_target(distance_threshold=60.0, timeout=20.0)

            self._reset_alt = np.random.uniform(5.0, 20.0)
            self._fly_to_altitude(self._reset_alt, timeout=15.0)

        # PHASE: Hover + forced turn to target (3s)
        # Drone must face the car before RL attack begins.
        # During this phase, step() will not be called — only position+yaw commands.
        self.get_logger().info("HOVER PHASE: turning camera toward car...")
        target_yaw = self._compute_yaw_to_target()
        self.get_logger().info(
            f"  Target yaw: {np.degrees(target_yaw):.1f} deg, "
            f"Alt: {-self.drone_pos[2]:.1f}m, "
            f"Dist: {self._get_distance():.1f}m"
        )
        self._hover_and_face_target(duration=5.0)
        self.get_logger().info("HOVER PHASE complete — attack starting")

        # 4. Reset internal state
        self.prev_distance = self._get_distance()
        self._prev_rel_pos = None
        self.episode_start_time = time.time()
        self.step_count = 0
        self.episode_reward = 0.0
        self._total_episodes += 1
        self._rpy_prev = self.drone_rpy.copy()  # Reset EMA state for new episode

        # 5. Get initial observation
        obs = self._get_obs()

        self.get_logger().info(
            f"✅ Episode reset complete. "
            f"Alt: {-self.drone_pos[2]:.1f}m, "
            f"Dist: {self._get_distance():.1f}m, "
            f"Episodes: {self._total_episodes}, Hits: {self._total_hits}"
        )

        # 6. Log attack start block (only after 4s from model creation)
        env_age = time.time() - self._env_created_time
        if env_age > 4.0:
            odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else -1
            self.get_logger().info(
                f"{'='*60}\n"
                f"ATTACK START — Episode\n"
                f"{'='*60}\n"
                f"  Drone NED: [{self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f}]\n"
                f"  Target NED: [{self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}, {self.target_pos[2]:.2f}]\n"
                f"  Distance: {self._get_distance():.1f}m\n"
                f"  Yaw to target: {np.degrees(self._compute_yaw_to_target()):.1f} deg\n"
                f"  Odom: {'fresh' if 0 <= odom_age < 2 else 'never' if odom_age < 0 else 'stale'}\n"
                f"  NN INPUT: [{obs[0]:.4f}, {obs[1]:.4f}, {obs[2]:.4f}, {obs[3]:.4f}, {obs[4]:.4f}, {obs[5]:.4f}, {obs[6]:.4f}, {obs[7]:.4f}, {obs[8]:.4f}]\n"
                f"{'='*60}"
            )

        return obs, {}

    def _fly_toward_target(self, distance_threshold=60.0, timeout=20.0):
        """Fly drone toward target until within distance_threshold.

        Used during reset when drone is >60m from car.
        Sends velocity commands toward target at 5 m/s.
        """
        start = time.time()
        speed = 5.0  # m/s toward target

        while time.time() - start < timeout:
            dist = self._get_distance()
            if dist <= distance_threshold:
                self.get_logger().info(f"✅ Within {distance_threshold}m (now {dist:.0f}m)")
                return

            target_yaw = self._compute_yaw_to_target()
            dx = self.target_pos[0] - self.drone_pos[0]
            dy = self.target_pos[1] - self.drone_pos[1]
            norm = max(dist, 0.1)
            vx = dx / norm * speed
            vy = dy / norm * speed

            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.velocity = True
            self.offboard_pub.publish(offboard_msg)

            ts = TrajectorySetpoint()
            ts.velocity = [float(vx), float(vy), float("nan")]
            ts.yaw = float(target_yaw)
            ts.timestamp = int(time.time() * 1e6)
            self.setpoint_pub.publish(ts)
            self._sim_sleep(0.1)

        self.get_logger().warn(f"Fly-to-target timeout ({timeout}s)")

    def _hover_and_face_target(self, duration=5.0):
        """Hover at current position and rotate camera to face target.

        Dedicated phase before RL attack begins. Sends position setpoints
        with NaN XY (hold position) + computed yaw to face car.
        Re-checks yaw every iteration because target may be moving.
        Attack does not start until this phase completes.
        """
        start = time.time()
        while time.time() - start < duration:
            target_yaw = self._compute_yaw_to_target()

            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.position = True
            self.offboard_pub.publish(offboard_msg)

            ts = TrajectorySetpoint()
            ts.position = [float("nan"), float("nan"), float("nan")]
            ts.velocity = [float("nan"), float("nan"), float("nan")]
            ts.yaw = float(target_yaw)
            ts.timestamp = int(time.time() * 1e6)
            self.setpoint_pub.publish(ts)
            self._sim_sleep(0.1)

    def _sim_sleep(self, duration):
        """Spin ROS for  sim-time seconds.
        At 5x real-time this completes in ~0.2x wall-time.
        """
        wall_start = time.time()
        sim_start = self.get_clock().now().nanoseconds * 1e-9
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0005)
            sim_now = self.get_clock().now().nanoseconds * 1e-9
            wall_elapsed = time.time() - wall_start
            if (sim_now - sim_start) >= duration:
                return
            if wall_elapsed > duration * 3.0:
                return
    def _sim_wait_for_sim_time(self, sim_duration, wall_timeout=10.0):
        """Wait until sim clock advances by sim_duration seconds."""
        self._sim_sleep(sim_duration)
    def _fly_to_altitude(self, target_alt, timeout=10.0):
        """Climb/descend to target altitude. NaN XY = hold current position.

        Used during reset to set drone to random altitude 5-20m.
        Sends position setpoints with NaN XY (PX4 holds current XY).
        """
        start = time.time()
        self.get_logger().info(f"Flying to altitude {target_alt:.1f}m (current: {-self.drone_pos[2]:.1f}m)")

        while time.time() - start < timeout:
            current_alt = -self.drone_pos[2]
            odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else 999

            if odom_age < 2.0 and abs(current_alt - target_alt) < 0.5:
                self.get_logger().info(f"Altitude {current_alt:.1f}m reached")
                return

            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.position = True
            self.offboard_pub.publish(offboard_msg)

            ts = TrajectorySetpoint()
            ts.position = [float("nan"), float("nan"), -float(target_alt)]
            ts.velocity = [float("nan"), float("nan"), float("nan")]
            ts.yaw = float("nan")
            ts.timestamp = int(time.time() * 1e6)
            self.setpoint_pub.publish(ts)
            self._sim_sleep(0.1)

        self.get_logger().warn(f"Altitude timeout ({timeout}s, wanted {target_alt:.1f}m, got {-self.drone_pos[2]:.1f}m)")

    def step(self, action):
        """Execute one RL step: send action, wait, observe, reward."""
        self.step_count += 1

        # Get observation first — _send_action needs it for visibility check
        obs = self._get_obs()
        self._last_obs = obs

        # 1. Send action to PX4
        self._send_action(action)

        # 2. Wait for control cycle (use sim time for faster-than-real-time)
        if hasattr(self, '_use_sim_sleep') and self._use_sim_sleep:
            self._sim_sleep(self.dt)
        else:
            time.sleep(self.dt)

        # 3. Compute reward
        reward = self._compute_reward(action)

        # 4. Check termination conditions
        terminated, truncated = self._check_done()

        # 6. Update tracking
        self.prev_distance = self._get_distance()
        self.episode_reward += reward

        # 6b. Periodic debug log — shows what RL agent sees and does.
        #     Prints every 20 steps (0.4s) to avoid log spam.
        odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else -1
        odom_ok = odom_age >= 0 and odom_age < 2.0
        if self.step_count % 20 == 0:
            R = self._rpy_to_rot(self.drone_rpy)
            acc_nav = R @ np.array(action[:3], dtype=np.float64)
            self.get_logger().info(
                f"{'─'*55}\n"
                f"  📥 NN INPUT (obs[{self.step_count}]):\n"
                f"     u_los = [{obs[0]:+.4f}, {obs[1]:+.4f}, {obs[2]:+.4f}]\n"
                f"     lambda= [{obs[3]:+.4f}, {obs[4]:+.4f}, {obs[5]:+.4f}]\n"
                f"     rpy   = [{np.degrees(obs[6]):+.1f} deg, {np.degrees(obs[7]):+.1f} deg, {np.degrees(obs[8]):+.1f} deg]\n"
                f"  🚀 NEURAL NETWORK ACCELERATION (3 values from NN):\n"
                f"     >>> acc_nn = [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}] m/s2 <<<\n"
                f"     (body FRD -> nav NED: [{acc_nav[0]:+.3f}, {acc_nav[1]:+.3f}, {acc_nav[2]:+.3f}])\n"
                f"  📊 State:\n"
                f"     pos   = [{self.drone_pos[0]:.1f}, {self.drone_pos[1]:.1f}, {self.drone_pos[2]:.1f}]\n"
                f"     alt   = {-self.drone_pos[2]:.1f}m\n"
                f"     dist  = {self._get_distance():.1f}m\n"
                f"     reward= {reward:+.3f}  total={self.episode_reward:.1f}\n"
                f"{'─'*55}"
            )

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

        # 4b. Horizontal velocity penalty (discourage sideways drift)
        # FIX: Without this, RL agent learns to hover at high altitude
        # instead of descending towards target. Penalising horizontal
        # velocity encourages the drone to move towards the target rather
        # than drifting sideways or maintaining altitude.
        horiz_vel = np.sqrt(self.drone_vel[0]**2 + self.drone_vel[1]**2)
        reward -= 0.01 * horiz_vel

        # 5. Terminal rewards (applied when episode ends)
        # FIX: Reward crash penalty only after grace period, same as _check_done().
        # Without this, reward=-100 accumulates during grace period, making
        # the policy learn that "being on the ground = bad" instead of learning
        # to fly towards the target.
        if distance < self.hit_threshold:
            reward += 200.0  # HIT!
        elif -self.drone_pos[2] < self.min_altitude and self.step_count > self._crash_grace_steps:
            odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else 999
            if odom_age < 2.0:  # only penalise if odom is fresh (we know it's a real crash)
                reward -= 100.0  # CRASH
        elif distance > self.max_distance:
            reward -= 50.0  # LOST TARGET

        # FIX: Altitude shaping — reward being at target altitude.
        # Without this, drone approaches from below (ground level),
        # hits the ground before reaching the car.
        target_alt = -self.target_pos[2]  # NED -> altitude
        drone_alt = -self.drone_pos[2]
        alt_error = abs(drone_alt - target_alt)
        if alt_error < 3.0:
            reward += 1.0  # good altitude
        elif alt_error < 10.0:
            reward += 0.3
        else:
            reward -= 0.5 * (alt_error / 10.0)  # penalise being far from target altitude

        return float(reward)

    def _check_done(self):
        """Check if episode should terminate or truncate.

        Terminal conditions (episode ends):
        - HIT: drone within hit_threshold of target
        - CRASH: altitude below min_altitude (after grace period)
        - TOO HIGH: altitude above 100m
        - LOST TARGET: horizontal or total distance > max_distance

        Truncation (episode times out):
        - TIMEOUT: elapsed time > max_episode_time
        """
        distance = self._get_distance()
        elapsed = time.time() - self.episode_start_time

        # Periodic debug: show distance check values
        if self.step_count % 20 == 0:
            odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else -1
            odom_status = 'fresh' if 0 <= odom_age < 2 else ('never' if odom_age < 0 else 'stale '+str(round(odom_age,1))+'s')
            self.get_logger().info(
                f"  DONE CHECK: dist={distance:.2f}m (threshold={self.hit_threshold}m) "
                f"odom={odom_status} "
                f"drone_pos=[{self.drone_pos[0]:.1f},{self.drone_pos[1]:.1f},{self.drone_pos[2]:.1f}]"
            )

        # Terminal conditions
        # FIX: Only trigger HIT when odom is fresh — when odom is stale,
        # drone_pos stays at [0,0,0] and distance to target becomes 0
        # when car passes origin, causing false positive HIT.
        odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else 999
        odom_fresh = odom_age < 2.0

        if distance < self.hit_threshold and odom_fresh:
            self._total_hits += 1
            self.get_logger().info(
                f"HIT! Distance: {distance:.2f}m | "
                f"HITS: {self._total_hits}/{self._total_episodes} "
                f"({100*self._total_hits/max(1,self._total_episodes):.0f}%)"
            )
            return True, False
        elif distance < self.hit_threshold and not odom_fresh:
            pass  # odom stale — distance is unreliable, skip HIT

        # FIX: NED coordinate system — PX4 uses z-down, so altitude = -z.
        # Without this, drone at 15m (z=-15) would read as altitude=-15
        # and always trigger CRASH (since -15 < 0.5).
        altitude = -self.drone_pos[2]  # NED: z-down, altitude = -z

        # FIX: Grace period check BEFORE altitude check.
        # During first 100 steps (2s), drone is on ground or taking off.
        # Random RL actions cause altitude fluctuations that would
        # immediately trigger CRASH. Skip crash check during this period.
        if self.step_count <= self._crash_grace_steps:
            pass  # grace period — no crash check during first 2s
        elif time.time() - self._last_odom_time > 2.0:
            pass  # odom stale — PX4 EKF lost after Gazebo reset, can't check altitude
        elif altitude < self.min_altitude:
            self.get_logger().warn(f"💥 CRASH! Altitude: {altitude:.2f}m (z={self.drone_pos[2]:.2f})")
            return True, False
        elif altitude > 100.0:
            self.get_logger().warn(f"📐 TOO HIGH! Altitude: {altitude:.1f}m")
            return True, False

        # Horizontal distance limit (XY plane)
        horiz_dist = np.sqrt(
            (self.target_pos[0] - self.drone_pos[0])**2
            + (self.target_pos[1] - self.drone_pos[1])**2
        )
        if horiz_dist > self.max_distance:
            self.get_logger().warn(f"📡 LOST TARGET! Horizontal distance: {horiz_dist:.1f}m")
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
        """Send acceleration command to PX4 via OFFBOARD mode.

        Action is in body frame (FRD: Forward/Right/Down).
        PX4 TrajectorySetpoint expects acceleration in local NED frame
        (North/East/Down). We apply body->nav rotation matrix to convert.

        FIX: Without this rotation, body-frame X (forward) maps to NED Y
        when drone yaw=90. The drone would fly sideways instead of forward,
        causing it to "fly away in a random direction" — the original bug.

        FIX: Gravity compensation — PX4 OFFBOARD acceleration mode does NOT
        compensate gravity. NN must output +9.81 upward to hover.

        FIX: Target visibility check — if no valid target in camera frame,
        hover instead of sending NN acceleration. Prevents drone from
        flying blindly when LightTrack reports false positive.
        """
        # Check if target is visible in observation (u_los != default [1,0,0])
        target_visible = not (abs(self._last_obs[0] - 1.0) < 0.01
                              and abs(self._last_obs[1]) < 0.01
                              and abs(self._last_obs[2]) < 0.01
                              and abs(self._last_obs[3]) < 0.001
                              and abs(self._last_obs[4]) < 0.001
                              and abs(self._last_obs[5]) < 0.001)

        if not target_visible:
            # Target not visible — hover in place, face target direction
            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.position = True
            self.offboard_pub.publish(offboard_msg)

            ts = TrajectorySetpoint()
            ts.position = [float("nan"), float("nan"), float("nan")]
            ts.velocity = [float("nan"), float("nan"), float("nan")]
            ts.yaw = float(self._compute_yaw_to_target())
            ts.timestamp = int(time.time() * 1e6)
            self.setpoint_pub.publish(ts)
            return

        # Rotate body-frame acceleration to navigation (NED) frame
        R = self._rpy_to_rot(self.drone_rpy)  # body→nav rotation matrix
        acc_nav = R @ np.array(action[:3], dtype=np.float64)

        # FIX: PX4 OFFBOARD acceleration mode does NOT compensate gravity.
        # The NN must output +9.81 upward (NED: Z=-9.81) to hover.
        # Without this, untrained agent outputs ~0 and drone falls.
        GRAVITY_COMP = 9.81  # m/s^2 upward in NED = negative Z
        acc_nav[2] -= GRAVITY_COMP

        # FIX: Safety altitude override — untrained PPO sends random
        # accelerations including downward. Without this, drone crashes
        # into ground every episode before it can learn anything.
        # Hard floor at 2m, soft floor at 5m.
        current_alt = -self.drone_pos[2]
        if current_alt < 2.0:
            acc_nav[2] = min(acc_nav[2], -5.0)  # force strong climb
        elif current_alt < 5.0 and acc_nav[2] > 0:
            acc_nav[2] = min(acc_nav[2], -2.0)  # prevent descent

        msg = TrajectorySetpoint()
        msg.timestamp = int(time.time() * 1e6)

        msg.acceleration = [float(acc_nav[0]), float(acc_nav[1]), float(acc_nav[2])]

        # NaN for unused fields (position/velocity control disabled)
        msg.position = [float("nan")] * 3
        msg.velocity = [float("nan")] * 3

        # FIX: Always face the target — without this, drone maintains
        # takeoff heading and camera never points at the car.
        # Drone turned right side to target without pi/2 offset.
        target_yaw = self._compute_yaw_to_target()
        msg.yaw = float(target_yaw)
        msg.yawspeed = float("nan")

        self.setpoint_pub.publish(msg)

        # Offboard mode heartbeat — every step to keep PX4 in OFFBOARD
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(time.time() * 1e6)
        offboard_msg.acceleration = True
        self.offboard_pub.publish(offboard_msg)

    def _hold_position(self, duration: float = 2.0, target_yaw: float = None):
        """Hold current position for `duration` seconds using position control.

        Used after takeoff to let the drone stabilise before RL takes over.
        Without this, random RL actions start immediately after takeoff,
        causing the drone to drift sideways before stabilising.

        Args:
            duration: How long to hold position (seconds)
            target_yaw: If set, rotate to face this yaw angle (rad) while holding.
                       Used to ensure camera points at target before RL starts.
        """
        # FIX: Use altitude only (NaN XY = hold current position).
        # Previously used hold_pos = drone_pos.copy() which captured
        # stale [0,0,0] when odom was not working, sending drone to ground.
        hold_alt = -self.drone_pos[2]  # capture current altitude
        hold_start = time.time()
        yaw_str = f", yaw={np.degrees(target_yaw):.1f}°" if target_yaw is not None else ""
        self.get_logger().info(f"Holding at alt={hold_alt:.1f}m{yaw_str} for {duration}s...")
        while time.time() - hold_start < duration:
            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.position = True
            self.offboard_pub.publish(offboard_msg)

            ts = TrajectorySetpoint()
            ts.position = [float("nan"), float("nan"), -float(hold_alt)]
            ts.velocity = [float("nan"), float("nan"), float("nan")]
            ts.yaw = float(target_yaw) if target_yaw is not None else float("nan")
            ts.timestamp = int(time.time() * 1e6)
            self.setpoint_pub.publish(ts)
            self._sim_sleep(0.05)
        self.get_logger().info("Position hold complete")

    # ──────────────────────────────────────────────────────────────────────
    # Simulation Control
    # ──────────────────────────────────────────────────────────────────────

    def _try_gazebo_reset(self):
        """Try Gazebo reset via service. Returns True if successful.

        Calls /gazebo/reset_simulation service provided by
        gazebo_reset_service node (from rl_pipeline_launch.py).
        This teleports drone + target back to spawn positions and
        resets Gazebo physics. If service is unavailable, returns
        False and caller falls back to manual descent.
        """
        try:
            if self.reset_sim_client.wait_for_service(timeout_sec=2.0):
                req = Trigger.Request()
                future = self.reset_sim_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if future.result() and future.result().success:
                    self.get_logger().info("✅ Gazebo reset successful")
                    return True
                else:
                    self.get_logger().warn("⚠️ Gazebo reset returned false")
            else:
                self.get_logger().warn("⚠️ Gazebo reset service not available")
        except Exception as e:
            self.get_logger().warn(f"⚠️ Gazebo reset failed: {e}")
        return False

    def _reset_simulation(self):
        """Reset Gazebo simulation via service calls."""
        reset_ok = False
        try:
            if self.reset_sim_client.wait_for_service(timeout_sec=2.0):
                req = Trigger.Request()
                future = self.reset_sim_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if future.result() and future.result().success:
                    self.get_logger().info("✅ Simulation reset confirmed")
                    reset_ok = True
                else:
                    self.get_logger().warn("⚠️ Simulation reset returned false")
            elif self.reset_world_client.wait_for_service(timeout_sec=2.0):
                req = Trigger.Request()
                future = self.reset_world_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if future.result() and future.result().success:
                    self.get_logger().info("✅ World reset confirmed")
                    reset_ok = True
                else:
                    self.get_logger().warn("⚠️ World reset returned false")
            else:
                self.get_logger().warn(
                    "⚠️ No Gazebo reset service available. "
                    "Resetting internal state only."
                )
        except Exception as e:
            self.get_logger().warn(f"⚠️ Reset service call failed: {e}")

        # Always reset internal state regardless of Gazebo service
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_rpy = np.zeros(3)
        self.target_pos = np.array(self.target_start_pose, dtype=float)
        self.prev_distance = np.inf
        self._prev_rel_pos = None

        self._sim_sleep(1.0)

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

    def _disarm(self):
        """Disarm the drone — command PX4 to turn off motors.

        Unlike the old _disarm that tried to descend (which fails when
        drone is flipped), this just sends the disarm command.
        PX4 will stop motors and EKF will re-align from IMU when on ground.
        """
        self.get_logger().info("🔒 Disarming motors...")
        disarm_cmd = VehicleCommand()
        disarm_cmd.command = 400  # MAV_CMD_COMPONENT_ARM_DISARM
        disarm_cmd.param1 = 0.0  # DISARM
        disarm_cmd.target_system = 1
        disarm_cmd.target_component = 1
        disarm_cmd.source_system = 101
        disarm_cmd.source_component = 1
        disarm_cmd.from_external = True
        disarm_cmd.timestamp = int(time.time() * 1e6)
        self.cmd_pub.publish(disarm_cmd)
        self._sim_sleep(1.0)

    def _soft_reset(self):
        """Recover from crash without losing odom/IMU continuity.

        Flow: DISARM → wait 5s for EKF to realign → REARM + OFFBOARD → takeoff.
        No Gazebo reset → EKF state is preserved, just re-initialized from IMU.
        """
        self.get_logger().info("🔄 Soft reset: disarm → wait → rearm...")

        # 1. DISARM — motors off, EKF starts re-aligning from IMU
        self._disarm()

        # 2. Wait for EKF to stabilize (5s on ground → clean IMU integration)
        self.get_logger().info("⏳ Waiting 5s for EKF to realign...")
        self._sim_sleep(5.0)

        # 3. REARM + OFFBOARD
        self._arm_and_takeoff()

        self.get_logger().info("✅ Soft reset complete — ready for next episode")

    def _arm_and_takeoff(self):
        """Arm the drone and switch to OFFBOARD mode.

        In OFFBOARD mode, PX4 ignores MAV_CMD_NAV_TAKEOFF. Instead,
        takeoff is commanded via TrajectorySetpoint with a position target.
        This method only handles ARM + OFFBOARD; the actual climb is driven
        by position setpoints published in reset() during the wait loop.

        FIX: OFFBOARD heartbeat must use position=True (not acceleration=True)
        during takeoff phase. PX4 rejects position setpoints when OFFBOARD
        mode is set to acceleration control, causing the drone to not climb.
        """
        self.get_logger().info("🚁 Arming and switching to OFFBOARD...")

        # 1. Publish OffboardControlMode heartbeat BEFORE mode switch
        #    PX4 needs ≥2 Hz for ~1 second to accept OFFBOARD
        self.get_logger().info("📡 Publishing OFFBOARD heartbeat (2s)...")
        for _ in range(40):  # 2 seconds at 20 Hz
            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.position = True
            self.offboard_pub.publish(offboard_msg)

            # PX4 requires TrajectorySetpoint alongside OffboardControlMode
            ts = TrajectorySetpoint()
            ts.position = [float(self.drone_pos[0]), float(self.drone_pos[1]), float(self.drone_pos[2])]
            ts.velocity = [float("nan"), float("nan"), float("nan")]
            ts.yaw = float("nan")
            ts.timestamp = int(time.time() * 1e6)
            self.setpoint_pub.publish(ts)
            self._sim_sleep(0.05)

        # 2. Switch to OFFBOARD mode
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
        self._sim_sleep(1.0)

        # 3. Keep sending heartbeat to confirm OFFBOARD is active
        for _ in range(20):  # 1 second at 20 Hz
            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.position = True
            self.offboard_pub.publish(offboard_msg)

            ts = TrajectorySetpoint()
            ts.position = [float(self.drone_pos[0]), float(self.drone_pos[1]), float(self.drone_pos[2])]
            ts.velocity = [float("nan"), float("nan"), float("nan")]
            ts.yaw = float("nan")
            ts.timestamp = int(time.time() * 1e6)
            self.setpoint_pub.publish(ts)
            self._sim_sleep(0.05)

        # 4. ARM (after OFFBOARD is confirmed)
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
        self._sim_sleep(1.0)

        self.get_logger().info("✅ Armed, OFFBOARD mode active — takeoff via position setpoints")

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

    def _compute_yaw_to_target(self):
        """Compute yaw angle (rad) from drone to target in NED frame.

        Returns the heading the drone needs to face so its camera
        (mounted on the nose/body X) points towards the target.

        PX4 yaw convention: yaw=0 means body X = NED X (North).
        arctan2(dy, dx) gives the angle from NED X to the target,
        which is exactly the yaw needed to face it with the nose.
        """
        dx = self.target_pos[0] - self.drone_pos[0]
        dy = self.target_pos[1] - self.drone_pos[1]
        return float(np.arctan2(dy, dx))

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
            raw_action = result[0].flatten()
            action = raw_action.copy()
            if self.bc_y_max is not None:
                action = action * self.bc_y_max

            # Clip to action space
            action = np.clip(action, self.action_space.low, self.action_space.high)

            # ── 🪵 Debug log: BC model inference ────────────────────────
            # Shows what the Behaviour Cloning model outputs for comparison
            # with the RL policy. Useful for debugging: if BC action points
            # towards target but RL action doesn't, the RL policy needs more
            # training or better reward shaping.
            self.get_logger().info(
                f"🧠 BC inference: "
                f"input=[{obs[0]:.3f},{obs[1]:.3f},{obs[2]:.3f} | "
                f"{obs[3]:.3f},{obs[4]:.3f},{obs[5]:.3f} | "
                f"{np.degrees(obs[6]):.1f},{np.degrees(obs[7]):.1f},{np.degrees(obs[8]):.1f}°] → "
                f"raw=[{raw_action[0]:.4f},{raw_action[1]:.4f},{raw_action[2]:.4f}] → "
                f"denorm=[{action[0]:.3f},{action[1]:.3f},{action[2]:.3f}] m/s²"
            )

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
    rpy_smooth_alpha: float = 0.3,
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
        rpy_smooth_alpha: EMA coefficient for RPY smoothing (0-1, smaller = smoother)

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
            rpy_smooth_alpha=rpy_smooth_alpha,
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