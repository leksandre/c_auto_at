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
from ai_service.msg import TrackerResult

import threading
import time
import subprocess
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
        hit_threshold: float = 1.0,
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
        self._sim_sleep_factor = 0.1  # sleep dt * factor (0.1 = 2ms at 50Hz, stable value)
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
        # FIX: [-5,+5] was too small — gravity compensation needs +9.81
        # in body Z to hover. With [-5,+5] the NN physically cannot hover
        # and always falls. [-20,+20] allows hovering and matches BC model
        # output range (y_max=[15.4, 0.68, 0.29]).
        self.action_space = spaces.Box(
            low=np.array([-20.0, -20.0, -20.0], dtype=np.float32),
            high=np.array([20.0, 20.0, 20.0], dtype=np.float32),
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
        self._odom_received = False
        self._target_received = False
        self._last_odom_time = 0.0
        # Last observation — used by _send_action to check target visibility
        self._last_obs = np.zeros(9, dtype=np.float32)

        # ── Recovery mode (when LightTrack loses tracking) ─────────────
        self._recovery_active = False
        self._recovery_target_pos = np.zeros(3)  # NED position to fly to
        self._recovery_speed = 5.0  # m/s toward recovery point
        # Attack phase — only count hits after hover completes
        self._attack_started = False

        # ── LightTrack visual LOS state ──────────────────────────────────
        self._visual_los_nav = np.zeros(3)       # LOS in nav frame from bbox
        self._visual_lambda_nav = np.zeros(3)    # LOS angular velocity from bbox
        self._visual_los_valid = False            # Is visual LOS available?
        self._visual_los_prev = np.zeros(3)       # Previous visual LOS for lambda
        self._visual_los_initialized = False
        self._visual_lambda_smooth = np.zeros(3)
        self._visual_lambda_initialized = False
        self._tracker_last_timestamp = 0.0
        # Camera intrinsics (match TargetTracker config in C++)
        self._cam_fx = 410.0
        self._cam_fy = 410.0
        self._cam_cx = 640.0   # half image width (1280/2)
        self._cam_cy = 360.0   # half image height (720/2)
        self._cam_install_pitch_deg = -10.0  # camera tilt down
        self._cam_install_roll_deg = 0.0
        self._cam_install_yaw_deg = 0.0
        # EMA smoothing
        self._pixel_smooth_alpha = 0.3
        self._los_smooth_alpha = 0.1
        self._lambda_smooth_alpha = 0.3
        self._pixel_u_prev = 0.0
        self._pixel_v_prev = 0.0
        self._pixel_initialized = False
        # Precompute camera rotation matrices (body→camera, camera→body)
        self._R_bc, self._R_cb = self._init_camera_matrices()

        # Episode tracking
        self.episode_start_time = 0.0
        self.step_count = 0
        self.episode_reward = 0.0
        # FIX: Grace period prevents false CRASH detection during takeoff.
        # After reset/Gazebo teleport, drone is at ground level (altitude=0).
        # RL agent sends random actions for ~2s before stabilising.
        # Without this, every episode immediately terminates with CRASH penalty.
        self._crash_grace_steps = 100  # 100 steps x 20ms = 2 seconds
        # Stuck detection: if altitude unchanged for too long, episode is stuck
        self._stuck_steps = 0
        self._last_alt_for_stuck = 0.0
        self._stuck_threshold = 500  # 10 seconds at 50Hz with no altitude change
        # Soft reset failure counter — after N failures, request full pipeline restart
        self._consecutive_reset_fails = 0
        self._max_reset_fails = 3

        # Consecutive crash counter — after N crashes, request full pipeline restart
        self._consecutive_crashes = 0
        self._max_consecutive_crashes = 2

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

        # LightTrack bbox subscriber — provides visual LOS from camera tracking
        self.tracker_sub = self.create_subscription(
            TrackerResult,
            "/tracker_result",
            self._tracker_cb,
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
    # LightTrack visual LOS (bbox → camera → body → nav)
    # ──────────────────────────────────────────────────────────────────────

    def _init_camera_matrices(self):
        """Precompute body→camera and camera→body rotation matrices.
        Matches TargetTracker C++ initCameraMatrices_()."""
        pitch_c = np.radians(self._cam_install_pitch_deg)
        roll_c = np.radians(self._cam_install_roll_deg)
        yaw_c = np.radians(self._cam_install_yaw_deg)
        # R_install: rotation by install angles (body → rotated frame)
        cr, sr = np.cos(roll_c), np.sin(roll_c)
        cp, sp = np.cos(pitch_c), np.sin(pitch_c)
        cy, sy = np.cos(yaw_c), np.sin(yaw_c)
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R_bc_install = R_yaw @ R_pitch @ R_roll
        # Axis swap: Body(FRD) → Camera(RDF)
        axis_swap = np.array([
            [0, 1, 0],   # Body Y → Camera X
            [0, 0, 1],   # Body Z → Camera Y
            [1, 0, 0],   # Body X → Camera Z
        ])
        R_bc = axis_swap @ R_bc_install
        R_cb = R_bc.T
        return R_bc, R_cb

    def _tracker_cb(self, msg):
        """LightTrack /tracker_result callback. Converts bbox → visual LOS."""
        ti = msg.target_info
        if ti.confidence < 0.4:
            self._visual_los_valid = False
            return
        # Normalized [0,1] bbox → pixel coordinates
        pixel_u = (ti.left + ti.right) * self._cam_cx
        pixel_v = (ti.top + ti.bottom) * self._cam_cy
        # EMA smoothing of pixels
        if not self._pixel_initialized:
            pu, pv = pixel_u, pixel_v
            self._pixel_initialized = True
        else:
            a = self._pixel_smooth_alpha
            pu = self._pixel_u_prev + a * (pixel_u - self._pixel_u_prev)
            pv = self._pixel_v_prev + a * (pixel_v - self._pixel_v_prev)
        self._pixel_u_prev = pu
        self._pixel_v_prev = pv
        # Roll/pitch correction
        pu_corr, pv_corr = self._correct_pixel_roll_pitch(pu, pv)
        # Pixel → camera frame vector (pinhole model)
        x_c = (pu_corr - self._cam_cx) / self._cam_fx
        y_c = (pv_corr - self._cam_cy) / self._cam_fy
        u_c = np.array([x_c, y_c, 1.0])
        norm = np.linalg.norm(u_c)
        if norm < 1e-6:
            self._visual_los_valid = False
            return
        u_c = u_c / norm
        # Camera → body → nav
        u_b = self._R_cb @ u_c
        # nav frame rotation (NED): R_nb^T * u_b
        r, p, y = self.drone_rpy
        R_nb = self._rpy_to_rot([r, p, y])  # body→nav
        u_n = R_nb @ u_b  # body→nav
        norm_n = np.linalg.norm(u_n)
        if norm_n < 1e-6:
            self._visual_los_valid = False
            return
        u_n = u_n / norm_n
        # EMA smoothing of LOS
        if not self._visual_los_initialized:
            self._visual_los_nav = u_n
            self._visual_los_initialized = True
        else:
            a = self._los_smooth_alpha
            self._visual_los_nav = self._visual_los_nav + a * (u_n - self._visual_los_nav)
        # Lambda (LOS angular velocity): cross(los, d_los/dt)
        timestamp = msg.timestamp * 1e-9  # ns → s
        dt = timestamp - self._tracker_last_timestamp if self._tracker_last_timestamp > 0 else 0.02
        dt = max(dt, 1e-3)
        if self._visual_los_prev is not None and np.linalg.norm(self._visual_los_prev) > 0:
            u_dot = (self._visual_los_nav - self._visual_los_prev) / dt
            lambda_raw = np.cross(self._visual_los_nav, u_dot)
            # EMA smoothing of lambda
            if not self._visual_lambda_initialized:
                self._visual_lambda_nav = lambda_raw
                self._visual_lambda_initialized = True
            else:
                a = self._lambda_smooth_alpha
                self._visual_lambda_nav = self._visual_lambda_nav + a * (lambda_raw - self._visual_lambda_nav)
        self._visual_los_prev = self._visual_los_nav.copy()
        self._tracker_last_timestamp = timestamp
        self._visual_los_valid = True

    def _correct_pixel_roll_pitch(self, u, v):
        """Correct pixel coordinates for drone roll/pitch. Matches C++ correctPixelRollPitch_()."""
        x = (u - self._cam_cx) / self._cam_fx
        y = (v - self._cam_cy) / self._cam_fy
        r_cam = np.array([x, y, 1.0])
        norm = np.linalg.norm(r_cam)
        if norm < 1e-6:
            return u, v
        r_cam = r_cam / norm
        # Correction rotation: undo body roll/pitch from camera ray
        roll, pitch = self.drone_rpy[0], self.drone_rpy[1]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R_corr = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ]) @ np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ])
        r_corr = self._R_bc @ R_corr @ self._R_cb @ r_cam
        if abs(r_corr[2]) < 1e-4:
            return u, v
        u_corr = self._cam_cx + self._cam_fx * (r_corr[0] / r_corr[2])
        v_corr = self._cam_cy + self._cam_fy * (r_corr[1] / r_corr[2])
        u_corr = np.clip(u_corr, 0.0, 2 * self._cam_cx)
        v_corr = np.clip(v_corr, 0.0, 2 * self._cam_cy)
        return u_corr, v_corr

    # ──────────────────────────────────────────────────────────────────────
    # ROS2 Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        """Cache vehicle odometry with timesync glitch protection.

        FIX: After Gazebo teleport (set_pose/world reset), PX4 EKF
        reinitializes and reports a large position jump. The old code
        rejected this message AND all subsequent ones (because drone_pos
        stayed at old value, so every new message looked like a glitch).
        Now we accept large jumps after disarming (soft reset), allowing
        odom to recover.
        """
        pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])

        # Timesync glitch filter: reject odom with NaN or large jumps
        if np.any(np.isnan(pos)) or np.any(np.isnan(vel)):
            return
        if self._odom_received:
            jump = np.linalg.norm(pos - self.drone_pos)
            if jump > 10.0:
                # After soft reset (disarm + rearm), large jumps are expected
                # because Gazebo teleport resets model position. Accept the
                # new position instead of rejecting it forever.
                if jump < 100.0:
                    self.get_logger().info(
                        f"⚠️ Large odom jump {jump:.1f}m — accepting (likely post-reset)")
                else:
                    # Truly impossible jump (>100m) — skip
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

        if is_crashed and not self._first_episode:
            self._consecutive_crashes += 1
            self.get_logger().warn(
                f"💥 Crash #{self._consecutive_crashes}/{self._max_consecutive_crashes}")
            if self._consecutive_crashes >= self._max_consecutive_crashes:
                self.get_logger().error(
                    f"💀 {self._consecutive_crashes} consecutive crashes — "
                    "requesting full pipeline restart")
                raise RuntimeError(
                    f"CONSECUTIVE_CRASHES: {self._consecutive_crashes} crashes in a row. "
                    "Full pipeline restart required.")
        elif not is_crashed and not self._first_episode:
            self._consecutive_crashes = 0

        if is_crashed or self._first_episode:
            if self._first_episode:
                self.get_logger().info("First episode — waiting for PX4 to be ready...")
                self._wait_for_px4_ready(timeout=30.0)
                self._arm_and_takeoff()
                self._first_episode = False
            else:
                self.get_logger().warn(
                    f"🔄 Crash detected (alt={altitude:.1f}m, roll={np.degrees(roll_abs):.0f}°, "
                    f"pitch={np.degrees(pitch_abs):.0f}°) — soft reset")
                reset_ok = self._soft_reset()

                if not reset_ok:
                    # Soft reset failed too many times — signal training to restart
                    self.get_logger().error(
                        "💀 Soft reset failed — returning restart signal to training loop")
                    return np.zeros(9, dtype=np.float32), {"needs_restart": True}

                # Check if soft reset worked (odom fresh after 10s)
                for _ in range(100):
                    rclpy.spin_once(self, timeout_sec=0.01)
                    odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else 999
                    if odom_age < 2.0:
                        break
                    self._sim_sleep(0.1)

                odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else 999
                if odom_age > 10.0:
                    self.get_logger().warn(
                        f"⚠️ Odom stale {odom_age:.0f}s after soft reset — "
                        f"episode will terminate. External launcher will restart pipeline.")
                    # Don't try _restart_pipeline() — it's unreliable (DDS doesn't reconnect).
                    # Raise exception to trigger training exit and launcher restart.
                    raise RuntimeError("ODOM_DEAD: EKF lost after soft reset, needs full pipeline restart")

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
                # Verify odom is still alive after fly-back
                odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else 999
                if odom_age > 5.0:
                    self.get_logger().warn("Odom stale after fly-back — triggering restart")
                    raise RuntimeError("ODOM_DEAD: lost during fly-back, needs full pipeline restart")

            self._reset_alt = np.random.uniform(5.0, 20.0)
            self._fly_to_altitude(self._reset_alt, timeout=30.0)

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

        # Ensure minimum altitude before attack — if below 2m, climb first
        min_attack_alt = 2.0
        climb_ok = False
        for alt_attempt in range(5):
            current_alt = -self.drone_pos[2]
            if current_alt >= min_attack_alt:
                climb_ok = True
                break
            # EKF divergence check — if altitude > 50m, EKF is lost
            if current_alt > 50.0:
                self.get_logger().error(
                    f"💀 EKF DIVERGENCE: altitude={current_alt:.1f}m — "
                    f"EKF position estimate is wrong. Full restart needed.")
                raise RuntimeError(
                    f"EKF_DIVERGENCE: altitude {current_alt:.1f}m, expected <50m")
            self.get_logger().info(
                f"⬆️ Alt {current_alt:.1f}m < {min_attack_alt}m — climbing (attempt {alt_attempt+1}/5)...")
            self._fly_to_altitude(min_attack_alt + 1.0, timeout=30.0)

        if not climb_ok:
            current_alt = -self.drone_pos[2]
            if current_alt > 50.0:
                self.get_logger().error(
                    f"💀 EKF DIVERGENCE: altitude={current_alt:.1f}m after climb attempts")
                raise RuntimeError(
                    f"EKF_DIVERGENCE: altitude {current_alt:.1f}m after 5 climb attempts")
            self.get_logger().warn(
                f"⚠️ Cannot climb above {min_attack_alt}m after 5 attempts "
                f"(alt={current_alt:.1f}m) — drone stuck, ending episode")
            return np.zeros(9, dtype=np.float32), {"drone_stuck": True}
        target_yaw = self._compute_yaw_to_target()

        self._hover_and_face_target(duration=5.0)
        self.get_logger().info("HOVER PHASE complete — attack starting")
        self._attack_started = True

        # 4. Reset internal state
        self.prev_distance = self._get_distance()
        self.episode_start_time = time.time()
        self.step_count = 0
        self.episode_reward = 0.0
        self._total_episodes += 1
        self._rpy_prev = self.drone_rpy.copy()  # Reset EMA state for new episode
        self._stuck_steps = 0
        self._last_alt_for_stuck = -self.drone_pos[2]
        # Reset visual LOS state for new episode
        self._visual_los_valid = False
        self._visual_los_initialized = False
        self._visual_lambda_initialized = False
        self._visual_los_nav = np.zeros(3)
        self._visual_lambda_nav = np.zeros(3)
        self._visual_los_prev = np.zeros(3)
        self._pixel_initialized = False
        self._tracker_last_timestamp = 0.0
        # Reset recovery mode
        self._recovery_active = False

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

    def _fly_to_starting_position(self, min_dist=10.0, max_dist=20.0, timeout=30.0):
        """Fly drone to a random position at min_dist-max_dist from target.

        Used during reset to ensure drone starts at a meaningful distance
        for the NN to learn approach behavior.
        """
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        dist = np.random.uniform(min_dist, max_dist)

        target_x = self.target_pos[0] + dist * np.cos(angle)
        target_y = self.target_pos[1] + dist * np.sin(angle)
        target_z = self.drone_pos[2]  # keep current altitude

        self.get_logger().info(
            f"🎯 Flying to start position: dist={dist:.1f}m angle={np.degrees(angle):.0f}°")

        start = time.time()
        speed = 5.0

        while time.time() - start < timeout:
            dx = target_x - self.drone_pos[0]
            dy = target_y - self.drone_pos[1]
            remaining = np.sqrt(dx**2 + dy**2)

            if remaining < 1.0:
                self.get_logger().info(f"✅ Start position reached (remaining={remaining:.1f}m)")
                return

            norm = max(remaining, 0.1)
            vx = dx / norm * speed
            vy = dy / norm * speed

            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.velocity = True
            self.offboard_pub.publish(offboard_msg)

            ts = TrajectorySetpoint()
            ts.velocity = [float(vx), float(vy), float("nan")]
            ts.yaw = float(self._compute_yaw_to_target())
            ts.timestamp = int(time.time() * 1e6)
            self.setpoint_pub.publish(ts)
            self._sim_sleep(0.1)

        self.get_logger().warn(f"Start position timeout ({timeout}s, remaining={remaining:.1f}m)")

    def _hover_and_face_target(self, duration=5.0):
        """Hover at current position and rotate camera to face target.

        Dedicated phase before RL attack begins. Sends position setpoints
        with NaN XY (hold position) + computed yaw to face car.
        Re-checks yaw every iteration because target may be moving.
        Attack does not start until this phase completes.
        """
        start = time.time()
        hold_alt = -self.drone_pos[2]  # capture current altitude
        while time.time() - start < duration:
            target_yaw = self._compute_yaw_to_target()

            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.position = True
            self.offboard_pub.publish(offboard_msg)

            ts = TrajectorySetpoint()
            ts.position = [float("nan"), float("nan"), -float(hold_alt)]
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
        """Climb/descend to target altitude.

        PX4 OFFBOARD position control doesn't work from ground level.
        We use velocity-based takeoff first (climb at 2 m/s), then
        switch to position control once airborne.
        """
        start = time.time()
        self.get_logger().info(f"Flying to altitude {target_alt:.1f}m (current: {-self.drone_pos[2]:.1f}m)")

        climb_speed = 4.0  # m/s upward (NED: negative Z)

        while time.time() - start < timeout:
            current_alt = -self.drone_pos[2]
            odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else 999

            if odom_age < 2.0 and abs(current_alt - target_alt) < 0.5:
                self.get_logger().info(f"Altitude {current_alt:.1f}m reached")
                return

            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)

            ts = TrajectorySetpoint()
            ts.yaw = float("nan")
            ts.timestamp = int(time.time() * 1e6)

            if current_alt < target_alt - 0.5:
                # Below target: velocity climb (NED: up = negative Z)
                offboard_msg.velocity = True
                ts.velocity = [0.0, 0.0, -climb_speed]
                ts.position = [float("nan")] * 3
            else:
                # At or above target: position hold
                offboard_msg.position = True
                ts.position = [float("nan"), float("nan"), -float(target_alt)]
                ts.velocity = [float("nan"), float("nan"), float("nan")]

            self.offboard_pub.publish(offboard_msg)
            self.setpoint_pub.publish(ts)
            self._sim_sleep(0.05)

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
            distance = self._get_distance()
            altitude = -self.drone_pos[2]
            target_alt = -self.target_pos[2]
            yaw_to_target = self._compute_yaw_to_target()
            vel_speed = np.linalg.norm(self.drone_vel)
            h_dist = np.sqrt((self.target_pos[0]-self.drone_pos[0])**2 + (self.target_pos[1]-self.drone_pos[1])**2)
            elapsed = time.time() - self.episode_start_time
            remaining = max(0, self.max_episode_time - elapsed)
            self.get_logger().info(
                f"{'─'*55}\n"
                f"  🎯 POS: alt={altitude:.1f}m (target={target_alt:.1f}m) "
                f"dist={distance:.1f}m h_dist={h_dist:.1f}m\n"
                f"  📐 ORIENT: roll={np.degrees(self.drone_rpy[0]):+.0f}° "
                f"pitch={np.degrees(self.drone_rpy[1]):+.0f}° "
                f"yaw={np.degrees(self.drone_rpy[2]):+.0f}° "
                f"yaw2target={np.degrees(yaw_to_target):+.0f}°\n"
                f"  🚀 VEL: speed={vel_speed:.1f}m/s "
                f"vel=[{self.drone_vel[0]:+.1f},{self.drone_vel[1]:+.1f},{self.drone_vel[2]:+.1f}]\n"
                f"  ⚡ NN: acc=[{action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}] → "
                f"nav=[{acc_nav[0]:+.2f},{acc_nav[1]:+.2f},{acc_nav[2]:+.2f}]\n"
                f"  📊 step={self.step_count} t={remaining:.1f}s "
                f"reward={reward:+.2f} total={self.episode_reward:.1f}\n"
                f"{'─'*55}"
            )

        # 7. Info dict
        info = {
            "distance": self._get_distance(),
            "hit": terminated and self._get_distance() < self.hit_threshold and self._attack_started,
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
        Build 9-dim observation vector from LightTrack visual tracking ONLY.

        NO GPS data. Only camera-based bbox → LOS conversion.
        Ground truth from Gazebo used ONLY for reward (distance, hit check).

        Returns:
            [u_los_x, u_los_y, u_los_z,     # 0:3 — unit LOS vector (nav frame)
             lambda_x, lambda_y, lambda_z,   # 3:6 — LOS angular velocity (rad/s)
             roll, pitch, yaw]               # 6:9 — drone attitude (rad)
        """
        if not self._visual_los_valid:
            # No valid LightTrack tracking — return default (drone will hover)
            self.get_logger().warn("⚠️ No valid visual LOS from LightTrack!")
            return np.zeros(9, dtype=np.float32)

        u_los = self._visual_los_nav.copy()
        lambda_vec = self._visual_lambda_nav.copy()

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
        #    FIX: Increased from 3.0 to 10.0 — at 50Hz, closing at 1 m/s
        #    gives d_distance=-0.02, reward=+0.2. Time penalty is -0.05.
        #    Net +0.15/step makes closing distance strongly rewarded.
        reward -= 10.0 * d_distance

        # 2. Time penalty (encourages fast attacks)
        reward -= 0.05

        # 3. Progressive proximity bonuses (stronger to motivate approach)
        if distance < 50.0:
            reward += 0.5
        if distance < 30.0:
            reward += 1.0
        if distance < 20.0:
            reward += 2.0
        if distance < 10.0:
            reward += 5.0
        if distance < 5.0:
            reward += 10.0
        if distance < 2.0:
            reward += 20.0

        # 4. Action regularisation (penalise large/jerky actions)
        reward -= 0.005 * np.sum(action ** 2)

        # 4a. Deceleration penalty — negative acc_x = braking = longer attack
        # BC data had acc_x ∈ [0, 15.4], all positive. NN must learn to always
        # accelerate forward. Penalty scales linearly with deceleration magnitude.
        # At acc_x=-5: penalty=-15.0/step, dominates closing velocity (+10.0).
        if action[0] < 0.0:
            reward -= 3.0 * abs(action[0])

        # 4b. Sideways drift penalty (only perpendicular to LOS)
        # FIX: Original code penalised ALL horizontal velocity, including
        # movement toward the target. This prevented the drone from closing
        # distance. Now we only penalise the component perpendicular to LOS.
        horiz_vel = np.sqrt(self.drone_vel[0]**2 + self.drone_vel[1]**2)
        if distance > 1.0:
            # LOS direction in XY plane
            los_x = (self.target_pos[0] - self.drone_pos[0]) / distance
            los_y = (self.target_pos[1] - self.drone_pos[1]) / distance
            # Velocity component along LOS (positive = toward target)
            vel_along_los = self.drone_vel[0] * los_x + self.drone_vel[1] * los_y
            # Perpendicular component (sideways drift)
            vel_perp = abs(horiz_vel - abs(vel_along_los)) if horiz_vel > 0.1 else 0
            reward -= 0.02 * vel_perp

        # 5. Terminal rewards (applied when episode ends)
        # FIX: Reward crash penalty only after grace period, same as _check_done().
        # Without this, reward=-100 accumulates during grace period, making
        # the policy learn that "being on the ground = bad" instead of learning
        # to fly towards the target.
        if distance < self.hit_threshold and self._attack_started:
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

        # HIT: horizontal distance < 1.0m AND altitude < 2m AND odom fresh
        altitude = -self.drone_pos[2]  # NED: altitude = -z
        horiz_dist = np.sqrt(
            (self.target_pos[0] - self.drone_pos[0])**2
            + (self.target_pos[1] - self.drone_pos[1])**2
        )
        if horiz_dist < self.hit_threshold and altitude < 2.0 and odom_fresh and self._attack_started:
            self._total_hits += 1
            self.get_logger().info(
                f"HIT! h_dist={horiz_dist:.2f}m alt={altitude:.1f}m | "
                f"HITS: {self._total_hits}/{self._total_episodes} "
                f"({100*self._total_hits/max(1,self._total_episodes):.0f}%)"
            )
            return True, False

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
            self.get_logger().warn(f"📐 TOO HIGH! Altitude: {altitude:.1f}m — EKF divergence")
            raise RuntimeError(
                f"EKF_DIVERGENCE: altitude {altitude:.1f}m > 100m limit")

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

        # Stale odom timeout — episode is stuck if odom dead for >15s
        if self.step_count > self._crash_grace_steps and odom_age > 15.0:
            self.get_logger().error(
                f"💀 ODOM DEAD {odom_age:.0f}s — PX4 EKF lost. "
                f"Raising exception for full pipeline restart.")
            raise RuntimeError(
                f"ODOM_DEAD: PX4 EKF lost for {odom_age:.0f}s. "
                "Full pipeline restart required.")

        # Stuck detection — if altitude doesn't change for 5s, drone is stuck
        if self.step_count > self._crash_grace_steps:
            current_alt = -self.drone_pos[2]
            if abs(current_alt - self._last_alt_for_stuck) < 0.3:
                self._stuck_steps += 1
            else:
                self._stuck_steps = 0
                self._last_alt_for_stuck = current_alt

            if self._stuck_steps > self._stuck_threshold:
                self.get_logger().warn(
                    f"⚠️ DRONE STUCK — altitude unchanged ({current_alt:.1f}m) "
                    f"for {self._stuck_steps} steps. Ending episode.")
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

        FIX: Target visibility check — when LightTrack loses tracking,
        _get_obs() returns zeros. Drone hovers in place and yaws toward
        target (from Gazebo ground truth) until tracking resumes.
        """
        # Check if target is visible: observation must have non-zero u_los
        # When LightTrack loses tracking, _get_obs() returns zeros
        obs_norm = np.linalg.norm(self._last_obs[:3])
        target_visible = obs_norm > 0.1

        if not target_visible:
            # Lost tracking — recovery: fly to random altitude + random distance from car
            if not self._recovery_active:
                self._recovery_active = True
                # Random altitude 2-15m (NED: negative Z)
                alt = np.random.uniform(2.0, 15.0)
                # Random direction from car, distance 10-120m
                dist = np.random.uniform(10.0, 120.0)
                angle = np.random.uniform(0.0, 2.0 * np.pi)
                # Recovery position in NED
                self._recovery_target_pos = np.array([
                    self.target_pos[0] + dist * np.cos(angle),
                    self.target_pos[1] + dist * np.sin(angle),
                    -alt,  # NED: z-down
                ])
                self.get_logger().info(
                    f"🔍 Lost tracking — recovery: alt={alt:.1f}m, "
                    f"dist={dist:.1f}m, angle={np.degrees(angle):.0f}°"
                )

            # Fly toward recovery position using velocity
            drone_pos = self.drone_pos
            diff = self._recovery_target_pos - drone_pos
            dist_to_recovery = np.linalg.norm(diff)
            if dist_to_recovery > 1.0:
                # Still flying to recovery point
                vel = diff / max(dist_to_recovery, 0.1) * self._recovery_speed
                offboard_msg = OffboardControlMode()
                offboard_msg.timestamp = int(time.time() * 1e6)
                offboard_msg.velocity = True
                self.offboard_pub.publish(offboard_msg)

                ts = TrajectorySetpoint()
                ts.velocity = [float(vel[0]), float(vel[1]), float("nan")]
                ts.yaw = float(self._compute_yaw_to_target())
                ts.timestamp = int(time.time() * 1e6)
                self.setpoint_pub.publish(ts)
            else:
                # At recovery point — hover and yaw toward car
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
        else:
            # Tracking restored — cancel recovery
            self._recovery_active = False

        # Rotate body-frame acceleration to navigation (NED) frame
        R = self._rpy_to_rot(self.drone_rpy)  # body→nav rotation matrix
        acc_nav = R @ np.array(action[:3], dtype=np.float64)

        # Safety: clamp NaN/inf to prevent PX4 weirdness
        acc_nav = np.where(np.isfinite(acc_nav), acc_nav, 0.0)

        # FIX: PX4 OFFBOARD acceleration mode does NOT compensate gravity.
        # The NN must output +9.81 upward (NED: Z=-9.81) to hover.
        # Without this, untrained agent outputs ~0 and drone falls.
        GRAVITY_COMP = 9.81  # m/s^2 upward in NED = negative Z
        acc_nav[2] -= GRAVITY_COMP

        # ── Forward push: DISABLED for testing ──
        # When NN is untrained, drone just hovers. With bias=3.0, NN
        # output is irrelevant — bias dominates. Removing bias forces
        # NN to learn actual navigation.
        forward_bias = 0.0
        acc_nav[0] += forward_bias * np.cos(float(self.drone_rpy[2]))  # North
        acc_nav[1] += forward_bias * np.sin(float(self.drone_rpy[2]))  # East

        # ── Altitude safety floors: DISABLED for NN training ──
        # NN must learn to control altitude from reward signal.
        # With safety floors, NN output is overridden and never learns.

        # ── Tilt safety: if drone tilts > 20°, send position hold ──
        # Prevents drone from flipping when NN outputs aggressive sideways accel
        roll_abs = abs(self.drone_rpy[0])
        pitch_abs = abs(self.drone_rpy[1])
        tilt_limit = np.radians(30.0)
        if roll_abs > tilt_limit or pitch_abs > tilt_limit:
            self.get_logger().warn(
                f"⚠️ Tilt safety: roll={np.degrees(roll_abs):.0f}° pitch={np.degrees(pitch_abs):.0f}°"
                f" — position hold to recover")
            msg = TrajectorySetpoint()
            msg.timestamp = int(time.time() * 1e6)
            msg.position = [float("nan"), float("nan"), float(self.drone_pos[2])]
            msg.velocity = [float("nan"), float("nan"), float("nan")]
            msg.acceleration = [float("nan"), float("nan"), float("nan")]
            msg.yaw = float(self._compute_yaw_to_target())
            msg.yawspeed = float("nan")
            self.setpoint_pub.publish(msg)
            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(time.time() * 1e6)
            offboard_msg.position = True
            self.offboard_pub.publish(offboard_msg)
            return

        msg = TrajectorySetpoint()
        msg.timestamp = int(time.time() * 1e6)

        # Safety clamp: prevent extreme acceleration values
        # Clip XY to prevent flip — NN outputs large sideways accel → instant flip
        acc_nav[0] = np.clip(acc_nav[0], -5.0, 5.0)
        acc_nav[1] = np.clip(acc_nav[1], -5.0, 5.0)
        acc_nav = np.clip(acc_nav, -20.0, 20.0)
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
        """Try Gazebo reset via service. Returns True if successful."""
        try:
            if not self.reset_sim_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warn("⚠️ Gazebo reset service not available")
                return False

            req = Trigger.Request()
            future = self.reset_sim_client.call_async(req)

            # Spin executor to process the response (uses the node's executor)
            deadline = time.time() + 15.0
            while time.time() < deadline:
                # Try both spin_once and checking future directly
                rclpy.spin_once(self, timeout_sec=0.1)
                if future.done():
                    break

            if future.done():
                result = future.result()
                if result is not None and result.success:
                    self.get_logger().info("✅ Gazebo reset successful")
                    # Wait for physics to settle after reset
                    self._sim_sleep(2.0)
                    return True
                else:
                    self.get_logger().warn("⚠️ Gazebo reset returned false")
                    return False
            else:
                self.get_logger().warn("⚠️ Gazebo reset timed out (15s)")
                # Cancel the future to avoid resource leak
                future.cancel()
                return False
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
        """Recover from crash: disarm → wait → rearm.

        Does NOT use Gazebo set_pose — it causes PX4 EKF divergence.
        After disarm, EKF settles on ground. After rearm, drone takes off.
        If drone is flipped and can't climb, stuck detection terminates
        the episode and consecutive failures trigger full pipeline restart.
        """
        self.get_logger().info("🔄 Soft reset: disarm → wait → rearm...")

        # 1. DISARM — motors off, drone settles on ground
        self._disarm()
        # Reset odom state so glitch filter is fresh after rearm
        self._odom_received = False
        self._last_odom_time = 0.0
        self._sim_sleep(5.0)

        # 2. Wait for EKF to settle (drone on ground, IMU consistent)
        self.get_logger().info("⏳ Waiting 15s for EKF to settle on ground...")
        settle_start = time.time()
        odom_recovered = False
        while time.time() - settle_start < 15.0:
            rclpy.spin_once(self, timeout_sec=0.01)
            odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else 999
            if odom_age < 2.0:
                odom_recovered = True
                self.get_logger().info(
                    f"✅ EKF settled after {time.time()-settle_start:.1f}s")
                break
            self._sim_sleep(0.1)

        if not odom_recovered:
            self.get_logger().warn("⚠️ EKF still stale — attempting rearm anyway")

        # 3. REARM + OFFBOARD
        self._arm_and_takeoff()

        # 4. Check if rearm succeeded
        ARMING_STATE_ARMED = 2
        if (self.vehicle_status is None or
                self.vehicle_status.arming_state != ARMING_STATE_ARMED):
            self._consecutive_reset_fails += 1
            self.get_logger().error(
                f"❌ Soft reset FAILED (rearm failed, "
                f"consecutive fails: {self._consecutive_reset_fails}/{self._max_reset_fails})")
            if self._consecutive_reset_fails >= self._max_reset_fails:
                self.get_logger().error(
                    "💀 Too many consecutive soft reset failures — "
                    "requesting full pipeline restart")
                return False  # Signal to caller: needs full restart

        self._consecutive_reset_fails = 0
        # Diagnostic: log PX4 state after successful soft reset
        if self.vehicle_status:
            self.get_logger().info(
                f"✅ Soft reset complete — PX4: nav_state={self.vehicle_status.nav_state}, "
                f"arming={self.vehicle_status.arming_state}")
        else:
            self.get_logger().info("✅ Soft reset complete")
        return True  # Signal: success

    def _restart_pipeline(self):
        """DEPRECATED: Full pipeline restart from inside Python is unreliable.

        DDS bridge does not reconnect after PX4 restart, causing odom to
        stay dead. Use external launcher (stable_rl_launcher.sh) instead.
        """
        self.get_logger().error(
            "❌ _restart_pipeline() called but is DEPRECATED — "
            "DDS does not reconnect after restart. "
            "External launcher (stable_rl_launcher.sh) handles full restart.")
        return False

    def _wait_for_px4_ready(self, timeout=30.0):
        """Wait until PX4 reports fresh odom and is ready for arming.

        Checks:
        - vehicle_odometry topic is alive (odom_age < 2s)
        - vehicle_status received (not None)
        """
        self.get_logger().info(f"⏳ Waiting up to {timeout}s for PX4 to be ready...")
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            odom_age = time.time() - self._last_odom_time if self._last_odom_time > 0 else 999
            if odom_age < 2.0 and self.vehicle_status is not None:
                self.get_logger().info(
                    f"✅ PX4 ready after {time.time()-start:.1f}s "
                    f"(odom_age={odom_age:.1f}s)")
                return True
            self._sim_sleep(0.2)
        self.get_logger().warn(
            f"⚠️ PX4 not fully ready after {timeout}s — attempting arm anyway")
        return False

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

        # 2. Switch to OFFBOARD mode — publish heartbeat DURING mode switch
        #    FIX: Remove 1s dead zone where no heartbeats were sent.
        #    PX4's COM_OF_LOSS_T (500ms) would fire and revert to POSCTL.
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

        # Keep heartbeats flowing during mode switch (no dead zone!)
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

        # 3. Verify OFFBOARD mode is active after mode switch
        NAVIGATION_STATE_OFFBOARD = 14
        for _ in range(20):  # 1 second verification
            rclpy.spin_once(self, timeout_sec=0.05)
            if (self.vehicle_status is not None and
                    self.vehicle_status.nav_state == NAVIGATION_STATE_OFFBOARD):
                self.get_logger().info("✅ OFFBOARD mode confirmed (nav_state=14)")
                break
            # Re-send OFFBOARD mode if not active
            self.cmd_pub.publish(offboard_cmd)
            self._sim_sleep(0.05)
        else:
            self.get_logger().warn(
                f"⚠️ OFFBOARD mode not confirmed — nav_state="
                f"{self.vehicle_status.nav_state if self.vehicle_status else 'None'}")

        # 4. ARM with retry — verify PX4 actually arms
        ARMING_STATE_ARMED = 2  # px4_msgs::VehicleStatus::ARMING_STATE_ARMED
        max_arm_attempts = 10
        for attempt in range(1, max_arm_attempts + 1):
            # Spin to receive fresh vehicle_status
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.01)
                self._sim_sleep(0.01)

            # Check if already armed + OFFBOARD active
            if (self.vehicle_status is not None and
                    self.vehicle_status.arming_state == ARMING_STATE_ARMED):
                nav = self.vehicle_status.nav_state
                self.get_logger().info(
                    f"✅ Armed (attempt {attempt}) — nav_state={nav}, "
                    f"{'OFFBOARD confirmed' if nav == NAVIGATION_STATE_OFFBOARD else 'WARNING: not OFFBOARD'}")
                break

            # Not armed yet — send ARM command + OFFBOARD heartbeat
            self.get_logger().info(
                f"⏳ ARM attempt {attempt}/{max_arm_attempts} — "
                f"arming_state={self.vehicle_status.arming_state if self.vehicle_status else 'None'}")

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
        else:
            self.get_logger().error(
                f"❌ Failed to arm after {max_arm_attempts} attempts — "
                f"arming_state={self.vehicle_status.arming_state if self.vehicle_status else 'None'}")

        # 5. Verify OFFBOARD after ARM — PX4 may switch to AUTO_LOITER
        for _ in range(30):  # 1.5s verification
            rclpy.spin_once(self, timeout_sec=0.05)
            if (self.vehicle_status is not None and
                    self.vehicle_status.nav_state == NAVIGATION_STATE_OFFBOARD):
                self.get_logger().info("✅ OFFBOARD confirmed after ARM")
                break
            # Re-send OFFBOARD mode command
            self.cmd_pub.publish(offboard_cmd)
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
        else:
            nav = self.vehicle_status.nav_state if self.vehicle_status else 'None'
            self.get_logger().warn(f"⚠️ OFFBOARD not confirmed after ARM — nav_state={nav}")

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
            hit_threshold=1.0,
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