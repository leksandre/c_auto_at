"""
rl_pipeline_launch.py — Launch file for the RL training pipeline.

Launches:
1. Gazebo reset service (for episode resets)
2. Moving target node (for target trajectory control)
3. Target pose bridge (Gazebo → ROS2)

Usage:
    ros2 launch los_guidance rl_pipeline_launch.py
    ros2 launch los_guidance rl_pipeline_launch.py trajectory:=circle
    ros2 launch los_guidance rl_pipeline_launch.py trajectory:=evasive speed:=5.0
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("los_guidance")

    # ── Launch Arguments ───────────────────────────────────────────────
    trajectory_arg = DeclareLaunchArgument(
        "trajectory",
        default_value="static",
        description="Target trajectory: static, linear, circle, random, evasive",
    )

    speed_arg = DeclareLaunchArgument(
        "speed",
        default_value="3.0",
        description="Target speed (m/s)",
    )

    turn_rate_arg = DeclareLaunchArgument(
        "turn_rate",
        default_value="0.3",
        description="Target turn rate (rad/s, for circle trajectory)",
    )

    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value="car_track",
        description="Gazebo world name",
    )

    # ── Nodes ──────────────────────────────────────────────────────────

    # 1. Gazebo Reset Service
    gazebo_reset_node = Node(
        package="los_guidance",
        executable="gazebo_reset_service",
        name="gazebo_reset_service",
        output="screen",
        parameters=[
            {
                "world_name": LaunchConfiguration("world_name"),
                "drone_model_name": "x500_0",
                "target_model_name": "prius_hybridx_apriltag",
                "default_drone_pose": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "default_target_pose": [50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ],
    )

    # 2. Moving Target Node
    moving_target_node = Node(
        package="los_guidance",
        executable="moving_target_node",
        name="moving_target_node",
        output="screen",
        parameters=[
            {
                "trajectory": LaunchConfiguration("trajectory"),
                "speed": LaunchConfiguration("speed"),
                "turn_rate": LaunchConfiguration("turn_rate"),
                "update_rate_hz": 50.0,
                "world_name": LaunchConfiguration("world_name"),
                "target_model_name": "prius_hybridx_apriltag",
                "random_seed": 42,
                "waypoint_radius": 20.0,
                "evasive_jerk_interval_s": 3.0,
            }
        ],
    )

    # 3. ROS2-Gazebo Bridge for target pose
    #    This bridges /target/pose from Gazebo to ROS2
    #    (Requires ros_gz_bridge package)
    target_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="target_pose_bridge",
        arguments=[
            "/target/pose@gz.msgs.Pose@ros2.msg.geometry_msgs.Pose",
        ],
        output="screen",
    )

    # ── Launch Description ─────────────────────────────────────────────
    ld = LaunchDescription()

    ld.add_action(trajectory_arg)
    ld.add_action(speed_arg)
    ld.add_action(turn_rate_arg)
    ld.add_action(world_name_arg)

    ld.add_action(gazebo_reset_node)
    ld.add_action(moving_target_node)
    ld.add_action(target_pose_bridge)

    return ld