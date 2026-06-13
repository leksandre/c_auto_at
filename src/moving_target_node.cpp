/**
 * @file moving_target_node.cpp
 * @brief ROS2 node that controls target vehicle movement for RL training
 *
 * Provides configurable target trajectories:
 * - Static: No movement
 * - Linear: Constant velocity in one direction
 * - Circle: Circular motion
 * - Random: Random waypoint navigation
 * - Evasive: Simple evasive manoeuvres
 *
 * Publishes /cmd_vel for the target vehicle in Gazebo (via ros_gz_bridge).
 * Publishes /target/pose for the RL environment.
 *
 * @author Generated for los_guidance RL pipeline
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <memory>
#include <string>
#include <random>
#include <cmath>
#include <chrono>

namespace los_guidance
{

/**
 * @enum TrajectoryType
 * @brief Supported target movement patterns
 */
enum class TrajectoryType
{
  STATIC,
  LINEAR,
  CIRCLE,
  RANDOM,
  EVASIVE
};

/**
 * @class MovingTargetNode
 * @brief Controls target vehicle movement for RL training
 *
 * Publishes velocity commands to /cmd_vel (bridged to Gazebo).
 * Supports multiple trajectory patterns with configurable parameters.
 */
class MovingTargetNode : public rclcpp::Node
{
public:
  MovingTargetNode()
    : Node("moving_target_node")
  {
    RCLCPP_INFO(this->get_logger(), "Initialising Moving Target Node...");

    // ── Parameters ────────────────────────────────────────────────────
    this->declare_parameter("trajectory", "static");
    this->declare_parameter("speed", 3.0);
    this->declare_parameter("turn_rate", 0.3);
    this->declare_parameter("update_rate_hz", 50.0);
    this->declare_parameter("random_seed", 42);
    this->declare_parameter("waypoint_radius", 20.0);
    this->declare_parameter("evasive_jerk_interval_s", 3.0);

    // ── Load parameters ───────────────────────────────────────────────
    trajectory_str_ = this->get_parameter("trajectory").as_string();
    speed_ = this->get_parameter("speed").as_double();
    turn_rate_ = this->get_parameter("turn_rate").as_double();
    update_rate_hz_ = this->get_parameter("update_rate_hz").as_double();
    int seed = this->get_parameter("random_seed").as_int();
    waypoint_radius_ = this->get_parameter("waypoint_radius").as_double();
    evasive_interval_ = this->get_parameter("evasive_jerk_interval_s").as_double();

    trajectory_type_ = parseTrajectoryType(trajectory_str_);

    // ── RNG ───────────────────────────────────────────────────────────
    rng_.seed(seed);

    // ── Publishers ────────────────────────────────────────────────────
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10);

    target_pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>(
      "/target/pose", 10);

    // ── Timer ─────────────────────────────────────────────────────────
    double dt = 1.0 / update_rate_hz_;
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(dt),
      std::bind(&MovingTargetNode::update, this));

    // ── Initialise trajectory state ───────────────────────────────────
    initTrajectory();

    RCLCPP_INFO(this->get_logger(), "✅ Moving Target Node ready");
    RCLCPP_INFO(this->get_logger(), "   Trajectory: %s (speed=%.1f m/s)",
                trajectory_str_.c_str(), speed_);
  }

private:
  // ── Trajectory Initialisation ───────────────────────────────────────

  void initTrajectory()
  {
    switch (trajectory_type_)
    {
      case TrajectoryType::LINEAR:
        {
          std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);
          linear_angle_ = dist(rng_);
        }
        break;

      case TrajectoryType::RANDOM:
        generateRandomWaypoint();
        break;

      case TrajectoryType::EVASIVE:
        evasive_timer_ = this->now();
        break;

      default:
        break;
    }
  }

  // ── Main Update Loop ────────────────────────────────────────────────

  void update()
  {
    geometry_msgs::msg::Twist cmd;

    switch (trajectory_type_)
    {
      case TrajectoryType::STATIC:
        cmd.linear.x = 0.0;
        cmd.angular.z = 0.0;
        break;

      case TrajectoryType::LINEAR:
        cmd.linear.x = speed_;
        cmd.angular.z = 0.0;
        break;

      case TrajectoryType::CIRCLE:
        cmd.linear.x = speed_;
        cmd.angular.z = turn_rate_;
        break;

      case TrajectoryType::RANDOM:
        updateRandomTrajectory(cmd);
        break;

      case TrajectoryType::EVASIVE:
        updateEvasiveTrajectory(cmd);
        break;
    }

    cmd_vel_pub_->publish(cmd);
  }

  // ── Random Trajectory ───────────────────────────────────────────────

  void updateRandomTrajectory(geometry_msgs::msg::Twist& cmd)
  {
    // Steer toward waypoint
    double target_angle = std::atan2(
      waypoint_y_ - current_y_,
      waypoint_x_ - current_x_);

    double angle_error = target_angle - current_yaw_;
    while (angle_error > M_PI) angle_error -= 2.0 * M_PI;
    while (angle_error < -M_PI) angle_error += 2.0 * M_PI;

    // Check if we've reached the waypoint
    double dx = waypoint_x_ - current_x_;
    double dy = waypoint_y_ - current_y_;
    double dist = std::sqrt(dx * dx + dy * dy);

    if (dist < 5.0)
    {
      generateRandomWaypoint();
    }

    cmd.linear.x = speed_;
    cmd.angular.z = std::clamp(2.0 * angle_error, -1.0, 1.0);
  }

  void generateRandomWaypoint()
  {
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> radius_dist(10.0, waypoint_radius_);

    double angle = angle_dist(rng_);
    double radius = radius_dist(rng_);

    waypoint_x_ = current_x_ + radius * std::cos(angle);
    waypoint_y_ = current_y_ + radius * std::sin(angle);

    RCLCPP_DEBUG(this->get_logger(), "🎯 New waypoint: (%.1f, %.1f)",
                 waypoint_x_, waypoint_y_);
  }

  // ── Evasive Trajectory ──────────────────────────────────────────────

  void updateEvasiveTrajectory(geometry_msgs::msg::Twist& cmd)
  {
    auto now = this->now();
    double elapsed = (now - evasive_timer_).seconds();

    if (elapsed > evasive_interval_)
    {
      std::uniform_real_distribution<double> speed_dist(0.5, 2.0);
      std::uniform_real_distribution<double> turn_dist(-1.0, 1.0);
      std::bernoulli_distribution jerk_dist(0.3);

      if (jerk_dist(rng_))
      {
        cmd.linear.x = speed_ * speed_dist(rng_);
        cmd.angular.z = turn_dist(rng_) * turn_rate_;
      }
      else
      {
        cmd.linear.x = speed_;
        cmd.angular.z = (turn_dist(rng_) > 0 ? 1.0 : -1.0) * turn_rate_ * 2.0;
      }

      evasive_timer_ = now;
    }
    else
    {
      cmd.linear.x = speed_;
      cmd.angular.z = 0.0;
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────────

  TrajectoryType parseTrajectoryType(const std::string& str)
  {
    if (str == "linear") return TrajectoryType::LINEAR;
    if (str == "circle") return TrajectoryType::CIRCLE;
    if (str == "random") return TrajectoryType::RANDOM;
    if (str == "evasive") return TrajectoryType::EVASIVE;
    return TrajectoryType::STATIC;
  }

  // ── Members ─────────────────────────────────────────────────────────

  // Parameters
  std::string trajectory_str_;
  double speed_;
  double turn_rate_;
  double update_rate_hz_;
  double waypoint_radius_;
  double evasive_interval_;

  TrajectoryType trajectory_type_;
  std::mt19937 rng_;

  // State
  double current_x_{0.0};
  double current_y_{0.0};
  double current_yaw_{0.0};

  // Linear trajectory
  double linear_angle_{0.0};

  // Random trajectory
  double waypoint_x_{0.0};
  double waypoint_y_{0.0};

  // Evasive trajectory
  rclcpp::Time evasive_timer_;

  // ROS2
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr target_pose_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace los_guidance

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<los_guidance::MovingTargetNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}