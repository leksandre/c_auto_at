/**
 * @file gazebo_reset_service.cpp
 * @brief ROS2 service node for resetting Gazebo simulation
 *
 * Provides ROS2 services that the RL environment calls to reset episodes.
 * The actual Gazebo interaction is done via ros_gz_bridge and gz CLI.
 *
 * Services:
 *   /gazebo/reset_simulation  (std_srvs/Trigger)  — Reset everything
 *   /gazebo/reset_world       (std_srvs/Trigger)  — Reset models only
 *   /gazebo/pause_physics     (std_srvs/SetBool)  — Pause/resume physics
 *   /gazebo/randomize_poses   (std_srvs/Trigger)  — Randomise drone + target poses
 *
 * @author Generated for los_guidance RL pipeline
 */

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <memory>
#include <string>
#include <cstdlib>

namespace los_guidance
{

/**
 * @class GazeboResetService
 * @brief ROS2 node that provides simulation reset services
 *
 * Uses gz CLI commands (executed via system()) to interact with Gazebo.
 * This avoids direct dependency on gz-transport C++ library.
 */
class GazeboResetService : public rclcpp::Node
{
public:
  GazeboResetService()
    : Node("gazebo_reset_service")
  {
    RCLCPP_INFO(this->get_logger(), "Initialising Gazebo Reset Service...");

    // ── Parameters ────────────────────────────────────────────────────
    this->declare_parameter("world_name", "car_track");
    this->declare_parameter("drone_model_name", "x500_0");
    this->declare_parameter("target_model_name", "prius_hybridx_apriltag");

    world_name_ = this->get_parameter("world_name").as_string();
    drone_model_name_ = this->get_parameter("drone_model_name").as_string();
    target_model_name_ = this->get_parameter("target_model_name").as_string();

    // ── Services ──────────────────────────────────────────────────────
    reset_sim_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "gazebo/reset_simulation",
      std::bind(&GazeboResetService::handleResetSimulation, this,
                std::placeholders::_1, std::placeholders::_2));

    reset_world_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "gazebo/reset_world",
      std::bind(&GazeboResetService::handleResetWorld, this,
                std::placeholders::_1, std::placeholders::_2));

    pause_physics_srv_ = this->create_service<std_srvs::srv::SetBool>(
      "gazebo/pause_physics",
      std::bind(&GazeboResetService::handlePausePhysics, this,
                std::placeholders::_1, std::placeholders::_2));

    randomize_poses_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "gazebo/randomize_poses",
      std::bind(&GazeboResetService::handleRandomizePoses, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "✅ Gazebo Reset Service ready");
    RCLCPP_INFO(this->get_logger(), "   World: %s", world_name_.c_str());
  }

private:
  // ── Service Handlers ────────────────────────────────────────────────

  void handleResetSimulation(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;
    RCLCPP_INFO(this->get_logger(), "🔄 Resetting simulation...");

    // Use gz CLI to reset the world
    std::string cmd = "gz service -s /world/" + world_name_ + "/reset "
                      "--reqtype gz.msgs.WorldReset "
                      "--reptype gz.msgs.Boolean "
                      "--timeout 2000 "
                      "--req 'all: true'";

    int ret = std::system(cmd.c_str());

    if (ret == 0) {
      response->success = true;
      response->message = "Simulation reset successful";
      RCLCPP_INFO(this->get_logger(), "✅ Simulation reset successful");
    } else {
      response->success = false;
      response->message = "Failed to reset simulation (exit code: " +
                          std::to_string(ret) + ")";
      RCLCPP_WARN(this->get_logger(), "⚠️ Simulation reset failed (code %d)", ret);
    }
  }

  void handleResetWorld(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;
    RCLCPP_INFO(this->get_logger(), "🔄 Resetting world (keeping time)...");

    std::string cmd = "gz service -s /world/" + world_name_ + "/reset "
                      "--reqtype gz.msgs.WorldReset "
                      "--reptype gz.msgs.Boolean "
                      "--timeout 2000 "
                      "--req 'all: false'";

    int ret = std::system(cmd.c_str());

    if (ret == 0) {
      response->success = true;
      response->message = "World reset successful";
    } else {
      response->success = false;
      response->message = "Failed to reset world";
    }
  }

  void handlePausePhysics(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    bool pause = request->data;
    RCLCPP_INFO(this->get_logger(), "%s physics...", pause ? "⏸️ Pausing" : "▶️ Resuming");

    std::string cmd = "gz service -s /world/" + world_name_ + "/pause "
                      "--reqtype gz.msgs.Boolean "
                      "--reptype gz.msgs.Boolean "
                      "--timeout 2000 "
                      "--req 'data: " + std::string(pause ? "true" : "false") + "'";

    int ret = std::system(cmd.c_str());

    if (ret == 0) {
      response->success = true;
      response->message = pause ? "Physics paused" : "Physics resumed";
    } else {
      response->success = false;
      response->message = "Failed to toggle physics";
    }
  }

  void handleRandomizePoses(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    (void)request;
    RCLCPP_INFO(this->get_logger(), "🎲 Randomising poses via gz CLI...");

    // Use gz CLI to set model poses
    // Note: Full randomisation is handled in Python (DroneAttackEnv._randomize_episode)
    // This service is a placeholder for C++-side pose manipulation

    response->success = true;
    response->message = "Randomisation triggered (handled by Python RL env)";
  }

  // ── Members ─────────────────────────────────────────────────────────

  std::string world_name_;
  std::string drone_model_name_;
  std::string target_model_name_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_sim_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_world_srv_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr pause_physics_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr randomize_poses_srv_;
};

}  // namespace los_guidance

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<los_guidance::GazeboResetService>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}