#include "los_guidance/guidance_config.hpp"

namespace los_guidance {

//void GuidanceConfig::loadFromNode(const rclcpp::Node::SharedPtr& node)
void GuidanceConfig::loadFromNode(const rclcpp::Node::SharedPtr& node)
{
  // === Базовые параметры управления ===
  node->declare_parameter("max_speed", max_speed_);
  node->get_parameter("max_speed", max_speed_);

  node->declare_parameter("max_vert_speed", max_vert_speed_);
  node->get_parameter("max_vert_speed", max_vert_speed_);

  node->declare_parameter("max_yaw_rate", max_yaw_rate_);
  node->get_parameter("max_yaw_rate", max_yaw_rate_);

  // === Фильтрация ручного управления ===
  node->declare_parameter("manual_roll_deadzone", manual_roll_deadzone);
  node->get_parameter("manual_roll_deadzone", manual_roll_deadzone);

  node->declare_parameter("manual_pitch_deadzone", manual_pitch_deadzone);
  node->get_parameter("manual_pitch_deadzone", manual_pitch_deadzone);

  node->declare_parameter("manual_yaw_deadzone", manual_yaw_deadzone);
  node->get_parameter("manual_yaw_deadzone", manual_yaw_deadzone);

  node->declare_parameter("manual_throttle_deadzone", manual_throttle_deadzone);
  node->get_parameter("manual_throttle_deadzone", manual_throttle_deadzone);

  // === Ограничения скорости изменения (slew rate limiting) ===
  node->declare_parameter("max_roll_slew_rate", max_roll_slew_rate);
  node->get_parameter("max_roll_slew_rate", max_roll_slew_rate);

  node->declare_parameter("max_pitch_slew_rate", max_pitch_slew_rate);
  node->get_parameter("max_pitch_slew_rate", max_pitch_slew_rate);

  node->declare_parameter("max_yaw_slew_rate", max_yaw_slew_rate);
  node->get_parameter("max_yaw_slew_rate", max_yaw_slew_rate);

  node->declare_parameter("max_throttle_slew_rate", max_throttle_slew_rate);
  node->get_parameter("max_throttle_slew_rate", max_throttle_slew_rate);

  // === Фильтр низких частот ===
  node->declare_parameter("manual_filter_time_constant", manual_filter_time_constant);
  node->get_parameter("manual_filter_time_constant", manual_filter_time_constant);

  // === Аварийные блокировки ===
  node->declare_parameter("enable_emergency_lockout", enable_emergency_lockout);
  node->get_parameter("enable_emergency_lockout", enable_emergency_lockout);

  node->declare_parameter("throttle_cutoff_threshold", throttle_cutoff_threshold);
  node->get_parameter("throttle_cutoff_threshold", throttle_cutoff_threshold);

  node->declare_parameter("throttle_spikes_threshold", throttle_spikes_threshold);
  node->get_parameter("throttle_spikes_threshold", throttle_spikes_threshold);

  node->declare_parameter("spike_filter_time_window", spike_filter_time_window);
  node->get_parameter("spike_filter_time_window", spike_filter_time_window);

  // === Режимы плавности ===
  node->declare_parameter("enable_ship_mode", enable_ship_mode);
  node->get_parameter("enable_ship_mode", enable_ship_mode);

  node->declare_parameter("ship_mode_response_factor", ship_mode_response_factor);
  node->get_parameter("ship_mode_response_factor", ship_mode_response_factor);

  // === Режимы переключения ===
  node->declare_parameter("allow_vision_attack", allow_vision_attack);
  node->get_parameter("allow_vision_attack", allow_vision_attack);

  node->declare_parameter("allow_autoland", allow_autoland);
  node->get_parameter("allow_autoland", allow_autoland);

  RCLCPP_INFO(node->get_logger(), "GuidanceConfig loaded successfully");
}

} // namespace los_guidance