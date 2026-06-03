#ifndef LOS_GUIDANCE_COMMANDER_HPP_
#define LOS_GUIDANCE_COMMANDER_HPP_

#include <thread>
#include <atomic>
#include <functional>
#include <string>
#include <rclcpp/rclcpp.hpp>

namespace los_guidance
{
  class Commander
  {
  public:
    using AttackCallback = std::function<void(bool)>;
    using AutoModeCallback = std::function<void()>;
    using GpsQueryCallback = std::function<bool(double &, double &, double &)>;
    using ArmCallback = std::function<void(bool)>; // 🆕 true=arm, false=disarm

    // 🆕 Домашняя точка
    struct HomeData
    {
      double lat{0}, lon{0}, alt{0};
      bool valid{false};
    };
    using HomeCallback = std::function<HomeData()>;

    // === Callback'и для новых команд ===
    using LandCallback = std::function<void(double)>;        // land speed (m/s)
    using SearchRoundCallback = std::function<void(double)>; // search rotation time (s for 360°)
    using AltitudeCallback = std::function<void(double)>;    // altitude change (m)
    using PatrolRoundCallback = std::function<void(double)>; // patrol diameter (m)
    using HoldSquareCallback = std::function<void(double)>;  // hold square size (m)
    using SearchModeCallback = std::function<void()>;        // 🆕 set_mode search
    using ManualControlCallback = std::function<void(float, float, float, float)>; // roll, pitch, yaw, throttle
    using WaypointModeCallback = std::function<void()>;      // 🆕 set_mode waypoint
    using WaypointDeleteCallback = std::function<void(int)>; // 🆕 waypoint_delete by index
    using WaypointAddCallback = std::function<void(int, const std::string&)>; // 🆕 waypoint_add at index
    using WaypointSkipCallback = std::function<void()>;      // 🆕 waypoint_skip
    using WaypointListCallback = std::function<std::string()>; // 🆕 waypoint_list — returns JSON
    using WaypointSpeedCallback = std::function<void(double)>;  // 🆕 waypoint_speed — set speed

    explicit Commander(int port, AttackCallback attack_cb, AutoModeCallback auto_cb,
                       GpsQueryCallback gps_cb, HomeCallback home_cb, rclcpp::Logger logger,
                       ArmCallback arm_cb = nullptr,                  // 🆕
                       LandCallback land_cb = nullptr,                // 🆕
                       SearchRoundCallback search_round_cb = nullptr, // 🆕
                       AltitudeCallback up_cb = nullptr,              // 🆕
                       AltitudeCallback down_cb = nullptr,            // 🆕
                       PatrolRoundCallback patrol_cb = nullptr,       // 🆕
                       HoldSquareCallback hold_cb = nullptr,          // 🆕
                       SearchModeCallback search_mode_cb = nullptr,   // 🆕
                       ManualControlCallback manual_cb = nullptr,     // 🆕
                       WaypointModeCallback waypoint_cb = nullptr,    // 🆕 set_mode waypoint
                       WaypointDeleteCallback waypoint_del_cb = nullptr,   // 🆕 waypoint_delete
                       WaypointAddCallback waypoint_add_cb = nullptr,      // 🆕 waypoint_add
                       WaypointSkipCallback waypoint_skip_cb = nullptr,   // 🆕 waypoint_skip
                       WaypointListCallback waypoint_list_cb = nullptr,  // 🆕 waypoint_list
                       WaypointSpeedCallback waypoint_speed_cb = nullptr); // 🆕 waypoint_speed

    ~Commander();

    void start();
    void stop();

  private:
    void listenLoop();
    void handleConnection(int client_fd);
    void processCommand(const std::string &data, int client_fd);
    bool parseAttackCommand(const std::string &data, bool &result);
    bool parseDoubleValue(const std::string &data, const std::string &key, double &result); // 🆕
    float parseFloat(const std::string &data, const std::string &key, float default_value); // 🆕
    void sendResponse(int fd, const std::string &json);

    int port_;
    // === Callback members ===
    AttackCallback attack_callback_;
    AutoModeCallback auto_mode_callback_;
    GpsQueryCallback gps_query_callback_;
    HomeCallback home_callback_;
    ArmCallback arm_callback_;
    LandCallback land_callback_;
    SearchRoundCallback search_callback_;     // для search_round
    SearchModeCallback search_mode_callback_; // для set_mode search 
    AltitudeCallback up_callback_;
    AltitudeCallback down_callback_;
    PatrolRoundCallback patrol_callback_;
    HoldSquareCallback hold_callback_;
    ManualControlCallback manual_control_callback_; // 🆕
    WaypointModeCallback waypoint_mode_callback_;   // 🆕 set_mode waypoint
    WaypointDeleteCallback waypoint_delete_callback_; // 🆕 waypoint_delete
    WaypointAddCallback waypoint_add_callback_;       // 🆕 waypoint_add
    WaypointSkipCallback waypoint_skip_callback_;     // 🆕 waypoint_skip
    WaypointListCallback waypoint_list_callback_;     // 🆕 waypoint_list
    WaypointSpeedCallback waypoint_speed_callback_;   // 🆕 waypoint_speed
    rclcpp::Logger logger_;
    std::thread listen_thread_;
    std::atomic<bool> running_{false};
    int server_fd_ = -1;
  };
} // namespace los_guidance
#endif