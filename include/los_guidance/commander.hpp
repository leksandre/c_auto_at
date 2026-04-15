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
  using GpsQueryCallback = std::function<bool(double&, double&, double&)>;
  using ArmCallback = std::function<void(bool)>;  // 🆕 true=arm, false=disarm

  // 🆕 Домашняя точка
  struct HomeData { double lat{0}, lon{0}, alt{0}; bool valid{false}; };
  using HomeCallback = std::function<HomeData()>;

  // 🆕 Callback'и для новых команд
  using LandCallback = std::function<void(double)>;              // land speed (m/s)
  using SearchRoundCallback = std::function<void(double)>;       // search rotation time (s for 360°)
  using AltitudeCallback = std::function<void(double)>;          // altitude change (m)
  using PatrolRoundCallback = std::function<void(double)>;       // patrol diameter (m)
  using HoldSquareCallback = std::function<void(double)>;        // hold square size (m)

  explicit Commander(int port, AttackCallback attack_cb, AutoModeCallback auto_cb,
                     GpsQueryCallback gps_cb, HomeCallback home_cb, rclcpp::Logger logger,
                     ArmCallback arm_cb = nullptr,                            // 🆕
                     LandCallback land_cb = nullptr,                          // 🆕
                     SearchRoundCallback search_cb = nullptr,                 // 🆕
                     AltitudeCallback up_cb = nullptr,                        // 🆕
                     AltitudeCallback down_cb = nullptr,                      // 🆕
                     PatrolRoundCallback patrol_cb = nullptr,                 // 🆕
                     HoldSquareCallback hold_cb = nullptr);                   // 🆕
  ~Commander();

  void start();
  void stop();

private:
  void listenLoop();
  void handleConnection(int client_fd);
  void processCommand(const std::string &data, int client_fd);
  bool parseAttackCommand(const std::string &data, bool &result);
  bool parseDoubleValue(const std::string &data, const std::string &key, double &result);  // 🆕
  void sendResponse(int fd, const std::string &json);

  int port_;
  AttackCallback attack_callback_;
  AutoModeCallback auto_mode_callback_;
  GpsQueryCallback gps_query_callback_;
  HomeCallback home_callback_; // 🆕
  ArmCallback arm_callback_;   // 🆕
  LandCallback land_callback_;         // 🆕
  SearchRoundCallback search_callback_; // 🆕
  AltitudeCallback up_callback_;       // 🆕
  AltitudeCallback down_callback_;     // 🆕
  PatrolRoundCallback patrol_callback_; // 🆕
  HoldSquareCallback hold_callback_;   // 🆕
  rclcpp::Logger logger_;
  std::thread listen_thread_;
  std::atomic<bool> running_{false};
  int server_fd_ = -1;
};
} // namespace los_guidance
#endif