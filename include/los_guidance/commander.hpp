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
  
  // 🆕 Домашняя точка
  struct HomeData { double lat{0}, lon{0}, alt{0}; bool valid{false}; };
  using HomeCallback = std::function<HomeData()>;

  explicit Commander(int port, AttackCallback attack_cb, AutoModeCallback auto_cb,
                     GpsQueryCallback gps_cb, HomeCallback home_cb, rclcpp::Logger logger);
  ~Commander();

  void start();
  void stop();

private:
  void listenLoop();
  void handleConnection(int client_fd);
  void processCommand(const std::string &data, int client_fd);
  bool parseAttackCommand(const std::string &data, bool &result);
  void sendResponse(int fd, const std::string &json);

  int port_;
  AttackCallback attack_callback_;
  AutoModeCallback auto_mode_callback_;
  GpsQueryCallback gps_query_callback_;
  HomeCallback home_callback_; // 🆕
  rclcpp::Logger logger_;
  std::thread listen_thread_;
  std::atomic<bool> running_{false};
  int server_fd_ = -1;
};
} // namespace los_guidance
#endif