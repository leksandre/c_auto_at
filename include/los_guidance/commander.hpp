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

    explicit Commander(int port, AttackCallback attack_cb, AutoModeCallback auto_cb, rclcpp::Logger logger);
    ~Commander();

    void start();
    void stop();

  private:
    void listenLoop();
    void handleConnection(int client_fd);
    void processCommand(const std::string &data);
    bool parseAttackCommand(const std::string &data, bool &result);

    int port_;
    AttackCallback attack_callback_;
    AutoModeCallback auto_mode_callback_; // 🆕 Новый колбэк
    rclcpp::Logger logger_;
    std::thread listen_thread_;
    std::atomic<bool> running_{false};
    int server_fd_ = -1;
  };

} // namespace los_guidance
#endif // LOS_GUIDANCE_COMMANDER_HPP_