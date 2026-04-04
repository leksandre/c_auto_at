#include "los_guidance/commander.hpp"
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <sstream>

namespace los_guidance
{

  Commander::Commander(int port, AttackCallback attack_cb, AutoModeCallback auto_cb,
                       GpsQueryCallback gps_cb, HomeCallback home_cb, rclcpp::Logger logger)
      : port_(port), attack_callback_(std::move(attack_cb)),
        auto_mode_callback_(std::move(auto_cb)), gps_query_callback_(std::move(gps_cb)),
        home_callback_(std::move(home_cb)), logger_(std::move(logger)) {}

  Commander::~Commander() { stop(); }

  void Commander::start()
  {
    if (running_)
      return;
    running_ = true;
    listen_thread_ = std::thread(&Commander::listenLoop, this);
    RCLCPP_INFO(logger_, "📡 Commander listening on TCP port %d", port_);
  }

  void Commander::stop()
  {
    if (!running_)
      return;
    running_ = false;
    if (server_fd_ != -1)
    {
      shutdown(server_fd_, SHUT_RDWR);
      close(server_fd_);
      server_fd_ = -1;
    }
    if (listen_thread_.joinable())
      listen_thread_.join();
    RCLCPP_INFO(logger_, "📡 Commander stopped");
  }

  void Commander::listenLoop()
  {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0)
    {
      RCLCPP_ERROR(logger_, "Failed to create socket");
      return;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address
    {
    };
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
      RCLCPP_ERROR(logger_, "Failed to bind to port %d", port_);
      close(server_fd_);
      return;
    }
    if (listen(server_fd_, 5) < 0)
    {
      RCLCPP_ERROR(logger_, "Failed to listen on port %d", port_);
      close(server_fd_);
      return;
    }

    while (running_)
    {
      fd_set readfds;
      FD_ZERO(&readfds);
      FD_SET(server_fd_, &readfds);
      struct timeval timeout
      {
      };
      timeout.tv_sec = 1;
      timeout.tv_usec = 0;
      if (select(server_fd_ + 1, &readfds, nullptr, nullptr, &timeout) > 0 && FD_ISSET(server_fd_, &readfds))
      {
        struct sockaddr_in client_addr
        {
        };
        socklen_t client_len = sizeof(client_addr);
        int new_socket = accept(server_fd_, (struct sockaddr *)&client_addr, &client_len);
        if (new_socket >= 0)
          std::thread(&Commander::handleConnection, this, new_socket).detach();
      }
    }
    close(server_fd_);
  }

  void Commander::handleConnection(int client_fd)
  {
    char buffer[512] = {0};
    int bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
    if (bytes_read > 0)
    {
      buffer[bytes_read] = '\0';
      processCommand(std::string(buffer), client_fd);
    }
    close(client_fd);
  }

  void Commander::sendResponse(int fd, const std::string &json)
  {
    // send(fd, json.c_str(), json.size(), MSG_NOSIGNAL);
    if (send(fd, json.c_str(), json.size(), MSG_NOSIGNAL) < 0)
    {
      RCLCPP_WARN(logger_, "⚠️ Failed to send response to client");
    }
  }

  void Commander::processCommand(const std::string &data, int client_fd)
  {
    // 🆕 1. Обработка showgps
    if (data.find("\"showgps\"") != std::string::npos)
    {
      double lat = 0, lon = 0, alt = 0;
      if (gps_query_callback_ && gps_query_callback_(lat, lon, alt))
      {
        std::ostringstream resp;
        resp << "{\"status\":\"success\",\"command\":\"SHOWGPS\",\"data\":{\"lat\":" << lat
             << ",\"lon\":" << lon << ",\"alt\":" << alt << "}}\n";
        sendResponse(client_fd, resp.str());
        RCLCPP_DEBUG(logger_, "📍 SHOWGPS response sent: %.6f, %.6f, %.2f", lat, lon, alt);
      }
      else
      {
        sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"SHOWGPS\",\"message\":\"GPS not available\"}\n");
        RCLCPP_WARN(logger_, "⚠️ SHOWGPS: GPS data not available");
      }
      return;
    }

    if (data.find("\"home\"") != std::string::npos)
    {
      RCLCPP_INFO(logger_, "🏠 Received command: home");
      if (home_callback_)
      {
        HomeData hd = home_callback_();
        if (hd.valid)
        {
          std::ostringstream resp;
          resp << "{\"status\":\"accepted\",\"command\":\"HOME\",\"data\":{\"home_lat\":" << hd.lat
               << ",\"home_lon\":" << hd.lon << ",\"home_alt\":" << hd.alt << "}}\n";
          sendResponse(client_fd, resp.str());
        }
        else
        {
          sendResponse(client_fd, "{\"status\":\"accepted\",\"command\":\"HOME\",\"data\":{\"home_lat\":0,\"home_lon\":0,\"home_alt\":0}}\n");
        }
      }
      return;
    }

    // 2. Обработка set_mode: auto
    if (data.find("\"set_mode\"") != std::string::npos && data.find("auto") != std::string::npos)
    {
      RCLCPP_INFO(logger_, "🤖 Received command: set_mode auto");
      if (auto_mode_callback_)
        auto_mode_callback_();
      sendResponse(client_fd, "{\"status\":\"success\",\"command\":\"SET_MODE\",\"mode\":\"auto\"}\n");
      return;
    }

    // 3. Обычная команда атаки
    bool attack_cmd = false;
    if (parseAttackCommand(data, attack_cmd))
    {
      RCLCPP_INFO(logger_, "🎯 Received attack command: %s", attack_cmd ? "START" : "STOP");
      if (attack_callback_)
        attack_callback_(attack_cmd);
      sendResponse(client_fd, attack_cmd ? "{\"status\":\"success\",\"command\":\"START_ATTACK\"}\n" : "{\"status\":\"success\",\"command\":\"STOP_ATTACK\"}\n");
    }
    else
    {
      RCLCPP_WARN(logger_, "⚠️ Invalid command format: %s", data.c_str());
      sendResponse(client_fd, "{\"status\":\"error\",\"message\":\"Unknown command\"}\n");
    }
  }

  bool Commander::parseAttackCommand(const std::string &data, bool &result)
  {
    size_t pos = data.find("start_attack");
    if (pos == std::string::npos)
      return false;
    size_t val_pos = data.find(':', pos);
    if (val_pos == std::string::npos)
      return false;
    std::string val = data.substr(val_pos + 1);
    val.erase(0, val.find_first_not_of(" \t\r\n"));
    if (val.substr(0, 4) == "true" || val.substr(0, 1) == "1")
    {
      result = true;
      return true;
    }
    if (val.substr(0, 5) == "false" || val.substr(0, 1) == "0")
    {
      result = false;
      return true;
    }
    return false;
  }

} // namespace los_guidance