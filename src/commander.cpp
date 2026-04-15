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
                       GpsQueryCallback gps_cb, HomeCallback home_cb, rclcpp::Logger logger,
                       ArmCallback arm_cb, LandCallback land_cb, SearchRoundCallback search_cb,
                       AltitudeCallback up_cb, AltitudeCallback down_cb,
                       PatrolRoundCallback patrol_cb, HoldSquareCallback hold_cb)
      : port_(port), attack_callback_(std::move(attack_cb)),
        auto_mode_callback_(std::move(auto_cb)), gps_query_callback_(std::move(gps_cb)),
        home_callback_(std::move(home_cb)), arm_callback_(std::move(arm_cb)),
        land_callback_(std::move(land_cb)), search_callback_(std::move(search_cb)),
        up_callback_(std::move(up_cb)), down_callback_(std::move(down_cb)),
        patrol_callback_(std::move(patrol_cb)), hold_callback_(std::move(hold_cb)),
        logger_(std::move(logger)) {}

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

    // 🆕 2.1 Обработка arm/disarm
    if (data.find("\"arm\"") != std::string::npos)
    {
      bool arm = false;
      // Парсим: "arm": true или "arm": false
      size_t key_pos = data.find("\"arm\"");
      size_t colon_pos = data.find(':', key_pos);
      if (colon_pos != std::string::npos)
      {
        std::string val = data.substr(colon_pos + 1);
        val.erase(0, val.find_first_not_of(" \t\r\n"));
        if (val.substr(0, 4) == "true" || val.substr(0, 1) == "1")
          arm = true;
        else if (val.substr(0, 5) == "false" || val.substr(0, 1) == "0")
          arm = false;
        else {
          sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"ARM\",\"message\":\"Invalid value, use true/false\"}\n");
          return;
        }
      }
      else
      {
        sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"ARM\",\"message\":\"Invalid format\"}\n");
        return;
      }

      RCLCPP_INFO(logger_, "🔑 Received command: %s", arm ? "ARM" : "DISARM");
      if (arm_callback_)
        arm_callback_(arm);

      if (arm)
        sendResponse(client_fd, "{\"status\":\"success\",\"command\":\"ARM\"}\n");
      else
        sendResponse(client_fd, "{\"status\":\"success\",\"command\":\"DISARM\"}\n");
      return;
    }

    // 🆕 Отдельная команда disarm (для совместимости с "disarm": true)
    if (data.find("\"disarm\"") != std::string::npos)
    {
      RCLCPP_INFO(logger_, "🔓 Received command: disarm");
      if (arm_callback_)
        arm_callback_(false);  // disarm = false
      sendResponse(client_fd, "{\"status\":\"success\",\"command\":\"DISARM\"}\n");
      return;
    }

    // 🆕 Команда прерывания взлёта
    if (data.find("\"abort_takeoff\"") != std::string::npos)
    {
      RCLCPP_INFO(logger_, "🛑 Received command: abort_takeoff");
      if (arm_callback_)
        arm_callback_(false);  // disarm = прерывание взлёта
      sendResponse(client_fd, "{\"status\":\"success\",\"command\":\"ABORT_TAKEOFF\"}\n");
      return;
    }

    // 🆕 Команда приземления
    if (data.find("\"land\"") != std::string::npos)
    {
      double speed = 0.5;  // default 0.5 m/s
      if (parseDoubleValue(data, "land", speed))
      {
        RCLCPP_INFO(logger_, "🛬 Received command: land with speed %.2f m/s", speed);
        if (land_callback_)
          land_callback_(speed);
        std::ostringstream resp;
        resp << "{\"status\":\"success\",\"command\":\"LAND\",\"data\":{\"speed\":" << speed << "}}\n";
        sendResponse(client_fd, resp.str());
      }
      else
      {
        RCLCPP_WARN(logger_, "⚠️ Invalid land command format");
        sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"LAND\",\"message\":\"Invalid format, use {\\\"land\\\": N}\"}\n");
      }
      return;
    }

    // 🆕 Команда кругового поиска
    if (data.find("\"search_round\"") != std::string::npos)
    {
      double rotation_time = 10.0;  // default 10s for 360°
      if (parseDoubleValue(data, "search_round", rotation_time))
      {
        RCLCPP_INFO(logger_, "🔍 Received command: search_round with rotation time %.2f s", rotation_time);
        if (search_callback_)
          search_callback_(rotation_time);
        std::ostringstream resp;
        resp << "{\"status\":\"success\",\"command\":\"SEARCH_ROUND\",\"data\":{\"rotation_time\":" << rotation_time << "}}\n";
        sendResponse(client_fd, resp.str());
      }
      else
      {
        RCLCPP_WARN(logger_, "⚠️ Invalid search_round command format");
        sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"SEARCH_ROUND\",\"message\":\"Invalid format, use {\\\"search_round\\\": N}\"}\n");
      }
      return;
    }

    // 🆕 Команда подъёма
    if (data.find("\"up_to\"") != std::string::npos)
    {
      double altitude = 10.0;
      if (parseDoubleValue(data, "up_to", altitude))
      {
        RCLCPP_INFO(logger_, "⬆️ Received command: up_to %.2f m", altitude);
        if (up_callback_)
          up_callback_(altitude);
        std::ostringstream resp;
        resp << "{\"status\":\"success\",\"command\":\"UP_TO\",\"data\":{\"altitude\":" << altitude << "}}\n";
        sendResponse(client_fd, resp.str());
      }
      else
      {
        RCLCPP_WARN(logger_, "⚠️ Invalid up_to command format");
        sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"UP_TO\",\"message\":\"Invalid format, use {\\\"up_to\\\": N}\"}\n");
      }
      return;
    }

    // 🆕 Команда снижения
    if (data.find("\"down_to\"") != std::string::npos)
    {
      double altitude = 10.0;
      if (parseDoubleValue(data, "down_to", altitude))
      {
        RCLCPP_INFO(logger_, "⬇️ Received command: down_to %.2f m", altitude);
        if (down_callback_)
          down_callback_(altitude);
        std::ostringstream resp;
        resp << "{\"status\":\"success\",\"command\":\"DOWN_TO\",\"data\":{\"altitude\":" << altitude << "}}\n";
        sendResponse(client_fd, resp.str());
      }
      else
      {
        RCLCPP_WARN(logger_, "⚠️ Invalid down_to command format");
        sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"DOWN_TO\",\"message\":\"Invalid format, use {\\\"down_to\\\": N}\"}\n");
      }
      return;
    }

    // 🆕 Команда кругового патрулирования
    if (data.find("\"patrol_round\"") != std::string::npos)
    {
      double diameter = 50.0;
      if (parseDoubleValue(data, "patrol_round", diameter))
      {
        RCLCPP_INFO(logger_, "🔄 Received command: patrol_round with diameter %.2f m", diameter);
        if (patrol_callback_)
          patrol_callback_(diameter);
        std::ostringstream resp;
        resp << "{\"status\":\"success\",\"command\":\"PATROL_ROUND\",\"data\":{\"diameter\":" << diameter << "}}\n";
        sendResponse(client_fd, resp.str());
      }
      else
      {
        RCLCPP_WARN(logger_, "⚠️ Invalid patrol_round command format");
        sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"PATROL_ROUND\",\"message\":\"Invalid format, use {\\\"patrol_round\\\": N}\"}\n");
      }
      return;
    }

    // 🆕 Команда удержания позиции в квадрате
    if (data.find("\"hold_to_square\"") != std::string::npos)
    {
      double square_size = 10.0;
      if (parseDoubleValue(data, "hold_to_square", square_size))
      {
        RCLCPP_INFO(logger_, "⏸️ Received command: hold_to_square with size %.2f m", square_size);
        if (hold_callback_)
          hold_callback_(square_size);
        std::ostringstream resp;
        resp << "{\"status\":\"success\",\"command\":\"HOLD_TO_SQUARE\",\"data\":{\"size\":" << square_size << "}}\n";
        sendResponse(client_fd, resp.str());
      }
      else
      {
        RCLCPP_WARN(logger_, "⚠️ Invalid hold_to_square command format");
        sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"HOLD_TO_SQUARE\",\"message\":\"Invalid format, use {\\\"hold_to_square\\\": N}\"}\n");
      }
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

  // 🆕 Универсальный метод парсинга double значения по ключу
  bool Commander::parseDoubleValue(const std::string &data, const std::string &key, double &result)
  {
    size_t key_pos = data.find("\"" + key + "\"");
    if (key_pos == std::string::npos)
      return false;
    
    size_t colon_pos = data.find(':', key_pos);
    if (colon_pos == std::string::npos)
      return false;
    
    // Извлекаем значение после ':'
    std::string val = data.substr(colon_pos + 1);
    val.erase(0, val.find_first_not_of(" \t\r\n"));
    
    // Удаляем запятую или закрывающую скобку если есть
    size_t end_pos = val.find_first_of(",} \t\r\n");
    if (end_pos != std::string::npos)
      val = val.substr(0, end_pos);
    
    try
    {
      result = std::stod(val);
      return true;
    }
    catch (...)
    {
      return false;
    }
  }

} // namespace los_guidance