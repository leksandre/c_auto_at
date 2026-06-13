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
    ArmCallback arm_cb, LandCallback land_cb, SearchRoundCallback search_round_cb,
    AltitudeCallback up_cb, AltitudeCallback down_cb,
    PatrolRoundCallback patrol_cb, HoldSquareCallback hold_cb,
    SearchModeCallback search_mode_cb, ManualControlCallback manual_cb,
    WaypointModeCallback waypoint_cb,
    WaypointDeleteCallback waypoint_del_cb,
    WaypointAddCallback waypoint_add_cb,
    WaypointSkipCallback waypoint_skip_cb,
    WaypointListCallback waypoint_list_cb,
    WaypointSpeedCallback waypoint_speed_cb,
    BamperModeCallback bamper_cb,
    SelfDestructModeCallback self_destruct_cb,
    HeartbeatCallback heartbeat_cb)
    : port_(port), attack_callback_(std::move(attack_cb)),
      auto_mode_callback_(std::move(auto_cb)), gps_query_callback_(std::move(gps_cb)),
      home_callback_(std::move(home_cb)), arm_callback_(std::move(arm_cb)),
      land_callback_(std::move(land_cb)), search_callback_(std::move(search_round_cb)),
      search_mode_callback_(std::move(search_mode_cb)),
      up_callback_(std::move(up_cb)), down_callback_(std::move(down_cb)),
      patrol_callback_(std::move(patrol_cb)), hold_callback_(std::move(hold_cb)),
      manual_control_callback_(std::move(manual_cb)),
      waypoint_mode_callback_(std::move(waypoint_cb)),
      waypoint_delete_callback_(std::move(waypoint_del_cb)),
      waypoint_add_callback_(std::move(waypoint_add_cb)),
      waypoint_skip_callback_(std::move(waypoint_skip_cb)),
      waypoint_list_callback_(std::move(waypoint_list_cb)),
      waypoint_speed_callback_(std::move(waypoint_speed_cb)),
      bamper_callback_(std::move(bamper_cb)),
      self_destruct_callback_(std::move(self_destruct_cb)),
      heartbeat_callback_(std::move(heartbeat_cb)),
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
    } else{
      RCLCPP_ERROR(logger_, "🔌 Manual control client disconnected");
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
    // 🆕 Обработка manual_control
    if (data.find("\"manual_control\"") != std::string::npos)
    {
      float roll = parseFloat(data, "roll", 0.0f);
      float pitch = parseFloat(data, "pitch", 0.0f);
      float yaw = parseFloat(data, "yaw", 0.0f);
      float throttle = parseFloat(data, "throttle", 0.5f);
      
      // Применяем deadzone и нормализацию
      const float DEADZONE = 0.05f;
      auto applyDeadzone = [](float v, float dz) -> float {
        if (std::abs(v) < dz) return 0.0f;
        return (v > 0 ? (v - dz) : (v + dz)) / (1.0f - dz);
      };
      
      if (manual_control_callback_) {
        manual_control_callback_(
          applyDeadzone(roll, DEADZONE),
          applyDeadzone(pitch, DEADZONE),
          applyDeadzone(yaw, DEADZONE),
          std::clamp(throttle, 0.0f, 1.0f)  // throttle без deadzone
        );
      }
      sendResponse(client_fd, "{\"status\":\"ok\",\"command\":\"manual_control\"}\n");
      return;
    }
    
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


        // 🆕 2.1 Обработка set_mode: search / searshА
    if (data.find("\"set_mode\"") != std::string::npos &&
        (data.find("search") != std::string::npos || data.find("searsh") != std::string::npos))
    {
        RCLCPP_INFO(logger_, "🔍 Received command: set_mode search");
        if (search_mode_callback_) search_mode_callback_();
        sendResponse(client_fd, "{\"status\":\"success\",\"command\":\"SET_MODE\",\"mode\":\"search\"}\n");
        return;
    }

    // 🆕 2.2 Обработка set_mode: waypoint
    if (data.find("\"set_mode\"") != std::string::npos &&
        data.find("waypoint") != std::string::npos)
    {
        RCLCPP_INFO(logger_, "🗺️ Received command: set_mode waypoint");
        if (waypoint_mode_callback_) waypoint_mode_callback_();
        sendResponse(client_fd, "{\"status\":\"success\",\"command\":\"SET_MODE\",\"mode\":\"waypoint\"}\n");
        return;
    }

    // 🆕 2.3 Обработка waypoint_delete
    if (data.find("\"waypoint_delete\"") != std::string::npos)
    {
        int index = -1;
        size_t key_pos = data.find("\"waypoint_delete\"");
        size_t colon_pos = data.find(':', key_pos);
        if (colon_pos != std::string::npos)
        {
            std::string val = data.substr(colon_pos + 1);
            val.erase(0, val.find_first_not_of(" \t\r\n"));
            size_t end_pos = val.find_first_of(",} \t\r\n");
            if (end_pos != std::string::npos)
                val = val.substr(0, end_pos);
            try { index = std::stoi(val); }
            catch (...) { index = -1; }
        }

        if (index >= 0)
        {
            RCLCPP_INFO(logger_, "🗑️ Received command: waypoint_delete index=%d", index);
            if (waypoint_delete_callback_) waypoint_delete_callback_(index);
            std::ostringstream resp;
            resp << "{\"status\":\"success\",\"command\":\"WAYPOINT_DELETE\",\"data\":{\"index\":" << index << "}}\n";
            sendResponse(client_fd, resp.str());
        }
        else
        {
            sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"WAYPOINT_DELETE\",\"message\":\"Invalid index\"}\n");
        }
        return;
    }

    // 🆕 2.4 Обработка waypoint_add (формат: {"waypoint_add": {"index": N, "data": "lon,lat,action,param"}})
    if (data.find("\"waypoint_add\"") != std::string::npos)
    {
        int add_index = -1;
        std::string waypoint_data;

        // Парсим index
        size_t idx_key = data.find("\"index\"");
        if (idx_key != std::string::npos)
        {
            size_t colon_pos = data.find(':', idx_key);
            if (colon_pos != std::string::npos)
            {
                std::string val = data.substr(colon_pos + 1);
                val.erase(0, val.find_first_not_of(" \t\r\n"));
                size_t end_pos = val.find_first_of(",} \t\r\n");
                if (end_pos != std::string::npos)
                    val = val.substr(0, end_pos);
                try { add_index = std::stoi(val); }
                catch (...) { add_index = -1; }
            }
        }

        // Парсим data (строка "lon,lat,action,param")
        size_t data_key = data.find("\"data\"");
        if (data_key != std::string::npos)
        {
            size_t colon_pos = data.find(':', data_key);
            if (colon_pos != std::string::npos)
            {
                std::string val = data.substr(colon_pos + 1);
                val.erase(0, val.find_first_not_of(" \t\r\n\""));
                size_t end_pos = val.find_first_of("\"}");
                if (end_pos != std::string::npos)
                    waypoint_data = val.substr(0, end_pos);
                else
                    waypoint_data = val;
                // Удаляем trailing кавычку если есть
                if (!waypoint_data.empty() && waypoint_data.back() == '\"')
                    waypoint_data.pop_back();
            }
        }

        if (add_index >= 0 && !waypoint_data.empty())
        {
            RCLCPP_INFO(logger_, "➕ Received command: waypoint_add index=%d data=%s", add_index, waypoint_data.c_str());
            if (waypoint_add_callback_) waypoint_add_callback_(add_index, waypoint_data);
            std::ostringstream resp;
            resp << "{\"status\":\"success\",\"command\":\"WAYPOINT_ADD\",\"data\":{\"index\":" << add_index << ",\"waypoint\":\"" << waypoint_data << "\"}}\n";
            sendResponse(client_fd, resp.str());
        }
        else
        {
            sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"WAYPOINT_ADD\",\"message\":\"Invalid index or data\"}\n");
        }
        return;
    }

    // 🆕 2.5 Обработка waypoint_skip
    if (data.find("\"waypoint_skip\"") != std::string::npos)
    {
        RCLCPP_INFO(logger_, "⏭️ Received command: waypoint_skip");
        if (waypoint_skip_callback_) waypoint_skip_callback_();
        sendResponse(client_fd, "{\"status\":\"success\",\"command\":\"WAYPOINT_SKIP\"}\n");
        return;
    }

    // 🆕 2.6 Обработка waypoint_list
    if (data.find("\"waypoint_list\"") != std::string::npos)
    {
        RCLCPP_INFO(logger_, "📋 Received command: waypoint_list");
        if (waypoint_list_callback_)
        {
            std::string json = waypoint_list_callback_();
            sendResponse(client_fd, json + "\n");
        }
        else
        {
            sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"WAYPOINT_LIST\",\"message\":\"No callback registered\"}\n");
        }
        return;
    }

    // 🆕 2.7 Обработка waypoint_speed
    if (data.find("\"waypoint_speed\"") != std::string::npos)
    {
        double speed = 7.0; // значение по умолчанию
        size_t key_pos = data.find("\"waypoint_speed\"");
        size_t colon_pos = data.find(':', key_pos);
        if (colon_pos != std::string::npos)
        {
            std::string val = data.substr(colon_pos + 1);
            val.erase(0, val.find_first_not_of(" \t\r\n"));
            size_t end_pos = val.find_first_of(",} \t\r\n");
            if (end_pos != std::string::npos)
                val = val.substr(0, end_pos);
            try { speed = std::stod(val); }
            catch (...) { speed = -1.0; }
        }

        if (speed > 0.0)
        {
            RCLCPP_INFO(logger_, "🚀 Received command: waypoint_speed speed=%.1f m/s", speed);
            if (waypoint_speed_callback_) waypoint_speed_callback_(speed);
            std::ostringstream resp;
            resp << "{\"status\":\"success\",\"command\":\"WAYPOINT_SPEED\",\"data\":{\"speed\":" << speed << "}}\n";
            sendResponse(client_fd, resp.str());
        }
        else
        {
            sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"WAYPOINT_SPEED\",\"message\":\"Invalid speed (must be > 0)\"}\n");
        }
        return;
    }

    // 🆕 2.8 Обработка arm/disarm
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




    // 🆕 Команда ping/heartbeat
    if (data.find("\"ping\"") != std::string::npos || data.find("\"heartbeat\"") != std::string::npos)
    {
      if (heartbeat_callback_)
        heartbeat_callback_();
      sendResponse(client_fd, "{\"status\":\"success\",\"command\":\"PONG\"}\n");
      return;
    }

    // 🆕 Команда set_bamper
    if (data.find("\"set_bamper\"") != std::string::npos)
    {
      bool bamper_val = false;
      size_t key_pos = data.find("\"set_bamper\"");
      size_t colon_pos = data.find(':', key_pos);
      if (colon_pos != std::string::npos)
      {
        std::string val = data.substr(colon_pos + 1);
        val.erase(0, val.find_first_not_of(" \t\r\n"));
        if (val.substr(0, 4) == "true" || val.substr(0, 1) == "1")
          bamper_val = true;
        else if (val.substr(0, 5) == "false" || val.substr(0, 1) == "0")
          bamper_val = false;
        else {
          sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"SET_BAMPER\",\"message\":\"Invalid value, use true/false\"}\n");
          return;
        }
      }

      RCLCPP_INFO(logger_, "💣 Received command: set_bamper = %s", bamper_val ? "true" : "false");
      if (bamper_callback_)
        bamper_callback_(bamper_val);
      sendResponse(client_fd, bamper_val ? "{\"status\":\"success\",\"command\":\"SET_BAMPER\",\"data\":{\"bamper\":true}}\n"
                                         : "{\"status\":\"success\",\"command\":\"SET_BAMPER\",\"data\":{\"bamper\":false}}\n");
      return;
    }

    // 🆕 Команда set_self_destruct_mode
    if (data.find("\"set_self_destruct_mode\"") != std::string::npos)
    {
      bool sd_val = false;
      size_t key_pos = data.find("\"set_self_destruct_mode\"");
      size_t colon_pos = data.find(':', key_pos);
      if (colon_pos != std::string::npos)
      {
        std::string val = data.substr(colon_pos + 1);
        val.erase(0, val.find_first_not_of(" \t\r\n"));
        if (val.substr(0, 4) == "true" || val.substr(0, 1) == "1")
          sd_val = true;
        else if (val.substr(0, 5) == "false" || val.substr(0, 1) == "0")
          sd_val = false;
        else {
          sendResponse(client_fd, "{\"status\":\"error\",\"command\":\"SET_SELF_DESTRUCT_MODE\",\"message\":\"Invalid value, use true/false\"}\n");
          return;
        }
      }

      RCLCPP_INFO(logger_, "☠️ Received command: set_self_destruct_mode = %s", sd_val ? "ARMED" : "DISARMED");
      if (self_destruct_callback_)
        self_destruct_callback_(sd_val);
      sendResponse(client_fd, sd_val ? "{\"status\":\"success\",\"command\":\"SET_SELF_DESTRUCT_MODE\",\"data\":{\"armed\":true}}\n"
                                     : "{\"status\":\"success\",\"command\":\"SET_SELF_DESTRUCT_MODE\",\"data\":{\"armed\":false}}\n");
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

  // 🆕 Универсальный метод парсинга float значения по ключу
  float Commander::parseFloat(const std::string &data, const std::string &key, float default_value)
  {
    double result;
    if (parseDoubleValue(data, key, result))
      return static_cast<float>(result);
    return default_value;
  }

} // namespace los_guidance