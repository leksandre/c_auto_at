#include "los_guidance/gps_parser.hpp"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath> // <-- ВАЖНО: Добавлено для std::floor

namespace los_guidance {

GpsParser::GpsParser() {}

GpsParser::~GpsParser() {
    stop();
    if (fd_port_ >= 0) {
        close(fd_port_);
    }
}

void GpsParser::start(const std::string& port, int baudrate) {
    if (running_) return;

    // Открытие порта
    fd_port_ = open(port.c_str(), O_RDONLY | O_NOCTTY);
    if (fd_port_ < 0) {
        std::cerr << "[GPS] Failed to open port: " << port << std::endl;
        return;
    }

    // Настройка терминала
    struct termios tty;
    if (tcgetattr(fd_port_, &tty) != 0) {
        std::cerr << "[GPS] Error from tcgetattr" << std::endl;
        close(fd_port_);
        return;
    }

    cfsetospeed(&tty, static_cast<speed_t>(baudrate));
    cfsetispeed(&tty, static_cast<speed_t>(baudrate));
    
    tty.c_cflag &= ~PARENB;      // No parity
    tty.c_cflag &= ~CSTOPB;      // 1 stop bit
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;          // 8 bits
    tty.c_cflag &= ~CRTSCTS;     // No hardware flow control
    tty.c_cflag |= CREAD | CLOCAL; // Enable receiver
    
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // Raw input
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);         // Disable software flow control
    tty.c_iflag &= ~(INLCR | ICRNL | IGNCR);        // Handle newlines simply
    tty.c_oflag &= ~OPOST;           // Raw output
    tty.c_cc[VMIN] = 0;              // Non-blocking read
    tty.c_cc[VTIME] = 1;             // 100ms timeout

    if (tcsetattr(fd_port_, TCSANOW, &tty) != 0) {
        std::cerr << "[GPS] Error from tcsetattr" << std::endl;
        close(fd_port_);
        return;
    }

    running_.store(true);
    gps_thread_ = std::thread(&GpsParser::runLoop, this);
    std::cout << "[GPS] Started reading from " << port << " at " << baudrate << " baud" << std::endl;
}

void GpsParser::stop() {
    running_.store(false);
    if (gps_thread_.joinable()) {
        gps_thread_.join();
    }
}

void GpsParser::runLoop() {
    char buffer[256];
    std::string line_buffer;

    while (running_.load()) {
        ssize_t bytes_read = read(fd_port_, buffer, sizeof(buffer) - 1);
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            // Исправление предупреждения: приводим bytes_read к size_t, так как он гарантированно > 0 здесь
            for (size_t i = 0; i < static_cast<size_t>(bytes_read); ++i) {
                if (buffer[i] == '\n') {
                    if (!line_buffer.empty()) {
                        parseNmeaLine(line_buffer);
                        line_buffer.clear();
                    }
                } else {
                    line_buffer += buffer[i];
                }
            }
        } else if (bytes_read < 0 && errno != EAGAIN) {
            std::cerr << "[GPS] Read error: " << strerror(errno) << std::endl;
            break;
        }
    }
}

bool GpsParser::parseNmeaLine(const std::string& line) {
    // Простой парсер для $GPGGA или $GNGGA
    if (line.find("$GPGGA") == std::string::npos && 
        line.find("$GNGGA") == std::string::npos) {
        return false;
    }

    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    
    while (std::getline(iss, token, ',')) {
        tokens.push_back(token);
    }

    // Формат GPGGA:
    // 0: $GPGGA
    // 1: Время UTC
    // 2: Широта (ddmm.mmmm)
    // 3: N/S
    // 4: Долгота (dddmm.mmmm)
    // 5: E/W
    // 6: Качество фиксации (1=GPS fix, 2=DGPS)
    // 7: Кол-во спутников
    // 8: HDOP
    // 9: Высота над эллипсоидом
    // 10: m
    // 11: Высота геоида
    // 12: m

    if (tokens.size() < 10) return false;

    try {
        // Проверка качества фиксации
        int fix_quality = std::stoi(tokens[6]);
        if (fix_quality == 0) return false;

        // Парсинг широты
        double lat_deg = std::stod(tokens[2]);
        double lat_int = std::floor(lat_deg / 100); // Теперь работает благодаря #include <cmath>
        double lat_frac = lat_deg - (lat_int * 100);
        double latitude = lat_int + (lat_frac / 60.0);
        if (tokens[3] == "S") latitude = -latitude;

        // Парсинг долготы
        double lon_deg = std::stod(tokens[4]);
        double lon_int = std::floor(lon_deg / 100); // Теперь работает благодаря #include <cmath>
        double lon_frac = lon_deg - (lon_int * 100);
        double longitude = lon_int + (lon_frac / 60.0);
        if (tokens[5] == "W") longitude = -longitude;

        // Парсинг высоты
        double altitude = std::stod(tokens[9]);

        // Парсинг спутников
        int satellites = std::stoi(tokens[7]);

        // Обновление данных
        std::lock_guard<std::mutex> lock(data_mutex_);
        current_data_.latitude = latitude;
        current_data_.longitude = longitude;
        current_data_.altitude = altitude;
        current_data_.satellites = satellites;
        current_data_.is_valid = true;
        
        // Текущее время системы
        auto now = std::chrono::system_clock::now();
        current_data_.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

    } catch (const std::exception& e) {
        // Игнорируем ошибки парсинга отдельных строк
    }

    return true;
}

GpsData GpsParser::getData() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex_));
    return current_data_;
}

} // namespace los_guidance