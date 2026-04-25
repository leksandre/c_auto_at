#ifndef LOS_GUIDANCE_GPS_PARSER_HPP_
#define LOS_GUIDANCE_GPS_PARSER_HPP_

#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <cstdint>

namespace los_guidance {

struct GpsData {
    double latitude{0.0};
    double longitude{0.0};
    double altitude{0.0};
    int satellites{0};
    bool is_valid{false};
    uint64_t timestamp_us{0}; // Unix timestamp in microseconds
};

class GpsParser {
public:
    GpsParser();
    ~GpsParser();

    /**
     * @brief Запуск потока чтения GPS
     * @param port Путь к порту (например, "/dev/ttyUSB0")
     * @param baudrate Скорость передачи (например, 9600, 115200)
     */
    void start(const std::string& port, int baudrate);

    /**
     * @brief Остановка потока
     */
    void stop();

    /**
     * @brief Получение последних валидных данных GPS
     * @return Структура с данными
     */
    GpsData getData() const;

private:
    void runLoop();
    bool parseNmeaLine(const std::string& line);
    
    std::thread gps_thread_;
    std::atomic<bool> running_{false};
    std::mutex data_mutex_;
    GpsData current_data_;
    
    // Для работы с файлом порта
    int fd_port_ = -1;
};

} // namespace los_guidance

#endif // LOS_GUIDANCE_GPS_PARSER_HPP_