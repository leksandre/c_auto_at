/**
 * @file synthetic_data_generator.hpp
 * @brief Генератор синтетических данных для обучения MLP-модели наведения
 *
 * Генерирует пары (входы → выходы) путём случайного перебора входных параметров
 * и вызова эталонного алгоритма GuidanceController::computeAcceleration().
 *
 * Входы (9 признаков):
 *   - u_los (3): единичный вектор LOS в навигационной СК
 *   - lambda (3): угловая скорость LOS (рад/с)
 *   - pitch, yaw, roll (3): углы дрона (рад)
 *
 * Выходы (3 значения):
 *   - acc_x, acc_y, acc_z: команда ускорения (м/с²)
 *
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#ifndef LOS_GUIDANCE_SYNTHETIC_DATA_GENERATOR_HPP_
#define LOS_GUIDANCE_SYNTHETIC_DATA_GENERATOR_HPP_

#pragma once

#include <string>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cmath>

#include <Eigen/Dense>
#include "los_guidance/guidance_controller.hpp"

namespace los_guidance
{

/**
 * @struct GeneratorConfig
 * @brief Конфигурация генератора синтетических данных
 */
struct GeneratorConfig
{
    // Диапазоны для lambda (угловая скорость LOS, рад/с)
    double lambda_min{-5.0};
    double lambda_max{5.0};

    // Диапазоны для углов дрона (радианы)
    double pitch_min{-1.0};    // ~[-57°, +57°]
    double pitch_max{1.0};
    double yaw_min{-M_PI};     // полный оборот
    double yaw_max{M_PI};
    double roll_min{-0.5};     // ~[-28°, +28°]
    double roll_max{0.5};

    // Количество сэмплов
    int num_samples{100000};

    // Seed для воспроизводимости (0 = случайный)
    uint32_t seed{0};
};

/**
 * @class SyntheticDataGenerator
 * @brief Генератор синтетических данных для обучения MLP-модели наведения
 */
class SyntheticDataGenerator
{
public:
    /**
     * @brief Конструктор
     * @param config Конфигурация диапазонов
     */
    explicit SyntheticDataGenerator(const GeneratorConfig& config = GeneratorConfig());

    /**
     * @brief Генерация датасета
     * @param guidance_ctrl Указатель на эталонный контроллер
     * @param output_path Путь к выходному CSV файлу
     * @return true если успешно
     */
    bool generateDataset(GuidanceController* guidance_ctrl,
                         const std::string& output_path);

    /**
     * @brief Генерация датасета с прогресс-колбэком
     * @param guidance_ctrl Указатель на эталонный контроллер
     * @param output_path Путь к выходному CSV файлу
     * @param progress_callback Функция обратного вызова (current, total)
     * @return true если успешно
     */
    bool generateDatasetWithProgress(
        GuidanceController* guidance_ctrl,
        const std::string& output_path,
        std::function<void(int, int)> progress_callback = nullptr);

    /**
     * @brief Получить статистику последней генерации
     */
    struct Stats {
        int total_samples{0};
        int valid_samples{0};
        int skipped_samples{0};  // NaN/Inf в выходе
        double duration_sec{0.0};
    };

    Stats getLastStats() const { return last_stats_; }

private:
    GeneratorConfig config_;
    std::mt19937 rng_;
    Stats last_stats_;

    // Генерация случайного единичного вектора (равномерно на сфере)
    Eigen::Vector3d randomUnitVector();

    // Генерация случайного значения в диапазоне
    double randomInRange(double min, double max);

    // Проверка валидности выхода (нет NaN/Inf)
    bool isValidOutput(const GuidanceOutput& output);

    // Запись заголовка CSV
    void writeHeader(std::ofstream& file);

    // Запись строки данных
    void writeRow(std::ofstream& file,
                  const Eigen::Vector3d& u_los,
                  const Eigen::Vector3d& lambda,
                  double pitch, double yaw, double roll,
                  const GuidanceOutput& output);
};

} // namespace los_guidance

#endif // LOS_GUIDANCE_SYNTHETIC_DATA_GENERATOR_HPP_