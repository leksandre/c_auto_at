/**
 * @file synthetic_data_generator.cpp
 * @brief Реализация генератора синтетических данных для обучения MLP
 *
 * @author Snow Pang, ssnowpang@gmail.com
 * @date 2025-09
 * @copyright BSD 3-Clause License
 */

#include "los_guidance/synthetic_data_generator.hpp"
#include <cmath>
#include <stdexcept>

namespace los_guidance
{

SyntheticDataGenerator::SyntheticDataGenerator(const GeneratorConfig& config)
    : config_(config)
{
    // Инициализация генератора случайных чисел
    if (config_.seed == 0) {
        rng_.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    } else {
        rng_.seed(config_.seed);
    }
}

Eigen::Vector3d SyntheticDataGenerator::randomUnitVector()
{
    // Равномерное распределение на единичной сфере (Marsaglia method)
    double z = randomInRange(-1.0, 1.0);
    double phi = randomInRange(0.0, 2.0 * M_PI);
    double r = std::sqrt(1.0 - z * z);

    return Eigen::Vector3d(
        r * std::cos(phi),
        r * std::sin(phi),
        z
    );
}

double SyntheticDataGenerator::randomInRange(double min, double max)
{
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng_);
}

bool SyntheticDataGenerator::isValidOutput(const GuidanceOutput& output)
{
    // Проверка на NaN/Inf в ускорении
    if (!output.acceleration_cmd.allFinite()) {
        return false;
    }

    // Проверка на разумные значения (защита от "взрывов" алгоритма)
    if (output.acceleration_cmd.norm() > 50.0) {
        return false;
    }

    return true;
}

void SyntheticDataGenerator::writeHeader(std::ofstream& file)
{
    file << "u_los_x,u_los_y,u_los_z,"
         << "lambda_x,lambda_y,lambda_z,"
         << "pitch,yaw,roll,"
         << "acc_x,acc_y,acc_z\n";
}

void SyntheticDataGenerator::writeRow(std::ofstream& file,
                                      const Eigen::Vector3d& u_los,
                                      const Eigen::Vector3d& lambda,
                                      double pitch, double yaw, double roll,
                                      const GuidanceOutput& output)
{
    file << std::fixed << std::setprecision(6)
         << u_los.x() << "," << u_los.y() << "," << u_los.z() << ","
         << lambda.x() << "," << lambda.y() << "," << lambda.z() << ","
         << pitch << "," << yaw << "," << roll << ","
         << output.acceleration_cmd.x() << ","
         << output.acceleration_cmd.y() << ","
         << output.acceleration_cmd.z() << "\n";
}

bool SyntheticDataGenerator::generateDataset(GuidanceController* guidance_ctrl,
                                              const std::string& output_path)
{
    return generateDatasetWithProgress(guidance_ctrl, output_path, nullptr);
}

bool SyntheticDataGenerator::generateDatasetWithProgress(
    GuidanceController* guidance_ctrl,
    const std::string& output_path,
    std::function<void(int, int)> progress_callback)
{
    if (!guidance_ctrl) {
        std::cerr << "❌ guidance_ctrl is null\n";
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "❌ Cannot open file: " << output_path << "\n";
        return false;
    }

    writeHeader(file);

    last_stats_ = Stats{};
    last_stats_.total_samples = config_.num_samples;

    std::cout << "🚀 Generating " << config_.num_samples << " samples...\n";

    for (int i = 0; i < config_.num_samples; i++) {
        // 1. Генерация входов
        Eigen::Vector3d u_los = randomUnitVector();

        Eigen::Vector3d lambda(
            randomInRange(config_.lambda_min, config_.lambda_max),
            randomInRange(config_.lambda_min, config_.lambda_max),
            randomInRange(config_.lambda_min, config_.lambda_max)
        );

        double pitch = randomInRange(config_.pitch_min, config_.pitch_max);
        double yaw = randomInRange(config_.yaw_min, config_.yaw_max);
        double roll = randomInRange(config_.roll_min, config_.roll_max);

        // 2. Вызов эталонного алгоритма
        GuidanceOutput output;
        try {
            output = guidance_ctrl->computeAcceleration(u_los, lambda, pitch, yaw, roll);
        } catch (const std::exception& e) {
            last_stats_.skipped_samples++;
            continue;
        }

        // 3. Проверка валидности
        if (!isValidOutput(output)) {
            last_stats_.skipped_samples++;
            continue;
        }

        // 4. Запись в CSV
        writeRow(file, u_los, lambda, pitch, yaw, roll, output);
        last_stats_.valid_samples++;

        // 5. Прогресс
        if (progress_callback && (i + 1) % 1000 == 0) {
            progress_callback(i + 1, config_.num_samples);
        }
    }

    file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.duration_sec = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "✅ Generated " << last_stats_.valid_samples << "/"
              << last_stats_.total_samples << " valid samples in "
              << std::fixed << std::setprecision(2) << last_stats_.duration_sec << "s\n";
    std::cout << "   Skipped: " << last_stats_.skipped_samples << " (NaN/Inf/exception)\n";
    std::cout << "   Output: " << output_path << "\n";

    return true;
}

} // namespace los_guidance