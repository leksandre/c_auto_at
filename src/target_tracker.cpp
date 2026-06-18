/**
 * @file target_tracker.cpp
 * @brief Реализация класса TargetTracker
 */

#include "los_guidance/target_tracker.hpp"
#include <cmath>
#include <algorithm>

namespace los_guidance {

TargetTracker::TargetTracker(const TrackerConfig& config)
  : config_(config)
{
  initCameraMatrices_();
}

void TargetTracker::initCameraMatrices_() {
  // Конвертация углов установки в радианы
  double roll_c = config_.install_roll_deg * M_PI / 180.0;
  double pitch_c = config_.install_pitch_deg * M_PI / 180.0;
  double yaw_c = config_.install_yaw_deg * M_PI / 180.0;
  
  // R_install (Body → rotated frame по углам установки)
  Eigen::Matrix3d R_bc_install =
    (Eigen::AngleAxisd(-yaw_c,   Eigen::Vector3d::UnitZ()) *
     Eigen::AngleAxisd(-pitch_c, Eigen::Vector3d::UnitY()) *
     Eigen::AngleAxisd(-roll_c,  Eigen::Vector3d::UnitX())).toRotationMatrix();
  
  // Axis swap: Body(FRD) → Camera(RDF)
  Eigen::Matrix3d axis_swap;
  axis_swap << 0, 1, 0,  // Body Y → Camera X
               0, 0, 1,  // Body Z → Camera Y
               1, 0, 0;  // Body X → Camera Z
  
  // Итоговая матрица body → camera
  R_bc_ = axis_swap * R_bc_install;
  R_cb_ = R_bc_.transpose();  // camera → body
}

TargetInfo TargetTracker::processTrackerResult(
  const ai_service::msg::TrackerResult::SharedPtr& msg,
  double roll_rad,
  double pitch_rad,
  const Eigen::Matrix3d& R_nb
) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  TargetInfo info;
  info.timestamp_us = msg->timestamp / 1000;  // ns → us
  info.confidence = msg->target_info.confidence;
  
  // Проверка минимального доверия
  if (info.confidence < config_.min_confidence) {
    info.is_valid = false;
    stats_.frames_dropped++;
    return info;
  }
  
  // Пиксельные координаты (центр цели)
  info.pixel_u = (msg->target_info.left + msg->target_info.right) * config_.cx;
  info.pixel_v = (msg->target_info.top + msg->target_info.bottom) * config_.cy;
  info.bbox_width = std::abs(msg->target_info.right - msg->target_info.left);
  info.bbox_height = std::abs(msg->target_info.bottom - msg->target_info.top);
  
  double bbox_frac_w = info.bbox_width / (config_.cx * 2.0);
  double bbox_frac_h = info.bbox_height / (config_.cy * 2.0);
  bool bbox_too_large = (bbox_frac_w > config_.max_bbox_fraction || bbox_frac_h > config_.max_bbox_fraction);
  if (bbox_too_large) {
    bool hard_reject = (bbox_frac_w > 0.85 || bbox_frac_h > 0.85);
    double img_w = config_.cx * 2.0;
    double img_h = config_.cy * 2.0;
    double cx_dist = std::abs(info.pixel_u - config_.cx) / img_w;
    double cy_dist = std::abs(info.pixel_v - config_.cy) / img_h;
    bool centered_enough = (cx_dist <= 0.25 && cy_dist <= 0.25);
    bool near_last_bbox = false;
    if (has_valid_bbox_) {
      double dx = std::abs(info.pixel_u / img_w - last_valid_bbox_cx_);
      double dy = std::abs(info.pixel_v / img_h - last_valid_bbox_cy_);
      near_last_bbox = (dx < config_.bbox_proximity_threshold && dy < config_.bbox_proximity_threshold);
    }
    if (hard_reject || (!centered_enough && !near_last_bbox)) {
      info.is_valid = false;
      stats_.frames_dropped++;
      return info;
    }
  }
  
  // Сглаживание пикселей
  if (!pixel_initialized_) {
    info.pixel_u_smooth = info.pixel_u;
    info.pixel_v_smooth = info.pixel_v;
    pixel_initialized_ = true;
  } else {
    info.pixel_u_smooth = smooth_(pixel_u_prev_, info.pixel_u, config_.pixel_smooth_alpha);
    info.pixel_v_smooth = smooth_(pixel_v_prev_, info.pixel_v, config_.pixel_smooth_alpha);
  }
  pixel_u_prev_ = info.pixel_u_smooth;
  pixel_v_prev_ = info.pixel_v_smooth;
  
  // Коррекция на roll/pitch дрона
  auto [u_correct, v_correct] = correctPixelRollPitch_(
    info.pixel_u_smooth, info.pixel_v_smooth,
    roll_rad, pitch_rad
  );
  
  // Pixel → Camera
  Eigen::Vector3d u_c = pixelToCamVec_(u_correct, v_correct);
  info.los_cam = u_c;
  
  // Camera → Nav
  info.los_nav = camToNav_(u_c,  R_nb);//R_bc_,
  
  // Сглаживание LOS вектора
  if (!los_initialized_) {
    los_nav_prev_ = info.los_nav;
    los_initialized_ = true;
  }
  info.los_nav = smooth_(los_nav_prev_, info.los_nav, config_.los_smooth_alpha);
  los_nav_prev_ = info.los_nav;
  
  // Расчёт угловой скорости LOS (lambda)
  double dt = (last_timestamp_us_ > 0) ? 
    (info.timestamp_us - last_timestamp_us_) * 1e-6 : 0.01;
  info.los_lambda = computeLosLambda_(info.los_nav, los_nav_history_, dt);
  los_nav_history_ = info.los_nav;
  last_timestamp_us_ = info.timestamp_us;
  
  info.is_valid = true;
  last_valid_bbox_cx_ = info.pixel_u / (config_.cx * 2.0);
  last_valid_bbox_cy_ = info.pixel_v / (config_.cy * 2.0);
  has_valid_bbox_ = true;
  stats_.frames_processed++;
  
  return info;
}

TargetInfo TargetTracker::processDetection2D(
  const vision_msgs::msg::Detection2D::SharedPtr& msg,
  double roll_rad,
  double pitch_rad,
  const Eigen::Matrix3d& R_nb
) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  TargetInfo info;
  info.timestamp_us = msg->header.stamp.sec * 1e6 + msg->header.stamp.nanosec * 1e-3;
  info.confidence = 1.0;  // Detection2D не имеет confidence
  
  // Пиксельные координаты из bbox
  info.pixel_u = msg->bbox.center.position.x;
  info.pixel_v = msg->bbox.center.position.y;
  info.bbox_width = msg->bbox.size_x;
  info.bbox_height = msg->bbox.size_y;
  
  double bbox_frac_w = info.bbox_width / (config_.cx * 2.0);
  double bbox_frac_h = info.bbox_height / (config_.cy * 2.0);
  bool bbox_too_large = (bbox_frac_w > config_.max_bbox_fraction || bbox_frac_h > config_.max_bbox_fraction);
  if (bbox_too_large) {
    bool hard_reject = (bbox_frac_w > 0.85 || bbox_frac_h > 0.85);
    double img_w = config_.cx * 2.0;
    double img_h = config_.cy * 2.0;
    double cx_dist = std::abs(info.pixel_u - config_.cx) / img_w;
    double cy_dist = std::abs(info.pixel_v - config_.cy) / img_h;
    bool centered_enough = (cx_dist <= 0.25 && cy_dist <= 0.25);
    bool near_last_bbox = false;
    if (has_valid_bbox_) {
      double dx = std::abs(info.pixel_u / img_w - last_valid_bbox_cx_);
      double dy = std::abs(info.pixel_v / img_h - last_valid_bbox_cy_);
      near_last_bbox = (dx < config_.bbox_proximity_threshold && dy < config_.bbox_proximity_threshold);
    }
    if (hard_reject || (!centered_enough && !near_last_bbox)) {
      info.is_valid = false;
      stats_.frames_dropped++;
      return info;
    }
  }
  
  // Сглаживание пикселей
  if (!pixel_initialized_) {
    info.pixel_u_smooth = info.pixel_u;
    info.pixel_v_smooth = info.pixel_v;
    pixel_initialized_ = true;
  } else {
    info.pixel_u_smooth = smooth_(pixel_u_prev_, info.pixel_u, config_.pixel_smooth_alpha);
    info.pixel_v_smooth = smooth_(pixel_v_prev_, info.pixel_v, config_.pixel_smooth_alpha);
  }
  pixel_u_prev_ = info.pixel_u_smooth;
  pixel_v_prev_ = info.pixel_v_smooth;
  
  // Коррекция на roll/pitch
  auto [u_correct, v_correct] = correctPixelRollPitch_(
    info.pixel_u_smooth, info.pixel_v_smooth,
    roll_rad, pitch_rad
  );
  
  // Pixel → Camera → Nav
  Eigen::Vector3d u_c = pixelToCamVec_(u_correct, v_correct);
  info.los_cam = u_c;
  info.los_nav = camToNav_(u_c,  R_nb); //R_bc_,
  
  // Сглаживание LOS
  if (!los_initialized_) {
    los_nav_prev_ = info.los_nav;
    los_initialized_ = true;
  }
  info.los_nav = smooth_(los_nav_prev_, info.los_nav, config_.los_smooth_alpha);
  los_nav_prev_ = info.los_nav;
  
  // Lambda
  double dt = (last_timestamp_us_ > 0) ?
    (info.timestamp_us - last_timestamp_us_) * 1e-6 : 0.01;
  info.los_lambda = computeLosLambda_(info.los_nav, los_nav_history_, dt);
  los_nav_history_ = info.los_nav;
  last_timestamp_us_ = info.timestamp_us;
  
  info.is_valid = true;
  last_valid_bbox_cx_ = info.pixel_u / (config_.cx * 2.0);
  last_valid_bbox_cy_ = info.pixel_v / (config_.cy * 2.0);
  has_valid_bbox_ = true;
  stats_.frames_processed++;
  
  return info;
}

Eigen::Vector3d TargetTracker::pixelToCamVec_(double u, double v) {
  Eigen::Vector3d vec;
  vec << (u - config_.cx) / config_.fx,
         (v - config_.cy) / config_.fy,
         1.0;
  return vec.normalized();
}

std::pair<double, double> TargetTracker::correctPixelRollPitch_(
  double u, double v,
  double roll_rad, double pitch_rad
) {
  // Back-project pixel to camera ray
  double x = (u - config_.cx) / config_.fx;
  double y = (v - config_.cy) / config_.fy;
  Eigen::Vector3d r_cam(x, y, 1.0);
  r_cam.normalize();
  
  // Rotation matrix для коррекции (undo roll/pitch)
  Eigen::Matrix3d R_body_corr =
    (Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY()) *
     Eigen::AngleAxisd(roll_rad,  Eigen::Vector3d::UnitX())).toRotationMatrix();
  
  // Apply correction: R_bc * R_corr * R_bc^T * r_cam
  Eigen::Vector3d r_corr = R_bc_ * R_body_corr * R_cb_ * r_cam;
  
  // Protection against degenerate case (z ~ 0)
  const double EPS_Z = 1e-4;
  if (std::abs(r_corr.z()) < EPS_Z) {
    return {u, v};  // Fallback: no correction
  }
  
  // Project back to pixel coordinates
  double u_correct = config_.cx + config_.fx * (r_corr.x() / r_corr.z());
  double v_correct = config_.cy + config_.fy * (r_corr.y() / r_corr.z());
  
  // Clamp to image plane
  u_correct = std::clamp(u_correct, 0.0, 2.0 * config_.cx);
  v_correct = std::clamp(v_correct, 0.0, 2.0 * config_.cy);
  
  return {u_correct, v_correct};
}

Eigen::Vector3d TargetTracker::camToNav_(
  const Eigen::Vector3d& u_c,
  // const Eigen::Matrix3d& R_bc,
  const Eigen::Matrix3d& R_nb
) {
  Eigen::Vector3d u_b = R_cb_ * u_c;      // camera → body
  Eigen::Vector3d u_n = R_nb.transpose() * u_b;  // body → nav
  return u_n.normalized();
}

Eigen::Vector3d TargetTracker::computeLosLambda_(
  const Eigen::Vector3d& los_current,
  const Eigen::Vector3d& los_prev,
  double dt
) {
  // lambda = u_n × u_dot_n (cross product)
  Eigen::Vector3d u_dot = (los_current - los_prev) / dt;
  Eigen::Vector3d lambda_raw = los_current.cross(u_dot);
  
  // Экспоненциальное сглаживание lambda для подавления шума дифференцирования
  if (!lambda_initialized_) {
    lambda_smooth_ = lambda_raw;
    lambda_initialized_ = true;
  } else {
    double alpha = 0.3; // Коэффициент сглаживания (0.0-1.0, меньше = плавнее)
    lambda_smooth_ = alpha * lambda_raw + (1.0 - alpha) * lambda_smooth_;
  }
  
  return lambda_smooth_;
}

double TargetTracker::smooth_(double prev, double current, double alpha) {
  return prev + alpha * (current - prev);
}

Eigen::Vector3d TargetTracker::smooth_(
  const Eigen::Vector3d& prev,
  const Eigen::Vector3d& current,
  double alpha
) {
  return prev + alpha * (current - prev);
}

Eigen::Matrix3d TargetTracker::skewSymmetric_(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m <<     0, -v.z(),  v.y(),
         v.z(),     0, -v.x(),
        -v.y(),  v.x(),     0;
  return m;
}

// === Public getters ===

Eigen::Vector3d TargetTracker::getLosVector() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return los_nav_prev_;
}

Eigen::Vector3d TargetTracker::getLosAngularVelocity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return lambda_smooth_;
}

bool TargetTracker::isValid() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return los_initialized_;
}

void TargetTracker::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  pixel_initialized_ = false;
  los_initialized_ = false;
  lambda_initialized_ = false;
  pixel_u_prev_ = 0.0;
  pixel_v_prev_ = 0.0;
  los_nav_prev_.setZero();
  los_nav_history_.setZero();
  lambda_smooth_.setZero();
  last_timestamp_us_ = 0;
  has_valid_bbox_ = false;
  last_valid_bbox_cx_ = 0.5;
  last_valid_bbox_cy_ = 0.5;
}

void TargetTracker::updateConfig(const TrackerConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  initCameraMatrices_();
}

TargetTracker::TrackerStats TargetTracker::getStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

} // namespace los_guidance