/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <boost/math/distributions/chi_squared.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <ack_msckf_lam/ack_msckf_lam.h>
#include <ack_msckf_lam/math_utils.hpp>
#include <ack_msckf_lam/utils.h>
#include <ack_msckf_lam/navsat_conversions.h>
#include <ack_msckf_lam/registration.hpp>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;

namespace ack_msckf_lam{
 
// Static member variables in ACKState class.
StateIDType ACKState::next_id = 0;
double ACKState::roll_noise = 0.001;
double ACKState::pitch_noise = 0.01;
double ACKState::steer_noise = 0.01;
double ACKState::vx_noise = 0.01;
double ACKState::vy_noise = 0.01;
double ACKState::vz_noise = 0.01;
double ACKState::WheelBase = 1.9;
double ACKState::AckRate = 150;
double ACKState::TireBase = 0.98;
double ACKState::SteerRatio = 360/21;

// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);

// Static member variables in CAMState class.
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables int Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> AckMsckfLam::chi_squared_test_table;

AckMsckfLam::AckMsckfLam(ros::NodeHandle& pnh):
  is_gravity_set(false),
  is_first_img(true),
  nh(pnh) {
  return;
}

/**
 * @brief batchAckProcessing
 *
 */
void AckMsckfLam::batchAckProcessing(const double& time_bound) {

  state_server.ack_state.ack_available = false;
  state_server.ack_state.delta_q = Eigen::Quaterniond::Identity();
  state_server.ack_state.delta_p = Eigen::Vector3d::Zero();
  state_server.ack_state.ackermann_x_ = 0;
  state_server.ack_state.ackermann_y_ = 0;
  state_server.ack_state.ackermann_heading_ = 0;
  // Clear the Ackermann Jacobian and noise
  state_server.ack_state.transferFunctionJacobian_.setZero();
  state_server.ack_state.processFunctionJacobian_.setZero();
  state_server.ack_state.estimateErrorCovariance_w_.setIdentity();
  state_server.ack_state.estimateErrorCovariance_w_ *= 1e-9;
  state_server.ack_state.estimateErrorCovariance_.setIdentity();
  state_server.ack_state.estimateErrorCovariance_ *= 1e-9;
  state_server.ack_state.processNoiseCovariance_ackerman_.setZero();
  state_server.ack_state.processNoiseCovariance_ackerman_(0,0) =  pow(state_server.ack_state.ackermann_speed_x_noise , 2); 
  state_server.ack_state.processNoiseCovariance_ackerman_(1,1) =  pow(state_server.ack_state.ackermann_steering_noise * M_PI / 180 , 2); 
  state_server.ack_state.processWhiteNoiseCovariance_.setZero();
  state_server.ack_state.processWhiteNoiseCovariance_(0,0) = pow(state_server.ack_state.ackermann_x_white_noise , 2); 
  state_server.ack_state.processWhiteNoiseCovariance_(1,1) = pow(state_server.ack_state.ackermann_y_white_noise , 2); 
  state_server.ack_state.processWhiteNoiseCovariance_(2,2) = pow(state_server.ack_state.ackermann_heading_white_noise * M_PI / 180 , 2);
  state_server.ack_state.delta_time = 0;

  // Counter how many IMU msgs in the buffer are used.
  int used_ack_msg_cntr = 0;
  int speed_cntr = 0;
  for (const auto& ack_msg : ack_msg_buffer) {
    double ack_time = ack_msg.header.stamp.toSec();
    if (ack_time < state_server.ack_state.time) {
      ++used_ack_msg_cntr;
      continue;
    }

    if (ack_time > time_bound) break;

    // Convert the msgs.
    Vector3d m_speed = Vector3d(ack_msg.drive.speed,0,0);
    double m_steering_angle = ack_msg.drive.steering_angle;

    speed_cntr++;
    // Execute processModel_ack.
    processModel_ack(ack_time, m_speed, m_steering_angle);
    ++used_ack_msg_cntr;
  }

  if(state_server.ack_state.ack_available){
  }else{
    ROS_INFO("ack is not available....");
  }
  
  // Set the state ID for the new ACK state.
  state_server.ack_state.id = ACKState::next_id++;

  // Remove all used IMU msgs.
  ack_msg_buffer.erase(ack_msg_buffer.begin(),
      ack_msg_buffer.begin()+used_ack_msg_cntr);

  return;
}

/**
 * @brief processModel_ack 
 *  estimateErrorCovariance_
 */
void AckMsckfLam::processModel_ack(const double& time,
    Eigen::Vector3d& m_speed_,
    double& m_steering_angle) {

  // ACK
  double dtime = time - state_server.ack_state.time;
  state_server.ack_state.delta_time += dtime;

  if(m_steering_angle < 0){
    state_server.ack_state.ackermann_TireBase_sign = -1 * ACKState::TireBase;
  }else{
    state_server.ack_state.ackermann_TireBase_sign = ACKState::TireBase;
  }
  
  state_server.ack_state.ackermann_WheelBase = ACKState::WheelBase - state_server.ack_state.ackermann_bl;
  state_server.ack_state.ackermann_TireBase = state_server.ack_state.ackermann_TireBase_sign - state_server.ack_state.ackermann_bt;

  Vector3d m_speed_est = (1 - state_server.ack_state.ackermann_kv) * m_speed_;
  Eigen::Vector3d m_speed = m_speed_est;

  double m_steering_angle_est = (1 - state_server.ack_state.ackermann_ks) * m_steering_angle - state_server.ack_state.ackermann_bs;
  double steering_angle_0 = m_steering_angle_est;

  double w_t_s = 2 * state_server.ack_state.ackermann_WheelBase - state_server.ack_state.ackermann_TireBase *  tan(steering_angle_0);
  double m_gyro_z = 2 * m_speed[0] * tan(steering_angle_0) / w_t_s;

  double steer_radius_inv = 2 * tan(steering_angle_0) / w_t_s;

  state_server.ack_state.delta_q = state_server.ack_state.delta_q * Quaterniond(1, 0, 0 , m_gyro_z * dtime / 2);
  state_server.ack_state.delta_q.normalize();
  state_server.ack_state.delta_p = state_server.ack_state.delta_p + state_server.ack_state.delta_q  * m_speed * dtime;

  double delta = dtime;
  const double linear = m_speed[0] * delta;
  double r_mul_tan = state_server.ack_state.ackermann_WheelBase - state_server.ack_state.ackermann_TireBase * tan(steering_angle_0) / 2;
  double pow_r_mul_tan = pow(r_mul_tan,2);

  // dr = ds/R
  const double deta_yaw = linear * steer_radius_inv;
  double & ackermann_heading_ = state_server.ack_state.ackermann_heading_;
  double & ackermann_x_ = state_server.ack_state.ackermann_x_;
  double & ackermann_y_ = state_server.ack_state.ackermann_y_;
  // Handle wrapping / Normalize
  if(ackermann_heading_ >= M_PI) ackermann_heading_ -= 2.0 * M_PI;
  if(ackermann_heading_ <= (-M_PI)) ackermann_heading_ += 2.0 * M_PI;
  if (dtime < 0.00001){
     return; 
  }
  double cos_y = cos(ackermann_heading_ + deta_yaw/2 );
  double sin_y = sin(ackermann_heading_ + deta_yaw/2 );                
  double pow_d = pow(linear,2);
  double pow_tan = pow(tan(steering_angle_0),2);
  double dFx_dY = -linear * sin(ackermann_heading_ + deta_yaw/2 );
  double dFy_dY = linear * cos(ackermann_heading_ + deta_yaw/2 );
  // Much of the transfer function Jacobian is identical to the transfer function
  state_server.ack_state.transferFunctionJacobian_(0, 0) = 1;
  state_server.ack_state.transferFunctionJacobian_(0, 2) = dFx_dY;
  state_server.ack_state.transferFunctionJacobian_(1, 1) = 1;
  state_server.ack_state.transferFunctionJacobian_(1, 2) = dFy_dY;
  state_server.ack_state.transferFunctionJacobian_(2, 2) = 1;
  state_server.ack_state.processFunctionJacobian_.resize(3,2);
  state_server.ack_state.processFunctionJacobian_.setZero();
  state_server.ack_state.processFunctionJacobian_(0,0) = -deta_yaw * delta  * sin_y / 2 + delta * cos_y;
  state_server.ack_state.processFunctionJacobian_(0,1) = pow_d * state_server.ack_state.ackermann_WheelBase * (pow_tan + 1) * sin_y / ( ( 2 * pow_r_mul_tan ) );
  state_server.ack_state.processFunctionJacobian_(1,0) = deta_yaw * delta * cos_y / 2 + delta * sin_y;
  state_server.ack_state.processFunctionJacobian_(1,1) = -pow_d * state_server.ack_state.ackermann_WheelBase * (pow_tan + 1) * sin_y / ( ( 2 * pow_r_mul_tan ) );
  state_server.ack_state.processFunctionJacobian_(2,0) = delta * steer_radius_inv;
  state_server.ack_state.processFunctionJacobian_(2,1) = linear * state_server.ack_state.ackermann_WheelBase * (pow_tan + 1) / pow_r_mul_tan;
  
  state_server.ack_state.estimateErrorCovariance_ = (state_server.ack_state.transferFunctionJacobian_ *
                              state_server.ack_state.estimateErrorCovariance_ *
                              state_server.ack_state.transferFunctionJacobian_.transpose());
  state_server.ack_state.estimateErrorCovariance_.noalias() += (state_server.ack_state.processFunctionJacobian_ * state_server.ack_state.processNoiseCovariance_ackerman_ * state_server.ack_state.processFunctionJacobian_.transpose()) + state_server.ack_state.processWhiteNoiseCovariance_;
  
  MatrixXd state_cov_fixed = (state_server.ack_state.estimateErrorCovariance_ +
      state_server.ack_state.estimateErrorCovariance_.transpose()) / 2.0;
  state_server.ack_state.estimateErrorCovariance_ = state_cov_fixed;

  /// Integrate odometry: RungeKutta2
  const double direction = ackermann_heading_ + deta_yaw * 0.5;
  /// Runge-Kutta 2nd order integration:
  ackermann_x_       += linear * cos(direction);
  ackermann_y_       += linear * sin(direction);
  ackermann_heading_ += deta_yaw;
  // Handle wrapping / Normalize
  if(ackermann_heading_ >= M_PI)ackermann_heading_ -= 2.0 * M_PI;
  if(ackermann_heading_ <= (-M_PI))ackermann_heading_ += 2.0 * M_PI;
  
  // state info
  state_server.ack_state.time = time;
  state_server.ack_state.ack_available = true;
  state_server.ack_state.velocity = m_speed;
  state_server.ack_state.angular_velocity = {0, 0, m_gyro_z};
  state_server.ack_state.processFunctionJacobian_.resize(1,2);
  state_server.ack_state.processFunctionJacobian_.setZero();
  state_server.ack_state.processFunctionJacobian_(0,0) = steer_radius_inv;
  state_server.ack_state.processFunctionJacobian_(0,1) = m_speed[0] * state_server.ack_state.ackermann_WheelBase * (pow_tan + 1) / pow_r_mul_tan;
  state_server.ack_state.estimateErrorCovariance_w_.setIdentity();
  state_server.ack_state.estimateErrorCovariance_w_ *= 1e-9;
  state_server.ack_state.estimateErrorCovariance_w_ = (state_server.ack_state.processFunctionJacobian_ * state_server.ack_state.processNoiseCovariance_ackerman_ * state_server.ack_state.processFunctionJacobian_.transpose());
  state_server.ack_state.estimateErrorCovariance_w_(0,0) += state_server.ack_state.processWhiteNoiseCovariance_(2,2);
  return;
}

/**
 * @brief initialize 
 *  loadParameters()
 *  createRosIO()
 */
 bool AckMsckfLam::initialize() {

  if (!loadParameters()) return false;
  ROS_INFO("Finish loading ROS parameters...");
  
  // Initialize state server IMU
  state_server.continuous_noise_cov =
    Matrix<double, 12, 12>::Zero();
  state_server.continuous_noise_cov.block<3, 3>(0, 0) =
    Matrix3d::Identity()*IMUState::gyro_noise;
  state_server.continuous_noise_cov.block<3, 3>(3, 3) =
    Matrix3d::Identity()*IMUState::gyro_bias_noise;
  state_server.continuous_noise_cov.block<3, 3>(6, 6) =
    Matrix3d::Identity()*IMUState::acc_noise;
  state_server.continuous_noise_cov.block<3, 3>(9, 9) =
    Matrix3d::Identity()*IMUState::acc_bias_noise;

  // Initialize the chi squared test table with confidence
  // level 0.95.
  for (int i = 1; i < 100; ++i) {
    boost::math::chi_squared chi_squared_dist(i);
    // quantile
    chi_squared_test_table[i] =
      boost::math::quantile(chi_squared_dist, 0.05);
  }

  if (!createRosIO()) return false;
  ROS_INFO("Finish creating ROS IO...");

  return true;
}

/**
 * @brief loadParameters 
 *  
 */
bool AckMsckfLam::loadParameters() {

  // raw_gnss fusion
  nh.param<bool>("use_raw_gnss", use_raw_gnss, false);

  nh.param<bool>("use_ackermann", use_ackermann, true);
  nh.param<bool>("use_a27_platform", use_a27_platform, true);
  nh.param<bool>("use_svd_ex", use_svd_ex, true);
  nh.param<bool>("use_debug", use_debug, true);
  nh.param<bool>("ackermann_update_v", ackermann_update_v, true);
  nh.param<bool>("ackermann_update_q", ackermann_update_q, true);
  nh.param<double>("ackermann_rate", ACKState::AckRate, 150);
  nh.param<double>("ackermann_steer_ratio", ACKState::SteerRatio, 360/21);
  nh.param<double>("noise/ackermann_velocity_x_std",
    state_server.ack_state.ackermann_speed_x_noise, 0.5);
  nh.param<double>("noise/ackermann_velocity_y_std",
    state_server.ack_state.ackermann_speed_y_noise, 0.5);
  nh.param<double>("noise/ackermann_velocity_z_std",
    state_server.ack_state.ackermann_speed_z_noise, 0.5);
  nh.param<double>("noise/ackermann_steerAngle_std",
    state_server.ack_state.ackermann_steering_noise, 30);
  nh.param<double>("noise/ackermann_heading_white_std",
    state_server.ack_state.ackermann_heading_white_noise, 3);
  nh.param<double>("noise/ackermann_x_white_std",
    state_server.ack_state.ackermann_x_white_noise, 0.1); 
  nh.param<double>("noise/ackermann_y_white_std",
    state_server.ack_state.ackermann_y_white_noise, 0.1);  
  state_server.ack_state.delta_q = Eigen::Quaterniond::Identity();
  nh.param<bool>("use_offline_bias", use_offline_bias, false);
  double gyro_bias_a, gyro_bias_b, gyro_bias_c;
  nh.param<double>("initial_state/gyro_bias_a",
    gyro_bias_a, 0.001);
  nh.param<double>("initial_state/gyro_bias_b",
    gyro_bias_b, 0.001);
  nh.param<double>("initial_state/gyro_bias_c",
    gyro_bias_c, 0.001);
  state_server.ack_state.initial_bias = {gyro_bias_a, gyro_bias_b, gyro_bias_c};
  state_server.imu_state.initial_bias = {gyro_bias_a, gyro_bias_b, gyro_bias_c};

  nh.getParam("ackermann_kv", state_server.ack_state.ackermann_kv);
  nh.getParam("ackermann_ks", state_server.ack_state.ackermann_ks);
  nh.getParam("ackermann_bs", state_server.ack_state.ackermann_bs);
  nh.getParam("ackermann_bl", state_server.ack_state.ackermann_bl);
  nh.getParam("ackermann_bt", state_server.ack_state.ackermann_bt);
  nh.getParam("ackermann_WheelBase", ACKState::WheelBase);
  nh.getParam("ackermann_TireBase", ACKState::TireBase);
  ROS_INFO_STREAM("ACKState::WheelBase: " << ACKState::WheelBase);
  ROS_INFO_STREAM("ackermann_TireBase: " << ACKState::TireBase);
  
  nh.param<string>("output_path",output_path,"");
    ROS_INFO_STREAM("Loaded ex_output_file: " << output_path);

  string file_time;
		[&]() { 
			 time_t timep;
		     time (&timep);
		     char tmp[64];
		     strftime(tmp, sizeof(tmp), "%Y-%m-%d-%H-%M-%S",localtime(&timep) );
		     file_time = tmp;
			 }();

  if(use_raw_gnss){

    // pose
    pose_file.open((output_path + "pose_ack_msckf_lm_lever_gnss" + "_" + file_time + ".txt"), std::ios::out);
    if(!pose_file.is_open())
    {
        cerr << "pose_ack_msckf_lm_lever_gnss is not open" << endl;
    }
    // odom
    odom_file.open((output_path + "odom_ack_msckf_lm_lever_gnss" + "_" + file_time + ".csv"), std::ios::out);
    if(!odom_file.is_open())
    {
        cerr << "odom_ack_msckf_lm_lever_gnss is not open" << endl;
    }
    // std
    std_file.open((output_path + "std_ack_msckf_lm_lever_gnss" + "_" + file_time + ".csv"), std::ios::out);
    if(!std_file.is_open())
    {
        cerr << "std_ack_msckf_lm_lever_gnss is not open" << endl;
    }
    // rmse
    rmse_file.open((output_path + "rmse_ack_msckf_lm_lever_gnss" + "_" + file_time + ".csv"), std::ios::out);
    if(!rmse_file.is_open())
    {
        cerr << "rmse_ack_msckf_lm_lever_gnss is not open" << endl;
    }
    // time
    time_file.open((output_path + "time_ack_msckf_lm_lever_gnss" + "_" + file_time + ".csv"), std::ios::out);
    if(!time_file .is_open())
    {
        cerr << "time_ack_msckf_lm_lever_gnss is not open" << endl;
    }

  }else{
  
    // pose
    pose_file.open((output_path + "pose_ack_msckf_lm_lever" + "_" + file_time + ".txt"), std::ios::out);
    if(!pose_file.is_open())
    {
        cerr << "pose_ack_msckf_lm_lever is not open" << endl;
    }
    // odom
    odom_file.open((output_path + "odom_ack_msckf_lm_lever" + "_" + file_time + ".csv"), std::ios::out);
    if(!odom_file.is_open())
    {
        cerr << "odom_ack_msckf_lm_lever is not open" << endl;
    }
    // std
    std_file.open((output_path + "std_ack_msckf_lm_lever" + "_" + file_time + ".csv"), std::ios::out);
    if(!std_file.is_open())
    {
        cerr << "std_ack_msckf_lm_lever is not open" << endl;
    }
    // rmse
    rmse_file.open((output_path + "rmse_ack_msckf_lm_lever" + "_" + file_time + ".csv"), std::ios::out);
    if(!rmse_file.is_open())
    {
        cerr << "rmse_ack_msckf_lm_lever is not open" << endl;
    }
    // time
    time_file.open((output_path + "time_ack_msckf_lm_lever" + "_" + file_time + ".csv"), std::ios::out);
    if(!time_file .is_open())
    {
        cerr << "time_ack_msckf_lm_lever is not open" << endl;
    }
  }

  std::string delim = ",";
  // odom
  odom_file << "#Time(sec),";
  odom_file << "Dtime(sec),";
  odom_file  << delim;
  odom_file << "x(m),y(m),z(m),";
  odom_file << "qx,qy,qz,qw,";
  odom_file << "roll_x_G(deg),pitch_y_G(deg),yaw_z_G(deg),";
  odom_file << "vx(m/s),vy(m/s),vz(m/s),";
  odom_file << "wx(rad/s),wy(rad/s),wz(rad/s),";
  odom_file  << delim;
  odom_file << "gt_x(m),gt_y(m),gt_z(m),";
  odom_file << "gt_qx,gt_qy,gt_qz,gt_qw,";
  odom_file << "gt_roll_x_G(deg),gt_pitch_y_G(deg),gt_yaw_z_G(deg),";
  odom_file << "gt_vx(m/s),gt_vy(m/s),gt_vz(m/s),";
  odom_file << "gt_wx(rad/s),gt_wy(rad/s),gt_wz(rad/s),";
  odom_file  << delim;
  odom_file << "Sr,Sr_avg,";
  odom_file  << delim;
  odom_file << "pibx(m),piby(m),pibz(m),";
  odom_file << "qibx,qiby,qibz,qibw,";
  odom_file << "roll_ibx(deg),pitch_iby(deg),yaw_ibz(deg),";
  odom_file  << delim;
  odom_file << "vx_la(m/s),vy_la(m/s),vz_la(m/s),";
  odom_file << std::endl;

  // std
  std_file << "#Time(sec),";
  std_file << "Dtime(sec),";
  std_file  << delim;
  std_file  << "err_rx(deg),err_ry(deg),err_rz(deg),";
  std_file  << "std3_rx(deg),std3_ry(deg),std3_rz(deg),";
  std_file  << "-std3_rx(deg),-std3_ry(deg),-std3_rz(deg),";
  std_file << delim;
  std_file  << "err_px(m),err_py(m),err_pz(m),";
  std_file  << "std3_px(m),std3_py(m),std3_pz(m),";
  std_file  << "-std3_px(m),-std3_py(m),-std3_pz(m),";
  std_file  << delim;
  std_file << "err_vx(m/s),err_vy(m/s),err_vz(m/s),";
  std_file  << "std3_vx(m/s),std3_vy(m/s),std3_vz(m/s),";
  std_file  << "-std3_vx(m/s),-std3_vy(m/s),-std3_vz(m/s),";
  std_file  << delim;
  std_file  << "err_ribx(deg),err_riby(deg),err_ribz(deg),";
  std_file  << "std3_ribx(deg),std3_riby(deg),std3_ribz(deg),";
  std_file  << "-std3_ribx(deg),-std3_riby(deg),-std3_ribz(deg),";
  std_file  << delim;
  std_file  << "err_pbix(m),err_pbiy(m),err_pbiz(m),";
  std_file  << "std3_pbix(m),std3_pbiy(m),std3_pbiz(m),";
  std_file  << "-std3_pbix(m),-std3_pbiy(m),-std3_pbiz(m),";
  std_file << delim;
  std_file  << "err_bgx(deg/h),err_bgy(deg/h),err_bgz(deg/h),";
  std_file  << "std3_bgx(deg/h),std3_bgy(deg/h),std3_bgz(deg/h),";
  std_file  << "-std3_bgx(deg/h),-std3_bgy(deg/h),-std3_bgz(deg/h),";
  std_file << delim;
  std_file  << "err_bax(m/s^2),err_bay(m/s^2),err_baz(m/s^2),";
  std_file  << "std3_bax(m/s^2),std3_bay(m/s^2),std3_baz(m/s^2),";
  std_file  << "-std3_bax(m/s^2),-std3_bay(m/s^2),-std3_baz(m/s^2),";
  std_file << std::endl;

  // rmse nees
  rmse_file  << "#Time(sec),";
  rmse_file << "Dtime(sec),";
  rmse_file  << delim;
  rmse_file  << "serr_rx(deg^2),serr_ry(deg^2),serr_rz(deg^2),serr_rxyz(deg^2),";
  rmse_file  << "nees_rx,nees_ry,nees_rz,";
  rmse_file  << delim;
  rmse_file  << "serr_px(m^2),serr_py(m^2),serr_pz(m^2),serr_pxyz(m^2),";
  rmse_file  << "nees_px,nees_py,nees_pz,";
  rmse_file  << delim;
  rmse_file  << "serr_ribx(deg^2),serr_riby(deg^2),serr_ribz(deg^2),";
  rmse_file  << "nees_ribx,nees_riby,nees_ribz,";
  rmse_file  << delim;
  rmse_file  << "serr_pbix(m^2),serr_pbiy(m^2),serr_pbiz(m^2),";
  rmse_file  << "nees_pbix,nees_pbiy,nees_pbiz,";
  rmse_file  << delim;
  rmse_file  << "serr_bgx,serr_bgy,serr_bgz,";
  rmse_file  << "nees_bgx,nees_bgy,nees_bgz,";
  rmse_file  << delim;
  rmse_file  << "serr_bax,serr_bay,serr_baz,";
  rmse_file  << "nees_bax,nees_bay,nees_baz,";
  rmse_file  << delim;
  rmse_file  << "serr_vx,serr_vy,serr_vz,";
  rmse_file  << "nees_vx,nees_vy,nees_vz,";
  rmse_file  << std::endl;

  // time
  time_file  << "#Time(sec),";
  time_file  << "Dtime(sec),";
  time_file  << delim;
  time_file  << "process_time(ms),total_time(ms),avg_time(ms),";
  time_file  << std::endl;

  // Frame id
  nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
  nh.param<string>("child_frame_id", child_frame_id, "robot");
  nh.param<bool>("publish_tf", publish_tf, true);
  nh.param<double>("frame_rate", frame_rate, 30.0);//30
  nh.param<double>("position_std_threshold", position_std_threshold, 8.0);

  nh.param<double>("rotation_threshold", rotation_threshold, 0.2618);
  nh.param<double>("translation_threshold", translation_threshold, 0.4);
  nh.param<double>("tracking_rate_threshold", tracking_rate_threshold, 0.5);

  // Feature optimization parameters
  nh.param<double>("feature/config/translation_threshold",
      Feature::optimization_config.translation_threshold, 0.2);

  // Noise related parameters
  nh.param<double>("noise/gyro", IMUState::gyro_noise, 0.001);
  nh.param<double>("noise/acc", IMUState::acc_noise, 0.01);
  nh.param<double>("noise/gyro_bias", IMUState::gyro_bias_noise, 0.001);
  nh.param<double>("noise/acc_bias", IMUState::acc_bias_noise, 0.01);
  nh.param<double>("noise/feature", Feature::observation_noise, 0.01);
  // Use variance instead of standard deviation.
  IMUState::gyro_noise *= IMUState::gyro_noise;
  IMUState::acc_noise *= IMUState::acc_noise;
  IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
  IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
  Feature::observation_noise *= Feature::observation_noise;

  // Set the initial state.
  nh.param<double>("initial_state/velocity/x",
      state_server.imu_state.velocity(0), 0.0);
  nh.param<double>("initial_state/velocity/y",
      state_server.imu_state.velocity(1), 0.0);
  nh.param<double>("initial_state/velocity/z",
      state_server.imu_state.velocity(2), 0.0);

  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);

  // state_cov
  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 12; i < 15; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;

  // Hamilton -> JPL
  Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
  state_server.imu_state.T_imu_cam0 = T_imu_cam0;
  state_server.imu_state.R_imu_cam0 = T_imu_cam0.linear();
  state_server.imu_state.t_cam0_imu = T_imu_cam0.inverse().translation();

  // platform
  if(use_a27_platform){
    // a27_platform 
    if(use_svd_ex){
          // is_svd
          // Hamilton -> JPL
          Eigen::Isometry3d T_imu_gps = utils::getTransformEigen(nh, "T_gps_imu");
          Eigen::Isometry3d T_body_gps = utils::getTransformEigen(nh, "T_gps_body");
          
          state_server.imu_state.T_imu_body = T_body_gps.inverse() * T_imu_gps;
      }else{
          // ib_hand
          // Hamilton -> JPL
          state_server.imu_state.T_imu_body =
            utils::getTransformEigen(nh, "T_body_imu");
      }   

  }else{
    // c11_platform 
    if(use_svd_ex){
          // is_svd
          // Hamilton -> JPL
          Eigen::Isometry3d T_imu_gps = utils::getTransformEigen(nh, "T_gps_imu");
          Eigen::Isometry3d T_body_gps = utils::getTransformEigen(nh, "T_gps_body");
          
          state_server.imu_state.T_imu_body = T_body_gps.inverse() * T_imu_gps;
      }else{
          // kalibr
          // Hamilton -> JPL
          CAMState::T_cam0_cam1 =
                utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
          Eigen::Isometry3d T_cam1_calibcam2 = utils::getTransformEigen(nh, "T_calibcam2_cam1");
          Eigen::Isometry3d T_calibcam2_body = utils::getTransformEigen(nh, "T_calibcam2_body").inverse();
          Eigen::Isometry3d T_imu_cam1 =  CAMState::T_cam0_cam1 * T_imu_cam0;

          state_server.imu_state.T_imu_body = T_calibcam2_body * T_cam1_calibcam2 * T_imu_cam1;
      }
  }

  state_server.imu_state.R_imu_body = state_server.imu_state.T_imu_body.linear();
  state_server.imu_state.t_imu_body = state_server.imu_state.T_imu_body.translation();
  state_server.imu_state.T_body_imu = state_server.imu_state.T_imu_body.inverse();
  state_server.imu_state.t_body_imu = state_server.imu_state.T_body_imu.translation();

  // body <-> camera  
  state_server.ack_state.T_body_cam0 = state_server.imu_state.T_imu_cam0 * state_server.imu_state.T_imu_body.inverse();
  state_server.ack_state.R_body_cam0 = state_server.ack_state.T_body_cam0.linear();
  state_server.ack_state.t_body_cam0 = state_server.ack_state.T_body_cam0.translation();
  state_server.ack_state.t_cam0_body = state_server.ack_state.T_body_cam0.inverse().translation();

  // Maximum number of camera states to be stored
  nh.param<int>("max_cam_state_size", max_cam_state_size, 30);
  ROS_INFO("===========================================");
  ROS_INFO("fixed frame id: %s", fixed_frame_id.c_str());
  ROS_INFO("child frame id: %s", child_frame_id.c_str());
  ROS_INFO("publish tf: %d", publish_tf);
  ROS_INFO("frame rate: %f", frame_rate);
  ROS_INFO("position std threshold: %f", position_std_threshold);
  ROS_INFO("Keyframe rotation threshold: %f", rotation_threshold);
  ROS_INFO("Keyframe translation threshold: %f", translation_threshold);
  ROS_INFO("Keyframe tracking rate threshold: %f", tracking_rate_threshold);
  ROS_INFO("gyro noise: %.10f", IMUState::gyro_noise);
  ROS_INFO("gyro bias noise: %.10f", IMUState::gyro_bias_noise);
  ROS_INFO("acc noise: %.10f", IMUState::acc_noise);
  ROS_INFO("acc bias noise: %.10f", IMUState::acc_bias_noise);
  ROS_INFO("observation noise: %.10f", Feature::observation_noise);
  ROS_INFO("initial velocity: %f, %f, %f",
      state_server.imu_state.velocity(0),
      state_server.imu_state.velocity(1),
      state_server.imu_state.velocity(2));
  ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
  ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
  ROS_INFO("initial velocity cov: %f", velocity_cov);
  ROS_INFO("initial extrinsic rotation cov: %f",
      extrinsic_rotation_cov);
  ROS_INFO("initial extrinsic translation cov: %f",
      extrinsic_translation_cov);

  cout << T_imu_cam0.linear() << endl;
  cout << T_imu_cam0.translation().transpose() << endl;

  ROS_INFO("max camera state #: %d", max_cam_state_size);
  ROS_INFO("===========================================");

  ROS_INFO("==================loadParameters OK=========================");

  return true;
}

/**
 * @brief createRosIO 
 *  
 */
bool AckMsckfLam::createRosIO() {
  odom_pub = nh.advertise<nav_msgs::Odometry>("ack_msckf_lm_odom", 10);
  feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
      "feature_point_cloud", 10);
  
  // gt
  gt_init_sub = nh.subscribe("gt", 50,
      &AckMsckfLam::gtInitCallback, this);

  // imu
  imu_sub = nh.subscribe("imu", 200,
  &AckMsckfLam::imuCallback, this);

  // camera features
  feature_sub = nh.subscribe("features", 30,
    &AckMsckfLam::featureCallback, this);
  
  // ackermann
  ack_sub = nh.subscribe("ackermann", 200,
      &AckMsckfLam::ackCallback, this);
  
  // raw_gnss fusion
  gnss_sub = nh.subscribe("raw_gnss", 100, & AckMsckfLam::rawGnssCallback, this);

  // save csv
  csv_timer = nh.createTimer(ros::Duration(1),&AckMsckfLam::csv_timer_callBack,this);

  return true;
}

/**
 * @brief rawGnssCallback 
 *  
 */
void AckMsckfLam::rawGnssCallback(const sensor_msgs::NavSatFix &msg)
{
    // raw_gnss fusion
    if(use_raw_gnss){
      double time = msg.header.stamp.toSec();
      gnss_msg_buffer.push_back(make_pair(time,msg));
    }

}

/**
 * @brief gtInitCallback 
 *  
 */
void AckMsckfLam::gtInitCallback(
    const nav_msgs::OdometryConstPtr& msg) {

  gt_msg_buffer.push_back(*msg);

  // If this is the first_gt_init_msg, set
  if (!is_gt_init_set) {
    Quaterniond orientation;
    Vector3d translation;
    tf::pointMsgToEigen(
        msg->pose.pose.position, translation);
    tf::quaternionMsgToEigen(
        msg->pose.pose.orientation, orientation);
    T_B0_GS_gt.linear() = orientation.toRotationMatrix();
    T_B0_GS_gt.translation() = translation;
    is_gt_init_set = true;
    double roll,pitch,yaw;
    tf::Matrix3x3(tf::Quaternion(orientation.x(),orientation.y(),orientation.z(),orientation.w())).getRPY(roll,pitch,yaw,1);    
    std::cout<< "ack_msckf_lm gt_q_B0_GB : roll="<< roll * 180 / M_PI <<",　pitch="<<pitch * 180 / M_PI <<",　yaw="<< yaw * 180 / M_PI << std::endl;
    std::cout<< "ack_msckf_lm gt_t_B0_GB : x="<< translation(0) <<", y="<< translation(1) <<", z="<< translation(2) << std::endl;
  
    gt_odom_last = *msg;
    gt_odom_curr = gt_odom_last;   
    t_GBi_GS_last = T_B0_GS_gt.translation();
  }
  
  return;
}

/**
 * @brief ackCallback 
 *  
 */
void AckMsckfLam::ackCallback(const ack_msckf_lam::AckermannDriveStamped::ConstPtr& msg){
  ack_msg_buffer.push_back(*msg);
  return;
}

/**
 * @brief gatingTest 
 *  
 */
bool AckMsckfLam::gatingTest(
    const MatrixXd& H, const VectorXd& r, const int& dof) {

  MatrixXd P1 = H * state_server.state_cov * H.transpose();
  MatrixXd P2 = Feature::observation_noise *
    MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

  if (gamma < chi_squared_test_table[dof]) {
        return true;
  } else {
        return false;
  }

}

/**
 * @brief findRedundantCamStates 
 *  
 */
void AckMsckfLam::findRedundantCamStates(
    vector<StateIDType>& rm_cam_state_ids) {

  // Move the iterator to the key position.
  auto key_cam_state_iter = state_server.cam_states.end();
  for (int i = 0; i < 4; ++i)//4
    --key_cam_state_iter;
  auto cam_state_iter = key_cam_state_iter;
  ++cam_state_iter;
  auto first_cam_state_iter = state_server.cam_states.begin();

  // Pose of the key camera state.
  const Vector3d key_position =
    key_cam_state_iter->second.position;
  const Matrix3d key_rotation = quaternionToRotation(
      key_cam_state_iter->second.orientation);

  // Mark the camera states to be removed based on the
  // motion between states.
  for (int i = 0; i < 2; ++i) {
    const Vector3d position =
      cam_state_iter->second.position;
    const Matrix3d rotation = quaternionToRotation(
        cam_state_iter->second.orientation);

    double distance = (position-key_position).norm();
    double angle = AngleAxisd(
        rotation*key_rotation.transpose()).angle();

    if (angle < rotation_threshold &&
        distance < translation_threshold &&
        tracking_rate > tracking_rate_threshold) {
      rm_cam_state_ids.push_back(cam_state_iter->first);
      ++cam_state_iter;
    } else {
      rm_cam_state_ids.push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
    }
  }

  // Sort the elements in the output vector.
  sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

  return;
}

/**
 * @brief measurementUpdate_gnss 
 *  
 */
void AckMsckfLam::measurementUpdate_gnss(
    const MatrixXd& H, const VectorXd& r,const Eigen::MatrixXd &noise) {

  if (H.rows() == 0 || r.rows() == 0) return;

  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
    r_thin = r_temp.head(21+state_server.cam_states.size()*6);

  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  // K = P * H_thin^T * (H_thin*P*H_thin^T + Rn)^-1
  // K * (H_thin*P*H_thin^T + Rn) = P * H_thin^T
  // -> (H_thin*P*H_thin^T + Rn)^T * K^T = H_thin * P^T
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() + noise;
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // delta_x_imu
  const VectorXd& delta_x_imu = delta_x.head<21>();

  if (
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      delta_x_imu.segment<3>(3).norm() > 1.0) {
      printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
      printf("delta position: %f\n", delta_x_imu.segment<3>(3).norm());
      ROS_WARN("Update change is too large.");
  }

  // update imu_state
  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);
  state_server.imu_state.position += delta_x_imu.segment<3>(3);
  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(15);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(18);
  
  // update extrinsic
  const Vector4d dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(9));
  state_server.imu_state.R_imu_body = quaternionToRotation(
      dq_extrinsic) * state_server.imu_state.R_imu_body;
  state_server.imu_state.t_body_imu += delta_x_imu.segment<3>(12);
  
  state_server.imu_state.T_body_imu.linear() = state_server.imu_state.R_imu_body.inverse();
  state_server.imu_state.T_body_imu.translation() = state_server.imu_state.t_body_imu;
  state_server.imu_state.T_imu_body = state_server.imu_state.T_body_imu.inverse();
  state_server.imu_state.t_imu_body = state_server.imu_state.T_imu_body.translation();
  // cam
  state_server.ack_state.T_body_cam0 = state_server.imu_state.T_imu_cam0 * state_server.imu_state.T_body_imu;
  state_server.ack_state.R_body_cam0 = state_server.ack_state.T_body_cam0.linear();
  state_server.ack_state.t_body_cam0 = state_server.ack_state.T_body_cam0.translation();
  state_server.ack_state.t_cam0_body = state_server.ack_state.T_body_cam0.inverse().translation();

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
    const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq_cam, cam_state_iter->second.orientation);
    cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

/**
 * @brief measurementUpdate_ack 
 *  
 */
void AckMsckfLam::measurementUpdate_ack(
    const MatrixXd& H, const VectorXd& r,const Eigen::MatrixXd &noise) {

  if (H.rows() == 0 || r.rows() == 0) return;

  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
    r_thin = r_temp.head(21+state_server.cam_states.size()*6);

  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  // K = P * H_thin^T * (H_thin*P*H_thin^T + Rn)^-1
  // K * (H_thin*P*H_thin^T + Rn) = P * H_thin^T
  // -> (H_thin*P*H_thin^T + Rn)^T * K^T = H_thin * P^T
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() + noise;
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // delta_x_imu
  const VectorXd& delta_x_imu = delta_x.head<21>();

  if (
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      delta_x_imu.segment<3>(3).norm() > 1.0) {
      printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
      printf("delta position: %f\n", delta_x_imu.segment<3>(3).norm());
      ROS_WARN("Update change is too large.");
  }

  // update imu_state
  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);
  state_server.imu_state.position += delta_x_imu.segment<3>(3);
  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(15);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(18);
  
  // update extrinsic
  const Vector4d dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(9));
  state_server.imu_state.R_imu_body = quaternionToRotation(
      dq_extrinsic) * state_server.imu_state.R_imu_body;
  state_server.imu_state.t_body_imu += delta_x_imu.segment<3>(12);
  
  state_server.imu_state.T_body_imu.linear() = state_server.imu_state.R_imu_body.inverse();
  state_server.imu_state.T_body_imu.translation() = state_server.imu_state.t_body_imu;
  state_server.imu_state.T_imu_body = state_server.imu_state.T_body_imu.inverse();
  state_server.imu_state.t_imu_body = state_server.imu_state.T_imu_body.translation();
  // cam
  state_server.ack_state.T_body_cam0 = state_server.imu_state.T_imu_cam0 * state_server.imu_state.T_body_imu;
  state_server.ack_state.R_body_cam0 = state_server.ack_state.T_body_cam0.linear();
  state_server.ack_state.t_body_cam0 = state_server.ack_state.T_body_cam0.translation();
  state_server.ack_state.t_cam0_body = state_server.ack_state.T_body_cam0.inverse().translation();

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
    const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq_cam, cam_state_iter->second.orientation);
    cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

/**
 * @brief imuCallback 
 *  
 */
void AckMsckfLam::imuCallback(
    const sensor_msgs::ImuConstPtr& msg) {

  imu_msg_buffer.push_back(*msg);

  if (!is_gravity_set) {
    if (imu_msg_buffer.size() < 200) return;
    initializeGravityAndBias();
    is_gravity_set = true;
  }

  return;
}

/**
 * @brief batchImuProcessing 
 *  
 */
void AckMsckfLam::batchImuProcessing(const double& time_bound) {
  // Counter how many IMU msgs in the buffer are used.
  int used_imu_msg_cntr = 0;
  for (const auto& imu_msg : imu_msg_buffer) {
    double imu_time = imu_msg.header.stamp.toSec();
    if (imu_time < state_server.imu_state.time) {
      ++used_imu_msg_cntr;
      continue;
    }
    if (imu_time > time_bound) break;

    // Convert the msgs.
    Vector3d m_gyro, m_acc;
    tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);

    // Execute process model.
    processModel(imu_time, m_gyro, m_acc);
    ++used_imu_msg_cntr;
  }

  // Set the state ID for the new IMU state.
  state_server.imu_state.id = IMUState::next_id++;

  // Remove all used IMU msgs.
  imu_msg_buffer.erase(imu_msg_buffer.begin(),
      imu_msg_buffer.begin()+used_imu_msg_cntr);

  return;
}

/**
 * @brief batchGtProcessing 
 *  
 */
void AckMsckfLam::batchGtProcessing(const double& time_bound) {

  // Counter how many gt msgs in the buffer are used.
  int used_gt_msg_cntr = 0;

  for (const auto& gt_msg : gt_msg_buffer) {
    double gt_time = gt_msg.header.stamp.toSec();
    if (gt_time < state_server.imu_state.gt_time) {
      ++used_gt_msg_cntr;
      continue;
    }
    if (gt_time > time_bound) break;

    // Save the newest gt msg
    gt_odom_curr = gt_msg;
    state_server.imu_state.gt_time = gt_time;

    ++used_gt_msg_cntr;
  }

  // Remove all used gt msgs.
  gt_msg_buffer.erase(gt_msg_buffer.begin(),
      gt_msg_buffer.begin() + used_gt_msg_cntr);

  return;
}

/**
 * @brief processModel 
 *  
 */
void AckMsckfLam::processModel(const double& time,
    const Vector3d& m_gyro,
    const Vector3d& m_acc) {

  // Remove the bias from the measured gyro and acceleration
  IMUState& imu_state = state_server.imu_state;
  Vector3d gyro_i = m_gyro - imu_state.gyro_bias;
  Vector3d acc = m_acc - imu_state.acc_bias;

  double dtime = time - imu_state.time;

  Vector3d gyro_b = imu_state.R_imu_body * gyro_i; 

  // Compute discrete transition and noise covariance matrix
  Matrix<double, 21, 21> F = Matrix<double, 21, 21>::Zero();
  Matrix<double, 21, 12> G = Matrix<double, 21, 12>::Zero();

  // F11
  F.block<3, 3>(0, 0) = -skewSymmetric(gyro_b);

  // F14
  F.block<3, 3>(0, 9) = skewSymmetric(gyro_b);

  // F16
  F.block<3, 3>(0, 15) = - imu_state.R_imu_body;

  // F21
  F.block<3, 3>(3, 0) = - quaternionToRotation(
      imu_state.orientation).transpose() * skewSymmetric(
        imu_state.R_imu_body * skewSymmetric(gyro_i) * imu_state.t_body_imu
      );

  // F23
  F.block<3, 3>(3, 6) = Matrix3d::Identity();

  // F24
  F.block<3, 3>(3, 9) = quaternionToRotation(
      imu_state.orientation).transpose() * skewSymmetric(
        imu_state.R_imu_body * skewSymmetric(gyro_i) * imu_state.t_body_imu
      );

  // F25
  F.block<3, 3>(3, 12) = quaternionToRotation(
      imu_state.orientation).transpose()* 
        imu_state.R_imu_body * skewSymmetric(gyro_i);

  // F26
  F.block<3, 3>(3, 15) = quaternionToRotation(
      imu_state.orientation).transpose()* 
        imu_state.R_imu_body * skewSymmetric(imu_state.t_body_imu);

  // F31
  F.block<3, 3>(6, 0) = - quaternionToRotation(
      imu_state.orientation).transpose() * skewSymmetric(imu_state.R_imu_body * acc);
 
  // F34
  F.block<3, 3>(6, 9) = quaternionToRotation(
      imu_state.orientation).transpose() * skewSymmetric(imu_state.R_imu_body * acc);
  
  // F37
  F.block<3, 3>(6, 18) = - quaternionToRotation(
      imu_state.orientation).transpose() * imu_state.R_imu_body;

  // G11
  G.block<3, 3>(0, 0) = - imu_state.R_imu_body;
  
  // G21
  G.block<3, 3>(3, 0) = quaternionToRotation(
      imu_state.orientation).transpose() * 
        imu_state.R_imu_body * skewSymmetric(imu_state.t_body_imu);
  
  // G33
  G.block<3, 3>(6, 6) = - quaternionToRotation(
      imu_state.orientation).transpose() * imu_state.R_imu_body;
  
  // G62
  G.block<3, 3>(15, 3) = Matrix3d::Identity();
  
  // G74
  G.block<3, 3>(18, 9) = Matrix3d::Identity();

  // Approximate matrix exponential to the 3rd order
  Matrix<double, 21, 21> Fdt = F * dtime;
  Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
  Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
  Matrix<double, 21, 21> Phi = Matrix<double, 21, 21>::Identity() +
    Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

  // Propogate the state using 4th order Runge-Kutta
  predictNewState(dtime, gyro_b, gyro_i, acc);

  // Propogate the state covariance matrix.
  Matrix<double, 21, 21> Q = Phi*G*state_server.continuous_noise_cov*
    G.transpose()*Phi.transpose()*dtime;
  state_server.state_cov.block<21, 21>(0, 0) =
    Phi*state_server.state_cov.block<21, 21>(0, 0)*Phi.transpose() + Q;

  //             [ P_I_I(k|k)      P_I_C(k|k)]
  // P_k_k(-)  = [                           ]
  //             [ P_I_C(k|k).T    P_C_C(k|k)]
  //          [ P_I_I(k+1|k)    Φ * P_I_C(k|k)]
  // P_k_k  = [                           ]
  //          [ P_I_C(k|k).T * Φ.T  P_C_C(k|k)]
  if (state_server.cam_states.size() > 0) {
    state_server.state_cov.block(
        0, 21, 21, state_server.state_cov.cols()-21) =
      Phi * state_server.state_cov.block(
        0, 21, 21, state_server.state_cov.cols()-21);
    state_server.state_cov.block(
        21, 0, state_server.state_cov.rows()-21, 21) =
      state_server.state_cov.block(
        21, 0, state_server.state_cov.rows()-21, 21) * Phi.transpose();
  }

  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  // Update the state info
  state_server.imu_state.time = time;
  return;
}

/**
 * @brief predictNewState 
 *  
 */
void AckMsckfLam::predictNewState(const double& dt,
    const Vector3d& gyro_b,
    const Vector3d& gyro_i,
    const Vector3d& acc) {

  double gyro_norm = gyro_b.norm();
  Matrix4d Omega = Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro_b);
  Omega.block<3, 1>(0, 3) = gyro_b;
  Omega.block<1, 3>(3, 0) = -gyro_b.transpose();

  Vector4d& q = state_server.imu_state.orientation;
  Vector3d& v = state_server.imu_state.velocity;
  Vector3d& p = state_server.imu_state.position;

  Vector3d v_r = quaternionToRotation(q).transpose() * state_server.imu_state.R_imu_body
    * skewSymmetric(gyro_i) * state_server.imu_state.t_body_imu;

  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
  }
  else {
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
      cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
      cos(gyro_norm*dt*0.25) * q;
  }
  Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
  Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

  // k1 = f(tn, yn)
  // v： k1 = f(tn, yn) = R（tn）*a + g
  // p： k1 = f(tn, pn) = v
  Vector3d k1_v_dot = quaternionToRotation(q).transpose() * state_server.imu_state.R_imu_body * acc +
     IMUState::gravity;
  Vector3d k1_p_dot = v + v_r;

  // k2 = f(tn+dt/2, yn+k1*dt/2) 
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose * state_server.imu_state.R_imu_body * acc +
     IMUState::gravity;
  Vector3d k1_v_r = dR_dt2_transpose * state_server.imu_state.R_imu_body
    * skewSymmetric(gyro_i) * state_server.imu_state.t_body_imu;
  Vector3d k2_p_dot = k1_v + k1_v_r;

  // k3 = f(tn+dt/2, yn+k2*dt/2)  
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose * state_server.imu_state.R_imu_body * acc +
     IMUState::gravity;
  Vector3d k2_v_r = dR_dt2_transpose * state_server.imu_state.R_imu_body
    * skewSymmetric(gyro_i) * state_server.imu_state.t_body_imu;
  Vector3d k3_p_dot = k2_v + k2_v_r;

  // k4 = f(tn+dt, yn+k3*dt)  
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose * state_server.imu_state.R_imu_body * acc +
     IMUState::gravity;
  Vector3d k3_v_r = dR_dt_transpose * state_server.imu_state.R_imu_body
    * skewSymmetric(gyro_i) * state_server.imu_state.t_body_imu;
  Vector3d k4_p_dot = k3_v + k3_v_r;

  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = dq_dt;
  quaternionNormalize(q);
  v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

  state_server.imu_state.angular_velocity = gyro_i;
  return;
}

/**
 * @brief stateAugmentation 
 *  
 */
void AckMsckfLam::stateAugmentation(const double& time) {

  const Matrix3d& R_b_c = state_server.ack_state.R_body_cam0;
  const Vector3d& t_c_b = state_server.ack_state.t_cam0_body;

  const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;

  const Matrix3d& R_i_b = state_server.imu_state.R_imu_body;
  const Vector3d& t_b_i = state_server.imu_state.t_body_imu;

  // Add a new camera state to the state server.
  Matrix3d R_w_b = quaternionToRotation(
      state_server.imu_state.orientation);
  Matrix3d R_w_c = R_b_c * R_w_b;
  Vector3d t_c_w = state_server.imu_state.position +
    R_w_b.transpose() * t_c_b;

  state_server.cam_states[state_server.imu_state.id] =
    CAMState(state_server.imu_state.id);
  CAMState& cam_state = state_server.cam_states[
    state_server.imu_state.id];

  cam_state.time = time;
  cam_state.orientation = rotationToQuaternion(R_w_c);
  cam_state.position = t_c_w;

  Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
  J.block<3, 3>(0, 0) = R_b_c;
  J.block<3, 3>(0, 9) = - R_b_c;

  J.block<3, 3>(3, 0) = - R_w_b.transpose() * skewSymmetric(t_c_b);
  J.block<3, 3>(3, 3) = Matrix3d::Identity();
  J.block<3, 3>(3, 9) = R_w_b.transpose() * skewSymmetric(t_c_b);
  J.block<3, 3>(3, 12) = - R_w_b.transpose() * R_i_b ;

  // Resize the state covariance matrix.
  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();
  state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Rename some matrix blocks for convenience.
  const Matrix<double, 21, 21>& P11 =
    state_server.state_cov.block<21, 21>(0, 0);
  const MatrixXd& P12 =
    state_server.state_cov.block(0, 21, 21, old_cols-21);

  // Fill in the augmented state covariance.
  //      [ I(21+6N) ]          [ I(21+6N) ]^T
  //  P = [          ] P11 P12  [          ]
  //      [    J J0  ] P21 P22  [    J J0  ]
  //
  state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
  state_server.state_cov.block(0, old_cols, old_rows, 6) =
    state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
  state_server.state_cov.block<6, 6>(old_rows, old_cols) =
    J * P11 * J.transpose();

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

/**
 * @brief addFeatureObservations 
 *  
 */
void AckMsckfLam::addFeatureObservations(
    const CameraMeasurementConstPtr& msg) {

  StateIDType state_id = state_server.imu_state.id;
  int curr_feature_num = map_server.size();
  int tracked_feature_num = 0;

  for (const auto& feature : msg->features) {
    if (map_server.find(feature.id) == map_server.end()) {
      map_server[feature.id] = Feature(feature.id);
      map_server[feature.id].observations[state_id] =
        Vector2d(feature.u0, feature.v0);
    } else {
      map_server[feature.id].observations[state_id] =
        Vector2d(feature.u0, feature.v0);
      ++tracked_feature_num;
    }
  }

  tracking_rate =
    static_cast<double>(tracked_feature_num) /
    static_cast<double>(curr_feature_num);

  return;
}

/**
 * @brief measurementJacobian 
 *  camera features 
 */ 
void AckMsckfLam::measurementJacobian(
    const StateIDType& cam_state_id,
    const FeatureIDType& feature_id,
    Matrix<double, 2, 6>& H_x, Matrix<double, 2, 3>& H_f, Vector2d& r) {
  
  // Prepare all the required data.
  const CAMState& cam_state = state_server.cam_states[cam_state_id];
  const Feature& feature = map_server[feature_id];

  // Cam0 pose. 
  Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
  const Vector3d& t_c0_w = cam_state.position;

  // 3d feature position in the world frame.
  const Vector3d& p_w = feature.position;
  const Vector2d& z = feature.observations.find(cam_state_id)->second;

  // Convert the feature position from the world frame to
  // the cam0 frame.
  Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
  
  // Compute the Jacobians.
  // Hc = z'/cp' * cp'/x'  Hf  = z'/ cp' * cp'/gp'
  // z'/cp'
  Matrix<double, 2, 3> dz_dpc0 = Matrix<double, 2, 3>::Zero();
  dz_dpc0(0, 0) = 1 / p_c0(2);
  dz_dpc0(1, 1) = 1 / p_c0(2);
  dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
  dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

  // cp'/x'
  Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
  dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
  dpc0_dxc.rightCols(3) = -R_w_c0;

  Matrix3d dpc0_dpg = R_w_c0;

  H_x = dz_dpc0*dpc0_dxc;
  H_f = dz_dpc0*dpc0_dpg;

  // Compute the residual.
  r = z - Vector2d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2));

  return;
}

/**
 * @brief measurementUpdate 
 *  camera features 
 */ 
void AckMsckfLam::measurementUpdate(
    const MatrixXd& H, const VectorXd& r) {

  if (H.rows() == 0 || r.rows() == 0) return;

  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
    r_thin = r_temp.head(21+state_server.cam_states.size()*6);
  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  // K = P * H_thin^T * (H_thin*P*H_thin^T + Rn)^-1
  // K * (H_thin*P*H_thin^T + Rn) = P * H_thin^T
  // -> (H_thin*P*H_thin^T + Rn)^T * K^T = H_thin * P^T
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() +
      Feature::observation_noise*MatrixXd::Identity(
        H_thin.rows(), H_thin.rows());
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // delta_x_imu
  const VectorXd& delta_x_imu = delta_x.head<21>();

  if (
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      delta_x_imu.segment<3>(3).norm() > 1.0) {
      printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
      printf("delta position: %f\n", delta_x_imu.segment<3>(3).norm());
      ROS_WARN("Update change is too large.");
  }
  
  // Update imu_state
  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);
  state_server.imu_state.position += delta_x_imu.segment<3>(3);
  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(15);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(18);

  // update extrinsic
  const Vector4d dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(9));
  state_server.imu_state.R_imu_body = quaternionToRotation(
      dq_extrinsic) * state_server.imu_state.R_imu_body;
  state_server.imu_state.t_body_imu += delta_x_imu.segment<3>(12);

  state_server.imu_state.T_body_imu.linear() = state_server.imu_state.R_imu_body.inverse();
  state_server.imu_state.T_body_imu.translation() = state_server.imu_state.t_body_imu;
  state_server.imu_state.T_imu_body = state_server.imu_state.T_body_imu.inverse();
  state_server.imu_state.t_imu_body = state_server.imu_state.T_imu_body.translation();
  state_server.ack_state.T_body_cam0 = state_server.imu_state.T_imu_cam0 * state_server.imu_state.T_body_imu;
  state_server.ack_state.R_body_cam0 = state_server.ack_state.T_body_cam0.linear();
  state_server.ack_state.t_body_cam0 = state_server.ack_state.T_body_cam0.translation();

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
    const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq_cam, cam_state_iter->second.orientation);
    cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

/**
 * @brief removeLostFeatures 
 *  
 */ 
void AckMsckfLam::removeLostFeatures() {

  int jacobian_row_size = 0;
  vector<FeatureIDType> invalid_feature_ids(0);
  vector<FeatureIDType> processed_feature_ids(0);

  for (auto iter = map_server.begin();
      iter != map_server.end(); ++iter) {
    auto& feature = iter->second;

    if (feature.observations.find(state_server.imu_state.id) !=
        feature.observations.end()) continue;
    if (feature.observations.size() < 3) {
      invalid_feature_ids.push_back(feature.id);
      continue;
    }

    if (!feature.is_initialized) {
      if (!feature.checkMotion(state_server.cam_states)) {
        invalid_feature_ids.push_back(feature.id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          invalid_feature_ids.push_back(feature.id);
          continue;
        }
      }
    }

    jacobian_row_size += 2*feature.observations.size() - 3;//4
    processed_feature_ids.push_back(feature.id);
  }

  // Remove the features that do not have enough measurements.
  for (const auto& feature_id : invalid_feature_ids)
    map_server.erase(feature_id);

  // Return if there is no lost feature to be processed.
  if (processed_feature_ids.size() == 0) return;

  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  // Process the features which lose track.
  for (const auto& feature_id : processed_feature_ids) {
    auto& feature = map_server[feature_id];

    vector<StateIDType> cam_state_ids(0);
    for (const auto& measurement : feature.observations)
      cam_state_ids.push_back(measurement.first);

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    if (stack_cntr > 1500) break;
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform the measurement update step.
  measurementUpdate(H_x, r);

  // Remove all processed features from the map.
  for (const auto& feature_id : processed_feature_ids)
    map_server.erase(feature_id);

  return;
}

/**
 * @brief pruneCamStateBuffer 
 *  
 */ 
void AckMsckfLam::pruneCamStateBuffer() {

  if (state_server.cam_states.size() < max_cam_state_size)
    return;

  // Find two camera states to be removed.
  vector<StateIDType> rm_cam_state_ids(0);
  findRedundantCamStates(rm_cam_state_ids);

  int jacobian_row_size = 0;
  for (auto& item : map_server) {
    auto& feature = item.second;
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }
    if (involved_cam_state_ids.size() == 0) continue;
    if (involved_cam_state_ids.size() == 1) {
      feature.observations.erase(involved_cam_state_ids[0]);
      continue;
    }

    if (!feature.is_initialized) {
      // Check if the feature can be initialize.
      if (!feature.checkMotion(state_server.cam_states)) {
        for (const auto& cam_id : involved_cam_state_ids)
          feature.observations.erase(cam_id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
          continue;
        }
      }
    }

    jacobian_row_size += 2*involved_cam_state_ids.size() - 3;//4
  }

  // Compute the Jacobian and residual.
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (auto& item : map_server) {
    auto& feature = item.second;
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    for (const auto& cam_id : involved_cam_state_ids)
      feature.observations.erase(cam_id);
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform measurement update.
  measurementUpdate(H_x, r);
  
  for (const auto& cam_id : rm_cam_state_ids) {
    int cam_sequence = std::distance(state_server.cam_states.begin(),
        state_server.cam_states.find(cam_id));
    int cam_state_start = 21 + 6*cam_sequence;
    int cam_state_end = cam_state_start + 6;

    // Remove the corresponding rows and columns in the state
    // covariance matrix.
    if (cam_state_end < state_server.state_cov.rows()) {
      state_server.state_cov.block(cam_state_start, 0,
          state_server.state_cov.rows()-cam_state_end,
          state_server.state_cov.cols()) =
        state_server.state_cov.block(cam_state_end, 0,
            state_server.state_cov.rows()-cam_state_end,
            state_server.state_cov.cols());

      state_server.state_cov.block(0, cam_state_start,
          state_server.state_cov.rows(),
          state_server.state_cov.cols()-cam_state_end) =
        state_server.state_cov.block(0, cam_state_end,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-cam_state_end);

      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    } else {
      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    }

    // Remove this camera state in the state vector.
    state_server.cam_states.erase(cam_id);
  }

  return;
}

/**
 * @brief onlineReset 
 *  
 */ 
void AckMsckfLam::onlineReset() {

  // Never perform online reset if position std threshold
  // is non-positive.
  if (position_std_threshold <= 0) return;
  static long long int online_reset_counter = 0;

  // Check the uncertainty of positions to determine if
  // the system can be reset.
  double position_x_std = std::sqrt(state_server.state_cov(3, 3));
  double position_y_std = std::sqrt(state_server.state_cov(4, 4));
  double position_z_std = std::sqrt(state_server.state_cov(5, 5));

  if (position_x_std < position_std_threshold &&
      position_y_std < position_std_threshold &&
      position_z_std < position_std_threshold) return;

  ROS_WARN("Start %lld online reset procedure...",
      ++online_reset_counter);
  ROS_INFO("Stardard deviation in xyz: %f, %f, %f",
      position_x_std, position_y_std, position_z_std);

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Clear all exsiting features in the map.
  map_server.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);

  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 12; i < 15; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;

  ROS_WARN("%lld online reset complete...", online_reset_counter);
  return;
}

/**
 * @brief publish 
 *  
 */ 
void AckMsckfLam::publish(const ros::Time& time) {

  const IMUState& imu_state = state_server.imu_state;

  // T_Bi_GBI　
  Eigen::Isometry3d T_Bi_GBI = Eigen::Isometry3d::Identity();
  T_Bi_GBI.linear() = quaternionToRotation(
      imu_state.orientation).transpose();
  T_Bi_GBI.translation() = imu_state.position;

  Eigen::Isometry3d T_GBI_B0 = T_B0_GBI.inverse();
  Eigen::Isometry3d T_GBI_W = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_GBI_GS = Eigen::Isometry3d::Identity();

  T_GBI_W = T_GBI_B0;
  T_GBI_GS = T_B0_GS_gt * T_GBI_B0;
  
  ////////////////////////////////////////////////////////////////////////////////
  // {EST and GT}
  Eigen::Quaterniond q_Bi_W;
  Vector3d t_Bi_W;
  Eigen::Quaterniond q_GBi_GS;
  Vector3d t_GBi_GS;
  Eigen::Quaterniond q_Bi_W_gt;
  Vector3d t_Bi_W_gt;
  Eigen::Quaterniond q_GBi_GS_gt;
  Vector3d t_GBi_GS_gt;

  // {EST}
  // {W}
  Eigen::Isometry3d T_Bi_W = T_GBI_W * T_Bi_GBI;
  q_Bi_W = T_Bi_W.linear();
  t_Bi_W = T_Bi_W.translation();
  // {G}
  Eigen::Isometry3d T_GBi_GS = T_GBI_GS * T_Bi_GBI;
  q_GBi_GS = T_GBi_GS.linear();
  t_GBi_GS = T_GBi_GS.translation();
  
  // {GT} 
  Eigen::Quaterniond gt_orientation;
  Eigen::Vector3d gt_translation;
  Eigen::Vector3d gt_translation_last;
  Eigen::Vector3d gt_bv;
  Eigen::Vector3d gt_bw;
  tf::pointMsgToEigen(
      gt_odom_last.pose.pose.position, gt_translation_last);
  tf::pointMsgToEigen(
      gt_odom_curr.pose.pose.position, gt_translation);
  tf::quaternionMsgToEigen(
      gt_odom_curr.pose.pose.orientation, gt_orientation);
  tf::vectorMsgToEigen(gt_odom_curr.twist.twist.linear,gt_bv);
  tf::vectorMsgToEigen(gt_odom_curr.twist.twist.angular,gt_bw);
  // {G}
  Eigen::Isometry3d gt_T_Bi_GS = Eigen::Isometry3d::Identity();
  gt_T_Bi_GS.linear() = gt_orientation.toRotationMatrix();
  gt_T_Bi_GS.translation() = gt_translation;
  q_GBi_GS_gt = gt_T_Bi_GS.linear();
  t_GBi_GS_gt = gt_T_Bi_GS.translation();
  // {W}
  Eigen::Isometry3d gt_T_Bi_W = Eigen::Isometry3d::Identity();
  gt_T_Bi_W = T_B0_GS_gt.inverse() * gt_T_Bi_GS;
  q_Bi_W_gt = gt_T_Bi_W.linear();
  t_Bi_W_gt = gt_T_Bi_W.translation();

  // save pose {W}
  Eigen::Vector3d p_wi = t_Bi_W;
  Eigen::Quaterniond q_wi = q_Bi_W;
  Eigen::Vector3d gt_p_wi = t_Bi_W_gt;
  Eigen::Quaterniond gt_q_wi =  q_Bi_W_gt;

  // EST roll pitch yaw
  double roll_GBi_GS, pitch_GBi_GS, yaw_GBi_GS;
  tf::Matrix3x3(tf::Quaternion(q_GBi_GS.x(),q_GBi_GS.y(),q_GBi_GS.z(),q_GBi_GS.w())).getRPY(roll_GBi_GS, pitch_GBi_GS, yaw_GBi_GS, 1); 
  roll_GBi_GS = roll_GBi_GS * 180 / M_PI;
  pitch_GBi_GS = pitch_GBi_GS * 180 / M_PI;
  yaw_GBi_GS = yaw_GBi_GS * 180 / M_PI;
  if(use_debug){
    std::cout.precision(16);    
    std::cout<<"ack_msckf_lm EST t_GBi_GS : x="<< t_GBi_GS[0] <<" m,　y="<< t_GBi_GS[1] <<" m,　z="<< t_GBi_GS[2] <<" m" << std::endl;
    std::cout<<"ack_msckf_lm EST q_GBi_GS : roll="<< roll_GBi_GS <<" deg,　pitch="<< pitch_GBi_GS <<" deg,　yaw_="<< yaw_GBi_GS <<" deg" << std::endl;
    std::cout << std::endl;
  }

   // GT roll pitch yaw
  double roll_GBi_GS_gt, pitch_GBi_GS_gt, yaw_GBi_GS_gt;
  tf::Matrix3x3(tf::Quaternion(q_GBi_GS_gt.x(),q_GBi_GS_gt.y(),q_GBi_GS_gt.z(),q_GBi_GS_gt.w())).getRPY(roll_GBi_GS_gt, pitch_GBi_GS_gt, yaw_GBi_GS_gt, 1);
  roll_GBi_GS_gt = roll_GBi_GS_gt * 180 / M_PI;
  pitch_GBi_GS_gt = pitch_GBi_GS_gt * 180 / M_PI;
  yaw_GBi_GS_gt = yaw_GBi_GS_gt * 180 / M_PI;
  if(use_debug){
    std::cout.precision(16);    
    std::cout<<"ack_msckf_lm GT gt_orientation : roll="<< roll_GBi_GS_gt <<" deg,　pitch="<< pitch_GBi_GS_gt <<" deg,　yaw_="<< yaw_GBi_GS_gt <<" deg" << std::endl;
    std::cout << std::endl;
  }
////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
  // Publish tf {W}
  if (publish_tf) {
    tf::Transform T_Bi_W_tf;
    tf::transformEigenToTF(T_Bi_W, T_Bi_W_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_Bi_W_tf, time, fixed_frame_id, child_frame_id));
  }
  // Publish the odometry
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = time;
  odom_msg.header.frame_id = fixed_frame_id;
  odom_msg.child_frame_id = child_frame_id;

  Eigen::Matrix3d R_Bi_GBI = T_Bi_GBI.linear();
  Eigen::Matrix3d R_GBI_Bi = R_Bi_GBI.transpose();
  // lever arm 
  // bv
  const Vector3d& w_G_i = state_server.imu_state.angular_velocity;
  Eigen::Vector3d b_v = R_GBI_Bi * imu_state.velocity + imu_state.T_imu_body.linear() * skewSymmetric(w_G_i) * (imu_state.T_imu_body.inverse().translation());
  Eigen::Vector3d b_v_la = imu_state.T_imu_body.linear() * skewSymmetric(w_G_i) * (imu_state.T_imu_body.inverse().translation());
  // bw
  Eigen::Vector3d b_w = imu_state.T_imu_body.linear() * w_G_i;
  tf::poseEigenToMsg(T_Bi_W, odom_msg.pose.pose);
  tf::vectorEigenToMsg(b_v, odom_msg.twist.twist.linear);
  tf::vectorEigenToMsg(b_w, odom_msg.twist.twist.angular);

  // Convert the covariance.
  Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
  Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 3);
  Matrix3d P_po = state_server.state_cov.block<3, 3>(3, 0);
  Matrix3d P_pp = state_server.state_cov.block<3, 3>(3, 3);
  Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
  P_imu_pose << P_pp, P_po, P_op, P_oo;
  Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
  H_pose.block<3, 3>(0, 0) =  T_GBI_W.linear();
  H_pose.block<3, 3>(3, 3) =  T_GBI_W.linear();
  Matrix<double, 6, 6> P_body_pose = H_pose *
    P_imu_pose * H_pose.transpose();
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);
  // Construct the covariance for the velocity.
  Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
  Matrix3d H_vel = R_GBI_Bi;
  Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);
  odom_pub.publish(odom_msg);

  // Publish the 3D positions of the features {W}
  pcl::PointCloud<pcl::PointXYZ>::Ptr feature_msg_ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  feature_msg_ptr->header.frame_id = fixed_frame_id;
  feature_msg_ptr->height = 1;
  for (const auto& item : map_server) {
    const auto& feature = item.second;
    if (feature.is_initialized) {

      Vector3d feature_position =
         T_GBI_W.linear() * feature.position + T_GBI_W.translation();

      feature_msg_ptr->points.push_back(pcl::PointXYZ(
            feature_position(0), feature_position(1), feature_position(2)));
    }
  }
  feature_msg_ptr->width = feature_msg_ptr->points.size();
  feature_pub.publish(feature_msg_ptr);
//////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////
  // save
  double dStamp = time.toSec();

  // GT
  Eigen::Vector3d gt_GIv;
  gt_GIv = R_GBI_Bi.transpose() * (gt_bv - imu_state.T_imu_body.linear() * skewSymmetric(w_G_i) * (imu_state.T_imu_body.inverse().translation()));

  // save odom
  double scale_ratio;
  double delta_t_est = (t_GBi_GS - t_GBi_GS_last).norm();
  double delta_t_gt = (gt_translation - gt_translation_last).norm();
  double delta_t_est_gt = delta_t_est - delta_t_gt;
  t_GBi_GS_last = t_GBi_GS;
  gt_odom_last = gt_odom_curr;
  Eigen::Quaterniond qib = Eigen::Quaterniond(state_server.imu_state.R_imu_body);
  Eigen::Vector3d pib = state_server.imu_state.t_imu_body;
  double roll_ibx, pitch_iby, yaw_ibz;
  tf::Matrix3x3(tf::Quaternion(qib.x(),qib.y(),qib.z(),qib.w())).getRPY(roll_ibx, pitch_iby, yaw_ibz, 1);   
  roll_ibx = roll_ibx * 180 / M_PI;
  pitch_iby = pitch_iby * 180 / M_PI;
  yaw_ibz = yaw_ibz * 180 / M_PI;
  if(is_first_sr){
    is_first_sr = false;
    scale_ratio = 0;
  }else{
    if(delta_t_est <= 0.01 && abs(delta_t_est_gt) <= 0.03){
      scale_ratio = 0;
    }else if(delta_t_gt <= 0.01 && abs(delta_t_est_gt) <= 0.03){
      scale_ratio = 0;
    }else if(delta_t_est <= 0.01 && abs(delta_t_est_gt) > 0.03){
        scale_ratio = -(delta_t_gt / (delta_t_est + 0.03) - 1);
    }else if(delta_t_gt <= 0.01 && abs(delta_t_est_gt) > 0.03){
        scale_ratio = delta_t_est / (delta_t_gt + 0.03) - 1;
    }else{
      if(delta_t_est_gt >=0){
        scale_ratio = delta_t_est / delta_t_gt - 1;
      }else{
        scale_ratio = -(delta_t_gt / delta_t_est - 1);
      }
    }
  }  
  global_count++;
  global_scale_ratio += scale_ratio;

  CSVDATA_ODOM csvdata_odom;
  csvdata_odom.time = dStamp;
  csvdata_odom.Dtime = Dtime;
  csvdata_odom.pB = p_wi;
  csvdata_odom.qB = q_wi;
  csvdata_odom.roll = roll_GBi_GS;
  csvdata_odom.pitch = pitch_GBi_GS;
  csvdata_odom.yaw = yaw_GBi_GS;
  csvdata_odom.vB = b_v;
  csvdata_odom.wB = b_w;
  csvdata_odom.gt_pB = gt_p_wi;
  csvdata_odom.gt_qB = gt_q_wi;
  csvdata_odom.gt_roll = roll_GBi_GS_gt;
  csvdata_odom.gt_pitch = pitch_GBi_GS_gt;
  csvdata_odom.gt_yaw = yaw_GBi_GS_gt;
  csvdata_odom.gt_vB = gt_bv;
  csvdata_odom.gt_wB = gt_bw;
  csvdata_odom.Sr = scale_ratio;
  csvdata_odom.Sr_avg = global_scale_ratio / global_count;
  csvdata_odom.pIBs = pib;
  csvdata_odom.qIBs = qib;
  csvdata_odom.roll_ibx = roll_ibx;
  csvdata_odom.pitch_iby = pitch_iby;
  csvdata_odom.yaw_ibz = yaw_ibz;
  csvdata_odom.b_v_la = b_v_la;
  csvData_odom.push_back(csvdata_odom);

  // save std {GB}
  Eigen::Vector3d err_p = gt_translation - t_GBi_GS; 
  Eigen::Vector3d err_v_GIv = gt_GIv - imu_state.velocity;
  // r err in {GB}
  Eigen::Quaterniond q_err_r_GBi_GB =  (q_GBi_GS.inverse()) * gt_orientation;
  double roll_err_r_GBI, pitch_err_r_GBI, yaw_err_r_GBI;
  tf::Matrix3x3(tf::Quaternion(q_err_r_GBi_GB.x(),q_err_r_GBi_GB.y(),q_err_r_GBi_GB.z(),q_err_r_GBi_GB.w())).getRPY(roll_err_r_GBI, pitch_err_r_GBI, yaw_err_r_GBI, 1);
  Vector3d err_r_GBI = Vector3d(roll_err_r_GBI,pitch_err_r_GBI,yaw_err_r_GBI);
  double err_rx = err_r_GBI[0] * 180 / M_PI;    
  double err_ry = err_r_GBI[1] * 180 / M_PI;
  double err_rz = err_r_GBI[2] * 180 / M_PI;
  double err_px = err_p(0);
  double err_py = err_p(1);
  double err_pz = err_p(2);
  double err_vx = err_v_GIv(0);
  double err_vy = err_v_GIv(1);
  double err_vz = err_v_GIv(2);
  double err_ribx = 0;
  double err_riby = 0;
  double err_ribz = 0;
  double err_pbix = 0;
  double err_pbiy = 0;
  double err_pbiz = 0;
  double err_bgx = 0;
  double err_bgy = 0;
  double err_bgz = 0; 
  double err_bax = 0;
  double err_bay = 0;
  double err_baz = 0; 
  double std_rx = std::sqrt(state_server.state_cov(0, 0));
  double std_ry = std::sqrt(state_server.state_cov(1, 1));
  double std_rz = std::sqrt(state_server.state_cov(2, 2));
  // r std in {GB}
  Vector3d std_r_GBI_vector = Vector3d(std_rx,std_ry,std_rz);
  Vector4d std_q_GBI =
      smallAngleQuaternion(std_r_GBI_vector);
  quaternionNormalize(std_q_GBI);
  double roll_std_r_GBI, pitch_std_r_GBI, yaw_std_r_GBI;
  tf::Matrix3x3(tf::Quaternion(std_q_GBI.x(),std_q_GBI.y(),std_q_GBI.z(),std_q_GBI.w())).getRPY(roll_std_r_GBI, pitch_std_r_GBI, yaw_std_r_GBI, 1);
  Vector3d std_r_GBI = Vector3d(roll_std_r_GBI,pitch_std_r_GBI,yaw_std_r_GBI);
  std_rx = std_r_GBI[0] * 180 / M_PI;
  std_ry = std_r_GBI[1] * 180 / M_PI;
  std_rz = std_r_GBI[2] * 180 / M_PI;
  if(use_debug){
    std::cout.precision(16);
    std::cout << "-------------------" << std::endl;
    std::cout<<"ack_msckf_lm ERR err_r_GBI : roll="<< err_rx <<" deg,　pitch="<< err_ry <<" deg,　yaw_="<< err_rz <<" deg" << std::endl;
    std::cout<<"ack_msckf_lm STD std_r_GBI : roll="<< std_rx <<" deg,　pitch="<< std_ry <<" deg,　yaw_="<< std_rz <<" deg" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << std::endl;
  }
  double std_px = std::sqrt(state_server.state_cov(3, 3));
  double std_py = std::sqrt(state_server.state_cov(4, 4));
  double std_pz = std::sqrt(state_server.state_cov(5, 5));
  double std_vx = std::sqrt(state_server.state_cov(6, 6));
  double std_vy = std::sqrt(state_server.state_cov(7, 7));
  double std_vz = std::sqrt(state_server.state_cov(8, 8));
  double std_ribx = std::sqrt(state_server.state_cov(9, 9)) * 180 / M_PI;
  double std_riby = std::sqrt(state_server.state_cov(10, 10)) * 180 / M_PI;
  double std_ribz = std::sqrt(state_server.state_cov(11, 11)) * 180 / M_PI;
  double std_pbix = std::sqrt(state_server.state_cov(12,12));
  double std_pbiy = std::sqrt(state_server.state_cov(13, 13));
  double std_pbiz = std::sqrt(state_server.state_cov(14, 14));
  double std_bgx = std::sqrt(state_server.state_cov(15, 15)) * 3600 * 180 / M_PI;
  double std_bgy = std::sqrt(state_server.state_cov(16, 16)) * 3600 * 180 / M_PI;
  double std_bgz = std::sqrt(state_server.state_cov(17, 17)) * 3600 * 180 / M_PI;
  double std_bax = std::sqrt(state_server.state_cov(18, 18));
  double std_bay = std::sqrt(state_server.state_cov(19, 19));
  double std_baz = std::sqrt(state_server.state_cov(20, 20));
  std_rx = abs(std_rx);
  std_ry = abs(std_ry);
  std_rz = abs(std_rz);
  std_px = abs(std_px);
  std_py = abs(std_py);
  std_pz = abs(std_pz);
  std_vx = abs(std_vx);
  std_vy = abs(std_vy);
  std_vz = abs(std_vz);
  std_ribx = abs(std_ribx);
  std_riby = abs(std_riby);
  std_ribz = abs(std_ribz);
  std_pbix = abs(std_pbix);
  std_pbiy = abs(std_pbiy);
  std_pbiz = abs(std_pbiz);
  std_bgx = abs(std_bgx);
  std_bgy = abs(std_bgy);
  std_bgz = abs(std_bgz);
  std_bax = abs(std_bax);
  std_bay = abs(std_bay);
  std_baz = abs(std_baz);
  
  CSVDATA_RMSE csvdata_rmse;
  csvdata_rmse.time = dStamp;
  csvdata_rmse.Dtime = Dtime;
  csvdata_rmse.err_rx = err_rx;
  csvdata_rmse.err_ry = err_ry;
  csvdata_rmse.err_rz = err_rz;
  csvdata_rmse.err_px = err_px;
  csvdata_rmse.err_py = err_py;
  csvdata_rmse.err_pz = err_pz;
  csvdata_rmse.err_vx = err_vx;
  csvdata_rmse.err_vy = err_vy;
  csvdata_rmse.err_vz = err_vz;
  csvdata_rmse.err_ribx = err_ribx;
  csvdata_rmse.err_riby = err_riby;
  csvdata_rmse.err_ribz = err_ribz;
  csvdata_rmse.err_pbix = err_pbix;
  csvdata_rmse.err_pbiy = err_pbiy;
  csvdata_rmse.err_pbiz = err_pbiz;
  csvdata_rmse.err_bgx = err_bgx;
  csvdata_rmse.err_bgy = err_bgy;
  csvdata_rmse.err_bgz = err_bgz;
  csvdata_rmse.err_bax = err_bax;
  csvdata_rmse.err_bay = err_bay;
  csvdata_rmse.err_baz = err_baz;

  csvdata_rmse.std_rx = std_rx;
  csvdata_rmse.std_ry = std_ry;
  csvdata_rmse.std_rz = std_rz;
  csvdata_rmse.std_px = std_px;
  csvdata_rmse.std_py = std_py;
  csvdata_rmse.std_pz = std_pz;
  csvdata_rmse.std_vx = std_vx;
  csvdata_rmse.std_vy = std_vy;
  csvdata_rmse.std_vz = std_vz;
  csvdata_rmse.std_ribx = std_ribx;
  csvdata_rmse.std_riby = std_riby;
  csvdata_rmse.std_ribz = std_ribz;
  csvdata_rmse.std_pbix = std_pbix;
  csvdata_rmse.std_pbiy = std_pbiy;
  csvdata_rmse.std_pbiz = std_pbiz;
  csvdata_rmse.std_bgx = std_bgx;
  csvdata_rmse.std_bgy = std_bgy;
  csvdata_rmse.std_bgz = std_bgz;
  csvdata_rmse.std_bax = std_bax;
  csvdata_rmse.std_bay = std_bay;
  csvdata_rmse.std_baz = std_baz;
  csvData_rmse.push_back(csvdata_rmse);

////////////////////////////////////////////////////////////////////////////////////////////

  return;
}

/**
 * @brief featureCallback 
 *  
 */ 
void AckMsckfLam::featureCallback(
    const CameraMeasurementConstPtr& msg) {

  if (!is_gravity_set) return;
  if (!is_gt_init_set) return;

  {
    // csv_curr_time
    csv_curr_time = ros::Time::now().toSec();
    is_csv_curr_time_init = true;
  }

  // is_first_img
  if (is_first_img) {
    is_first_img = false;
    state_server.imu_state.time = msg->header.stamp.toSec();
    state_server.imu_state.gt_time = msg->header.stamp.toSec();
    state_server.ack_state.time = msg->header.stamp.toSec();
    state_server.ack_state.kaist_time = msg->header.stamp.toSec();

    // save time
    DfirstTime = msg->header.stamp.toSec();
  }
  // total Dtime
  Dtime = msg->header.stamp.toSec() - DfirstTime;

  static double max_processing_time = 0.0;
  static int critical_time_cntr = 0;
  double processing_start_time = ros::Time::now().toSec();
  double processing_time = 0;

  // Publish the odometry.
  ros::Time start_time = ros::Time::now();
  publish(msg->header.stamp);
  double publish_time = (
      ros::Time::now()-start_time).toSec();

  // Propogate the IMU state.
  start_time = ros::Time::now();
  batchImuProcessing(msg->header.stamp.toSec());
  batchGtProcessing(msg->header.stamp.toSec());
  batchAckProcessing(msg->header.stamp.toSec());
  
  // Augment the state vector.
  start_time = ros::Time::now();
  stateAugmentation(msg->header.stamp.toSec());
  double state_augmentation_time = (
      ros::Time::now()-start_time).toSec();

  // Add new observations
  start_time = ros::Time::now();
  addFeatureObservations(msg);
  double add_observations_time = (
      ros::Time::now()-start_time).toSec();

  // Perform measurement update if necessary.
  start_time = ros::Time::now();
  removeLostFeatures();
  double remove_lost_features_time = (
      ros::Time::now()-start_time).toSec();
  start_time = ros::Time::now();
  pruneCamStateBuffer();
  double prune_cam_states_time = (
      ros::Time::now()-start_time).toSec();

  // update Ackermann measurement
  ackUpdate(msg);

  // raw_gnss fusion
  if(is_gnss_aligned){
    gnssUpdate(msg);
  }

  // Reset the system if necessary.
  onlineReset();

  double processing_end_time = ros::Time::now().toSec();
  processing_time =
    processing_end_time - processing_start_time;
  if (processing_time > 1.0/frame_rate) {
    ++critical_time_cntr;
    ROS_INFO("\033[1;31mTotal processing time %f/%d...\033[0m",
        processing_time, critical_time_cntr);
    printf("Remove lost features time: %f/%f\n",
        remove_lost_features_time, remove_lost_features_time/processing_time);
    printf("Remove camera states time: %f/%f\n",
        prune_cam_states_time, prune_cam_states_time/processing_time);
  }
  
  // raw_gnss fusion
  if(use_raw_gnss){
    if(!is_gnss_aligned)
      {
          vio_position_buffer.push_back(make_pair(state_server.imu_state.time, state_server.imu_state.position));
          rawGnssAlign();
      }
  }

  CSVDATA_TIME csvdata_time;
  csvdata_time.time = msg->header.stamp.toSec();
  csvdata_time.Dtime = Dtime;
  csvdata_time.process_time = processing_time * 1000;
  total_time += csvdata_time.process_time;
  csvdata_time.total_time = total_time;
  csvdata_time.avg_time = total_time / global_count;
  csvData_time.push_back(csvdata_time);

  return;
}

/**
 * @brief rawGnssAlign 
 *  
 */ 
void AckMsckfLam::rawGnssAlign()
{ 
    ROS_INFO("TRY RAW GNSS ALIGN!");
    if(gnss_msg_buffer.size()==0 || vio_position_buffer.size()==0 || is_gnss_aligned==true) return ;
    if(gnss_msg_buffer.back().first - last_check_gnss_time < 1.0) return ;
    last_check_gnss_time = gnss_msg_buffer.back().first;
    ROS_INFO("%f",gnss_msg_buffer.back().first - gnss_msg_buffer.front().first);
    if(gnss_msg_buffer.back().first - gnss_msg_buffer.front().first < 5) return;

    while(gnss_msg_buffer.front().first < last_check_gnss_time - 15) gnss_msg_buffer.erase(gnss_msg_buffer.begin());
    while(vio_position_buffer.front().first < last_check_gnss_time - 15) vio_position_buffer.erase(vio_position_buffer.begin());

    // construct GPS/vio trajectories to do ICP
    vector<pair<double,Vector3d>> gnss_pos_buffer(gnss_msg_buffer.size());
    vector<pair<double,Vector3d>> vio_pos_buffer(vio_position_buffer.size());

    int i=0;
    for(auto ite = gnss_msg_buffer.begin() ; ite!=gnss_msg_buffer.end();ite++, i++)
    {
        auto & gnss_msg= ite->second;

        string UTMZone;
        double north_y;
        double east_x;
        // LLtoUTM
        // NE. -> EN.
        NavsatConversions::LLtoUTM(gnss_msg.latitude * 180 / M_PI, gnss_msg.longitude * 180 / M_PI,
                                north_y , east_x,
                                UTMZone);
        // ENU
        gnss_pos_buffer[i]=make_pair(ite->first,
                                    Vector3d(east_x, north_y, gnss_msg.altitude));

    }
    i=0;
    for(auto ite = vio_position_buffer.begin() ; ite!=vio_position_buffer.end();ite++, i++)
    {
        vio_pos_buffer[i]=make_pair(ite->first,ite->second);
    }

    is_gnss_aligned=registration_4DOF(vio_pos_buffer,gnss_pos_buffer,state_server.gnss_state.R_GB_US,state_server.gnss_state.t_GB_US);
    if(is_gnss_aligned)
    {
        double roll_GB_US, pitch_GB_US, yaw_GB_US;
        Eigen::Quaterniond q_GB_US;
        q_GB_US = state_server.gnss_state.R_GB_US;
        tf::Matrix3x3(tf::Quaternion(q_GB_US.x(),q_GB_US.y(),q_GB_US.z(),q_GB_US.w())).getRPY(roll_GB_US, pitch_GB_US, yaw_GB_US, 1);
        roll_GB_US = roll_GB_US * 180 / M_PI;
        pitch_GB_US = pitch_GB_US * 180 / M_PI;
        yaw_GB_US = yaw_GB_US * 180 / M_PI;
        std::cout.precision(16); 
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        ROS_INFO("RANSS ALIGNED SUCCESSFULLY!");
        std::cout << "R_GB_US : roll = " << roll_GB_US << " deg,　pitch = " << pitch_GB_US << " deg,　yaw = " << yaw_GB_US << " deg" << std::endl;
        cout << "t_GB_US: " << state_server.gnss_state.t_GB_US.transpose() << endl;
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    }

}

/**
 * @brief gnssUpdate 
 *  
 */ 
void AckMsckfLam::gnssUpdate(const CameraMeasurementConstPtr& msg){

        while(gnss_msg_buffer.size()!=0 && gnss_msg_buffer.front().first < msg->header.stamp.toSec()){
          gnss_msg_buffer.erase(gnss_msg_buffer.begin());
        }
        
        if(gnss_msg_buffer.size()!=0 )
        {

            auto & gnss_msg =gnss_msg_buffer.front().second;

            string UTMZone;
            double north_y;
            double east_x;
            // LLtoUTM
            // NE. -> EN.
            NavsatConversions::LLtoUTM(gnss_msg.latitude * 180 / M_PI, gnss_msg.longitude * 180 / M_PI,
                                        north_y , east_x,
                                        UTMZone);
            if(use_debug){
              // ENU
              cout.precision(16);
              cout << "---------------------------------------------------" << endl;
              cout << "east_x: " << east_x << endl;
              cout << "north_y: " << north_y << endl;
              cout << "up_z: " << gnss_msg.altitude << endl;
              cout << "state_server.imu_state.position: " << state_server.imu_state.position.transpose() << endl;
              cout << "---------------------------------------------------" << endl;
            }
            
            // ENU
            state_server.gnss_state.position = Vector3d(east_x , north_y, gnss_msg.altitude);

            // construct H and r
            MatrixXd H_x = MatrixXd::Zero(3,21+6*state_server.cam_states.size());
            VectorXd r = VectorXd::Zero(3);

            //-------------------------------------------------
            Eigen::Vector3d t_Bk_GB_est = state_server.imu_state.position;
            Eigen::Matrix3d R_GB_Bk_est = quaternionToRotation(
                state_server.imu_state.orientation);
            Eigen::Matrix3d R_i_b = state_server.imu_state.R_imu_body;
            Eigen::Vector3d t_b_i = state_server.imu_state.t_body_imu;
            Eigen::Matrix3d R_GB_US = state_server.gnss_state.R_GB_US;
            Eigen::Vector3d t_GB_US = state_server.gnss_state.t_GB_US;
            
            //-------------------------------------------------
            Eigen::Vector3d t_Sk_Us_est = Vector3d::Zero();
            t_Sk_Us_est = t_GB_US + R_GB_US * t_Bk_GB_est + R_GB_US * (R_GB_Bk_est.transpose()) * R_i_b * ( t_s_i - t_b_i);
            r.head<3>(3) = state_server.gnss_state.position - t_Sk_Us_est;
            cout.precision(16);
            cout << "-----------------------" << endl;
            // cout<<"Before state_server.gnss_state.position (m)= "<< state_server.gnss_state.position.transpose() << endl;  
            // cout<<"Before t_Sk_Us_est (m)= " << t_Sk_Us_est.transpose() << endl;  
            cout<<"Before gnssUpdate delta_t_Sk_Us_est (m)= "<< r.transpose() << endl;
            cout << endl;

            // gnssUpdate H11
            H_x.block<3,3>(0,0) = R_GB_US * (R_GB_Bk_est.transpose()) * skewSymmetric( R_i_b * ( t_b_i - t_s_i) );

            // gnssUpdate H12
            H_x.block<3,3>(0,3) = R_GB_US;

            // gnssUpdate H14
            H_x.block<3,3>(0,9) = R_GB_US * (R_GB_Bk_est.transpose()) * skewSymmetric( R_i_b * ( t_s_i - t_b_i) );

            // gnssUpdate H15
            H_x.block<3,3>(0,12) = - R_GB_US * (R_GB_Bk_est.transpose()) * R_i_b;
            
            Matrix3d US_t_cov;
            // std 10m
            US_t_cov << pow(10 , 2) ,0,0,
                        0, pow(10 , 2),0,
                        0, 0,pow(10 , 2);

            MatrixXd noise = MatrixXd::Identity(3,3);
            noise = US_t_cov;

            // GNSS error measurementUpdate
            measurementUpdate_gnss(H_x,r,noise);

            //-------------------------------------------------
            t_Bk_GB_est = state_server.imu_state.position;
            R_GB_Bk_est = quaternionToRotation(
                state_server.imu_state.orientation);
            R_i_b = state_server.imu_state.R_imu_body;
            t_b_i = state_server.imu_state.t_body_imu;
            R_GB_US = state_server.gnss_state.R_GB_US;
            t_GB_US = state_server.gnss_state.t_GB_US;
            //-------------------------------------------------
            t_Sk_Us_est = t_GB_US + R_GB_US * t_Bk_GB_est + R_GB_US * (R_GB_Bk_est.transpose()) * R_i_b * ( t_s_i - t_b_i);
            r.head<3>(3) = state_server.gnss_state.position - t_Sk_Us_est;
            cout.precision(16);
            cout<<"After gnssUpdate delta_t_Sk_Us_est (m)= " << r.transpose() << endl;
            cout << "-----------------------" << endl;
            cout << endl;

        }
 
}

/**
 * @brief ackUpdate 
 *  vehicle speed and yaw angular velocity measurement model
 */ 
void AckMsckfLam::ackUpdate(const CameraMeasurementConstPtr& msg){
  if(!state_server.ack_state.ack_available){
    ROS_INFO("ackUpdate: !state_server.ack_state.ack_available．．．");
    return;
  }

  // construct H and r
  MatrixXd H_x = MatrixXd::Zero(6,21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(6);

  //-------------------------------------------------
  Eigen::Vector3d G_v_est = state_server.imu_state.velocity;
  Eigen::Matrix3d R_G_Bj_est = quaternionToRotation(
      state_server.imu_state.orientation);
  Eigen::Vector3d i_w_est = state_server.imu_state.angular_velocity;
  Eigen::Matrix3d R_i_b = state_server.imu_state.R_imu_body;
  Eigen::Vector3d t_b_i = state_server.imu_state.t_body_imu;
  Eigen::Matrix3d R_i_c = state_server.imu_state.R_imu_cam0;
  Eigen::Vector3d t_c_i = state_server.imu_state.t_cam0_imu;
  Eigen::Vector3d t_body_imu = state_server.imu_state.t_body_imu;
  Eigen::Matrix3d R_body_cam0 = state_server.ack_state.R_body_cam0;
  Eigen::Vector3d t_body_cam0= state_server.ack_state.t_body_cam0;
  //-------------------------------------------------
  
  // tunable parameter configurations 
  if(ackermann_update_q){
      
      Eigen::Vector3d b_w_ack;
      
      b_w_ack = state_server.ack_state.angular_velocity;

      // lever arm effect of vehicle angular velocity
      Eigen::Vector3d b_w_est = R_i_b * i_w_est;
      Eigen::Vector3d delta_w = b_w_ack - b_w_est;
      
      r.head<3>(3) = delta_w;

      if(use_debug){
        cout.precision(16);  
        cout<<"Before i_w_est (deg/s)= "<< i_w_est.transpose() *  180 / M_PI << endl;
        cout<<"Before b_w_ack (deg/s)= "<< b_w_ack.transpose() *  180 / M_PI << endl;
        cout<<"Before b_w_est (deg/s)= "<< b_w_est.transpose() *  180 / M_PI << endl;
        cout<<"Before delta_w (deg/s)= "<< delta_w.transpose() *  180 / M_PI << endl;
        cout << endl;
      }

      // H14
      H_x.block<3,3>(0,9) = skewSymmetric(R_i_b * i_w_est);

      // H16
      H_x.block<3,3>(0,15) = - R_i_b;

  }

  //-------------------------------------------------
  if(ackermann_update_v){
      Eigen::Vector3d B_v_est = Vector3d::Zero();

      // lever arm effect of vehicle speed
      B_v_est = R_G_Bj_est * G_v_est + R_i_b * skewSymmetric(i_w_est) * t_b_i;
      
      r.tail<3>() = state_server.ack_state.velocity - B_v_est;

      if(use_debug){
        cout << "Before B_V: " << state_server.ack_state.velocity.transpose() << endl;
        cout << "Before B_v_est: " << B_v_est.transpose() << endl;
      }

      // H21
      H_x.block<3,3>(3,0) = skewSymmetric(R_G_Bj_est * G_v_est);

      // H23
      H_x.block<3,3>(3,6) = R_G_Bj_est;

      // H24
      H_x.block<3,3>(3,9) = skewSymmetric(R_i_b * skewSymmetric(i_w_est) * t_b_i);
      
      // H25
      H_x.block<3,3>(3,12) = R_i_b * skewSymmetric(i_w_est);

      // H26
      H_x.block<3,3>(3,15) = R_i_b * skewSymmetric(t_b_i);

  }
  
  Matrix3d B_R_cov;
  double estimateErrorCovariance_q = state_server.ack_state.estimateErrorCovariance_w_(0,0);   
  B_R_cov <<  estimateErrorCovariance_q * 10, 0, 0,
            0, estimateErrorCovariance_q * 10, 0,
            0, 0, estimateErrorCovariance_q;
  Matrix3d B_v_cov;
  B_v_cov <<  pow(state_server.ack_state.ackermann_speed_x_noise , 2) ,0,0,
              0, pow(state_server.ack_state.ackermann_speed_y_noise , 2),0,
              0, 0,pow(state_server.ack_state.ackermann_speed_z_noise , 2);

  MatrixXd noise=MatrixXd::Identity(6,6);
  noise.block<3,3>(0,0)= B_R_cov;
  noise.block<3,3>(3,3)= B_v_cov;

  if(use_debug){
    cout.precision(16);
    cout<< fixed << "Before Ackermann Update r: " << r.head<3>().transpose() *  180 / M_PI << " " << r.tail<3>().transpose() <<endl;
  }
  
  // ACK error measurementUpdate
  measurementUpdate_ack(H_x,r,noise);

  // debug
  //-------------------------------------------------
  G_v_est = state_server.imu_state.velocity;
  R_G_Bj_est = quaternionToRotation(
      state_server.imu_state.orientation);
  i_w_est = state_server.imu_state.angular_velocity;
  R_i_b = state_server.imu_state.R_imu_body;
  t_b_i = state_server.imu_state.t_body_imu;
  //-------------------------------------------------
  
  if(ackermann_update_q){
      Eigen::Vector3d b_w_ack;
      
      b_w_ack = state_server.ack_state.angular_velocity;

      Eigen::Vector3d b_w_est = R_i_b * i_w_est;
      Eigen::Vector3d delta_w = b_w_ack - b_w_est;
      
      r.head<3>(3) = delta_w;

      if(use_debug){
        cout.precision(16);  
        cout<<"After delta_w (deg/s)= "<< delta_w.transpose() *  180 / M_PI << endl;
      }

  }
  
  if(ackermann_update_v){
      Eigen::Vector3d B_v_est = Vector3d::Zero();

      B_v_est = R_G_Bj_est * G_v_est + R_i_b * skewSymmetric(i_w_est) * t_b_i;

      r.tail<3>() = state_server.ack_state.velocity - B_v_est;

  }

  if(use_debug){
    cout.precision(16);
    cout<< fixed << "After Ackermann Update r: " << r.head<3>().transpose() *  180 / M_PI << " " << r.tail<3>().transpose() <<endl;
    cout << "---------------------------" << endl;
  }   
    
}

/**
 * @brief featureJacobian 
 *  
 */ 
void AckMsckfLam::featureJacobian(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) {

  const auto& feature = map_server[feature_id];

  vector<StateIDType> valid_cam_state_ids(0);
  for (const auto& cam_id : cam_state_ids) {
    if (feature.observations.find(cam_id) ==
        feature.observations.end()) continue;

    valid_cam_state_ids.push_back(cam_id);
  }

  int jacobian_row_size = 0;
  jacobian_row_size = 2* valid_cam_state_ids.size();//4

  MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
      21+state_server.cam_states.size()*6);
  MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
  VectorXd r_j = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (const auto& cam_id : valid_cam_state_ids) {

    Matrix<double, 2, 6> H_xi = Matrix<double, 2, 6>::Zero();
    Matrix<double, 2, 3> H_fi = Matrix<double, 2, 3>::Zero();
    Vector2d r_i = Vector2d::Zero();

    measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

    auto cam_state_iter = state_server.cam_states.find(cam_id);
    int cam_state_cntr = std::distance(
        state_server.cam_states.begin(), cam_state_iter);

    // Stack the Jacobians.
    H_xj.block<2, 6>(stack_cntr, 21+6*cam_state_cntr) = H_xi;
    H_fj.block<2, 3>(stack_cntr, 0) = H_fi;
    r_j.segment<2>(stack_cntr) = r_i;
    stack_cntr += 2;

  }

  JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
  MatrixXd A = svd_helper.matrixU().rightCols(
      jacobian_row_size - 3);

  H_x = A.transpose() * H_xj;
  r = A.transpose() * r_j;

  return;

}

/**
 * @brief initializeGravityAndBias 
 *  
 */ 
void AckMsckfLam::initializeGravityAndBias() {

  // Initialize gravity and gyro bias.
  Vector3d sum_angular_vel = Vector3d::Zero();
  Vector3d sum_linear_acc = Vector3d::Zero();

  for (const auto& imu_msg : imu_msg_buffer) {
    Vector3d angular_vel = Vector3d::Zero();
    Vector3d linear_acc = Vector3d::Zero();

    tf::vectorMsgToEigen(imu_msg.angular_velocity, angular_vel);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, linear_acc);

    sum_angular_vel += angular_vel;
    sum_linear_acc += linear_acc;
  }

  // if use_offline_bias
  if(use_offline_bias){
    state_server.imu_state.gyro_bias = state_server.imu_state.initial_bias;
  }else{
    state_server.imu_state.gyro_bias =
      sum_angular_vel / imu_msg_buffer.size();
  }
  cout << "state_server.imu_state.gyro_bias: " << state_server.imu_state.gyro_bias.transpose() << endl;

  // This is the gravity in the IMU frame.
  Vector3d gravity_imu =
    sum_linear_acc / imu_msg_buffer.size();

  double gravity_norm = gravity_imu.norm();
  IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);

  // This is the gravity in the Body frame for the static initialization
  Vector3d gravity_body = state_server.imu_state.T_imu_body.linear() * gravity_imu;
  Quaterniond q_B0_GB = Quaterniond::FromTwoVectors(
    gravity_body, -IMUState::gravity);
  T_B0_GBI.linear() = q_B0_GB.toRotationMatrix();
  T_B0_GBI.translation() = Eigen::Vector3d (0,0,0);
  double roll_B0_GB,pitch_B0_GB,yaw_B0_GB;
  tf::Matrix3x3(tf::Quaternion(q_B0_GB.x(),q_B0_GB.y(),q_B0_GB.z(),q_B0_GB.w())).getRPY(roll_B0_GB,pitch_B0_GB,yaw_B0_GB,1); 
  state_server.imu_state.orientation =
    rotationToQuaternion(q_B0_GB.toRotationMatrix().transpose());
  state_server.imu_state.position = T_B0_GBI.translation();

  std::cout.precision(16);
  std::cout << "----------------------" << std::endl;
  std::cout <<"ack_msckf_lm {GB} q_B0_GB : roll="<< roll_B0_GB * 180 / M_PI <<",　pitch="<< pitch_B0_GB * 180 / M_PI <<",　yaw="<< yaw_B0_GB * 180 / M_PI << std::endl;
  std::cout << "----------------------" << std::endl;
  std::cout << endl;

  return;

}

/**
 * @brief csv_timer_callBack 
 *  
 */ 
void AckMsckfLam::csv_timer_callBack(const ros::TimerEvent& event){
  
  if (!is_csv_curr_time_init) return;

  if(ros::Time::now().toSec() - csv_curr_time > 120){
    pose_file.close();
    odom_file.close();
    std_file.close();
    rmse_file.close();
    time_file.close();
    ros::shutdown();
  }

  if(csvData_odom.empty() && csvData_rmse.empty() && csvData_time.empty()){
    return;
  }

  std::string delim = ",";

  if(ros::Time::now().toSec() - csv_curr_time > 1){

    // save pose odom
    for (const auto& csvdata_odom : csvData_odom) {

      // save pose 
      // TUM #timestamp(sec) x y z q_x q_y q_z q_w
      pose_file.precision(16);
      pose_file << fixed << csvdata_odom.time << " " << csvdata_odom.pB(0) << " " << csvdata_odom.pB(1) << " " << csvdata_odom.pB(2) << " "
              << csvdata_odom.qB.x() << " " << csvdata_odom.qB.y() << " " << csvdata_odom.qB.z() << " " << csvdata_odom.qB.w() << endl;
      
      odom_file.precision(16);
      odom_file << fixed << csvdata_odom.time << delim << csvdata_odom.Dtime << delim;
      odom_file  << delim;
      odom_file << csvdata_odom.pB(0) << delim << csvdata_odom.pB(1) << delim << csvdata_odom.pB(2) << delim;
      odom_file << csvdata_odom.qB.x() << delim <<  csvdata_odom.qB.y() << delim << csvdata_odom.qB.z() << delim << csvdata_odom.qB.w() << delim;
      odom_file << csvdata_odom.roll << delim << csvdata_odom.pitch << delim << csvdata_odom.yaw << delim;
      odom_file << csvdata_odom.vB(0) << delim << csvdata_odom.vB(1) << delim << csvdata_odom.vB(2) << delim;
      odom_file << csvdata_odom.wB(0) << delim << csvdata_odom.wB(1) << delim << csvdata_odom.wB(2) << delim;
      odom_file  << delim;
      // gt
      odom_file << csvdata_odom.gt_pB(0) << delim << csvdata_odom.gt_pB(1) << delim << csvdata_odom.gt_pB(2) << delim;
      odom_file << csvdata_odom.gt_qB.x() << delim <<  csvdata_odom.gt_qB.y() << delim << csvdata_odom.gt_qB.z() << delim << csvdata_odom.gt_qB.w() << delim;
      odom_file << csvdata_odom.gt_roll << delim << csvdata_odom.gt_pitch << delim << csvdata_odom.gt_yaw << delim;
      odom_file << csvdata_odom.gt_vB(0) << delim << csvdata_odom.gt_vB(1) << delim << csvdata_odom.gt_vB(2) << delim;
      odom_file << csvdata_odom.gt_wB(0) << delim << csvdata_odom.gt_wB(1) << delim << csvdata_odom.gt_wB(2) << delim;
      odom_file  << delim;
      odom_file << csvdata_odom.Sr << delim << csvdata_odom.Sr_avg << delim;
      odom_file << delim;
      odom_file << csvdata_odom.pIBs(0) << delim << csvdata_odom.pIBs(1) << delim << csvdata_odom.pIBs(2) << delim;
      odom_file << csvdata_odom.qIBs.x() << delim <<  csvdata_odom.qIBs.y() << delim << csvdata_odom.qIBs.z() << delim << csvdata_odom.qIBs.w() << delim;
      odom_file << csvdata_odom.roll_ibx << delim << csvdata_odom.pitch_iby << delim << csvdata_odom.yaw_ibz << delim;
      odom_file << delim;
      odom_file << csvdata_odom.b_v_la(0) << delim << csvdata_odom.b_v_la(1) << delim << csvdata_odom.b_v_la(2) << delim;
      odom_file << std::endl;

    } 
    std::cout << "-------pose_file odom_file save done!!!---------" << std::endl;

    // save rmse
    for (const auto& csvdata_rmse : csvData_rmse) {

        double dStamp = csvdata_rmse.time;;
        double Dtime = csvdata_rmse.Dtime;

        double err_rx = csvdata_rmse.err_rx;
        double err_ry = csvdata_rmse.err_ry;
        double err_rz = csvdata_rmse.err_rz;
        double err_px = csvdata_rmse.err_px;
        double err_py = csvdata_rmse.err_py;
        double err_pz = csvdata_rmse.err_pz;
        double err_vx = csvdata_rmse.err_vx;
        double err_vy = csvdata_rmse.err_vy;
        double err_vz = csvdata_rmse.err_vz;
        double err_ribx = csvdata_rmse.err_ribx;
        double err_riby = csvdata_rmse.err_riby;
        double err_ribz = csvdata_rmse.err_ribz;
        double err_pbix = csvdata_rmse.err_pbix;
        double err_pbiy = csvdata_rmse.err_pbiy;
        double err_pbiz = csvdata_rmse.err_pbiz;
        double err_bgx = csvdata_rmse.err_bgx;
        double err_bgy = csvdata_rmse.err_bgy;
        double err_bgz = csvdata_rmse.err_bgz;
        double err_bax = csvdata_rmse.err_bax;
        double err_bay = csvdata_rmse.err_bay;
        double err_baz = csvdata_rmse.err_baz;
        
        double std_rx = csvdata_rmse.std_rx;
        double std_ry = csvdata_rmse.std_ry; 
        double std_rz = csvdata_rmse.std_rz;
        double std_px = csvdata_rmse.std_px;
        double std_py = csvdata_rmse.std_py;
        double std_pz = csvdata_rmse.std_pz;
        double std_vx = csvdata_rmse.std_vx;
        double std_vy = csvdata_rmse.std_vy;
        double std_vz = csvdata_rmse.std_vz;
        double std_ribx = csvdata_rmse.std_ribx;
        double std_riby = csvdata_rmse.std_riby;
        double std_ribz = csvdata_rmse.std_ribz;
        double std_pbix = csvdata_rmse.std_pbix;
        double std_pbiy = csvdata_rmse.std_pbiy;
        double std_pbiz = csvdata_rmse.std_pbiz;
        double std_bgx = csvdata_rmse.std_bgx;
        double std_bgy = csvdata_rmse.std_bgy;
        double std_bgz = csvdata_rmse.std_bgz;
        double std_bax = csvdata_rmse.std_bax;
        double std_bay = csvdata_rmse.std_bay;
        double std_baz = csvdata_rmse.std_baz;

        std_file.precision(16);
        std_file << fixed << dStamp << delim << Dtime << delim;
        std_file  << delim;
        std_file  << err_rx << delim << err_ry << delim << err_rz << delim;
        std_file  << 3 * std_rx << delim << 3 * std_ry << delim << 3 * std_rz << delim;
        std_file  << -3 * std_rx << delim << -3 * std_ry << delim << -3 * std_rz << delim;
        std_file << delim;
        std_file  << err_px << delim << err_py << delim << err_pz << delim;
        std_file  << 3 * std_px << delim << 3 * std_py << delim << 3 * std_pz << delim;
        std_file  << -3 * std_px << delim << -3 * std_py << delim << -3 * std_pz << delim;
        std_file << delim;
        std_file << err_vx << delim << err_vy << delim << err_vz << delim;
        std_file << 3 * std_vx << delim << 3 * std_vy << delim << 3 * std_vz << delim;
        std_file << -3 * std_vx << delim << -3 * std_vy << delim << -3 * std_vz << delim;
        std_file  << delim;
        std_file  << err_ribx << delim << err_riby << delim << err_ribz << delim;
        std_file  << 3 * std_ribx << delim << 3 * std_riby << delim << 3 * std_ribz << delim;
        std_file  << -3 * std_ribx << delim << -3 * std_riby << delim << -3 * std_ribz << delim;
        std_file  << delim;
        std_file  << err_pbix << delim << err_pbiy << delim << err_pbiz << delim;
        std_file  << 3 * std_pbix << delim << 3 * std_pbiy << delim << 3 * std_pbiz << delim;
        std_file  << -3 * std_pbix << delim << -3 * std_pbiy << delim << -3 * std_pbiz << delim;
        std_file  << delim;
        std_file  << err_bgx << delim << err_bgy << delim << err_bgz << delim;
        std_file  << 3 * std_bgx << delim << 3 * std_bgy << delim << 3 * std_bgz << delim;
        std_file  << -3 * std_bgx << delim << -3 * std_bgy << delim << -3 * std_bgz << delim;
        std_file << delim;
        std_file  << err_bax << delim << err_bay << delim << err_baz << delim;
        std_file  << 3 * std_bax << delim << 3 * std_bay << delim << 3 * std_baz << delim;
        std_file  << -3 * std_bax << delim << -3 * std_bay << delim << -3 * std_baz << delim;
        std_file << std::endl;

        // save rmse
        double nees_rx;
        double nees_ry;
        double nees_rz;
        double nees_px;
        double nees_py;
        double nees_pz;
        double nees_ribx;
        double nees_riby;
        double nees_ribz;
        double nees_pbix;
        double nees_pbiy;
        double nees_pbiz;
        double nees_bgx;
        double nees_bgy;
        double nees_bgz;
        double nees_bax;
        double nees_bay;
        double nees_baz;
        double nees_vx;
        double nees_vy;
        double nees_vz;
        if(is_first_nees){
          is_first_nees = false;
          nees_rx = 0;
          nees_ry = 0;
          nees_rz = 0;
          nees_px = 0;
          nees_py = 0;
          nees_pz = 0;
          nees_ribx = 0;
          nees_riby = 0;
          nees_ribz = 0;
          nees_pbix = 0;
          nees_pbiy = 0;
          nees_pbiz = 0;
          nees_bgx = 0;
          nees_bgy = 0;
          nees_bgz = 0;
          nees_bax = 0;
          nees_bay = 0;
          nees_baz = 0;
          nees_vx = 0;
          nees_vy = 0;
          nees_vz = 0;
        }else{
          nees_rx = err_rx * err_rx / (std_rx * std_rx);
          nees_ry = err_ry * err_ry / (std_ry * std_ry);
          nees_rz = err_rz * err_rz / (std_rz * std_rz);
          nees_px = err_px * err_px / (std_px * std_px);
          nees_py = err_py * err_py / (std_py * std_py);
          nees_pz = err_pz * err_pz / (std_pz * std_pz);
          nees_ribx = 0;
          nees_riby = 0;
          nees_ribz = 0;
          nees_pbix = 0;
          nees_pbiy = 0;
          nees_pbiz = 0;
          nees_bgx = err_bgx * err_bgx / (std_bgx * std_bgx);
          nees_bgy = err_bgy * err_bgy / (std_bgy * std_bgy);
          nees_bgz = err_bgz * err_bgz / (std_bgz * std_bgz);
          nees_bax = err_bax * err_bax / (std_bax * std_bax);
          nees_bay = err_bay * err_bay / (std_bay * std_bay);
          nees_baz = err_baz * err_baz / (std_baz * std_baz);
          nees_vx = err_vx * err_vx / (std_vx * std_vx);
          nees_vy = err_vy * err_vy / (std_vy * std_vy);
          nees_vz = err_vz * err_vz / (std_vz * std_vz);
        }
        rmse_file.precision(16);
        rmse_file << fixed << dStamp << delim << Dtime << delim;
        rmse_file  << delim;
        rmse_file << err_rx * err_rx << delim << err_ry * err_ry << delim << err_rz * err_rz << delim;
        rmse_file << err_rx * err_rx + err_ry * err_ry + err_rz * err_rz << delim;
        rmse_file << nees_rx << delim << nees_ry << delim << nees_rz << delim;
        rmse_file << delim;
        rmse_file << err_px * err_px << delim << err_py * err_py << delim << err_pz * err_pz << delim;
        rmse_file << err_px * err_px + err_py * err_py + err_pz * err_pz << delim;
        rmse_file << nees_px << delim << nees_py << delim << nees_pz << delim;
        rmse_file  << delim;
        rmse_file << 0 << delim << 0 << delim << 0 << delim;
        rmse_file << nees_ribx << delim << nees_riby << delim << nees_ribz << delim;
        rmse_file  << delim;
        rmse_file << 0 << delim << 0 << delim << 0 << delim;
        rmse_file << nees_pbix << delim << nees_pbiy << delim << nees_pbiz << delim;
        rmse_file  << delim;
        rmse_file << err_bgx * err_bgx << delim << err_bgy * err_bgy << delim << err_bgz * err_bgz << delim;
        rmse_file << nees_bgx << delim << nees_bgy << delim << nees_bgz << delim;
        rmse_file  << delim;
        rmse_file << err_bax * err_bax << delim << err_bay * err_bay << delim << err_baz * err_baz << delim;
        rmse_file << nees_bax << delim << nees_bay << delim << nees_baz << delim;
        rmse_file << delim;
        rmse_file << err_vx * err_vx << delim << err_vy * err_vy << delim << err_vz * err_vz << delim;
        rmse_file << nees_vx << delim << nees_vy << delim << nees_vz << delim;
        rmse_file  << std::endl;

    }
    std::cout << "-------rmse_file save done!!!---------" << std::endl;

    // save time
    for (const auto& csvdata_time : csvData_time) {

        double dStamp = csvdata_time.time;;
        double Dtime = csvdata_time.Dtime;
        double process_time = csvdata_time.process_time;
        double total_time = csvdata_time.total_time;
        double avg_time = csvdata_time.avg_time;

        time_file.precision(16);
        time_file << fixed << dStamp << delim << Dtime << delim;
        time_file << delim;
        time_file << process_time << delim << total_time << delim << avg_time << delim;
        time_file << std::endl;

    }
    std::cout << "-------time_file save done!!!---------" << std::endl;

    std::cout << std::endl;
    csvData_odom.clear();
    csvData_rmse.clear();
    csvData_time.clear();

  }

  return;

}

} // namespace ack_msckf_lm

