/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_IMU_STATE_H
#define MSCKF_VIO_IMU_STATE_H

#include <map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include"math_utils.hpp"

#define GRAVITY_ACCELERATION 9.81

namespace ack_msckf_lam {

/*
 * @brief IMUState State for IMU
 */
struct IMUState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef long long int StateIDType;

  // An unique identifier for the IMU state.
  StateIDType id;

  // id for next IMU state
  static StateIDType next_id;

  // Time when the state is recorded
  double time;

  // Orientation
  // Take a vector from the world frame to
  // the IMU (body) frame.
  Eigen::Vector4d orientation;

  // Position of the IMU (body) frame
  // in the world frame.
  Eigen::Vector3d position;

  // Velocity of the IMU (body) frame
  // in the world frame.
  Eigen::Vector3d velocity;

  // angular_velocity of the IMU frame
  // in the IMU frame.
  Eigen::Vector3d angular_velocity;

  // Bias for measured angular velocity
  // and acceleration.
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d acc_bias;

  // Transformation between the IMU and the
  // left camera (cam0)
  Eigen::Isometry3d T_imu_cam0;
  Eigen::Matrix3d R_imu_cam0;
  Eigen::Vector3d t_cam0_imu;
  

  // These three variables should have the same physical
  // interpretation with `orientation`, `position`, and
  // `velocity`. There three variables are used to modify
  // the transition matrices to make the observability matrix
  // have proper null space.
  Eigen::Vector4d orientation_null;
  Eigen::Vector3d position_null;
  Eigen::Vector3d velocity_null;


  Eigen::Vector4d orientation_null_ack;
  Eigen::Vector3d position_null_ack;
  Eigen::Vector3d velocity_null_ack;


  // Process noise
  static double gyro_noise;
  static double acc_noise;
  static double gyro_bias_noise;
  static double acc_bias_noise;

  // Gravity vector in the world frame
  static Eigen::Vector3d gravity;

  // Transformation offset from the IMU frame to
  // the body frame. The transformation takes a
  // vector from the IMU frame to the body frame.
  // The z axis of the body frame should point upwards.
  // Normally, this transform should be identity.
  Eigen::Isometry3d T_imu_body;


  Eigen::Matrix3d R_imu_body;
  Eigen::Vector3d t_imu_body;

  Eigen::Isometry3d T_body_imu;
  Eigen::Vector3d t_body_imu;
  
  Eigen::Vector3d initial_bias;

  double gt_time;

  IMUState(): id(0), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    velocity(Eigen::Vector3d::Zero()),
    angular_velocity(Eigen::Vector3d::Zero()),
    gyro_bias(Eigen::Vector3d::Zero()),
    acc_bias(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d::Zero()),
    velocity_null(Eigen::Vector3d::Zero()) {}

  IMUState(const StateIDType& new_id): id(new_id), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    velocity(Eigen::Vector3d::Zero()),
    angular_velocity(Eigen::Vector3d::Zero()),
    gyro_bias(Eigen::Vector3d::Zero()),
    acc_bias(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d::Zero()),
    velocity_null(Eigen::Vector3d::Zero()) {}

};

typedef IMUState::StateIDType StateIDType;






/*
 * @brief ACKState State for IMU
 */
struct ACKState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // An unique identifier for the IMU state.
  StateIDType id;

  // id for next IMU state
  static StateIDType next_id;

  // Time when the state is recorded
  double time;

  // Orientation
  // Take a vector from the world frame to
  // the IMU (body) frame.
  Eigen::Vector4d orientation;

  // Position of the IMU (body) frame
  // in the world frame.
  Eigen::Vector3d position;

  // Velocity of the IMU (body) frame
  // in the world frame.
  Eigen::Vector3d velocity;


  double steer_angle;

  // angular_velocity of the IMU frame
  // in the IMU frame.
  Eigen::Vector3d angular_velocity;


  Eigen::Vector3d initial_bias;


  // Bias for measured angular velocity
  // and acceleration.
  // Eigen::Vector3d gyro_bias;
  // Eigen::Vector3d acc_bias;


  // Transformation between the IMU and the
  // left camera (cam0)
  Eigen::Isometry3d T_body_cam0;
  Eigen::Matrix3d R_body_cam0;
  Eigen::Vector3d t_body_cam0;
  Eigen::Vector3d t_cam0_body;
  Eigen::Vector3d ground_norm_vector;

  Eigen::Isometry3d Ci_Cj_T;
  Eigen::Matrix3d Ci_Cj_R;
  Eigen::Vector3d Ci_Cj_t;
  cv::Matx33d Ci_Cj_R_cv;
  cv::Vec3d   Ci_Cj_t_cv;

  Eigen::Isometry3d Cj_Ci_T;
  Eigen::Matrix3d Cj_Ci_R;
  Eigen::Vector3d Cj_Ci_t;
  cv::Matx33d Cj_Ci_R_cv;
  cv::Vec3d   Cj_Ci_t_cv;
  cv::Vec3d ground_norm_vector_cv;

  // These three variables should have the same physical
  // interpretation with `orientation`, `position`, and
  // `velocity`. There three variables are used to modify
  // the transition matrices to make the observability matrix
  // have proper null space.
  Eigen::Vector4d orientation_null;
  Eigen::Vector3d position_null;
  Eigen::Vector3d velocity_null;


  static double roll_noise;
  static double pitch_noise;
  static double steer_noise;
  static double vx_noise;
  static double vy_noise;
  static double vz_noise;




  static double WheelBase;
  static double AckRate;
  static double TireBase;
  static double SteerRatio;


  double ackermann_kv;
  double ackermann_ks;
  double ackermann_bs;
  double ackermann_bl;
  double ackermann_bt;
  double ackermann_WheelBase;
  double ackermann_TireBase;
  double ackermann_TireBase_sign;


  double gt_time;

  double kaist_time;


  Eigen::MatrixXd estimateErrorCovariance_w_; 
  Eigen::MatrixXd estimateErrorCovariance_;
  Eigen::MatrixXd transferFunctionJacobian_;
  Eigen::MatrixXd processFunctionJacobian_;
  Eigen::MatrixXd processNoiseCovariance_ackerman_;
  Eigen::MatrixXd processWhiteNoiseCovariance_;
  double speed_x;
  double gyro_z;
  Eigen::Quaterniond delta_q;
  Eigen::Vector3d delta_p;
  Eigen::Vector3d B_V_avg;
  double ackermann_heading_;
  double ackermann_x_;
  double ackermann_y_;
  double ackermann_speed_x_noise;
  double ackermann_speed_y_noise;
  double ackermann_speed_z_noise;
  double ackermann_steering_noise;
  double ackermann_heading_white_noise;
  double ackermann_x_white_noise;
  double ackermann_y_white_noise;
  double delta_time;
  bool ack_available;


  ACKState(): id(0), time(0), gt_time(0), kaist_time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    velocity(Eigen::Vector3d::Zero()),
    angular_velocity(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d::Zero()),
    velocity_null(Eigen::Vector3d::Zero()),
    estimateErrorCovariance_w_(Eigen::MatrixXd::Zero(1,1)),
    estimateErrorCovariance_(Eigen::MatrixXd::Zero(3,3)),
    transferFunctionJacobian_(Eigen::MatrixXd::Zero(3,3)),
    processFunctionJacobian_(Eigen::MatrixXd::Zero(3,2)),
    processNoiseCovariance_ackerman_(Eigen::MatrixXd::Zero(2,2)),
    processWhiteNoiseCovariance_(Eigen::MatrixXd::Zero(3,3)){}

  ACKState(const StateIDType& new_id): id(new_id), time(0), gt_time(0), kaist_time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    velocity(Eigen::Vector3d::Zero()),
    angular_velocity(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d::Zero()),
    velocity_null(Eigen::Vector3d::Zero()),
    estimateErrorCovariance_w_(Eigen::MatrixXd::Zero(1,1)),
    estimateErrorCovariance_(Eigen::MatrixXd::Zero(3,3)),
    transferFunctionJacobian_(Eigen::MatrixXd::Zero(3,3)),
    processFunctionJacobian_(Eigen::MatrixXd::Zero(3,2)),
    processNoiseCovariance_ackerman_(Eigen::MatrixXd::Zero(2,2)),
    processWhiteNoiseCovariance_(Eigen::MatrixXd::Zero(3,3)) {}




};


/*
 * @brief GNSSState State
 * raw_gnss fusion
 */
struct GNSSState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // An unique identifier for the GNSSS state.
  StateIDType id;

  // id for next GNSSS state
  static StateIDType next_id;

  // Time when the state is recorded
  double time;

  // Orientation
  // Take a vector from the world frame to
  // the GNSSS frame.
  Eigen::Vector4d orientation;

  // Position of the GNSSS frame
  // in the world frame.
  Eigen::Vector3d position;

  double heading;

  // ICP 
  // JPL
  Eigen::Matrix3d R_GB_US;
  Eigen::Vector3d t_GB_US;

  GNSSState(): id(0), time(0), heading(0),
    R_GB_US(Eigen::MatrixXd::Identity(3,3)),
    t_GB_US(Eigen::Vector3d::Zero()),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()){}

  GNSSState(const StateIDType& new_id): id(new_id), time(0), heading(0),
    R_GB_US(Eigen::MatrixXd::Identity(3,3)),
    t_GB_US(Eigen::Vector3d::Zero()),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()){}

};

} // namespace ack_msckf_lam

#endif // MSCKF_VIO_IMU_STATE_H
