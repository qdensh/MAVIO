/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <std_srvs/Trigger.h>

#include "imu_state.h"
#include "cam_state.h"

#include <ack_msckf_lam/ack_msckf_lam.h>
#include <ack_msckf_lam/math_utils.hpp>
#include <ack_msckf_lam/utils.h>
#include "feature.hpp"
#include <ack_msckf_lam/CameraMeasurement.h>
#include <ack_msckf_lam/AckermannDriveStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/NavSatFix.h>

namespace ack_msckf_lam {
/*
 * @brief AckMsckfLam Implements the algorithm in
 *    Anatasios I. Mourikis, and Stergios I. Roumeliotis,
 *    "A Multi-State Constraint Kalman Filter for Vision-aided
 *    Inertial Navigation",
 *    http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf
 */
class AckMsckfLam {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    AckMsckfLam(ros::NodeHandle& pnh);
    // Disable copy and assign constructor
    AckMsckfLam(const AckMsckfLam&) = delete;
    AckMsckfLam operator=(const AckMsckfLam&) = delete;

    // Destructor
    ~AckMsckfLam() {}

    /*
     * @brief initialize Initialize the VIO.
     */
    bool initialize();

    /*
     * @brief reset Resets the VIO to initial status.
     */
    void reset();

    typedef boost::shared_ptr<AckMsckfLam> Ptr;
    typedef boost::shared_ptr<const AckMsckfLam> ConstPtr;

  private:
    
    /*
     * @brief StateServer Store one IMU states and several
     *    camera states for constructing measurement
     *    model.
     */
    struct StateServer {
        
      // raw_gnss fusion
      GNSSState gnss_state;

      ACKState ack_state;

      IMUState imu_state;
      CamStateServer cam_states;

      // State covariance matrix
      Eigen::MatrixXd state_cov;
      Eigen::Matrix<double, 12, 12> continuous_noise_cov;

      // ack State covariance matrix
      Eigen::MatrixXd state_cov_ack;
      Eigen::Matrix<double, 6, 6> continuous_noise_cov_ack;

    };


    /*
     * @brief loadParameters
     *    Load parameters from the parameter server.
     */
    bool loadParameters();

    /*
     * @brief createRosIO
     *    Create ros publisher and subscirbers.
     */
    bool createRosIO();

     /*
     * @brief ackCallback
     *    Callback function for the ack message.
     * @param msg ACK msg.
     */
    void ackCallback(const ack_msckf_lam::AckermannDriveStamped::ConstPtr& msg);

    /*
     * @brief imuCallback
     *    Callback function for the imu message.
     * @param msg IMU msg.
     */
    void imuCallback(const sensor_msgs::ImuConstPtr& msg);

    /*
     * @brief featureCallback
     *    Callback function for feature measurements.
     * @param msg Stereo feature measurements.
     */
    void featureCallback(const CameraMeasurementConstPtr& msg);

    /*
     * @brief initializegravityAndBias
     *    Initialize the IMU bias and initial orientation
     *    based on the first few IMU readings.
     */
    void initializeGravityAndBias();

    bool gatingTest(const Eigen::MatrixXd& H,
        const Eigen::VectorXd&r, const int& dof);
    void findRedundantCamStates(
        std::vector<StateIDType>& rm_cam_state_ids);

    // Chi squared test table.
    static std::map<int, double> chi_squared_test_table;

    // State vector
    StateServer state_server;
    // Maximum number of camera states
    int max_cam_state_size;

    // Features used
    MapServer map_server;

    // ACK data buffer
    std::vector<ack_msckf_lam::AckermannDriveStamped> ack_msg_buffer;

    // IMU data buffer
    // This is buffer is used to handle the unsynchronization or
    // transfer delay between IMU and Image messages.
    std::vector<sensor_msgs::Imu> imu_msg_buffer;

    // Indicate if the gravity vector is set.
    bool is_gravity_set;

    // Indicate if the received image is the first one. The
    // system will start after receiving the first image.
    bool is_first_img;

    // The position uncertainty threshold is used to determine
    // when to reset the system online. Otherwise, the ever-
    // increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning.
    // Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;

    // Tracking rate
    double tracking_rate;

    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;
    
    // Ros node handle
    ros::NodeHandle nh;

    // Subscribers and publishers
    ros::Subscriber feature_sub;
    ros::Subscriber feature_ack_sub;
    ros::Publisher odom_pub;
    ros::Publisher feature_pub;
    tf::TransformBroadcaster tf_pub;
    ros::ServiceServer reset_srv;

    // Frame id
    std::string fixed_frame_id;
    std::string child_frame_id;

    // Whether to publish tf or not.
    bool publish_tf;

    // Framte rate of the stereo images. This variable is
    // only used to determine the timing threshold of
    // each iteration of the filter.
    double frame_rate;

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(
        const double& time_bound);
    void processModel(const double& time,
        const Eigen::Vector3d& m_gyro,
        const Eigen::Vector3d& m_acc);
    void predictNewState(const double& dt,
        const Eigen::Vector3d& gyro_b,
        const Eigen::Vector3d& gyro_i,
        const Eigen::Vector3d& acc);
     // Measurement update
    void stateAugmentation(const double& time);
    void addFeatureObservations(const CameraMeasurementConstPtr& msg);
    // This function is used to compute the measurement Jacobian
    // for a single feature observed at a single camera frame.
    void measurementJacobian(const StateIDType& cam_state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 2, 6>& H_x,
        Eigen::Matrix<double, 2, 3>& H_f,
        Eigen::Vector2d& r);
    // This function computes the Jacobian of all measurements viewed
    // in the given camera states of this feature.
    void featureJacobian(const FeatureIDType& feature_id,
        const std::vector<StateIDType>& cam_state_ids,
        Eigen::MatrixXd& H_x, Eigen::VectorXd& r);
    void measurementUpdate(const Eigen::MatrixXd& H,
        const Eigen::VectorXd& r);
    void removeLostFeatures();
    void pruneCamStateBuffer();
    // Reset the system online if the uncertainty is too large.
    void onlineReset();
    void batchGtProcessing(
        const double& time_bound);
    /*
     * @brief publish Publish the results of VIO.
     * @param time The time stamp of output msgs.
     */
    void publish(const ros::Time& time);

    // save csv
    std::string output_path;
    std::ofstream pose_file;
    std::ofstream odom_file;
    std::ofstream std_file;
    std::ofstream rmse_file;

    // gt
    ros::Subscriber gt_init_sub;
    void gtInitCallback(const nav_msgs::OdometryConstPtr& msg);
    Eigen::Isometry3d T_B0_GS_gt = Eigen::Isometry3d::Identity();
    bool is_gt_init_set = false;
    std::vector<nav_msgs::Odometry> gt_msg_buffer;
    nav_msgs::Odometry gt_odom_curr;
    nav_msgs::Odometry gt_odom_last;
    Eigen::Vector3d t_GBi_GS_last;
    void batchGtProcessing_ack(
        const double& time_bound);
    
    bool is_first_sr = true;
    bool is_first_nees = true;
    unsigned long long int global_count = 0;
    double global_scale_ratio = 0;

    // debug
    bool use_a27_platform;
    bool use_svd_ex;
    bool use_debug;
    bool use_offline_bias;
    bool use_ackermann;
    bool ackermann_update_v;
    bool ackermann_update_q;
    
    double DfirstTime;
    double Dtime;
    ros::Subscriber imu_sub;
    ros::Subscriber ack_sub;

    Eigen::Isometry3d T_GI_B0 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_B0_GI = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_GI_I0 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_B0_GBI = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_GI_GBI = Eigen::Isometry3d::Identity();

    void measurementUpdate_ack(
        const Eigen::MatrixXd& H, const Eigen::VectorXd& r,const Eigen::MatrixXd &noise);
    void batchAckProcessing(const double& time_bound);
    void ackUpdate(const CameraMeasurementConstPtr& msg);
    void processModel_ack(const double& time,
        Eigen::Vector3d& m_speed,
        double& steering_angle_0);

    // ODOM
    struct CSVDATA_ODOM {
        double time;
        double Dtime;
        Eigen::Vector3d pB;
        Eigen::Quaterniond qB;
        double roll, pitch, yaw;
        Eigen::Vector3d vB;
        Eigen::Vector3d wB;

        Eigen::Vector3d gt_pB;
        Eigen::Quaterniond gt_qB;
        double gt_roll, gt_pitch, gt_yaw;
        Eigen::Vector3d gt_vB;
        Eigen::Vector3d gt_wB;

        double Sr;
        double Sr_avg;

        Eigen::Vector3d pIBs;
        Eigen::Quaterniond qIBs;
        double roll_ibx, pitch_iby, yaw_ibz;
        Eigen::Vector3d b_v_la;

        Eigen::Matrix<double, 5, 1> k_b;
    };
    std::vector<struct CSVDATA_ODOM> csvData_odom;

    ros::Timer csv_timer;
    void csv_timer_callBack(const ros::TimerEvent& event);
    double csv_curr_time = 0;
    
    // RMSE
    struct CSVDATA_RMSE {
        double time;
        double Dtime;

        double err_rx, err_ry, err_rz;
        double err_px;
        double err_py;
        double err_pz;
        double err_vx;
        double err_vy;
        double err_vz;
        double err_ribx;
        double err_riby;
        double err_ribz;
        double err_pbix;
        double err_pbiy;
        double err_pbiz;
        double err_bgx;
        double err_bgy;
        double err_bgz;
        double err_bax;
        double err_bay;
        double err_baz;
        
        double std_rx, std_ry, std_rz;
        double std_px;
        double std_py;
        double std_pz;
        double std_vx;
        double std_vy;
        double std_vz;
        double std_ribx;
        double std_riby;
        double std_ribz;
        double std_pbix;
        double std_pbiy;
        double std_pbiz;
        double std_bgx;
        double std_bgy;
        double std_bgz;
        double std_bax;
        double std_bay;
        double std_baz;
        
    };
    std::vector<struct CSVDATA_RMSE> csvData_rmse;

    std::ofstream time_file;
    // TIME
    struct CSVDATA_TIME {
        double time;
        double Dtime;

        double process_time, avg_time, total_time;
    };
    std::vector<struct CSVDATA_TIME> csvData_time;
    double total_time = 0;
    bool is_csv_curr_time_init = false;

    // raw_gnss fusion
    ros::Subscriber gnss_sub;
    bool is_gnss_aligned = false;
    void rawGnssCallback(const sensor_msgs::NavSatFix & msg);
    bool use_raw_gnss = false;
    void gnssUpdate(const CameraMeasurementConstPtr& msg);
    Eigen::Vector3d t_s_i;
    void measurementUpdate_gnss(
        const Eigen::MatrixXd& H, const Eigen::VectorXd& r,const Eigen::MatrixXd &noise);
    std::deque<std::pair<double,sensor_msgs::NavSatFix>> gnss_msg_buffer;
    std::deque<std::pair<double,Eigen::Vector3d>> vio_position_buffer;
    void rawGnssAlign();
    double last_check_gnss_time = 0;

};

typedef AckMsckfLam::Ptr AckMsckfLamPtr;
typedef AckMsckfLam::ConstPtr AckMsckfLamConstPtr;

} // namespace ack_msckf_lam

#endif
