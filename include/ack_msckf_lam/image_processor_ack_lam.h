/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_IMAGE_PROCESSOR_H
#define MSCKF_VIO_IMAGE_PROCESSOR_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <ack_msckf_lam/CameraMeasurement.h>
#include <ack_msckf_lam/AckermannDriveStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


#include "imu_state.h"
#include "cam_state.h"
#include "utils.h"
#include "math_utils.hpp"

#include <nav_msgs/Odometry.h>


// #include<geometry_msgs/Vector3Stamped>

using namespace std;
using namespace Eigen;
using namespace cv;

namespace ack_msckf_lam {

/*
 * @brief ImageProcessorAckLam Detects and tracks features
 *    in image sequences.
 */
class ImageProcessorAckLam {
public:
  // Constructor
  ImageProcessorAckLam(ros::NodeHandle& n);
  // Disable copy and assign constructors.
  ImageProcessorAckLam(const ImageProcessorAckLam&) = delete;
  ImageProcessorAckLam operator=(const ImageProcessorAckLam&) = delete;

  // Destructor
  ~ImageProcessorAckLam();

  // Initialize the object.
  bool initialize();

  typedef boost::shared_ptr<ImageProcessorAckLam> Ptr;
  typedef boost::shared_ptr<const ImageProcessorAckLam> ConstPtr;

private:

  /*
   * @brief ProcessorConfig Configuration parameters for
   *    feature detection and tracking.
   */
  struct ProcessorConfig {
    int grid_row;
    int grid_col;
    int grid_min_feature_num;
    int grid_max_feature_num;

    int pyramid_levels;
    int patch_size;
    int fast_threshold;
    int max_iteration;
    double track_precision;
    double ransac_threshold;
    //double stereo_threshold;

    // ack matching threshold
    double ack_threshold;
  };

  /*
   * @brief FeatureIDType An alias for unsigned long long int.
   */
  typedef unsigned long long int FeatureIDType;

  /*
   * @brief FeatureMetaData Contains necessary information
   *    of a feature for easy access.
   */
  struct FeatureMetaData {
    FeatureIDType id;
    float response;
    int lifetime;
    cv::Point2f cam0_point;
  //  cv::Point2f cam1_point;
  };

  /*
   * @brief GridFeatures Organize features based on the grid
   *    they belong to. Note that the key is encoded by the
   *    grid index.
   */
  typedef std::map<int, std::vector<FeatureMetaData> > GridFeatures;

  /*
   * @brief keyPointCompareByResponse
   *    Compare two keypoints based on the response.
   */
  static bool keyPointCompareByResponse(
      const cv::KeyPoint& pt1,
      const cv::KeyPoint& pt2) {
    // Keypoint with higher response will be at the
    // beginning of the vector.
    return pt1.response > pt2.response;
  }
  /*
   * @brief featureCompareByResponse
   *    Compare two features based on the response.
   */
  static bool featureCompareByResponse(
      const FeatureMetaData& f1,
      const FeatureMetaData& f2) {
    // Features with higher response will be at the
    // beginning of the vector.
    return f1.response > f2.response;
  }
  /*
   * @brief featureCompareByLifetime
   *    Compare two features based on the lifetime.
   */
  static bool featureCompareByLifetime(
      const FeatureMetaData& f1,
      const FeatureMetaData& f2) {
    // Features with longer lifetime will be at the
    // beginning of the vector.
    return f1.lifetime > f2.lifetime;
  }

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
   * @brief stereoCallback
   *    Callback function for the stereo images.
   * @param cam0_img left image.
   * @param cam1_img right image.
   */
  void cam0Callback(
      const sensor_msgs::ImageConstPtr& cam0_img);

  /*
   * @brief imuCallback
   *    Callback function for the imu message.
   * @param msg IMU msg.
   */
  void imuCallback(const sensor_msgs::ImuConstPtr& msg);

  /*
   * @initializeFirstFrame
   *    Initialize the image processing sequence, which is
   *    bascially detect new features on the first set of
   *    stereo images.
   */
  void initializeFirstFrame();

  /*
   * @brief trackFeatures
   *    Tracker features on the newly received stereo images.
   */
  void trackFeatures();

  /*
   * @addNewFeatures
   *    Detect new features on the image to ensure that the
   *    features are uniformly distributed on the image.
   */
  void addNewFeatures();

  /*
   * @brief pruneGridFeatures
   *    Remove some of the features of a grid in case there are
   *    too many features inside of that grid, which ensures the
   *    number of features within each grid is bounded.
   */
  void pruneGridFeatures();

  /*
   * @brief publish
   *    Publish the features on the current image including
   *    both the tracked and newly detected ones.
   */
  void publish();

  /*
   * @brief drawFeaturesMono
   *    Draw tracked and newly detected features on the left
   *    image only.
   */
  void drawFeaturesMono();
  /*
   * @brief drawFeaturesStereo
   *    Draw tracked and newly detected features on the
   *    stereo images.
   */
  void drawFeaturesStereo();

  /*
   * @brief createImagePyramids
   *    Create image pyramids used for klt tracking.
   */
  void createImagePyramids();

  /*
   * @brief integrateImuData Integrates the IMU gyro readings
   *    between the two consecutive images, which is used for
   *    both tracking prediction and 2-point RANSAC.
   * @return cam0_R_p_c: a rotation matrix which takes a vector
   *    from previous cam0 frame to current cam0 frame.
   * @return cam1_R_p_c: a rotation matrix which takes a vector
   *    from previous cam1 frame to current cam1 frame.
   */
  void integrateImuData(cv::Matx33f& cam0_R_p_c);

  /*
   * @brief predictFeatureTracking Compensates the rotation
   *    between consecutive camera frames so that feature
   *    tracking would be more robust and fast.
   * @param input_pts: features in the previous image to be tracked.
   * @param R_p_c: a rotation matrix takes a vector in the previous
   *    camera frame to the current camera frame.
   * @param intrinsics: intrinsic matrix of the camera.
   * @return compensated_pts: predicted locations of the features
   *    in the current image based on the provided rotation.
   *
   * Note that the input and output points are of pixel coordinates.
   */
  void predictFeatureTracking(
      const std::vector<cv::Point2f>& input_pts,
      const cv::Matx33f& R_p_c,
      const cv::Vec4d& intrinsics,
      std::vector<cv::Point2f>& compenstated_pts);

  /*
   * @brief twoPointRansac Applies two point ransac algorithm
   *    to mark the inliers in the input set.
   * @param pts1: first set of points.
   * @param pts2: second set of points.
   * @param R_p_c: a rotation matrix takes a vector in the previous
   *    camera frame to the current camera frame.
   * @param intrinsics: intrinsics of the camera.
   * @param distortion_model: distortion model of the camera.
   * @param distortion_coeffs: distortion coefficients.
   * @param inlier_error: acceptable error to be considered as an inlier.
   * @param success_probability: the required probability of success.
   * @return inlier_flag: 1 for inliers and 0 for outliers.
   */
  void twoPointRansac(
      const std::vector<cv::Point2f>& pts1,
      const std::vector<cv::Point2f>& pts2,
      const cv::Matx33f& R_p_c,
      const cv::Vec4d& intrinsics,
      const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs,
      const double& inlier_error,
      const double& success_probability,
      std::vector<int>& inlier_markers);
  void undistortPoints(
      const std::vector<cv::Point2f>& pts_in,
      const cv::Vec4d& intrinsics,
      const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs,
      std::vector<cv::Point2f>& pts_out,
      const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
      const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));
  void rescalePoints(
      std::vector<cv::Point2f>& pts1,
      std::vector<cv::Point2f>& pts2,
      float& scaling_factor);
  std::vector<cv::Point2f> distortPoints(
      const std::vector<cv::Point2f>& pts_in,
      const cv::Vec4d& intrinsics,
      const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs);

  
  template <typename T>
  void removeUnmarkedElements(
      const std::vector<T>& raw_vec,
      const std::vector<unsigned char>& markers,
      std::vector<T>& refined_vec) {
    if (raw_vec.size() != markers.size()) {
      ROS_WARN("The input size of raw_vec(%lu) and markers(%lu) does not match...",
          raw_vec.size(), markers.size());
    }
    for (int i = 0; i < markers.size() ; ++i) {
      if (markers[i] == 0) continue;
      refined_vec.push_back(raw_vec[i]);
    }
    return;
  }

  // Indicate if this is the first image message.
  bool is_first_img;

  // ID for the next new feature.
  FeatureIDType next_feature_id;

  // Feature detector
  ProcessorConfig processor_config;
  cv::Ptr<cv::Feature2D> detector_ptr;

  // IMU message buffer.
  std::vector<sensor_msgs::Imu> imu_msg_buffer;

  // Camera calibration parameters
  std::string cam0_distortion_model;
  cv::Vec2i cam0_resolution;
  cv::Vec4d cam0_intrinsics;
  cv::Vec4d cam0_distortion_coeffs;

  // Take a vector from cam0 frame to the IMU frame.
  // JPL
  cv::Matx33d R_cam0_imu;
  cv::Vec3d t_cam0_imu;


  // Previous and current images
  cv_bridge::CvImageConstPtr cam0_prev_img_ptr;
  cv_bridge::CvImageConstPtr cam0_curr_img_ptr;

  // Pyramids for previous and current image
  std::vector<cv::Mat> prev_cam0_pyramid_;
  std::vector<cv::Mat> curr_cam0_pyramid_;

  // Features in the previous and current image.
  boost::shared_ptr<GridFeatures> prev_features_ptr;
  boost::shared_ptr<GridFeatures> curr_features_ptr;

  // Number of features after each outlier removal step.
  int before_tracking;
  int after_tracking;
  int after_matching;
  int after_ransac;

  // Ros node handle
  ros::NodeHandle nh;

  // Subscribers and publishers.
  message_filters::Subscriber<
    sensor_msgs::Image> cam0_img_sub;
  //message_filters::Subscriber<
  //  sensor_msgs::Image> cam1_img_sub;
  //message_filters::TimeSynchronizer<
   // sensor_msgs::Image> stereo_sub;
  ros::Subscriber imu_sub;
  ros::Publisher feature_pub;
  ros::Publisher tracking_info_pub;
  image_transport::Publisher debug_stereo_pub;

  // Debugging
  std::map<FeatureIDType, int> feature_lifetime;
  void updateFeatureLifetime();
  void featureLifetimeStatistics();

  static double WheelBase;
  static double TireBase;
  static double AckRate;
  static double SteerRatio;

  void ackCallback(const ack_msckf_lam::AckermannDriveStamped::ConstPtr& msg);
  // ACK data buffer
  std::vector<ack_msckf_lam::AckermannDriveStamped> ack_msg_buffer;

  bool is_init_bias_set = false;
  void initializeGravityAndBias();

  // State vector
  ACKState ack_state;
  IMUState imu_state;
  CamStateServer cam_states;

  void batchAckProcessing(const double& time_bound);
  void processModel_ack(const double& time,
    const Eigen::Vector3d& m_speed,
    const Eigen::Vector3d& m_gyro);
  void predictNewState_ack(const double& dt,
    const Eigen::Vector3d& gyro,
    const Eigen::Vector3d& speed);
  void predictFeatureTrackingAck(
    const std::vector<cv::Point2f>& input_pts,
    const cv::Vec4d& intrinsics,
    vector<cv::Point2f>& compensated_pts);
  void ackMatch(
    const vector<cv::Point2f>& prev_points,
    const vector<cv::Point2f>& curr_points,
    vector<unsigned char>& inlier_markers);

  void generateInitialGuess(
    const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
    const Eigen::Vector2d& z2, Eigen::Vector3d& p); 

  void twoPointRansac_ack(
      const std::vector<cv::Point2f>& pts1,
      const std::vector<cv::Point2f>& pts2,
      const cv::Matx33f& R_p_c,
      const cv::Vec3d& t_p_c,
      const cv::Vec4d& intrinsics,
      const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs,
      const double& inlier_error,
      const double& success_probability,
      std::vector<int>& inlier_markers);
    
  void initializeGravityAndBias_ack();


  // debug
  bool use_a27_platform;
  bool use_svd_ex;
  bool use_debug;
  bool use_gyro_bias;
  bool use_offline_bias;

  // orb
  cv::Ptr<ORB> orb_detector;
  cv::Ptr<cv::CLAHE> m_clahe;
  bool use_orb;
  bool use_ack_camHeight;
  ros::Subscriber gyro_bias_sub;
  void gyroBiasCallback(const nav_msgs::OdometryConstPtr& gyro_bias_msg_);
  double image_camHeight;
  void predictFeatureTrackingCamHeight(
    const std::vector<cv::Point2f>& input_pts,
    const cv::Vec4d& intrinsics,
    vector<cv::Point2f>& compensated_pts);

  // save time_file
  std::string output_path;
  std::ofstream time_file;
  unsigned long long int global_count =0;
  // TIME
  struct CSVDATA_TIME {
      double time;
      double Dtime;

      double process_time, avg_time, total_time;
  };
  std::vector<struct CSVDATA_TIME> csvData_time;
  double total_time = 0;
  double DfirstTime = 0;
  double Dtime = 0;
  double csv_curr_time = 0;
  ros::Timer csv_timer;
  void csv_timer_callBack(const ros::TimerEvent& event);
  bool is_csv_curr_time_init = false;

};

typedef ImageProcessorAckLam::Ptr ImageProcessorAckLamPtr;
typedef ImageProcessorAckLam::ConstPtr ImageProcessorAckLamConstPtr;

} // end namespace ack_msckf_lam

#endif
