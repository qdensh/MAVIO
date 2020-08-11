cmake_minimum_required(VERSION 2.8.12)
project(ack_msckf_lam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -g")
add_definitions(-std=c++11)
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
            STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Modify cmake module path if new .cmake files are required
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
  nav_msgs
  sensor_msgs
  geometry_msgs
  eigen_conversions
  tf_conversions
  random_numbers
  message_generation
  nodelet
  image_transport
  cv_bridge
  message_filters
  pcl_conversions
  pcl_ros
  std_srvs
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)

##################
## ROS messages ##
##################
add_message_files(
  FILES

  FeatureMeasurement.msg
  CameraMeasurement.msg
  TrackingInfo.msg

  AckermannDrive.msg 
  AckermannDriveStamped.msg
  
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ack_msckf_lam image_processor_ack_lam
  CATKIN_DEPENDS
    roscpp std_msgs tf nav_msgs sensor_msgs geometry_msgs
    eigen_conversions tf_conversions random_numbers message_runtime
    image_transport cv_bridge message_filters pcl_conversions
    pcl_ros std_srvs
  DEPENDS Boost EIGEN3 OpenCV SUITESPARSE
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Sophus_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
)

# ack_msckf_lam
add_library(ack_msckf_lam
  src/ack_msckf_lam.cpp
  src/utils.cpp
)
add_dependencies(ack_msckf_lam
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(ack_msckf_lam
  ${catkin_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  ${Sophus_LIBRARIES}
)
# ack_msckf_lam nodelet
add_library(ack_msckf_lam_nodelet
  src/ack_msckf_lam_nodelet.cpp
)
add_dependencies(ack_msckf_lam_nodelet
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(ack_msckf_lam_nodelet
  ack_msckf_lam
  ${catkin_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Sophus_LIBRARIES}
)

# Image processor
add_library(image_processor_ack_lam
  src/image_processor_ack_lam.cpp
  src/utils.cpp
)
add_dependencies(image_processor_ack_lam
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(image_processor_ack_lam
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
)
# Image processor nodelet
add_library(image_processor_ack_lam_nodelet
  src/image_processor_ack_lam_nodelet.cpp
)
add_dependencies(image_processor_ack_lam_nodelet
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(image_processor_ack_lam_nodelet
  image_processor_ack_lam
  ${catkin_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Sophus_LIBRARIES}
)

#############
## Install ##
#############

install(TARGETS
  ack_msckf_lam ack_msckf_lam_nodelet image_processor_ack_lam image_processor_ack_lam_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN "*_nodelet.h" EXCLUDE
)

install(FILES nodelets.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

# if(CATKIN_ENABLE_TESTING)
#   # Feature initialization test
#   catkin_add_gtest(test_feature_init
#     test/feature_initialization_test.cpp
#   )
#   add_dependencies(test_feature_init
#     ${${PROJECT_NAME}_EXPORTED_TARGETS}
#     ${catkin_EXPORTED_TARGETS}
#   )
#   target_link_libraries(test_feature_init
#     ${catkin_LIBRARIES}
#   )

#   # Math utils test
#   catkin_add_gtest(test_math_utils
#     test/math_utils_test.cpp
#   )
# endif()
