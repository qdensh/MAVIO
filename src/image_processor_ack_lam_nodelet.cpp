/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <ack_msckf_lam/image_processor_ack_lam_nodelet.h>

namespace ack_msckf_lam {
void ImageProcessorAckLamNodelet::onInit() {
  img_processor_ack_lam_ptr.reset(new ImageProcessorAckLam(getPrivateNodeHandle()));
  if (!img_processor_ack_lam_ptr->initialize()) {
    ROS_ERROR("Cannot initialize Image Processor Wgo...");
    return;
  }
  return;
}

PLUGINLIB_EXPORT_CLASS(ack_msckf_lam::ImageProcessorAckLamNodelet, nodelet::Nodelet);

} // end namespace ack_msckf_lam