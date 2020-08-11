/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <ack_msckf_lam/ack_msckf_lam_nodelet.h>


namespace ack_msckf_lam {
void AckMsckfLamNodelet::onInit() {
  ack_msckf_lam_ptr.reset(new AckMsckfLam(getPrivateNodeHandle()));
  if (!ack_msckf_lam_ptr->initialize()) {
    ROS_ERROR("Cannot initialize WGO MSCKF...");
    return;
  }
  return;
}

PLUGINLIB_EXPORT_CLASS(ack_msckf_lam::AckMsckfLamNodelet, nodelet::Nodelet);

} // end namespace ack_msckf_lam