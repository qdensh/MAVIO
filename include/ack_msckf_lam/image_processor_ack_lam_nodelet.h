/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef IMAGE_PROCESSOR_NODELET_H
#define IMAGE_PROCESSOR_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ack_msckf_lam/image_processor_ack_lam.h>

namespace ack_msckf_lam {
class ImageProcessorAckLamNodelet : public nodelet::Nodelet {
public:
  ImageProcessorAckLamNodelet() { return; }
  ~ImageProcessorAckLamNodelet() { return; }

private:
  virtual void onInit();
  ImageProcessorAckLamPtr img_processor_ack_lam_ptr;
};
} // end namespace ack_msckf_lam

#endif

