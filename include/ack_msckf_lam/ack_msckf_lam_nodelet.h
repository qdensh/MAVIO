/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_NODELET_H
#define MSCKF_VIO_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ack_msckf_lam/ack_msckf_lam.h>

namespace ack_msckf_lam {
class AckMsckfLamNodelet : public nodelet::Nodelet {
public:
  AckMsckfLamNodelet() { return; }
  ~AckMsckfLamNodelet() { return; }

private:
  virtual void onInit();
  AckMsckfLamPtr ack_msckf_lam_ptr;
};
} // end namespace ack_msckf_lam

#endif

