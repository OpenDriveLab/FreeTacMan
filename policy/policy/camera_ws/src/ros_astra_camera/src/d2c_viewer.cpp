/***************************************************************************/
/*                                                                        */
/* Copyright (c) 2013-2022 Orbbec 3D Technology, Inc                      */
/*                                                                        */
/* PROPRIETARY RIGHTS of Orbbec 3D Technology are involved in the         */
/* subject matter of this material. All manufacturing, reproduction, use, */
/* and sales rights pertaining to this subject matter are governed by the */
/* license agreement. The recipient of this software implicitly accepts   */
/* the terms of the license.                                              */
/*                                                                        */
/**************************************************************************/

#include "astra_camera/d2c_viewer.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>

namespace astra_camera {
D2CViewer::D2CViewer(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private) {
  rgb_sub_.subscribe(nh_, "color/image_raw", 1);
  depth_sub_.subscribe(nh_, "depth/image_raw", 1);
  sync_ = std::make_unique<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(10), rgb_sub_,
                                                                        depth_sub_);
  sync_->registerCallback(boost::bind(&D2CViewer::messageCallback, this, _1, _2));
  d2c_pub_ = nh_.advertise<sensor_msgs::Image>("depth_to_color/image_raw", 1);
}
D2CViewer::~D2CViewer() = default;

void D2CViewer::messageCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                                const sensor_msgs::ImageConstPtr& depth_msg) {
  if (rgb_msg->width != depth_msg->width || rgb_msg->height != depth_msg->height) {
    ROS_ERROR("rgb and depth image size not match(%d, %d) vs (%d, %d)", rgb_msg->width,
              rgb_msg->height, depth_msg->width, depth_msg->height);
    return;
  }
  auto rgb_img_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);
  auto depth_img_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  cv::Mat gray_depth, depth_img, d2c_img;
  depth_img_ptr->image.convertTo(gray_depth, CV_8UC1);
  cv::cvtColor(gray_depth, depth_img, cv::COLOR_GRAY2RGB);
  cv::bitwise_or(rgb_img_ptr->image, depth_img, d2c_img);
  sensor_msgs::ImagePtr d2c_msg =
      cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::RGB8, d2c_img)
          .toImageMsg();
  d2c_msg->header = rgb_msg->header;
  d2c_pub_.publish(d2c_msg);
}
}  // namespace astra_camera
