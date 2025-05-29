/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include "astra_camera/point_cloud_proc/point_cloud_xyzrgb.h"
namespace astra_camera {
PointCloudXyzrgbNode::PointCloudXyzrgbNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private) {
  rgb_nh_.reset(new ros::NodeHandle(nh, "color"));
  ros::NodeHandle depth_nh(nh, "depth");
  rgb_it_.reset(new image_transport::ImageTransport(*rgb_nh_));
  depth_it_.reset(new image_transport::ImageTransport(depth_nh));

  // Read parameters
  int queue_size;
  nh_private_.param("queue_size", queue_size, 5);
  bool use_exact_sync;
  nh_private_.param("exact_sync", use_exact_sync, false);

  // Synchronize inputs. Topic subscriptions happen on demand in the connection callback.
  if (use_exact_sync) {
    exact_sync_ = std::make_shared<ExactSynchronizer>(ExactSyncPolicy(queue_size), sub_depth_,
                                                      sub_rgb_, sub_info_);
    exact_sync_->registerCallback(boost::bind(&PointCloudXyzrgbNode::imageCb, this, _1, _2, _3));

  } else {
    sync_ = std::make_shared<Synchronizer>(SyncPolicy(queue_size), sub_depth_, sub_rgb_, sub_info_);
    sync_->registerCallback(boost::bind(&PointCloudXyzrgbNode::imageCb, this, _1, _2, _3));
  }

  // Monitor whether anyone is subscribed to the output
  ros::SubscriberStatusCallback connect_cb = boost::bind(&PointCloudXyzrgbNode::connectCb, this);
  ros::SubscriberStatusCallback disconnectCb =
      boost::bind(&PointCloudXyzrgbNode::disconnectCb, this);
  // Make sure we don't enter connectCb() between advertising and assigning to pub_point_cloud_
  std::lock_guard<std::mutex> lock(connect_mutex_);
  pub_point_cloud_ = depth_nh.advertise<PointCloud>("color/points", 1, connect_cb, disconnectCb);
  save_point_cloud_srv_ = nh_.advertiseService<std_srvs::EmptyRequest, std_srvs::EmptyResponse>(
      "save_point_cloud_xyz_rgb",
      [this](auto&& req, auto&& res) { return this->SavePointCloudXyzrgbCallback(req, res); });
}

PointCloudXyzrgbNode::~PointCloudXyzrgbNode() = default;

void PointCloudXyzrgbNode::connectCb() {
  std::lock_guard<std::mutex> lock(connect_mutex_);
  // parameter for depth_image_transport hint
  std::string depth_image_transport_param = "depth_image_transport";

  // depth image can use different transport.(e.g. compressedDepth)
  image_transport::TransportHints depth_hints("raw", ros::TransportHints(), nh_private_,
                                              depth_image_transport_param);
  sub_depth_.subscribe(*depth_it_, "image_raw", 1, depth_hints);

  // rgb uses normal ros transport hints.
  image_transport::TransportHints hints("raw", ros::TransportHints(), nh_private_);
  sub_rgb_.subscribe(*rgb_it_, "image_raw", 1, hints);
  sub_info_.subscribe(nh_, "color/camera_info", 1);
}

void PointCloudXyzrgbNode::disconnectCb() {
  if (pub_point_cloud_.getNumSubscribers() == 0) {
    sub_depth_.unsubscribe();
    sub_rgb_.unsubscribe();
    sub_info_.unsubscribe();
  }
}

void PointCloudXyzrgbNode::imageCb(const sensor_msgs::ImageConstPtr& depth_msg,
                                   const sensor_msgs::ImageConstPtr& rgb_msg_in,
                                   const sensor_msgs::CameraInfoConstPtr& info_msg) {
  // Check for bad inputs
  if (depth_msg->header.frame_id != rgb_msg_in->header.frame_id) {
    ROS_ERROR_THROTTLE(5, "Depth image frame id [%s] doesn't match RGB image frame id [%s]",
                       depth_msg->header.frame_id.c_str(), rgb_msg_in->header.frame_id.c_str());
    return;
  }

  // Update camera model
  model_.fromCameraInfo(info_msg);

  // Check if the input image has to be resized
  sensor_msgs::ImageConstPtr rgb_msg = rgb_msg_in;
  if (depth_msg->width != rgb_msg->width || depth_msg->height != rgb_msg->height) {
    sensor_msgs::CameraInfo info_msg_tmp = *info_msg;
    info_msg_tmp.width = depth_msg->width;
    info_msg_tmp.height = depth_msg->height;
    float ratio = float(depth_msg->width) / float(rgb_msg->width);
    info_msg_tmp.K[0] *= ratio;
    info_msg_tmp.K[2] *= ratio;
    info_msg_tmp.K[4] *= ratio;
    info_msg_tmp.K[5] *= ratio;
    info_msg_tmp.P[0] *= ratio;
    info_msg_tmp.P[2] *= ratio;
    info_msg_tmp.P[5] *= ratio;
    info_msg_tmp.P[6] *= ratio;
    model_.fromCameraInfo(info_msg_tmp);

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(rgb_msg, rgb_msg->encoding);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv_bridge::CvImage cv_rsz;
    cv_rsz.header = cv_ptr->header;
    cv_rsz.encoding = cv_ptr->encoding;
    // FIXME: avoid out of range exception
    int end_row = std::min<int>(depth_msg->height / ratio, rgb_msg->height);
    cv::resize(cv_ptr->image.rowRange(0, end_row), cv_rsz.image,
               cv::Size(depth_msg->width, depth_msg->height));
    if ((rgb_msg->encoding == enc::RGB8) || (rgb_msg->encoding == enc::BGR8) ||
        (rgb_msg->encoding == enc::MONO8)) {
      rgb_msg = cv_rsz.toImageMsg();
    } else {
      rgb_msg = cv_bridge::toCvCopy(cv_rsz.toImageMsg(), enc::RGB8)->toImageMsg();
    }

  } else {
    rgb_msg = rgb_msg_in;
  }

  // Supported color encodings: RGB8, BGR8, MONO8
  int red_offset, green_offset, blue_offset, color_step;
  if (rgb_msg->encoding == enc::RGB8) {
    red_offset = 0;
    green_offset = 1;
    blue_offset = 2;
    color_step = 3;
  }
  if (rgb_msg->encoding == enc::RGBA8) {
    red_offset = 0;
    green_offset = 1;
    blue_offset = 2;
    color_step = 4;
  } else if (rgb_msg->encoding == enc::BGR8) {
    red_offset = 2;
    green_offset = 1;
    blue_offset = 0;
    color_step = 3;
  } else if (rgb_msg->encoding == enc::BGRA8) {
    red_offset = 2;
    green_offset = 1;
    blue_offset = 0;
    color_step = 4;
  } else if (rgb_msg->encoding == enc::MONO8) {
    red_offset = 0;
    green_offset = 0;
    blue_offset = 0;
    color_step = 1;
  } else {
    try {
      rgb_msg = cv_bridge::toCvCopy(rgb_msg, enc::RGB8)->toImageMsg();
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR_THROTTLE(5, "Unsupported encoding [%s]: %s", rgb_msg->encoding.c_str(), e.what());
      return;
    }
    red_offset = 0;
    green_offset = 1;
    blue_offset = 2;
    color_step = 3;
  }

  // Allocate new point cloud message
  PointCloud::Ptr cloud_msg(new PointCloud);
  cloud_msg->header = depth_msg->header;  // Use depth image time stamp
  cloud_msg->height = depth_msg->height;
  cloud_msg->width = depth_msg->width;
  cloud_msg->is_dense = false;
  cloud_msg->is_bigendian = false;

  sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
  pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

  if (depth_msg->encoding == enc::TYPE_16UC1 || depth_msg->encoding == enc::MONO16) {
    convert<uint16_t>(depth_msg, rgb_msg, cloud_msg, red_offset, green_offset, blue_offset,
                      color_step);
  } else if (depth_msg->encoding == enc::TYPE_32FC1) {
    convert<float>(depth_msg, rgb_msg, cloud_msg, red_offset, green_offset, blue_offset,
                   color_step);
  } else {
    ROS_ERROR_THROTTLE(5, "Depth image has unsupported encoding [%s]", depth_msg->encoding.c_str());
    return;
  }

  pub_point_cloud_.publish(cloud_msg);
  if (save_cloud_) {
    save_cloud_ = false;
    auto now = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    auto current_path = boost::filesystem::current_path().string();
    std::string filename = current_path + "/point_cloud/points_xyz_rgb_" + ss.str() + ".ply";
    if (!boost::filesystem::exists(current_path + "/point_cloud")) {
      boost::filesystem::create_directory(current_path + "/point_cloud");
    }
    ROS_INFO_STREAM("Saving point cloud to " << filename);
    saveRGBPointToPly(cloud_msg, filename);
  }
}

template <typename T>
void PointCloudXyzrgbNode::convert(const sensor_msgs::ImageConstPtr& depth_msg,
                                   const sensor_msgs::ImageConstPtr& rgb_msg,
                                   const PointCloud2::Ptr& cloud_msg, int red_offset,
                                   int green_offset, int blue_offset, int color_step) {
  // Use correct principal point from calibration
  float center_x = model_.cx();
  float center_y = model_.cy();

  // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
  double unit_scaling = DepthTraits<T>::toMeters(T(1));
  float constant_x = unit_scaling / model_.fx();
  float constant_y = unit_scaling / model_.fy();
  float bad_point = std::numeric_limits<float>::quiet_NaN();

  const T* depth_row = reinterpret_cast<const T*>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(T);
  const uint8_t* rgb = &rgb_msg->data[0];
  int rgb_skip = rgb_msg->step - rgb_msg->width * color_step;

  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud_msg, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud_msg, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud_msg, "b");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_a(*cloud_msg, "a");

  for (int v = 0; v < int(cloud_msg->height); ++v, depth_row += row_step, rgb += rgb_skip) {
    for (int u = 0; u < int(cloud_msg->width); ++u, rgb += color_step, ++iter_x, ++iter_y, ++iter_z,
             ++iter_a, ++iter_r, ++iter_g, ++iter_b) {
      T depth = depth_row[u];

      // Check for invalid measurements
      if (!DepthTraits<T>::valid(depth)) {
        *iter_x = *iter_y = *iter_z = bad_point;
      } else {
        // Fill in XYZ
        *iter_x = (u - center_x) * depth * constant_x;
        *iter_y = (v - center_y) * depth * constant_y;
        *iter_z = DepthTraits<T>::toMeters(depth);
      }

      // Fill in color
      *iter_a = 255;
      *iter_r = rgb[red_offset];
      *iter_g = rgb[green_offset];
      *iter_b = rgb[blue_offset];
    }
  }
}

bool PointCloudXyzrgbNode::SavePointCloudXyzrgbCallback(std_srvs::Empty::Request& request,
                                                        std_srvs::Empty::Response& response) {
  (void)request;
  (void)response;
  ROS_INFO("SavePointCloudXyzrgbCallback");
  save_cloud_ = true;
  return true;
}

}  // namespace astra_camera
