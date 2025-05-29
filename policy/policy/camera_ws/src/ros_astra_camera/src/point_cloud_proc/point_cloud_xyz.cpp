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

#include "astra_camera/point_cloud_proc/point_cloud_xyz.h"

namespace astra_camera {
PointCloudXyzNode::PointCloudXyzNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private)
    : nh_(nh), nh_private_(nh_private) {
  nh_private_.param<int>("queue_size", queue_size_, 5);
  ros::SubscriberStatusCallback connect_cb = std::bind(&PointCloudXyzNode::connectCb, this);
  ros::SubscriberStatusCallback disconnect_cb = std::bind(&PointCloudXyzNode::disconnectCb, this);
  it_.reset(new image_transport::ImageTransport(nh));

  std::lock_guard<decltype(connect_mutex_)> lock_(connect_mutex_);
  pub_point_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("depth/points", queue_size_,
                                                             connect_cb, disconnect_cb);
  save_point_cloud_srv_ = nh_.advertiseService<std_srvs::EmptyRequest, std_srvs::EmptyResponse>(
      "save_point_cloud_xyz",
      [this](auto &&req, auto &&res) { return this->SavePointCloudXyzCallback(req, res); });
}

PointCloudXyzNode::~PointCloudXyzNode() = default;

void PointCloudXyzNode::connectCb() {
  std::lock_guard<std::mutex> lock(connect_mutex_);
  if (!sub_depth_) {
    image_transport::TransportHints hints("raw", ros::TransportHints(), nh_private_);
    sub_depth_ = it_->subscribeCamera("depth/image_raw", queue_size_, &PointCloudXyzNode::depthCb,
                                      this, hints);
  }
}

void PointCloudXyzNode::disconnectCb() {
  if (pub_point_cloud_.getNumSubscribers() == 0) {
    sub_depth_.shutdown();
  }
}

void PointCloudXyzNode::depthCb(const sensor_msgs::ImageConstPtr &depth_msg,
                                const sensor_msgs::CameraInfoConstPtr &info_msg) {
  PointCloud2::Ptr cloud_msg(new PointCloud2);
  cloud_msg->header = depth_msg->header;
  cloud_msg->height = depth_msg->height;
  cloud_msg->width = depth_msg->width;
  cloud_msg->is_dense = false;
  cloud_msg->is_bigendian = false;
  sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
  pcd_modifier.setPointCloud2FieldsByString(1, "xyz");

  // Update camera model
  model_.fromCameraInfo(info_msg);

  if (depth_msg->encoding == enc::TYPE_16UC1 || depth_msg->encoding == enc::MONO16) {
    convert<uint16_t>(depth_msg, cloud_msg, model_);
  } else if (depth_msg->encoding == enc::TYPE_32FC1) {
    convert<float>(depth_msg, cloud_msg, model_);
  } else {
    ROS_WARN_THROTTLE(5, "Depth image has unsupported encoding [%s]", depth_msg->encoding.c_str());
    return;
  }
  if (save_cloud_) {
    save_cloud_ = false;
    auto now = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    auto current_path = boost::filesystem::current_path().string();
    std::string filename = current_path + "/point_cloud/points_xyz_" + ss.str() + ".ply";
    if (!boost::filesystem::exists(current_path + "/point_cloud")) {
      boost::filesystem::create_directory(current_path + "/point_cloud");
    }
    ROS_INFO_STREAM("Saving point cloud to " << filename);
    savePointToPly(cloud_msg, filename);
  }
  pub_point_cloud_.publish(cloud_msg);
}

bool PointCloudXyzNode::SavePointCloudXyzCallback(std_srvs::Empty::Request &request,
                                                  std_srvs::Empty::Response &response) {
  (void)request;
  (void)response;
  ROS_INFO("SavePointCloudXyzCallback");
  save_cloud_ = true;
  return true;
}

}  // namespace astra_camera
