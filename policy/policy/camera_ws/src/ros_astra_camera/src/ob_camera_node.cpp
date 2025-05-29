/**************************************************************************/
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

#include "astra_camera/ob_camera_node.h"

#include <cv_bridge/cv_bridge.h>

#include <boost/algorithm/string.hpp>
#include <utility>

namespace astra_camera {
OBCameraNode::OBCameraNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private,
                           std::shared_ptr<openni::Device> device, bool use_uvc_camera)
    : nh_(nh),
      nh_private_(nh_private),
      device_(std::move(device)),
      use_uvc_camera_(use_uvc_camera) {
  init();
}

OBCameraNode::~OBCameraNode() {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  ROS_INFO_STREAM("OBCameraNode::~OBCameraNode");
  clean();
  ROS_INFO_STREAM("OBCameraNode::~OBCameraNode done.");
}

void OBCameraNode::clean() {
  is_running_.store(false);
  ROS_INFO_STREAM("OBCameraNode::clean stop poll frame");
  run_poll_frame_thread_ = false;
  if (poll_frame_thread_ && poll_frame_thread_->joinable()) {
    poll_frame_thread_->join();
  }
  ROS_INFO_STREAM("OBCameraNode::clean stop poll frame done");
  ROS_INFO_STREAM("OBCameraNode::clean stop tf");
  if (tf_thread_ != nullptr && tf_thread_->joinable()) {
    tf_thread_->join();
  }
  ROS_INFO_STREAM("OBCameraNode::clean stop tf done.");
  stopStreams();
  for (const auto& stream_index : IMAGE_STREAMS) {
    if (streams_[stream_index]) {
      streams_[stream_index]->destroy();
      streams_[stream_index].reset();
    }
  }
  ROS_INFO_STREAM("OBCameraNode::clean stop streams done.");
  if (uvc_camera_driver_ != nullptr) {
    ROS_INFO_STREAM("OBCameraNode::stop uvc camera.");
    uvc_camera_driver_.reset();
    ROS_INFO_STREAM("OBCameraNode::stop uvc camera done.");
  }
  if (device_ && device_->isValid()) {
    ROS_INFO_STREAM("OBCameraNode::clean close device");
    ROS_INFO_STREAM("OBCameraNode::clean close device done.");
  }
  ROS_INFO("OBCameraNode::clean stop streams done.");
}

void OBCameraNode::init() {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  if (!device_->isValid()) {
    ROS_ERROR_STREAM("OBCameraNode::init device is not valid");
    throw std::runtime_error("OBCameraNode::init device is not valid");
  }
  is_running_.store(true);
  device_info_ = device_->getDeviceInfo();
  setupConfig();
  setupTopics();
  if (enable_d2c_viewer_) {
    d2c_filter_ = std::make_unique<D2CViewer>(nh_, nh_private_);
  }
  setupUVCCamera();
  setupD2CConfig();
  for (const auto& stream_index : IMAGE_STREAMS) {
    if (streams_[stream_index]) {
      save_images_[stream_index] = false;
    }
  }
  init_ir_gain_ = getIRGain();
  init_ir_exposure_ = getIRExposure();
  if (enable_reconfigure_) {
    reconfigure_server_ = std::make_unique<ReconfigureServer>(nh_private_);
    reconfigure_server_->setCallback([this](const AstraConfig& config, uint32_t level) {
      this->reconfigureCallback(config, level);
    });
  }
  if (keep_alive_) {
    keep_alive_timer_ =
        nh_.createTimer(ros::Duration(keep_alive_interval_), &OBCameraNode::sendKeepAlive, this);
  }
  if (enable_pointcloud_) {
    point_cloud_xyz_node_ = std::make_unique<PointCloudXyzNode>(nh_, nh_private_);
  }
  if (enable_pointcloud_xyzrgb_) {
    point_cloud_xyzrgb_node_ = std::make_unique<PointCloudXyzrgbNode>(nh_, nh_private_);
  }
  run_poll_frame_thread_ = true;
  poll_frame_thread_ = std::make_unique<std::thread>(&OBCameraNode::pollFrame, this);
  initialized_ = true;

  for (const auto& stream_index : IMAGE_STREAMS) {
    if (!enable_[stream_index]) {
      continue;
    }
    if (use_uvc_camera_ && stream_index == COLOR) {
      continue;
    }
    sensor_msgs::CameraInfo camera_info;
    if (stream_index == COLOR) {
      camera_info = getColorCameraInfo();
    } else if (stream_index == DEPTH) {
      camera_info = getDepthCameraInfo();
    } else {
      int width = width_[stream_index];
      int height = height_[stream_index];
      double f = getFocalLength(stream_index, width);
      camera_info = getIRCameraInfo(width, height, f);
    }

    camera_info.header.stamp = ros::Time::now();
    camera_info.header.frame_id = optical_frame_id_[stream_index];
    camera_info_publishers_.at(stream_index).publish(camera_info);
  }
  ROS_INFO_STREAM("OBCameraNode initialized");
}

void OBCameraNode::setupConfig() {
  stream_name_[DEPTH] = "depth";
  unit_step_size_[DEPTH] = sizeof(uint16_t);
  format_[DEPTH] = openni::PIXEL_FORMAT_DEPTH_1_MM;
  image_format_[DEPTH] = CV_16UC1;
  encoding_[DEPTH] = sensor_msgs::image_encodings::TYPE_16UC1;

  stream_name_[COLOR] = "color";
  unit_step_size_[COLOR] = 3;
  format_[COLOR] = openni::PIXEL_FORMAT_RGB888;
  image_format_[COLOR] = CV_8UC3;
  encoding_[COLOR] = sensor_msgs::image_encodings::RGB8;

  stream_name_[INFRA1] = "ir";
  unit_step_size_[INFRA1] = sizeof(uint8_t);
  format_[INFRA1] = openni::PIXEL_FORMAT_GRAY8;
  image_format_[INFRA1] = CV_8UC1;
  encoding_[INFRA1] = sensor_msgs::image_encodings::MONO8;

  stream_name_[INFRA2] = "ir2";
  unit_step_size_[INFRA2] = sizeof(uint8_t);
  format_[INFRA2] = openni::PIXEL_FORMAT_GRAY8;
  image_format_[INFRA2] = CV_8UC1;
  encoding_[INFRA2] = sensor_msgs::image_encodings::MONO8;
}

void OBCameraNode::setupCameraInfoManager() {
  ir_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(
      nh_private_, "ir_camera", ir_info_uri_);
  if (!use_uvc_camera_ && device_->hasSensor(openni::SENSOR_COLOR)) {
    color_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(
        nh_, "rgb_camera", color_info_uri_);
  }
}

void OBCameraNode::setupDevices() {
  // first check
  for (const auto& stream_index : IMAGE_STREAMS) {
    if (enable_[stream_index] && !device_->hasSensor(stream_index.first)) {
      std::stringstream ss;
      if(stream_index == COLOR && use_uvc_camera_){
        //Do nothing
      } else {
        ss << "OBCameraNode:: Failed to get sensor:" << stream_name_[stream_index]
           << " from device. maybe hardware has some problem, or the device is not supported.";
      }
      ROS_WARN_STREAM("check device status " << device_->isValid());
      ROS_WARN_STREAM(ss.str());
    }
  }
  for (const auto& stream_index : IMAGE_STREAMS) {
    stream_started_[stream_index] = false;
    if (enable_[stream_index] && device_->hasSensor(stream_index.first)) {
      if (use_uvc_camera_ && stream_index == COLOR) {
        continue;
      }
      auto stream = std::make_shared<openni::VideoStream>();
      auto status = stream->create(*device_, stream_index.first);
      if (status != openni::STATUS_OK) {
        std::stringstream ss;
        ss << "OBCameraNode::setupDevices failed to create stream " << stream_name_[stream_index]
           << " with error: " << openni::OpenNI::getExtendedError();
        ROS_ERROR_STREAM(ss.str());
        throw std::runtime_error(ss.str());
      }
      streams_[stream_index] = stream;
    } else {
      if(stream_index == COLOR && use_uvc_camera_){
        //Do nothing
      } else {
        ROS_WARN_STREAM("OBCameraNode::setupDevices failed to create stream "
                      << stream_name_[stream_index] << ", stream is disabled or sensor not found");
      }
      if (streams_[stream_index]) {
        ROS_ERROR_STREAM("code should not reach here, it's MUST a hidden bug");
        ROS_WARN_STREAM("OBCameraNode::setupDevices stream " << stream_name_[stream_index]
                                                             << " is already created, closing it");
        streams_[stream_index].reset();
      }
      enable_[stream_index] = false;
    }
  }
}

void OBCameraNode::setupFrameCallback() {
  for (const auto& stream_index : IMAGE_STREAMS) {
    if (enable_[stream_index] && device_->hasSensor(stream_index.first)) {
      auto frame_callback = [this,
                             stream_index = stream_index](const openni::VideoFrameRef& frame) {
        this->onNewFrameCallback(frame, stream_index);
      };
      stream_frame_callback_[stream_index] = frame_callback;
    }
  }
}

void OBCameraNode::setupSyncMode() {
  OBDEVICE_MODE mode = OBDEVICE_MODE::OBDEVICE_MODE_ROUTINE;
  std::string mode_str = "none";
  if (multi_device_sync_mode_ == MultiDeviceSyncMode::Master) {
    mode = OBDEVICE_MODE::OBDEVICE_MODE_MASTER;
    mode_str = "master";
  } else if (multi_device_sync_mode_ == MultiDeviceSyncMode::Slave) {
    mode = OBDEVICE_MODE::OBDEVICE_MODE_SLAVE;
    mode_str = "slave";
  }
  if (multi_device_sync_mode_ != MultiDeviceSyncMode::None) {
    ROS_ERROR_STREAM("Set device" << getSerialNumber() << " sync mode to " << mode_str);
    int value = mode;
    auto status = device_->setProperty(XN_MODULE_PROPERTY_SET_DEVICE_MODE, value);
    if (status != openni::STATUS_OK) {
      ROS_ERROR_STREAM("Set multi device sync mode failed with error "
                       << openni::OpenNI::getExtendedError());
    }
  }

}

void OBCameraNode::setupVideoMode() {
  if (!use_uvc_camera_ && enable_[INFRA1] && enable_[COLOR]) {
    ROS_WARN_STREAM(
        "Infrared and Color streams are enabled. "
        "Infrared stream will be disabled.");
    enable_[INFRA1] = false;
  }
  for (const auto& stream_index : IMAGE_STREAMS) {
    supported_video_modes_[stream_index] = std::vector<openni::VideoMode>();
    if (device_->hasSensor(stream_index.first) && enable_[stream_index]) {
      if (use_uvc_camera_ && stream_index == COLOR) {
        continue;
      }
      auto stream = streams_[stream_index];
      const auto& sensor_info = stream->getSensorInfo();
      const auto& supported_video_modes = sensor_info.getSupportedVideoModes();
      int size = supported_video_modes.getSize();
      for (int i = 0; i < size; i++) {
        supported_video_modes_[stream_index].emplace_back(supported_video_modes[i]);
      }
      openni::VideoMode video_mode, default_video_mode;
      video_mode.setResolution(width_[stream_index], height_[stream_index]);
      default_video_mode.setResolution(width_[stream_index], height_[stream_index]);
      video_mode.setFps(fps_[stream_index]);
      video_mode.setPixelFormat(format_[stream_index]);
      default_video_mode.setPixelFormat(format_[stream_index]);
      bool is_supported_mode = false;
      bool is_default_mode_supported = false;
      for (const auto& item : supported_video_modes_[stream_index]) {
        if (video_mode == item) {
          is_supported_mode = true;
          stream_video_mode_[stream_index] = video_mode;
          break;
        }
        if (default_video_mode.getResolutionX() == item.getResolutionX() &&
            default_video_mode.getResolutionY() == item.getResolutionY() &&
            default_video_mode.getPixelFormat() == item.getPixelFormat()) {
          default_video_mode.setFps(item.getFps());
          is_default_mode_supported = true;
        }
      }
      if (!is_supported_mode) {
        ROS_WARN_STREAM("Video mode " << video_mode << " is not supported. ");
        if (is_default_mode_supported) {
          ROS_WARN_STREAM("Default video mode " << default_video_mode
                                                << " is supported. "
                                                   "Stream will be enabled.");
          stream_video_mode_[stream_index] = default_video_mode;
          video_mode = default_video_mode;
          is_supported_mode = true;
        } else {
          ROS_WARN_STREAM("Default video mode " << default_video_mode
                                                << "is not supported. "
                                                   "Stream will be disabled.");
          enable_[stream_index] = false;
          ROS_WARN_STREAM("Supported video modes: ");
          for (const auto& item : supported_video_modes_[stream_index]) {
            ROS_WARN_STREAM(item);
          }
        }
      }
      if (is_supported_mode) {
        ROS_INFO_STREAM("set " << stream_name_[stream_index] << " video mode " << video_mode);
        images_[stream_index] = cv::Mat(height_[stream_index], width_[stream_index],
                                        image_format_[stream_index], cv::Scalar(0, 0, 0));
      }
    }
  }
}

void OBCameraNode::setupD2CConfig() {
  if (!depth_align_ && !enable_pointcloud_xyzrgb_) {
    return;
  }
  auto color_width = width_[COLOR];
  auto color_height = height_[COLOR];
  setImageRegistrationMode(depth_align_);
  setDepthColorSync(color_depth_synchronization_);
  if (depth_align_ || enable_pointcloud_xyzrgb_) {
    setDepthToColorResolution(color_width, color_height);
  }
}

void OBCameraNode::startStream(const stream_index_pair& stream_index) {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  if (!is_running_) {
    ROS_ERROR_STREAM("Device is not running.");
    return;
  }
  if (!enable_[stream_index]) {
    ROS_WARN_STREAM("Stream " << stream_name_[stream_index] << " is not enabled.");
    return;
  }
  if (stream_index == COLOR && use_uvc_camera_) {
    return;
  }
  auto pid = device_->getDeviceInfo().getUsbProductId();
  if (pid == ASTRA_PRO_DEPTH_PID && stream_index == COLOR) {
    return;
  }
  if (stream_started_[stream_index]) {
    ROS_WARN_STREAM("Stream " << stream_name_[stream_index] << " is already started.");
    return;
  }
  ROS_INFO_STREAM("Start " << stream_name_[stream_index] << " stream.");
  bool has_subscribers = image_publishers_[stream_index].getNumSubscribers() > 0;
  if (!has_subscribers) {
    ROS_INFO_STREAM("No subscribers for " << stream_name_[stream_index]
                                          << " stream. "
                                             "Stream will be disabled.");
    return;
  }
  CHECK(stream_video_mode_.count(stream_index));
  auto video_mode = stream_video_mode_.at(stream_index);
  CHECK(streams_.count(stream_index));
  if (!streams_[stream_index].get() && device_->hasSensor(stream_index.first)) {
    ROS_WARN_STREAM("stream is not created, that is anomalous, but we'll try to create it.");
    auto stream = std::make_shared<openni::VideoStream>();
    auto status = stream->create(*device_, stream_index.first);
    if (status != openni::STATUS_OK) {
      ROS_ERROR_STREAM("Can't create " << stream_name_[stream_index] << " stream. "
                                       << "OpenNI error: " << openni::OpenNI::getExtendedError());
      return;
    }
    streams_[stream_index] = stream;
  }
  CHECK(streams_[stream_index].get());
  streams_[stream_index]->setVideoMode(video_mode);
  streams_[stream_index]->setMirroringEnabled(false);

  if(stream_index == INFRA1 && !ir_ae_) {
    setIRAutoExposure(false);
    setIRGain(ir_gain_);
    setIRExposure(ir_exposure_);
  }

  if (stream_index == DEPTH) {
    streams_[stream_index]->setProperty(XN_STREAM_PROPERTY_SOFTWARE_FILTER, soft_filter_);
    if (soft_filter_max_diff_ != 0) {
      streams_[stream_index]->setProperty(XN_STREAM_PROPERTY_DEPTH_MAX_DIFF, soft_filter_max_diff_);
    }
    if (soft_filter_max_speckle_size_ != 0) {
      streams_[stream_index]->setProperty(XN_STREAM_PROPERTY_DEPTH_MAX_SPECKLE_SIZE,
                                          soft_filter_max_speckle_size_);
    }
  }

  auto status = streams_[stream_index]->start();
  if (status == openni::STATUS_OK) {
    stream_started_[stream_index] = true;
    ROS_INFO_STREAM(stream_name_[stream_index] << " is started");
  } else {
    ROS_ERROR_STREAM("openni status " << status);
    ROS_ERROR_STREAM("current errno " << errno << " system error string " << strerror(errno));
    ROS_ERROR_STREAM(stream_name_[stream_index] << " start failed with ERROR "
                                                << openni::OpenNI::getExtendedError());
  }
}

void OBCameraNode::stopStream(const stream_index_pair& stream_index) {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  if (!enable_[stream_index]) {
    ROS_WARN_STREAM("Stream " << stream_name_[stream_index] << " is not enabled.");
    return;
  }
  if (!stream_started_[stream_index]) {
    ROS_WARN_STREAM("Stream " << stream_name_[stream_index] << " is not started.");
    return;
  }
  ROS_INFO_STREAM("Stop " << stream_name_[stream_index] << " stream.");
  ROS_INFO_STREAM("OBCameraNode::stopStream stop");
  streams_[stream_index]->stop();
  ROS_INFO_STREAM("OBCameraNode::stopStream stop");
  ROS_INFO_STREAM("Stop stream " << stream_name_[stream_index] << " done.");
  stream_started_[stream_index] = false;
}

void OBCameraNode::startStreams() {
  for (const auto& stream_index : IMAGE_STREAMS) {
    if (enable_[stream_index] && !stream_started_[stream_index]) {
      startStream(stream_index);
    }
  }
}

void OBCameraNode::stopStreams() {
  for (const auto& stream_index : IMAGE_STREAMS) {
    if (stream_started_[stream_index]) {
      stopStream(stream_index);
    }
  }
}

void OBCameraNode::getParameters() {
  camera_name_ = nh_private_.param<std::string>("camera_name", "camera");
  base_frame_id_ = camera_name_ + "_link";
  for (const auto& stream_index : IMAGE_STREAMS) {
    frame_id_[stream_index] = camera_name_ + "_" + stream_name_[stream_index] + "_frame";
    optical_frame_id_[stream_index] =
        camera_name_ + "_" + stream_name_[stream_index] + "_optical_frame";
  }
  for (const auto& stream_index : IMAGE_STREAMS) {
    std::string param_name = stream_name_[stream_index] + "_width";
    width_[stream_index] = nh_private_.param<int>(param_name, IMAGE_WIDTH);
    param_name = stream_name_[stream_index] + "_height";
    height_[stream_index] = nh_private_.param<int>(param_name, IMAGE_HEIGHT);
    param_name = stream_name_[stream_index] + "_fps";
    fps_[stream_index] = nh_private_.param<int>(param_name, IMAGE_FPS);
    param_name = "enable_" + stream_name_[stream_index];
    enable_[stream_index] = nh_private_.param<bool>(param_name, false);
    param_name = "flip_" + stream_name_[stream_index];
    flip_image_[stream_index] = nh_private_.param<bool>(param_name, false);
  }

  ir_ae_ = nh_private_.param<bool>("ir_ae", true);
  ir_gain_ = nh_private_.param<int>("ir_gain", init_ir_gain_);
  ir_exposure_ = nh_private_.param<int>("ir_exposure", init_ir_exposure_);

  for (const auto& stream_index : IMAGE_STREAMS) {
    depth_aligned_frame_id_[stream_index] = optical_frame_id_[COLOR];
  }
  publish_tf_ = nh_private_.param<bool>("publish_tf", true);
  color_depth_synchronization_ = nh_private_.param<bool>("color_depth_synchronization", false);
  tf_publish_rate_ = nh_private_.param<double>("tf_publish_rate", 10.0);
  color_roi_.x = nh_private_.param<int>("color_roi_x", -1);
  color_roi_.y = nh_private_.param<int>("color_roi_y", -1);
  color_roi_.width = nh_private_.param<int>("color_roi_width", -1);
  color_roi_.height = nh_private_.param<int>("color_roi_height", -1);
  depth_roi_.x = nh_private_.param<int>("depth_roi_x", -1);
  depth_roi_.y = nh_private_.param<int>("depth_roi_y", -1);
  depth_roi_.width = nh_private_.param<int>("depth_roi_width", -1);
  depth_roi_.height = nh_private_.param<int>("depth_roi_height", -1);
  depth_scale_ = nh_private_.param<int>("depth_scale", 1);
  enable_reconfigure_ = nh_private_.param<bool>("enable_reconfigure", false);
  depth_align_ = nh_private_.param<bool>("depth_align", false);
  ir_info_uri_ = nh_private_.param<std::string>("ir_info_uri", "");
  color_info_uri_ = nh_private_.param<std::string>("color_info_uri", "");
  enable_d2c_viewer_ = nh_private_.param<bool>("enable_d2c_viewer", false);
  keep_alive_ = nh_private_.param<bool>("keep_alive", false);
  keep_alive_interval_ = nh_private_.param<int>("keep_alive_interval", 15);
  enable_pointcloud_ = nh_private_.param<bool>("enable_point_cloud", false);
  enable_pointcloud_xyzrgb_ = nh_private_.param<bool>("enable_point_cloud_xyzrgb", false);
  enable_publish_extrinsic_ = nh_private_.param<bool>("enable_publish_extrinsic", false);
  std::string sync_mode = nh_private_.param<std::string>("multi_device_sync_mode", "none");
  multi_device_sync_mode_ = getMultiDeviceSyncMode(sync_mode);
  if (depth_align_ && !device_->hasSensor(openni::SENSOR_COLOR) && !use_uvc_camera_) {
    ROS_WARN("No color sensor found, depth align will be disabled");
    depth_align_ = false;
  }
  if (enable_pointcloud_xyzrgb_) {
    depth_align_ = true;
  }
  soft_filter_ = nh_private_.param<int>("soft_filter", 2);
  soft_filter_max_diff_ = nh_private_.param<int>("soft_filter_max_diff", 0);
  soft_filter_max_speckle_size_ = nh_private_.param<int>("soft_filter_max_speckle_size", 0);
}

void OBCameraNode::setupTopics() {
  getParameters();
  setupCameraInfoManager();
  setupFrameCallback();
  setupDevices();
  setupCameraCtrlServices();
  setupPublishers();
  setupSyncMode();
  setupVideoMode();
  getCameraParams();
  publishStaticTransforms();
}

void OBCameraNode::setupUVCCamera() {
  if (use_uvc_camera_) {
    ROS_INFO("OBCameraNode::setupUVCCamera");
    auto color_camera_info = getColorCameraInfo();
    auto serial_number = getSerialNumber();
    try {
      uvc_camera_driver_ =
          std::make_shared<UVCCameraDriver>(nh_, nh_private_, color_camera_info, serial_number);
    } catch (const std::exception& e) {
      ROS_ERROR_STREAM("Failed to initialize UVC camera: " << e.what());
      clean();
      throw e;
    }
  } else {
    uvc_camera_driver_ = nullptr;
  }
}

void OBCameraNode::imageSubscribedCallback(const stream_index_pair& stream_index) {
  ROS_INFO_STREAM("Image stream " << stream_name_[stream_index] << " subscribed");
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  if (!initialized_ || !is_running_) {
    ROS_WARN_STREAM("Camera not initialized, subscribing to stream " << stream_name_[stream_index]);
    return;
  }
  if (stream_started_[stream_index]) {
    return;
  }
  startStream(stream_index);
}

void OBCameraNode::imageUnsubscribedCallback(const stream_index_pair& stream_index) {
  ROS_INFO_STREAM("Image stream " << stream_name_[stream_index] << " unsubscribed");
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  if (!stream_started_[stream_index]) {
    return;
  }
  auto subscriber_count = image_publishers_[stream_index].getNumSubscribers();
  if (subscriber_count == 0) {
    stopStream(stream_index);
  }
}

void OBCameraNode::setupPublishers() {
  for (const auto& stream_index : IMAGE_STREAMS) {
    std::string name = stream_name_[stream_index];
    camera_info_publishers_[stream_index] =
        nh_.advertise<sensor_msgs::CameraInfo>(name + "/camera_info", 1, true);
    if (enable_[stream_index] && device_->hasSensor(stream_index.first)) {
      ros::SubscriberStatusCallback image_subscribed_cb =
          boost::bind(&OBCameraNode::imageSubscribedCallback, this, stream_index);
      ros::SubscriberStatusCallback image_unsubscribed_cb =
          boost::bind(&OBCameraNode::imageUnsubscribedCallback, this, stream_index);
      image_publishers_[stream_index] = nh_.advertise<sensor_msgs::Image>(
          name + "/image_raw", 1, image_subscribed_cb, image_unsubscribed_cb);
    }
  }
}

void OBCameraNode::publishStaticTF(const ros::Time& t, const tf2::Vector3& trans,
                                   const tf2::Quaternion& q, const std::string& from,
                                   const std::string& to) {
  geometry_msgs::TransformStamped msg;
  msg.header.stamp = t;
  msg.header.frame_id = from;
  msg.child_frame_id = to;
  msg.transform.translation.x = trans[2] / 1000.0;
  msg.transform.translation.y = -trans[0] / 1000.0;
  msg.transform.translation.z = -trans[1] / 1000.0;
  msg.transform.rotation.x = q.getX();
  msg.transform.rotation.y = q.getY();
  msg.transform.rotation.z = q.getZ();
  msg.transform.rotation.w = q.getW();
  static_tf_msgs_.push_back(msg);
}

void OBCameraNode::calcAndPublishStaticTransform() {
  tf2::Quaternion quaternion_optical, zero_rot;
  zero_rot.setRPY(0.0, 0.0, 0.0);
  quaternion_optical.setRPY(-M_PI / 2, 0.0, -M_PI / 2);
  tf2::Vector3 zero_trans(0, 0, 0);
  std::vector<float> rotation, transition;
  for (float& i : camera_params_->r2l_r) {
    rotation.emplace_back(i);
  }
  for (float& i : camera_params_->r2l_t) {
    transition.emplace_back(i);
  }
  auto Q = rotationMatrixToQuaternion(rotation);
  for (int i = 0; i < 9; i++) {
    if (std::isnan(rotation[i])) {
      Q.setRPY(0, 0, 0);
      break;
    }
  }
  if (!use_uvc_camera_ && !device_->hasSensor(openni::SENSOR_COLOR)) {
    Q.setRPY(0, 0, 0);
  }
  Q = quaternion_optical * Q * quaternion_optical.inverse();
  tf2::Vector3 trans(transition[0], transition[1], transition[2]);
  if ((!use_uvc_camera_ && !device_->hasSensor(openni::SENSOR_COLOR)) ||
      std::isnan(transition[0]) || std::isnan(transition[1]) || std::isnan(transition[2])) {
    ROS_WARN("No color sensor found or transition is invalid , setting translation to 0");
    trans[0] = 0;
    trans[1] = 0;
    trans[2] = 0;
  }
  tf2::Transform transform(Q, trans);
  transform = transform.inverse();
  Q = transform.getRotation();
  trans = transform.getOrigin();
  auto tf_timestamp = ros::Time::now();
  publishStaticTF(tf_timestamp, zero_trans, quaternion_optical, frame_id_[COLOR],
                  optical_frame_id_[COLOR]);
  publishStaticTF(tf_timestamp, zero_trans, quaternion_optical, frame_id_[DEPTH],
                  optical_frame_id_[DEPTH]);
  publishStaticTF(tf_timestamp, zero_trans, quaternion_optical, frame_id_[INFRA1],
                  optical_frame_id_[INFRA1]);
  publishStaticTF(tf_timestamp, zero_trans, zero_rot, base_frame_id_, frame_id_[DEPTH]);
  publishStaticTF(tf_timestamp, zero_trans, zero_rot, base_frame_id_, frame_id_[INFRA1]);
  publishStaticTF(tf_timestamp, trans, Q, base_frame_id_, frame_id_[COLOR]);
}

void OBCameraNode::publishDynamicTransforms() {
  ROS_WARN("Publishing dynamic camera transforms (/tf) at %g Hz", tf_publish_rate_);
  static std::mutex mu;
  std::unique_lock<std::mutex> lock(mu);
  while (ros::ok() && is_running_) {
    tf_cv_.wait_for(lock, std::chrono::milliseconds((int)(1000.0 / tf_publish_rate_)),
                    [this] { return (!(is_running_)); });
    {
      auto t = ros::Time::now();
      for (auto& msg : static_tf_msgs_) {
        msg.header.stamp = t;
      }
      CHECK_NOTNULL(dynamic_tf_broadcaster_.get());
      dynamic_tf_broadcaster_->sendTransform(static_tf_msgs_);
    }
  }
}

void OBCameraNode::publishStaticTransforms() {
  if (!publish_tf_) {
    return;
  }
  static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>();
  dynamic_tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>();
  calcAndPublishStaticTransform();
  if (tf_publish_rate_ > 0) {
    CHECK(tf_thread_ == nullptr);
    tf_thread_ = std::make_shared<std::thread>([this]() { this->publishDynamicTransforms(); });
  } else {
    CHECK_NOTNULL(static_tf_broadcaster_.get());
    static_tf_broadcaster_->sendTransform(static_tf_msgs_);
  }
}

void OBCameraNode::setImageRegistrationMode(bool data) {
  if (!device_->isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR)) {
    ROS_WARN_STREAM("Current do not support IMAGE_REGISTRATION_DEPTH_TO_COLOR");
    return;
  }
  auto mode = data ? openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR : openni::IMAGE_REGISTRATION_OFF;
  auto rc = device_->setImageRegistrationMode(mode);
  if (rc != openni::STATUS_OK) {
    ROS_ERROR("Enabling image registration mode failed: \n%s\n",
              openni::OpenNI::getExtendedError());
  }
}

void OBCameraNode::onNewFrameCallback(const openni::VideoFrameRef& frame,
                                      const stream_index_pair& stream_index) {
  int width = frame.getWidth();
  int height = frame.getHeight();
  int dst_width = width;
  int dst_height = height;
  CHECK(images_.count(stream_index));
  auto& image = images_.at(stream_index);
  if (image.size() != cv::Size(width, height)) {
    image.create(height, width, image.type());
  }
  image.data = (uint8_t*)frame.getData();
  auto pid = device_info_.getUsbProductId();
  if (stream_index == DEPTH && depth_align_ && depth_scale_ != 1) {
    dst_width = width * depth_scale_;
    dst_height = height * depth_scale_;
    cv::Mat dst(dst_height, dst_width, CV_16UC1);
    cv::resize(image, dst, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_NEAREST);
    image.create(dst_height, dst_width, image.type());
    image = dst;

  } else if (stream_index == DEPTH && depth_align_ && pid == DABAI_DCW2_DEPTH_PID) {
    dst_width = width_[COLOR];
    dst_height = height_[COLOR];
    cv::Mat dst(dst_height, dst_width, CV_16UC1);
    dcw2Align(image, dst);
    image.create(dst_height, dst_width, image.type());
    image = dst;
  } else if (stream_index == DEPTH && depth_align_ && (pid == DABAI_MAX_PRO_PID || pid == GEMINI_UW_PID)) {
    dst_width = width_[COLOR];
    dst_height = height_[COLOR];
    cv::Mat dst(dst_height, dst_width, CV_16UC1);
    maxProAlign(image, dst);
    image.create(dst_height, dst_width, image.type());
    image = dst;
  }
  if (flip_image_[stream_index]) {
    cv::flip(image, image, 1);
  }
  auto image_msg =
      *(cv_bridge::CvImage(std_msgs::Header(), encoding_.at(stream_index), image).toImageMsg());
  auto timestamp = ros::Time::now();
  image_msg.header.stamp = timestamp;
  image_msg.header.frame_id =
      depth_align_ ? depth_aligned_frame_id_[stream_index] : optical_frame_id_[stream_index];
  image_msg.width = dst_width;
  image_msg.height = dst_height;
  image_msg.step = image_msg.width * unit_step_size_[stream_index];
  image_msg.is_bigendian = false;
  auto& image_publisher = image_publishers_.at(stream_index);
  image_publisher.publish(image_msg);
  sensor_msgs::CameraInfo camera_info;
  if (stream_index == DEPTH) {
    camera_info = getDepthCameraInfo();
  } else if (stream_index == COLOR) {
    camera_info = getColorCameraInfo();
  } else if (stream_index == INFRA1 || stream_index == INFRA2) {
    double f = getFocalLength(stream_index, width);
    camera_info = getIRCameraInfo(width, height, f);
  }

  camera_info.header.stamp = timestamp;
  camera_info_publishers_.at(stream_index).publish(camera_info);

  if (save_images_[stream_index]) {
    auto now = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    auto current_path = boost::filesystem::current_path().string();
    auto fps = stream_video_mode_[stream_index].getFps();
    std::string filename = current_path + "/image/" + stream_name_[stream_index] + "_" +
                           std::to_string(image_msg.width) + "x" +
                           std::to_string(image_msg.height) + "_" + std::to_string(fps) + "hz_" +
                           ss.str() + ".png";
    if (!boost::filesystem::exists(current_path + "/image")) {
      boost::filesystem::create_directory(current_path + "/image");
    }
    ROS_INFO_STREAM("Saving image to " << filename);

    ROS_INFO_STREAM("Saving image to :" << stream_index.first);
    if (stream_index.first == openni::SENSOR_DEPTH) {
      auto image_to_save = cv_bridge::toCvCopy(image_msg, encoding_[stream_index])->image;
      cv::imwrite(filename, image_to_save);
    } else if (stream_index.first == openni::SENSOR_COLOR) {
      auto image_to_save =
          cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8)->image;
      cv::imwrite(filename, image_to_save);
    } else if (stream_index.first == openni::SENSOR_IR) {
      cv::imwrite(filename, image);
    } else {
      ROS_ERROR_STREAM("Unsupported stream type: " << stream_index.first);
    }
    
    save_images_[stream_index] = false;
  }
}

void OBCameraNode::setDepthColorSync(bool data) {
  auto rc = device_->setDepthColorSyncEnabled(data);
  if (rc != openni::STATUS_OK) {
    ROS_ERROR_STREAM(
        "Enabling depth color synchronization failed: " << openni::OpenNI::getExtendedError());
  }
}

void OBCameraNode::setDepthToColorResolution(int width, int height) {
  const auto pid = device_info_.getUsbProductId();
  if (pid != DABAI_DCW_DEPTH_PID && pid != GEMINI_E_DEPTH_PID) {
    return;
  }
  if (!depth_align_ && !enable_pointcloud_xyzrgb_) {
    return;
  }
  if (width * 9 == height * 16) {
    // 16:9
    auto status = device_->setProperty(XN_MODULE_PROPERTY_D2C_RESOLUTION, RGBResolution16_9);
    if (status != openni::STATUS_OK) {
      ROS_ERROR_STREAM("setProperty XN_MODULE_PROPERTY_D2C_RESOLUTION "
                       << openni::OpenNI::getExtendedError());
    }
  } else if (width * 3 == height * 4) {
    // 4:3
    auto status = device_->setProperty(XN_MODULE_PROPERTY_D2C_RESOLUTION, RGBResolution4_3);
    if (status != openni::STATUS_OK) {
      ROS_ERROR_STREAM("setProperty XN_MODULE_PROPERTY_D2C_RESOLUTION "
                       << openni::OpenNI::getExtendedError());
    }
  } else {
    ROS_ERROR_STREAM("NOT 16x9 or 4x3 resolution");
  }
}

boost::optional<openni::VideoMode> OBCameraNode::lookupVideoModeFromDynConfig(int index) {
  auto it = video_modes_lookup_table_.find(index);
  if (it != video_modes_lookup_table_.end()) {
    return it->second;
  }
  return {};
}

void OBCameraNode::reconfigureCallback(const AstraConfig& config, uint32_t level) {
  (void)level;
  ROS_INFO_STREAM("Received configuration");
  if (!enable_reconfigure_) {
    ROS_WARN_STREAM("Dynamic reconfigure is disabled");
    return;
  }
  auto json_data = nlohmann::json::parse(config.edited_video_modes);
  auto edited_video_modes = json_data["enum"].get<std::vector<nlohmann::json>>();
  video_modes_lookup_table_.clear();
  for (const auto& json_mode : edited_video_modes) {
    std::string name = json_mode["name"].get<std::string>();
    int value = json_mode["value"].get<int>();
    std::vector<std::string> arr;
    boost::split(arr, name, boost::is_any_of("_"));
    assert(arr.size() == 3);
    openni::VideoMode video_mode;
    int x_resolution = std::stoi(arr[0]);
    int y_resolution = std::stoi(arr[1]);
    int fps = std::stoi(arr[2]);
    video_mode.setResolution(x_resolution, y_resolution);
    video_mode.setFps(fps);
    video_modes_lookup_table_[value] = video_mode;
  }
  auto ir_mode = lookupVideoModeFromDynConfig(config.ir_mode);
  if (ir_mode && device_->hasSensor(openni::SENSOR_IR)) {
    ROS_INFO_STREAM("Setting IR mode to " << ir_mode->getResolutionX() << "x"
                                          << ir_mode->getResolutionY() << "@" << ir_mode->getFps()
                                          << " fps");
    width_[INFRA1] = ir_mode->getResolutionX();
    height_[INFRA1] = ir_mode->getResolutionY();
    fps_[INFRA1] = ir_mode->getFps();
  }
  auto color_mode = lookupVideoModeFromDynConfig(config.color_mode);
  if (color_mode && device_->hasSensor(openni::SENSOR_COLOR)) {
    ROS_INFO_STREAM("Setting color mode to " << color_mode->getResolutionX() << "x"
                                             << color_mode->getResolutionY() << "@"
                                             << color_mode->getFps() << " fps");
    width_[COLOR] = color_mode->getResolutionX();
    height_[COLOR] = color_mode->getResolutionY();
    fps_[COLOR] = color_mode->getFps();
  }
  auto depth_mode = lookupVideoModeFromDynConfig(config.depth_mode);
  if (depth_mode && device_->hasSensor(openni::SENSOR_DEPTH)) {
    ROS_INFO_STREAM("Setting depth mode to " << depth_mode->getResolutionX() << "x"
                                             << depth_mode->getResolutionY() << "@"
                                             << depth_mode->getFps() << " fps");
    width_[DEPTH] = depth_mode->getResolutionX();
    height_[DEPTH] = depth_mode->getResolutionY();
    fps_[DEPTH] = depth_mode->getFps();
  }
  depth_align_ = config.depth_align;
  if (depth_align_ && !device_->hasSensor(openni::SENSOR_COLOR) && !use_uvc_camera_) {
    ROS_WARN("No color sensor found, depth align will be disabled");
    depth_align_ = false;
  }
  color_depth_synchronization_ = config.color_depth_synchronization;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  stopStreams();
  setupVideoMode();
  camera_params_.reset();
  getCameraParams();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  startStreams();
  ROS_INFO("Configuration applied");
}

void OBCameraNode::sendKeepAlive(const ros::TimerEvent& event) {
  (void)event;
  ROS_INFO_STREAM("Sending keep alive");
  openni::Status status;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  status = device_->setProperty(XN_MODULE_PROPERTY_LASER_SECURE_KEEPALIVE, nullptr, 0);
  if (status != openni::STATUS_OK) {
    ROS_ERROR_STREAM("openni status " << status);
    ROS_ERROR_STREAM("current errno " << errno << " system error string " << strerror(errno));
    ROS_INFO("Sending keep alive Error: %s\n", openni::OpenNI::getExtendedError());
  } else {
    ROS_INFO("Sending keep alive success\n");
  }
}

void OBCameraNode::pollFrame() {
  openni::VideoFrameRef frame;
  while (run_poll_frame_thread_ && ros::ok()) {
    std::unique_lock<decltype(poll_frame_thread_lock_)> lock(poll_frame_thread_lock_);
    auto has_stream_started =
        poll_frame_thread_cv_.wait_for(lock, std::chrono::milliseconds(1000), [this] {
          return std::any_of(
              IMAGE_STREAMS.begin(), IMAGE_STREAMS.end(),
              [this](const stream_index_pair& stream_index) {
                if (stream_started_[stream_index] && streams_[stream_index]->isValid()) {
                  return true;
                }
                return false;
              });
        });
    if (!has_stream_started) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      continue;
    }
    openni::VideoStream* streams[3];
    std::map<int, stream_index_pair> idx_map;
    int stream_count = 0;
    for (const auto& stream_index : IMAGE_STREAMS) {
      if (enable_[stream_index] && streams_[stream_index].get()) {
        streams[stream_count] = streams_[stream_index].get();
        idx_map[stream_count] = stream_index;
        stream_count++;
      }
    }
    int ready_stream = -1;
    const static int timeout_ms(2000);
    auto status =
        openni::OpenNI::waitForAnyStream(streams, stream_count, &ready_stream, timeout_ms);
    if (status != openni::STATUS_OK) {
      continue;
    }
    CHECK(ready_stream != -1);
    auto stream_index = idx_map[ready_stream];
    status = streams[ready_stream]->readFrame(&frame);
    if (status != openni::STATUS_OK) {
      ROS_ERROR_STREAM("read " << stream_name_[stream_index] << " stream failed "
                               << openni::OpenNI::getExtendedError());
      continue;
    }
    onNewFrameCallback(frame, stream_index);
  }
}

void OBCameraNode::dcw2Align(const cv::Mat& src, cv::Mat& dst) {
  int color_width = width_[COLOR];
  int color_height = height_[COLOR];
  int depth_width = width_[DEPTH];
  int depth_height = height_[DEPTH];
  if (color_width == 640 && color_height == 360 && depth_width == 640 && depth_height == 400) {
    // take src up 360 line
    cv::Mat src_up = src(cv::Rect(0, 0, 640, 360));
    dst = src_up;
  } else if (color_width == 640 && color_height == 360 && depth_width == 540 && depth_height == 400) {
    // 创建一个新的 640x400 的图像
    cv::Mat newImage(400, 640, src.type(), cv::Scalar(0));
    // 定义原始图像在新图像中的位置
    cv::Rect roi(cv::Rect(50, 0, src.cols, src.rows));
    // 复制原始图像到新图像的指定位置
    src.copyTo(newImage(roi));

    cv::Mat src_up = newImage(cv::Rect(0, 0, 640, 360));
    dst = src_up;
  } else if (color_width == 640 && color_height == 480 && depth_width == 640 && depth_height == 400) {
    // take src up 480x360, then upscale to 640x680
    cv::Mat src_up = src(cv::Rect(80, 0, 480, 360));
    cv::resize(src_up, dst, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);

  } else if (color_width == 320 && color_height == 180 && depth_width == 320 &&
             depth_height == 200) {
    // take src up 180 line
    cv::Mat src_up = src(cv::Rect(0, 0, 320, 180));
    dst = src_up;
  } else if (color_width == 320 && color_height == 240 && depth_width == 320 &&
             depth_height == 200) {
    // take src up 240x180, then upscale to 320x240
    cv::Mat src_up = src(cv::Rect(40, 0, 240, 180));
    cv::resize(src_up, dst, cv::Size(320, 240), 0, 0, cv::INTER_NEAREST);
  } else if (color_width == 640 && color_height == 480 && depth_width == 540 &&
             depth_height == 400) {
    // take src up 480x360, then upscale to 640x680
    cv::Mat src_up = src(cv::Rect(30, 0, 480, 360));
    cv::resize(src_up, dst, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
  } else if (color_width == 320 && color_height == 240 && depth_width == 270 &&
             depth_height == 200) {
    // take src up 240x180, then upscale to 320x240
    cv::Mat src_up = src(cv::Rect(15, 0, 240, 180));
    cv::resize(src_up, dst, cv::Size(320, 240), 0, 0, cv::INTER_NEAREST);
  } else {
    ROS_ERROR_STREAM("Not support D2C config  color_width "
                     << color_width << " color_height " << color_height << " depth_width "
                     << depth_width << " depth_height " << depth_height);
    return;
  }
}

void OBCameraNode::maxProAlign(const cv::Mat& src, cv::Mat& dst) {
  int color_width = width_[COLOR];
  int color_height = height_[COLOR];
  int depth_width = width_[DEPTH];
  int depth_height = height_[DEPTH];
  if (color_width == 640 && color_height == 480 && depth_width == 640 && depth_height == 400) {
    // take src up 480x360, then upscale to 640x680
    cv::Mat src_up = src(cv::Rect(96, 32, 448, 336));
    cv::resize(src_up, dst, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);

  } else {
    ROS_ERROR_STREAM("Not support D2C config  color_width "
                     << color_width << " color_height " << color_height << " depth_width "
                     << depth_width << " depth_height " << depth_height);
    return;
  }
}

}  // namespace astra_camera
