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
#pragma once

#include <astra_camera/AstraConfig.h>
#include <camera_info_manager/camera_info_manager.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <openni2/OpenNI.h>
#include <openni2/PS1080.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <boost/optional.hpp>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "constants.h"
#include "d2c_viewer.h"
#include "point_cloud_proc/point_cloud_proc.h"
#include "types.h"
#include "utils.h"
#include "uvc_camera_driver.h"

namespace astra_camera {

using ReconfigureServer = dynamic_reconfigure::Server<AstraConfig>;

class OBCameraNode {
 public:
  OBCameraNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private,
               std::shared_ptr<openni::Device> device, bool use_uvc_camera = false);

  ~OBCameraNode();

  void clean();

  void init();

 private:
  void setupCameraCtrlServices();

  void setupConfig();

  void setupCameraInfoManager();

  void setupDevices();

  void setupD2CConfig();

  void setupFrameCallback();

  void setupSyncMode();

  void setupVideoMode();

  void startStreams();

  void stopStreams();

  void startStream(const stream_index_pair& stream_index);

  void stopStream(const stream_index_pair& stream_index);

  void getParameters();

  void setupTopics();

  void setupUVCCamera();

  void imageSubscribedCallback(const stream_index_pair& stream_index);

  void imageUnsubscribedCallback(const stream_index_pair& stream_index);

  void setupPublishers();

  void publishStaticTF(const ros::Time& t, const tf2::Vector3& trans, const tf2::Quaternion& q,
                       const std::string& from, const std::string& to);

  void calcAndPublishStaticTransform();

  void publishDynamicTransforms();

  void publishStaticTransforms();

  void setImageRegistrationMode(bool data);

  bool setMirrorCallback(std_srvs::SetBoolRequest& request, std_srvs::SetBoolResponse& response,
                         const stream_index_pair& stream_index);

  bool getExposureCallback(GetInt32Request& request, GetInt32Response& response,
                           const stream_index_pair& stream_index);

  bool setExposureCallback(SetInt32Request& request, SetInt32Response& response,
                           const stream_index_pair& stream_index);

  bool getGainCallback(GetInt32Request& request, GetInt32Response& response,
                       const stream_index_pair& stream_index);

  bool setGainCallback(SetInt32Request& request, SetInt32Response& response,
                       const stream_index_pair& stream_index);

  bool getIRTemperatureCallback(GetDoubleRequest& request, GetDoubleResponse& response);

  void setIRAutoExposure(bool status);

  int getIRExposure();

  void setIRExposure(uint32_t data);

  int getIRGain();

  std::string getSerialNumber();

  void setIRGain(int data);

  bool resetIRGainCallback(std_srvs::EmptyRequest& request, std_srvs::EmptyResponse& response);

  bool resetIRExposureCallback(std_srvs::EmptyRequest& request, std_srvs::EmptyResponse& response);

  bool getAutoWhiteBalanceEnabledCallback(GetInt32Request& request, GetInt32Response& response,
                                          const stream_index_pair& stream_index);

  bool setAutoWhiteBalanceEnabledCallback(SetInt32Request& request, SetInt32Response& response);

  bool setAutoExposureCallback(std_srvs::SetBoolRequest& request,
                               std_srvs::SetBoolResponse& response,
                               const stream_index_pair& stream_index);

  bool setLaserEnableCallback(std_srvs::SetBoolRequest& request,
                              std_srvs::SetBoolResponse& response);

  bool setIRFloodCallback(std_srvs::SetBoolRequest& request, std_srvs::SetBoolResponse& response);

  bool setLdpEnableCallback(std_srvs::SetBoolRequest& request, std_srvs::SetBoolResponse& response);

  bool setFanEnableCallback(std_srvs::SetBoolRequest& request, std_srvs::SetBoolResponse& response);

  bool getDeviceInfoCallback(GetDeviceInfoRequest& request, GetDeviceInfoResponse& response);

  bool getCameraInfoCallback(GetCameraInfoRequest& request, GetCameraInfoResponse& response);

  bool getSDKVersionCallback(GetStringRequest& request, GetStringResponse& response);

  bool getDeviceTypeCallback(GetStringRequest& request, GetStringResponse& response);

  bool getSerialNumberCallback(GetStringRequest& request, GetStringResponse& response);

  bool switchIRCameraCallback(SetStringRequest& request, SetStringResponse& response);

  bool getCameraParamsCallback(GetCameraParamsRequest& request, GetCameraParamsResponse& response);

  bool toggleSensorCallback(std_srvs::SetBoolRequest& request, std_srvs::SetBoolResponse& response,
                            const stream_index_pair& stream_index);
  bool saveImagesCallback(std_srvs::EmptyRequest& request, std_srvs::EmptyResponse& response);

  bool getSupportedVideoModesCallback(GetStringRequest& request, GetStringResponse& response,
                                      const stream_index_pair& stream_index);

  //bool getLaserStatusCallback(GetBoolRequest& request, GetBoolResponse& response);

  bool getLdpStatusCallback(GetBoolRequest& request, GetBoolResponse& response);

  bool toggleSensor(const stream_index_pair& stream_index, bool enabled, std::string& msg);

  void onNewFrameCallback(const openni::VideoFrameRef& frame,
                          const stream_index_pair& stream_index);

  void setDepthColorSync(bool data);

  void setDepthToColorResolution(int width, int height);

  OBCameraParams getCameraParams();

  OBCameraParams getColorCameraParams();

  static sensor_msgs::CameraInfo OBCameraParamsToCameraInfo(const OBCameraParams& params);

  double getFocalLength(const stream_index_pair& stream_index, int y_resolution);

  sensor_msgs::CameraInfo getIRCameraInfo(int width, int height, double f);

  sensor_msgs::CameraInfo getDepthCameraInfo();

  sensor_msgs::CameraInfo getColorCameraInfo();

  sensor_msgs::CameraInfo getDefaultCameraInfo(int width, int height, double f);

  boost::optional<openni::VideoMode> lookupVideoModeFromDynConfig(int index);

  // NOTE: This interface only for testing purposes.
  void reconfigureCallback(const AstraConfig& config, uint32_t level);

  void sendKeepAlive(const ros::TimerEvent& event);

  void pollFrame();

  void dcw2Align(const cv::Mat& src, cv::Mat& dst);
  void maxProAlign(const cv::Mat& src, cv::Mat& dst);

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  std::shared_ptr<openni::Device> device_;
  std::shared_ptr<UVCCameraDriver> uvc_camera_driver_ = nullptr;
  std::string camera_name_ = "camera";
  std::unique_ptr<ReconfigureServer> reconfigure_server_ = nullptr;
  std::map<int, openni::VideoMode> video_modes_lookup_table_;
  bool use_uvc_camera_ = false;
  openni::DeviceInfo device_info_{};
  std::atomic_bool is_running_{false};
  std::map<stream_index_pair, bool> enable_;
  std::map<stream_index_pair, bool> stream_started_;
  std::map<stream_index_pair, int> width_;
  std::map<stream_index_pair, int> height_;
  std::map<stream_index_pair, int> fps_;
  std::map<stream_index_pair, openni::PixelFormat> format_;
  std::map<stream_index_pair, int> image_format_;
  std::map<stream_index_pair, std::string> encoding_;
  std::map<stream_index_pair, cv::Mat> images_;
  std::vector<int> compression_params_;
  std::string base_frame_id_;
  std::map<stream_index_pair, std::string> frame_id_;
  std::map<stream_index_pair, std::string> optical_frame_id_;
  std::map<stream_index_pair, std::string> depth_aligned_frame_id_;
  std::map<stream_index_pair, std::string> stream_name_;
  std::map<stream_index_pair, std::shared_ptr<openni::VideoStream>> streams_;
  std::map<stream_index_pair, openni::VideoMode> stream_video_mode_;
  std::map<stream_index_pair, std::vector<openni::VideoMode>> supported_video_modes_;
  std::map<stream_index_pair, FrameCallbackFunction> stream_frame_callback_;
  std::map<stream_index_pair, int> unit_step_size_;
  std::map<stream_index_pair, ros::Publisher> image_publishers_;
  std::map<stream_index_pair, ros::Publisher> camera_info_publishers_;
  std::map<stream_index_pair, bool> flip_image_;

  std::map<stream_index_pair, ros::ServiceServer> get_exposure_srv_;
  std::map<stream_index_pair, ros::ServiceServer> set_exposure_srv_;
  std::map<stream_index_pair, ros::ServiceServer> get_gain_srv_;
  std::map<stream_index_pair, ros::ServiceServer> set_gain_srv_;
  std::map<stream_index_pair, ros::ServiceServer> set_mirror_srv_;
  std::map<stream_index_pair, ros::ServiceServer> toggle_sensor_srv_;
  std::map<stream_index_pair, ros::ServiceServer> get_supported_video_modes_srv_;

  ros::ServiceServer get_sdk_version_srv_;
  std::map<stream_index_pair, ros::ServiceServer> set_auto_exposure_srv_;
  ros::ServiceServer get_device_srv_;
  ros::ServiceServer set_laser_enable_srv_;
  //ros::ServiceServer get_laser_status_srv_;
  ros::ServiceServer set_ldp_enable_srv_;
  ros::ServiceServer set_fan_enable_srv_;
  ros::ServiceServer get_camera_info_srv_;
  ros::ServiceServer switch_ir_camera_srv_;
  ros::ServiceServer get_white_balance_srv_;
  ros::ServiceServer set_white_balance_srv_;
  ros::ServiceServer get_camera_params_srv_;
  ros::ServiceServer get_device_type_srv_;
  ros::ServiceServer get_serial_srv_;
  ros::ServiceServer save_images_srv_;
  ros::ServiceServer reset_ir_gain_srv_;
  ros::ServiceServer reset_ir_exposure_srv_;
  ros::ServiceServer set_ir_flood_srv_;
  ros::ServiceServer get_ldp_status_srv_;
  ros::ServiceServer get_ir_temperature_srv_;

  bool publish_tf_ = true;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> dynamic_tf_broadcaster_;
  std::vector<geometry_msgs::TransformStamped> static_tf_msgs_;
  ros::Publisher extrinsics_publisher_;
  std::shared_ptr<std::thread> tf_thread_;
  std::condition_variable tf_cv_;
  double tf_publish_rate_ = 10.0;
  bool depth_align_ = false;
  boost::optional<OBCameraParams> camera_params_;
  boost::optional<OBCameraParams> color_camera_params_;
  double depth_ir_x_offset_ = 0.0;
  double depth_ir_y_offset_ = 0.0;
  bool color_depth_synchronization_ = false;
  int depth_scale_ = 1;
  ImageROI color_roi_;
  ImageROI depth_roi_;
  bool enable_reconfigure_ = false;
  std::map<stream_index_pair, std::atomic_bool> save_images_;
  std::recursive_mutex device_lock_;
  int init_ir_gain_ = 0;
  uint32_t init_ir_exposure_ = 0;
  bool ir_ae_ = true;
  int ir_gain_ = 0;
  uint32_t ir_exposure_ = 0;
  bool enable_d2c_viewer_ = false;
  std::unique_ptr<D2CViewer> d2c_filter_ = nullptr;
  std::unique_ptr<camera_info_manager::CameraInfoManager> color_info_manager_ = nullptr;
  std::unique_ptr<camera_info_manager::CameraInfoManager> ir_info_manager_ = nullptr;
  std::string ir_info_uri_;
  std::string color_info_uri_;
  bool keep_alive_ = false;
  int keep_alive_interval_ = 15;
  ros::Timer keep_alive_timer_;
  std::atomic_bool initialized_{false};
  std::unique_ptr<PointCloudXyzNode> point_cloud_xyz_node_ = nullptr;
  std::unique_ptr<PointCloudXyzrgbNode> point_cloud_xyzrgb_node_ = nullptr;
  bool enable_pointcloud_ = false;
  bool enable_pointcloud_xyzrgb_ = false;
  std::unique_ptr<std::thread> poll_frame_thread_ = nullptr;
  std::atomic_bool run_poll_frame_thread_{false};
  std::mutex poll_frame_thread_lock_;
  std::condition_variable poll_frame_thread_cv_;
  bool enable_publish_extrinsic_ = false;
  int soft_filter_ = 2;
  //Default 16
  int soft_filter_max_diff_ = 16;
  //Default 480
  int soft_filter_max_speckle_size_ = 480;
  MultiDeviceSyncMode multi_device_sync_mode_;
};
}  // namespace astra_camera
