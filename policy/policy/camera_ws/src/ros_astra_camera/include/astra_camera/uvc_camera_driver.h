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

#include <camera_info_manager/camera_info_manager.h>
#include <image_transport/image_transport.h>
#include <libuvc/libuvc.h>
#include <openni2/OpenNI.h>
#include <sensor_msgs/CameraInfo.h>

#include <boost/optional.hpp>

#include "constants.h"
#include "types.h"
#include "utils.h"

#if defined(USE_RK_MPP)
#include <rga/RgaApi.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_err.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_log.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_rc_defs.h>
#include <rockchip/mpp_task.h>
#include <rockchip/rk_mpi.h>
#define MPP_ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))
#endif

namespace astra_camera {
struct UVCCameraConfig {
  int vendor_id = 0;
  int product_id = 0;
  int width = 0;
  int height = 0;
  int fps = 0;
  std::string serial_number;
  std::string format;
  std::string frame_id;
  std::string optical_frame_id;
  int retry_count = 0;
  UVCCameraConfig() = default;
  UVCCameraConfig(const UVCCameraConfig&) = default;
  UVCCameraConfig(UVCCameraConfig&&) = default;
  UVCCameraConfig& operator=(const UVCCameraConfig&) = default;
  UVCCameraConfig& operator=(UVCCameraConfig&&) = default;
};

std::ostream& operator<<(std::ostream& os, const UVCCameraConfig& config);

class UVCCameraDriver {
 public:
  explicit UVCCameraDriver(ros::NodeHandle& nh, ros::NodeHandle& nh_private,
                           const sensor_msgs::CameraInfo& camera_info,
                           const std::string& serial_number = "");

  ~UVCCameraDriver();

  void updateConfig(const UVCCameraConfig& config);

  void setVideoMode();

  void imageSubscribedCallback();

  void imageUnsubscribedCallback();

  void startStreaming();

  void stopStreaming() noexcept;

  int getResolutionX() const;

  int getResolutionY() const;
#if defined(USE_RK_MPP)
  void mppInit();
  void mppDeInit();
  void convertFrameToRGB(MppFrame frame, uint8_t* rgb_data);
  bool MPPDecodeFrame(uvc_frame_t* frame, uint8_t* rgb_data);
#endif

 private:
  void setupCameraControlService();

  sensor_msgs::CameraInfo getCameraInfo();

  static enum uvc_frame_format UVCFrameFormatString(const std::string& format);

  static void frameCallbackWrapper(uvc_frame_t* frame, void* ptr);

  void frameCallback(uvc_frame_t* frame);

  void autoControlsCallback(enum uvc_status_class status_class, int event, int selector,
                            enum uvc_status_attribute status_attribute, void* data,
                            size_t data_len);
  static void autoControlsCallbackWrapper(enum uvc_status_class status_class, int event,
                                          int selector, enum uvc_status_attribute status_attribute,
                                          void* data, size_t data_len, void* ptr);

  void openCamera();

  bool getUVCExposureCb(GetInt32Request& request, GetInt32Response& response);

  bool setUVCExposureCb(SetInt32Request& request, SetInt32Response& response);

  bool getUVCGainCb(GetInt32Request& request, GetInt32Response& response);

  bool setUVCGainCb(SetInt32Request& request, SetInt32Response& response);

  bool getUVCWhiteBalanceCb(GetInt32Request& request, GetInt32Response& response);

  bool setUVCWhiteBalanceCb(SetInt32Request& request, SetInt32Response& response);

  bool setUVCAutoExposureCb(std_srvs::SetBoolRequest& request, std_srvs::SetBoolResponse& response);

  bool setUVCAutoWhiteBalanceCb(std_srvs::SetBoolRequest& request,
                                std_srvs::SetBoolResponse& response);

  bool getUVCMirrorCb(GetInt32Request& request, GetInt32Response& response);

  bool setUVCMirrorCb(std_srvs::SetBoolRequest& request, std_srvs::SetBoolResponse& response);

  bool toggleUVCCamera(std_srvs::SetBoolRequest& request, std_srvs::SetBoolResponse& response);

  bool saveImageCallback(std_srvs::EmptyRequest& request, std_srvs::EmptyResponse& response);

  int UVCGetControl(int control, int unit, int len, uvc_req_code req_code);

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  UVCCameraConfig config_;
  std::string camera_name_ = "camera";
  std::string frame_id_;
  std::string color_info_uri_;
  ImageROI roi_;
  uvc_context_t* ctx_ = nullptr;
  uvc_device_t* device_ = nullptr;
  uvc_device_handle_t* device_handle_ = nullptr;
  uvc_frame_t* frame_buffer_ = nullptr;
  uvc_stream_ctrl_t ctrl_{};
  std::atomic_bool uvc_flip_{false};
  std::atomic_bool is_streaming_started{false};
  std::atomic_bool save_image_{false};
  std::atomic_bool is_camera_opened_{false};
  bool flip_color_ = false;

  ros::ServiceServer get_uvc_exposure_srv_;
  ros::ServiceServer set_uvc_exposure_srv_;
  ros::ServiceServer get_uvc_gain_srv_;
  ros::ServiceServer set_uvc_gain_srv_;
  ros::ServiceServer get_uvc_white_balance_srv_;
  ros::ServiceServer set_uvc_white_balance_srv_;
  ros::ServiceServer set_uvc_auto_exposure_srv_;
  ros::ServiceServer set_uvc_auto_white_balance_srv_;
  ros::ServiceServer get_uvc_mirror_srv_;
  ros::ServiceServer set_uvc_mirror_srv_;
  ros::ServiceServer toggle_uvc_camera_srv_;
  ros::ServiceServer save_image_srv_;
  ros::Publisher image_publisher_;
  ros::Publisher camera_info_publisher_;
  sensor_msgs::CameraInfo camera_info_;
  std::recursive_mutex device_lock_;
  std::shared_ptr<camera_info_manager::CameraInfoManager> color_info_manager_ = nullptr;
  int device_num_ = 1;
#if defined(USE_RK_MPP)
  MppCtx mpp_ctx_ = nullptr;
  MppApi* mpp_api_ = nullptr;
  MppPacket mpp_packet_ = nullptr;
  MppFrame mpp_frame_ = nullptr;
  uint8_t* rgb_data_ = nullptr;
  MppDecCfg mpp_dec_cfg_ = nullptr;
  MppBuffer mpp_frame_buffer_ = nullptr;
  MppBuffer mpp_packet_buffer_ = nullptr;
  uint8_t* data_buffer_ = nullptr;
  MppBufferGroup mpp_frame_group_ = nullptr;
  MppBufferGroup mpp_packet_group_ = nullptr;
  MppTask mpp_task_ = nullptr;
  uint32_t need_split_ = 0;
#endif
};
}  // namespace astra_camera
