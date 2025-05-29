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
#pragma once
#include <openni2/OpenNI.h>
#include <ros/ros.h>
#include <mutex>

namespace astra_camera {

using DeviceConnectedCb = std::function<void(const openni::DeviceInfo*)>;

using DeviceDisconnectedCb = std::function<void(const openni::DeviceInfo*)>;

class Context : public openni::OpenNI::DeviceConnectedListener,
                public openni::OpenNI::DeviceDisconnectedListener,
                public openni::OpenNI::DeviceStateChangedListener {
 public:
  explicit Context(DeviceDisconnectedCb device_disconnected_cb);

  ~Context() override;

  void onDeviceStateChanged(const openni::DeviceInfo* pInfo, openni::DeviceState state) override;

  void onDeviceConnected(const openni::DeviceInfo* pInfo) override;

  void onDeviceDisconnected(const openni::DeviceInfo* pInfo) override;

  std::vector<openni::DeviceInfo> queryDeviceList();

 private:
  std::recursive_mutex mutex_;
  bool first_time_query_ = true;
  DeviceDisconnectedCb device_disconnected_cb_;
  std::map<std::string, openni::DeviceInfo> device_info_list_;
};

}  // namespace astra_camera
