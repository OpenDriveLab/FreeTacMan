/**************************************************************************/
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

#include <glog/logging.h>
#include <openni2/OpenNI.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/version.hpp>
#include <mutex>
#include <atomic>

#include <cstdlib>
#include <chrono>
#include <ctime>
#include <boost/filesystem.hpp>
#include "astra_camera/DeviceInfo.h"
#include "astra_camera/Extrinsics.h"
#include "astra_camera/GetBool.h"
#include "astra_camera/GetCameraInfo.h"
#include "astra_camera/GetDeviceInfo.h"
#include "astra_camera/GetDouble.h"
#include "astra_camera/GetInt32.h"
#include "astra_camera/GetString.h"
#include "astra_camera/Metadata.h"
#include "astra_camera/SetInt32.h"
#include "std_srvs/SetBool.h"
#include "std_srvs/Empty.h"
#include "astra_camera/SetString.h"
#include "astra_camera/GetCameraParams.h"
#include "std_msgs/Empty.h"
#include "json.hpp"

namespace astra_camera {
using FrameCallbackFunction = std::function<void(const openni::VideoFrameRef& frame)>;
using stream_index_pair = std::pair<openni::SensorType, int>;

const stream_index_pair COLOR{openni::SENSOR_COLOR, 0};
const stream_index_pair DEPTH{openni::SENSOR_DEPTH, 0};
const stream_index_pair INFRA1{openni::SENSOR_IR, 0};
const stream_index_pair INFRA2{openni::SENSOR_IR, 1};

const std::vector<stream_index_pair> IMAGE_STREAMS = {DEPTH, INFRA1, COLOR};

typedef enum {
  RGBResolution4_3 = 0,   // 4:3分辨率 如：640x480
  RGBResolution16_9 = 1,  // 16:9分辨率 如：1920x1280
} RgbResolution;

struct ImageROI {
  int x = -1;
  int y = -1;
  int width = -1;
  int height = -1;
};

enum class MultiDeviceSyncMode {
  None = 0,  // 不同步
  Master = 1,
  Slave = 2,
};

}  // namespace astra_camera
