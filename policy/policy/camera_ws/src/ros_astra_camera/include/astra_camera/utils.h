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

#include <glog/logging.h>
#include <openni2/OpenNI.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <tf2/LinearMath/Quaternion.h>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/PointCloud2.h>

#include "astra_camera/Extrinsics.h"
#include "constants.h"
#include "types.h"

namespace astra_camera {
bool operator==(const openni::VideoMode& lhs, const openni::VideoMode& rhs);

bool operator!=(const openni::VideoMode& lhs, const openni::VideoMode& rhs);

std::ostream& operator<<(std::ostream& os, const openni::VideoMode& video_mode);

tf2::Quaternion rotationMatrixToQuaternion(const std::vector<float>& rotation);

Extrinsics obExtrinsicsToMsg(const std::vector<float>& rotation,const std::vector<float>& transition,
                             const std::string& frame_id);


bool isValidCameraParams(const OBCameraParams& params);

void cameraParameterPrinter(const std::vector<float>& rotation,
                            const std::vector<float>& transition);

std::string PixelFormatToString(const openni::PixelFormat& format);

void savePointToPly(sensor_msgs::PointCloud2::Ptr cloud, const std::string& filename);

void saveRGBPointToPly(sensor_msgs::PointCloud2::Ptr cloud, const std::string& filename);

MultiDeviceSyncMode getMultiDeviceSyncMode(const std::string& mode);


}  // namespace astra_camera
