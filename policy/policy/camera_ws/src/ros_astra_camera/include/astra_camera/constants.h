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

#include <cstdlib>
#include <string>

#define OB_ROS_MAJOR_VERSION 1
#define OB_ROS_MINOR_VERSION 2
#define OB_ROS_PATCH_VERSION 7

#ifndef STRINGIFY
#define STRINGIFY(arg) #arg
#endif
#ifndef VAR_ARG_STRING
#define VAR_ARG_STRING(arg) STRINGIFY(arg)
#endif

#define STRINGIFY(arg) #arg
#define VAR_ARG_STRING(arg) STRINGIFY(arg)
/* Return version in "X.Y.Z" format */
#define OB_ROS_VERSION_STR \
  (VAR_ARG_STRING(OB_ROS_MAJOR_VERSION.OB_ROS_MINOR_VERSION.OB_ROS_PATCH_VERSION))

namespace astra_camera {

const bool ALIGN_DEPTH = false;
const bool POINTCLOUD = false;
const bool ALLOW_NO_TEXTURE_POINTS = false;
const bool SYNC_FRAMES = false;
const bool ORDERED_POINTCLOUD = false;

const bool PUBLISH_TF = true;
const double TF_PUBLISH_RATE = 0;     // Static transform
const double DIAGNOSTICS_PERIOD = 0;  // Static transform

const int IMAGE_WIDTH = 640;
const int IMAGE_HEIGHT = 480;
const int IMAGE_FPS = 30;

const std::string IMAGE_QOS = "SYSTEM_DEFAULT";
const std::string DEFAULT_QOS = "DEFAULT";
const std::string HID_QOS = "HID_DEFAULT";
const std::string EXTRINSICS_QOS = "EXTRINSICS_DEFAULT";

const double IMU_FPS = 0;

const bool ENABLE_DEPTH = true;
const bool ENABLE_INFRA1 = true;
const bool ENABLE_INFRA2 = true;
const bool ENABLE_COLOR = true;
const bool ENABLE_FISHEYE = true;
const bool ENABLE_IMU = true;
const bool HOLD_BACK_IMU_FOR_FRAMES = false;
const bool PUBLISH_ODOM_TF = true;

const std::string DEFAULT_BASE_FRAME_ID = "camera_link";
const std::string DEFAULT_ODOM_FRAME_ID = "odom_frame";
const std::string DEFAULT_DEPTH_FRAME_ID = "camera_depth_frame";
const std::string DEFAULT_INFRA1_FRAME_ID = "camera_infra1_frame";
const std::string DEFAULT_INFRA2_FRAME_ID = "camera_infra2_frame";
const std::string DEFAULT_COLOR_FRAME_ID = "camera_color_frame";
const std::string DEFAULT_FISHEYE_FRAME_ID = "camera_fisheye_frame";
const std::string DEFAULT_IMU_FRAME_ID = "camera_imu_frame";

const std::string DEFAULT_DEPTH_OPTICAL_FRAME_ID = "camera_depth_optical_frame";
const std::string DEFAULT_INFRA1_OPTICAL_FRAME_ID = "camera_infra1_optical_frame";
const std::string DEFAULT_INFRA2_OPTICAL_FRAME_ID = "camera_infra2_optical_frame";
const std::string DEFAULT_COLOR_OPTICAL_FRAME_ID = "camera_color_optical_frame";
const std::string DEFAULT_FISHEYE_OPTICAL_FRAME_ID = "camera_fisheye_optical_frame";
const std::string DEFAULT_ACCEL_OPTICAL_FRAME_ID = "camera_accel_optical_frame";
const std::string DEFAULT_GYRO_OPTICAL_FRAME_ID = "camera_gyro_optical_frame";
const std::string DEFAULT_IMU_OPTICAL_FRAME_ID = "camera_imu_optical_frame";

const std::string DEFAULT_ALIGNED_DEPTH_TO_COLOR_FRAME_ID = "camera_aligned_depth_to_color_frame";
const std::string DEFAULT_ALIGNED_DEPTH_TO_INFRA1_FRAME_ID = "camera_aligned_depth_to_infra1_frame";
const std::string DEFAULT_ALIGNED_DEPTH_TO_INFRA2_FRAME_ID = "camera_aligned_depth_to_infra2_frame";
const std::string DEFAULT_ALIGNED_DEPTH_TO_FISHEYE_FRAME_ID =
    "camera_aligned_depth_to_fisheye_frame";

const std::string DEFAULT_UNITE_IMU_METHOD = "";
const std::string DEFAULT_FILTERS = "";
const std::string DEFAULT_TOPIC_ODOM_IN = "";
const std::string DEFAULT_D2C_MODE = "sw";  // sw = software mode, hw=hardware mode, none,
const float ROS_DEPTH_SCALE = 0.001;
const int TIME_FILTER_LENGTH = 10;

constexpr static uint32_t DABAI_DCW_RGB_PID = 0x0559;
constexpr static uint32_t DABAI_PRO_PID = 0x0655;
constexpr static uint32_t DABAI_MINI_PID = 0x0656;
constexpr static uint32_t DABAI_DC1_PID = 0x0657;
constexpr static uint32_t DABAI_D1_PID = 0x0658;
constexpr static uint32_t DABAI_DCW_DEPTH_PID = 0x0659;
constexpr static uint32_t DABAI_DW_PID = 0x065a;
constexpr static uint32_t DEEYEA_UVC_PID = 0x050b;
constexpr static uint32_t GEMIN_E_UVC_PID = 0x055c;
constexpr static uint32_t GEMINI_E_DEPTH_PID = 0x065c;
constexpr static uint32_t GEMINI_E_LITE_DEPTH_PID = 0x065d;
constexpr static uint32_t DABAI_MAX_PID = 0x069a;
constexpr static uint32_t DABAI_MAX_PRO_PID = 0x069e;
constexpr static uint32_t GEMINI_UW_PID = 0x06aa;
constexpr static uint32_t ASTRA_PRO_DEPTH_PID = 0x0403;
constexpr static uint32_t DABAI_DCW2_DEPTH_PID = 0x06a0;
constexpr static uint32_t DABAI_DW2_DEPTH_PID = 0x069f;
const std::string OB_STEREO_S = "Orbbec Canglong";
const std::string OB_EMBEDDED_S = "Astra SL1000S_U3";
const std::string OB_STEREO_S_U3 = "Astra SV1301S_U3";
const std::string OB_ASTRA_PRO = "Orbbec Astra Pro";
const std::string OB_ASTRA_PRO_PLUS = "Orbbec Astra Pro Plus";
const std::string OB_DABAI = "Orbbec Dabai";
const std::string OB_ASTRA_PLUS = "Orbbec Astra+";
const std::string OB_ASTRA_PLUS_S = "Orbbec Astra+_S";
const std::string OB_DABAI_PRO = "Orbbec Astra DaBai Pro";
const std::string OB_DABAI_DCW = "Orbbec DaBai DCW";
const std::string OB_DABAI_DW = "Orbbec DaBai DW";
const std::string DEFAULT_LOCK_FILE = "astra_device.lock";
const key_t DEFAULT_SEM_KEY = 0x0401;

}  // namespace astra_camera
