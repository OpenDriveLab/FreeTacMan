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

namespace astra_camera {

OBCameraParams OBCameraNode::getCameraParams() {
  if (camera_params_) {
    return camera_params_.value();
  }
  auto pid = device_info_.getUsbProductId();
  if (pid == DABAI_DCW2_DEPTH_PID || pid == DABAI_MAX_PID || pid == DABAI_DW2_DEPTH_PID ||
      pid == DABAI_MAX_PRO_PID || pid == GEMINI_UW_PID) {
    OBCameraParamList cameraParamList;
    int data_size = sizeof(OBCameraParamList);
    OBCameraParam cameraParams[10], depthParams;
    OBCameraParams params;
    cameraParamList.pCameraParams = (OBCameraParam*)cameraParams;
    cameraParamList.nParamSize = 10;
    openni::Status rc = device_->getProperty(openni::OBEXTENSION_ID_CAM_PARAMS,
                                             (uint8_t*)&cameraParamList, &data_size);
    if (rc != openni::STATUS_OK) {
      ROS_WARN_STREAM("Failed to get camera params");
      return {};
    }
    bool matched = false;
    for (int i = 0; i < cameraParamList.nParamSize; i++) {
      if (width_[DEPTH] == cameraParamList.pCameraParams[i].depthIntrinsic.width &&
          height_[DEPTH] == cameraParamList.pCameraParams[i].depthIntrinsic.height &&
          width_[COLOR] == cameraParamList.pCameraParams[i].rgbIntrinsic.width &&
          height_[COLOR] == cameraParamList.pCameraParams[i].rgbIntrinsic.height) {
        depthParams = cameraParamList.pCameraParams[i];
        matched = true;
        break;
      }
    }

    if (!matched) {
      for (int i = 0; i < cameraParamList.nParamSize; i++) {
        if (width_[DEPTH] == cameraParamList.pCameraParams[i].depthIntrinsic.width &&
            height_[DEPTH] == cameraParamList.pCameraParams[i].depthIntrinsic.height) {
          depthParams = cameraParamList.pCameraParams[i];
          matched = true;
          break;
        }
      }
    }

    if (!matched) {
      ROS_WARN_STREAM("Failed to get camera params");
      depthParams = cameraParamList.pCameraParams[0];
    }
    params.l_intr_p[0] = depthParams.depthIntrinsic.fx;
    params.l_intr_p[1] = depthParams.depthIntrinsic.fy;
    params.l_intr_p[2] = depthParams.depthIntrinsic.cx;
    params.l_intr_p[3] = depthParams.depthIntrinsic.cy;
    params.r_intr_p[0] = depthParams.rgbIntrinsic.fx;
    params.r_intr_p[1] = depthParams.rgbIntrinsic.fy;
    params.r_intr_p[2] = depthParams.rgbIntrinsic.cx;
    params.r_intr_p[3] = depthParams.rgbIntrinsic.cy;
    params.l_k[0] = depthParams.depthDistortion.k1;
    params.l_k[1] = depthParams.depthDistortion.k2;
    params.l_k[2] = depthParams.depthDistortion.k3;
    params.l_k[3] = depthParams.depthDistortion.p1;
    params.l_k[4] = depthParams.depthDistortion.p2;
    params.r_k[0] = depthParams.rgbDistortion.k1;
    params.r_k[1] = depthParams.rgbDistortion.k2;
    params.r_k[2] = depthParams.rgbDistortion.k3;
    params.r_k[3] = depthParams.rgbDistortion.p1;
    params.r_k[4] = depthParams.rgbDistortion.p2;
    for (int i = 0; i < 9; i++) {
      if (i < 3) {
        params.r2l_t[i] = depthParams.transform.trans[i];
      }
      params.r2l_r[i] = depthParams.transform.rot[i];
    }
    camera_params_ = params;
    return params;
  } else if (pid == DABAI_DCW_DEPTH_PID || pid == DABAI_DW_PID || pid == GEMINI_E_DEPTH_PID ||
             pid == GEMINI_E_LITE_DEPTH_PID) {
    OBCameraParamsData camera_params_data;
    int data_size = sizeof(camera_params_data);
    memset(&camera_params_data, 0, data_size);
    auto depth_width = width_[DEPTH];
    auto depth_height = height_[DEPTH];
    auto color_width = width_[COLOR];
    auto color_height = height_[COLOR];
    if (depth_width == 1024 && depth_height == 768) {
      camera_params_data.depthRes = XN_CAMERA_PARAMS_DEPTH_RES_1024_768;
      camera_params_data.colorRes = XN_CAMERA_PARAMS_COLOR_RES_DEFAULT;
    } else if (depth_width == 512 && depth_height == 384) {
      camera_params_data.depthRes = XN_CAMERA_PARAMS_DEPTH_RES_512_384;
      camera_params_data.colorRes = XN_CAMERA_PARAMS_COLOR_RES_DEFAULT;
    } else if (depth_width == 640 && depth_height == 480) {
      camera_params_data.depthRes = XN_CAMERA_PARAMS_DEPTH_RES_640_480;
      camera_params_data.colorRes = XN_CAMERA_PARAMS_COLOR_RES_DEFAULT;
    } else if (depth_width == 320 && depth_height == 240) {
      camera_params_data.depthRes = XN_CAMERA_PARAMS_DEPTH_RES_320_240;
      camera_params_data.colorRes = XN_CAMERA_PARAMS_COLOR_RES_DEFAULT;
    } else if (depth_width == 160 && depth_height == 120) {
      camera_params_data.depthRes = XN_CAMERA_PARAMS_DEPTH_RES_160_120;
      camera_params_data.colorRes = XN_CAMERA_PARAMS_COLOR_RES_DEFAULT;
    } else if (depth_width == 480 && depth_height == 360) {
      camera_params_data.depthRes = XN_CAMERA_PARAMS_DEPTH_RES_480_360;
      camera_params_data.colorRes = XN_CAMERA_PARAMS_COLOR_RES_DEFAULT;
    } else if (depth_width == 240 && depth_height == 180) {
      camera_params_data.depthRes = XN_CAMERA_PARAMS_DEPTH_RES_240_180;
      camera_params_data.colorRes = XN_CAMERA_PARAMS_COLOR_RES_DEFAULT;
    } else if (depth_width == 640 && depth_height == 360 && color_width == 640 &&
               color_height == 360 && (pid == DABAI_DCW_DEPTH_PID || pid == GEMINI_E_DEPTH_PID)) {
      camera_params_data.depthRes = XN_CAMERA_PARAMS_DEPTH_RES_640_360;
      camera_params_data.colorRes = XN_CAMERA_PARAMS_COLOR_RES_640_360;
    } else if (depth_width == 320 && depth_height == 180 && color_width == 320 &&
               color_height == 180 && (pid == DABAI_DCW_DEPTH_PID || pid == GEMINI_E_DEPTH_PID)) {
      camera_params_data.depthRes = XN_CAMERA_PARAMS_DEPTH_RES_320_180;
      camera_params_data.colorRes = XN_CAMERA_PARAMS_COLOR_RES_320_180;
    } else {
      ROS_WARN_STREAM("dose not match  resolution: " << depth_width << "x" << depth_height);
      OBCameraParams params;
      data_size = sizeof(OBCameraParams);
      device_->getProperty(openni::OBEXTENSION_ID_CAM_PARAMS, (uint8_t*)&params, &data_size);
      camera_params_ = params;
      return params;
    }
    device_->getProperty(openni::OBEXTENSION_ID_CAM_PARAMS, (uint8_t*)&camera_params_data,
                         &data_size);
    camera_params_ = camera_params_data.params;
    return camera_params_data.params;
  } else {
    OBCameraParams params;
    int data_size = sizeof(OBCameraParams);
    device_->getProperty(openni::OBEXTENSION_ID_CAM_PARAMS, (uint8_t*)&params, &data_size);
    camera_params_ = params;
    return params;
  }
}

OBCameraParams OBCameraNode::getColorCameraParams() {
  if (color_camera_params_) {
    return color_camera_params_.value();
  }

  OBCameraParamList cameraParamList;
  int data_size = sizeof(OBCameraParamList);
  OBCameraParam cameraParams[10], depthParams;
  OBCameraParams params;
  cameraParamList.pCameraParams = (OBCameraParam*)cameraParams;
  cameraParamList.nParamSize = 10;

  openni::Status rc = device_->getProperty(openni::OBEXTENSION_ID_CAM_PARAMS,
                                           (uint8_t*)&cameraParamList, &data_size);
  if (rc != openni::STATUS_OK) {
    ROS_WARN_STREAM("Failed to get camera params");
    return {};
  }
  bool matched = false;
  for (int i = 0; i < cameraParamList.nParamSize; i++) {
    if (width_[DEPTH] == cameraParamList.pCameraParams[i].depthIntrinsic.width &&
        height_[DEPTH] == cameraParamList.pCameraParams[i].depthIntrinsic.height &&
        width_[COLOR] == cameraParamList.pCameraParams[i].rgbIntrinsic.width &&
        height_[COLOR] == cameraParamList.pCameraParams[i].rgbIntrinsic.height) {
      depthParams = cameraParamList.pCameraParams[i];
      matched = true;
      break;
    }
  }

  if (!matched) {
    for (int i = 0; i < cameraParamList.nParamSize; i++) {
      if (width_[COLOR] == cameraParamList.pCameraParams[i].rgbIntrinsic.width &&
          height_[COLOR] == cameraParamList.pCameraParams[i].rgbIntrinsic.height) {
        depthParams = cameraParamList.pCameraParams[i];
        matched = true;
        break;
      }
    }
  }

  if (!matched) {
    ROS_WARN_STREAM("Failed to get color camera params");
    depthParams = cameraParamList.pCameraParams[0];
  }
  params.l_intr_p[0] = depthParams.depthIntrinsic.fx;
  params.l_intr_p[1] = depthParams.depthIntrinsic.fy;
  params.l_intr_p[2] = depthParams.depthIntrinsic.cx;
  params.l_intr_p[3] = depthParams.depthIntrinsic.cy;
  params.r_intr_p[0] = depthParams.rgbIntrinsic.fx;
  params.r_intr_p[1] = depthParams.rgbIntrinsic.fy;
  params.r_intr_p[2] = depthParams.rgbIntrinsic.cx;
  params.r_intr_p[3] = depthParams.rgbIntrinsic.cy;
  params.l_k[0] = depthParams.depthDistortion.k1;
  params.l_k[1] = depthParams.depthDistortion.k2;
  params.l_k[2] = depthParams.depthDistortion.k3;
  params.l_k[3] = depthParams.depthDistortion.p1;
  params.l_k[4] = depthParams.depthDistortion.p2;
  params.r_k[0] = depthParams.rgbDistortion.k1;
  params.r_k[1] = depthParams.rgbDistortion.k2;
  params.r_k[2] = depthParams.rgbDistortion.k3;
  params.r_k[3] = depthParams.rgbDistortion.p1;
  params.r_k[4] = depthParams.rgbDistortion.p2;
  for (int i = 0; i < 9; i++) {
    if (i < 3) {
      params.r2l_t[i] = depthParams.transform.trans[i];
    }
    params.r2l_r[i] = depthParams.transform.rot[i];
  }

  color_camera_params_ = params;
  return params;
}

sensor_msgs::CameraInfo OBCameraNode::OBCameraParamsToCameraInfo(const OBCameraParams& params) {
  sensor_msgs::CameraInfo camera_info;
  camera_info.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
  camera_info.D.resize(5, 0.0);
  camera_info.D[0] = params.r_k[0];
  camera_info.D[1] = params.r_k[1];
  camera_info.D[2] = params.r_k[3];
  camera_info.D[3] = params.r_k[4];
  camera_info.D[4] = params.r_k[2];

  camera_info.K.fill(0.0);
  camera_info.K[0] = params.r_intr_p[0];  // fx
  camera_info.K[2] = params.r_intr_p[2];  // cx
  camera_info.K[4] = params.r_intr_p[1];  // fy
  camera_info.K[5] = params.r_intr_p[3];  // cy
  camera_info.K[8] = 1.0;

  camera_info.R.fill(0.0);
  camera_info.R[0] = 1.0;
  camera_info.R[4] = 1.0;
  camera_info.R[8] = 1.0;

  camera_info.P.fill(0.0);
  camera_info.P[0] = camera_info.K[0];  // fx
  camera_info.P[2] = camera_info.K[2];  // cx
  camera_info.P[3] = 0;                 // Tx
  camera_info.P[5] = camera_info.K[4];  // fy
  camera_info.P[6] = camera_info.K[5];  // cy
  camera_info.P[7] = 0;                 // Ty
  camera_info.P[10] = 1.0;
  camera_info.P[11] = 0.0;
  return camera_info;
}

double OBCameraNode::getFocalLength(const stream_index_pair& stream_index, int y_resolution) {
  if (!streams_.count(stream_index)) {
    return 0.0;
  }
  auto stream = streams_.at(stream_index);
  if (stream == nullptr) {
    return 0.0;
  }
  return static_cast<double>(y_resolution) / (2 * tan(stream->getVerticalFieldOfView() / 2));
}

sensor_msgs::CameraInfo OBCameraNode::getIRCameraInfo(int width, int height, double f) {
  if (ir_info_manager_ && ir_info_manager_->isCalibrated()) {
    auto camera_info = ir_info_manager_->getCameraInfo();
    if (camera_info.width != static_cast<uint32_t>(width) ||
        camera_info.height != static_cast<uint32_t>(height)) {
      ROS_WARN_ONCE(
          "Image resolution doesn't match calibration of the IR camera. Using default "
          "parameters.");
      ROS_INFO("camera info width = %d, height = %d", camera_info.width, camera_info.height);
      return getDefaultCameraInfo(width, height, f);
    } else {
      return camera_info;
    }
  }

  auto camera_info = getDefaultCameraInfo(width, height, f);
  auto camera_params = getCameraParams();
  if (isValidCameraParams(camera_params)) {
    camera_info.K.fill(0.0);

    camera_info.K.fill(0.0);
    camera_info.K[0] = camera_params.l_intr_p[0];
    camera_info.K[2] = camera_params.l_intr_p[2];
    camera_info.K[4] = camera_params.l_intr_p[1];
    camera_info.K[5] = camera_params.l_intr_p[3];
    camera_info.K[8] = 1.0;
    for (int i = 0; i < 9; i++) {
      camera_info.R[i] = camera_params.r2l_r[i];
    }

    camera_info.P.fill(0.0);
    camera_info.P[0] = camera_info.K[0];
    camera_info.P[2] = camera_info.K[2];
    camera_info.P[5] = camera_info.K[4];
    camera_info.P[6] = camera_info.K[5];
    camera_info.P[10] = 1.0;
    auto pid = device_info_.getUsbProductId();
    if (pid != DABAI_DCW_DEPTH_PID && pid != DABAI_DW_PID && pid != DABAI_MAX_PID &&
        pid != DABAI_DCW2_DEPTH_PID && pid != DABAI_DW2_DEPTH_PID && pid != DABAI_MAX_PRO_PID &&
        pid != GEMINI_UW_PID) {
      /* 02122020 Scale IR Params */
      double scaling = static_cast<double>(width) / 640;
      camera_info.K[0] *= scaling;  // fx
      camera_info.K[2] *= scaling;  // cx
      camera_info.K[4] *= scaling;  // fy
      camera_info.K[5] *= scaling;  // cy
      camera_info.P[0] *= scaling;  // fx
      camera_info.P[2] *= scaling;  // cx
      camera_info.P[5] *= scaling;  // fy
      camera_info.P[6] *= scaling;  // cy

      /* 02122020 end */
    }
  }
  return camera_info;
}

sensor_msgs::CameraInfo OBCameraNode::getDepthCameraInfo() {
  // The depth image has essentially the same intrinsics as the IR image, BUT the
  // principal point is offset by half the size of the hardware correlation window
  // (probably 9x9 or 9x7 in 640x480 mode). See
  // http://www.ros.org/wiki/kinect_calibration/technical
  int width = width_[DEPTH];
  int height = height_[DEPTH];
  double f = getFocalLength(DEPTH, height);
  double scaling = (double)width / 640;
  auto camera_info = getIRCameraInfo(width, height, f);
  auto camera_params = getCameraParams();
  if (!isValidCameraParams(camera_params)) {
    for (int i = 0; i < 9; i++) {
      camera_info.R[i] = camera_params.r2l_r[i];
    }
    if (depth_align_ || enable_pointcloud_xyzrgb_) {
      camera_info.K[0] = camera_params.r_intr_p[0];
      camera_info.K[2] = camera_params.r_intr_p[2];
      camera_info.K[4] = camera_params.r_intr_p[1];
      camera_info.K[5] = camera_params.r_intr_p[3];
      auto pid = device_info_.getUsbProductId();
      if (pid != DABAI_DCW_DEPTH_PID && pid != DABAI_DW_PID && pid != DABAI_MAX_PID &&
          pid != DABAI_DCW2_DEPTH_PID && pid != DABAI_DW2_DEPTH_PID && pid != DABAI_MAX_PRO_PID &&
          pid != GEMINI_UW_PID) {
        camera_info.K[0] *= scaling;  // fx
        camera_info.K[2] *= scaling;  // cx
        camera_info.K[4] *= scaling;  // fy
        camera_info.K[5] *= scaling;  // cy
      }
    }
  }
  camera_info.K[2] -= depth_ir_x_offset_ * scaling;
  camera_info.K[5] -= depth_ir_y_offset_ * scaling;
  camera_info.K[2] -= depth_ir_x_offset_ * scaling;
  camera_info.K[6] -= depth_ir_y_offset_ * scaling;

  // TODO: Could put this in projector frame so as to encode the baseline in P[3]
  return camera_info;
}

sensor_msgs::CameraInfo OBCameraNode::getColorCameraInfo() {
  int width = width_[COLOR];
  int height = height_[COLOR];
  if (color_info_manager_ && color_info_manager_->isCalibrated()) {
    auto camera_info = color_info_manager_->getCameraInfo();
    if (camera_info.width != static_cast<uint32_t>(width) ||
        camera_info.height != static_cast<uint32_t>(height)) {
      ROS_WARN(
          "Image resolution doesn't match calibration of the RGB camera. Using default "
          "parameters.");
      double color_focal_length = getFocalLength(COLOR, height);
      return getDefaultCameraInfo(width, height, color_focal_length);
    } else {
      return camera_info;
    }
  }
  // If uncalibrated, fill in default values

  auto pid = device_info_.getUsbProductId();
  OBCameraParams camera_params;
  memset(&camera_params, 0, sizeof(OBCameraParams));
  if (pid == DABAI_DCW2_DEPTH_PID || pid == DABAI_MAX_PRO_PID || pid == GEMINI_UW_PID) {
    camera_params = getColorCameraParams();
  } else {
    camera_params = getCameraParams();
  }
  if (isValidCameraParams(camera_params)) {
    auto default_camera_info = OBCameraParamsToCameraInfo(camera_params);
    sensor_msgs::CameraInfo camera_info;
    camera_info.D.resize(5, 0.0);
    camera_info.K.fill(0.0);
    camera_info.P.fill(0.0);
    camera_info.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
    camera_info.width = width;
    camera_info.height = height;

    for (int i = 0; i < 9; i++) {
      camera_info.K[i] = default_camera_info.K[i];
    }
    for (int i = 0; i < 9; i++) {
      camera_info.R[i] = camera_params.r2l_r[i];
    }
    for (int i = 0; i < 12; i++) {
      camera_info.P[i] = default_camera_info.P[i];
    }

    if (pid != DABAI_DCW_DEPTH_PID && pid != GEMINI_E_DEPTH_PID && pid != DABAI_DCW2_DEPTH_PID &&
        pid != DABAI_MAX_PRO_PID && pid != GEMINI_UW_PID) {
      /*02112020 color camera param change according to resolution */
      double scaling = (double)width / 640;
      camera_info.K[0] *= scaling;  // fx
      camera_info.K[2] *= scaling;  // cx
      camera_info.K[4] *= scaling;  // fy
      camera_info.K[5] *= scaling;  // cy
      camera_info.P[0] *= scaling;  // fx
      camera_info.P[2] *= scaling;  // cx
      camera_info.P[5] *= scaling;  // fy
      camera_info.P[6] *= scaling;  // cy

      /* 02112020 end*/
    }
    return camera_info;
    // end if (isValidCameraParams(camera_params))
  } else {
    double color_focal_length = getFocalLength(COLOR, height);
    return getDefaultCameraInfo(width, height, color_focal_length);
  }
}
sensor_msgs::CameraInfo OBCameraNode::getDefaultCameraInfo(int width, int height, double f) {
  sensor_msgs::CameraInfo info;

  info.width = width;
  info.height = height;

  // No distortion
  info.D.resize(5, 0.0);
  info.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;

  // Simple camera matrix: square pixels (fx = fy), principal point at center
  info.K.fill(0.0);
  info.K[0] = info.K[4] = f;

  info.K[2] = (static_cast<double>(width) / 2) - 0.5;
  // Aspect ratio for the camera center on Astra (and other devices?) is 4/3
  // This formula keeps the principal point the same in VGA and SXGA modes
  info.K[5] = (static_cast<double>(width) * (3. / 8.)) - 0.5;
  info.K[8] = 1.0;

  // No separate rectified image plane, so R = I
  info.R[0] = info.R[4] = info.R[8] = 1.0;

  // Then P=K(I|0) = (K|0)
  info.P.fill(0.0);
  info.P[0] = info.P[5] = f;  // fx, fy
  info.P[2] = info.K[2];      // cx
  info.P[6] = info.K[5];      // cy
  info.P[10] = 1.0;

  return info;
}

}  // namespace astra_camera
