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

void OBCameraNode::setupCameraCtrlServices() {
  for (const auto& stream_index : IMAGE_STREAMS) {
    if (!enable_[stream_index] || !device_->hasSensor(stream_index.first)) {
      //ROS_INFO_STREAM("Stream " << stream_name_[stream_index] << " is disabled");
      continue;
    }
    auto stream_name = stream_name_[stream_index];
    std::string service_name = "get_" + stream_name + "_exposure";
    get_exposure_srv_[stream_index] = nh_.advertiseService<GetInt32Request, GetInt32Response>(
        service_name, [this, stream_index = stream_index](auto&& request, auto&& response) {
          response.success = this->getExposureCallback(request, response, stream_index);
          return response.success;
        });
    service_name = "set_" + stream_name + "_exposure";
    set_exposure_srv_[stream_index] = nh_.advertiseService<SetInt32Request, SetInt32Response>(
        service_name, [this, stream_index = stream_index](auto&& request, auto&& response) {
          response.success = this->setExposureCallback(request, response, stream_index);
          return response.success;
        });
    service_name = "get_" + stream_name + "_gain";
    get_gain_srv_[stream_index] = nh_.advertiseService<GetInt32Request, GetInt32Response>(
        service_name, [this, stream_index = stream_index](auto&& request, auto&& response) {
          response.success = this->getGainCallback(request, response, stream_index);
          return response.success;
        });
    service_name = "set_" + stream_name + "_gain";
    set_gain_srv_[stream_index] = nh_.advertiseService<SetInt32Request, SetInt32Response>(
        service_name, [this, stream_index = stream_index](auto&& request, auto&& response) {
          response.success = this->setGainCallback(request, response, stream_index);
          return response.success;
        });
    service_name = "set_" + stream_name + "_mirror";
    set_mirror_srv_[stream_index] =
        nh_.advertiseService<std_srvs::SetBoolRequest, std_srvs::SetBoolResponse>(
            service_name, [this, stream_index = stream_index](auto&& request, auto&& response) {
              response.success = this->setMirrorCallback(request, response, stream_index);
              return response.success;
            });
    service_name = "set_" + stream_name + "_auto_exposure";
    set_auto_exposure_srv_[stream_index] =
        nh_.advertiseService<std_srvs::SetBoolRequest, std_srvs::SetBoolResponse>(
            service_name, [this, stream_index = stream_index](auto&& request, auto&& response) {
              response.success = this->setAutoExposureCallback(request, response, stream_index);
              return response.success;
            });
    service_name = "toggle_" + stream_name;
    toggle_sensor_srv_[stream_index] =
        nh_.advertiseService<std_srvs::SetBoolRequest, std_srvs::SetBoolResponse>(
            service_name, [this, stream_index = stream_index](auto&& request, auto&& response) {
              response.success = this->toggleSensorCallback(request, response, stream_index);
              return response.success;
            });
    service_name = "get_" + stream_name + "_supported_video_modes";
    get_supported_video_modes_srv_[stream_index] =
        nh_.advertiseService<GetStringRequest, GetStringResponse>(
            service_name, [this, stream_index = stream_index](auto&& request, auto&& response) {
              response.success =
                  this->getSupportedVideoModesCallback(request, response, stream_index);
              return response.success;
            });
  }
  get_white_balance_srv_ = nh_.advertiseService<GetInt32Request, GetInt32Response>(
      "get_auto_white_balance", [this](auto&& request, auto&& response) {
        response.success = this->getAutoWhiteBalanceEnabledCallback(request, response, COLOR);
        return response.success;
      });
  set_white_balance_srv_ = nh_.advertiseService<SetInt32Request, SetInt32Response>(
      "set_auto_white_balance", [this](auto&& request, auto&& response) {
        response.success = this->setAutoWhiteBalanceEnabledCallback(request, response);
        return response.success;
      });
  set_fan_enable_srv_ = nh_.advertiseService<std_srvs::SetBoolRequest, std_srvs::SetBoolResponse>(
      "set_fan", [this](auto&& request, auto&& response) {
        response.success = this->setFanEnableCallback(request, response);
        return response.success;
      });
  set_laser_enable_srv_ = nh_.advertiseService<std_srvs::SetBoolRequest, std_srvs::SetBoolResponse>(
      "set_laser", [this](auto&& request, auto&& response) {
        response.success = this->setLaserEnableCallback(request, response);
        return response.success;
      });
  //Laser is read only.
  // get_laser_status_srv_ = nh_.advertiseService<GetBoolRequest, GetBoolResponse>(
  //     "get_laser_status", [this](auto&& request, auto&& response) {
  //       response.success = this->getLaserStatusCallback(request, response);
  //       return response.success;
  //     });
  set_ldp_enable_srv_ = nh_.advertiseService<std_srvs::SetBoolRequest, std_srvs::SetBoolResponse>(
      "set_ldp", [this](auto&& request, auto&& response) {
        response.success = this->setLdpEnableCallback(request, response);
        return response.success;
      });

  get_ldp_status_srv_ = nh_.advertiseService<GetBoolRequest, GetBoolResponse>(
      "get_ldp_status", [this](auto&& request, auto&& response) {
        response.success = this->getLdpStatusCallback(request, response);
        return response.success;
      });
  get_ir_temperature_srv_ = nh_.advertiseService<GetDoubleRequest, GetDoubleResponse>(
      "get_ir_temperature", [this](auto&& request, auto&& response) {
        response.success = this->getIRTemperatureCallback(request, response);
        return response.success;
      });

  get_device_srv_ = nh_.advertiseService<GetDeviceInfoRequest, GetDeviceInfoResponse>(
      "get_device_info", [this](auto&& request, auto&& response) {
        response.success = this->getDeviceInfoCallback(request, response);
        return response.success;
      });
  get_sdk_version_srv_ = nh_.advertiseService<GetStringRequest, GetStringResponse>(
      "get_version", [this](auto&& request, auto&& response) {
        response.success = this->getSDKVersionCallback(request, response);
        return response.success;
      });
  get_camera_info_srv_ = nh_.advertiseService<GetCameraInfoRequest, GetCameraInfoResponse>(
      "get_camera_info", [this](auto&& request, auto&& response) {
        response.success = this->getCameraInfoCallback(request, response);
        return response.success;
      });
  switch_ir_camera_srv_ = nh_.advertiseService<SetStringRequest, SetStringResponse>(
      "switch_ir_camera", [this](auto&& request, auto&& response) {
        response.success = this->switchIRCameraCallback(request, response);
        return response.success;
      });
  get_camera_params_srv_ = nh_.advertiseService<GetCameraParamsRequest, GetCameraParamsResponse>(
      "get_camera_params", [this](auto&& request, auto&& response) {
        response.success = this->getCameraParamsCallback(request, response);
        return response.success;
      });
  get_device_type_srv_ = nh_.advertiseService<GetStringRequest, GetStringResponse>(
      "get_device_type", [this](auto&& request, auto&& response) {
        response.success = this->getDeviceTypeCallback(request, response);
        return response.success;
      });
  get_serial_srv_ = nh_.advertiseService<GetStringRequest, GetStringResponse>(
      "get_serial", [this](auto&& request, auto&& response) {
        response.success = this->getSerialNumberCallback(request, response);
        return response.success;
      });
  save_images_srv_ = nh_.advertiseService<std_srvs::EmptyRequest, std_srvs::EmptyResponse>(
      "save_images", [this](auto&& request, auto&& response) {
        return this->saveImagesCallback(request, response);
      });
  reset_ir_exposure_srv_ = nh_.advertiseService<std_srvs::EmptyRequest, std_srvs::EmptyResponse>(
      "reset_ir_exposure", [this](auto&& request, auto&& response) {
        return this->resetIRExposureCallback(request, response);
      });
  reset_ir_gain_srv_ = nh_.advertiseService<std_srvs::EmptyRequest, std_srvs::EmptyResponse>(
      "reset_ir_gain", [this](auto&& request, auto&& response) {
        return this->resetIRGainCallback(request, response);
      });
  set_ir_flood_srv_ = nh_.advertiseService<std_srvs::SetBoolRequest, std_srvs::SetBoolResponse>(
      "set_ir_flood", [this](auto&& request, auto&& response) {
        response.success = this->setIRFloodCallback(request, response);
        return response.success;
      });
}

bool OBCameraNode::setMirrorCallback(std_srvs::SetBoolRequest& request,
                                     std_srvs::SetBoolResponse& response,
                                     const stream_index_pair& stream_index) {
  (void)response;
  if (!stream_started_[stream_index] || !device_->hasSensor(stream_index.first)) {
    std::stringstream ss;
    ss << "Stream " << stream_name_[stream_index] << " is not started or does not have a sensor";
    response.message = ss.str();
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  auto stream = streams_.at(stream_index);
  stream->setMirroringEnabled(request.data);
  return true;
}

bool OBCameraNode::getExposureCallback(GetInt32Request& request, GetInt32Response& response,
                                       const stream_index_pair& stream_index) {
  (void)request;
  if (!stream_started_[stream_index] || !device_->hasSensor(stream_index.first)) {
    std::stringstream ss;
    ss << "Stream " << stream_name_[stream_index] << " is not started or does not have a sensor";
    response.message = ss.str();
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  if (stream_index == COLOR) {
    auto stream = streams_.at(stream_index);
    auto camera_settings = stream->getCameraSettings();
    if (camera_settings == nullptr) {
      response.data = 0;
      response.success = false;
      response.message = stream_name_[stream_index] + " Camera settings not available";
      return false;
    }
    response.data = camera_settings->getExposure();
  } else if (stream_index == INFRA1 || stream_index == INFRA2 || stream_index == DEPTH) {
    response.data = getIRExposure();
    return true;
  } else {
    response.message = "Stream not supported get exposure";
    return false;
  }
  return true;
}

bool OBCameraNode::setExposureCallback(SetInt32Request& request, SetInt32Response& response,
                                       const stream_index_pair& stream_index) {
  if (!stream_started_[stream_index] || !device_->hasSensor(stream_index.first)) {
    std::stringstream ss;
    ss << "Stream " << stream_name_[stream_index] << " is not started or does not have a sensor";
    response.message = ss.str();
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  if (stream_index == COLOR) {
    auto stream = streams_.at(stream_index);
    auto camera_settings = stream->getCameraSettings();
    if (camera_settings == nullptr) {
      response.success = false;
      response.message = stream_name_[stream_index] + " Camera settings not available";
      return false;
    }
    auto rc = camera_settings->setExposure(request.data);
    std::stringstream ss;
    if (rc != openni::STATUS_OK) {
      ss << "Couldn't set color exposure: " << openni::OpenNI::getExtendedError();
      response.message = ss.str();
      ROS_ERROR_STREAM(response.message);
      return false;
    } else {
      return true;
    }
  } else if (stream_index == INFRA1 || stream_index == INFRA2 || stream_index == DEPTH) {
    auto data = static_cast<uint32_t>(request.data);
    setIRExposure(data);
    return true;
  } else {
    response.message = "stream not support get gain";
    return false;
  }
}

bool OBCameraNode::getGainCallback(GetInt32Request& request, GetInt32Response& response,
                                   const stream_index_pair& stream_index) {
  (void)request;
  if (!stream_started_[stream_index] || !device_->hasSensor(stream_index.first)) {
    std::stringstream ss;
    ss << "Stream " << stream_name_[stream_index] << " is not started or does not have a sensor";
    response.message = ss.str();
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  if (stream_index == COLOR) {
    auto stream = streams_.at(stream_index);
    auto camera_settings = stream->getCameraSettings();
    if (camera_settings == nullptr) {
      response.success = false;
      response.message = stream_name_[stream_index] + " Camera settings not available";
      return false;
    }
    response.data = camera_settings->getGain();
  } else if (stream_index == INFRA1 || stream_index == INFRA2 || stream_index == DEPTH) {
    response.data = getIRGain();
  } else {
    response.message = "stream not support get gain";
    return false;
  }
  return true;
}

bool OBCameraNode::setGainCallback(SetInt32Request& request, SetInt32Response& response,
                                   const stream_index_pair& stream_index) {
  (void)response;
  if (!stream_started_[stream_index] || !device_->hasSensor(stream_index.first)) {
    std::stringstream ss;
    ss << "Stream " << stream_name_[stream_index] << " is not started or does not have a sensor";
    response.message = ss.str();
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  if (stream_index == COLOR) {
    auto stream = streams_.at(stream_index);
    auto camera_settings = stream->getCameraSettings();
    if (camera_settings == nullptr) {
      response.success = false;
      response.message = stream_name_[stream_index] + " Camera settings not available";
      return false;
    }
    camera_settings->setGain(request.data);
  } else if (stream_index == INFRA1 || stream_index == INFRA2 || stream_index == DEPTH) {
    setIRGain(request.data);
  }
  return true;
}

bool OBCameraNode::getIRTemperatureCallback(GetDoubleRequest& request,
                                            GetDoubleResponse& response) {
  (void)request;
  double data = 0;
  auto ret = device_->getProperty(XN_MODULE_PROPERTY_RT_IR_TEMP, &data);
  if (ret != openni::STATUS_OK) {
    response.success = false;
    response.message = "Failed to get IR temperature";
    return false;
  }
  response.data = data;
  return true;
}

void OBCameraNode::setIRAutoExposure(bool status) {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->setProperty(XN_MODULE_PROPERTY_AE, (uint64_t)status);
}

int OBCameraNode::getIRExposure() {
  int data = 0;
  int data_size = 4;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->getProperty(openni::OBEXTENSION_ID_IR_EXP, (uint32_t*)&data, &data_size);
  return data;
}

void OBCameraNode::setIRExposure(uint32_t data) {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->setProperty(openni::OBEXTENSION_ID_IR_EXP, data);
}

int OBCameraNode::getIRGain() {
  int data = 0;
  int data_size = 4;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->getProperty(openni::OBEXTENSION_ID_IR_GAIN, (uint8_t*)&data, &data_size);
  return data;
}

void OBCameraNode::setIRGain(int data) {
  int data_size = 4;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->setProperty(openni::OBEXTENSION_ID_IR_GAIN, (uint8_t*)&data, data_size);
}

std::string OBCameraNode::getSerialNumber() {
  char serial_number_str[128] = {0};
  int data_size = sizeof(serial_number_str);
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->getProperty(openni::OBEXTENSION_ID_SERIALNUMBER, (uint8_t*)&serial_number_str,
                       &data_size);
  return serial_number_str;
}

bool OBCameraNode::getAutoWhiteBalanceEnabledCallback(GetInt32Request& request,
                                                      GetInt32Response& response,
                                                      const stream_index_pair& stream_index) {
  (void)request;
  if (!stream_started_[stream_index] || !device_->hasSensor(stream_index.first)) {
    std::stringstream ss;
    ss << "Stream " << stream_name_[stream_index] << " is not started or does not have a sensor";
    response.message = ss.str();
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  auto stream = streams_.at(stream_index);
  auto camera_settings = stream->getCameraSettings();
  if (camera_settings == nullptr) {
    response.data = 0;
    response.message = stream_name_[stream_index] + " Camera settings not available";
    return false;
  }
  response.data = camera_settings->getAutoWhiteBalanceEnabled();
  return true;
}

bool OBCameraNode::setAutoWhiteBalanceEnabledCallback(SetInt32Request& request,
                                                      SetInt32Response& response) {
  if (!device_->hasSensor(openni::SENSOR_COLOR)) {
    response.message = "Color sensor not available";
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  auto stream = streams_.at(COLOR);
  auto camera_settings = stream->getCameraSettings();
  if (camera_settings == nullptr) {
    response.message = stream_name_[COLOR] + " Camera settings not available";
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  auto data = request.data;
  auto rc = camera_settings->setAutoWhiteBalanceEnabled(data);
  if (rc != openni::STATUS_OK) {
    std::stringstream ss;
    ss << " Couldn't set auto white balance: " << openni::OpenNI::getExtendedError();
    response.message = ss.str();
    ROS_ERROR_STREAM(ss.str());
    return false;
  } else {
    return true;
  }
}

bool OBCameraNode::setAutoExposureCallback(std_srvs::SetBoolRequest& request,
                                           std_srvs::SetBoolResponse& response,
                                           const stream_index_pair& stream_index) {
  if (!stream_started_[stream_index] || !device_->hasSensor(stream_index.first)) {
    std::stringstream ss;
    ss << "Stream " << stream_name_[stream_index] << " is not started or does not have a sensor";
    response.message = ss.str();
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  openni::Status status;
  if (stream_index == COLOR) {
    auto stream = streams_.at(stream_index);
    auto camera_settings = stream->getCameraSettings();
    if (camera_settings == nullptr) {
      response.success = false;
      response.message = stream_name_[stream_index] + " Camera settings not available";
      ROS_ERROR_STREAM(response.message);
      return false;
    }
    status = camera_settings->setAutoExposureEnabled(request.data);
  } else if (stream_index == INFRA1 || stream_index == INFRA2 || stream_index == DEPTH) {
    std::lock_guard<decltype(device_lock_)> lock(device_lock_);
    status = device_->setProperty(XN_MODULE_PROPERTY_AE, request.data);
  } else {
    response.message = "Stream not supported set auto exposure";
    return false;
  }

  if (status != openni::STATUS_OK) {
    std::stringstream ss;
    ss << "Couldn't set auto exposure: " << openni::OpenNI::getExtendedError();
    response.message = ss.str();
    ROS_ERROR_STREAM(response.message);
    return false;
  } else {
    return true;
  }
}

bool OBCameraNode::setLaserEnableCallback(std_srvs::SetBoolRequest& request,
                                          std_srvs::SetBoolResponse& response) {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->setProperty(openni::OBEXTENSION_ID_LASER_EN, request.data);
  device_->setProperty(XN_MODULE_PROPERTY_EMITTER_STATE, request.data);
  response.success = true;
  return true;
}

bool OBCameraNode::setIRFloodCallback(std_srvs::SetBoolRequest& request,
                                      std_srvs::SetBoolResponse& response) {
  (void)response;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  ROS_INFO_STREAM("Setting IR flood to " << (request.data ? "true" : "false"));
  int data = static_cast<int>(request.data);
  device_->setProperty(XN_MODULE_PROPERTY_IRFLOOD_STATE, data);
  return true;
}

bool OBCameraNode::setLdpEnableCallback(std_srvs::SetBoolRequest& request,
                                        std_srvs::SetBoolResponse& response) {
  (void)response;
  //stopStreams();
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  auto status = device_->setProperty(XN_MODULE_PROPERTY_LDP_ENABLE, request.data);
  //startStreams();
  if (status != openni::STATUS_OK) {
    std::stringstream ss;
    ss << "Couldn't set LDP enable: " << openni::OpenNI::getExtendedError();
    ROS_ERROR_STREAM(ss.str());
    response.message = ss.str();
    return false;
  } else {
    return true;
  }
}

bool OBCameraNode::setFanEnableCallback(std_srvs::SetBoolRequest& request,
                                        std_srvs::SetBoolResponse& response) {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->setProperty(XN_MODULE_PROPERTY_FAN_ENABLE, request.data);
  response.success = true;
  return true;
}

bool OBCameraNode::getDeviceInfoCallback(GetDeviceInfoRequest& request,
                                         GetDeviceInfoResponse& response) {
  (void)request;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  auto device_info = device_->getDeviceInfo();
  response.info.name = device_info.getName();
  response.info.pid = device_info.getUsbProductId();
  response.info.vid = device_info.getUsbVendorId();
  char serial_number[64];
  int data_size = sizeof(serial_number);
  auto rc = device_->getProperty(openni::OBEXTENSION_ID_SERIALNUMBER, serial_number, &data_size);
  if (rc == openni::STATUS_OK) {
    response.info.serial_number = serial_number;
  } else {
    response.info.serial_number = "";
  }
  return true;
}

bool OBCameraNode::getCameraInfoCallback(GetCameraInfoRequest& request,
                                         GetCameraInfoResponse& response) {
  (void)request;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  auto camera_info = getColorCameraInfo();
  response.info = camera_info;
  return true;
}

bool OBCameraNode::getSDKVersionCallback(GetStringRequest& request, GetStringResponse& response) {
  (void)request;
  nlohmann::json data;
  data["ros_sdk_version"] = OB_ROS_VERSION_STR;
  data["openni_version"] = ONI_VERSION_STRING;
  char buffer[128] = {0};
  int data_size = sizeof(buffer);
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->getProperty(XN_MODULE_PROPERTY_SENSOR_PLATFORM_STRING, buffer, &data_size);
  data["firmware_version"] = buffer;
  response.data = data.dump(2);
  return true;
}

bool OBCameraNode::getDeviceTypeCallback(GetStringRequest& request, GetStringResponse& response) {
  (void)request;
  char device_type_str[128] = {0};
  int data_size = sizeof(device_type_str);
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  device_->getProperty(openni::OBEXTENSION_ID_DEVICETYPE, (uint8_t*)&device_type_str, &data_size);
  response.data = device_type_str;
  return true;
}

bool OBCameraNode::getSerialNumberCallback(GetStringRequest& request, GetStringResponse& response) {
  (void)request;
  response.data = getSerialNumber();
  return true;
}

bool OBCameraNode::switchIRCameraCallback(SetStringRequest& request, SetStringResponse& response) {
  const int data_size = 4;
  int data;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  if (request.data == "left") {
    data = 0;
    device_->setProperty(XN_MODULE_PROPERTY_SWITCH_IR, (uint8_t*)&data, data_size);
  } else if (request.data == "right") {
    data = 1;
    device_->setProperty(XN_MODULE_PROPERTY_SWITCH_IR, (uint8_t*)&data, data_size);
  } else {
    response.message = "Invalid IR camera name";
    ROS_ERROR_STREAM(response.message);
    return false;
  }
  return true;
}

bool OBCameraNode::getCameraParamsCallback(GetCameraParamsRequest& request,
                                           GetCameraParamsResponse& response) {
  (void)request;
  auto camera_params = getCameraParams();
  for (int i = 0; i < 9; i++) {
    response.r2l_r[i] = camera_params.r2l_r[i];
    if (i < 4) {
      response.l_intr_p[i] = camera_params.l_intr_p[i];
      response.r_intr_p[i] = camera_params.r_intr_p[i];
    }
    if (i < 3) {
      response.r2l_t[i] = camera_params.r2l_t[i];
    }
    if (i < 5) {
      response.l_k[i] = camera_params.l_k[i];
      response.r_k[i] = camera_params.r_k[i];
    }
  }
  return true;
}

bool OBCameraNode::toggleSensorCallback(std_srvs::SetBoolRequest& request,
                                        std_srvs::SetBoolResponse& response,
                                        const stream_index_pair& stream_index) {
  if (request.data) {
    ROS_INFO_STREAM(stream_name_[stream_index] << " ON");
  } else {
    ROS_INFO_STREAM(stream_name_[stream_index] << " OFF");
  }
  response.success = toggleSensor(stream_index, request.data, response.message);
  return true;
}

bool OBCameraNode::saveImagesCallback(std_srvs::EmptyRequest& request,
                                      std_srvs::EmptyResponse& response) {
  (void)request;
  (void)response;
  ROS_INFO_STREAM("Saving images");
  if (enable_[INFRA1]) {
    save_images_[INFRA1] = true;
  }
  if (enable_[COLOR]) {
    save_images_[COLOR] = true;
  }
  if (enable_[DEPTH]) {
    save_images_[DEPTH] = true;
  }
  return true;
}

bool OBCameraNode::getSupportedVideoModesCallback(GetStringRequest& request,
                                                  GetStringResponse& response,
                                                  const stream_index_pair& stream_index) {
  (void)request;
  if (!supported_video_modes_.count(stream_index)) {
    response.data = "";
    response.message = "No supported video modes";
    return false;
  } else {
    auto modes = supported_video_modes_[stream_index];
    nlohmann::json data;
    for (auto& mode : modes) {
      std::stringstream ss;
      ss << mode.getResolutionX() << "x" << mode.getResolutionY() << "@" << mode.getFps();
      if (data.empty() || data.back() != ss.str()) {
        data.push_back(ss.str());
      }
    }
    response.data = data.dump(2);
    return true;
  }
}

// bool OBCameraNode::getLaserStatusCallback(GetBoolRequest& request, GetBoolResponse& response) {
//   (void)request;
//   std::lock_guard<decltype(device_lock_)> lock(device_lock_);
//   int data = 0;
//   int data_size = 4;
//   device_->getProperty(XN_MODULE_PROPERTY_EMITTER_STATE, (uint8_t*)&data, &data_size);
//   response.data = data;
//   return true;
// }

bool OBCameraNode::getLdpStatusCallback(GetBoolRequest& request, GetBoolResponse& response) {
  (void)request;
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  int data = 0;
  int data_size = 4;
  device_->getProperty(XN_MODULE_PROPERTY_LDP_STATUS, (uint8_t*)&data, &data_size);
  response.data = data;
  return true;
}

bool OBCameraNode::resetIRGainCallback(std_srvs::EmptyRequest& request,
                                       std_srvs::EmptyResponse& response) {
  (void)request;
  (void)response;
  ROS_INFO_STREAM("Resetting IR gain");
  setIRGain(init_ir_gain_);
  return true;
}

bool OBCameraNode::resetIRExposureCallback(std_srvs::EmptyRequest& request,
                                           std_srvs::EmptyResponse& response) {
  (void)request;
  (void)response;
  ROS_INFO_STREAM("Resetting IR exposure");
  setIRExposure(init_ir_exposure_);
  return true;
}

bool OBCameraNode::toggleSensor(const stream_index_pair& stream_index, bool enabled,
                                std::string& msg) {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  if (!device_->hasSensor(stream_index.first)) {
    std::stringstream ss;

    ss << "doesn't  have " << stream_name_[stream_index];
    msg = ss.str();
    ROS_WARN_STREAM(msg);
    return false;
  }

  if (enabled) {
    enable_[stream_index] = true;
  } else {
    enable_[stream_index] = false;
  }
  stopStreams();
  startStreams();
  return true;
}

}  // namespace astra_camera
