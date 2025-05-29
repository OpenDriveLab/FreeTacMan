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
#include <ros/ros.h>

#include "astra_camera/ob_context.h"

void DeviceConnectedCallback(const openni::DeviceInfo* device_info) {
  std::cout << "Device connected: " << device_info->getName() << std::endl;
  auto device = std::make_shared<openni::Device>();
  auto uri = device_info->getUri();
  std::cout << "URI: " << uri << std::endl;
  device->open(uri);
  char serial_number[64];
  int data_size = sizeof(serial_number);
  device->getProperty(openni::OBEXTENSION_ID_SERIALNUMBER, serial_number, &data_size);
  std::cout << "Serial number: " << serial_number << std::endl;
  device->close();
}

int main() {
  openni::OpenNI::initialize();
  auto disconnected_cb = [](const openni::DeviceInfo* device_info) 
  {
    std::cout << "device " << device_info->getUri() << " disconnected" << std::endl;
  };

  auto context = std::make_unique<astra_camera::Context>(disconnected_cb);
  auto device_list = context->queryDeviceList();
  
  for (auto& device_info : device_list) 
  {
    DeviceConnectedCallback(&device_info);
  }
  openni::OpenNI::shutdown();
  return 0;
}
