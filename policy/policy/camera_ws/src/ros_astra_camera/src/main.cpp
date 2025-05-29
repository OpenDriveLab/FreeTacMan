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

#include "astra_camera/ob_camera_node_factory.h"
using namespace astra_camera;
int main(int argc, char** argv) {
  ROS_INFO_STREAM("Starting camera node...");
  ros::init(argc, argv, "astra_camera_node");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  ROS_INFO_STREAM("Creating camera node...");
  OBCameraNodeFactory node_factory(nh, nh_private);
  ROS_INFO_STREAM("Creating camera node done...");
  ros::spin();
  return 0;
}
