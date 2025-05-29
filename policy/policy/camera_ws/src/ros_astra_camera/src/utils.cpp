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
#include "astra_camera/utils.h"

#include <sensor_msgs/point_cloud_conversion.h>

#include "sensor_msgs/PointCloud.h"
#include "sensor_msgs/distortion_models.h"
#include "sensor_msgs/point_cloud2_iterator.h"

namespace astra_camera {

bool operator==(const openni::VideoMode& lhs, const openni::VideoMode& rhs) {
  return lhs.getResolutionY() == rhs.getResolutionY() &&
         lhs.getResolutionX() == rhs.getResolutionX() && lhs.getFps() == rhs.getFps() &&
         lhs.getPixelFormat() == rhs.getPixelFormat();
}

bool operator!=(const openni::VideoMode& lhs, const openni::VideoMode& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const openni::VideoMode& video_mode) {
  os << "Resolution :" << video_mode.getResolutionX() << "x" << video_mode.getResolutionY() << "@"
     << video_mode.getFps() << "Hz" << std::endl
     << "format " << PixelFormatToString(video_mode.getPixelFormat());
  return os;
}

tf2::Quaternion rotationMatrixToQuaternion(const std::vector<float>& rotation) {
  CHECK_EQ(rotation.size(), 9u);
  Eigen::Matrix3f m;
  // We need to be careful about the order, as RS2 rotation matrix is
  // column-major, while Eigen::Matrix3f expects row-major.
  m << rotation[0], rotation[3], rotation[6], rotation[1], rotation[4], rotation[7], rotation[2],
      rotation[5], rotation[8];
  Eigen::Quaternionf q(m);
  return {q.x(), q.y(), q.z(), q.w()};
}

Extrinsics obExtrinsicsToMsg(const std::vector<float>& rotation,
                             const std::vector<float>& transition, const std::string& frame_id) {
  CHECK_EQ(rotation.size(), 9u);
  CHECK_EQ(transition.size(), 3u);
  Extrinsics msg;
  for (int i = 0; i < 9; ++i) {
    msg.rotation[i] = rotation[i];
    if (i < 3) {
      msg.translation[i] = transition[i];
    }
  }

  msg.header.frame_id = frame_id;
  return msg;
}

bool isValidCameraParams(const OBCameraParams& params) {
  if (std::isnan(params.l_intr_p[0]) || std::isnan(params.l_intr_p[1]) ||
      std::isnan(params.l_intr_p[2]) || std::isnan(params.l_intr_p[3])) {
    return false;
  }
  return true;
}

void cameraParameterPrinter(const std::vector<float>& rotation,
                            const std::vector<float>& transition) {
  CHECK_EQ(rotation.size(), 9u);
  CHECK_EQ(transition.size(), 3u);
  std::cout << "Rotation: " << std::endl;
  for (int i = 0; i < 9; ++i) {
    std::cout << rotation[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Translation: " << std::endl;
  for (int i = 0; i < 3; ++i) {
    std::cout << transition[i] << " ";
  }
  std::cout << std::endl;
}

std::string PixelFormatToString(const openni::PixelFormat& format) {
  switch (format) {
    case openni::PIXEL_FORMAT_DEPTH_1_MM:
      return "PIXEL_FORMAT_DEPTH_1_MM";
    case openni::PIXEL_FORMAT_DEPTH_100_UM:
      return "PIXEL_FORMAT_DEPTH_100_UM";
    case openni::PIXEL_FORMAT_SHIFT_9_2:
      return "PIXEL_FORMAT_SHIFT_9_2";
    case openni::PIXEL_FORMAT_SHIFT_9_3:
      return "PIXEL_FORMAT_SHIFT_9_3";
    case openni::PIXEL_FORMAT_RGB888:
      return "PIXEL_FORMAT_RGB888";
    case openni::PIXEL_FORMAT_YUV422:
      return "PIXEL_FORMAT_YUV422";
    case openni::PIXEL_FORMAT_GRAY8:
      return "PIXEL_FORMAT_GRAY8";
    case openni::PIXEL_FORMAT_GRAY16:
      return "PIXEL_FORMAT_GRAY16";
    case openni::PIXEL_FORMAT_JPEG:
      return "PIXEL_FORMAT_JPEG";
    case openni::PIXEL_FORMAT_YUYV:
      return "PIXEL_FORMAT_YUYV";
    case openni::PIXEL_FORMAT_LOG:
      return "PIXEL_FORMAT_LOG";
  }
  return "Unknown";
}
void savePointToPly(sensor_msgs::PointCloud2::Ptr cloud, const std::string& filename) {
  sensor_msgs::PointCloud out_point_cloud;
  sensor_msgs::convertPointCloud2ToPointCloud(*cloud, out_point_cloud);
  size_t point_size = 0;
  FILE* fp = fopen(filename.c_str(), "wb+");
  for (auto& point : out_point_cloud.points) {
    if (!std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z)) {
      point_size++;
    }
  }
  ROS_INFO_STREAM("Point size: " << point_size);
  fprintf(fp, "ply\n");
  fprintf(fp, "format ascii 1.0\n");
  fprintf(fp, "element vertex %zu\n", point_size);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "end_header\n");
  for (const auto& point : out_point_cloud.points) {
    if (!std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z)) {
      fprintf(fp, "%.3f %.3f %.3f\n", point.x, point.y, point.z);
    }
  }
  fflush(fp);
  fclose(fp);
}

void saveRGBPointToPly(sensor_msgs::PointCloud2::Ptr cloud, const std::string& filename) {
  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud, "b");
  size_t point_size = cloud->width * cloud->height;
  std::vector<geometry_msgs::Point32> points;
  std::vector<std::vector<uint8_t>> rgb;
  for (size_t i = 0; i < point_size; i++) {
    if (!std::isnan(*iter_x) && !std::isnan(*iter_y) && !std::isnan(*iter_z)) {
      geometry_msgs::Point32 point;
      point.x = *iter_x;
      point.y = *iter_y;
      point.z = *iter_z;
      points.push_back(point);
      std::vector<uint8_t> rgb_point;
      rgb_point.push_back(*iter_r);
      rgb_point.push_back(*iter_g);
      rgb_point.push_back(*iter_b);
      rgb.push_back(rgb_point);
    }
    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_r;
    ++iter_g;
    ++iter_b;
  }
  point_size = points.size();
  FILE* fp = fopen(filename.c_str(), "wb+");

  ROS_INFO_STREAM("Point size: " << point_size);
  fprintf(fp, "ply\n");
  fprintf(fp, "format ascii 1.0\n");
  fprintf(fp, "element vertex %zu\n", point_size);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  for (size_t i = 0; i < point_size; ++i) {
    fprintf(fp, "%.3f %.3f %.3f %d %d %d\n", points[i].x, points[i].y, points[i].z, rgb[i][0],
            rgb[i][1], rgb[i][2]);
  }
  fflush(fp);
  fclose(fp);
}

MultiDeviceSyncMode getMultiDeviceSyncMode(const std::string& mode) {
  if (mode == "none") {
    return MultiDeviceSyncMode::None;
  } else if (mode == "master" || mode == "main") {
    return MultiDeviceSyncMode::Master;
  } else if (mode == "slave" || mode == "sub") {
    return MultiDeviceSyncMode::Slave;
  } else {
    return MultiDeviceSyncMode::None;
  }
}


}  // namespace astra_camera
