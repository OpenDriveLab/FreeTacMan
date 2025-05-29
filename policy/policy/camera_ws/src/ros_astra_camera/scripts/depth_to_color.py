#!/usr/bin/env python3

import message_filters
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from tf2_ros import TransformListener,Buffer
import math
import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point


tf_listener = None
tf_buffer = None
def callback( depth_msg : Image):
  cv_bridge = CvBridge()
  depth_img = cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
  center_x = depth_img.shape[1] / 2
  center_y = depth_img.shape[0] / 2
  try:
      trans = tf_buffer.lookup_transform('camera_depth_frame', 'camera_color_frame', rospy.Time(0))
  except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
      rospy.loginfo(e)
      return

  # Create a point in the depth camera's coordinate frame
  depth_point = PointStamped()
  depth_point.header.frame_id = 'camera_depth_frame'
  depth_point.header.stamp = rospy.Time.now()
  depth_point.point.x = center_x
  depth_point.point.y = center_y
  depth_point.point.z = depth_img[int(center_y), int(center_x)]

  # Now we transform the point from the depth camera's coordinate frame to the color camera's coordinate frame
  try:
      color_point = do_transform_point(depth_point, trans)
  except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
      rospy.loginfo(e)
      return

  # The point's position in the color frame is now stored in color_point.point
  color_x = color_point.point.x
  color_y = color_point.point.y
  print("from depth {},{} to color {},{}".format(center_x,center_y,color_x,color_y))

  

def main():
    rospy.init_node('test_sync', anonymous=True)
    depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, callback)
    global tf_listener, tf_buffer
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer)
    rospy.spin()


if __name__ == '__main__':
    main()
