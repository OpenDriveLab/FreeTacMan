#!/usr/bin/env python3

from numpy import mat
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import rospy
import math

def pointCloudCb(data):
  for p in pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True):
     dist = p[0]*p[0] + p[1]*p[1] + p[2] *p[2]
     print("dist %s" % math.sqrt(dist))

 
def main():
    rospy.init_node('get_point_cloud_dist', anonymous=True)
    rospy.Subscriber("/camera/depth/points", PointCloud2, pointCloudCb)
    rospy.spin()

if __name__ == '__main__':
    main()
