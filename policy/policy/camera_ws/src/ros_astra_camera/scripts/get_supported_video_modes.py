#!/usr/bin/env python3

from astra_camera.srv import GetString

import rospy
import math
import sys
import json


def main():
    rospy.init_node('GetSupportedVideoModes', anonymous=True)
    stream = sys.argv[1]
    service = "/camera/get_" + stream + "_supported_video_modes"
    rospy.wait_for_service(service)
    try:
      res = rospy.ServiceProxy(service, GetString)
      response = res()
      data = json.loads(response.data)
      for item in data:
        print("%s" % item)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    rospy.spin()


if __name__ == '__main__':
    main()