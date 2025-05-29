#!/usr/bin/env python3
import imp
import rospy
import time
from sensor_msgs.msg import Image
import sys

get_msg = False


def callback(data):
    global get_msg
    get_msg = True


def main():
    rospy.init_node('test_sub', anonymous=True)
    try:
        for i in range(100000):
            sub = rospy.Subscriber("/camera/depth/image_raw", Image, callback)
            global get_msg
            while not get_msg:
                time.sleep(0.1)
            get_msg = False
            sub.unregister()
            print("%d test case passed" % i)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    main()