import message_filters
from sensor_msgs.msg import Image
import rospy


def callback(depth_msg, rgb_msg):
    print("hello")


def main():
    rospy.init_node('test_sync', anonymous=True)
    rgb_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)

    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 1)
    ts.registerCallback(callback)
    rospy.spin()


if __name__ == '__main__':
    main()
