import debugpy
try:
    # 9501 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("127.0.0.1", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


import numpy as np
import collections
from collections import deque
import rospy
from geometry_msgs.msg import PoseArray


class RosOperator:
    def __init__(self):
        self.eep_right_deque = None
        self.eep_left_deque = None
        self.init()
        self.init_ros()

    def init(self):
        self.eep_left_deque = deque()
        self.eep_right_deque = deque()

    def get_frame(self):
        # import ipdb;ipdb.set_trace()

        if len(self.eep_left_deque) == 0:
            return False
        if len(self.eep_right_deque) == 0:
            return False

        eep_left  = None
        eep_right = None

        self.eep_left_deque.popleft()
        eep_left = self.eep_left_deque.popleft()
        self.eep_right_deque.popleft()
        eep_right = self.eep_right_deque.popleft()

        return (eep_left, eep_right)

    def eep_left_callback(self, msg):
        if len(self.eep_left_deque) >= 2000:
            self.eep_left_deque.popleft()
        self.eep_left_deque.append(msg)

    def eep_right_callback(self, msg):
        if len(self.eep_right_deque) >= 2000:
            self.eep_right_deque.popleft()
        self.eep_right_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        # import ipdb;ipdb.set_trace()
        rospy.Subscriber('/end_pose_left', PoseArray, self.eep_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber('/end_pose_right', PoseArray, self.eep_right_callback, queue_size=1000, tcp_nodelay=True)

    def process(self):

        count = 0

        while (count < 99999) and not rospy.is_shutdown():
            # 2 收集数据
            result = self.get_frame()
            if not result:
                print("result none")
                continue
            else:
                print("result exis")
            count += 1

            (eep_left, eep_right) = result

        obs = collections.OrderedDict()
        obs['eep'] = np.concatenate((np.array(eep_left.position), np.array(eep_left.orientation), np.array(eep_right.position), np.array(eep_right.orientation)), axis=0)
       
        return None
    

def main():
    ros_operator = RosOperator()
    timesteps, actions = ros_operator.process()


if __name__ == '__main__':
    main()