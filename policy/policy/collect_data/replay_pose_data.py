#coding=utf-8
import os
import math
import numpy as np
import cv2
import h5py
import argparse
import rospy

from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, Pose, PoseArray
from piper_msgs.msg import PosCmd

import sys
sys.path.append("./")

POS_FACTOR = 1e3
ROT_FACTOR = 180 / math.pi

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]

        action = root['/action'][()]
        
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                image_len = int(compress_len[cam_id, frame_id])
                
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list

    return qpos, action, image_dict

def main(args):
    rospy.init_node("replay_node")
    bridge = CvBridge()
    img_left_publisher = rospy.Publisher(args.img_left_topic, Image, queue_size=10)
    img_right_publisher = rospy.Publisher(args.img_right_topic, Image, queue_size=10)
    img_front_publisher = rospy.Publisher(args.img_front_topic, Image, queue_size=10)
    
    end_pose_left_publisher = rospy.Publisher(args.left_end_pos_cmd_topic, PosCmd, queue_size=10)
    end_pose_right_publisher = rospy.Publisher(args.right_end_pos_cmd_topic, PosCmd, queue_size=10)

    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    task_name = args.task_name
    dataset_name = f'episode_{episode_idx}'
    
    end_pose_msg = PosCmd()
    # end_pose_msg.header = Header()
    # end_pose_msg.header.stamp = rospy.Time.now()

    rate = rospy.Rate(args.frame_rate)
    
    qposs, actions, image_dicts = load_hdf5(os.path.join(dataset_dir, task_name), dataset_name)
    
    
    i = 0
    while(not rospy.is_shutdown() and i < len(actions)):
        cam_names = [k for k in image_dicts.keys()]

        for cam_name in cam_names:
            if cam_name == "cam_high":
                image0 = image_dicts[cam_names[0]][i] 
                image0 = image0[:, :, [2, 1, 0]]  # swap B and R channel
            elif cam_name == "cam_left_wrist":
                image1 = image_dicts[cam_names[1]][i] 
                image1 = image1[:, :, [2, 1, 0]]  # swap B and R channel
            elif cam_name == "cam_right_wrist":
                image2 = image_dicts[cam_names[2]][i] 
                image2 = image2[:, :, [2, 1, 0]]  # swap B and R channel

        # cur_timestamp = rospy.Time.now()  # 设置时间戳
        # end_pose_msg.header.stamp = cur_timestamp 

        if "left" in args.arm_names:
            end_pose_msg.x, end_pose_msg.y, end_pose_msg.z = actions[i][:3] * POS_FACTOR
            end_pose_msg.roll, end_pose_msg.pitch, end_pose_msg.yaw = actions[i][3:6] * ROT_FACTOR
            end_pose_msg.gripper = actions[i][6]
            end_pose_left_publisher.publish(end_pose_msg)
            
        elif "right" in args.arm_names:
            end_pose_msg.x, end_pose_msg.y, end_pose_msg.z = actions[i][:3] * POS_FACTOR
            end_pose_msg.roll, end_pose_msg.pitch, end_pose_msg.yaw = actions[i][3:6] * ROT_FACTOR
            end_pose_msg.gripper = actions[i][6]
            end_pose_right_publisher.publish(end_pose_msg)

        if "cam_high" in cam_names:
            img_front_publisher.publish(bridge.cv2_to_imgmsg(image0, "bgr8"))
        if "cam_left_wrist" in cam_names:
            img_left_publisher.publish(bridge.cv2_to_imgmsg(image1, "bgr8"))
        if "cam_right_wrist" in cam_names:
            img_right_publisher.publish(bridge.cv2_to_imgmsg(image2, "bgr8"))


        i += 1
        rate.sleep() 
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)

    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',default=0, required=False)
    
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    parser.add_argument('--arm_names', action='store', type=int, help='replay which arm data',  
                        # default=['left', 'right'], required=False)
                        default=['right'], required=False)
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--left_end_pos_cmd_topic', action='store', type=str, help='left_end_pos_cmd_topic',
                        default='pos_cmd_left', required=False)
    parser.add_argument('--right_end_pos_cmd_topic', action='store', type=str, help='right_end_pos_cmd_topic',
                        default='pos_cmd_right', required=False)
    
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=20, required=False)
    
    args = parser.parse_args()
    main(args)
    # python collect_data.py --max_timesteps 500 --is_compress --episode_idx 0 