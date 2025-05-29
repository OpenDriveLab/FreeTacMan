#coding=utf-8
import os
import numpy as np
import cv2
import h5py
import argparse
import rospy

from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist

import sys
sys.path.append("./")
def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        if 'qvel' in root.keys():
            qvel = root['/observations/qvel'][()]
        else:
            qvel = None

        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        action = root['/action'][()]
        
        if 'base_action' in root.keys():
            base_action = root['/base_action'][()]
        else:
            base_action = None
        
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

    return qpos, qvel, effort, action, base_action, image_dict

def main(args):
    rospy.init_node("replay_node")
    bridge = CvBridge()
    img_left_publisher = rospy.Publisher(args.img_left_topic, Image, queue_size=10)
    img_right_publisher = rospy.Publisher(args.img_right_topic, Image, queue_size=10)
    img_front_publisher = rospy.Publisher(args.img_front_topic, Image, queue_size=10)
    
    puppet_arm_left_publisher = rospy.Publisher(args.puppet_arm_left_topic, JointState, queue_size=10)
    puppet_arm_right_publisher = rospy.Publisher(args.puppet_arm_right_topic, JointState, queue_size=10)
    
    master_arm_left_publisher = rospy.Publisher(args.master_arm_left_topic, JointState, queue_size=10)
    master_arm_right_publisher = rospy.Publisher(args.master_arm_right_topic, JointState, queue_size=10)
    
    robot_base_publisher = rospy.Publisher(args.robot_base_topic, Twist, queue_size=10)


    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    task_name   = args.task_name
    dataset_name = f'episode_{episode_idx}'

    origin_left = [-0.0057,-0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279]  
    origin_right = [ 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]

    
    joint_state_msg = JointState()
    joint_state_msg.header =  Header()
    joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
    twist_msg = Twist()

    rate = rospy.Rate(args.frame_rate)
    
    qposs, qvels, efforts, actions, base_actions, image_dicts = load_hdf5(os.path.join(dataset_dir, task_name), dataset_name)
    
    
    if args.only_pub_master:
        last_action = [-0.0057,-0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279, 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
        rate = rospy.Rate(100)
        for action in actions:
            if(rospy.is_shutdown()):
                    break
            
            new_actions = np.linspace(last_action, action, 20) # 插值
            last_action = action
            for act in new_actions:
                print(np.round(act[:7], 4))
                cur_timestamp = rospy.Time.now()  # 设置时间戳
                joint_state_msg.header.stamp = cur_timestamp 
                
                joint_state_msg.position = act[:7]
                master_arm_left_publisher.publish(joint_state_msg)

                joint_state_msg.position = act[7:]
                master_arm_right_publisher.publish(joint_state_msg)   

                if(rospy.is_shutdown()):
                    break
                rate.sleep() 
    
    else:
        i = 0
        while(not rospy.is_shutdown() and i < len(actions)):
            print("left: ", np.round(qposs[i][:7], 4), " right: ", np.round(qposs[i][7:], 4))
            
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

            cur_timestamp = rospy.Time.now()  # 设置时间戳
            
            joint_state_msg.header.stamp = cur_timestamp 

            if "left" in args.arm_names:
                joint_state_msg.position = actions[i][:7]
                master_arm_left_publisher.publish(joint_state_msg) 

                joint_state_msg.position = actions[i][7:]
                master_arm_right_publisher.publish(joint_state_msg)

                joint_state_msg.position = qposs[i][:7]
                puppet_arm_left_publisher.publish(joint_state_msg) 

                joint_state_msg.position = qposs[i][7:]
                puppet_arm_right_publisher.publish(joint_state_msg) 
            else:
                ###################### below is exchange(left & right) #############################
                joint_state_msg.position = actions[i][:7]
                master_arm_right_publisher.publish(joint_state_msg)  #############################

                joint_state_msg.position = actions[i][7:]
                master_arm_left_publisher.publish(joint_state_msg) ################################

                joint_state_msg.position = qposs[i][:7]
                puppet_arm_right_publisher.publish(joint_state_msg)##############################


                joint_state_msg.position = qposs[i][7:]
                puppet_arm_left_publisher.publish(joint_state_msg)   ################################


            if "cam_high" in cam_names:
                img_front_publisher.publish(bridge.cv2_to_imgmsg(image0, "bgr8"))
            if "cam_left_wrist" in cam_names:
                img_left_publisher.publish(bridge.cv2_to_imgmsg(image1, "bgr8"))
            if "cam_right_wrist" in cam_names:
                img_right_publisher.publish(bridge.cv2_to_imgmsg(image2, "bgr8"))
    
            
            # twist_msg.linear.x = base_actions[i][0]
            # twist_msg.angular.z = base_actions[i][1]
            # robot_base_publisher.publish(twist_msg)

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
    
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=20, required=False)
    
    parser.add_argument('--only_pub_master', action='store_true', help='only_pub_master',required=False)
    
    

    args = parser.parse_args()
    main(args)
    # python collect_data.py --max_timesteps 500 --is_compress --episode_idx 0 