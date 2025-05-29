#coding=utf-8
import os
import time
import numpy as np
import h5py
import argparse
import dm_env
import cv2
import queue

import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import threading
import collections

import sys
sys.path.append("./")



# 保存数据函数
def save_data(args, timesteps, actions, dataset_path):
    use_depth_image = args.use_depth_image
    is_compress = args.is_compress
    
    # 数据字典
    data_dict = {
        # obs包含qpos，qvel， effort ,acition, image, base_action
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
    }

    # 相机字典  观察的图像
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if use_depth_image:
             data_dict[f'/observations/depths/{cam_name}'] = []
        

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)   # 动作  当前动作
        ts = timesteps.pop(0)     # 奖励  前一帧

        # 往字典里面添值
        # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])

        # 实际发的action
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        # 相机数据
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            
            if use_depth_image:
                data_dict[f'/observations/depths/{cam_name}'].append(ts.observation['depths'][cam_name])

    pass
    if is_compress:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
        compressed_len = []
        # 3个相机
        for cam_name in args.camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])   # 压缩的长度
            
            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            
            # 更新图像
            data_dict[f'/observations/images/{cam_name}'] = compressed_list

        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()  # 取最大的图像长度，图像压缩后就是一个buf序列
        
        for cam_name in args.camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list

        # 深度图压缩处理
        if use_depth_image:
            compressed_len_depth = []
            # 3个相机
            for cam_name in args.camera_names:
                depth_list = data_dict[f'/observations/depths/{cam_name}']
                compressed_list_depth = []
                compressed_len_depth.append([])   # 压缩的长度
                
                for depth in depth_list:
                    result, encoded_depth = cv2.imencode('.jpg', depth, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    
                    compressed_list_depth.append(encoded_depth)
                    compressed_len_depth[-1].append(len(encoded_depth))
                
                # 更新图像
                data_dict[f'/observations/depths/{cam_name}'] = compressed_list_depth

            compressed_len_depth = np.array(compressed_len_depth)
            padded_size_depth = compressed_len_depth.max()  # 取最大的图像长度，图像压缩后就是一个buf序列
            
            for cam_name in args.camera_names:
                compressed_depth_list = data_dict[f'/observations/depths/{cam_name}']
                padded_compressed_depth_list = []
                for compressed_depth in compressed_depth_list:
                    padded_compressed_depth = np.zeros(padded_size_depth, dtype='uint8')
                    depth_len = len(compressed_depth)
                    padded_compressed_depth[:depth_len] = compressed_depth
                    padded_compressed_depth_list.append(padded_compressed_depth)
                data_dict[f'/observations/depths/{cam_name}'] = padded_compressed_depth_list




    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        #
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        if args.is_compress:
            root.attrs['compress'] = True

        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        
        image = obs.create_group('images')

        depth = obs.create_group('depths')

        for cam_name in args.camera_names:
            if args.is_compress:
                _ = image.create_dataset(cam_name, (args.max_timesteps, padded_size), dtype='uint8',
                                         chunks=(1, padded_size), )        
                if use_depth_image:
                    # _ = depth.create_dataset(cam_name, (args.max_timesteps, 400, 640), dtype='uint8',
                    #                      chunks=(1, 400, 640), )
                    _ = depth.create_dataset(cam_name, (args.max_timesteps, padded_size_depth), dtype='uint8',
                                         chunks=(1, padded_size_depth), )       


            else:
                _ = image.create_dataset(cam_name, (args.max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
                if use_depth_image:
                    _ = depth.create_dataset(cam_name, (args.max_timesteps, 400, 640), dtype='uint8',
                                         chunks=(1, 400, 640), )

        _ = obs.create_dataset('qpos', (args.max_timesteps, 14))
        _ = obs.create_dataset('qvel', (args.max_timesteps, 14))
        _ = obs.create_dataset('effort', (args.max_timesteps, 14))
        _ = root.create_dataset('action', (args.max_timesteps, 14))
        _ = root.create_dataset('base_action', (args.max_timesteps, 2))


        if is_compress:
            _ = root.create_dataset('compress_len', (len(args.camera_names), args.max_timesteps))
            root['/compress_len'][...] = compressed_len

            
        # data_dict写入h5py.File
        for name, array in data_dict.items():   # 名字+值
            root[name][...] = array

    print(f'Saving: {time.time() - t0:.1f} secs', dataset_path)


class RosOperator:
    def __init__(self, args):
        self.init(args)

    def init(self, args):
        self.args = args
        self.use_depth_image = self.args.use_depth_image
        self.use_robot_base  = self.args.use_robot_base

        self.bridge = CvBridge()
        self.imgl_queue, self.imgr_queue, self.imgf_queue = [queue.Queue() for _ in range(3)]
        self.depthl_queue, self.depthr_queue, self.depthf_queue = [queue.Queue() for _ in range(3)]
        self.masterl_queue, self.masterr_queue, self.puppetl_queue, self.puppetr_queue = [queue.Queue() for _ in range(4)]
        self.robot_base_deque = queue.Queue()

        dataset_dir = os.path.join(args.dataset_dir, args.task_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        self.dataset_path = os.path.join(dataset_dir, "episode_" + str(args.episode_idx))

        self.register_sub()   # 定义订阅消息话题

    def register_sub(self):
        rospy.init_node('record_episodes', anonymous=True)
        
        rospy.Subscriber(self.args.img_left_topic,  Image, lambda msg: self.imgl_queue.put(msg), queue_size=60, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, lambda msg: self.imgr_queue.put(msg), queue_size=60, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, lambda msg: self.imgf_queue.put(msg), queue_size=60, tcp_nodelay=True)
        
        if self.use_depth_image:
            rospy.Subscriber(self.args.depth_left_topic,  Image, lambda msg: self.depthl_queue.put(msg), queue_size=60, tcp_nodelay=True)
            rospy.Subscriber(self.args.depth_right_topic, Image, lambda msg: self.depthr_queue.put(msg), queue_size=60, tcp_nodelay=True)
            rospy.Subscriber(self.args.depth_front_topic, Image, lambda msg: self.depthf_queue.put(msg), queue_size=60, tcp_nodelay=True)


        rospy.Subscriber(self.args.master_arm_left_topic,  JointState, lambda msg: self.masterl_queue.put(msg), queue_size=200, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState, lambda msg: self.masterr_queue.put(msg), queue_size=200, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic,  JointState, lambda msg: self.puppetl_queue.put(msg), queue_size=200, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, lambda msg: self.puppetr_queue.put(msg), queue_size=200, tcp_nodelay=True)
        
        if self.use_robot_base:
            rospy.Subscriber(self.args.robot_base_topic, Odometry, lambda msg: self.robot_base_deque.put(msg), queue_size=60, tcp_nodelay=True)

 
    def collet_data(self):
        rate = rospy.Rate(30)
        input("\033[32m\nPlease press any key to continue\n\033[0m")
        
        count = 0      # 统计 timesteps
        timesteps = [] # obs
        actions = []   # acticn
        
        # 循环max_timesteps次
        while (count < self.args.max_timesteps + 1 and not rospy.is_shutdown()):
            time1 = time.time()     # 起始时间
            # 判读图像和深度图队列是否为空
            if self.imgl_queue.empty() or self.imgr_queue.empty() or self.imgf_queue.empty() or \
                (self.use_depth_image and (self.depthl_queue.empty() or self.depthr_queue.empty() or self.depthf_queue.empty())):
                print("\033[31mPlease check topic\n\033[0m")
                rate.sleep()
                continue
            
            if self.masterl_queue.empty() or self.masterr_queue.empty() or \
                self.puppetl_queue.empty() or self.puppetr_queue.empty() or \
                (  self.use_robot_base and ( self.robot_base_deque.empty() )  ):
                print("\033[31mPlease check topic\n\033[0m")
                rate.sleep()
                continue
            
            imgl, imgr, imgl, depthl, depthr, depthl = [None for _ in range(6)]
            masterl, masterr, puppetl, puppetr = [None for _ in range(4)]
            robot_base = None

            min_timestamps = min(self.imgl_queue.queue[-1].header.stamp.to_sec(),
                                 self.imgr_queue.queue[-1].header.stamp.to_sec(),
                                 self.imgf_queue.queue[-1].header.stamp.to_sec())  - 0.01
            
            if(self.use_depth_image):
                min_timestamps = min(min_timestamps, 
                                     self.depthl_queue.queue[-1].header.stamp.to_sec(),
                                     self.depthr_queue.queue[-1].header.stamp.to_sec(),
                                     self.depthf_queue.queue[-1].header.stamp.to_sec(),
                                    ) - 0.01

            while (self.imgl_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                self.imgl_queue.get()
            imgl = self.bridge.imgmsg_to_cv2(self.imgl_queue.get(), 'passthrough')

            while (self.imgr_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                self.imgr_queue.get()
            imgr = self.bridge.imgmsg_to_cv2(self.imgr_queue.get(), 'passthrough')

            while (self.imgf_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                self.imgf_queue.get()
            imgf = self.bridge.imgmsg_to_cv2(self.imgf_queue.get(), 'passthrough')
            
            if(self.use_depth_image):
                while (self.depthl_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                    self.depthl_queue.get()
                depthl = self.bridge.imgmsg_to_cv2(self.depthl_queue.get(), 'passthrough')

                while (self.depthr_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                    self.depthr_queue.get()
                depthr = self.bridge.imgmsg_to_cv2(self.depthr_queue.get(), 'passthrough')

                while (self.depthf_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                    self.depthf_queue.get()
                depthf = self.bridge.imgmsg_to_cv2(self.depthf_queue.get(), 'passthrough')
            


            while (self.masterl_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                self.masterl_queue.get()
            masterl = self.masterl_queue.get()

            while (self.masterr_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                self.masterr_queue.get()
            masterr = self.masterr_queue.get()

            while (self.puppetl_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                self.puppetl_queue.get()
            puppetl = self.puppetl_queue.get()

            while (self.puppetr_queue.queue[0].header.stamp.to_sec() < min_timestamps):
                self.puppetr_queue.get()
            puppetr = self.puppetl_queue.get()

            if(self.use_robot_base):
                while self.robot_base_deque[0].header.stamp.to_sec() < min_timestamps:
                    self.robot_base_deque.get()
                robot_base = self.robot_base_deque.get()

            # cv2.imshow("imgl", imgl)
            # cv2.imshow("imgr", imgr)
            # cv2.imshow("imgf", imgf)
            # cv2.waitKey(20)

            count += 1
           

            # 2.1 图像信息
            image_dict = dict()
            image_dict[self.args.camera_names[0]] = imgf
            image_dict[self.args.camera_names[1]] = imgl
            image_dict[self.args.camera_names[2]] = imgr

            # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
            obs = collections.OrderedDict()  # 有序的字典
            obs['images'] = image_dict
            obs['qpos'] = np.concatenate((np.array(puppetl.position), np.array(puppetr.position)), axis=0)
            obs['qvel'] = np.concatenate((np.array(puppetl.velocity), np.array(puppetr.velocity)), axis=0)
            obs['effort'] = np.concatenate((np.array(puppetl.effort), np.array(puppetr.effort)), axis=0)
            
            # 2.3 底盘数据
            if self.use_robot_base:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            else:
                obs['base_vel'] = [0.0, 0.0]

            # 2.4 深度图数据
            if(self.use_depth_image):
                depth_dict = dict()
                depth_dict[self.args.camera_names[0]] = depthf
                depth_dict[self.args.camera_names[1]] = depthl
                depth_dict[self.args.camera_names[2]] = depthr
                obs['depths'] = depth_dict

            # 第一帧 只包含first， fisrt只保存StepType.FIRST
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                
                timesteps.append(ts)
                print("frame %s cost_time: %s" % (count, round(1/(time.time() - time1), 6)))
                rate.sleep()
                continue

            # 时间步
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            # 主臂保存状态
            action = np.concatenate((np.array(masterl.position), np.array(masterr.position)), axis=0)
            actions.append(action)
            timesteps.append(ts)

            if rospy.is_shutdown():
                exit(-1)
            
            print("frame %s cost_time: %s" % (count, round((time.time() - time1), 6)))
            rate.sleep()
            
        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        save_data(self.args, timesteps, actions, self.dataset_path)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=500, required=False)

    # 相机名称话题
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    
    # 相机彩色图话题
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    # 相机深度图话题
    parser.add_argument('--use_depth_image', action='store_true', help='--use_depth_image', required=False)
    
    parser.add_argument('--depth_front_topic', action='store', type=str, help='depth_front_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--depth_left_topic', action='store', type=str, help='depth_left_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--depth_right_topic', action='store', type=str, help='depth_right_topic',
                        default='/camera_r/depth/image_raw', required=False)

    # 机械臂话题
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    # 底盘话题
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    
    # 发布频率
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    
    # 图像是否压缩
    parser.add_argument('--is_compress', action='store_true', help='is_compress', required=False)
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)   # 初始化ros节点，订阅消息

    t1 = threading.Thread(target=ros_operator.collet_data)
    t1.start()
    t1.join()
    

if __name__ == '__main__':
    main()

# python scripts/record_data.py  --dataset_dir ~/data0301 --max_timesteps 500 --is_compress --episode_idx 0
