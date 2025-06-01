#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import torch
import cv2
import numpy as np
import os
import pickle
import argparse
from einops import rearrange
import torch.nn.functional as F


from policy.act.utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy.act.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from robomimic.models.base_nets import SpatialSoftmax

import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import math
import threading
import matplotlib.pyplot as plt


import sys
sys.path.append("./")

# task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}
task_config = {
    'camera_names': ['front'],
    'tactile_camera_names': ['tactile']
}
# task_config = {'camera_names': ['cam_left_wrist','cam_high']}
inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None
############################dp
_original_spatial_softmax_forward = SpatialSoftmax.forward

def patched_spatial_softmax_forward(self, feature):
    # feature shape: [B, C, H, W]
    B, C, H, W = feature.shape
    target_h = self._in_h
    if H != target_h:
        if H < target_h:
            pad_h = target_h - H
            feature = F.pad(feature, (0, 0, 0, pad_h), mode='constant', value=0)
        else:
            feature = feature[:, :, :target_h, :]
    return _original_spatial_softmax_forward(self, feature)

SpatialSoftmax.forward = patched_spatial_softmax_forward
############################dp
def actions_interpolation(args, pre_action, actions, stats):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    result = [pre_action]
    post_action = post_process(actions[0])
    # print("pre_action:", pre_action[7:])
    # print("actions_interpolation1:", post_action[:, 7:])
    max_diff_index = 0
    max_diff = -1
    for i in range(post_action.shape[0]):
        diff = 0
        for j in range(pre_action.shape[0]):
            if j == 6 or j == 13:
                continue
            diff += math.fabs(pre_action[j] - post_action[i][j])
        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    for i in range(max_diff_index, post_action.shape[0]):
        step = max([math.floor(math.fabs(result[-1][j] - post_action[i][j])/steps[j]) for j in range(pre_action.shape[0])])
        inter = np.linspace(result[-1], post_action[i], step+2)
        result.extend(inter[1:])
    while len(result) < args.chunk_size+1:
        result.append(result[-1])
    result = np.array(result)[1:args.chunk_size+1]
    # print("actions_interpolation2:", result.shape, result[:, 7:])
    result = pre_process(result)
    result = result[np.newaxis, :]
    return result


def get_model_config(args):
    set_seed(1)
   
    if args.use_tactile_image:
        tactile_camera_names = task_config['tactile_camera_names']
    else:
        tactile_camera_names = []
        
    # fixed parameters
    if args.policy_class == 'ACT':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'pretrained_tactile_backbone': args.pretrained_tactile_backbone,
                         'tactile_backbone_path': args.tactile_backbone_path,
                         'pretrained_vision_backbone': args.pretrained_vision_backbone,
                         'vision_backbone_path': args.vision_backbone_path,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,
                         'camera_names': task_config['camera_names'],
                         'tactile_camera_names': tactile_camera_names,
                         'use_tactile_image': args.use_tactile_image,
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'kl_weight': args.kl_weight,
                         'hidden_dim': args.hidden_dim,
                         'dim_feedforward': args.dim_feedforward,
                         'enc_layers': args.enc_layers,
                         'dec_layers': args.dec_layers,
                         'nheads': args.nheads,
                         'dropout': args.dropout,
                         'pre_norm': args.pre_norm,
                         'state_dim': len(args.arm_names) * 7
                         }
    elif args.policy_class == 'CNNMLP':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': 1,
                         'camera_names': task_config['camera_names'],
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base
                         }

    elif args.policy_class == 'Diffusion':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'pretrained_tactile_backbone': args.pretrained_tactile_backbone,
                         'tactile_backbone_path': args.tactile_backbone_path,
                         'pretrained_vision_backbone': args.pretrained_vision_backbone,
                         'vision_backbone_path': args.vision_backbone_path,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,
                         'camera_names': task_config['camera_names'],
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'observation_horizon': args.observation_horizon,
                         'action_horizon': args.action_horizon,
                         'num_inference_timesteps': args.num_inference_timesteps,
                         'ema_power': args.ema_power,
                         'state_dim': len(args.arm_names) * 7
                         }
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': args.ckpt_dir,
        'ckpt_name': args.ckpt_name,
        'ckpt_stats_name': args.ckpt_stats_name,
        'episode_len': args.max_publish_step,
        'state_dim': len(args.arm_names) * 7,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'temporal_agg': args.temporal_agg,
        'camera_names': task_config['camera_names'],
        'tactile_camera_names': tactile_camera_names
    }
    return config

cross_attn_maps = []
def save_cross_attn(module, inp, output):
    attn_weights = output[1]
    cross_attn_maps.append(attn_weights.detach().cpu())
    
    
def make_policy(policy_class, policy_config):
    
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
        
        # TODO: vis_attn
        # for layer in policy.model.transformer.decoder.layers:
        #     layer.multihead_attn.register_forward_hook(save_cross_attn)
            
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
    
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def get_tactile_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['tactile_images'][cam_name], 'h w c -> c h w')
    
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def inference_process(args, config, ros_operator, policy, stats, t, pre_action):
    global inference_lock
    global inference_actions
    global inference_timestep
    print_flag = True
    pre_pos_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]
    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        

        # img_front = img_front[60:420,:,:]
        # img_left = img_left[60:420,:,:]
        # img_right = img_right[60:420,:,:]
        
        combined = cv2.vconcat([img_front, img_left, img_right])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite("combined_vertical.jpg", combined_bgr)
        # combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("combined_vertical.jpg", combined)

        obs = collections.OrderedDict()
        image_dict = dict()

        for cam_name in config['camera_names']:
            if cam_name == "front":
                image_dict[cam_name] = img_front
            elif cam_name == "cam_left_wrist":
                image_dict[cam_name] = img_left
            elif cam_name == "cam_right_wrist":
                image_dict[cam_name] = img_right

        tactile_dict = dict()
        for cam_name in config['tactile_camera_names']:
            if cam_name == "tactile":
                tactile_dict[cam_name] = img_right # TODO:

        obs['images'] = image_dict
        obs['tactile_images'] = tactile_dict

        if len(args.arm_names) == 2:
            obs['qpos'] = np.concatenate(
                (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            obs['qvel'] = np.concatenate(
                (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
            obs['effort'] = np.concatenate(
                (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        elif "left" in args.arm_names:
            obs['qpos'] = np.array(puppet_arm_left.position)
            obs['qvel'] = np.array(puppet_arm_left.velocity)
            obs['effort'] = np.array(puppet_arm_left.effort)
        elif "right" in args.arm_names:
            obs['qpos'] = np.array(puppet_arm_right.position)
            obs['qvel'] = np.array(puppet_arm_right.velocity)
            obs['effort'] = np.array(puppet_arm_right.effort)

            
        if args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]
        # qpos_numpy = np.array(obs['qpos'])

        qpos = pre_pos_process(obs['qpos'])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        curr_image = get_image(obs, config['camera_names'])
        curr_depth_image = None
        if args.use_depth_image:
            curr_depth_image = get_depth_image(obs, config['camera_names'])
        curr_tactile_image = None
        if args.use_tactile_image:
            curr_tactile_image = get_tactile_image(obs, config['tactile_camera_names'])
        start_time = time.time()
        all_actions = policy(curr_image, curr_depth_image, curr_tactile_image, qpos)
        end_time = time.time()
        print("model cost time: ", end_time -start_time)
        
        # TODO: visualize attention map
        # attn = cross_attn_maps[0][0, 0, 0]   # TODO: [seq_len] [N_vis + N_tac] ?
        # attn_mean = attn.mean(dim=(1, 2))
        # N_vis = 16 * 16                      # TODO: vision patch16x16
        # N_tac = 16 * 16                      # TODO: tactile patch16x16
        # H_vis, W_vis = 16, 16
        # H_img, W_img = 480, 640       
        
        # vis_score = attn_mean[0, :N_vis].sum().item()
        # tac_score = attn_mean[0, N_vis:N_vis + N_tac].sum().item()
        # vis_ratio = vis_score / (vis_score + tac_score)
        # tac_ratio = tac_score / (vis_score + tac_score)
               
        # vis_attn = attn[:N_vis].reshape(H_vis, W_vis).numpy()
        # vis_attn = cv2.resize(vis_attn, (W_img, H_img))  
        # img = obs['images']['front']  # [H,W,3]，uint8
        # img_overlay = (img * 0.6 + (plt.cm.jet(vis_attn / vis_attn.max())[:, :, :3] * 255 * 0.4)).astype(np.uint8)
        # cv2.imwrite(f"vis_attn_step_{t}.jpg", cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
        
        # tac_attn = attn[N_vis:].reshape(H_vis, W_vis).numpy()
        # tac_attn = cv2.resize(tac_attn, (W_img, H_img))  # 插值到原图大小
        # tac = obs['tactile_images']['tactile']  # [H,W,3]，uint8
        # tac_overlay = (tac * 0.6 + (plt.cm.jet(tac_attn / tac_attn.max())[:, :, :3] * 255 * 0.4)).astype(np.uint8)
        # cv2.imwrite(f"tac_attn_step_{t}.jpg", cv2.cvtColor(tac_overlay, cv2.COLOR_RGB2BGR))
        # cross_attn_maps.clear()
        
        inference_lock.acquire()
        
        inference_actions = all_actions.cpu().detach().numpy()
        if pre_action is None:
            pre_action = obs['qpos']
        # print("obs['qpos']:", obs['qpos'][7:])
        if args.use_actions_interpolation:
            inference_actions = actions_interpolation(args, pre_action, inference_actions, stats)
        inference_timestep = t
        inference_lock.release()
        break


def model_inference(args, config, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    set_seed(1000)

    print("Start to load model...")
    policy = make_policy(config['policy_class'], config['policy_config'])
    
    print("Start to load checkpoint...")
    ckpt_path = os.path.join(config['ckpt_dir'], args.task_name, config['ckpt_name'])
    print(f"Loading checkpoint: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not exist: {ckpt_path}")
        return False
        
    state_dict = torch.load(ckpt_path)
    print("Successfully load checkpoint.")
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
            continue
        if key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
            continue
        new_state_dict[key] = value
    loading_status = policy.deserialize(new_state_dict)
    if not loading_status:
        print("ckpt path not exist")
        return False

    policy.cuda()
    policy.eval()
    
    # TODO

    print("Start to load stats...")
    stats_path = os.path.join(config['ckpt_dir'], args.task_name, config['ckpt_stats_name'])
    if not os.path.exists(stats_path):
        print(f"Error: stats not exist: {stats_path}")
        return False
        
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    print("Successfully load stats.")

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']

    max_publish_step = config['episode_len']
    chunk_size = config['policy_config']['chunk_size']

    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.006]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.005]
    # right0 = [0.008146348000000001, 2.2958920600000003, -1.554365064, 0.079265536, -0.24855955600000001, -0.13806926, 0.002]
    left1 = [0, 0, 0, 0, 0, 0, 0.0] # puppet left
    # right1 = [0.0034713560000000005, 1.2548864720000001, -0.8694089600000001, -0.06735128400000001, 1.214782716, -0.08769098800000001, -0.00539] # puppet right
    right1 = [0, 0, 0, 0, 0, 0, 0.0]

    ################## DP open cabinet
    # right1 = [0.41106786000000006, 1.210334496, -0.784648564, 0.003645796, 1.135848616, 0.113490664, 0.02968]
 
    # ################## wly
    # right1 = [0.063653156, 0.413824012, -0.4410889840000001, -0.75000478, 0.17339336, 0.733467868, -0.00532]


    ros_operator.puppet_arm_publish_continuous(left0, right0)
    # input("Enter any key to continue :")
    # right0[6] = 0.00525
    # ros_operator.puppet_arm_publish_continuous(left0, right0)
    ########
    input("Enter any key to continue :")
    # ros_operator.puppet_arm_publish(left1, right1)
    # action = None
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            t = 0
            max_t = 0
            rate = rospy.Rate(args.publish_rate)
            if config['temporal_agg']:
                all_time_actions = np.zeros([max_publish_step, max_publish_step + chunk_size, config['state_dim']])
            while t < max_publish_step and not rospy.is_shutdown():
                # start_time = time.time()
                # query policy
                if config['policy_class'] == "ACT":
                    if t >= max_t:
                        # pre_action = action
                        pre_action = None
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(args, config, ros_operator,
                                                                  policy, stats, t, pre_action))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + args.pos_lookahead_step
                            if config['temporal_agg']:
                                all_time_actions[[t], t:t + chunk_size] = all_actions
                        inference_lock.release()
                    if config['temporal_agg']:
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.1
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                elif config['policy_class'] == "Diffusion":
                    if t >= max_t:
                        # pre_action = action
                        pre_action = None
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(args, config, ros_operator,
                                                                  policy, stats, t, pre_action))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        # chunk_size = 16
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + args.pos_lookahead_step
                            if config['temporal_agg']:
                                all_time_actions[[t], t:t + chunk_size] = all_actions#[:, 16:]
                        inference_lock.release()
                    if config['temporal_agg']:
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.3
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                else:
                    raise NotImplementedError
                action = post_process(raw_action[0])
                print(action)
                if len(args.arm_names) == 2:
                    left_action = action[:7]
                    right_action = action[7:14]
                elif "left" in args.arm_names:
                    left_action = action[:7]
                    right_action = torch.zeros(7)
                elif "right" in args.arm_names:
                    left_action = torch.zeros(7)
                    right_action = action[:7]
                    right_action[6] -= 0.05
                ros_operator.puppet_arm_publish(left_action, right_action)# puppet_arm_publish_continuous_thread
                if args.use_robot_base:
                    vel_action = action[14:16]
                    ros_operator.robot_base_publish(vel_action)
                t += 1
                # end_time = time.time()
                # print("publish: ", t)
                # print("time:", end_time - start_time)
                # print("left_action:", left_action)
                # print("right_action:", right_action)
                rate.sleep()


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now() 
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        
        # 配置三个相机的话题
        camera_configs = {
            'front': {
                'device': '/dev/video0',
                'topic': '/front_camera/image_raw',
                'camera_name': 'front'
            },
            'left': {
                'device': '/dev/video2',
                'topic': '/left_camera/image_raw',
                'camera_name': 'cam_left_wrist'
            },
            'right': {
                'device': '/dev/video4',
                'topic': '/right_camera/image_raw',
                'camera_name': 'tactile'
            }
        }
        
        for position, config in camera_configs.items():
            if not os.path.exists(config['device']):
                rospy.logwarn(f"Camera device {config['device']} for {position} camera not found!")
                continue
                
        home_dir = os.environ['HOME']
        for position, config in camera_configs.items():
            if not os.path.exists(config['device']):
                continue
                
            camera_info_dir = f"{home_dir}/.ros/camera_info"
            os.makedirs(camera_info_dir, exist_ok=True)
            import subprocess

            cmd = [
                "rosrun",
                "usb_cam",
                "usb_cam_node",
                f"_video_device:={config['device']}",
                f"__name:={position}_camera",
                f"_camera_name:={position}_camera",
                f"_camera_frame_id:={position}_camera",
                "_image_width:=640",
                "_image_height:=480",
                "_pixel_format:=yuyv",
                "_io_method:=mmap",
                "_framerate:=30",
                "_brightness:=100",
                "_autofocus:=false",
                "_focus:=51",
                f"_camera_info_url:=file://{home_dir}/.ros/camera_info/{position}_camera.yaml"
            ]
            
            try:
                rospy.loginfo(f"Starting {position} camera...")
                process = subprocess.Popen(cmd)
                rospy.sleep(2.0)
                
                timeout = 10.0
                try:
                    rospy.wait_for_message(config['topic'], Image, timeout=timeout)
                    rospy.loginfo(f"{position} camera is ready!")
                except rospy.ROSException:
                    rospy.logerr(f"Timeout waiting for {position} camera topic!")
                    process.terminate()
                    continue
                    
            except Exception as e:
                rospy.logerr(f"Failed to start {position} camera: {str(e)}")
                continue
        
        for position, config in camera_configs.items():
            if position == 'front':
                rospy.Subscriber(config['topic'], Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
            elif position == 'left':
                rospy.Subscriber(config['topic'], Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
            elif position == 'right':
                rospy.Subscriber(config['topic'], Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
            
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)  # policy_epoch_6000_seed_0.ckpt / policy_best.ckpt
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='Diffusion', required=False)  # ACT / Diffusion
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--pretrained_tactile_backbone', default=False, type=bool, required=False)
    parser.add_argument('--tactile_backbone_path', default=None, type=str, required=False)
    parser.add_argument('--pretrained_vision_backbone', default=False, type=bool, required=False)
    parser.add_argument('--vision_backbone_path', default=None, type=str, required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=7, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)
    parser.add_argument('--arm_names', action='store', type=int, help='inference which arm data',  
                        # default=['left', 'right'], required=False)
                        default=['right'], required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/front_camera/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/left_camera/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/right_camera/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=40, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=12, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_tactile_image', action='store', type=bool, help='use_tactile_image',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=15, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_model_config(args)
    model_inference(args, config, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()
# python act/inference.py --ckpt_dir ~/train0314/
