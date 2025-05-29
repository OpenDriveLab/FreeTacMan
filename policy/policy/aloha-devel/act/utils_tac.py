import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import random
import IPython
import math

import torchvision.transforms as transforms
from torchvision.transforms import Normalize
from PIL import Image, ImageOps

e = IPython.embed
import cv2

image_transform = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x.astype('uint8'))),
    transforms.Lambda(lambda x: ImageOps.pad(x, (max(x.size), max(x.size)))),
    transforms.Resize(224),
    transforms.ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
tactile_transform = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x.astype('uint8'))),
    transforms.Lambda(lambda x: ImageOps.pad(x, (max(x.size), max(x.size)))),
    transforms.Resize(224),
    transforms.ToTensor(),
    Normalize(mean=[0.6234687568296101, 0.4473433504166392, 0.5247292468417972], std=[0.16458128838663075, 0.17134032216538317, 0.21147269102429556]),
])

class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, tactile_camera_names, norm_stats, arm_delay_time,
                 use_depth_image, use_tactile_image, use_robot_base, backbone):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids  # 1000
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.tactile_camera_names = tactile_camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_depth_image = use_depth_image
        self.use_tactile_image = use_tactile_image
        self.arm_delay_time = arm_delay_time
        self.use_robot_base = use_robot_base
        self.backbone = backbone
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        # 读取数据
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            try:
                is_compress = root.attrs['compress']
            except:
                is_compress = False
            original_action_shape = root['/action'].shape
            max_action_len = original_action_shape[0]  # max_episode
            if self.use_robot_base:
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 2)

            start_ts = np.random.choice(max_action_len)  # 随机抽取一个索引
            actions = root['/observations/qpos'][1:]
            actions = np.append(actions, actions[-1][np.newaxis, :], axis=0)
            qpos = root['/observations/qpos'][start_ts]
            if self.use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][start_ts]), axis=0)
            image_dict = dict()
            image_depth_dict = dict()
            tactile_image_dict = dict()
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                    # print(image_dict[cam_name].shape)
                    # exit(-1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                if self.use_depth_image:
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_ts]
            
            if self.use_tactile_image:
                for tactile_name in self.tactile_camera_names:
                    if is_compress:
                        decoded_image = root[f'/observations/images/{tactile_name}'][start_ts]
                        tactile_image_dict[tactile_name] = cv2.imdecode(decoded_image, 1)
                        # print(image_dict[cam_name].shape)
                        # exit(-1)
                    else:
                        tactile_image_dict[tactile_name] = root[f'/observations/images/{tactile_name}'][start_ts]

            start_action = min(start_ts, max_action_len - 1)
            index = max(0, start_action - self.arm_delay_time)
            action = actions[index:]  # hack, to make timesteps more aligned
            if self.use_robot_base:
                action = np.concatenate((action, root['/base_action'][index:]), axis=1)
            action_len = max_action_len - index  # hack, to make timesteps more aligned

        self.is_sim = is_sim

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        action_is_pad = np.zeros(max_action_len)
        action_is_pad[action_len:] = 1
        
        image_data = []
        if self.backbone == 'vit':
            for cam_name in self.camera_names:
                image_data.append(image_transform(image_dict[cam_name]))
            image_data = torch.stack(image_data, axis=0)
            
            if self.use_tactile_image:
                tactile_image_data = []
                for tactile_name in self.tactile_camera_names:
                    tactile_image_data.append(tactile_transform(tactile_image_dict[tactile_name]))
                tactile_image_data = torch.stack(tactile_image_data, axis=0)
            else:
                tactile_image_data = np.zeros(1, dtype=np.float32)
        else:
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)
            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            image_data = torch.einsum('k h w c -> k c h w', image_data)
            image_data = image_data / 255.0
            
            if self.use_tactile_image:
                all_tactile_images = []
                for tactile_name in self.tactile_camera_names:
                    all_tactile_images.append(tactile_image_dict[tactile_name])
                all_tactile_images = np.stack(all_tactile_images, axis=0)
                # construct observations
                tactile_image_data = torch.from_numpy(all_tactile_images) / 255.0
                tactile_image_data = torch.einsum('k h w c -> k c h w', tactile_image_data)
            else:
                tactile_image_data = np.zeros(1, dtype=np.float32)
        

        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = []
            for cam_name in self.camera_names:
                all_cam_images_depth.append(image_depth_dict[cam_name])
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            # construct observations
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            # image_depth_data = torch.einsum('k h w c -> k c h w', image_depth_data)
            image_depth_data = image_depth_data / 255.0
        
        qpos_data = torch.from_numpy(qpos).float()
        mean_tensor = torch.tensor(self.norm_stats["qpos_mean"], device=qpos_data.device, dtype=qpos_data.dtype)
        std_tensor = torch.tensor(self.norm_stats["qpos_std"], device=qpos_data.device, dtype=qpos_data.dtype)
        qpos_data = (qpos_data - mean_tensor) / std_tensor
        action_data = torch.from_numpy(padded_action).float()
        action_is_pad = torch.from_numpy(action_is_pad).bool()
        action_data = (action_data - mean_tensor) / std_tensor
        # qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        # action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # torch.set_printoptions(precision=10, sci_mode=False)
        # torch.set_printoptions(threshold=float('inf'))
        # print("qpos_data:", qpos_data[7:])
        # print("action_data:", action_data[:, 7:])

        return image_data, image_depth_data, tactile_image_data, qpos_data, action_data, action_is_pad


def get_norm_stats(dataset_dir, num_episodes, use_robot_base):
    all_qpos_data = []
    all_action_data = []

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            # qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            if use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][()]), axis=1)
                action = np.concatenate((action, root['/base_action'][()]), axis=1)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data
    # all_qpos_data = torch.cat(all_qpos_data, dim=0)
    # all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    # action_mean = all_action_data.mean(dim=0, keepdim=True)
    # action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    action_max = all_action_data.max(dim=0, keepdim=True)[0][0].max(dim=0)[0]
    action_min = all_action_data.min(dim=0, keepdim=True)[0][0].min(dim=0)[0]
    # action_max = all_action_data.max(dim=0, keepdim=True)[0]
    # action_min = all_action_data.min(dim=0, keepdim=True)[0]

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    # qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    # qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(), 
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos, 
        "action_max":action_max, 
        "action_min":action_min
    }
    # stats = {
    #     "action_mean": action_mean.numpy().squeeze().astype(np.float32), 
    #     "action_std": action_std.numpy().squeeze().astype(np.float32),
    #     "qpos_mean": qpos_mean.numpy().squeeze().astype(np.float32), 
    #     "qpos_std": qpos_std.numpy().squeeze().astype(np.float32),
    #     "example_qpos": qpos.astype(np.float32), 
    #     "action_max":action_max.numpy().squeeze().astype(np.float32), 
    #     "action_min":action_min.numpy().squeeze().astype(np.float32)
    # }

    return stats


def load_data(dataset_dir, num_episodes, arm_delay_time, use_depth_image, use_tactile_image,
              use_robot_base, camera_names, tactile_camera_names, batch_size_train, batch_size_val, backbone):
    print(f'\nData from: {dataset_dir}\n')

    # obtain train test split
    train_ratio = 0.99  # 数据集比例
    shuffled_indices = np.random.permutation(num_episodes)  # 打乱

    train_indices = shuffled_indices[:math.ceil(train_ratio * num_episodes)]
    val_indices = shuffled_indices[math.floor(train_ratio * num_episodes):]

    print(train_indices, val_indices)


    # obtain normalization stats for qpos and action  返回均值和方差
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_robot_base)

    # construct dataset and dataloader 归一化处理  结构化处理数据
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, tactile_camera_names, norm_stats, arm_delay_time,
                                    use_depth_image, use_tactile_image, use_robot_base, backbone)

    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, tactile_camera_names,norm_stats, arm_delay_time,
                                  use_depth_image, use_tactile_image, use_robot_base, backbone)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1,
                                prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])

    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
