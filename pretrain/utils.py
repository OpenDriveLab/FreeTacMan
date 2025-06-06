import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import cv2
from torchvision import transforms
from multiprocessing import Pool
from tqdm import tqdm
import json

import IPython
e = IPython.embed

class NormalizeActionQpos:
    def __init__(self, norm_stats):
        # since the values of the qpos and action are tied together
        # (current position, goal position), we normalize them together
        self.mean = (norm_stats["qpos_mean"] + norm_stats["action_mean"])/2
        self.std = (norm_stats["qpos_std"] + norm_stats["action_std"])/2
    
    def __call__(self, qpos, action):
        qpos = (qpos - self.mean) / self.std
        action = (action - self.mean) / self.std
        return qpos, action
    
    def normalize_qpos(self, qpos):
        return (qpos - self.mean) / self.std
    
    def normalize_action(self, action):
        return (action - self.mean) / self.std
    
    def unnormalize_qpos(self, qpos):
        return qpos * self.std + self.mean
    
    def unnormalize_action(self, action):
        return action * self.std + self.mean
    
    def unnormalize(self, qpos, action):
        new_qpos = qpos * self.std + self.mean
        new_action = action * self.std + self.mean
        return new_qpos, new_action
    
class NormalizeDeltaActionQpos:
    def __init__(self, norm_stats):
        self.qpos_mean = norm_stats["qpos_mean"]
        self.qpos_std = norm_stats["qpos_std"]
        self.delta_mean = norm_stats["delta_mean"]
        self.delta_std = norm_stats["delta_std"]
    
    def __call__(self, qpos, action):
        delta = action - qpos
        delta[:, 3] = action[:, 3] # keep the gripper action the same

        # normalize the qpos and delta
        qpos = (qpos - self.qpos_mean) / self.qpos_std
        delta = (delta - self.delta_mean) / self.delta_std
        return qpos, delta
    
    def normalize_qpos(self, qpos):
        return (qpos - self.qpos_mean) / self.qpos_std
    
    def unnormalize_qpos(self, normalized_qpos):
        return normalized_qpos * self.qpos_std + self.qpos_mean
    
    def unnormalize_delta(self, normalized_delta):
        return normalized_delta * self.delta_std + self.delta_mean
    
    def unnormalize(self, normalized_qpos, normalized_delta):
        qpos = normalized_qpos * self.qpos_std + self.qpos_mean
        delta = normalized_delta * self.delta_std + self.delta_mean
        action = qpos + delta
        action[:, 3] = delta[:, 3] # keep the gripper action the same
        return qpos, action

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size, image_size = None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names

        self.action_qpos_normalize = NormalizeActionQpos(norm_stats)

        self.is_sim = None
        self.image_size = image_size # image size in (H, W)
        if "tactile_mean" in norm_stats: # if tactile data exists
            self.tactile_mean = norm_stats["tactile_mean"]
            self.tactile_std = norm_stats["tactile_std"]

        self.chunk_size = chunk_size

        # image normalization for resnet. 
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.__getitem__(0) # initialize self.is_sim, self.image_size


        # image = normalize(image)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts] # unused

            if self.image_size is None: # if image size is not specified, use the saved image size
                self.image_size = (root.attrs['image_height'], root.attrs['image_width'])

            all_cam_images = []

            for cam_name in self.camera_names:
                # seperate processing for tactile:
                if cam_name == 'tactile':
                    size = (root.attrs['tactile_height'], root.attrs['tactile_width'])
                    tactile_data = root['observations/tactile/depth_strain_image'][start_ts]
                    # tactile_data = cv2.resize(tactile_data, (self.image_size[1], self.image_size[0]))
                    # adjust tactile data using the mean and std
                    tactile_data = (tactile_data - self.tactile_mean) / self.tactile_std
                    tactile_data = torch.tensor(tactile_data, dtype=torch.float32)
                    tactile_data = torch.einsum('h w c -> c h w', tactile_data) # change to c h w
                    all_cam_images.append(tactile_data)
                
                elif cam_name == "blank":
                    image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)
                    image = torch.tensor(image, dtype=torch.float32)
                    image = torch.einsum('h w c -> c h w', image)
                    all_cam_images.append(image)

                else:
                    image = root[f'/observations/images/{cam_name}'][start_ts]
                    # resize image
                    if self.image_size != image.shape[:2]:
                        print('reshaping image')
                        raise ValueError('Image size does not match the specified image size')
                        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

                    # normalize image
                    image = torch.tensor(image, dtype=torch.float32)/255.0
                    image = torch.einsum('h w c -> c h w', image) # change to c h w
                    image = self.image_normalize(image)
                    all_cam_images.append(image)



            # get all actions after and including start_ts, with the max length of chunk_size
            action_len = min(episode_len - start_ts, self.chunk_size) 
            action = root['/action'][start_ts:start_ts + action_len]

        # normalize action and qpos
        qpos, action = self.action_qpos_normalize(qpos=qpos, action=action)

        self.is_sim = is_sim
        padded_action = np.zeros([self.chunk_size, action.shape[1]], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return all_cam_images, qpos_data, action_data, is_pad
    
# create a new class for the delta action - different normalization
class EpisodicDatasetDelta(EpisodicDataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size, image_size = None):
        super(EpisodicDatasetDelta, self).__init__(episode_ids, dataset_dir, camera_names, norm_stats, chunk_size, image_size)
        self.action_qpos_normalize = NormalizeDeltaActionQpos(norm_stats)


def process_episode(args):
    episode_idx, dataset_dir = args
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    tactile_mean = []
    tactile_std = []

    with h5py.File(dataset_path, 'r') as root:
        if 'observations/images/tactile' in root:
            tactile_data = root['observations/images/tactile'][()]
            tactile_mean = np.mean(tactile_data, axis=(1, 2))
            tactile_std = np.std(tactile_data, axis=(1, 2))

    return tactile_mean, tactile_std

def tactile_norm_stats(dataset_dir, num_episodes, use_existing = False) -> tuple:
    
    if use_existing and os.path.exists('tactile_data.json'):
        with open('tactile_data.json', 'r') as f:
            tactile_data = json.load(f)
            assert 'tactile_mean' in tactile_data and 'tactile_std' in tactile_data, "tactile data must exist"
            return np.array(tactile_data['tactile_mean']), np.array(tactile_data['tactile_std'])
        
    elif not os.path.exists('tactile_data.json'):
        with open('tactile_data.json', 'w') as f:
            json.dump({}, f)
        
    tactile_means = [] 
    tactile_stds = []

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_episode, [(i, dataset_dir) for i in range(num_episodes)]), total=num_episodes, desc="Get tactile Stats"))
    tactile_means, tactile_stds = zip(*results)
    tactile_means, tactile_stds = np.concatenate(tactile_means), np.concatenate(tactile_stds)
    # for episode_idx in tqdm(range(num_episodes), desc="Get tactile Stats"):
    #     dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    #     with h5py.File(dataset_path, 'r') as root:

    #         # if tactile data exists, get the average
    #         if 'observations/images/tactile' in root:
    #             use_tactile = True
    #             tactile_data = root['observations/images/tactile'][()]
    #             tactile_mean = np.mean(tactile_data, axis=(1, 2))
    #             tactile_std = np.std(tactile_data, axis=(1, 2))   
    #             tactile_means.extend(tactile_mean)
    #             tactile_stds.extend(tactile_std)
    
    tactile_mean = np.mean(np.array(tactile_means), axis=0)
    tactile_std = np.mean(np.array(tactile_stds), axis=0) 
    tactile_std = np.clip(tactile_std, 1e-2, np.inf)
    
    # TODO: normalize the tactile data to [0, 1]
    tactile_mean /= 255.0
    tactile_std /= 255.0
    
    
    with open('tactile_data.json', 'w') as f:
        json.dump({'tactile_mean': tactile_mean.tolist(), 'tactile_std': tactile_std.tolist()}, f)

    return tactile_mean, tactile_std


def get_norm_stats(dataset_dir, num_episodes, use_existing = False):
    qpos_data_list = []
    action_data_list = []
    use_tactile = False

    for episode_idx in tqdm(range(num_episodes), desc="Get Norm Stats"):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            #qvel = root['/observations/qvel'][()] # unused
            action = root['/action'][()]

            # if tactile data exists, get the average
            if 'observations/images/tactile' in root:
                use_tactile = True

        qpos_data_list.append(qpos)
        action_data_list.append(action)
    all_qpos_data = np.concatenate(qpos_data_list, axis=0)
    all_action_data = np.concatenate(action_data_list, axis=0)

    # get mean of the action data
    action_mean = all_action_data.mean(axis=0)
    action_std = all_action_data.std(axis=0)
    action_std = np.clip(action_std, 1e-2, np.inf) # clipping

    # get mean of the qpos data
    qpos_mean = all_qpos_data.mean(axis=0)
    qpos_std = all_qpos_data.std(axis=0)
    qpos_std = np.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean, "action_std": action_std,
            "qpos_mean": qpos_mean, "qpos_std": qpos_std}

    # check to see if norm stats already exists
    if use_tactile:
        tactile_mean, tactile_std = tactile_norm_stats(dataset_dir, num_episodes, use_existing)

        stats["tactile_mean"] = tactile_mean
        stats["tactile_std"] = tactile_std

    # if chunk_size != 0:
    #     # calculate the mean and std of the delta (position) actions:
    #     all_deltas = []
    #     for episode in range(num_episodes):
    #         len_episode = len(action_data_list[episode])
    #         for t in range(len_episode - chunk_size):
    #             deltas = action_data_list[episode][t:t+chunk_size, 0:3] - qpos_data_list[episode][t][0:3]
    #             all_deltas.append(deltas)

    #     all_deltas = np.concatenate(all_deltas, axis=0)
    #     delta_mean = all_deltas.mean(axis=0)
    #     delta_std = all_deltas.std(axis=0)
    #     delta_std = np.clip(delta_std, 1e-3, np.inf)
    #     stats["delta_mean"] = np.concatenate([delta_mean, [action_mean[3]]])
    #     stats["delta_std"] = np.concatenate([delta_std, [action_std[3]]])
            
    return stats


## helper functions, same as ACT

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
