import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm

from policy.act.utils import load_data 
from policy.act.utils import compute_dict_mean, set_seed, detach_dict
from policy.act.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from robomimic.models.base_nets import SpatialSoftmax

import sys
sys.path.append("./")

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

def train(args):
    set_seed(1)
    DATA_DIR = os.path.expanduser(args.dataset_dir) 
    
    TASK_CONFIGS = {
        args.task_name: {
            'dataset_dir': os.path.join(DATA_DIR, args.task_name),
            'camera_names': ['front'],
            'tactile_camera_names': ['tactile'],
            'num_episodes': args.num_episodes
        }
    }

    task_config = TASK_CONFIGS[args.task_name]
    
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']
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
                         'chunk_size': args.chunk_size,     # chunk_size
                         'camera_names': camera_names,
                         'use_depth_image': args.use_depth_image,
                         'use_tactile_image': args.use_tactile_image,
                         'tactile_camera_names': tactile_camera_names,
                         'use_robot_base': args.use_robot_base,
                         'kl_weight': args.kl_weight,        # kl
                         'hidden_dim': args.hidden_dim,      # Hidden dim
                         'dim_feedforward': args.dim_feedforward,
                         'enc_layers': args.enc_layers,
                         'dec_layers': args.dec_layers,
                         'nheads': args.nheads,
                         'dropout': args.dropout,
                         'pre_norm': args.pre_norm,
                         'state_dim': args.state_dim, 
                         'vis_blind': args.vis_blind, 
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
                         'chunk_size': 1,     # 查询
                         'camera_names': camera_names,
                         'use_depth_image': args.use_depth_image,
                         'use_tactile_image': args.use_tactile_image,
                         'tactile_camera_names': tactile_camera_names,
                         'use_robot_base': args.use_robot_base,
                         'hidden_dim': args.hidden_dim,
                         'state_dim': args.state_dim
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
                         'chunk_size': args.chunk_size,     # 查询
                         'camera_names': camera_names,
                         'use_depth_image': args.use_depth_image,
                         'use_tactile_image': args.use_tactile_image,
                         'tactile_camera_names': tactile_camera_names,
                         'use_robot_base': args.use_robot_base,
                         'observation_horizon': args.observation_horizon,
                         'action_horizon': args.action_horizon,
                         'num_inference_timesteps': args.num_inference_timesteps,
                         'ema_power': args.ema_power,
                         'hidden_dim': args.hidden_dim,
                         'state_dim': args.state_dim,
                         'vis_blind': args.vis_blind, 
                         }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': args.num_epochs,
        'ckpt_dir': args.ckpt_dir,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'seed': args.seed,
        'pretrain_ckpt_dir': args.pretrain_ckpt,
        'save_every_epochs': args.save_every_epochs,
    }

    print(config)

    # data Preprocess
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, args.arm_delay_time,
                                                           args.use_depth_image, args.use_tactile_image, args.use_robot_base, camera_names, tactile_camera_names,
                                                           args.batch_size, args.batch_size, args.backbone)

    # save dataset stats
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    stats_path = os.path.join(args.ckpt_dir, args.ckpt_stats_name)
    print("saving stats...")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_process(train_dataloader, val_dataloader, config, stats)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    best_ckpt_name=f'policy_best_epoch_{best_epoch}.ckpt'
    ckpt_path = os.path.join(args.ckpt_dir, best_ckpt_name)
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config, pretrain_ckpt_dir):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            state_dict = torch.load(pretrain_ckpt_dir)
            loading_status = policy.deserialize(state_dict)
            if not loading_status:
                print("ckpt path not exist")

    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.deserialize(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.deserialize(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(policy_config, data, policy):
    image_data, image_depth_data, tactile_image_data, qpos_data, action_data, action_is_pad = data

    if policy_config['vis_blind']:
        image_data = torch.zeros_like(image_data)

    (image_data, qpos_data, action_data, action_is_pad) = (image_data.cuda(), qpos_data.cuda(),
                                                           action_data.cuda(), action_is_pad.cuda())
    if policy_config['use_depth_image']:
        image_depth_data = image_depth_data.cuda()
    else:
        image_depth_data = None
        
    if policy_config['use_tactile_image']:
        tactile_image_data = tactile_image_data.cuda()
    else:
        tactile_image_data = None
        
    return policy(image_data, image_depth_data, tactile_image_data, qpos_data, action_data, action_is_pad)


def train_process(train_dataloader, val_dataloader, config, stats):
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    pretrain_ckpt_dir = config['pretrain_ckpt_dir']
    set_seed(seed)

    policy = make_policy(policy_class, policy_config, pretrain_ckpt_dir)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict, result = forward_pass(policy_config, data, policy)
                # print("result:", post_process(result.cpu().detach().numpy())[0, :, 7:])
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.serialize()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict, result = forward_pass(policy_config, data, policy)
            # print("result:", post_process(result.cpu().detach().numpy())[0, :, 7:])
            # backward
            loss = forward_dict['loss']
            if policy_class == "Diffusion":
                policy.model.ema.update()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % config["save_every_epochs"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', default='./dataset', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=True)
   
    parser.add_argument('--pretrain_ckpt', action='store', type=str, help='pretrain_ckpt', default='', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize, CNNMLP, ACT, Diffusion', default='Diffusion', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=32, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=3000, required=False)

    parser.add_argument('--lr', action='store', type=float, help='lr', default=4e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--save_every_epochs', type=float, help='save every epochs', default=500, required=False)
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

    # parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=7, required=False)
    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=7, required=False)   # 7 - single; arm 14 - dual arm
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=4e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=8, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg',  action='store', type=bool, help='temporal_agg', default=True, required=False)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=15, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)

    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base', default=False, required=False)

    parser.add_argument('--arm_delay_time', action='store', type=int, help='arm_delay_time', default=0, required=False)

    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image', default=False, required=False)
    parser.add_argument('--use_tactile_image', action='store', type=bool, help='use_tactile_image', default=False, required=False)
    parser.add_argument('--vis_blind', action='store', type=bool, default=False, required=False)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    train(args)

if __name__ == '__main__':
    main()
# python act/train.py --dataset_dir ~/data --pretrain_ckpt policy_best.ckpt --ckpt_dir ~/train_dir/ --num_episodes 20 --batch_size 10 --num_epochs 2000 
