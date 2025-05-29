# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model_tac, build_CNNMLP_model_tac, build_diffusion_model_tac

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)  # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # will be overridden
    parser.add_argument('--backbone', default='resnet18', type=str,  # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2',
                        default='l1', required=False)
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=100, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image', default=False, required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base', default=False, required=False)

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)

    return parser


def build_diffusion_model_and_optimizer(args_override):
    # 加载参数
    # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    #
    # # 加载的模型config 字典形式
    # for k, v in args_override.items():
    #     # 设置 args的k键值对应的value为v
    #     setattr(args, k, v)
    args = argparse.Namespace(**args_override)
    model = build_diffusion_model_tac(args)
    model.cuda()

    # param_dicts = [
    #     {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    # ]

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            # "lr": args.lr_backbone 
        },
    ]

    # 优化器
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_ACT_model_and_optimizer(args_override):
    
    # 加载参数
    # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    #
    # # 加载的模型config 字典形式
    # for k, v in args_override.items():
    #     # 设置 args的k键值对应的value为v
    #     setattr(args, k, v)
    args = argparse.Namespace(**args_override)
    model = build_ACT_model_tac(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    # 优化器
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    #
    # for k, v in args_override.items():
    #     setattr(args, k, v)
    args = argparse.Namespace(**args_override)
    model = build_CNNMLP_model_tac(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

