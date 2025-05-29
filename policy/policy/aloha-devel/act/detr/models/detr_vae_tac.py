# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone, DepthNet
from .position_encoding import build_position_encoding
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np
from collections import OrderedDict
import sys
# sys.path.append("/home/agilex/cobot_magic/aloha-devel")
sys.path.append("/cpfs01/user/yuchecheng/FastUMI_Data/cobot_magic/aloha-devel")
from act.TactileACT.clip_pretraining import modified_resnet18
from act.TactileACT.clip_pretraining_vit import load_vit_encoder
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from ema_pytorch import EMA
from transformers.modeling_outputs import BaseModelOutputWithPooling

import IPython
e = IPython.embed
import math

class MyJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        output = self[0](tensor_list)

        if isinstance(output, BaseModelOutputWithPooling):
            # last_hidden_state: [B, num_patches+1, C]
            hs = output.last_hidden_state
            patch_tokens = hs[:, 1:, :]           # [B, N, C]
            B, N, C = patch_tokens.shape
            H = W = int(math.sqrt(N))
            # reshape to [B, C, H, W]
            xs = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        elif isinstance(output, (tuple, list)):
            xs = output[0]
        else:
            xs = output

        out = [xs]
        pos = [self[1](xs).to(xs.dtype)]
        return out, pos


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def load_backbone(model):
    if model == 'vit':
        return load_vit_encoder()
    else:
        return modified_resnet18()

class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, depth_backbones, tactile_backbones, transformer, encoder, state_dim, num_queries, camera_names, tactile_camera_names, kl_weight):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.tactile_camera_names = tactile_camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.kl_weight = kl_weight

        if backbones is not None:
            # print("backbones[0]", backbones[0])
            if depth_backbones is not None:
                self.depth_backbones = nn.ModuleList(depth_backbones)
                self.input_proj = nn.Conv2d(backbones[0].num_channels + depth_backbones[0].num_channels, hidden_dim, kernel_size=1)
            else:
                self.depth_backbones = None
                self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
                
            if tactile_backbones is not None:
                self.tactile_backbones = nn.ModuleList(tactile_backbones)
                self.input_proj_tactile = nn.Conv2d(tactile_backbones[0].num_channels, hidden_dim, kernel_size=1)
            else:
                self.tactile_backbones = None
                
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding

        self.latent_pos = nn.Embedding(1, hidden_dim)
        self.robot_state_pos = nn.Embedding(1, hidden_dim)

        if kl_weight != 0:
            self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)  # project action to embedding
            self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)  # project hidden state to latent std, var
            self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))  # [CLS], qpos, a_seq

    def forward(self, image, depth_image, tactile_image, robot_state, actions=None, action_is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None                                   
        actions: batch, seq, action_dim                    
        """

        # print("forward: ", qpos.shape, image.shape, env_state, actions.shape, action_is_pad.shape)

        is_training = actions is not None  # train or val
        bs, _ = robot_state.shape

        # Obtain latent z from action sequence
        if is_training and self.kl_weight != 0:  # hidden_dim输入参数是512
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            robot_state_embed = self.encoder_joint_proj(robot_state)  # (bs, hidden_dim)
            robot_state_embed = torch.unsqueeze(robot_state_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], axis=1)  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            cls_joint_is_pad = torch.full((bs, 2), False).to(robot_state.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, action_is_pad], axis=1)  # (bs, seq+1)

            # obtain position embedding  合并位置编码
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            
            # 线性层  hidden_dim扩大到64
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(robot_state.device)
            latent_input = self.latent_out_proj(latent_sample)
        # Image observation features and position embeddings
        all_cam_features = []
        all_tactile_features = []
        all_tactile_pos = []
        all_cam_depth_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            # features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
            features, src_pos = self.backbones[cam_id](image[:, cam_id]) # HARDCODED
            # image_test = image[:, cam_id][:, 0].unsqueeze(dim=1)
            # print("depth_encoder:", self.depth_encoder(image_test))
            features = features[0]  # take the last layer feature
            src_pos = src_pos[0]
            if self.depth_backbones is not None and depth_image is not None:
                features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                all_cam_features.append(self.input_proj(torch.cat([features, features_depth], axis=1)))
            else:
                all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(src_pos)
            
        # TODO:
        if self.tactile_backbones is not None and tactile_image is not None:
            for tac_id, tac_name in enumerate(self.tactile_camera_names):
                tac_features, tac_src_pos = self.tactile_backbones[tac_id](tactile_image[:, tac_id])
                tac_features = tac_features[0]
                tac_src_pos = tac_src_pos[0]
                all_tactile_features.append(self.input_proj_tactile(tac_features))
                all_tactile_pos.append(tac_src_pos)
            all_cam_features = all_cam_features + all_tactile_features
            all_cam_pos = all_cam_pos + all_tactile_pos
                
        # proprioception features
        robot_state_input = self.input_proj_robot_state(robot_state)
        robot_state_input = torch.unsqueeze(robot_state_input, axis=0)
        latent_input = torch.unsqueeze(latent_input, axis=0)
        # fold camera dimension into width dimension
        # TODO: cross attention
        src = torch.cat(all_cam_features, axis=3)
        src_pos = torch.cat(all_cam_pos, axis=3)
        hs = self.transformer(self.query_embed.weight,
                              src, src_pos, None,
                              robot_state_input, self.robot_state_pos.weight,
                              latent_input, self.latent_pos.weight)[0]
        a_hat = self.action_head(hs)
        return a_hat, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(self, backbones, depth_backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.depth_backbones = depth_backbones
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            
            for i, backbone in enumerate(backbones):
                num_channels = backbone.num_channels
                if self.depth_backbones is not None:
                    num_channels += depth_backbones[i].num_channels
                down_proj = nn.Sequential(
                    nn.Conv2d(num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=state_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, image, depth_image, robot_state, actions=None, action_is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        bs, _ = robot_state.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            if self.depth_backbones is not None and depth_image is not None:
                features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                all_cam_features.append(self.backbone_down_projs[cam_id](torch.cat([features, features_depth], axis=1)))
            else:
                all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, robot_state], axis=1)  # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


class Diffusion(nn.Module):
    def __init__(self, backbones, pools, linears, depth_backbones, state_dim, chunk_size,
                 observation_horizon, action_horizon, num_inference_timesteps,
                 ema_power, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.backbones = nn.ModuleList(backbones)
        self.backbones = replace_bn_with_gn(self.backbones)  # TODO
        self.pools = nn.ModuleList(pools)
        self.linears = nn.ModuleList(linears)
        self.depth_backbones = depth_backbones
        if depth_backbones is not None:
            self.depth_backbones = nn.ModuleList(depth_backbones)
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.chunk_size = chunk_size
        self.num_inference_timesteps = num_inference_timesteps
        self.ema_power = ema_power
        self.state_dim = state_dim
        self.weight_decay = 0
        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = state_dim
        self.obs_dim = self.feature_dimension * len(self.camera_names) + state_dim  # camera features and proprio
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.state_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon
        )
        if depth_backbones is not None:
            nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'backbones': self.backbones,
                    'depth_backbones': self.depth_backbones,
                    'pools': self.pools,
                    'linears': self.linears,
                    'noise_pred_net': self.noise_pred_net
                })
            })
        else:
            nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'backbones': self.backbones,
                    'pools': self.pools,
                    'linears': self.linears,
                    'noise_pred_net': self.noise_pred_net
                })
            })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            # ema = EMAModel(model=nets, power=self.ema_power)
            ema = EMA(nets, power=self.ema_power, beta = 0.9999, update_every = 1)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        # self.noise_scheduler = DDIMScheduler(
        #     num_train_timesteps=50,
        #     beta_schedule='squaredcos_cap_v2',
        #     clip_sample=True,
        #     set_alpha_to_one=True,
        #     steps_offset=0,
        #     prediction_type='epsilon' # "sample"
        # )

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type= "sample" #'epsilon'
        )

    def forward(self, image, depth_image, robot_state, actions=None, action_is_pad=None):
        B = robot_state.shape[0]
        pred_type = self.noise_scheduler.config.prediction_type
        if actions is not None:  # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                if depth_image is not None:
                    features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                    cam_features = torch.cat([cam_features, features_depth], axis=1)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)
            # TODO: add cross-attention
            obs_cond = torch.cat(all_features + [robot_state], dim=1)
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            # predict the noise residual
            pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            # if self.ema is not None:
            #     self.ema.step(nets)
            if pred_type == "epsilon":
                return noise, pred
            elif pred_type == "sample":
                return actions, pred
        else:
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.chunk_size

            nets = self.nets
            nets.eval()
            if self.ema is not None:
                nets = self.ema.ema_model

            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                if depth_image is not None:
                    features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                    cam_features = torch.cat([cam_features, features_depth], axis=1)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)
            obs_cond = torch.cat(all_features + [robot_state], dim=1)
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, self.state_dim), device=obs_cond.device)
            naction = noisy_action
            # init scheduler
            print("inference steps: ", self.num_inference_timesteps)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            # "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            # status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status_ema = self.ema.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status


def build_diffusion(args):
    if args.use_robot_base:
        state_dim = args.state_dim + 2  # TODO hardcode
    else:
        state_dim = args.state_dim

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    pools = []
    linears = []
    depth_backbones = None
    if args.use_depth_image:
        depth_backbones = []
    
    for cam_name in args.camera_names:
        if 'tactile' in cam_name and args.pretrained_tactile_backbone:
            tactile_weights = torch.load(args.tactile_backbone_path)
            backbone = modified_resnet18()
            backbone.load_state_dict(tactile_weights)
            backbone.requires_grad_(False) # TODO: freeze the tactile encoder
        elif args.pretrained_vision_backbone:
            vision_weights = torch.load(args.vision_backbone_path)
            backbone = modified_resnet18()
            backbone.load_state_dict(vision_weights)
        elif args.backbone == 'vit':
            backbone = load_vit_encoder()
        else:
            backbone = ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False})
        
        backbones.append(backbone)
        num_channels = 512
        
        if args.use_depth_image:
            depth_backbones.append(DepthNet())
            num_channels += depth_backbones[-1].num_channels
        
        pools.append(SpatialSoftmax(**{'input_shape': [num_channels, 15, 20], 'num_kp': 32, 'temperature': 1.0,
                                       'learnable_temperature': False, 'noise_std': 0.0}))
        linears.append(torch.nn.Linear(int(np.prod([32, 2])), 64))


    model = Diffusion(
        backbones,
        pools,
        linears,
        depth_backbones,
        state_dim=state_dim,
        chunk_size=args.chunk_size,
        observation_horizon=args.observation_horizon,
        action_horizon=args.action_horizon,
        num_inference_timesteps=args.num_inference_timesteps,
        ema_power=args.ema_power,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))
    return model


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    
    d_model = args.hidden_dim  # 256
    dropout = args.dropout     # 0.1
    nhead = args.nheads        # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    if args.use_robot_base:
        state_dim = args.state_dim + 2  # TODO hardcode
    else:
        state_dim = args.state_dim

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []   # 空的网络list
    depth_backbones = None
    tactile_backbones = None
    if args.use_depth_image:
        depth_backbones = []
    if args.use_tactile_image:
        tactile_backbones = []

    # backbone = build_backbone(args)  # 位置编码和主干网络组合成特征提取器
    # backbones.append(backbone)
    # if args.use_depth_image:
    #     depth_backbones.append(DepthNet())

    for cam_name in args.camera_names:
        if args.pretrained_vision_backbone:
            position_embedding = build_position_encoding(args)
            backbone_model = load_backbone(args.backbone)
            backbone_model.load_state_dict(torch.load(args.vision_backbone_path))
            backbone = MyJoiner(backbone_model, position_embedding)
            backbone.num_channels = 768 if args.backbone == "vit" else 512# resnet-18 hardcoded
            
        elif args.backbone == 'vit':
            position_embedding = build_position_encoding(args)
            backbone_model = load_vit_encoder()
            backbone = MyJoiner(backbone_model, position_embedding)
            backbone.num_channels = 768 # TODO: vit channel
        else:
            backbone = build_backbone(args)
        backbones.append(backbone)
        if args.use_depth_image:
            depth_backbones.append(DepthNet())
        
    if args.use_tactile_image:
        for tactile_name in args.tactile_camera_names:
            if args.pretrained_tactile_backbone:
                position_embedding = build_position_encoding(args)
                backbone_model = load_backbone(args.backbone)
                backbone_model.load_state_dict(torch.load(args.tactile_backbone_path))
                
                backbone_model.requires_grad_(False) # TODO: freeze the tactile encoder
                backbone = MyJoiner(backbone_model, position_embedding)
                backbone.num_channels = 768 if args.backbone == "vit" else 512# resnet-18 hardcoded
                
            elif args.backbone == 'vit':
                position_embedding = build_position_encoding(args)
                backbone_model = load_vit_encoder()
                backbone = MyJoiner(backbone_model, position_embedding)
                backbone.num_channels = 768 # TODO: vit channel
            else:
                backbone = build_backbone(args)
            tactile_backbones.append(backbone)


    transformer = build_transformer(args)  # 构建trans层

    encoder = None
    if args.kl_weight != 0:
        encoder = build_encoder(args)          # 构建编码成和解码层

    model = DETRVAE(
        backbones,
        depth_backbones,
        tactile_backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.chunk_size,
        camera_names=args.camera_names,
        tactile_camera_names=args.tactile_camera_names,
        kl_weight=args.kl_weight
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


def build_cnnmlp(args):
    if args.use_robot_base:
        state_dim = 16  # TODO hardcode
    else:
        state_dim = 14

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []   # 空的网络list
    depth_backbones = None
    if args.use_depth_image:
        depth_backbones = []

    # backbone = build_backbone(args)  # 位置编码和主干网络组合成特征提取器
    # backbones.append(backbone)
    # if args.use_depth_image:
    #     depth_backbones.append(DepthNet())

    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
        if args.use_depth_image:
            depth_backbones.append(DepthNet())

    model = CNNMLP(
        backbones,
        depth_backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
