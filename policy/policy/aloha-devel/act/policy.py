import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer, build_diffusion_model_and_optimizer

import IPython
e = IPython.embed


class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_diffusion_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

    def configure_optimizers(self):
        return self.optimizer

    def __call__(self, image, depth_image, robot_state, actions=None, action_is_pad=None):
        B = robot_state.shape[0]
        if actions is not None:
            self.model.train()
            actions = actions[:, :self.model.chunk_size]
            action_is_pad = action_is_pad[:, :self.model.chunk_size]
            noise, noise_pred = self.model(image, depth_image, robot_state, actions, action_is_pad)
            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~action_is_pad.unsqueeze(-1)).mean()
            joint_loss = (all_l2 * ~action_is_pad.unsqueeze(-1)).mean(dim=(0, 1))

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss
            for i in range(len(joint_loss)):
                loss_dict[f'joint_{i}_loss'] = joint_loss[i]
            return loss_dict, (noise, noise_pred)
        else:  # inference time
            return self.model(image, depth_image, robot_state, actions, action_is_pad)

    def serialize(self):
        return self.model.serialize()

    def deserialize(self, model_dict):
        return self.model.deserialize(model_dict)


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)

        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.loss_function = args_override['loss_function']

        print(f'KL Weight {self.kl_weight}')

    def __call__(self, image, depth_image, robot_state, actions=None, action_is_pad=None):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        depth_normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        image = normalize(image)  # 图像归一化
        if depth_image is not None:
            depth_image = depth_normalize(depth_image)

        # 总共max个步 只取前model.num_queries个 
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            action_is_pad = action_is_pad[:, :self.model.num_queries]

            a_hat, (mu, logvar) = self.model(image, depth_image, robot_state, actions, action_is_pad)

            loss_dict = dict()
            if self.loss_function == 'l1':
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            elif self.loss_function == 'l2':
                all_l1 = F.mse_loss(actions, a_hat, reduction='none')
            else:
                all_l1 = F.smooth_l1_loss(actions, a_hat, reduction='none')

            l1 = (all_l1 * ~action_is_pad.unsqueeze(-1)).mean()

            loss_dict['l1'] = l1
            if self.kl_weight != 0:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            else:
                loss_dict['loss'] = loss_dict['l1']

            return loss_dict, a_hat
        else:  # inference time
            a_hat, (_, _) = self.model(image, depth_image, robot_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer
        self.loss_function = args_override['loss_function']

    # 而 __call__ 在对象被调用时执行
    def __call__(self, image, depth_image, robot_state, actions=None,
                 action_is_pad=None):
        env_state = None  # TODO

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        depth_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize(image)  # 图像归一化
        if depth_image is not None:
            depth_image = depth_normalize(depth_image)
        if actions is not None:  # training time
            actions = actions[:, 0]  # 动作
            a_hat = self.model(image, depth_image, robot_state, actions, action_is_pad)
            # 均方误差
            if self.loss_function == 'l1':
                mse = F.l1_loss(actions, a_hat)
            elif self.loss_function == 'l2':
                mse = F.mse_loss(actions, a_hat)
            else:
                mse = F.smooth_l1_loss(actions, a_hat)

            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict, a_hat

        else:  # inference time
            a_hat = self.model(image, depth_image, robot_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
