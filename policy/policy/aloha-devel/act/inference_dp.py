import sys
sys.path.append("./")  # 视具体目录结构而定
import torch
import numpy as np
import os
import pickle
import argparse
import time

from einops import rearrange

# 如果你已经实现了以下这些类或函数，请根据你的工程路径改写import
from utils import set_seed
from policy import DiffusionPolicy  # 你的 DiffusionPolicy 类
# 也可以导入或自定义你需要的 connect_device、get_obs 等
# 这里示例以 Feetech 电机 + opencv 相机为例

import cv2
from common.utils.utils import load_config_from_yaml
from common.devices.motor.feetech import FeetechMotorsBus
from common.devices.motor.feetech import SCS_SERIES_BAUDRATE_TABLE as BAUDRATE_TABLE


def connect_device(inference_config):
    """
    根据你的硬件配置进行设备连接。
    inference_config 是一个从 YAML/JSON 等文件里读取的配置字典。
    如果不需要连接任何设备，可将此函数简化/跳过。
    """
    port_read = inference_config['port_read']
    model = inference_config['model']     # 电机型号(如 SCS15)
    n = inference_config['n']            # 电机数量
    baudrate = inference_config['baudrate']

    # 初始化电机总线
    read_motor_bus = FeetechMotorsBus(port=port_read, motors={f"motor_{i}": (i, model) for i in range(1, n+1)})
    read_motor_bus.connect()
    read_motor_bus.set_bus_baudrate(baudrate)
    
    # 初始化相机
    cameras = inference_config['camera0']  # 假设只用一台相机
    cap = cv2.VideoCapture(cameras)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: {cameras}")
    
    return cap, read_motor_bus


def get_obs(cap, state_dim, read_motor_bus):
    """
    从相机和电机总线中获取当前观测:
    - 相机画面
    - 机械臂/电机的关节角
    """
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"无法从相机读取图像")
    
    # BGR -> RGB (如你后面模型需要 BGR，可自行修改)
    camera0_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 读取电机角度，这里假设你用 state_dim 个电机
    positions = np.zeros(state_dim)
    for i in range(1, state_dim + 1):
        # Present_Position 通常返回 [0,4095] 之类，需要你自己映射到实际角度
        raw_val = read_motor_bus.read_with_motor_ids(read_motor_bus.motor_models, i, "Present_Position")
        # 根据需要做一下 mod 或者放缩
        positions[i - 1] = raw_val % 4096

    obs = {
        'qpos': positions,
        'images': {
            "camera0": camera0_data,
        }
    }
    return obs


def get_image(obs, cam_names):
    """
    将 obs['images'][cam_name] 转成 (B, C, H, W) 的 float Tensor。
    若你有多相机, 可以对多张图像做拼接或分别处理再 cat。
    """
    img = obs['images'][cam_names[0]]  # 这里简化为只取第一台相机
    # [H, W, C] -> [C, H, W]
    img_chw = rearrange(img, 'h w c -> c h w')
    # 归一化到 [0, 1]
    img_chw = img_chw.astype(np.float32) / 255.0
    # 增加 batch 维度 (B=1)
    img_chw = np.expand_dims(img_chw, axis=0)  
    # 转成 torch.tensor
    img_tensor = torch.from_numpy(img_chw).cuda()  # shape: (1, 3, H, W)
    return img_tensor


def load_diffusion_policy(policy, ckpt_path, map_location='cuda'):
    """
    从 ckpt_path 加载 DiffusionPolicy (含 nets + ema)。 
    ckpt_path 是训练时使用 policy.serialize() 的结果。

    如果你的 policy 里已经封装了 deserialize，可以直接调用 policy.deserialize(checkpoint_dict)。
    """
    checkpoint_dict = torch.load(ckpt_path, map_location=map_location)
    load_status = policy.deserialize(checkpoint_dict)
    policy.cuda()
    policy.eval()
    print(f"Loaded DiffusionPolicy from {ckpt_path}.")
    return load_status


def main(args):
    """
    推理入口
    """
    # 1. 载入你自定义的推理配置(如电机串口、相机ID等)
    # inference_config = load_config_from_yaml("./config/inference/inference.yaml")  
    inference_config = None
    
    # 2. 设置随机种子(可选)
    set_seed(args.seed)

    # 3. 连接真实设备(相机、电机), 如果是仿真环境请自行修改/注释
    cap, read_motor_bus = connect_device(inference_config)

    # 4. 组装 policy_config，这通常跟训练时保持一致
    #    这里只示例一些必要参数，更多的看你自己的 DiffusionPolicy 和 build_diffusion
    policy_config = {
        'lr': args.lr,
        'lr_backbone': 1e-5,
        'backbone': args.backbone,
        'masks': args.masks,
        'weight_decay': args.weight_decay,
        'dilation': args.dilation,
        'position_embedding': args.position_embedding,
        'loss_function': args.loss_function,
        'chunk_size': args.chunk_size,
        'camera_names': ['camera0'],  # 跟 get_obs 中要对应
        'use_depth_image': False,
        'use_robot_base': False,
        'observation_horizon': args.observation_horizon,
        'action_horizon': args.action_horizon,
        'num_inference_timesteps': args.num_inference_timesteps,
        'ema_power': args.ema_power,
        'hidden_dim': args.hidden_dim,
        'state_dim': args.state_dim,
    }

    # 5. 构建一个空的 DiffusionPolicy 并加载权重
    policy = DiffusionPolicy(policy_config)
    ckpt_path = os.path.join(args.ckpt_dir, "policy_best.ckpt")
    load_diffusion_policy(policy, ckpt_path=ckpt_path, map_location='cuda')

    # 6. 加载训练时保存的数据集统计(均值方差), 用于后处理
    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 定义预处理 / 后处理
    pre_process = lambda qpos: (qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda act: act * stats['qpos_std'] + stats['qpos_mean']

    # 7. 进行多次 rollout (演示多次执行)
    num_rollouts = args.num_rollouts
    max_timesteps = args.max_timesteps  # 每个rollout执行多少步

    for rollout_id in range(num_rollouts):
        print(f"\nStart rollout {rollout_id} / {num_rollouts} ...")
        # 如果需要存放某些信息，可在此处初始化列表

        with torch.inference_mode():
            for t in range(max_timesteps):
                # (1) 读取当前观测
                obs = get_obs(cap, args.state_dim, read_motor_bus)
                qpos_numpy = obs['qpos']

                # (2) 预处理
                qpos_tensor = torch.from_numpy(pre_process(qpos_numpy)).float().cuda().unsqueeze(0)  # shape: (1, state_dim)
                img_tensor = get_image(obs, policy_config['camera_names'])  # shape: (1, 3, H, W)

                # (3) 调用 diffusion policy 进行推理
                out_actions = policy(
                    image=img_tensor, 
                    depth_image=None,          # 如果有深度图，请传这里
                    robot_state=qpos_tensor,
                    actions=None,              # None 表示 inference
                    action_is_pad=None
                )  
                # out_actions 形状: (1, chunk_size, state_dim)，比如 (1, 8, 7)

                # (4) 选取要执行的动作，这里示例只拿第 0 个
                predicted_action = out_actions[:, 0, :]  # shape: (1, state_dim)
                predicted_action = predicted_action.squeeze(0).cpu().numpy()  # (state_dim,)

                # (5) 后处理到真实角度量纲
                predicted_action = post_process(predicted_action)

                # (6) 将动作下发到电机
                for i in range(1, args.state_dim + 1):
                    read_motor_bus.write_with_motor_ids(
                        read_motor_bus.motor_models, 
                        i, 
                        "Goal_Position", 
                        int(predicted_action[i - 1])
                    )

                # 如果机械臂需要一段时间才能到达指定位置，可以做短暂 sleep
                # time.sleep(0.05)
            
            print(f"Rollout {rollout_id} finished.")

        # 如果需要的话可以在这里对 rollout_id 进行一些统计或记录

    # 所有rollout完成后，释放资源
    cap.release()
    # read_motor_bus.disconnect()  # 如果有需求也可断开串口

    print("All inference rollouts done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='训练好的模型所在目录')
    parser.add_argument('--task_name', type=str, default='dummy_task', help='任务名称(仅打印)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--state_dim', type=int, default=7, help='关节维度(电机数量)')
    parser.add_argument('--num_rollouts', type=int, default=1, help='要执行多少条完整的rollout')
    parser.add_argument('--max_timesteps', type=int, default=40, help='每条rollout执行多少步')

    # diffusion相关参数(与训练时对应)
    parser.add_argument('--chunk_size', type=int, default=8)
    parser.add_argument('--observation_horizon', type=int, default=1)
    parser.add_argument('--action_horizon', type=int, default=8)
    parser.add_argument('--num_inference_timesteps', type=int, default=10)
    parser.add_argument('--ema_power', type=float, default=0.75)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--loss_function', type=str, default='l1')
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', type=str, default='sine', choices=('sine', 'learned'))
    parser.add_argument('--masks', action='store_true')

    args = parser.parse_args()

    main(args)