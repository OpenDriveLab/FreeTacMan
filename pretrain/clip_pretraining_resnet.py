import os
import cv2
import h5py
import torch
import torchvision
import argparse
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from typing import Tuple, Dict, Union, Callable, List
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class MemoryBank:
    def __init__(self, bank_size: int, feat_dim: int, device: torch.device):
        self.bank_size = bank_size
        self.device = device
        self.register = F.normalize(torch.randn(bank_size, feat_dim, device=device), dim=1)
        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, emb: torch.Tensor):
        M = emb.shape[0]
        if M >= self.bank_size:
            emb = emb[-self.bank_size:]
            M = self.bank_size
        idx = (self.ptr + torch.arange(M, device=self.device)) % self.bank_size
        self.register[idx] = emb.detach()
        self.ptr = (self.ptr + M) % self.bank_size

    def get_bank(self) -> torch.Tensor:
        return self.register.detach()


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


def replace_bn_with_gn(
        root_module: nn.Module,
        features_per_group: int = 16) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features)
    )
    return root_module


class ClipProjectionHead(nn.Module):
    def __init__(self, out_dim: int, conditioning_dim: int = 0, num_channels: int = 512, normalize: bool = True):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1, -1)
        self.linear = nn.Linear(num_channels + conditioning_dim, out_dim)
        self.normalize = normalize

    def forward(self, feature_map, conditioning=None) -> torch.Tensor:
        x = self.pooling(feature_map)
        x = self.flatten(x)
        if conditioning is not None:
            x = torch.cat((x, conditioning), dim=-1)
        x = self.linear(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x


def modified_resnet18(weights=None, features_per_group=16) -> nn.Module:
    resnet18 = getattr(torchvision.models, 'resnet18')(pretrained=True)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
    resnet18 = replace_bn_with_gn(resnet18, features_per_group=features_per_group)
    return resnet18


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 episode_ids: List[int], 
                 dataset_dir: str, 
                 camera_names: List[str], 
                 norm_stats: Dict[str, Union[float, np.ndarray]],
                 image_size: Tuple[int, int] = None, 
                 tactile_size: Tuple[int, int] = None,
                 min_distance: int = 5,
                 n_images: int = 10,
                 is_cluster: bool = False):
        super(ClipDataset).__init__()
        self.n_images = n_images
        self.min_distance = min_distance
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.image_size = image_size
        self.n_cameras = len(camera_names)
        self.is_cluster = is_cluster

        assert "tactile_mean" in norm_stats, "tactile data must exist"
        tactile_mean = norm_stats["tactile_mean"]
        tactile_std = norm_stats["tactile_std"]
        self.position_mean = norm_stats["qpos_mean"]
        self.position_std = norm_stats["qpos_std"]

        self.image_normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229,0.224,0.225])
        self.tactile_normalize = Normalize(mean=tactile_mean, std=tactile_std)

        self.episode_lengths = []
        for episode_id in self.episode_ids:
            dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                self.episode_lengths.append(len(root['action'][()]))


    def __len__(self):
        return len(self.episode_ids)


    def __getitem__(self, index):
        timesteps = []
        while len(timesteps) < self.n_images:
            t = np.random.randint(0, self.episode_lengths[index])
            good_timestep = True
            for prev_t in timesteps:
                if abs(t - prev_t) < self.min_distance:
                    good_timestep = False
            if good_timestep:
                timesteps.append(t)

        dataset_path = os.path.join(self.dataset_dir, f'episode_{self.episode_ids[index]}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            all_cam_images = []
            all_tactile_images = []
            all_positions = []
            for timestep in timesteps:
                timestep_cam_images = []
                for cam_name in self.camera_names:
                    image = root[f'/observations/images/{cam_name}'][timestep]
                    image = torch.tensor(image, dtype=torch.float32) / 255.0
                    image = torch.einsum('h w c -> c h w', image)
                    image = self.image_normalize(image)
                    timestep_cam_images.append(image)
                images = torch.stack(timestep_cam_images, axis=0)

                tactile_data = root['observations/images/tactile'][timestep]
                tactile_data = torch.tensor(tactile_data, dtype=torch.float32) / 255.0
                tactile_data = torch.einsum('h w c -> c h w', tactile_data)
                tactile_data = self.tactile_normalize(tactile_data)

                position = root['observations/qpos'][timestep]
                position = (position - self.position_mean) / self.position_std
                position = torch.tensor(position[:7], dtype=torch.float32)

                all_cam_images.append(images)
                all_tactile_images.append(tactile_data)
                all_positions.append(position)

        return torch.stack(all_cam_images, axis=0), torch.stack(all_tactile_images, axis=0), torch.stack(all_positions, axis=0)


def clip_loss(image_embeddings: torch.Tensor, tactile_embeddings: torch.Tensor, target_matrix: torch.Tensor, logit_scale: float = 1.0):
    n_cameras = image_embeddings.shape[2]
    batch_size = image_embeddings.shape[0]
    loss = torch.zeros(n_cameras, device=image_embeddings.device)

    image_embeddings = image_embeddings.permute(0, 2, 1, 3)
    tactile_embeddings = tactile_embeddings.unsqueeze(1)

    image_logits = logit_scale * image_embeddings @ tactile_embeddings.permute(0, 1, 3, 2)
    tactile_logits = logit_scale * tactile_embeddings @ image_embeddings.permute(0, 1, 3, 2)

    image_logits = image_logits.flatten(0, 1)
    tactile_logits = tactile_logits.flatten(0, 1)

    image_loss = F.cross_entropy(image_logits, target_matrix.repeat(image_logits.shape[0], 1, 1), reduce=False).mean(dim=1)
    tactile_loss = F.cross_entropy(tactile_logits, target_matrix.T.repeat(tactile_logits.shape[0], 1, 1), reduce=False).mean(dim=1)

    image_loss = image_loss.view(batch_size, n_cameras)
    tactile_loss = tactile_loss.view(batch_size, n_cameras)

    loss = ((image_loss + tactile_loss) / 2.0).mean(dim=0)

    return loss


def multi_positive_contrastive_loss(image_emb: torch.Tensor, tactile_emb: torch.Tensor, memory_bank: MemoryBank = None, temperature: float = 0.07):
    B, N, D = image_emb.shape
    img = F.normalize(image_emb.view(-1, D), dim=1)
    tac = F.normalize(tactile_emb.view(-1, D), dim=1)

    pos_primary = torch.sum(img * tac, dim=1, keepdim=True) / temperature
    idx = torch.arange(B * N, device=img.device)
    sec_idx = idx // N * N + ((idx % N + 1) % N)
    pos_secondary = torch.sum(img * tac[sec_idx], dim=1, keepdim=True) / temperature
    pos_logits = torch.cat([pos_primary, pos_secondary], dim=1)

    if memory_bank is not None:
        neg_bank = memory_bank.get_bank()
        neg_logits = img @ neg_bank.t() / temperature
        logits = torch.cat([pos_logits, neg_logits], dim=1)
    else:
        sim = img @ tac.t() / temperature
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask[idx, idx] = False
        mask[idx, sec_idx] = False
        neg_logits = sim[mask].view(B * N, -1)
        logits = torch.cat([pos_logits, neg_logits], dim=1)

    labels = torch.zeros(B * N, dtype=torch.long, device=img.device)
    loss = F.cross_entropy(logits, labels)

    return loss


def clip_pretraining(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    save_freq: int = 100,
    plot_freq: int = 50,
    n_epochs: int = 1000,
    clip_dim: int = 512,
    features_per_group: int = 16,
    resnet_lr: float = 1e-5,
    projection_lr: float = 1e-4,
    use_multi_positive: bool = False,
    memory_bank_size: int = 4096
):
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]

    dataset: ClipDataset = train_loader.dataset
    n_cameras = dataset.n_cameras
    state_size = 7

    vision_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)
    for param in vision_encoder.parameters():
        param.requires_grad = False

    vision_projection = ClipProjectionHead(out_dim=clip_dim).to(device)

    tactile_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)
    tactile_projection = ClipProjectionHead(out_dim=clip_dim, conditioning_dim=state_size).to(device)

    optim_params = [
        {"params": tactile_encoder.parameters(), "lr": resnet_lr},
        {"params": tactile_projection.parameters(), "lr": projection_lr},
        {"params": vision_projection.parameters(), "lr": projection_lr},
    ]
    optimizer = torch.optim.Adam(optim_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    memory_bank = MemoryBank(bank_size=memory_bank_size, feat_dim=clip_dim, device=device) if use_multi_positive else None

    training_losses = []
    testing_losses = []

    writer = SummaryWriter(log_dir=f"{save_dir}/runs")
    global_step = 0

    for epoch in tqdm(range(n_epochs)):
        training_loss = 0.0
        tactile_encoder.train()
        tactile_projection.train()
        vision_encoder.eval()
        vision_projection.train()

        for batch_idx, (images, tactile, position) in enumerate(train_loader):
            images = images.to(device)
            tactile = tactile.to(device)
            position = position.to(device)

            batch_size = images.shape[0]
            clip_N = images.shape[1]

            images = images.view(-1, images.shape[3], images.shape[4], images.shape[5])
            image_embeddings = vision_projection(vision_encoder(images))
            image_embeddings = image_embeddings.view(batch_size, clip_N, n_cameras, clip_dim)

            tactile = tactile.view(-1, tactile.shape[2], tactile.shape[3], tactile.shape[4])
            position = position.view(-1, position.shape[2])
            tactile_embeddings = tactile_projection(tactile_encoder(tactile), position)
            tactile_embeddings = tactile_embeddings.view(batch_size, clip_N, clip_dim)

            img_flat = image_embeddings.view(-1, clip_N, clip_dim)
            tac_flat = tactile_embeddings.unsqueeze(1).expand(-1, n_cameras, -1, -1).reshape(-1, clip_N, clip_dim)

            if use_multi_positive:
                loss = multi_positive_contrastive_loss(img_flat, tac_flat, memory_bank)
            else:
                target_matrix = torch.eye(clip_N).to(device)
                loss = clip_loss(image_embeddings, tactile_embeddings, target_matrix)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if use_multi_positive:
                memory_bank.enqueue(tac_flat.view(-1, clip_dim))

            training_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step=global_step)
            global_step += 1

        training_losses.append(training_loss / len(train_loader))

        testing_loss = 0.0
        tactile_encoder.eval()
        tactile_projection.eval()
        vision_encoder.eval()
        vision_projection.eval()

        with torch.no_grad():
            for batch_idx, (images, tactile, position) in enumerate(test_loader):
                images = images.to(device)
                tactile = tactile.to(device)
                position = position.to(device)

                batch_size = images.shape[0]
                clip_N = images.shape[1]

                images = images.view(-1, images.shape[3], images.shape[4], images.shape[5])
                image_embeddings = vision_projection(vision_encoder(images))
                image_embeddings = image_embeddings.view(batch_size, clip_N, n_cameras, clip_dim)

                tactile = tactile.view(-1, tactile.shape[2], tactile.shape[3], tactile.shape[4])
                position = position.view(-1, position.shape[2])
                tactile_embeddings = tactile_projection(tactile_encoder(tactile), position)
                tactile_embeddings = tactile_embeddings.view(batch_size, clip_N, clip_dim)

                img_flat = image_embeddings.view(-1, clip_N, clip_dim)
                tac_flat = tactile_embeddings.unsqueeze(1).expand(-1, n_cameras, -1, -1).reshape(-1, clip_N, clip_dim)

                if use_multi_positive:
                    loss = multi_positive_contrastive_loss(img_flat, tac_flat, memory_bank)
                else:
                    target_matrix = torch.eye(clip_N).to(device)
                    loss = clip_loss(image_embeddings, tactile_embeddings, target_matrix)

                testing_loss += loss.item()
                writer.add_scalar("Loss/test", loss.item(), global_step=global_step)

        testing_losses.append(testing_loss / len(test_loader))

        if epoch % plot_freq == 0:
            plt.figure()
            plt.plot(training_losses, label='Training Loss')
            plt.plot(testing_losses, label='Testing Loss')
            plt.legend()
            plt.savefig(f'{save_dir}/graphs/training_loss.png')
            plt.close()

        if (epoch + 1) % save_freq == 0:
            torch.save(vision_encoder.state_dict(), f'{save_dir}/epoch_{epoch}_vision_encoder.pth')
            torch.save(vision_projection.state_dict(), f'{save_dir}/epoch_{epoch}_vision_projection.pth')
            torch.save(tactile_encoder.state_dict(), f'{save_dir}/epoch_{epoch}_tactile_encoder.pth')
            torch.save(tactile_projection.state_dict(), f'{save_dir}/epoch_{epoch}_tactile_projection.pth')

    writer.close()


def run_clip_pretraining(
    dataset_dir: str = None,
    save_dir: str = "./",
    num_episodes: int = 200,
    camera_names: list = ['front'],
    use_existing: bool = False,
    batch_size: int = 3,
    n_clip_images: int = 5,
    min_distance: int = 20,
    n_epochs: int = 1500,
    plot_freq: int = 50,
    save_freq: int = 10,
    features_per_group: int = 16,
    clip_dim: int = 512,
    resnet_lr: float = 1e-5,
    projection_lr: float = 1e-4,
    use_multi_positive: bool = False,
    memory_bank_size: int = 4096
):
    from utils import get_norm_stats  # Assuming get_norm_stats is in utils.py

    if not os.path.exists(dataset_dir):
        raise ValueError("Dataset directory does not exist")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_existing=use_existing)
    batch_size_train = batch_size
    batch_size_test = batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    train_dataset = ClipDataset(train_indices, dataset_dir, camera_names, norm_stats, n_images=n_clip_images, min_distance=min_distance)
    test_dataset = ClipDataset(val_indices, dataset_dir, camera_names, norm_stats, n_images=n_clip_images, min_distance=min_distance)

    if device == torch.device("cuda"):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=10, prefetch_factor=10, pin_memory_device='cuda')
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, pin_memory=True, num_workers=10, prefetch_factor=10, pin_memory_device='cuda')
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

    ns = [-1]
    for folder in os.listdir(save_dir):
        if folder.isdigit():
            ns.append(int(folder))
    n = max(ns) + 1
    os.makedirs(f'{save_dir}/{n}')
    os.makedirs(f'{save_dir}/{n}/graphs')
    os.makedirs(f'{save_dir}/{n}/runs')

    with open(f'{save_dir}/{n}/run_stats.txt', 'w') as f:
        f.write(f'num_episodes: {num_episodes}\n')
        f.write(f'dataset_dir: {dataset_dir}\n')
        f.write(f'camera_names: {camera_names}\n')
        f.write(f'norm_stats: {norm_stats}\n')
        f.write(f'batch_size_train: {batch_size_train}\n')
        f.write(f'batch_size_test: {batch_size_test}\n')
        f.write(f'n_clip_images: {n_clip_images}\n')
        f.write(f'min_distance: {min_distance}\n')
        f.write(f'train_indices: {train_indices}\n')
        f.write(f'val_indices: {val_indices}\n')
        f.write(f'use_multi_positive: {use_multi_positive}\n')

    clip_pretraining(
        train_dataloader,
        test_dataloader,
        device,
        save_dir=f'{save_dir}/{n}',
        save_freq=save_freq,
        plot_freq=plot_freq,
        n_epochs=n_epochs,
        clip_dim=clip_dim,
        features_per_group=features_per_group,
        resnet_lr=resnet_lr,
        projection_lr=projection_lr,
        use_multi_positive=use_multi_positive,
        memory_bank_size=memory_bank_size
    )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--dataset_dir', type=str, default='dataset/whole_dataset_freetacman_1')
    args_parser.add_argument('--save_dir', type=str, default='checkpoint/clip_resnet')
    args_parser.add_argument('--num_episodes', type=int, default=1000)
    args_parser.add_argument('--camera_names', type=list, default=['front'])
    args_parser.add_argument('--use_existing', type=bool, default=True)
    args_parser.add_argument('--batch_size', type=int, default=45)
    args_parser.add_argument('--n_clip_images', type=int, default=5)
    args_parser.add_argument('--min_distance', type=int, default=20)
    args_parser.add_argument('--save_freq', type=int, default=50)
    args_parser.add_argument('--plot_freq', type=int, default=50)
    args_parser.add_argument('--n_epochs', type=int, default=3000)
    args_parser.add_argument('--resnet_lr', type=float, default=1e-5)
    args_parser.add_argument('--projection_lr', type=float, default=1e-4)
    args_parser.add_argument('--clip_dim', type=int, default=512)
    args_parser.add_argument('--features_per_group', type=int, default=64)
    args_parser.add_argument('--use_multi_positive', type=bool, default=False)
    args_parser.add_argument('--memory_bank_size', type=int, default=4096)

    args = args_parser.parse_args()

    run_clip_pretraining(
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir,
        num_episodes=args.num_episodes,
        camera_names=args.camera_names,
        use_existing=args.use_existing,
        batch_size=args.batch_size,
        n_clip_images=args.n_clip_images,
        min_distance=args.min_distance,
        save_freq=args.save_freq,
        plot_freq=args.plot_freq,
        n_epochs=args.n_epochs,
        resnet_lr=args.resnet_lr,
        projection_lr=args.projection_lr,
        clip_dim=args.clip_dim,
        features_per_group=args.features_per_group,
        use_multi_positive=args.use_multi_positive,
        memory_bank_size=args.memory_bank_size
    )