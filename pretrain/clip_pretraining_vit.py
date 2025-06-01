import os
import cv2
import h5py
import torch
import timm
import argparse
import torchvision
import numpy as np

from torch import nn
from tqdm import tqdm
from typing import Tuple, Dict, Union, Callable, List
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import (
    Normalize, RandomRotation, RandomResizedCrop, ColorJitter, 
    RandomGrayscale, GaussianBlur, RandomAdjustSharpness, 
    RandomAutocontrast, RandomEqualize, RandomPosterize, RandomSolarize
)
from transformers import AutoModel, pipeline

from PIL import Image, ImageOps

os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'
VIT_MODEL_PATH = "/path/to/vit_model" 
# Load CLIP ViT model
# transform = transforms.Compose([
#     transforms.Lambda(lambda x: Image.fromarray(x.astype('uint8'))),
#     transforms.Lambda(lambda x: ImageOps.pad(x, (max(x.size), max(x.size)))),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # -> FloatTensor, C×H×W, values ∈ [0,1]
#     transforms.Normalize(mean=[0.485,0.456,0.406],
#                          std =[0.229,0.224,0.225]),
# ])


# Replace BatchNorm layers with GroupNorm
def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.
    """
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

def replace_bn_with_gn(root_module: nn.Module, features_per_group: int=16) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

class AttentionPool(nn.Module):
    """
    Learnable attention pooling over patch tokens using a separate CLS query.
    """
    def __init__(self, token_dim: int, num_heads: int = 8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.attn = nn.MultiheadAttention(embed_dim=token_dim,
                                          num_heads=num_heads,
                                          batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D]
        B, N, D = tokens.shape
        # expand cls token to batch
        cls = self.cls_token.expand(B, -1, -1)        # [B,1,D]
        x = torch.cat([cls, tokens], dim=1)            # [B,1+N,D]
        # query=cls, key/value=all
        out, _ = self.attn(query=x[:, :1], key=x, value=x)
        return out.squeeze(1)                          # [B,D]

# Projection head for CLIP model
class ClipProjectionHead(nn.Module):
    def __init__(self,
                 out_dim: int,
                 token_dim: int = 768,
                 conditioning_dim: int = 0,
                 normalize: bool = True,
                 dropout_prob: float = 0.2,
                 pool: str = 'attention',
                 num_heads: int = 8):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.normalize_out = normalize
        # choose pooling
        if pool == 'attention':
            self.pool = AttentionPool(token_dim, num_heads)
            in_dim = token_dim + conditioning_dim
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)  # fallback avg over tokens
            in_dim = token_dim + conditioning_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, tokens: torch.Tensor, conditioning: torch.Tensor = None) -> torch.Tensor:
        # tokens: [B, N, D]
        if isinstance(self.pool, AttentionPool):
            x = self.pool(tokens)            # [B, D]
        else:
            x = tokens.permute(0,2,1)        # [B, D, N]
            x = self.pool(x).squeeze(-1)     # [B, D]
        if conditioning is not None:
            x = torch.cat([x, conditioning], dim=-1)
        x = self.dropout(x)
        x = self.linear(x)
        if self.normalize_out:
            x = F.normalize(x, dim=-1)
        return x


# Modified ResNet18 (for tactile encoder)
def modified_resnet18(weights=None, features_per_group=16) -> nn.Module:
    resnet18 = getattr(torchvision.models, 'resnet18')(pretrained=True)  # Load pre-trained ResNet-18
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])  # Remove fully connected layer and average pooling
    resnet18 = replace_bn_with_gn(resnet18, features_per_group=features_per_group)  # Replace BatchNorm with GroupNorm
    return resnet18    

def load_vit_encoder(use_timm=False, local_model_path=VIT_MODEL_PATH) -> nn.Module:
    """
    Load the ViT encoder model, either from timm or Hugging Face.
    
    Args:
    - use_timm (bool): Whether to use timm to load the model. Defaults to False.
    - local_model_path (str): Local path to the pre-trained model. Required if use_timm is False and model is downloaded locally.

    Returns:
    - model (nn.Module): Loaded ViT model.
    """
    
    if use_timm:
        if local_model_path:
            # If the model is already downloaded locally, use it directly
            model = timm.create_model("vit_base_patch16_224", pretrained=False)  # Don't download again
            model.load_state_dict(torch.load(local_model_path))  # Load local weights
        else:
            # Download the model from timm and load the weights
            model = timm.create_model("hf_hub:timm/vit_base_patch16_clip_224.openai", pretrained=True)
    else:
        if local_model_path:
            # Load model from local path for Hugging Face pre-trained ViT
            model = AutoModel.from_pretrained(local_model_path)
            model = model.vision_model
        else:
            model_name = "timm/vit_base_patch16_clip_224.openai"
            model = AutoModel.from_pretrained(model_name)  # Load pre-trained CLIP ViT model
            model = model.vision_model

    return model


def setup_optimizer_and_scheduler(model_components: List[Dict], lr: float, weight_decay: float,
                                  n_epochs: int):
    optimizer = torch.optim.AdamW(
        model_components,
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6
    )
    return optimizer, scheduler
        
class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids: List[int], dataset_dir: str, camera_names: List[str], norm_stats: Dict[str, Union[float, np.ndarray]], image_size: Tuple[int, int] = None, tactile_size: Tuple[int, int] = None, min_distance = 5, n_images = 10, is_cluster=False, data_augmentation:bool=True):
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

        if data_augmentation:
            self.image_transform = transforms.Compose([
                transforms.Lambda(lambda x: Image.fromarray(x.astype('uint8'))),
                transforms.Lambda(lambda x: ImageOps.pad(x, (max(x.size), max(x.size)))),
                RandomResizedCrop(224, scale=(0.8, 1.0)),
                RandomRotation(10),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                RandomGrayscale(p=0.1),
                transforms.RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                RandomAutocontrast(p=0.3),
                RandomEqualize(p=0.3),
                RandomPosterize(bits=4, p=0.3),
                RandomSolarize(threshold=192.0, p=0.3),
                transforms.ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.tactile_transform = transforms.Compose([
                transforms.Lambda(lambda x: Image.fromarray(x.astype('uint8'))),
                transforms.Lambda(lambda x: ImageOps.pad(x, (max(x.size), max(x.size)))),
                RandomResizedCrop(224, scale=(0.8, 1.0)),
                RandomRotation(10),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),  # Gaussian noise injection
                Normalize(mean=tactile_mean, std=tactile_std)
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Lambda(lambda x: Image.fromarray(x.astype('uint8'))),
                transforms.Lambda(lambda x: ImageOps.pad(x, (max(x.size), max(x.size)))),
                transforms.Resize(224),
                transforms.ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.tactile_transform = transforms.Compose([
                transforms.Lambda(lambda x: Image.fromarray(x.astype('uint8'))),
                transforms.Lambda(lambda x: ImageOps.pad(x, (max(x.size), max(x.size)))),
                transforms.Resize(224),
                transforms.ToTensor(),
                Normalize(mean=tactile_mean, std=tactile_std)
            ])

        # Get episode lengths
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
                # Get camera images
                timestep_cam_images = []
                for cam_name in self.camera_names:
                    image = root[f'/observations/images/{cam_name}'][timestep]
                    image = self.image_transform(image)
                    timestep_cam_images.append(image)
                images = torch.stack(timestep_cam_images, axis=0)

                # Get tactile data
                tactile_data = root['observations/images/tactile'][timestep]
                tactile_data = self.tactile_transform(tactile_data)

                # Get qpos (robot position)
                position = root['observations/qpos'][timestep]
                position = (position - self.position_mean) / self.position_std
                position = torch.tensor(position[:7], dtype=torch.float32)

                all_cam_images.append(images)
                all_tactile_images.append(tactile_data)
                all_positions.append(position)

        return torch.stack(all_cam_images, axis=0), torch.stack(all_tactile_images, axis=0), torch.stack(all_positions, axis=0)

# CLIP Loss Function
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, bank_size: int, feat_dim: int, device: torch.device):
        self.bank_size = bank_size
        self.device = device
        # initialize random normalized
        self.register = F.normalize(
            torch.randn(bank_size, feat_dim, device=device),
            dim=1)
        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, emb: torch.Tensor):
        # emb: [M, D]
        M = emb.shape[0]
        if M >= self.bank_size:
            emb = emb[-self.bank_size:]
            M = self.bank_size
        idx = (self.ptr + torch.arange(M, device=self.device)) % self.bank_size
        self.register[idx] = emb.detach()
        self.ptr = (self.ptr + M) % self.bank_size

    def get_bank(self) -> torch.Tensor:
        return self.register.detach()
    
def clip_loss(image_embeddings: torch.Tensor, tactile_embeddings: torch.Tensor, target_matrix: torch.Tensor, logit_scale=1.0, visualize=False):
    n_cameras = image_embeddings.shape[2]
    batch_size = image_embeddings.shape[0]

    visualizations = []
    image_embeddings = image_embeddings.permute(0, 2, 1, 3)  # batch, camera, clip_N, clip_dim
    tactile_embeddings = tactile_embeddings.unsqueeze(1)  # batch, 1, clip_N, clip_dim
    image_logits = logit_scale * image_embeddings @ tactile_embeddings.permute(0, 1, 3, 2)
    tactile_logits = logit_scale * tactile_embeddings @ image_embeddings.permute(0, 1, 3, 2)

    if visualize:
        visualizations = image_logits[0].clone().detach().cpu().numpy()/logit_scale
    
    image_logits = image_logits.flatten(0, 1)
    tactile_logits = tactile_logits.flatten(0, 1)

    image_loss = F.cross_entropy(image_logits, target_matrix.repeat(image_logits.shape[0], 1, 1), reduce=False).mean(dim=1)
    tactile_loss = F.cross_entropy(tactile_logits, target_matrix.T.repeat(tactile_logits.shape[0], 1, 1), reduce=False).mean(dim=1)

    image_loss = image_loss.view(batch_size, n_cameras)
    tactile_loss = tactile_loss.view(batch_size, n_cameras)

    loss = ((image_loss + tactile_loss) / 2.0).mean(dim=0)

    return loss, visualizations

def multi_positive_contrastive_loss(
        image_emb: torch.Tensor,
        tactile_emb: torch.Tensor,
        memory_bank: MemoryBank = None,
        temperature: float = 0.07
    ) -> torch.Tensor:
    """
    image_emb, tactile_emb: [B, N, D]
    Positives: same frame and next frame as weak positive.
    Negative samples: from memory bank if provided.
    Returns scalar loss.
    """
    B, N, D = image_emb.shape
    # flatten
    img = F.normalize(image_emb.view(-1, D), dim=1)    # [B*N, D]
    tac = F.normalize(tactile_emb.view(-1, D), dim=1)  # [B*N, D]
    # compute positive logits
    # primary pos = same frame
    pos_primary = torch.sum(img * tac, dim=1, keepdim=True) / temperature
    # secondary pos = next frame (cyclic)
    idx = torch.arange(B*N, device=img.device)
    sec_idx = idx//N * N + ((idx % N + 1) % N)
    pos_secondary = torch.sum(img * tac[sec_idx], dim=1, keepdim=True) / temperature
    pos_logits = torch.cat([pos_primary, pos_secondary], dim=1)  # [BN, 2]
    # negative logits
    if memory_bank is not None:
        neg_bank = memory_bank.get_bank()       # [M, D]
        neg_logits = img @ neg_bank.t() / temperature  # [BN, M]
        logits = torch.cat([pos_logits, neg_logits], dim=1)
    else:
        # only use batch negatives excluding prim/sec
        sim = img @ tac.t() / temperature      # [BN, BN]
        # mask out prim/sec
        mask = torch.ones_like(sim, dtype=bool)
        mask[idx, idx] = False
        mask[idx, sec_idx] = False
        neg_logits = sim[mask].view(B*N, -1)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
    # labels: positive positions are 0 and 1; use cross_entropy by treating label=0
    labels = torch.zeros(B*N, dtype=torch.long, device=img.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# CLIP pretraining function
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
    vit_lr: float = 1e-5, 
    projection_lr: float = 1e-4
):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]
    dataset: ClipDataset = train_loader.dataset
    n_cameras = dataset.n_cameras
    state_size = 7  # Joint state size

    # Load vision encoder (CLIP ViT model)
    vision_encoder = load_vit_encoder(local_model_path=VIT_MODEL_PATH).to(device)
    tactile_encoder = load_vit_encoder(local_model_path=VIT_MODEL_PATH).to(device)
    
    # Freeze vision encoder parameters
    for param in vision_encoder.parameters():
        param.requires_grad = False

    # Projection heads for vision and tactile
    vision_projection = ClipProjectionHead(out_dim=clip_dim, pool='attention').to(device)
    tactile_projection = ClipProjectionHead(out_dim=clip_dim, conditioning_dim=state_size, pool='attention').to(device)

    # Create tactile encoder (ResNet18 or any other model)

    # Optimizer only for tactile encoder and projection heads
    optim_params = [
        {"params": tactile_encoder.parameters(), "lr": vit_lr},
        {"params": tactile_projection.parameters(), "lr": projection_lr},
        {"params": vision_projection.parameters(), "lr": projection_lr},
    ]

    optimizer, scheduler = setup_optimizer_and_scheduler(optim_params, lr=vit_lr, weight_decay=1e-4, n_epochs=n_epochs)
    # optimizer = torch.optim.AdamW(optim_params)

    # training_losses = np.empty([n_epochs, n_cameras])
    # testing_losses = np.empty([n_epochs, n_cameras])
    
    memory_bank = MemoryBank(bank_size=4096, feat_dim=clip_dim, device=device)
    writer = SummaryWriter(log_dir=f"{save_dir}/runs")
    
    global_step = 0
    for epoch in tqdm(range(n_epochs)):
    # train the model
        training_loss = np.zeros(n_cameras)

        tactile_encoder.train()
        tactile_projection.train()
        # vision_encoder.train()
        vision_encoder.eval()
        vision_projection.train()
        
        for batch_idx, (images, tactile, position) in enumerate(train_loader):
            images = images.to(device)
            tactile = tactile.to(device)
            position = position.to(device)

            # forward pass
            
            batch_size = images.shape[0]
            clip_N = images.shape[1]
            # images are in form batch, clip_N, camera, c, h, w. We want to flatten the batch and camera dimensions
            images = images.view(-1, images.shape[3], images.shape[4], images.shape[5])
            vision_output = vision_encoder(images)
            vision_tokens = vision_output.last_hidden_state
            image_embeddings = vision_projection(vision_tokens)
            # vision_cls_token = vision_output.last_hidden_state[:, 0, :]
            
            tactile = tactile.view(-1, tactile.shape[2], tactile.shape[3], tactile.shape[4])
            tactile_output = tactile_encoder(tactile)
            tactile_tokens = tactile_output.last_hidden_state
            position = position.view(-1, position.shape[2])
            tactile_embeddings = tactile_projection(tactile_tokens, position)
            # tactile_cls_token = tactile_output.last_hidden_state[:, 0, :]
            
            # image_embeddings = vision_projection(vision_encoder(images))
            # now reshape the image_embeddings to be batch, clip_N, camera, clip_dim
            image_embeddings = image_embeddings.view(batch_size, clip_N, n_cameras, clip_dim)
            

            # flatten the batch and clip_N dimensions
            # tactile_embeddings = tactile_projection(tactile_encoder(tactile), position)
            # reshape the tactile_embeddings to be batch, clip_N, clip_dim
            tactile_embeddings = tactile_embeddings.view(batch_size, clip_N, clip_dim)
            
            B, N, C, D = image_embeddings.shape
            img_flat = image_embeddings.view(-1, N, D)  # [B*cams, N, D]
            tac_flat = tactile_embeddings.unsqueeze(1).expand(-1, C, -1, -1)
            tac_flat = tac_flat.reshape(-1, N, D)  # [B*cams, N, D]
            
            
            # calculate target matrix if using clip loss
            # target_matrix = torch.eye(clip_N).to(device)
            # calculate loss - vector of per-camera losses
            # if batch_idx == 0 and epoch%plot_freq == 0: # visualize the first batch in each epoch
            #     loss, plot_maps = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=True)
            #     try:
            #         for cam_num, plot_map in enumerate(plot_maps):
            #             plt.figure()
            #             plt.imshow(plot_map)
            #             plt.colorbar()
            #             plt.title(f'Average Softmax Map, Epoch {epoch}, Cam {cam_num} - Train')
            #             plt.savefig(f'{save_dir}/graphs/epoch_{epoch}_cam_{cam_num}_train.png')
            #             plt.close()
            #     except:
            #         print('Error in train plots')
            #         raise
            # else:
            #     loss, _ = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=False)
                # loss_vect, _ = clip_loss_vectorized(image_embeddings, tactile_embeddings, target_matrix, visualize=False)
                # print('loss diff', loss - loss_vect, (loss - loss_vect).mean())
                # assert torch.allclose(loss, loss_vect, atol=1e-5), f'loss: {loss}, loss_vect: {loss_vect}'
            
            # Multi-positive contrastive loss using memeory bank
            loss = multi_positive_contrastive_loss(img_flat, tac_flat, memory_bank)
            # training_loss += loss.clone().detach().cpu().numpy()
            optimizer.zero_grad()
            # loss.mean().backward()
            loss.backward()
            optimizer.step()
            scheduler.step()
            memory_bank.enqueue(tac_flat.view(-1, D))
            
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), global_step=global_step)

            # # clamp the logit scale to be between 0.05 and 100
            # logit_scale.data = torch.clamp(logit_scale.data, 0.05, 100)

        # training_losses[epoch] = training_loss/len(train_loader)

        # test the model
        tactile_encoder.eval()
        tactile_projection.eval()
        vision_encoder.eval()
        vision_projection.eval()

        # test_loss = np.zeros(n_cameras)
        with torch.no_grad():
            for batch_idx, (images, tactile, position) in enumerate(test_loader):
                images = images.to(device)
                tactile = tactile.to(device)
                position = position.to(device)

                # forward pass
                
                batch_size = images.shape[0]
                clip_N = images.shape[1]
                # images are in form batch, clip_N, camera, c, h, w. We want to flatten the batch and camera dimensions
                images = images.view(-1, images.shape[3], images.shape[4], images.shape[5])
                vision_output = vision_encoder(images)
                vision_tokens = vision_output.last_hidden_state
                image_embeddings = vision_projection(vision_tokens)
                # vision_cls_token = vision_output.last_hidden_state[:, 0, :]
                
                tactile = tactile.view(-1, tactile.shape[2], tactile.shape[3], tactile.shape[4])
                tactile_output = tactile_encoder(tactile)
                tactile_tokens = tactile_output.last_hidden_state
                position = position.view(-1, position.shape[2])
                tactile_embeddings = tactile_projection(tactile_tokens, position)
                # tactile_cls_token = tactile_output.last_hidden_state[:, 0, :]
                
                # image_embeddings = vision_projection(vision_encoder(images))
                # now reshape the image_embeddings to be batch, clip_N, camera, clip_dim
                image_embeddings = image_embeddings.view(batch_size, clip_N, n_cameras, clip_dim)
                

                # flatten the batch and clip_N dimensions
                # tactile_embeddings = tactile_projection(tactile_encoder(tactile), position)
                # reshape the tactile_embeddings to be batch, clip_N, clip_dim
                tactile_embeddings = tactile_embeddings.view(batch_size, clip_N, clip_dim)
                
                B, N, C, D = image_embeddings.shape
                img_flat = image_embeddings.view(-1, N, D)  # [B*cams, N, D]
                tac_flat = tactile_embeddings.unsqueeze(1).expand(-1, C, -1, -1)
                tac_flat = tac_flat.reshape(-1, N, D)  # [B*cams, N, D]
                
                loss = multi_positive_contrastive_loss(img_flat, tac_flat, memory_bank)
                
                if writer is not None:
                    writer.add_scalar("Loss/test", loss.item(), global_step=global_step)

                # calculate loss - vector of per-camera losses
                            # calculate loss - vector of per-camera losses
                # if batch_idx == 0 and epoch%plot_freq == 0: # visualize the first batch in each epoch
                #     loss, plot_maps = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=True)
                #     try:
                #         for cam_num, plot_map in enumerate(plot_maps):
                #             plt.figure()
                #             plt.imshow(plot_map)
                #             plt.colorbar()
                #             plt.title(f'Average Softmax Map, Epoch {epoch}, Cam {cam_num} - Test')
                #             plt.savefig(f'{save_dir}/graphs/epoch_{epoch}_cam_{cam_num}_test.png')
                #             plt.close()
                #     except:
                #         print('Error in test plots')
                #         raise
                # else:
                #     loss, _ = clip_loss(image_embeddings, tactile_embeddings, target_matrix, visualize=False)
                # test_loss += loss.clone().detach().cpu().numpy()
        # testing_losses[epoch] = test_loss/len(test_loader)


        # plot the training and testing losses
        # if epoch%plot_freq == 0:
        #     plt.figure()
        #     for i in range(n_cameras):
        #         plt.plot(training_losses[:epoch+1, i], label=f'camera {i+1} train', c=f'C{i}')
        #         plt.plot(testing_losses[:epoch+1, i], label=f'camera {i+1} test', linestyle='dashed', c=f'C{i}')
        #     plt.legend(loc='best')
        #     plt.title(f'Training and Testing Loss - Epoch {epoch+1}/{n_epochs}')
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Loss')
        #     plt.savefig(f'{save_dir}/graphs/training_loss.png')
        #     plt.close()

        # # save the losses as a np file
        # np.save(f'{save_dir}/graphs/training_losses.npy', training_losses)
        # np.save(f'{save_dir}/graphs/testing_losses.npy', testing_losses)

        # save the models
        if (epoch+1) % save_freq == 0:
            torch.save(vision_encoder.state_dict(), f'{save_dir}/epoch_{epoch}_vision_encoder.pth')
            torch.save(vision_projection.state_dict(), f'{save_dir}/epoch_{epoch}_vision_projection.pth')
            torch.save(tactile_encoder.state_dict(), f'{save_dir}/epoch_{epoch}_tactile_encoder.pth')
            torch.save(tactile_projection.state_dict(), f'{save_dir}/epoch_{epoch}_tactile_projection.pth')

def run_clip_pretraining(
    dataset_dir:str = None, 
    save_dir:str = "./", 
    num_episodes:int = 200, 
    camera_names:list = ['front'], 
    use_existing:bool = False, 
    batch_size:int = 3, 
    n_clip_images:int = 5, 
    min_distance:int = 20, 
    n_epochs:int = 1500, 
    plot_freq:int = 50, 
    save_freq:int = 10, 
    features_per_group:int = 16,
    clip_dim:int = 512,
    vit_lr:float = 1e-5,
    projection_lr:float = 1e-4,
    data_augmentation:bool = False
):
    from utils import get_norm_stats
    if not os.path.exists(dataset_dir):
        raise ValueError("Dataset directory does not exist")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_existing=use_existing)
    batch_size_train = batch_size
    batch_size_test = batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    train_dataset = ClipDataset(train_indices, dataset_dir, camera_names, norm_stats, n_images=n_clip_images, min_distance=min_distance, data_augmentation=data_augmentation)
    test_dataset = ClipDataset(val_indices, dataset_dir, camera_names, norm_stats, n_images=n_clip_images, min_distance=min_distance, data_augmentation=data_augmentation)

    if device == torch.device("cuda"):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=10, prefetch_factor=10, pin_memory_device='cuda')
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, pin_memory=True, num_workers=10, prefetch_factor=10, pin_memory_device='cuda')
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

    # create directory to save models and plots
    # get all folders in the clip_models directory
    ns = [-1]
    for folder in os.listdir(save_dir):
        ns.append(int(folder))

    n = max(ns) + 1
    os.makedirs(f'{save_dir}/{n}')
    os.makedirs(f'{save_dir}/{n}/graphs')
    os.makedirs(f'{save_dir}/{n}/runs')

    # save run stats:
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
        
    clip_pretraining(
        train_dataloader, 
        test_dataloader, 
        device, 
        save_dir=f'{save_dir}/{n}', 
        save_freq=save_freq,
        plot_freq=plot_freq,
        clip_dim=clip_dim, 
        features_per_group=features_per_group, 
        n_epochs=n_epochs,
        vit_lr=vit_lr,
        projection_lr=projection_lr
    )

def replot_loss_graph(training_losses, testing_losses):
    """
    Plot the training and testing losses from the saved npy files.
    Applies a running average to smooth the losses.
    """
    # training_losses: N X cameras
    # testing_losses: N X cameras
    from matplotlib import pyplot as plt

    total_train = training_losses.mean(axis=1)
    total_test = testing_losses.mean(axis=1)

    # smooth the losses (running average)
    window_size = 10
    smooth_train =  np.zeros_like(total_train)
    smooth_test =  np.zeros_like(total_test)
    for i in range(len(total_train)):
        if i < window_size:
            smooth_train[i] = total_train[:i].mean()
            smooth_test[i] = total_test[:i].mean()
        else:
            smooth_train[i] = total_train[i-window_size:i].mean()
            smooth_test[i] = total_test[i-window_size:i].mean()


    plt.figure()
    plt.plot(smooth_train, label=f'Training loss', c='r')
    plt.plot(smooth_test, label=f'Testing loss', c='b')
    plt.legend(loc='best')
    plt.title(f'Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--dataset_dir', type=str, default='')
    args_parser.add_argument('--save_dir', type=str, default='')
    args_parser.add_argument('--num_episodes', type=int, default=1000)
    args_parser.add_argument('--camera_names', type=list, default=['front'])
    args_parser.add_argument('--use_existing', type=bool, default=True)
    args_parser.add_argument('--batch_size', type=int, default=45) # 128
    args_parser.add_argument('--n_clip_images', type=int, default=5)
    args_parser.add_argument('--min_distance', type=int, default=20)
    args_parser.add_argument('--save_freq', type=int, default=100)
    args_parser.add_argument('--plot_freq', type=int, default=50)
    args_parser.add_argument('--n_epochs', type=int, default=5000)
    args_parser.add_argument('--vit_lr', type=float, default=1e-4)
    args_parser.add_argument('--projection_lr', type=float, default=1e-4)
    args_parser.add_argument('--clip_dim', type=int, default=512)
    args_parser.add_argument('--features_per_group', type=int, default=64)
    args_parser.add_argument('--data_augmentation', type=bool, default=True)
    
    
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
        vit_lr=args.vit_lr,
        projection_lr=args.projection_lr,
        clip_dim=args.clip_dim,
        features_per_group=args.features_per_group,
        data_augmentation=args.data_augmentation
    )