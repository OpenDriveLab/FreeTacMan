from .attention import flash_attention
import torch
import torch.nn as nn
import numpy as np

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight.to(x.dtype)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)



class MLP(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(MLP, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.fc1 = nn.Linear(inputdim, 128)
        self.fc2 = nn.Linear(128, outputdim)
        self.relu = nn.ReLU()

    def forward(self, x ,t):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attn_mode="torch", split_attn=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        
        self.mlp = MLP(inputdim=64, outputdim=1)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k_img = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, vembedding, tembedding, t):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]; 
            vembeding(Tensor): Shape [B, L2, C]
            tembedding(Tensor): Shape [B, L3, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.q(x)
        q = self.norm_q(q)
        q = q.view(b, -1, n, d)
        k = self.k(vembedding)
        k = self.norm_k(k).view(b, -1, n, d)#这里你看下是不是也要用image的norm
        v = self.v(vembedding).view(b, -1, n, d)
        qkv = [q, k, v]
        x = flash_attention(qkv, k_lens=None, attn_mode=self.attn_mode, split_attn=self.split_attn)

        k_img = self.norm_k_img(self.k_img(tembedding)).view(b, -1, n, d)
        v_img = self.v_img(tembedding).view(b, -1, n, d)
        qkv = [q, k_img, v_img]
        img_x = flash_attention(qkv, k_lens=None, attn_mode=self.attn_mode, split_attn=self.split_attn)

        x = x.flatten(2)
        img_x = img_x.flatten(2)
        
        p=self.mlp(x, t)
        x = p*x + (1-p)*img_x
        return x