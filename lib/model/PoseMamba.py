# PoseFormer adaptation of the timm vision transformer:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
from functools import partial

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.mambablocks_my import BiSTSSMBlock


class PoseMamba(nn.Module):
    """StructMamPose with paired spatial and temporal state-space blocks."""

    def __init__(
        self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256,
        depth=6, mlp_ratio=2., drop_rate=0., drop_path_rate=0.2,
        norm_layer=None, ffn_hidden_dim=None,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        drop_rates = [rate.item() for rate in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        def make_blocks():
            return nn.ModuleList(
                BiSTSSMBlock(
                    hidden_dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=rate,
                    norm_layer=norm_layer,
                    ffn_hidden_dim=ffn_hidden_dim,
                )
                for rate in drop_rates
            )

        self.STEblocks = make_blocks()
        self.TTEblocks = make_blocks()
        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)
        self.head_1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.head_2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.proj_side = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.cross_attn = CrossAttention(embed_dim=embed_dim, num_heads=1, dropout=0)

    def get_side_view_90(self, pose):
        relative_pose = pose - pose[:, :, 0:1, :]
        return torch.stack([relative_pose[..., 2], relative_pose[..., 1]], dim=-1)

    def STE_forward(self, x):
        frames = x.shape[1]
        x = rearrange(x, 'b f n c -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, '(b f) n c -> b f n c', f=frames)
        return self.Spatial_norm(self.STEblocks[0](x))

    def TTE_foward(self, x):
        frames, joints = x.shape[1:3]
        x = rearrange(x, 'b f n c -> (b n) f c', f=frames)
        x += self.Temporal_pos_embed[:, :frames, :]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f c -> b f n c', n=joints)
        return self.Temporal_norm(self.TTEblocks[0](x))

    def ST1_foward(self, x):
        for i in range(1, 9):
            x = self.Spatial_norm(self.STEblocks[i](x))
            x = self.Temporal_norm(self.TTEblocks[i](x))
        return x

    def ST2_foward(self, x):
        for i in range(9, self.block_depth):
            x = self.Spatial_norm(self.STEblocks[i](x))
            x = self.Temporal_norm(self.TTEblocks[i](x))
        return x

    def forward(self, x):
        batch, frames, joints, _ = x.shape
        x = self.STE_forward(x)
        x = self.TTE_foward(x)
        x = self.ST1_foward(x)
        intermediate_pose = self.head_1(x)
        side = self.proj_side(self.get_side_view_90(intermediate_pose))
        x = x + self.cross_attn(x, side)
        x = self.ST2_foward(x)
        return (intermediate_pose + self.head_2(x)).view(batch, frames, joints, -1)


class CrossAttention(nn.Module):
    """Project each joint with shared weights and attend over frames."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.total_dim = 17 * embed_dim
        self.head_dim = self.total_dim // num_heads
        assert self.total_dim % num_heads == 0, "17 * embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, y, attn_mask=None):
        batch, frames, joints, embed_dim = x.shape
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.flatten(2).view(batch, frames, self.num_heads, self.head_dim)
        k = k.flatten(2).view(batch, frames, self.num_heads, self.head_dim)
        v = v.flatten(2).view(batch, frames, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, frames, joints, embed_dim)
        return self.norm(self.out_proj(out))
