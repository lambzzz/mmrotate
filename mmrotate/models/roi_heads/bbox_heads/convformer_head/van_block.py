import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ....builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=2, groups=dim, dilation=1)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn



class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class VANBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 mlp_ratio=4., 
                 drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 mask_cfg=False,
                 align_cfg=False):
        
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        
        self.mask_cfg = mask_cfg
        self.align_cfg = align_cfg

        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, theta=None):

        x_res = x
        if theta is not None and self.mask_cfg:
            B, C, W, H = x.shape
            x_ones = torch.ones([B, 1, W, H], dtype=x.dtype, device=x.device, requires_grad=False)
            grid = F.affine_grid(theta, x_ones.size(), align_corners=False)                    # 
            grid = grid.type(x_ones.type())                                 # avoid fp16/fp32 confusion
            actv_mask = F.grid_sample(x_ones, grid, align_corners=False) # + x_ones * 0.1        # 
            x = x * actv_mask
        x = self.norm1(x)
        x = self.attn(x)
        x = x_res + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x)

        # if theta is not None and self.align_cfg:
        #     # 计算 determinant
        #     det = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
        #     # 计算逆变换的线性部分
        #     a11 = theta[:, 1, 1] / det
        #     a12 = -theta[:, 0, 1] / det
        #     a21 = -theta[:, 1, 0] / det
        #     a22 = theta[:, 0, 0] / det
        #     # 计算逆变换的平移部分
        #     linear_inv = torch.stack([
        #         torch.stack([a11, a12], dim=-1),
        #         torch.stack([a21, a22], dim=-1)
        #     ], dim=1)  # [B, 2, 2]
        #     t = theta[:, :, 2].unsqueeze(-1)  # [B, 2, 1]
        #     trans_inv = -torch.bmm(linear_inv, t).squeeze(-1)  # [B, 2]
        #     # 合并为完整的 theta_inv
        #     theta_inv = torch.cat([linear_inv, trans_inv.unsqueeze(-1)], dim=2)  # [B, 2, 3]
        #     # 应用 affine_grid 和 grid_sample
        #     grid = F.affine_grid(theta_inv, x.size(), align_corners=False)
        #     x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)


        # x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


