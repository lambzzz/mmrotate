import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d, deform_conv2d
from mmcv.runner import BaseModule, ModuleList, Sequential

from .delta2theta import delta2theta
from .van_block import VANBlock


class LayerReg(BaseModule):
    """x has shape (B, C, H, W)

    reshape -> B, C, A, A
    conv3x3 -> B, C, (A - 2), (A - 2)
    conv3x3 -> B, C, (A - 4), (A - 4)
    conv3x3 -> B, C, (A - 6), (A - 6)
    flatten -> B, F
    fc      -> B, out_channels
    """
    def __init__(self, 
                 in_channels=256, 
                 out_channels=2, 
                 reg_channels=256, 
                 num_convs=0, 
                 feat_size=7):
        super().__init__()
        self.num_convs = num_convs
        self.feat_size = feat_size
        self.with_avg_pool = False
        self.double_ConvCh = False

        if self.num_convs > 0:
            self.norms = nn.ModuleList()
            self.convs = nn.ModuleList()
            self.relu = nn.ReLU(True)
            for i in range(self.num_convs):
                in_ch = in_channels if i == 0 else out_ch
                # out_ch = in_ch*2 if self.double_ConvCh else in_ch
                out_ch = reg_channels
                self.norms.append(nn.GroupNorm(32, in_ch))
                self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0))
            self.last_feat_area = (feat_size - num_convs * 2) ** 2

        if self.with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # in_channels = in_channels * (2**self.num_convs) if self.double_ConvCh else in_channels
        # conv_out_channels = in_channels if self.with_avg_pool else in_channels * self.last_feat_area
        conv_out_channels = reg_channels * self.last_feat_area
        self.norm_reg = nn.LayerNorm(conv_out_channels)
        self.fc_reg = nn.Linear(conv_out_channels, out_channels)

        self.init_cfg = [
            dict(type='Kaiming', layer='Conv2d'),
            dict(type='Constant', val=1.0, bias=0.0, override=dict(type='LayerNorm')),
            dict(type='Constant', val=1.0, bias=0.0, override=dict(type='GroupNorm')),
            dict(type='Normal', std=0.001, override=dict(name='fc_reg')),
        ]

    def forward(self, x):

        if self.num_convs > 0:
            B, C, H, W = x.shape
            for i in range(self.num_convs):
                norm = self.norms[i]
                conv = self.convs[i]
                x = self.relu(norm(conv(x)))
            # x = x.reshape(B, C, self.last_feat_area).transpose(-2, -1)
        if self.with_avg_pool:
            x = self.avgpool(x)
        bbox_pr = self.fc_reg(self.norm_reg(x.flatten(1)))
        return bbox_pr
    

class Block(BaseModule):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 reg_channels,
                 num_convs=0,
                 predict_cfg="",
                 mask_cfg=False,
                 align_cfg=False):
        
        super().__init__()
        predict_channels = len(predict_cfg)
        self.predict_cfg = predict_cfg
        self.mask_cfg = mask_cfg
        self.align_cfg = align_cfg
        self.calculate_theta = mask_cfg or align_cfg

        if predict_channels > 0:
            self.reg_branch = LayerReg(in_channels=in_channels, 
                                    out_channels=predict_channels, 
                                    reg_channels=reg_channels,
                                    num_convs=num_convs)

        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.norm = nn.GroupNorm(32, out_channels)
        # self.actfunc = nn.ReLU(True)
        self.mixer = VANBlock(in_channels, mlp_ratio=2, mask_cfg=mask_cfg, align_cfg=align_cfg)
        


    def forward(self, x, rois=None, deltas=None):
        B, C, W, H = x.shape

        if len(self.predict_cfg) > 0:
            bbox_pr = self.reg_branch(x)
        else:
            bbox_pr = torch.zeros([B, 0], dtype=x.dtype, device=x.device, requires_grad=False)

        if deltas is None:
            deltas = torch.zeros([B, 5], dtype=x.dtype, device=x.device, requires_grad=False)

        if 'XY' in self.predict_cfg:
            deltas[:, :2] = bbox_pr
        if 'WH' in self.predict_cfg:
            deltas[:, 2:4] = bbox_pr
        if 'A'  in self.predict_cfg:
            deltas[:, 4:] = bbox_pr

        if not self.calculate_theta:
            theta_c = None
        else:
            theta_c = delta2theta(rois=rois[:, 1:], deltas=deltas, rois_mode='rbbox')
            theta_c = theta_c.reshape(-1, 2, 3)

        # x = self.conv(x)
        # x = self.norm(x)
        # x = self.actfunc(x)
        x = self.mixer(x, theta_c)

        return x, bbox_pr
    

