import math
import einops
import torch
import torch.nn as nn
from .weight_init import trunc_normal_


class LayerNormProxy(nn.Module):
    # copy from https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    

class RountingFunction(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_rate=0.2, proportion=40.0, with_chanatten=True):
        super().__init__()
        self.out_channels = out_channels
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if with_chanatten:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc_alpha = nn.Linear(in_channels, out_channels, bias=True)
            self.act_func1 = nn.Sigmoid()
        else:
            self.dropout1 = nn.Identity()
            self.fc_alpha = nn.Identity()
            self.act_func1 = nn.Identity()

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, out_channels, bias=False)
        self.act_func2 = nn.Softsign()

        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        if with_chanatten:
            trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = self.act_func1(alphas)

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func2(angles)
        angles = angles * self.proportion

        return alphas, angles

    def extra_repr(self):
        s = (f'angle_groups={self.out_channels}')
        return s.format(**self.__dict__)
