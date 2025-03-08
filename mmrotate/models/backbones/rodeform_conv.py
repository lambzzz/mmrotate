import torch
import torch.nn as nn
import math
import einops

from mmcv.cnn import CONV_LAYERS
from mmcv.ops import DeformConv2d, deform_conv2d
from mmcv.utils import print_log

from torch import Tensor
from torch.nn.modules.utils import _pair

class StripBlock(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels = 3, 
                 kernel_size = 5,
                 strip_size1 = 1,
                 strip_size2 = 19,
                 stride = 1,):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size, 
                               stride=stride, padding=kernel_size//2, groups=in_channels)
        self.conv_spatial1 = nn.Conv2d(in_channels,in_channels,kernel_size=(strip_size1, strip_size2), 
                                       stride=1, padding=(strip_size1//2, strip_size2//2), groups=in_channels)     
        self.conv_spatial2 = nn.Conv2d(in_channels,in_channels,kernel_size=(strip_size2, strip_size1), 
                                       stride=1, padding=(strip_size2//2, strip_size1//2), groups=in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

        self.init_weights()

    def init_weights(self):
        self.conv0.weight.data.zero_()
        self.conv_spatial1.weight.data.zero_()
        self.conv_spatial2.weight.data.zero_()
        self.conv1.weight.data.zero_()
        self.conv0.bias.data.zero_()
        self.conv_spatial1.bias.data.zero_()
        self.conv_spatial2.bias.data.zero_()
        self.conv1.bias.data.zero_()

    def forward(self, x):   
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)

        return attn

class LayerNormProxy(nn.Module):
    # copy from https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class routing_function(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels = 3, 
                 kernel_size = 5,
                 stride = 1,
                 bias = True,
                 deform_groups = 1):
        super(routing_function, self).__init__()

        # pass in stride for compatibility with downsampled feature maps
        # self.DWConv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels, bias=bias)
        # self.norm = LayerNormProxy(in_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.PWConv = nn.Conv2d(in_channels, deform_groups*out_channels, kernel_size=1, bias=bias)

        self.Conv = nn.Conv2d(in_channels, deform_groups*out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=bias)
        self.DialConv = nn.Conv2d(out_channels, out_channels, 5, padding=4, dilation=2)

        # self.StripConv = StripBlock(in_channels, out_channels, kernel_size, 1, 19, stride)

        self.actfunc_theta = nn.Tanh()
        self.actfunc_alpha = nn.Sigmoid()

        self.init_offset()

    def init_offset(self):
        # self.DWConv.weight.data.zero_()
        # self.PWConv.weight.data.zero_()

        # self.DWConv.bias.data.zero_()
        # self.PWConv.bias.data.zero_()

        self.Conv.weight.data.zero_()
        self.Conv.bias.data.zero_()

        self.DialConv.weight.data.zero_()
        self.DialConv.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor: 

        # x = self.DWConv(x)
        # x = self.norm(x)
        # x = self.relu(x)
        # x = self.PWConv(x)

        x = self.Conv(x)
        x_ = self.DialConv(x)
        x = x + x_

        # x = self.StripConv(x)

        theta, alpha_x, alpha_y = torch.split(x, 1, dim=1)
        theta = self.actfunc_theta(theta) * math.pi / 2
        alpha_x = self.actfunc_alpha(alpha_x) * 2
        alpha_y = self.actfunc_alpha(alpha_y) * 2
        return theta, alpha_x, alpha_y



@CONV_LAYERS.register_module('RDCN')
class DeformConv2dPack(DeformConv2d):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
    The spatial arrangement is like:

    .. code:: text

        (x0, y0) (x1, y1) (x2, y2)
        (x3, y3) (x4, y4) (x5, y5)
        (x6, y6) (x7, y7) (x8, y8)

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing_func = routing_function(self.in_channels, stride=self.stride, deform_groups=self.deform_groups)

        self.origin_coors = torch.tensor(
            [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0],
                [1, 1]],
            dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.origin_coors_x = self.origin_coors[:, 1].reshape(1, 9, 1, 1)
        self.origin_coors_y = self.origin_coors[:, 0].reshape(1, 9, 1, 1)
        self.origin_coors = self.origin_coors.reshape(1, 18, 1, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        theta, alphas_x, alphas_y = self.routing_func(x)
        offset = self._get_offset(theta, alphas_x, alphas_y)

        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups,
                             False, self.im2col_step)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, DeformConvPack loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'DeformConv2dPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)



    def _get_offset(self, theta: Tensor, alphas_x: Tensor, alpha_y: Tensor) -> Tensor:
        """Get offset from theta and alphas.

        Args:
            theta (Tensor): The theta tensor with shape (B, G, H, W).
            alphas_x (Tensor): The alphas_x tensor with shape (B, G, H, W).
            alpha_y (Tensor): The alpha_y tensor with shape (B, G, H, W).

        Returns:
            Tensor: The offset tensor with shape (B, G*18, H, W).
        """
        B, G, H, W = theta.size()
        origin_coors_expanded = self.origin_coors.repeat(1, G, 1, 1) # shape: [1, 18, 1, 1] -> [1, G*18, 1, 1]
        origin_coors_expanded_x = self.origin_coors_x.repeat(1, G, 1, 1)
        origin_coors_expanded_y = self.origin_coors_y.repeat(1, G, 1, 1)
        alphas_x = alphas_x.reshape(B, G, 1, H, W).repeat(1, 1, 9, 1, 1).reshape(B, G*9, H, W)    # [B, G, H, W] -> [B, G, 1, H, W] -> [B, G, 9, H, W] -> [B, G*9, H, W]
        alphas_y = alpha_y.reshape(B, G, 1, H, W).repeat(1, 1, 9, 1, 1).reshape(B, G*9, H, W)    # [B, G, H, W] -> [B, G, 1, H, W] -> [B, G, 9, H, W] -> [B, G*9, H, W]
        coors_x = torch.mul(alphas_x, origin_coors_expanded_x) # shape: [B, G*9, H, W]
        coors_y = torch.mul(alphas_y, origin_coors_expanded_y) # shape: [B, G*9, H, W]
        coors = torch.stack((coors_y, coors_x), dim=2).reshape(B, G*18, H, W)   # shape: [B, G*18, H, W]

        coors = coors.permute(0, 2, 3, 1).reshape(-1, 9, 2) # [B, H, W, G*18] -> [B*H*W*G, 9, 2]
        rotated_coors = _batch_rotate_coordinates(coors, theta.permute(0, 2, 3, 1).reshape(-1))
        rotated_coors = rotated_coors.reshape(B, H, W, G*18).permute(0, 3, 1, 2).contiguous()   # [B*H*W*G, 9, 2] -> [B, G*18, H, W]
        offset = rotated_coors - origin_coors_expanded
        
        return offset

def _batch_rotate_coordinates(coors: Tensor, angles: Tensor) -> torch.Tensor:
    """Rotate coordinates by a given angle.

    Args:
        coors (torch.Tensor): The coordinates tensor with shape (B, N, 2).
        angles (torch.Tensor): The rotation angles tensor with shape (B,).

    Returns:
        torch.Tensor: The rotated coordinates.
    """
    batch_size = coors.size(0)
    num_points = coors.size(1)

    # Create rotation matrices for each angle in the batch
    rotation_matrices = torch.zeros((batch_size, 2, 2), dtype=torch.float32, device=coors.device)
    rotation_matrices[:, 0, 0] = torch.cos(angles)
    rotation_matrices[:, 0, 1] = -torch.sin(angles)
    rotation_matrices[:, 1, 0] = torch.sin(angles)
    rotation_matrices[:, 1, 1] = torch.cos(angles)

    # Rotate the coordinates
    rotated_coors = torch.matmul(coors, rotation_matrices)

    return rotated_coors

# def normalize_sin_cos(sin_cos):
#     sin_theta, cos_theta = sin_cos[:, 0], sin_cos[:, 1]
#     norm = torch.sqrt(sin_theta**2 + cos_theta**2 + 1e-8)  # 避免除零
#     sin_theta = sin_theta / norm
#     cos_theta = cos_theta / norm
#     return sin_theta, cos_theta

# def construct_rotation_matrix(sin_theta, cos_theta):
#     # 构造2D旋转矩阵
#     rotation_matrix = torch.stack([
#         torch.stack([cos_theta, -sin_theta], dim=-1),
#         torch.stack([sin_theta, cos_theta], dim=-1)
#     ], dim=-2)  # shape: [batch_size, 2, 2]
#     return rotation_matrix

# def _get_offset_sincos(sincos: Tensor, alphas: Tensor) -> Tensor:
#     """Get offset from theta and alphas.

#     Args:
#         sincos (Tensor): The sincos tensor with shape (B, 2, H, W).
#         alphas (Tensor): The alphas tensor with shape (B, 1, H, W).

#     Returns:
#         Tensor: The offset.
#     """
#     B, _, H, W = sincos.size()
#     coors = torch.mul(alphas, origin_coors)
#     coors = coors.permute(0, 2, 3, 1).reshape(-1, 9, 2)
#     sincos = sincos.permute(0, 2, 3, 1).reshape(-1, 2)
#     sin_theta, cos_theta = normalize_sin_cos(sincos)
#     rotation_matrix = construct_rotation_matrix(sin_theta, cos_theta)

#     rotated_coors = torch.matmul(coors, rotation_matrix)
#     rotated_coors = rotated_coors.reshape(B, H, W, 18).permute(0, 3, 1, 2).contiguous()
#     offset = rotated_coors - origin_coors
    
#     return offset

