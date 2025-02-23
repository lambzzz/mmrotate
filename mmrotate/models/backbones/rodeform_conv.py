import torch
import torch.nn as nn
import math

from mmcv.cnn import CONV_LAYERS
from mmcv.ops import DeformConv2d, deform_conv2d
from mmcv.utils import print_log

from torch import Tensor
from torch.nn.modules.utils import _pair



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

    def __init__(self, *args, use_rotate=True, use_expand=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rotate = use_rotate
        self.use_expand = use_expand
        if self.use_rotate:
            self.conv_theta = nn.Conv2d(
                self.in_channels,
                self.deform_groups,
                kernel_size=self.kernel_size,
                stride=_pair(self.stride),
                padding=_pair(self.padding),
                dilation=_pair(self.dilation),
                bias=True)
            
            # self.conv_sin = nn.Conv2d(
            #     self.in_channels,
            #     self.deform_groups,
            #     kernel_size=self.kernel_size,
            #     stride=_pair(self.stride),
            #     padding=_pair(self.padding),
            #     dilation=_pair(self.dilation),
            #     bias=True)
            # self.conv_cos = nn.Conv2d(
            #     self.in_channels,
            #     self.deform_groups,
            #     kernel_size=self.kernel_size,
            #     stride=_pair(self.stride),
            #     padding=_pair(self.padding),
            #     dilation=_pair(self.dilation),
            #     bias=True)

            self.act_func_theta = nn.Tanh()
        else:
            self.conv_theta = None
        if self.use_expand:
            self.conv_alpha = nn.Conv2d(
                self.in_channels,
                self.deform_groups,
                kernel_size=self.kernel_size,
                stride=_pair(self.stride),
                padding=_pair(self.padding),
                dilation=_pair(self.dilation),
                bias=True)
            self.act_func_alpha = nn.Sigmoid()
        else:
            self.conv_alpha = None

        self.init_offset()

    def init_offset(self):
        if self.use_rotate:
            self.conv_theta.weight.data.zero_()
            self.conv_theta.bias.data.zero_()
            # self.conv_sin.weight.data.fill_(0)
            # self.conv_cos.weight.data.fill_(0)
            # self.conv_sin.bias.data.fill_(0)
            # self.conv_cos.bias.data.fill_(5)
        if self.use_expand:
            self.conv_alpha.weight.data.zero_()
            self.conv_alpha.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        theta = self.conv_theta(x)
        theta = self.act_func_theta(theta) * math.pi / 2
        alphas = self.conv_alpha(x)
        alphas = self.act_func_alpha(alphas) * 2
        # theta = alphas.new_zeros(alphas.size(0), self.deform_groups, alphas.size(2), alphas.size(3))
        # alphas = theta.new_ones(theta.size(0), self.deform_groups, theta.size(2), theta.size(3))
        offset = _get_offset(theta, alphas)

        # sin_theta = self.conv_sin(x)
        # cos_theta = self.conv_cos(x)
        # sin_theta = self.act_func_theta(sin_theta)
        # cos_theta = torch.sigmoid(cos_theta)
        # theta = torch.cat([sin_theta, cos_theta], dim=1)
        # alphas = theta.new_ones(theta.size(0), self.deform_groups, theta.size(2), theta.size(3))
        # offset = _get_offset_sincos(theta, alphas)
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


origin_coors = torch.tensor(
    [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0],
        [1, 1]],
    dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu').reshape(1, 18, 1, 1)

def _get_offset(theta: Tensor, alphas: Tensor) -> Tensor:
    """Get offset from theta and alphas.

    Args:
        theta (Tensor): The theta tensor with shape (B, G, H, W).
        alphas (Tensor): The alphas tensor with shape (B, G, H, W).

    Returns:
        Tensor: The offset tensor with shape (B, G*18, H, W).
    """
    B, G, H, W = theta.size()
    origin_coors_expanded = origin_coors.repeat(1, G, 1, 1) # shape: [1, 18, 1, 1] -> [1, G*18, 1, 1]
    alphas = alphas.reshape(B, G, 1, H, W).repeat(1, 1, 18, 1, 1).reshape(B, G*18, H, W)
    coors = torch.mul(alphas, origin_coors_expanded) # shape: [B, G*18, H, W]
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

