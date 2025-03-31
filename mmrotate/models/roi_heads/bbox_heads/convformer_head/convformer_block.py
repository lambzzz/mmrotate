import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d, deform_conv2d

from .delta2theta import delta2theta
from .van_block import VANBlock


class LayerReg(nn.Module):
    """x has shape (B, C, H, W)

    reshape -> B, C, A, A
    conv3x3 -> B, C, (A - 2), (A - 2)
    conv3x3 -> B, C, (A - 4), (A - 4)
    conv3x3 -> B, C, (A - 6), (A - 6)
    flatten -> B, F
    fc      -> B, out_channels
    """
    def __init__(self, in_channels=256, out_channels=2, reg_channels=256, num_convs=0, feat_size=7):
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
    

class Block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 reg_channels,
                 num_convs=0,
                 predict_cfg="",
                 complement_cfg=""):
        
        super().__init__()
        predict_channels = len(predict_cfg)
        self.predict_cfg = predict_cfg
        self.complement_cfg = complement_cfg

        if predict_channels > 0:
            self.reg_branch = LayerReg(in_channels=in_channels, 
                                    out_channels=predict_channels, 
                                    reg_channels=reg_channels,
                                    num_convs=num_convs)

        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.norm = nn.GroupNorm(32, out_channels)
        # self.actfunc = nn.ReLU(True)
        self.mixer = VANBlock(in_channels, mlp_ratio=4)
        


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

        # theta_c = delta2theta(rois=rois[:, 1:], deltas=deltas, rois_mode='rbbox')
        # theta_c = theta_c.reshape(-1, 2, 3)

        # mask activation
        # x_ones = torch.ones([B, C, W, H], dtype=x.dtype, device=x.device, requires_grad=False)
        # grid = F.affine_grid(theta_c, x_ones.size())                    # 
        # grid = grid.type(x_ones.type())                                 # avoid fp16/fp32 confusion
        # actv_mask = F.grid_sample(x_ones, grid) # + x_ones * 0.1        # 

        # # feature alignment
        # grid = F.affine_grid(theta_c, x.size(), align_corners=False)
        # x_aligned = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)


        # x = self.conv(x)
        # x = self.norm(x)
        # x = self.actfunc(x)
        x = self.mixer(x)

        return x, bbox_pr
    


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

        self.origin_coors = torch.tensor(
            [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0],
                [1, 1]],
            dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.origin_coors_x = self.origin_coors[:, 1].reshape(1, 9, 1, 1)
        self.origin_coors_y = self.origin_coors[:, 0].reshape(1, 9, 1, 1)
        self.origin_coors = self.origin_coors.reshape(1, 18, 1, 1)

    def forward(self, x: Tensor, offset: Tensor) -> Tensor:  # type: ignore

        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups,
                             False, self.im2col_step)


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