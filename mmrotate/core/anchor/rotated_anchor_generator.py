# Copyright (c) OpenMMLab. All rights reserved.
import torch
import math
from mmcv.utils import to_2tuple
from mmdet.core.anchor import AnchorGenerator

from .builder import ROTATED_ANCHOR_GENERATORS


@ROTATED_ANCHOR_GENERATORS.register_module()
class RealRotatedAnchorGenerator(AnchorGenerator):
    """Real rotate anchor generator for 2D anchor-based detectors.

    Horizontal bounding box represented by (x,y,w,h,theta).
    """

    def __init__(self, 
                 angles = [0],
                 **kwargs):
        self.angles = angles
        super(RealRotatedAnchorGenerator, self).__init__(**kwargs)

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [base_anchors.size(0)*len(self.angles) for base_anchors in self.base_anchors]
    

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
            ``torch.float32``.
            device (str, optional): The device the tensor will be put on.
            Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        anchors = super(RealRotatedAnchorGenerator, self).single_level_grid_priors(
            featmap_size, level_idx, dtype=dtype, device=device)

        # The correct usage is：
        #       from ..bbox.transforms import hbb2obb
        #       anchors = hbb2obb(anchors, self.angle_version)
        # instead of rudely setting the angle to all 0.
        # However, the experiment shows that the performance has decreased.
        num_anchors = anchors.size(0)
        xy = (anchors[:, 2:] + anchors[:, :2]) / 2
        wh = anchors[:, 2:] - anchors[:, :2]
        anchors = torch.tensor([], dtype=torch.float32, device=device)
        for angle in self.angles:
            theta = torch.full((num_anchors, 1), angle/180*math.pi, device=device)
            anchor = torch.cat([xy, wh, theta], dim=1)
            anchors = torch.cat([anchors, anchor], dim=0)

        return anchors


