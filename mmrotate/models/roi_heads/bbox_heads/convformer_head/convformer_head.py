# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32, Sequential
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from ....builder import ROTATED_HEADS
from ..rotated_bbox_head import RotatedBBoxHead
from .convformer_block import Block


@ROTATED_HEADS.register_module()
class ConvFormerHead(RotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

        ConvFormer -> ConvFormer -> ConvFormer -> cls
            \             \             \-> reg_HW
             \             \-> reg_A
              \-> reg_XY

    Args:
        num_blocks (int): Number of blocks in the head. Default: 3.
        feat_channels (int): Number of channels in the feature map. Default: 1024.
        predict_seq (list[str]): Sequence of prediction. Default: ['XY', 'A', 'WH'].
    """

    def __init__(self,
                 num_blocks=3,
                 feat_channels=256,
                 reg_channels=256,
                 predict_cfgs=['XY', 'A', 'WH'],
                 mask_cfgs=[False, False, False],
                 align_cfgs=[False, False, False],
                 *args,
                 **kwargs):
        super(ConvFormerHead, self).__init__(
            *args, 
            # with_avg_pool=True,
            **kwargs)

        # self.PWConv = nn.Conv2d(self.in_channels, feat_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [Block(in_channels=feat_channels, 
                   out_channels=feat_channels, 
                   reg_channels=reg_channels,
                   num_convs=num_blocks-i, 
                   predict_cfg=predict_cfgs[i],
                   mask_cfg=mask_cfgs[i],
                   align_cfg=align_cfgs[i]) 
            for i in range(num_blocks)]
        )

        # reconstruct fc_cls and fc_reg since input channels are changed
        self.cls_last_dim = feat_channels if self.with_avg_pool else feat_channels*self.roi_feat_area
        self.layernorm = nn.LayerNorm(self.cls_last_dim)
        # self.cls_fc = nn.Linear(self.cls_last_dim, 1024)
        self.relu = nn.ReLU(inplace=True)
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            del self.fc_reg

        self.init_cfg = [
            dict(type='Normal', std=0.01, override=dict(name='fc_cls')),
        ]
        # self.init_cfg += [
        #     dict(
        #         type='Xavier',
        #         layer='Linear',
        #         override=[
        #             dict(name='shared_fcs'),
        #             dict(name='cls_fcs'),
        #             dict(name='reg_fcs')
        #         ])
        # ]


    def forward(self, x, rois=None):
        """Forward function."""
        B, C, W, H = x.shape
        bbox_pred = torch.zeros([B, 5], dtype=x.dtype, device=x.device, requires_grad=False)

        # x = self.PWConv(x)
        for block in self.blocks:
            x, reg_result = block(x, rois, bbox_pred)
            if 'XY' in block.predict_cfg:
                bbox_pred[:, :2] = reg_result
            if 'WH' in block.predict_cfg:
                bbox_pred[:, 2:4] = reg_result
            if 'A'  in block.predict_cfg:
                bbox_pred[:, 4:] = reg_result

        x_cls = x
        if self.with_avg_pool:
            x_cls = self.avg_pool(x_cls)
        x_cls = x_cls.flatten(1)
        x_cls = self.layernorm(x_cls)
        # x_cls = self.relu(self.cls_fc(x_cls))
        
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        # bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
    