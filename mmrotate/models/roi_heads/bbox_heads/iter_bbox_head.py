# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from ...builder import ROTATED_HEADS
from .rotated_bbox_head import RotatedBBoxHead

# import os 
# os.environ['CUDA_LAUNCH_BLOCKING']='1'


@ROTATED_HEADS.register_module()
class IterBBoxHead(RotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg

    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 stage = 0,
                 *args,
                 **kwargs):
        super(IterBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.num_instance = 2
        self.stage = stage
        self.additional_channels = self.fc_out_channels if self.stage == 1 else 0

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
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
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    (last_layer_dim + self.additional_channels) if i == 0 else self.fc_out_channels)  # in_channel: 7*7*256 + 1024
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x, last_feats = None):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            if self.stage == 0:
                x = x.flatten(1)
            else:
                x = torch.cat([x.flatten(1), last_feats.detach()], 1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, x

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat: bool = True) -> tuple:
        
        labels = []
        bbox_targets = []
        bbox_weights = []
        label_weights = []
        if self.stage == 0:
            for i in range(len(sampling_results)):
                sample_bboxes = torch.cat([
                    sampling_results[i].pos_gt_bboxes,
                    sampling_results[i].neg_gt_bboxes
                ])
                sample_priors = sampling_results[i].priors
                # sample_priors = sample_priors.repeat(1, self.num_instance).reshape(
                #     -1, 5)
                sample_bboxes = sample_bboxes.reshape(-1, 10)

                if not self.reg_decoded_bbox:
                    _bbox_targets = self.bbox_coder.encode(sample_priors,
                                                            sample_bboxes[:, 0:5])
                else:
                    # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                    # is applied directly on the decoded bounding boxes, both
                    # the predicted boxes and regression targets should be with
                    # absolute coordinate format.
                    _bbox_targets = sample_priors
                # _bbox_targets = _bbox_targets.reshape(-1, self.num_instance * 5)
                _bbox_weights = _bbox_targets.new_ones(_bbox_targets.shape)
                _labels = torch.cat([
                    sampling_results[i].pos_gt_labels,
                    sampling_results[i].neg_gt_labels
                ])
                _labels = _labels[:, 0]
                _labels_weights = _labels.new_ones(_labels.shape)

                bbox_targets.append(_bbox_targets)
                bbox_weights.append(_bbox_weights)
                labels.append(_labels)
                label_weights.append(_labels_weights)

        else:
            for i in range(len(sampling_results)):
                sample_bboxes = torch.cat([
                    sampling_results[i].pos_gt_bboxes,
                    sampling_results[i].neg_gt_bboxes
                ])
                num_samples = sample_bboxes.size(0)
                _labels = torch.cat([
                    sampling_results[i].pos_gt_labels,
                    sampling_results[i].neg_gt_labels
                ])
                pos_mask = _labels[:, 1] == 1
                num_pos = pos_mask.nonzero().size(0)

                num_expected_neg = (num_pos+1)*7
                if num_expected_neg + num_pos > num_samples:
                    num_expected_neg = num_samples - num_pos
                    neg_mask = ~pos_mask
                else:
                    neg_mask = random_choice((~pos_mask).nonzero(), num_expected_neg)
                pos_mask = pos_mask.nonzero()
                mask = torch.cat([pos_mask, neg_mask]).reshape(-1)
                
                sample_priors = sampling_results[i].priors[mask]
                # sample_priors = sample_priors.repeat(1, self.num_instance).reshape(
                #     -1, 5)
                sample_bboxes = sample_bboxes.reshape(-1, 10)[mask]

                if not self.reg_decoded_bbox:
                    _bbox_targets = self.bbox_coder.encode(sample_priors,
                                                            sample_bboxes[:,5:10])
                else:
                    # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                    # is applied directly on the decoded bounding boxes, both
                    # the predicted boxes and regression targets should be with
                    # absolute coordinate format.
                    _bbox_targets = sample_priors

                _bbox_weights = _bbox_targets.new_ones(_bbox_targets.shape)

                _labels = torch.cat([_labels[mask, 1:2], mask.reshape(-1, 1)], 1)
                # _labels = _labels[:, 1]
                _labels_weights = _labels.new_ones(_labels.shape[0])

                bbox_targets.append(_bbox_targets)
                bbox_weights.append(_bbox_weights)
                labels.append(_labels)
                label_weights.append(_labels_weights)


        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, 
             cls_score: Tensor, 
             bbox_pred: Tensor, 
             rois: Tensor,
             labels: Tensor, 
             label_weights: Tensor, 
             bbox_targets: Tensor,
             bbox_weights: Tensor, 
             **kwargs) -> dict:
        
        losses = dict()
        labels = labels.long()
        if self.stage == 0:
            if cls_score is not None:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                if cls_score.numel() > 0:
                    loss_cls_ = self.loss_cls(
                        cls_score,
                        labels,
                        label_weights,
                        avg_factor=avg_factor)
                    if isinstance(loss_cls_, dict):
                        losses.update(loss_cls_)
                    else:
                        losses['loss_cls'] = loss_cls_

            if bbox_pred is not None:
                pos_inds = labels > 0
                # do not perform bounding box regression for BG anymore.
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0))

        # stage 1 
        else:
            if cls_score is not None:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                if cls_score.numel() > 0:
                    if len(labels.shape) > 1:
                        mask = labels[:, 1]
                        cls_score = cls_score[mask, :]
                        labels = labels[:, 0].clone()
                        bbox_pred = bbox_pred[mask, :].clone()

                    loss_cls_ = self.loss_cls(
                        cls_score,
                        labels,
                        label_weights,
                        avg_factor=avg_factor
                        # avg_factor=len(pos_inds.nonzero())*4+1
                        )
                    if isinstance(loss_cls_, dict):
                        losses.update(loss_cls_)
                    else:
                        losses['loss_cls'] = loss_cls_

            if bbox_pred is not None:
                
                pos_inds = labels > 0
                # do not perform bounding box regression for BG anymore.
                # if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    # avg_factor=bbox_targets.size(0)
                    avg_factor=pos_bbox_pred.size(0)*6+20
                    )
                
        # targets = bbox_targets.reshape(-1, 5)
        # labels = labels.long().flatten()

        # # masks
        # valid_masks = labels >= 0
        # fg_masks = labels > 0

        # # multiple class
        # bbox_pred = bbox_pred.reshape(-1, 5)
        # fg_gt_classes = labels[fg_masks]
        # bbox_pred = bbox_pred[fg_masks, :]

        # # loss for regression
        # loss_bbox = self.loss_bbox(bbox_pred, targets[fg_masks])

        # # loss for classification
        # labels = labels * valid_masks
        # loss_cls = self.loss_cls(cls_score, labels)

        # losses['loss_cls'] = loss_cls
        # losses['loss_bbox'] = loss_bbox

        return losses
    

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 6) and last
                dimension 6 represent (cx, cy, w, h, a, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = bboxes.view(bboxes.size(0), -1, 5)
            bboxes[..., :4] = bboxes[..., :4] / scale_factor
            bboxes = bboxes.view(bboxes.size(0), -1)

        # if cfg is None:
        #     return bboxes, scores
        # else:
        #     det_bboxes, det_labels = multiclass_nms_rotated(
        #         bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        #     return det_bboxes, det_labels
        return bboxes, scores
    

def random_choice(gallery, num):
    """Random select some elements from the gallery.

    If `gallery` is a Tensor, the returned indices will be a Tensor;
    If `gallery` is a ndarray or list, the returned indices will be a
    ndarray.

    Args:
        gallery (Tensor | ndarray | list): indices pool.
        num (int): expected sample num.

    Returns:
        Tensor or ndarray: sampled indices.
    """
    assert len(gallery) >= num

    is_tensor = isinstance(gallery, torch.Tensor)
    if not is_tensor:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = 'cpu'
        gallery = torch.tensor(gallery, dtype=torch.long, device=device)
    # This is a temporary fix. We can revert the following code
    # when PyTorch fixes the abnormal return of torch.randperm.
    # See: https://github.com/open-mmlab/mmdetection/pull/5014
    perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
    rand_inds = gallery[perm]
    if not is_tensor:
        rand_inds = rand_inds.cpu().numpy()
    return rand_inds