# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch import Tensor
from mmcv.ops import batched_nms, bbox_overlaps
from mmdet.core import anchor_inside_flags, unmap, images_to_levels, multi_apply
from mmcv.runner import force_fp32
from mmdet.utils import get_root_logger

# from scipy.optimize import linear_sum_assignment
from torch_linear_assignment import batch_linear_assignment

from mmrotate.core.bbox.iou_calculators.builder import *
from mmrotate.core import obb2xyxy
from ..builder import ROTATED_HEADS
from .rotated_rpn_head import RotatedRPNHead
from .oriented_rpn_head import OrientedRPNHead
from . import cluster_nms_module


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


@ROTATED_HEADS.register_module()
class MultiInstanceOrientedRPNHead(OrientedRPNHead):
    """Oriented RPN head for Oriented R-CNN."""

    def __init__(self,
                 num_instance = 2,
                 **kwargs):
        self.num_instance = num_instance
        super(OrientedRPNHead, self).__init__(**kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        
        # self.rpn_cls = nn.ModuleList()
        # self.rpn_reg = nn.ModuleList()
        # for k in range(self.num_instance):
        #     self.rpn_cls.append(nn.Conv2d(self.feat_channels,
        #                                   self.num_anchors * self.cls_out_channels, 1))
        #     self.rpn_reg.append(nn.Conv2d(self.feat_channels, 
        #                                   self.num_anchors * 6, 1)) 
            
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_instance * self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, 
                                 self.num_instance * self.num_anchors * 6, 1)

    # def forward_single(self, x):
    #     """Forward feature map of a single scale level."""
    #     # x has shape of (batch_size, channels, w, h)
    #     x = self.rpn_conv(x)
    #     x = F.relu(x, inplace=True)
    #     # separate branches
    #     cls_score = list()
    #     bbox_pred = list()
    #     for k in range(self.num_instance):
    #         cls_score.append(self.rpn_cls[k](x))
    #         bbox_pred.append(self.rpn_reg[k](x))

    #     rpn_cls_score = torch.cat(cls_score, dim=1)
    #     shape = rpn_cls_score.shape
    #     rpn_cls_score = rpn_cls_score.view(shape[0], 2, -1, shape[2], shape[3])
    #     rpn_cls_score = rpn_cls_score.permute(0, 2, 1, 3, 4).contiguous().view(shape)

    #     rpn_bbox_pred = torch.cat(bbox_pred, dim=1)
    #     shape = rpn_cls_score.shape
    #     rpn_cls_score = rpn_cls_score.view(shape[0], 2, -1, shape[2], shape[3])
    #     rpn_cls_score = rpn_cls_score.permute(0, 2, 1, 3, 4).contiguous().view(shape)

    #     return rpn_cls_score, rpn_bbox_pred

    # # RPN验证
    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    # def valid_rpn(self, proposal_list, gt_bboxes, img_metas):
    #     """
    #     【20210825-INCREASE】 验证RPN的有效性:(参考函数_get_targets_single中的用法)
    #     GT有没有取到: RPN输出的数据中, 每个GT上有没有roi, 有的话有多少个, iou是多少
    #     ROIs类别是否判断正确: 由于RPN默认只区分fg/bg, 暂不考虑
    #     Args:
    #         proposal_list: batch_size * [num_rois, [4; cls_scores]]
    #         gt_bboxes: batch_size * [num_gts, 4]
    #         img_metas: {}
    #     Returns:

    #     """
    #     logger = get_root_logger('INFO')
    #     log_str = ""
    #     num_imgs = len(img_metas)

    #     # 与GT进行匹配，考虑最简单的情况，取消了assign的后2个参数; 由于assign返回的是字典类型，所以不能使用multi_apply
    #     # assign_results = multi_apply(self.assigner.assign, proposal_anchor_list, gt_bboxes)
    #     for i in range(num_imgs):
    #         if 'assigner_' not in locals().keys() :
    #             self.assigner_ = copy.deepcopy(self.assigner)
    #             self.assigner_.iou_calculator = build_iou_calculator(dict(type='RBboxOverlaps2D'))
    #         assign_results = self.assigner_.assign(proposal_list[i], gt_bboxes[i])
    #         gt_inds = assign_results.gt_inds
    #         max_overlaps = assign_results.max_overlaps
    #         # 是不是每一个GT都被取到了
    #         num_gts = gt_bboxes[i].shape[0]
    #         log_str += f'\ti:g{num_gts}'
    #         cnt_miss_gt = 0
    #         for j in range(num_gts):
    #             # XXX: 注意：输出的gt_inds，默认为-1，小于neg_iou_thr为0，n个gt的索引为1~n而非0~n-1
    #             j_inds = (gt_inds == j+1)
    #             if j_inds.sum().item() == 0:
    #                 cnt_miss_gt += 1
    #                 continue
    #             # j_cnt07 = (max_overlaps[j_inds] > 0.7).sum().item()
    #             # j_cnt05 = (max_overlaps[j_inds] > 0.5).sum().item()
    #             # log_str += f'{j_cnt05 - j_cnt07}-{j_cnt07}_'

    #         cnt07 = (max_overlaps > 0.7).sum().item()
    #         cnt05 = (max_overlaps > 0.5).sum().item()
    #         if cnt_miss_gt > 0:
    #             log_str += f'\tm{cnt_miss_gt}'
    #         else:
    #             log_str += '\t'
    #         log_str += f'\t{cnt05}\t{cnt07}'
    #     logger.info(log_str)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors ,4)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level
                - label_weights_list (list[Tensor]): Label weights of each \
                  level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        gt_hbboxes = obb2xyxy(gt_bboxes, self.version)
        gt_labels = gt_hbboxes.new_full((gt_hbboxes.shape[0],), -1, dtype=int)
        # num_gts = gt_hbboxes.shape[0]
        # if num_gts < self.num_instance:
        #     pseudo_gt = torch.tensor([0, 0, 1e-6, 1e-6], dtype=gt_hbboxes.dtype, device=gt_hbboxes.device).repeat(self.num_instance-num_gts, 1)
        #     pseudo_ogt = torch.tensor([0, 0, 1e-6, 1e-6, 0], dtype=gt_hbboxes.dtype, device=gt_hbboxes.device).repeat(self.num_instance-num_gts, 1)
        #     gt_hbboxes = torch.cat([gt_hbboxes, pseudo_gt], dim=0)
        #     gt_bboxes = torch.cat([gt_bboxes, pseudo_ogt], dim=0)
        assign_result = self.assigner.assign(
            anchors, gt_hbboxes, gt_bboxes_ignore,
            # None if self.sampling else 
            gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_hbboxes)

        if gt_bboxes.numel() == 0:
            sampling_result.pos_gt_bboxes = gt_bboxes.new(
                (0, gt_bboxes.size(-1))).zero_()
        else:
            sampling_result.pos_gt_bboxes = \
                gt_bboxes[sampling_result.pos_assigned_gt_inds, :]

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((anchors.size(0), self.num_instance * 6))
        bbox_weights = anchors.new_zeros((anchors.size(0), self.num_instance * 6))
        labels = anchors.new_full((num_valid_anchors, self.num_instance),
                                #   self.num_classes,
                                  0,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros((num_valid_anchors, self.num_instance), dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            sample_bboxes = sampling_result.pos_gt_bboxes
            sample_bboxes = sample_bboxes.reshape(-1, 5)
            sample_priors = sampling_result.pos_bboxes
            sample_priors = sample_priors.repeat(1, self.num_instance).reshape(-1, 4)

            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sample_priors, 
                                                          sample_bboxes)
            else:
                pos_bbox_targets = sample_bboxes
            pos_bbox_targets = pos_bbox_targets.reshape(-1, self.num_instance * 6)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, 0:6] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds, :] = 0
            else:
                # cls_score : [bg, fg]
                labels[pos_inds, :] = sampling_result.pos_gt_labels

            bbox_weights[pos_inds, 6:12] = torch.where(labels[pos_inds, 1] == 1, 1.0, 0.0).reshape(-1, 1).repeat(1, 6)
            
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds, :] = 1.0
            else:
                label_weights[pos_inds, :] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds, :] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=0)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)


        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_single(self, 
                    cls_score, 
                    bbox_pred, 
                    anchors, 
                    labels, 
                    label_weights,
                    bbox_targets, 
                    bbox_weights, 
                    num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 4).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
      
        # Hungarian matching
        
        # classification cost
        labels = labels.reshape(-1, self.num_instance)
        label_weights = label_weights.reshape(-1, self.num_instance)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_instance, self.cls_out_channels)
        cls_cost = _binary_cross_entropy(cls_score, labels)

        # regression cost
        bbox_targets = bbox_targets.reshape(-1, self.num_instance, 6)
        bbox_weights = bbox_weights.reshape(-1, self.num_instance, 6)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.num_instance, 6)
        if self.reg_decoded_bbox:   #false
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        reg_cost = _smooth_l1_loss(bbox_pred, bbox_targets, bbox_weights)

        cost = torch.stack([cls_cost, reg_cost]).sum(dim=0)
        
        match_result = batch_linear_assignment(cost)
        bbox_pred = torch.gather(bbox_pred, dim=1, index=match_result.unsqueeze(-1).expand(-1, -1, bbox_pred.size(2)))
        cls_score = torch.gather(cls_score, dim=1, index=match_result.unsqueeze(-1).expand(-1, -1, cls_score.size(2)))


        # classification loss
        cls_score = cls_score.reshape(-1)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        loss_cls = self.loss_cls(cls_score, 
                                 labels, 
                                 label_weights)
        loss_cls = loss_cls.reshape(-1, self.num_instance).sum(dim=1)
        
        # regression loss
        bbox_pred = bbox_pred.reshape(-1, 6)
        bbox_targets = bbox_targets.reshape(-1, 6)
        bbox_weights = bbox_weights.reshape(-1, 6)
        loss_bbox = self.loss_bbox(bbox_pred,
                                   bbox_targets,
                                   bbox_weights)
        loss_bbox = loss_bbox.sum(dim=1)
        loss_bbox = loss_bbox.reshape(-1, self.num_instance).sum(dim=1)

        loss_cls = loss_cls.mean()
        loss_bbox = loss_bbox.mean()
        return loss_cls, loss_bbox
        
        # classification loss
        labels = labels.reshape(-1, self.num_instance)
        label_weights = label_weights.reshape(-1, self.num_instance)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_instance * self.cls_out_channels)
        # loss_cls = self.loss_cls(
        #     cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, self.num_instance * 6)
        bbox_weights = bbox_weights.reshape(-1, self.num_instance * 6)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.num_instance * 6)
        if self.reg_decoded_bbox:   #false
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        # loss_bbox = self.loss_bbox(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     avg_factor=num_total_samples)

        losses = dict()
        if bbox_pred.numel():
            loss_0 = self.emd_loss(bbox_pred[:, 0:6], cls_score[:, 0:1],
                                   bbox_pred[:, 6:12], cls_score[:, 1:2],
                                   bbox_targets, labels,
                                   bbox_weights, label_weights)
            loss_1 = self.emd_loss(bbox_pred[:, 6:12], cls_score[:, 1:2],
                                   bbox_pred[:, 0:6], cls_score[:, 0:1],
                                   bbox_targets, labels,
                                   bbox_weights, label_weights)
            loss = torch.cat([loss_0, loss_1], dim=1)
            _, min_indices = loss.min(dim=1)
            loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
            loss_emd = loss_emd.mean()
            # loss_emd = loss_0.mean()
        else:
            loss_emd = bbox_pred.sum()
        losses['loss_rpn_emd'] = loss_emd
        return loss_emd, None

    def emd_loss(self, 
                 bbox_pred_0: Tensor, 
                 cls_score_0: Tensor,
                 bbox_pred_1: Tensor, 
                 cls_score_1: Tensor, 
                 targets: Tensor,
                 labels: Tensor,
                 bbox_weights: Tensor,
                 label_weights: Tensor) -> Tensor:
        """Calculate the emd loss.

        Note:
            This implementation is modified from https://github.com/Purkialo/
            CrowdDet/blob/master/lib/det_oprs/loss_opr.py

        Args:
            bbox_pred_0 (Tensor): Part of regression prediction results, has
                shape (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            cls_score_0 (Tensor): Part of classification prediction results,
                has shape (batch_size * num_proposals_single_image,
                (num_classes + 1)), where 1 represents the background.
            bbox_pred_1 (Tensor): The other part of regression prediction
                results, has shape (batch_size*num_proposals_single_image, 4).
            cls_score_1 (Tensor):The other part of classification prediction
                results, has shape (batch_size * num_proposals_single_image,
                (num_classes + 1)).
            targets (Tensor):Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image,
                4 * k), the last dimension 4 represents [tl_x, tl_y, br_x,
                br_y], k represents the number of prediction boxes generated
                by each proposal box.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, k).

        Returns:
            torch.Tensor: The calculated loss.
        """

        bbox_pred = torch.cat([bbox_pred_0, bbox_pred_1],
                              dim=1).reshape(-1, bbox_pred_0.shape[-1])
        cls_score = torch.cat([cls_score_0, cls_score_1],
                              dim=1).reshape(-1, cls_score_0.shape[-1])
        targets = targets.reshape(-1, 6)
        labels = labels.long().flatten()
        bbox_weights = bbox_weights.reshape(-1, 6)
        label_weights = label_weights.reshape(-1)

        # # masks
        # valid_masks = labels >= 0
        # # fg_masks = labels > 0
        # fg_masks = labels == 0

        # # multiple class
        # bbox_pred = bbox_pred.reshape(-1, self.num_classes, 6)
        # fg_gt_classes = labels[fg_masks]
        # bbox_pred = bbox_pred[fg_masks, fg_gt_classes - 1, :]

        # loss for regression
        if bbox_pred.numel():
            loss_bbox = self.loss_bbox(bbox_pred, targets, bbox_weights)
            loss_bbox = loss_bbox.sum(dim=1)

        # loss for classification
        cls_score = cls_score.reshape(-1)
        loss_cls = self.loss_cls(cls_score, labels, label_weights)
        # loss_cls = loss_cls.reshape(-1)
        if bbox_pred.numel():
            loss_cls= loss_cls + loss_bbox
        loss = loss_cls.reshape(-1, 2).sum(dim=1)
        return loss.reshape(-1, 1)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        
        loss = dict()
        # loss['loss_rpn_emd'] = losses[0]
        loss['loss_rpn_cls'] = losses[0]
        loss['loss_rpn_bbox'] = losses[1]
        return loss

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

          Args:
            cls_scores (list[Tensor]): Box scores of all scale level
                each item has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas of all
                scale level, each item has shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx, _ in enumerate(cls_scores):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
                scores = scores.reshape(-1, self.num_instance)
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                # scores = rpn_cls_score.softmax(dim=1)[:, 0]
                scores = rpn_cls_score.softmax(dim=1)
                scores = scores.reshape(-1, 4)
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, self.num_instance * 6)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # ranked_scores, rank_inds = scores[:, [1, 3]].max(dim=1)[0].sort(descending=True)
                ranked_scores, rank_inds = scores.max(dim=1)[0].sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = scores[topk_inds, :]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        # scores = torch.cat(mlvl_scores).reshape(-1, 2)[:, 1]    #scores: [bg, fg]
        scores = torch.cat(mlvl_scores).reshape(-1)
        anchors = torch.cat(mlvl_valid_anchors)
        anchors = anchors.repeat_interleave(self.num_instance, dim=0)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds).reshape(-1, 6)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)
        ids = ids.repeat_interleave(self.num_instance, dim=0)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2]
            h = proposals[:, 3]
            valid_mask = (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
        if proposals.numel() > 0:
            hproposals = obb2xyxy(proposals, self.version)

            if cfg.use_setnms:
            
                roi_idx = np.tile(
                    np.arange(hproposals.shape[0] / self.num_instance)[:, None],
                    (1, self.num_instance)).reshape(-1, 1)[:, 0]
                roi_idx = torch.from_numpy(roi_idx).to(hproposals.device).reshape(
                    -1, 1).to(hproposals.dtype)
                hproposals = torch.cat([hproposals, roi_idx], dim=1)
                _, keep = set_nms(hproposals, scores, ids, cfg.nms['iou_threshold'])

            else:
                _, keep = batched_nms(hproposals, scores, ids, cfg.nms)

            dets = torch.cat([proposals, scores[:, None]], dim=1)
            dets = dets[keep]
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]
    
def set_nms(bboxes: Tensor,
            scores: Tensor,
            idxs: Tensor,
            iou_threshold: float):
    total_mask = torch.zeros_like(scores, dtype=torch.bool, device=bboxes.device)

    for id in torch.unique(idxs):


        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        ordered_scores, order = scores[mask].sort(descending=True)
        ordered_bboxes = bboxes[mask][order]
        roi_idx = ordered_bboxes[:, -1]

        keep = torch.ones(len(ordered_bboxes), dtype=torch.bool, device=bboxes.device)

        overlaps = bbox_overlaps(ordered_bboxes[:, :4], ordered_bboxes[:, :4])

        # 创建相同 idx 矩阵
        roi_idx_matrix = roi_idx.unsqueeze(0) == roi_idx.unsqueeze(1)

        # 找到需要去除的边界框的索引
        indices = (overlaps > iou_threshold) & ~roi_idx_matrix
        indices = torch.triu(indices, diagonal=1)

        # Cluster NMS(0.068s)
        # indices_prev = ~indices
        # indices_next = indices
        # while(not (~indices_prev.any(dim=0)).equal(~indices_next.any(dim=0))):
        #     indices_prev = indices_next
        #     indices_next = indices & (~indices_prev.any(dim=0)).unsqueeze(1)
        # # 更新保留的边界框
        # keep = keep & ~indices_next.any(dim=0)

        # Cluster NMS C++ implement(0.05s)
        cluster_nms_module.cluster_nms(indices, keep)
        

        # Fast NMS
        # keep = keep & ~indices.any(dim=0)

        # Traditional NMS
        # indices = ~indices
        # set_nms_module.keepbbox(indices, keep, len(ordered_bboxes))

        total_mask[mask[keep]] = True

    keep = total_mask.nonzero(as_tuple=False).view(-1)
    ordered_scores, order = scores[keep].sort(descending=True)
    keep = keep[order]

    return bboxes[keep], keep

def _binary_cross_entropy(cls_pred: Tensor,
                                  gt_labels: Tensor) -> Tensor:
    """
    Args:
        cls_pred (Tensor): The prediction with shape (num_queries, 1, *) or
            (num_queries, *).
        gt_labels (Tensor): The learning label of prediction with
            shape (num_gt, *).

    Returns:
        Tensor: Cross entropy cost matrix in shape (num_queries, num_gt).
    """
    with torch.no_grad():
        cls_pred = cls_pred.reshape(cls_pred.size(0), cls_pred.size(1), -1).float()
        gt_labels = gt_labels.reshape(gt_labels.size(0), gt_labels.size(1), -1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('anc,amc->anm', pos, gt_labels) + \
            torch.einsum('anc,amc->anm', neg, 1 - gt_labels)
        # cls_cost = cls_cost / n

    return cls_cost
        
        
def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_weights):

    with torch.no_grad():
        # 扩展input和target的形状以计算每个实例之间的loss
        bbox_pred = bbox_pred.unsqueeze(2)
        bbox_targets = bbox_targets.unsqueeze(1)
        diff = bbox_pred - bbox_targets

        # 计算loss
        reg_cost = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none')

        # 将 loss 乘以权重
        reg_cost = reg_cost * bbox_weights.unsqueeze(1)  

        # 求和在最后一个维度
        reg_cost = reg_cost.sum(-1)

    return reg_cost

