# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import box_iou_rotated
from mmcv.runner import auto_fp16, force_fp32
from torch import Tensor, nn

from mmdet.models.utils import build_linear_layer
from mmrotate.core import rbbox2roi, multiclass_nms_rotated
from .rotated_bbox_head import RotatedBBoxHead
from ...builder import ROTATED_HEADS


@ROTATED_HEADS.register_module()
class MultiInstanceBBoxHead(RotatedBBoxHead):
    r"""Bbox head used in CrowdDet.

    .. code-block:: none

                                      /-> cls convs_1 -> cls fcs_1 -> cls_1
                                   |--
                                   |  \-> reg convs_1 -> reg fcs_1 -> reg_1
                                   |
                                   |  /-> cls convs_2 -> cls fcs_2 -> cls_2
        shared convs -> shared fcs |--
                                   |  \-> reg convs_2 -> reg fcs_2 -> reg_2
                                   |
                                   |                     ...
                                   |
                                   |  /-> cls convs_k -> cls fcs_k -> cls_k
                                   |--
                                      \-> reg convs_k -> reg fcs_k -> reg_k


    Args:
        num_instance (int): The number of branches after shared fcs.
            Defaults to 2.
        with_refine (bool): Whether to use refine module. Defaults to False.
        num_shared_convs (int): The number of shared convs. Defaults to 0.
        num_shared_fcs (int): The number of shared fcs. Defaults to 2.
        num_cls_convs (int): The number of cls convs. Defaults to 0.
        num_cls_fcs (int): The number of cls fcs. Defaults to 0.
        num_reg_convs (int): The number of reg convs. Defaults to 0.
        num_reg_fcs (int): The number of reg fcs. Defaults to 0.
        conv_out_channels (int): The number of conv out channels.
            Defaults to 256.
        fc_out_channels (int): The number of fc out channels. Defaults to 1024.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """  # noqa: W605

    def __init__(self,
                 num_instance: int = 2,
                 with_refine: bool = False,
                 num_shared_convs: int = 0,
                 num_shared_fcs: int = 2,
                 num_cls_convs: int = 0,
                 num_cls_fcs: int = 0,
                 num_reg_convs: int = 0,
                 num_reg_fcs: int = 0,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 init_cfg = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        assert num_instance == 2, 'Currently only 2 instances are supported'
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_instance = num_instance
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.with_refine = with_refine

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim
        self.relu = nn.ReLU(inplace=True)

        if self.with_refine:
            refine_model_cfg = {
                'type': 'Linear',
                'in_features': self.shared_out_channels + 20,
                'out_features': self.shared_out_channels
            }
            self.shared_fcs_ref = build_linear_layer(refine_model_cfg)
            self.fc_cls_ref = nn.ModuleList()
            self.fc_reg_ref = nn.ModuleList()

        self.cls_convs = nn.ModuleList()
        self.cls_fcs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_fcs = nn.ModuleList()
        self.cls_last_dim = list()
        self.reg_last_dim = list()
        self.fc_cls = nn.ModuleList()
        self.fc_reg = nn.ModuleList()
        for k in range(self.num_instance):
            # add cls specific branch
            cls_convs, cls_fcs, cls_last_dim = self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
            self.cls_convs.append(cls_convs)
            self.cls_fcs.append(cls_fcs)
            self.cls_last_dim.append(cls_last_dim)

            # add reg specific branch
            reg_convs, reg_fcs, reg_last_dim = self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)
            self.reg_convs.append(reg_convs)
            self.reg_fcs.append(reg_fcs)
            self.reg_last_dim.append(reg_last_dim)

            if self.num_shared_fcs == 0 and not self.with_avg_pool:
                if self.num_cls_fcs == 0:
                    self.cls_last_dim *= self.roi_feat_area
                if self.num_reg_fcs == 0:
                    self.reg_last_dim *= self.roi_feat_area

            if self.with_cls:
                if self.custom_cls_channels:
                    cls_channels = self.loss_cls.get_cls_channels(
                        self.num_classes)
                else:
                    cls_channels = self.num_classes + 1
                cls_predictor_cfg_ = self.cls_predictor_cfg.copy()  # deepcopy
                cls_predictor_cfg_.update(
                    in_features=self.cls_last_dim[k],
                    out_features=cls_channels)
                self.fc_cls.append(build_linear_layer(cls_predictor_cfg_))
                if self.with_refine:
                    self.fc_cls_ref.append(build_linear_layer(cls_predictor_cfg_))

            if self.with_reg:
                out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                               self.num_classes)
                reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
                reg_predictor_cfg_.update(
                    in_features=self.reg_last_dim[k], out_features=out_dim_reg)
                self.fc_reg.append(build_linear_layer(reg_predictor_cfg_))
                if self.with_refine:
                    self.fc_reg_ref.append(build_linear_layer(reg_predictor_cfg_))

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False) -> tuple:
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
                        conv_in_channels, self.conv_out_channels, 3,
                        padding=1))
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
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    @auto_fp16()
    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all scale
                  levels, each is a 4D-tensor, the channels number is
                  num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all scale
                  levels, each is a 4D-tensor, the channels number is
                  num_base_priors * 4.
                - cls_score_ref (Tensor): The cls_score after refine model.
                - bbox_pred_ref (Tensor): The bbox_pred after refine model.
        """

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls = x
        x_reg = x
        # separate branches
        cls_score = list()
        bbox_pred = list()
        for k in range(self.num_instance):
            for conv in self.cls_convs[k]:
                x_cls = conv(x_cls)
            if x_cls.dim() > 2:
                if self.with_avg_pool:
                    x_cls = self.avg_pool(x_cls)
                x_cls = x_cls.flatten(1)
            for fc in self.cls_fcs[k]:
                x_cls = self.relu(fc(x_cls))

            for conv in self.reg_convs[k]:
                x_reg = conv(x_reg)
            if x_reg.dim() > 2:
                if self.with_avg_pool:
                    x_reg = self.avg_pool(x_reg)
                x_reg = x_reg.flatten(1)
            for fc in self.reg_fcs[k]:
                x_reg = self.relu(fc(x_reg))

            cls_score.append(self.fc_cls[k](x_cls) if self.with_cls else None)
            bbox_pred.append(self.fc_reg[k](x_reg) if self.with_reg else None)

        if self.with_refine:
            x_ref = x
            cls_score_ref = list()
            bbox_pred_ref = list()
            for k in range(self.num_instance):
                feat_ref = cls_score[k].softmax(dim=-1)
                feat_ref = torch.cat((bbox_pred[k], feat_ref[:, 1][:, None]),
                                     dim=1).repeat(1, 5)
                feat_ref = torch.cat((x_ref, feat_ref), dim=1)
                feat_ref = F.relu_(self.shared_fcs_ref(feat_ref))

                cls_score_ref.append(self.fc_cls_ref[k](feat_ref))
                bbox_pred_ref.append(self.fc_reg_ref[k](feat_ref))

            cls_score = torch.cat(cls_score, dim=1)
            bbox_pred = torch.cat(bbox_pred, dim=1)
            cls_score_ref = torch.cat(cls_score_ref, dim=1)
            bbox_pred_ref = torch.cat(bbox_pred_ref, dim=1)
            return cls_score, bbox_pred, cls_score_ref, bbox_pred_ref

        cls_score = torch.cat(cls_score, dim=1)
        bbox_pred = torch.cat(bbox_pred, dim=1)

        return cls_score, bbox_pred

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all proposals in a
              batch, each tensor in list has shape (num_proposals,) when
              `concat=False`, otherwise just a single tensor has shape
              (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
              all proposals in a batch, each tensor in list has shape
              (num_proposals,) when `concat=False`, otherwise just a single
              tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target for all
              proposals in a batch, each tensor in list has shape
              (num_proposals, 4) when `concat=False`, otherwise just a single
              tensor has shape (num_all_proposals, 4), the last dimension 4
              represents [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
              all proposals in a batch, each tensor in list has shape
              (num_proposals, 4) when `concat=False`, otherwise just a
              single tensor has shape (num_all_proposals, 4).
        """
        labels = []
        bbox_targets = []
        bbox_weights = []
        label_weights = []
        for i in range(len(sampling_results)):
            sample_bboxes = torch.cat([
                sampling_results[i].pos_gt_bboxes,
                sampling_results[i].neg_gt_bboxes
            ])
            sample_priors = sampling_results[i].priors
            sample_priors = sample_priors.repeat(1, self.num_instance).reshape(
                -1, 5)
            sample_bboxes = sample_bboxes.reshape(-1, 5)

            if not self.reg_decoded_bbox:
                _bbox_targets = self.bbox_coder.encode(sample_priors,
                                                       sample_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                _bbox_targets = sample_priors
            _bbox_targets = _bbox_targets.reshape(-1, self.num_instance * 5)
            _bbox_weights = torch.ones(_bbox_targets.shape)
            _labels = torch.cat([
                sampling_results[i].pos_gt_labels,
                sampling_results[i].neg_gt_labels
            ])
            _labels_weights = torch.ones(_labels.shape)

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
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction results of all class,
                has shape (batch_size * num_proposals_single_image,
                (num_classes + 1) * k), k represents the number of prediction
                boxes generated by each proposal box.
            bbox_pred (Tensor): Regression prediction results, has shape
                (batch_size * num_proposals_single_image, 4 * k), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, k).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, k).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image,
                4 * k), the last dimension 4 represents [tl_x, tl_y, br_x,
                br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image,
                4 * k).

        Returns:
            dict: A dictionary of loss.
        """
        losses = dict()
        if bbox_pred.numel():
            num_instance0 = len((labels.reshape(-1, 2)[:, 0]>0).nonzero())
            num_instance1 = len((labels.reshape(-1, 2)[:, 1]>0).nonzero())
            print(num_instance0, ' ', num_instance1, '', len(labels))

            loss_0 = self.emd_loss(bbox_pred[:, 0:5], cls_score[:, 0:2],
                                   bbox_pred[:, 5:10], cls_score[:, 2:4],
                                   bbox_targets, labels)
            loss_1 = self.emd_loss(bbox_pred[:, 5:10], cls_score[:, 2:4],
                                   bbox_pred[:, 0:5], cls_score[:, 0:2],
                                   bbox_targets, labels)
            loss = torch.cat([loss_0, loss_1], dim=1)
            _, min_indices = loss.min(dim=1)
            loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
            # loss_emd = loss_0
            loss_emd = loss_emd.mean()
        else:
            loss_emd = bbox_pred.sum()
        losses['loss_rcnn_emd'] = loss_emd
        return losses

    def emd_loss(self, 
                 bbox_pred_0: Tensor, 
                 cls_score_0: Tensor,
                 bbox_pred_1: Tensor, 
                 cls_score_1: Tensor, 
                 targets: Tensor,
                 labels: Tensor) -> Tensor:
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
        targets = targets.reshape(-1, 5)
        labels = labels.long().flatten()

        # masks
        valid_masks = labels >= 0
        fg_masks = labels > 0

        # multiple class
        bbox_pred = bbox_pred.reshape(-1, self.num_classes, 5)
        fg_gt_classes = labels[fg_masks]
        bbox_pred = bbox_pred[fg_masks, fg_gt_classes - 1, :]

        # loss for regression
        loss_bbox = self.loss_bbox(bbox_pred, targets[fg_masks])
        if len(loss_bbox.shape) == 2:
            loss_bbox = loss_bbox.sum(dim=1)

        # loss for classification
        labels = labels * valid_masks
        loss_cls = self.loss_cls(cls_score, labels)

        loss_cls[fg_masks] = loss_cls[fg_masks] + loss_bbox
        loss = loss_cls.reshape(-1, 2).sum(dim=1)
        return loss.reshape(-1, 1)
    
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        img_meta = {'img_shape': img_shape, 'scale_factor': scale_factor}
        return self._predict_by_feat_single(roi=rois,
                                            cls_score=cls_score,
                                            bbox_pred=bbox_pred,
                                            img_meta=img_meta,
                                            rescale=rescale,
                                            rcnn_test_cfg=cfg)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg = None,
                        rescale: bool = False):
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        det_bboxes = []
        det_labels = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            det_bbox, det_label = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg = None):
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas. has shape
                (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        cls_score = cls_score.reshape(-1, self.num_classes + 1)
        bbox_pred = bbox_pred.reshape(-1, 5)
        roi = roi.repeat_interleave(self.num_instance, dim=0)

        if roi.shape[0] == 0:
            # There is no proposal in the single image
            det_bbox = roi.new_zeros(0, 6)
            det_label = roi.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :5]
                det_label = roi.new_zeros(
                    (0, self.fc_cls.out_features))
            return det_bbox, det_label

        scores = cls_score.softmax(dim=-1) if cls_score is not None else None
        img_shape = img_meta['img_shape']
        bboxes = self.bbox_coder.decode(
            roi[..., 1:], bbox_pred, max_shape=img_shape)

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = bboxes.new_tensor(img_meta['scale_factor'])
            # scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            #     (1, 2))
            # bboxes = (bboxes.view(bboxes.size(0), -1, 5) / scale_factor).view(
            #     bboxes.size()[0], -1)
            bboxes = bboxes.view(bboxes.size(0), -1, 5)
            bboxes[..., :4] = bboxes[..., :4] / scale_factor
            bboxes = bboxes.view(bboxes.size(0), -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            return bboxes, scores
        else:
            roi_idx = np.tile(
                np.arange(bboxes.shape[0] / self.num_instance)[:, None],
                (1, self.num_instance)).reshape(-1, 1)[:, 0]
            roi_idx = torch.from_numpy(roi_idx).to(bboxes.device).reshape(
                -1, 1).to(bboxes.dtype)
            bboxes = torch.cat([bboxes, roi_idx], dim=1)
            det_bboxes, det_scores = self.set_nms(
                bboxes, scores[:, 1], rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms['iou_thr'], rcnn_test_cfg.max_per_img)

            det_bboxes = det_bboxes[:, :-1]
            det_bboxes = torch.cat([det_bboxes, det_scores[:, None]], dim=1)
            det_labels = torch.zeros_like(det_scores)

            # scores = torch.cat([scores[:, 1:2], scores[:, 0:1]], dim=1)
            # bboxes = bboxes.reshape(-1, 10)[:, :5]
            # scores = scores.reshape(-1, 4)[:, :2]
            # bboxes = bboxes.reshape(-1, 10)[:, 5:]
            # scores = scores.reshape(-1, 4)[:, 2:]
            # det_bboxes, det_labels = multiclass_nms_rotated(
            #     bboxes, scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)

        return det_bboxes, det_labels

    @staticmethod
    def set_nms(bboxes: Tensor,
                scores: Tensor,
                score_thr: float,
                iou_threshold: float,
                max_num: int = -1) -> Tuple[Tensor, Tensor]:
        """NMS for multi-instance prediction. Please refer to
        https://github.com/Purkialo/CrowdDet for more details.

        Args:
            bboxes (Tensor): predict bboxes.
            scores (Tensor): The score of each predict bbox.
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            iou_threshold (float): IoU threshold to be considered as
                conflicted.
            max_num (int, optional): if there are more than max_num bboxes
                after NMS, only top max_num will be kept. Default to -1.

        Returns:
            Tuple[Tensor, Tensor]: (bboxes, scores).
        """

        bboxes = bboxes[scores > score_thr]
        scores = scores[scores > score_thr]

        ordered_scores, order = scores.sort(descending=True)
        ordered_bboxes = bboxes[order]
        roi_idx = ordered_bboxes[:, -1]

        keep = torch.ones(len(ordered_bboxes)) == 1
        ruler = torch.arange(len(ordered_bboxes))
        while ruler.shape[0] > 0:
            basement = ruler[0]
            ruler = ruler[1:]
            idx = roi_idx[basement]
            # calculate the body overlap
            basement_bbox = ordered_bboxes[:, :5][basement].reshape(-1, 5)
            ruler_bbox = ordered_bboxes[:, :5][ruler].reshape(-1, 5)
            overlap = rbbox_overlaps(basement_bbox, ruler_bbox)
            indices = torch.where(overlap > iou_threshold)[1]
            loc = torch.where(roi_idx[ruler][indices] == idx)
            # the mask won't change in the step
            mask = keep[ruler[indices][loc]]
            keep[ruler[indices]] = False
            keep[ruler[indices][loc][mask]] = True
            ruler[~keep[ruler]] = -1
            ruler = ruler[ruler > 0]

        keep = keep[order.sort()[1]]
        return bboxes[keep][:max_num, :], scores[keep][:max_num]
    
    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes_(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (torch.Tensor): Shape (n*bs, 6), where n is image number per
                GPU, and bs is the sampled RoIs per image. The first column is
                the image id and the next 5 columns are cx, cy, w, h, a.
            labels (torch.Tensor): Shape (n*bs, ).
            bbox_preds (torch.Tensor): Shape (n*bs, 5) or (n*bs, 5*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i, _ in enumerate(img_metas):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class_(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes = bboxes.reshape(-1, 10)
            bboxes_list.append(bboxes[keep_inds.type(torch.bool)].reshape(-1, 5))

        return bboxes_list

    # override
    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class_(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade head.

        Args:
            rois (torch.Tensor): shape (n, 5) or (n, 6)
            label (torch.Tensor): shape (n, )
            bbox_pred (torch.Tensor): shape (n, 10*(#class)) or (n, 10)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 5 or rois.size(1) == 6, repr(rois.shape)
        bbox_pred = bbox_pred.reshape(-1, 5)
        rois = rois.repeat_interleave(self.num_instance, dim=0)

        # 是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，后续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
        # 若True从bbox_pred中提取对应类别的值
        if not self.reg_class_agnostic:   
            label = label * 5
            inds = torch.stack((label, label + 1, label + 2, label + 3, label + 4), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 5

        if rois.size(1) == 5:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)
        
        return new_rois
    
def rbbox_overlaps(bboxes1,
                   bboxes2,
                   mode='iou',
                   is_aligned=False,
                   version='oc'):
    """Calculate IoU between 2D bboxes.

    Args:
        bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
            <cx, cy, w, h, a> format, or shape (m, 6) in
                <cx, cy, w, h, a2, score> format.
        bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
            <cx, cy, w, h, a> format, shape (m, 6) in
                <cx, cy, w, h, a, score> format, or be empty.
                If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection
            over foreground), or "giou" (generalized intersection over
            union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        version (str, optional): Angle representations. Defaults to 'oc'.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert bboxes1.size(-1) in [0, 5, 6]
    assert bboxes2.size(-1) in [0, 5, 6]

    if bboxes2.size(-1) == 6:
        bboxes2 = bboxes2[..., :5]
    if bboxes1.size(-1) == 6:
        bboxes1 = bboxes1[..., :5]
    return rbbox_overlaps(bboxes1.contiguous(), bboxes2.contiguous(), mode,
                            is_aligned)

def rbbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    clamped_bboxes1 = bboxes1.detach().clone()
    clamped_bboxes2 = bboxes2.detach().clone()
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)
