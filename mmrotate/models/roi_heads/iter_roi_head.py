# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

import torch
import numpy as np
from torch import Tensor
from mmcv.runner import BaseModule, ModuleList
from mmcv.ops import box_iou_rotated
from mmdet.core import bbox2roi

from mmrotate.core import (build_assigner, build_sampler, obb2xyxy,
                           rbbox2result, rbbox2roi, multiclass_nms_rotated)
from mmrotate.models.roi_heads.bbox_heads.multi_instance_bbox_head import MultiInstanceBBoxHead
from ..builder import ROTATED_HEADS, build_head, build_roi_extractor



@ROTATED_HEADS.register_module()
class IterRoIHead(BaseModule, metaclass=ABCMeta):
    """RoI Trans cascade roi head including one bbox head.

    Args:
        num_stages (int): number of cascade stages.
        stage_loss_weights (list[float]): loss weights of cascade stages.
        bbox_roi_extractor (dict, optional): Config of ``bbox_roi_extractor``.
        bbox_head (dict, optional): Config of ``bbox_head``.
        shared_head (dict, optional): Config of ``shared_head``.
        train_cfg (dict, optional): Config of train.
        test_cfg (dict, optional): Config of test.
        pretrained (str, optional): Path of pretrained weight.
        version (str, optional): Angle representations. Defaults to 'oc'.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 version='oc',
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        super(IterRoIHead, self).__init__(init_cfg)
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained
        self.version = version

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        self.init_assigner_sampler()

        self.with_bbox = True if self.bbox_head is not None else False

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = ModuleList()

        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_head) == self.num_stages
        for idx, head in enumerate(bbox_head):
            head.update(stage = idx)
            self.bbox_head.append(build_head(head))

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        # bbox head
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                if i > 0:
                    rois = rbbox2roi([bbox_results['bbox_pred']])
                bbox_results = self._bbox_forward(i, x, rois)
                proposals = torch.randn(1000, 6).to(proposals.device)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        return outs

    def _bbox_forward(self, stage, x, rois, last_feats = None, extracted_feats = None):
        """Box head forward function used in both training and testing.

        Args:
            x (list[Tensor]): list of multi-level img features.
            rois (list[Tensors]): list of region of interests.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        if extracted_feats is None:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
        else:
            bbox_feats = extracted_feats
        bbox_head = self.bbox_head[stage]
        
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred, fcs_feats = bbox_head(bbox_feats, last_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, fcs_feats=fcs_feats, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg, last_feats = None, extracted_feats = None):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        # if stage == 0:
        #     rois = bbox2roi([res.bboxes for res in sampling_results])
        # else:
        #     rois = rbbox2roi([res.bboxes for res in sampling_results])
        rois = rbbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois, last_feats, extracted_feats)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        first_stage_feats = None
        extracted_feats = None

        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg
            lw = self.stage_loss_weights[i]

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg, first_stage_feats, 
                                                    extracted_feats)
            extracted_feats = bbox_results['bbox_feats']

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                first_stage_feats = bbox_results['fcs_feats']

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        first_stage_feats = None
        extracted_feats = None

        # "ms" in variable names means multi-stage
        ms_bbox_result = []
        ms_scores = []
        cls_score = []
        bbox_pred = []
        rcnn_test_cfg = self.test_cfg
        

        # rois = rbbox2roi(proposal_list)
        for i in range(self.num_stages):
            rois = rbbox2roi(proposal_list)
            bbox_results = self._bbox_forward(i, x, rois, first_stage_feats, extracted_feats)
            extracted_feats = bbox_results['bbox_feats']

            # split batch bbox prediction back to each image
            cls_score.append(bbox_results['cls_score'])
            bbox_pred.append(bbox_results['bbox_pred'])
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score[i] = cls_score[i].split(num_proposals_per_img, 0)
            if bbox_pred[i] is not None:
                bbox_pred[i] = bbox_pred[i].split(num_proposals_per_img, 0)
            else:
                bbox_pred[i] = (None, ) * len(proposal_list)

            if i < self.num_stages - 1:
                first_stage_feats = bbox_results['fcs_feats']

        # average scores of each image by stages
        # cls_score = [
        #     sum([score[i] for score in ms_scores]) / float(len(ms_scores))
        #     for i in range(num_imgs)
        # ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox0, det_score0 = self.bbox_head[0].get_bboxes(
                rois[i],
                cls_score[0][i],
                bbox_pred[0][i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bbox1, det_score1 = self.bbox_head[1].get_bboxes(
                rois[i],
                cls_score[1][i],
                bbox_pred[1][i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            bboxes = torch.cat([det_bbox0, det_bbox1], 1).reshape(-1, 5)
            scores = torch.cat([det_score0, det_score1], 1).reshape(-1, 2)
            # bboxes = det_bbox0
            # scores = det_score0

            roi_idx = np.tile(
                np.arange(bboxes.shape[0] / self.num_stages)[:, None],
                (1, self.num_stages)).reshape(-1, 1)[:, 0]
            roi_idx = torch.from_numpy(roi_idx).to(bboxes.device).reshape(
                -1, 1).to(bboxes.dtype)
            bboxes = torch.cat([bboxes, roi_idx], dim=1)
            det_bbox, det_score = self.set_nms(
                bboxes, scores[:, 1], rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms['iou_thr'], rcnn_test_cfg.max_per_img)
            det_bbox = det_bbox[:, :-1]
            det_bbox = torch.cat([det_bbox, det_score[:, None]], dim=1)
            det_label = torch.zeros_like(det_score)

            # scores = torch.cat([scores[:, 1:2], scores[:, 0:1]], dim=1)
            # det_bbox, det_label = multiclass_nms_rotated(
            #     bboxes, scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
            

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        results = bbox_results

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError
    
    @staticmethod
    def set_nms(bboxes: Tensor,
                scores: Tensor,
                score_thr: float,
                iou_threshold: float,
                max_num: int = -1):
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
