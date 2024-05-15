# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from torch import Tensor

from ..builder import ROTATED_HEADS, build_loss
from mmrotate.core import rbbox2roi, rbbox2result
from .oriented_standard_roi_head import OrientedStandardRoIHead


@ROTATED_HEADS.register_module()
class MultiInstanceRoIHead(OrientedStandardRoIHead):
    """The roi head for Multi-instance prediction."""

    def __init__(self, num_instance: int = 2, *args, **kwargs) -> None:
        self.num_instance = num_instance
        super().__init__(*args, **kwargs)
        # loss_bbox_iou = dict(type='RotatedIoULoss', loss_weight=1.0)
        # self.loss_bbox_iou = build_loss(loss_bbox_iou)

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `cls_score_ref` (Tensor): The cls_score after refine model.
                - `bbox_pred_ref` (Tensor): The bbox_pred after refine model.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        if self.bbox_head.with_refine:
            raise Exception("Refine Module unimplemented")
            # bbox_results = dict(
            #     cls_score=bbox_results[0],
            #     bbox_pred=bbox_results[1],
            #     cls_score_ref=bbox_results[2],
            #     bbox_pred_ref=bbox_results[3],
            #     bbox_feats=bbox_feats)
        else:
            bbox_results = dict(
                cls_score=cls_score,
                bbox_pred=bbox_pred,
                bbox_feats=bbox_feats)

        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = rbbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        # If there is a refining process, add refine loss.
        if 'cls_score_ref' in bbox_results:
            raise Exception("Refine Module unimplemented")
            # bbox_loss_and_target = self.bbox_head.loss_and_target(
            #     cls_score=bbox_results['cls_score'],
            #     bbox_pred=bbox_results['bbox_pred'],
            #     rois=rois,
            #     sampling_results=sampling_results,
            #     rcnn_train_cfg=self.train_cfg)
            # bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
            # bbox_loss_and_target_ref = self.bbox_head.loss_and_target(
            #     cls_score=bbox_results['cls_score_ref'],
            #     bbox_pred=bbox_results['bbox_pred_ref'],
            #     rois=rois,
            #     sampling_results=sampling_results,
            #     rcnn_train_cfg=self.train_cfg)
            # bbox_results['loss_bbox']['loss_rcnn_emd_ref'] = \
            #     bbox_loss_and_target_ref['loss_bbox']['loss_rcnn_emd']
        else:
            bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)

            bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

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

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def simple_test(self,
                    x,
                    proposals,
                    img_metas,
                    rescale=False):
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        rcnn_test_cfg = self.test_cfg
        num_imgs = len(proposals)
        # proposals = [res.bboxes for res in rpn_results_list]
        rois = rbbox2roi(proposals)

        if rois.shape[0] == 0:
            pass
            # return empty_instances(
            #     img_metas, rois.device, task_type='bbox')

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        if 'cls_score_ref' in bbox_results:
            cls_scores = bbox_results['cls_score_ref']
            bbox_preds = bbox_results['bbox_pred_ref']
        else:
            cls_scores = bbox_results['cls_score']
            bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        if bbox_preds is not None:
            bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
        else:
            bbox_preds = (None, ) * len(proposals)

        det_bboxes, det_labels = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        
        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head.num_classes)
            for i in range(num_imgs)
        ]
        return bbox_results
