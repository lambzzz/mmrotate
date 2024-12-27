# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from ..builder import ROTATED_BBOX_ASSIGNERS
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.max_iou_assigner import MaxIoUAssigner


@ROTATED_BBOX_ASSIGNERS.register_module()
class MultiInstanceAssigner(MaxIoUAssigner):
    """Assign a corresponding gt bbox or background to each proposal bbox. If
    we need to use a proposal box to generate multiple predict boxes,
    `MultiInstanceAssigner` can assign multiple gt to each proposal box.

    Args:
        num_instance (int): How many bboxes are predicted by each proposal box.
    """

    def __init__(self, 
                 num_instance: int = 2, 
                 add_gt_as_proposals=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_instance = num_instance
        self.add_gt_as_proposals = add_gt_as_proposals

    def assign(self, priors, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, gt_labels_ignore = None, **kwargs):

        """Assign gt to bboxes.

        This method assign gt bboxes to every bbox (proposal/anchor), each bbox
        is assigned a set of gts, and the number of gts in this set is defined
        by `self.num_instance`.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        # gt_bboxes = gt_instances.bboxes
        # priors = pred_instances.priors
        # Set the FG label to 1 and add ignored annotations
        # gt_labels = gt_instances.labels + 1
        if gt_labels is not None:
            gt_labels = gt_labels + 1
        # if gt_instances_ignore is not None:
        #     gt_bboxes_ignore = gt_instances_ignore.bboxes
        #     if hasattr(gt_instances_ignore, 'labels'):
        #         gt_labels_ignore = gt_instances_ignore.labels
        #     else:
        #         gt_labels_ignore = torch.ones_like(gt_bboxes_ignore)[:, 0] * -1
        # else:
        #     gt_bboxes_ignore = None
        #     gt_labels_ignore = None

        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = priors.device
            priors = priors.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
                gt_labels_ignore = gt_labels_ignore.cpu()

        if gt_bboxes_ignore is not None:
            all_bboxes = torch.cat([gt_bboxes, gt_bboxes_ignore], dim=0)
            all_labels = torch.cat([gt_labels, gt_labels_ignore], dim=0)
        else:
            all_bboxes = gt_bboxes
            all_labels = gt_labels
        if self.add_gt_as_proposals:
            all_priors = torch.cat([priors[:, :5], all_bboxes], dim=0)
        else:
            all_priors = priors[:, :5]


        overlaps_normal = self.iou_calculator(
            all_priors, all_bboxes, mode='iou')
        overlaps_ignore = self.iou_calculator(
            all_priors, all_bboxes, mode='iof')
        
        num_bboxes, num_gts = overlaps_normal.size(0), overlaps_normal.size(1)
        # if num_gts == 0 or num_bboxes == 0:
        #     # No ground truth or boxes, return empty assignment
        #     max_overlaps = overlaps_normal.new_zeros((num_bboxes, ))
        #     assigned_gt_inds = overlaps_normal.new_full((num_bboxes, ),
        #                                      -1,
        #                                      dtype=torch.long)
        #     if num_gts == 0:
        #         # No truth, assign everything to background
        #         assigned_gt_inds[:] = 1
        #     if gt_labels is None:
        #         assigned_labels = None
        #     else:
        #         assigned_labels = overlaps_normal.new_full((num_bboxes, ),
        #                                             -1,
        #                                             dtype=torch.long)
        #     return AssignResult(
        #         num_gts,
        #         assigned_gt_inds,
        #         max_overlaps,
        #         labels=assigned_labels)

        if gt_labels is not None:
            gt_ignore_mask = all_labels.eq(-1).repeat(all_priors.shape[0], 1)
        else:
            gt_ignore_mask = all_bboxes.new_full((all_priors.shape[0], all_bboxes.shape[0]), False, dtype=bool)
        overlaps_normal = overlaps_normal * ~gt_ignore_mask
        overlaps_ignore = overlaps_ignore * gt_ignore_mask

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps_normal.max(dim=0)
        _overlaps_normal = overlaps_normal

        overlaps_normal, overlaps_normal_indices = overlaps_normal.sort(
            descending=True, dim=1)
        overlaps_ignore, overlaps_ignore_indices = overlaps_ignore.sort(
            descending=True, dim=1)

        # select the roi with the higher score
        max_overlaps_normal = overlaps_normal[:, :self.num_instance].flatten()
        gt_assignment_normal = overlaps_normal_indices[:, :self.
                                                       num_instance].flatten()
        max_overlaps_ignore = overlaps_ignore[:, :self.num_instance].flatten()
        gt_assignment_ignore = overlaps_ignore_indices[:, :self.
                                                       num_instance].flatten()

        # ignore or not
        ignore_assign_mask = (max_overlaps_normal < self.pos_iou_thr) * (
            max_overlaps_ignore > max_overlaps_normal)
        overlaps = (max_overlaps_normal * ~ignore_assign_mask) + (
            max_overlaps_ignore * ignore_assign_mask)
        gt_assignment = (gt_assignment_normal * ~ignore_assign_mask) + (
            gt_assignment_ignore * ignore_assign_mask)

        if all_labels is None:
            all_labels = all_bboxes.new_zeros(all_bboxes.shape[0], dtype=torch.long)
        assigned_labels = all_labels[gt_assignment]
        fg_mask = (overlaps >= self.pos_iou_thr) * (assigned_labels != -1)
        bg_mask = (overlaps < self.neg_iou_thr) * (overlaps >= 0)
        assigned_labels[fg_mask] = 1
        assigned_labels[bg_mask] = 0

        if self.match_low_quality:
            # Low-quality matching will overwrite the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox 2.
            # This might be the reason that it is not used in ROI Heads.
            gt_assignment = gt_assignment.reshape(-1, self.num_instance)
            assigned_labels = assigned_labels.reshape(-1, self.num_instance)
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = (_overlaps_normal[:, i] == gt_max_overlaps[i]).nonzero(as_tuple=True)[0]
                        for ind in max_iou_inds:        # 遍历目标GT最大重叠的所有anchor

                            # assigned_labels：anchor是否为GT. gt_assignment：anchor关联的gt索引
                            for j in range(self.num_instance):
                                if assigned_labels[ind, j] == 1 and j < self.num_instance-1:
                                    if gt_assignment[ind, j] != i:
                                        continue        # 首选框非目标GT，在次选框中关联
                                    else:
                                        break           # 如果当前anchor已经关联了目标GT，跳过
                                else:
                                    gt_assignment[ind, j] = i 
                                    assigned_labels[ind, j] = 1
                                    break
                    else:
                        gt_assignment[gt_argmax_overlaps[i], :] = i 
                        assigned_labels[gt_argmax_overlaps[i], :] = 1



        overlaps = overlaps.reshape(-1, self.num_instance)
        gt_assignment = gt_assignment.reshape(-1, self.num_instance)
        assigned_labels = assigned_labels.reshape(-1, self.num_instance)

        assign_result = AssignResult(
            num_gts=all_bboxes.size(0),
            gt_inds=gt_assignment,
            max_overlaps=overlaps,
            labels=assigned_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result
