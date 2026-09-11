# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

"""RoI 采样与目标分配层。

对应论文 Sec. 3.4 的监督构造：从第一阶段检测出的 RoI 中按前景/简单背景/困难
背景比例采样固定数量，计算每个 RoI 与 GT 的旋转 3D IoU，据此生成回归前景掩码、
分类标签（二元或 IoU 软化标签）以及匹配到的 GT。

主要类：
    - ProposalTargetLayer: 完成 RoI 采样、IoU 计算与标签生成。

与其他模块关系：
    - 被 roi_head_template.py 的 RoIHeadTemplate.assign_targets 调用；
    - 依赖 iou3d_nms 的 boxes_iou3d_gpu 计算旋转 3D IoU。
"""

import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


class ProposalTargetLayer(nn.Module):
    """负责 RoI 采样、IoU 计算与训练标签生成的目标分配层。"""

    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """采样一批 RoI 并生成回归/分类标签。

        Args:
            batch_dict (dict): 含 rois/roi_scores/roi_labels/gt_boxes_and_cls/
                roi_features 等字段。

        Returns:
            dict: 含 rois/gt_of_rois/gt_iou_of_rois/roi_scores/roi_labels/
                roi_features/reg_valid_mask/rcnn_cls_labels 的目标字典。
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels, \
        batch_roi_features = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        )
        # 回归有效掩码：IoU 超过前景阈值的 RoI 才参与 box 回归监督。
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # 生成分类标签（论文 Sec. 3.4 的 IoU 引导置信度监督）。
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            # 二元标签：IoU 超前景阈值记 1，低于背景阈值记 0，中间区间记 -1（忽略）。
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            # 用 IoU 本身作为回归目标（软化标签），而不是硬二元标签。
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            # 中间区间按 IoU 在 [bg, fg] 内的相对位置线性映射到 [0, 1] 连续值。
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        else:
            raise NotImplementedError

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'roi_features': batch_roi_features,  'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """对每张图计算 RoI 与 GT 的 IoU 并采样固定数量的 RoI。

        Args:
            batch_dict (dict): 含 batch_size/rois/roi_scores/roi_labels/
                gt_boxes_and_cls/roi_features 等字段。

        Returns:
            tuple: 采样后的 rois、匹配的 GT、IoU、分数、标签、特征。
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes_and_cls']
        roi_features = batch_dict['roi_features']

        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_roi_features = roi_features.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, 
            roi_features.shape[-1])

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores, cur_roi_features = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index], \
                roi_features[index]

            # 去掉 GT 末尾的零填充（数据加载时按最大目标数对齐的占位框）。
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                # 按类别分组计算 IoU，保证同类之间匹配。
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi[:, :7], roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )
            else:
                # 直接对所有 GT 计算旋转 3D IoU，取每个 RoI 的最佳匹配。
                iou3d = boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]
            batch_roi_features[index] = cur_roi_features[sampled_inds]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels, batch_roi_features

    def subsample_rois(self, max_overlaps):
        """按前景/简单背景/困难背景比例从候选 RoI 中采样。

        Args:
            max_overlaps (Tensor): 每个 RoI 与其最佳匹配 GT 的 IoU。

        Returns:
            Tensor: 采样出的 RoI 索引（前景在前、背景在后）。
        """
        # 采样前景(fg)、简单背景(easy_bg)、困难背景(hard_bg)三类。
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # 采样前景：随机排列后取前 fg_rois_per_this_image 个，且不超过实际前景数。
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # 采样背景
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # 采样前景
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # 采样背景
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        """按硬背景比例在困难/简单背景中采样指定数量的背景 RoI。

        困难背景（IoU 在中阈值与前景阈值之间）更难分辨，通常在训练中占有更大
        权重；简单背景（IoU 极低）则补充易分负样本。

        Args:
            hard_bg_inds (Tensor): 困难背景索引。
            easy_bg_inds (Tensor): 简单背景索引。
            bg_rois_per_this_image (int): 需要采样的背景 RoI 数量。
            hard_bg_ratio (float): 困难背景占比。

        Returns:
            Tensor: 采样出的背景 RoI 索引。
        """
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # 采样困难背景
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # 采样简单背景
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # 采样困难背景
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # 采样简单背景
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """按类别分组计算每个 RoI 与其同类 GT 的最大 IoU 及 GT 索引。

        只在同类之间匹配，避免不同类框的几何重叠被误判为正样本。

        Args:
            rois (Tensor): (N, 7) 候选 RoI。
            roi_labels (Tensor): (N,) RoI 的类别标签。
            gt_boxes (Tensor): (M, 7) GT 框。
            gt_labels (Tensor): (M,) GT 的类别标签。

        Returns:
            (Tensor, Tensor): 每个 RoI 的最大 IoU 与对应的 GT 索引。
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment
