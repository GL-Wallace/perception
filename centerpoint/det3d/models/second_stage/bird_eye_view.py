"""第二阶段 BEV（鸟瞰视角）点特征提取器。

对应论文 Sec. 3.4：在第一阶段检测框的中心（以及可选的前/后/左/右四个面中心，
见 TwoStageDetector.get_box_center）处，从第一阶段输出的 BEV 特征图上做双线性
插值采样点特征，作为 RoI 特征输入后续 RoIHead 的 MLP，预测 IoU 引导的置信度
分数并回归 box 精炼量，从而恢复因下采样 stride 与感受野丢失的局部几何信息。

整体流程：第一阶段框 -> 取中心/面中心点 -> 双线性插值采样 BEV 点特征 ->
（多点拼接）-> RoIHead 的 MLP -> score + box refine。

主要类：
    - BEVFeatureExtractor: 负责绝对坐标到 BEV 网格坐标的换算与点特征采样。

与其他模块关系：
    - 由 det3d/models/detectors/two_stage.py 作为 second_stage 模块调用；
    - 采样点坐标由 TwoStageDetector.get_box_center 生成。
"""

import torch
from torch import nn

from ..registry import SECOND_STAGE
from det3d.core.utils.center_utils import (
    bilinear_interpolate_torch,
)

@SECOND_STAGE.register_module
class BEVFeatureExtractor(nn.Module): 
    """在 BEV 特征图上按给定采样点做双线性插值提取 RoI 点特征。"""

    def __init__(self, pc_start, 
            voxel_size, out_stride):
        """记录坐标换算所需的参数。

        Args:
            pc_start (list[float]): 点云范围起点（x/y/z 最小值）。
            voxel_size (list[float]): 每个 voxel 的物理尺寸。
            out_stride (int): backbone 相对原始输入的下采样倍率。
        """
        super().__init__()
        self.pc_start = pc_start 
        self.voxel_size = voxel_size
        self.out_stride = out_stride

    def absl_to_relative(self, absolute):
        """把绝对物理坐标换算成 BEV 特征图的网格坐标。

        网格坐标 = (绝对坐标 - 范围起点) / voxel 尺寸 / 下采样倍率。

        Args:
            absolute (Tensor): (..., 2或3) 绝对坐标（只用前两维 x/y）。

        Returns:
            (Tensor, Tensor): BEV 网格坐标 (x, y)。
        """
        a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride 
        a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride 

        return a1, a2

    def forward(self, example, batch_centers, num_point):
        """采样每个 RoI 在 BEV 特征图上的点特征。

        Args:
            example (dict): 含 'bev_feature'（形状 (B, H, W, C)，通道在最后一维）。
            batch_centers (list[Tensor]): 每个 batch 的采样点，形状 (N*num_point, 3)，
                当 num_point=5 时依次为中心与前后左右四个面中心。
            num_point (int): 每个框采样的点数（1 或 5）。

        Returns:
            list[Tensor]: 每个 batch 一个 (N, C*num_point) 特征张量。
        """
        batch_size = len(example['bev_feature'])
        ret_maps = [] 

        for batch_idx in range(batch_size):
            # 采样点绝对坐标 -> BEV 网格坐标。
            xs, ys = self.absl_to_relative(batch_centers[batch_idx])
            
            # 双线性插值：对每个采样点取 BEV 特征向量，得到 (N*num_point, C)。
            feature_map = bilinear_interpolate_torch(example['bev_feature'][batch_idx],
             xs, ys)

            if num_point > 1:
                # 把 num_point 个面/中心点的特征在通道维拼接，(N, C*num_point)。
                # 这里采用拼接而非额外池化，直接保留各点的原始特征供后续 MLP 使用。
                section_size = len(feature_map) // num_point
                feature_map = torch.cat([feature_map[i*section_size: (i+1)*section_size] for i in range(num_point)], dim=1)

            ret_maps.append(feature_map)

        return ret_maps 