"""体素特征提取器（VoxelFeatureExtractorV3）。

对每个非空体素内的点特征求均值，得到固定维度的体素特征，供稀疏卷积 backbone
使用。这是 CenterPoint 论文 Sec 3 骨架中 reader（体素编码）部分的最简实现。

主要类：
    - VoxelFeatureExtractorV3: 基于点均值的体素特征编码。
"""

from torch import nn
from torch.nn import functional as F

from ..registry import READERS


@READERS.register_module
class VoxelFeatureExtractorV3(nn.Module):
    """体素特征提取器：对体素内点特征求均值，得到体素特征。"""

    def __init__(
        self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV3"
    ):
        """构造体素特征提取器。

        Args:
            num_input_features (int): 每个点的输入特征维度（如 x,y,z,intensity）。
            norm_cfg (dict, optional): 归一化配置（本实现未使用）。
            name (str): 模块名称。
        """
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        """对每个体素内的点特征求均值，输出 (M, C) 的体素特征。

        Args:
            features (Tensor): (M, max_points, C) 体素内点特征（含 padding）。
            num_voxels (Tensor): (M,) 每个体素内的真实点数。
            coors: 体素坐标（本实现不使用）。

        Returns:
            Tensor: (M, C) 的体素均值特征。
        """
        # 校验体素特征通道数与配置一致。
        assert self.num_input_features == features.shape[-1]

        # 求体素内点特征之和，再除以真实点数得到均值（padding 点为 0，不影响求和）。
        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        return points_mean.contiguous()
