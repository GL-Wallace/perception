"""动态体素特征提取器（DynamicVoxelEncoder）。

训练时动态地对原始点云做体素化（无需预先固定体素数量），对每个体素内的点做
均值池化，得到体素特征与坐标。这是 CenterPoint 论文 Sec 3 骨架中 reader
（体素编码）的动态变体。

主要函数 / 类：
    - voxelization: 动态体素化（裁剪 + 量化坐标 + scatter_mean 均值池化）。
    - voxelization_virtual: 支持虚拟点的动态体素化（面向 nuScenes）。
    - DynamicVoxelEncoder: 对点云列表逐帧体素化并拼接为一个 batch。
"""

from det3d.core.utils.scatter import scatter_mean
from torch.nn import functional as F
from ..registry import READERS
from torch import nn
import numpy as np
import torch 

def voxelization(points, pc_range, voxel_size):
    """将单帧点云体素化：裁剪到范围、量化坐标并按体素做均值池化。

    Args:
        points (Tensor): (N, C) 点云（前 3 维为 xyz，后可为 intensity 等特征）。
        pc_range: 点云范围 [xmin, ymin, zmin, xmax, ymax, zmax]。
        voxel_size: 体素尺寸 [vx, vy, vz]。

    Returns:
        Tuple[Tensor, Tensor]: (体素特征 (M, C), 体素坐标 (M, 3)，顺序为 zyx)。
    """    
    # 过滤出点云范围内的点。
    keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    points = points[keep, :]    
    # 坐标转换到体素网格（逆序为 z,y,x）并取整。
    coords = ((points[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) /  voxel_size[[2, 1, 0]]).to(torch.int64)
    # 得到唯一体素坐标及每个点所属体素的索引（inverse_indices）。
    unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)

    # 按体素索引对点特征做均值池化。
    voxels = scatter_mean(points, inverse_indices, dim=0)
    return voxels, unique_coords

def voxelization_virtual(points, pc_range, voxel_size):
    """支持虚拟点的动态体素化（面向 nuScenes，通道布局被硬编码）。

    将点分为真实点（1）/ 绘制点 painted（0）/ 虚拟点 virtual（-1）三类，按各自
    规则填充到 22 通道的扩展特征中，再按体素做均值池化；对混合体素依真实点占比
    分别归一化两组特征。

    Args:
        points (Tensor): 点云，倒数第 2 维为点类型指示（1 / 0 / -1）。
        pc_range: 点云范围。
        voxel_size: 体素尺寸。

    Returns:
        Tuple[Tensor, Tensor]: (体素特征, 体素坐标，顺序为 zyx)。
    """
    # 当前实现针对 nuScenes 做了硬编码。
    # TODO: 修复这些硬编码的魔法数字。 
    # 过滤出点云范围内的点。
    keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    points = points[keep, :]    

    real_points_mask = points[:, -2] == 1 
    painted_points_mask = points[:, -2] == 0 
    virtual_points_mask = points[:, -2] == -1 

    # 去除真实点中的零填充。 
    real_points = points[real_points_mask][:, [0, 1, 2, 3, -1]]
    painted_point = points[painted_points_mask]  
    virtual_point = points[virtual_points_mask] 

    padded_points = torch.zeros(len(points), 22, device=points.device, dtype=points.dtype)

    # 真实点占用通道 0~4 与 -1。 
    padded_points[:len(real_points), :5] = real_points
    padded_points[:len(real_points), -1] = 1 

    # 绘制点占用通道 5~21。 
    padded_points[len(real_points):len(real_points)+len(painted_point), 5:19] = painted_point[:, :-2]
    padded_points[len(real_points):len(real_points)+len(painted_point), 19] = painted_point[:, -1]
    padded_points[len(real_points):len(real_points)+len(painted_point), 20] = 1
    padded_points[len(real_points):len(real_points)+len(painted_point), 21] = 0

    # 虚拟点占用通道 5~21。 
    padded_points[len(real_points)+len(painted_point):, 5:19] = virtual_point[:, :-2]
    padded_points[len(real_points)+len(painted_point):, 19] = virtual_point[:, -1]
    padded_points[len(real_points)+len(painted_point):, 20] = 0
    padded_points[len(real_points)+len(painted_point):, 21] = 0

    points_xyz = torch.cat([real_points[:, :3], painted_point[:, :3], virtual_point[:, :3]], dim=0)

    # 三类点的 xyz 拼接后统一转换到体素网格并取整。
    coords = ((points_xyz[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) /  voxel_size[[2, 1, 0]]).to(torch.int64)
    # 得到唯一体素坐标及每个点所属体素的索引。
    unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)

    # 按体素索引对扩展特征做均值池化。
    voxels = scatter_mean(padded_points, inverse_indices, dim=0)

    # indicator 记录体素内真实点的占比（均值池化后的指示位）。
    indicator = voxels[:, -1] 
    mix_mask = (indicator > 0) * (indicator < 1)
    # 去掉末尾的指示位。 
    voxels = voxels[:, :-1] 

    # 混合体素（同时含真实点与虚拟/绘制点）按真实点占比分别归一化两组特征。
    voxels[mix_mask, :5] = voxels[mix_mask, :5] / indicator[mix_mask].unsqueeze(-1)
    voxels[mix_mask, 5:] = voxels[mix_mask, 5:] / (1-indicator[mix_mask].unsqueeze(-1))
    return voxels, unique_coords

@READERS.register_module
class DynamicVoxelEncoder(nn.Module):
    """动态体素编码器：对点云列表逐帧体素化并拼接为一个 batch。"""

    def __init__(
        self, pc_range, voxel_size, virtual=False
    ):
        """构造动态体素编码器，预先计算体素网格维度。

        Args:
            pc_range: 点云范围 [xmin, ymin, zmin, xmax, ymax, zmax]。
            voxel_size: 体素尺寸 [vx, vy, vz]。
            virtual (bool): 是否使用虚拟点体素化（nuScenes）。
        """
        super(DynamicVoxelEncoder, self).__init__()
        self.pc_range = torch.tensor(pc_range) 
        self.voxel_size = torch.tensor(voxel_size) 
        # 计算体素网格每个维度的数量。
        self.shape = torch.round((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size)
        self.shape_np = self.shape.numpy().astype(np.int32)
        self.virtual = virtual 

    @torch.no_grad()
    def forward(self, points):
        """对点云列表逐帧体素化，拼接为 batch 维度的体素特征与坐标。

        Args:
            points (list[Tensor]): 每帧一个点云。

        Returns:
            Tuple[Tensor, Tensor, np.ndarray]: (体素特征, 带 batch 索引的体素坐标, 网格形状)。
        """
        # points 为 list[torch.Tensor]，逐帧体素化。
        coors = []
        voxels = []  
        for res in points:
            if self.virtual:
                voxel, coor = voxelization_virtual(res, self.pc_range.to(res.device), self.voxel_size.to(res.device))
            else:
                voxel, coor = voxelization(res, self.pc_range.to(res.device), self.voxel_size.to(res.device))
            voxels.append(voxel)
            coors.append(coor)

        coors_batch = [] 
        for i in range(len(voxels)):
            # 在体素坐标前补一列 batch 索引，形成 (batch, z, y, x) 格式的坐标。
            coor_pad = F.pad(coors[i], (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)
        voxels_batch = torch.cat(voxels, dim=0)
        return voxels_batch, coors_batch, self.shape_np

