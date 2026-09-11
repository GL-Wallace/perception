"""体素生成器。

封装点云到体素的转换参数（体素尺寸、点云范围、每体素最多点数、体素上限），
并调用底层 CUDA 算子 points_to_voxel 完成实际的稀疏体素划分。
"""
import numpy as np
from det3d.ops.point_cloud.point_cloud_ops import points_to_voxel


class VoxelGenerator:
    """将点云离散化为稀疏体素的生成器。

    保存体素化所需的全部几何参数，generate 时把无序点云分配到三维体素网格。

    Args:
        voxel_size (list/tuple): 每个体素的尺寸 [dx, dy, dz]。
        point_cloud_range (list/tuple): 点云范围 [xmin, ymin, zmin, xmax, ymax, zmax]。
        max_num_points (int): 每个体素最多保留的点数。
        max_voxels (int): 单帧最多保留的体素数。
    """

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # point_cloud_range 形如 [xmin, ymin, zmin, xmax, ymax, zmax]。
        voxel_size = np.array(voxel_size, dtype=np.float32)
        # 由点云范围与体素尺寸反推 x/y/z 三方向上的体素网格数量。
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels=-1):
        """执行体素化。

        Args:
            points (np.ndarray): 点云，shape 为 [N, >=3]，前 3 维为 x/y/z 坐标。
            max_voxels (int): 覆盖默认体素上限；-1 表示使用构造时配置的上限。

        Returns:
            voxels / coordinates / num_points 三者，具体结构与底层算子一致。
        """
        if max_voxels == -1:
            max_voxels = self._max_voxels

        return points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            max_voxels,
        )

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
