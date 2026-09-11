"""det3d.core.input 输入处理子包。

当前仅包含 voxel_generator 模块，提供体素生成器 VoxelGenerator，
负责把点云映射为体素编号与坐标，供特征构建阶段使用。
"""

from . import voxel_generator
