"""点云到 BEV 视图的转换算子。

基于体素化思路，将点云转换为鸟瞰图（BEV）特征图：沿高度方向划分为多个柱状
切片，每列统计点数密度与最高点归一化高度，可选叠加反射率通道。输出 [C, H, W]
特征图，供基于 BEV 的检测网络使用。

主要函数：
    - points_to_bev: 点云转 BEV 特征图的主入口。
    - _points_to_bevmap_reverse_kernel: JIT 内核，逐点填充高度切片与反射率。
"""

import math

import numba
import numpy as np


@numba.jit(nopython=True)
def _points_to_bevmap_reverse_kernel(
    points,
    voxel_size,
    coors_range,
    coor_to_voxelidx,
    # coors_2d,
    bev_map,
    height_lowers,
    # density_norm_num=16,
    with_reflectivity=False,
    max_voxels=40000,
):
    """BEV 地图填充 JIT 内核。

    逐个点计算其体素索引，更新对应柱的密度计数与最高点归一化高度，
    with_reflectivity 为 True 时同时写入反射率。

    Args:
        points: [N, 4] 点云（xyz + 反射率）。
        voxel_size: [3] 体素尺寸（对应 xyz）。
        coors_range: [6] 体素范围，格式 xyzxyz。
        coor_to_voxelidx: [D, H, W] 体素索引表。
        bev_map: 待填充的 BEV 特征图，末尾通道为密度计数，反射率通道在倒数第二。
        height_lowers: [D] 各高度切片的下边界。
        with_reflectivity: 是否写入反射率通道。
        max_voxels: 最多容纳的体素数量。
    """
    # 所有计算放在单个循环中完成，避免在 JIT 主体中创建大数组导致性能下降。
    N = points.shape[0]
    ndim = points.shape[1] - 1
    # ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    height_slice_size = voxel_size[-1]
    coor = np.zeros(shape=(3,), dtype=np.int32)  # DHW
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            # 计算点在体素网格中的索引，越界则跳过该点。
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            # 首次遇到该体素：分配新索引。
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # coors_2d[voxelidx] = coor[1:]
        # 最后一个通道累加密度计数。
        bev_map[-1, coor[1], coor[2]] += 1
        height_norm = bev_map[coor[0], coor[1], coor[2]]
        # 计算当前点相对所在柱底部的高度归一化值。
        incomimg_height_norm = (
            points[i, 2] - height_lowers[coor[0]]
        ) / height_slice_size
        if incomimg_height_norm > height_norm:
            # 保留该柱内最高点对应的归一化高度与反射率。
            bev_map[coor[0], coor[1], coor[2]] = incomimg_height_norm
            if with_reflectivity:
                bev_map[-2, coor[1], coor[2]] = points[i, 3]
    # return voxel_num


def points_to_bev(
    points,
    voxel_size,
    coors_range,
    with_reflectivity=False,
    density_norm_num=16,
    max_voxels=40000,
):
    """将 KITTI 点云 (N, 4) 转换为 BEV 地图，返回 [C, H, W] 特征图。

    此函数基于 points_to_voxel 的算法。在体素尺寸为 [0.1, 0.1, 0.8] 的降采样
    点云上约需 5ms。

    Args:
        points: [N, ndim] 浮点张量，points[:, :3] 为 xyz 坐标，
            points[:, 3] 为反射率。
        voxel_size: [3] list/tuple/array，xyz 三个方向的体素尺寸。
        coors_range: [6] list/tuple/array，体素范围，格式 xyzxyz（minmax）。
        with_reflectivity: bool，为 True 时在 BEV 地图中追加反射率（intensity）通道。
        density_norm_num: 密度归一化参考点数，当前实现中未在 kernel 中使用。
        max_voxels: 最多容纳的体素数量。

    Returns:
        bev_map: [num_height_maps + 1(或 +2), H, W] 浮点张量。
            注意：bev_map[-1] 是点数计数图而非密度图，因为密度图在 CPU 上计算
            比 GPU 更耗时；with_reflectivity 为 True 时 bev_map[-2] 为反射率图。
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]  # DHW format
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # coors_2d = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
    bev_map_shape = list(voxelmap_shape)
    bev_map_shape[0] += 1
    # 计算每个高度切片的下边界，用于后续高度归一化。
    height_lowers = np.linspace(
        coors_range[2], coors_range[5], voxelmap_shape[0], endpoint=False
    )
    if with_reflectivity:
        bev_map_shape[0] += 1
    bev_map = np.zeros(shape=bev_map_shape, dtype=points.dtype)
    _points_to_bevmap_reverse_kernel(
        points,
        voxel_size,
        coors_range,
        coor_to_voxelidx,
        bev_map,
        height_lowers,
        with_reflectivity,
        max_voxels,
    )
    return bev_map
