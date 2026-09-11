"""点云到体素的转换算子。

利用 numba JIT 将原始点云按体素大小划分到规则网格中，输出每个体素内的点、
体素坐标以及每个体素包含的点数，供后续 VoxelNet 风格的 3D 检测网络（如本仓库
的 CenterPoint / SECOND backbone）使用。

主要函数：
    - points_to_voxel: 点云转体素的主入口，返回体素、坐标与每体素点数。
    - _points_to_voxel_kernel / _points_to_voxel_reverse_kernel: JIT 内核，
      分别输出 xyz 与 zyx 顺序的体素坐标。
    - bound_points_jit: 判定点是否落在给定范围内。

设计思路：
    体素化在单个循环内完成，避免在 JIT 代码中创建大数组以控制内存与性能。
    coor_to_voxelidx 表记录每个体素坐标对应的体素索引，减少查找开销。
"""

import time

import numba
import numpy as np


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(
    points,
    voxel_size,
    coors_range,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=35,
    max_voxels=20000,
):
    """体素化 JIT 内核：输出 zyx 顺序的体素坐标。

    与 _points_to_voxel_kernel 的区别在于，坐标按 z/y/x 倒序写入 coor，
    用于需要 reverse_index 的场景（体素坐标与点云特征维度顺序不同）。

    Returns:
        int: 实际生成的体素数量（可能小于 max_voxels）。
    """
    # 所有计算放在单个循环中完成，避免在 JIT 主体中创建大数组以降低开销。
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3,), dtype=np.int32)
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
            # 首次遇到该体素：分配新索引并记录坐标。
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue 
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel(
    points,
    voxel_size,
    coors_range,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxels,
    coors,
    max_points=35,
    max_voxels=20000,
):
    """体素化 JIT 内核：输出 xyz 顺序的体素坐标。

    对每个点计算体素索引并填充进对应体素；超过 max_points 或 max_voxels 会丢弃。

    Returns:
        int: 实际生成的体素数量。
    """
    # 若在 CUDA 上写入需要互斥锁，但 numba.cuda 不支持互斥锁；
    # 且 PyTorch dataloader 不支持 CUDA，故这里运行在 CPU 上。
    # 所有计算放在单个循环中完成，避免在 JIT 主体中创建大数组导致性能下降。
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3,), dtype=np.int32)
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
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            # 首次遇到该体素：分配新索引并记录坐标。
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue 
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(
    points, voxel_size, coors_range, max_points=35, reverse_index=True, max_voxels=20000
):
    """将 KITTI 点云 (N, >=3) 转换为体素。

    此版本在单个循环中完成全部计算：在 3.2GHz CPU 上加 JIT 仅需约 4.2ms
    （不含其他特征计算）。注意：本函数在 Ubuntu 上通常比 Windows 10 更快。

    Args:
        points: [N, ndim] 浮点张量，points[:, :3] 为 xyz 坐标，
            points[:, 3:] 为反射率等其他信息。
        voxel_size: [3] list/tuple/array，xyz 三个方向的体素尺寸。
        coors_range: [6] list/tuple/array，体素范围，格式为 xyzxyz（前 3 个为最小角，
            后 3 个为最大角）。
        max_points: int，单个体素最多包含的点数。
        reverse_index: bool，是否返回倒序坐标。若点为 xyz 格式且 reverse_index 为 True，
            输出坐标将为 zyx 格式；但特征中的点仍保持 xyz 格式。
        max_voxels: int，本函数最多创建的体素数量。对 SECOND 而言 20000 是合适取值；
            由于体素数量受限可能丢弃部分点，调用前建议先对点云做 shuffle。

    Returns:
        voxels: [M, max_points, ndim] 浮点张量，仅包含点数据。
        coordinates: [M, 3] int32 张量，体素坐标。
        num_points_per_voxel: [M] int32 张量，每个体素实际包含的点数。
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # 不在 JIT(nopython=True) 代码中创建大数组。
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype
    )
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    else:
        voxel_num = _points_to_voxel_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    """判断点是否位于给定范围内。

    Args:
        points: [N, ndim] 点坐标数组。
        upper_bound: [ndim] 各维度上界（不包含）。
        lower_bound: [ndim] 各维度下界（包含）。

    Returns:
        np.ndarray: [N] int32 数组，1 表示点在范围内，0 表示越界。
    """
    # nopython=True 下 numba 不支持 np.bool，因此用 int32 标记，
    # 需要在函数返回后自行转换为 bool。
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N,), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices
