"""PyTorch 版旋转框张量算子。

提供基于 torch 的旋转框几何与坐标变换工具：从尺寸生成相对角点、绕轴旋转、
中心转角点、相机/雷达坐标系互转、图像投影，以及把 CenterPoint 的 7 维框
适配到 PCDet CUDA 旋转框 NMS 的 rotate_nms_pcdet。

主要函数：
    - corners_nd / corners_2d: 由各维尺寸与原点生成相对角点。
    - rotation_3d_in_axis / rotation_2d / rotate_points_along_z: 各种旋转变换。
    - center_to_corner_box3d / center_to_corner_box2d: 中心+尺寸+角度转角点。
    - camera_to_lidar / lidar_to_camera 及 box 版本: 相机与雷达坐标互转。
    - project_to_image: 3D 点投影到图像平面。
    - rotate_nms_pcdet: 调用 iou3d_nms_cuda 的旋转框 NMS。

与 box_np_ops 不同，本模块全程使用张量，服务于训练/推理的前向数据流；张量
算子内部用 einsum 做批量旋转以提高效率。
"""
import math
from functools import reduce

import numpy as np
import torch
from torch import stack as tstack
try:
    from det3d.ops.iou3d_nms import iou3d_nms_cuda, iou3d_nms_utils
except:
    print("iou3d cuda not built. You don't need this if you use circle_nms. Otherwise, refer to the advanced installation part to build this cuda extension")

def torch_to_np_dtype(ttype):
    """把 torch dtype 映射为对应的 numpy dtype。

    Args:
        ttype (torch.dtype): 输入的 torch 数据类型。

    Returns:
        np.dtype: 对应的 numpy 数据类型。

    注意:
        torch.float16 同时映射到 float16 与 float64(后者疑似历史遗留写法)，
        映射表缺失的类型会抛出 KeyError。
    """
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]


def corners_nd(dims, origin=0.5):
    """根据各维尺寸与原点生成相对盒角点(0/1 组合再线性映射)。

    Args:
        dims (torch.Tensor): 形状 [N, ndim]，每维的尺寸。
        origin (list or array or float): 原点相对最小角点的比例，0.5 表示中心。

    Returns:
        torch.Tensor: 形状 [N, 2**ndim, ndim] 的相对角点。
            布局示例(2d): x0y0, x0y1, x1y0, x1y1；(3d) 8 个角点为 x0< x1、
            y0< y1、z0< z1 的排列。

    注意:
        2d 面结果为顺时针(从最小点起)排列，3d 面做了 [0,1,3,2,4,5,7,6]
        重排以使角点顺序与后续面构造约定一致。
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    # unravel_index 生成 [2**ndim, ndim] 的 0/1 二进制角点索引，再按各维尺寸缩放
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    # 广播相乘：角点相对于原点的偏移乘上各维尺寸
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners


def corners_2d(dims, origin=0.5):
    """生成 2D 相对盒角点。

    Args:
        dims (torch.Tensor): 形状 [N, 2] 的各维尺寸。
        origin (list or array or float): 原点相对最小角点的比例。

    Returns:
        torch.Tensor: 形状 [N, 4, 2] 的角点，布局 x0y0, x0y1, x1y1, x1y0。
    """
    return corners_nd(dims, origin)


def corner_to_standup_nd(boxes_corner):
    """由角点求各轴外接(与坐标轴对齐的)最小/最大包围盒。

    Args:
        boxes_corner (torch.Tensor): 形状 [N, num_corners, ndim] 的角点。

    Returns:
        torch.Tensor: 形状 [N, 2*ndim]，前 ndim 列为各轴最小值，后 ndim 为最大值。
    """
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)


def rotation_3d_in_axis(points, angles, axis=0):
    """沿指定坐标轴批量旋转点集。

    Args:
        points (torch.Tensor): 形状 [N, point_size, 3] 的待旋转点。
        angles (torch.Tensor): 形状 [N] 的旋转角。
        axis (int): 0(x 轴)、1(y 轴)或 2/-1(z 轴)。

    Returns:
        torch.Tensor: 与 points 同形状的旋转结果。

    注意:
        返回结果用 einsum "aij,jka->aik" 表示 points @ rot_mat_T^T，即把
        旋转矩阵的转置从右侧作用到每个点上。
    """
    # points: [N, point_size, 3]
    # angles: [N]
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = tstack(
            [
                tstack([rot_cos, zeros, -rot_sin]),
                tstack([zeros, ones, zeros]),
                tstack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = tstack(
            [
                tstack([rot_cos, -rot_sin, zeros]),
                tstack([rot_sin, rot_cos, zeros]),
                tstack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = tstack(
            [
                tstack([zeros, rot_cos, -rot_sin]),
                tstack([zeros, rot_sin, rot_cos]),
                tstack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError("axis should in range")
    # print(points.shape, rot_mat_T.shape)
    return torch.einsum("aij,jka->aik", points, rot_mat_T)

def rotate_points_along_z(points, angle):
    """沿 z 轴旋转一批点(保留前 3 维后的附加特征不变)。

    Args:
        points (torch.Tensor): 形状 (B, N, 3 + C)，前 3 维为 xyz 坐标。
        angle (torch.Tensor): 形状 (B)，绕 z 轴角度，角度增大方向为 x -> y。

    Returns:
        torch.Tensor: 与 points 同形状，仅前 3 维被旋转。
    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    # 逐样本构造 3x3 绕 z 轴旋转矩阵
    rot_matrix = torch.stack((
        cosa,  -sina, zeros,
        sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    # 附加特征(如反射强度)原样拼接不参与旋转
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot


def rotation_2d(points, angles):
    """绕原点旋转 2D 点(角度为正时顺时针)。

    Args:
        points (torch.Tensor): 形状 [N, point_size, 2] 的点。
        angles (torch.Tensor): 形状 [N] 的旋转角。

    Returns:
        torch.Tensor: 与 points 同形状的旋转结果。
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack([tstack([rot_cos, -rot_sin]), tstack([rot_sin, rot_cos])])
    return torch.einsum("aij,jka->aik", (points, rot_mat_T))


def center_to_corner_box3d(centers, dims, angles, origin=(0.5, 0.5, 0.5), axis=1):
    """由中心、尺寸、角度把 kitti 风格框转为 8 角点。

    Args:
        centers (torch.Tensor): 形状 [N, 3] 的中心坐标。
        dims (torch.Tensor): 形状 [N, 3] 的尺寸。
        angles (torch.Tensor): 形状 [N] 的旋转角(绕 axis)。
        origin (list or array or float): 原点相对最小角点的比例，相机用
            [0.5, 1.0, 0.5]，雷达用 [0.5, 0.5, 0]。
        axis (int): 旋转轴，1 为相机、2 为雷达。

    Returns:
        torch.Tensor: 形状 [N, 8, 3] 的角点。
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """由中心、尺寸、角度把 2D 框转为 4 角点。

    Args:
        centers (torch.Tensor): 形状 [N, 2] 的中心坐标。
        dims (torch.Tensor): 形状 [N, 2] 的尺寸。
        angles (torch.Tensor): 形状 [N] 的旋转角，可为 None。
        origin (list or array or float): 原点比例。

    Returns:
        torch.Tensor: 形状 [N, 4, 2] 的角点。
    """
    # 'length' in kitti format is in x axis.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


def project_to_image(points_3d, proj_mat):
    """把 3D 点投影到图像平面(透视除法)。

    Args:
        points_3d (torch.Tensor): 形状 [..., 3] 的 3D 点。
        proj_mat (torch.Tensor): 3x4 投影矩阵(如 KITTI P2)。

    Returns:
        torch.Tensor: 形状 [..., 2] 的归一化像素坐标。
    """
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    # 齐次化后乘投影矩阵，再做透视除法得到像素坐标
    points_4 = torch.cat(
        [points_3d, torch.ones(*points_shape).type_as(points_3d)], dim=-1
    )
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def camera_to_lidar(points, r_rect, velo2cam):
    """相机坐标 -> 雷达坐标。

    Args:
        points (torch.Tensor): 形状 [N, 3] 的相机坐标点。
        r_rect (torch.Tensor): 3x3 整流旋转矩阵。
        velo2cam (torch.Tensor): 4x4 激光雷达到相机的外参(velo -> cam)。

    Returns:
        torch.Tensor: 形状 [N, 3] 的雷达坐标点。
    """
    num_points = points.shape[0]
    points = torch.cat([points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    lidar_points = points @ torch.inverse((r_rect @ velo2cam).t())
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    """雷达坐标 -> 相机坐标。

    Args:
        points (torch.Tensor): 形状 [N, 3] 的雷达坐标点。
        r_rect (torch.Tensor): 3x3 整流旋转矩阵。
        velo2cam (torch.Tensor): 4x4 激光雷达到相机的外参。

    Returns:
        torch.Tensor: 形状 [N, 3] 的相机坐标点。
    """
    num_points = points.shape[0]
    points = torch.cat([points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    camera_points = points @ (r_rect @ velo2cam).t()
    return camera_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    """相机框 -> 雷达框(含尺寸与 yaw 顺序调整)。

    相机框编码为 [x, y, z, l, h, w, r]，雷达框编码为 [x, y, z, w, l, h, r]，
    尺寸顺序由 lhw 与 wlh 的不同做重排。

    Args:
        data (torch.Tensor): 形状 [..., 7] 的相机坐标系框。
        r_rect, velo2cam: 见 camera_to_lidar。

    Returns:
        torch.Tensor: 形状 [..., 7] 的雷达坐标系框。
    """
    xyz = data[..., 0:3]
    l, h, w = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return torch.cat([xyz_lidar, w, l, h, r], dim=-1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    """雷达框 -> 相机框(含尺寸与 yaw 顺序调整)。

    Args:
        data (torch.Tensor): 形状 [..., 7] 的雷达坐标系框 [x, y, z, w, l, h, r]。
        r_rect, velo2cam: 见 lidar_to_camera。

    Returns:
        torch.Tensor: 形状 [..., 7] 的相机坐标系框 [x, y, z, l, h, w, r]。
    """
    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return torch.cat([xyz, l, h, w, r], dim=-1)


def rotate_nms_pcdet(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """把框适配到 PCDet 约定后调用 CUDA 旋转框 NMS。

    CenterPoint 的框编码为 [x, y, z, dx, dy, dz, yaw](dx 在 x 方向)，而
    PCDet/CUDA NMS 期待 [x, y, z, dx, dy, dz, yaw] 但 dx/dy 语义相反且 yaw
    定义为相对 y 轴，因此这里进行维度重排与 yaw 变换。

    Args:
        boxes (torch.Tensor): 形状 (N, 7) 的框 [x, y, z, dx, dy, dz, yaw]。
        scores (torch.Tensor): 形状 (N) 的分数。
        thresh (float): NMS 的 IoU 阈值。
        pre_maxsize (Optional[int]): 按分数取前 k 个后再做 NMS。
        post_max_size (Optional[int]): NMS 后最多保留的框数。

    Returns:
        torch.Tensor: 按分数降序排列的保留框索引。
    """
    # transform back to pcdet's coordinate
    # 交换 dx/dy 使长宽语义与 PCDet 一致，并把 yaw 映射到 PCDet 的约定
    boxes = boxes[:, [0, 1, 2, 4, 3, 5, -1]]
    boxes[:, -1] = -boxes[:, -1] - np.pi /2

    # 按分数降序排序，必要时先截断候选
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))

    if len(boxes) == 0:
        num_out =0
    else:
        # 调用 CUDA NMS，写入 keep 中前 num_out 个位置
        num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)

    selected = order[keep[:num_out].cuda()].contiguous()

    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected