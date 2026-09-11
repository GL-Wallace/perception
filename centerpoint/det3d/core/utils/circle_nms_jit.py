"""圆形距离非极大值抑制(NMS)。

基于 numba 加速的 2D 中心距离 NMS：按分数从高到低遍历检测框，若某个框
与已保留框的中心欧氏距离平方小于阈值则被抑制。与基于 IoU 的旋转框 NMS
不同，这里用圆形判据，属于 CenterPoint 官方提供的轻量后处理方案。

主要函数：
    - circle_nms: 对给定检测结果执行圆形距离 NMS，返回保留的索引。

注意:
    距离判据用的是平方距离 (dx^2 + dy^2) 与阈值 thresh 直接比较，因此
    调用方传入的 thresh 通常是距离平方的阈值。
"""
import numba 
import numpy as np 

@numba.jit(nopython=True)
def circle_nms(dets, thresh):
    """执行圆形距离 NMS 并返回保留框索引。

    Args:
        dets (np.ndarray): 形状 [N, >=3] 的检测结果，前三列依次为
            x、y 中心坐标与置信分数。
        thresh (float): 抑制判据阈值，为距离平方阈值(单位与 x/y 一致)。

    Returns:
        list: 保留下来的检测框索引(按分数降序排列)。

    注意:
        返回值为 Python list，供后续用整数索引选取保留框。
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    # 分数从高到低排序，得到遍历顺序索引
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    # 抑制标记数组，1 表示该框已被更高分框抑制
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            # 计算两框中心距离的平方，避免开根号
            dist = (x1[i]-x1[j])**2 + (y1[i]-y1[j])**2

            # ovr = inter / areas[j]
            # 中心距离平方小于阈值则抑制分数较低的框
            if dist <= thresh:
                suppressed[j] = 1
    return keep
