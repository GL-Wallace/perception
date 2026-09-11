"""通用杂项工具函数。

提供与具体任务解耦的小工具：张量转可视化图像、多输入并行映射以及把
子集数据反向展开(map back)到原始集合的 unmap。它们被检测头预测、损失
计算等下游流程复用。

主要函数：
    - tensor2imgs: 把归一化的图像张量批还原为 uint8 RGB 图像列表。
    - multi_apply: 把同一函数应用到多组输入并转置返回结果。
    - unmap: 把按索引子集计算的结果填回到完整集合对应位置。
"""
from functools import partial

import numpy as np
from det3d import torchie
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """把一批图像张量还原为 uint8 图像列表(便于可视化或保存)。

    Args:
        tensor (torch.Tensor): 形状 [B, C, H, W] 的归一化图像张量。
        mean (tuple): 归一化时减去的均值。
        std (tuple): 归一化时除以的标准差。
        to_rgb (bool): 是否转为 RGB 顺序(经 imdenormalize 的 to_bgr 参数控制)。

    Returns:
        list[np.ndarray]: 每张图的 uint8 数组，形状 [H, W, C]。
    """
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        # 单张图从 [C, H, W] 转为 [H, W, C]，再做去归一化
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = torchie.imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    """将 func 并行应用到多组输入，并把结果列表按输出维度转置。

    Args:
        func (callable): 要应用的函数。
        *args: 每组输入按位置展开传给 func(每组长度需一致)。
        **kwargs: 传给 func 的公共关键字参数。

    Returns:
        tuple: 每个输出维度一个列表，第 i 项是所有输入在该维度第 i 个结果。

    注意:
        常用于对每个 gt 目标并行调用同一函数后，把逐目标的多个返回量
        重组为按量分组的列表。
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """把按索引选出的子集数据填回到原始完整集合。

    Args:
        data (torch.Tensor): 子集数据，第一维为子集元素数。
        count (int): 原始集合大小。
        inds (torch.Tensor): 子集元素在原始集合中的索引。
        fill (int/float): 非子集位置填充的默认值。

    Returns:
        torch.Tensor: 大小为 count(或 count + 其余维度)的张量，子集位置
            填入 data，其余位置填 fill。
    """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
