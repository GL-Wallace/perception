# The following code are copied from pytorch_scatter https://github.com/rusty1s/pytorch_scatter
# Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
# MIT License 
"""点/体素特征的散布(Scatter)操作。

从 pytorch_scatter 移植而来，提供按索引把源张量累加/求均值散布到目标张量
指定维度上的原语。体素稀疏特征应用中常需要在 NCHW 与 NHWC 间切换，这里的
broadcast 负责把索引对齐到源张量维度，scatter_sum/scatter_mean 完成归约。

主要函数：
    - broadcast: 将源张量在给定维度上广播到与目标张量同形状。
    - scatter_sum: 沿指定维按索引对源张量做加和散布。
    - scatter_mean: 沿指定维按索引对源张量做均值散布。

全部函数使用 torch.jit.script 编译以提升运行效率。
"""
from typing import Optional, Tuple
import torch 

@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """把 src 广播到与 other 相同的形状。

    先按 dim 的语义在 src 头部/尾部补齐维度，再用 expand_as 复制数据。

    Args:
        src (torch.Tensor): 待广播的源张量(通常为索引或计数)。
        other (torch.Tensor): 目标形状参考张量。
        dim (int): 需要对齐的维度，可为负数(按 other.dim() 取模)。

    Returns:
        torch.Tensor: 与 other 形状一致的广播结果张量。
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        # 一维索引需在 dim 之前补足 dim 个前导维度
        for _ in range(dim):
            src = src.unsqueeze(0)
    # 其余在末尾补齐到与 other 相同的维数
    for _ in range(other.dim()-src.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """沿 dim 维度按索引对 src 做加和散布。

    Args:
        src (torch.Tensor): 源数据张量。
        index (torch.Tensor): 与 src 对齐的整数索引，决定每个元素落到目标哪个位置。
        dim (int): 散布的维度，默认为最后一维。
        out (Optional[torch.Tensor]): 可选的输出张量，就地累加。
        dim_size (Optional[int]): 目标维度大小，缺省取 index 最大值加一。

    Returns:
        torch.Tensor: 散布累加后的结果张量。

    注意:
        依赖 torch.scatter_add_，其原位语义意味着同一索引多次出现会被累加。
    """
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

@torch.jit.script
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    """沿 dim 维度按索引对 src 做均值散布。

    实现思路：先做 scatter_sum 得到求和结果，再统计每个索引位置出现的次数
    (对全一张量做 scatter_sum)，最后用除法得到均值。

    Args:
        src (torch.Tensor): 源数据张量。
        index (torch.Tensor): 与 src 对齐的整数索引。
        dim (int): 散布的维度。
        out (Optional[torch.Tensor]): 可选输出张量。
        dim_size (Optional[int]): 目标维度大小。

    Returns:
        torch.Tensor: 散布求均值后的结果张量。

    注意:
        计数张量先 clamp(1) 以避免除零；对浮点张量使用原地除法，
        整型张量路径被 assert 拦截(未实现)。
    """
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    # 用全一张量统计每个目标位置被写入的次数，作为求平均的分母
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count.clamp_(1)
    count = broadcast(count, out, dim)
    if torch.is_floating_point(out):
        out.div_(count)
    else:
        assert 0 
        # out.floor_divide_(count)
    return out