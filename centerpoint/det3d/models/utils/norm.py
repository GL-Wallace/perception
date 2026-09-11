"""归一化层相关工具。

包含分布式同步 BatchNorm 的自实现（NaiveSyncBatchNorm）、AllReduce 自定义算子，
以及根据配置构建归一化层的 build_norm_layer 工厂函数。

主要类 / 函数：
    - AllReduce: 以 all_gather 求和、all_reduce 回传梯度的自定义算子。
    - NaiveSyncBatchNorm: 简单同步 BatchNorm 实现。
    - build_norm_layer: 按配置构建 BN / BN1d / GN 归一化层。
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from det3d.utils.dist import dist_common as comm
from torch.autograd.function import Function
from torch.nn import BatchNorm2d


class AllReduce(Function):
    """分布式统计量汇总的自定义算子（可微）。"""

    @staticmethod
    def forward(ctx, input):
        """前向：收集所有进程的输入并求和，得到全局统计量。"""
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # 使用 all_gather 而非 all_reduce，避免依赖原地操作。
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """反向：all_reduce 求和梯度，使各进程梯度一致。"""
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class NaiveSyncBatchNorm(BatchNorm2d):
    """简单同步 BatchNorm 实现。

    torch 自带的 SyncBatchNorm 在 batch 大小不平衡（如尺度增广、mask head）时
    可能精度下降甚至出现 NaN；这里手动 all_gather 各进程的均值与均方，再对各
    进程做统一的全局归一化，运行速度略慢但更稳定。
    """

    def forward(self, input):
        """训练且多进程时按全局统计量做同步归一化，否则退回普通 BN。

        Args:
            input (Tensor): (N, C, H, W) 输入。

        Returns:
            Tensor: 归一化后的输出。
        """
        if comm.get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty input"
        C = input.shape[1]
        # 计算本进程的均值与均方。
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        # all_gather 各进程统计量并按进程数求平均，得到全局均值与均方。
        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        # 更新 running 统计量。
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        # 用全局统计量对输入做归一化与仿射。
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias


norm_cfg = {
    # 归一化类型到 (名称缩写, 层类) 的映射。
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}


def build_norm_layer(cfg, num_features, postfix=""):
    """按配置构建归一化层。

    Args:
        cfg (dict): 归一化配置，需含 type 字段及对应层参数；可选 requires_grad
            控制参数是否参与梯度更新。
        num_features (int): 输入通道数。
        postfix (int, str): 拼接到名称后缀，用于命名生成的层。

    Returns:
        Tuple[str, nn.Module]: (层名称缩写 + 后缀, 创建的归一化层)。
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
