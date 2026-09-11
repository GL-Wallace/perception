"""微调（finetune）相关工具：冻结 BatchNorm。

提供 FrozenBatchNorm2d，用于在微调 / 冻结阶段把 BatchNorm 的统计量与仿射参数
固定，保持预训练模型的归一化行为。

主要类：
    - FrozenBatchNorm2d: 统计量与仿射参数固定的 BatchNorm2d。
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import Function
from torch.nn import functional as F
import logging

class FrozenBatchNorm2d(nn.Module):
    """固定的 BatchNorm2d：统计量与仿射参数均不随训练更新。

    内部保留 weight / bias / running_mean / running_var 作为不可训练 buffer，
    以恒等变换初始化；forward 使用固定统计量实现
    ``(x - running_mean) / sqrt(running_var) * weight + bias`` 的等价计算，
    常用于含预训练 BN 模型的微调场景。
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        """构造冻结的 BatchNorm2d。

        Args:
            num_features (int): 通道数。
            eps (float): 数值稳定项。
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # 以恒等变换初始化：weight=1、bias=0。
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        """使用固定统计量执行归一化与仿射。

        Args:
            x (Tensor): (N, C, H, W) 输入。

        Returns:
            Tensor: 归一化后的输出。
        """
        if x.requires_grad:
            # 需要梯度时，F.batch_norm 的反向会为 weight/bias 额外算梯度多占内存，
            # 这里手动展开归一化 + 仿射以节省显存。
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # 无需梯度时，F.batch_norm 是单个融合算子，更利于优化。
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """加载 state_dict 时做版本兼容处理。"""
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # 旧版本没有 running_mean/var，补充默认值以避免报 warning。
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # 版本 < 3 时保存的 running_var 不含 eps，这里减去 eps 以保持一致。
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        """返回模块描述字符串。"""
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """递归把模块中的 BatchNorm / SyncBatchNorm 转换为 FrozenBatchNorm。

        Args:
            module (nn.Module): 待转换的模块。

        Returns:
            如果 module 本身是 BN，返回新建的 FrozenBatchNorm；否则原地转换各个
            子模块后返回原模块。
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm1d, bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res