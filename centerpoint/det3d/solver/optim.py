"""混合精度优化器包装。

提供 param_fp32_copy / set_grad 等 FP32 主权重与梯度拷贝工具，以及
MixedPrecisionWrapper：将优化器参数复制为 FP32 后进行梯度缩放与参数更新，
再写回原模型，实现混合精度训练。
"""

from collections import Iterable, defaultdict
from copy import deepcopy
from itertools import chain

import torch
from torch.autograd import Variable

required = object()


def param_fp32_copy(params):
    """把参数列表复制为 FP32 CUDA 张量，并开启 requires_grad。"""
    param_copy = [
        param.clone().type(torch.cuda.FloatTensor).detach() for param in params
    ]
    for param in param_copy:
        param.requires_grad = True
    return param_copy


def set_grad(params, params_with_grad, scale=1.0):
    """把 params_with_grad 的梯度拷贝到 params（可选先除以 scale）。

    Returns:
        bool: 若梯度中出现 nan/inf 返回 True（表示无效梯度）。
    """
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(
                param.data.new().resize_(*param.data.size())
            )
        grad = param_w_grad.grad.data
        if scale is not None:
            grad /= scale
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            return True  # 无效梯度
        param.grad.data.copy_(grad)
    return False


class MixedPrecisionWrapper(object):
    """混合精度优化器包装。

    Arguments:
        optimizer (torch.optim.Optimizer): torch.optim.Optimizer 的实例。
        scale (float): 梯度缩放系数。
        auto_scale (bool): 是否启用自动缩放。自动缩放算法参见
            http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
    """

    def __init__(
        self,
        optimizer,
        scale=None,
        auto_scale=True,
        inc_factor=2.0,
        dec_factor=0.5,
        num_iters_be_stable=500,
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("must provide a torch.optim.Optimizer")
        self.optimizer = optimizer
        if hasattr(self.optimizer, "name"):
            self.name = self.optimizer.name  # 供 checkpoint 系统使用
        param_groups_copy = []
        for i, group in enumerate(optimizer.param_groups):
            # 拷贝参数组配置（不含 params），并将参数替换为 FP32 副本。
            group_copy = {n: v for n, v in group.items() if n != "params"}
            group_copy["params"] = param_fp32_copy(group["params"])
            param_groups_copy.append(group_copy)

        # 替换优化器参数组，可能有一定风险。
        self.param_groups = optimizer.param_groups
        optimizer.param_groups = param_groups_copy
        self.grad_scale = scale
        self.auto_scale = auto_scale
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        self.stable_iter_count = 0
        self.num_iters_be_stable = num_iters_be_stable

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        return self.optimizer.__setstate__(state)

    def __repr__(self):
        return self.optimizer.__repr__()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        """执行一步优化。

        先将原始参数的梯度按 grad_scale 缩放并拷贝到 FP32 参数组，出现无效梯度时
        按 dec_factor 缩小 grad_scale 并跳过本轮；否则执行 FP32 优化后写回原参数。
        """
        for g, g_copy in zip(self.param_groups, self.optimizer.param_groups):
            invalid = set_grad(g_copy["params"], g["params"], self.grad_scale)
            if invalid:
                if self.grad_scale is None or self.auto_scale is False:
                    raise ValueError("nan/inf detected but auto_scale disabled.")
                self.grad_scale *= self.dec_factor
                print("scale decay to {}".format(self.grad_scale))
                return
        if self.auto_scale is True:
            # 连续稳定若干步后上调 grad_scale。
            self.stable_iter_count += 1
            if self.stable_iter_count > self.num_iters_be_stable:
                if self.grad_scale is not None:
                    self.grad_scale *= self.inc_factor
                self.stable_iter_count = 0

        if closure is None:
            self.optimizer.step()
        else:
            self.optimizer.step(closure)
        # 把 FP32 优化结果写回原始参数。
        for g, g_copy in zip(self.param_groups, self.optimizer.param_groups):
            for p_copy, p in zip(g_copy["params"], g["params"]):
                p.data.copy_(p_copy.data)
