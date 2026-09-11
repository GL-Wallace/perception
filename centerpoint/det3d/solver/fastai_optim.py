"""fastai 风格优化器工具与包装。

提供 FP16 模型参数 / FP32 主参数（master）的管理工具，以及 OptimWrapper /
FastAIMixedOptim 优化器包装类，支持权重衰减、动量和混合精度训练。

主要函数/类：
    - split_bn_bias: 将层按 BN 与非 BN 拆分成两组。
    - get_master: 生成 FP16 模型参数与 FP32 主参数列表。
    - model_g2master_g / master2model: 模型与主参数间的梯度/参数拷贝。
    - OptimWrapper: 优化器包装，简化超参数修改。
    - FastAIMixedOptim: 混合精度优化器包装（FP32 主权重）。
"""

from collections import Iterable, defaultdict
from copy import deepcopy
from itertools import chain

import torch
from torch import nn
from torch._utils import _unflatten_dense_tensors
from torch.autograd import Variable
from torch.nn.utils import parameters_to_vector
try:
    from apex.parallel.optimized_sync_batchnorm import SyncBatchNorm
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.modules.batchnorm._BatchNorm, SyncBatchNorm)
except:
    print('no apex')
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,nn.modules.batchnorm._BatchNorm)

def split_bn_bias(layer_groups):
    """把每组层拆分为非 BN 组与 BN 组（bn_types）。

    BN 层通常不施加权重衰减，分拆后便于分别设置超参数。
    """
    split_groups = []
    for l in layer_groups:
        l1, l2 = [], []
        for c in l.children():
            if isinstance(c, bn_types):
                l2.append(c)
            else:
                l1.append(c)
        split_groups += [nn.Sequential(*l1), nn.Sequential(*l2)]
    return split_groups


def get_master(layer_groups, flat_master: bool = False):
    """返回两个列表：FP16 模型参数与 FP32 主参数。"""
    split_groups = split_bn_bias(layer_groups)
    model_params = [
        [param for param in lg.parameters() if param.requires_grad]
        for lg in split_groups
    ]
    if flat_master:
        # flat_master：每组参数展平为一个 1D 主参数。
        master_params = []
        for lg in model_params:
            if len(lg) != 0:
                mp = parameters_to_vector([param.data.float() for param in lg])
                mp = torch.nn.Parameter(mp, requires_grad=True)
                if mp.grad is None:
                    mp.grad = mp.new(*mp.size())
                master_params.append([mp])
            else:
                master_params.append([])
        return model_params, master_params
    else:
        # 非 flat：逐参数复制为 FP32 主参数。
        master_params = [
            [param.clone().float().detach() for param in lg] for lg in model_params
        ]
        for mp in master_params:
            for param in mp:
                param.requires_grad = True
        return model_params, master_params


def model_g2master_g(model_params, master_params, flat_master: bool = False) -> None:
    """把模型参数的梯度拷贝到主参数，供优化器更新使用。"""
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(master_group) != 0:
                master_group[0].grad.data.copy_(
                    parameters_to_vector([p.grad.data.float() for p in model_group])
                )
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group):
                if model.grad is not None:
                    if master.grad is None:
                        master.grad = master.data.new(*master.data.size())
                    master.grad.data.copy_(model.grad.data)
                else:
                    master.grad = None


def master2model(model_params, master_params, flat_master: bool = False) -> None:
    """把主参数拷贝回模型参数。"""
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(model_group) != 0:
                for model, master in zip(
                    model_group,
                    _unflatten_dense_tensors(master_group[0].data, model_group),
                ):
                    model.data.copy_(master)
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group):
                model.data.copy_(master.data)


def listify(p=None, q=None):
    """把 p 转换为列表，并使其长度与 q 一致（单元素时复制扩展）。"""
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1:
        p = p * n
    assert len(p) == n, f"List len mismatch ({len(p)} vs {n})"
    return list(p)


def trainable_params(m: nn.Module):
    """返回模块 m 中所有可训练参数。"""
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res


def is_tuple(x) -> bool:
    return isinstance(x, tuple)


# 拷贝自 fastai。
class OptimWrapper:
    """优化器基础包装，简化超参数（lr/mom/wd/beta）的读取与修改。"""

    def __init__(self, opt, wd, true_wd: bool = False, bn_wd: bool = True):
        self.opt, self.true_wd, self.bn_wd = opt, true_wd, bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove("params")
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func, lr, layer_groups, **kwargs):
        """用 opt_func 创建优化器，并为各层组设置学习率。"""
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{"params": trainable_params(l), "lr": 0} for l in split_groups])
        opt = cls(opt, **kwargs)
        opt.lr, opt.opt_func = listify(lr, layer_groups), opt_func
        return opt

    def new(self, layer_groups):
        """基于自身超参数，用另一组层创建新的 OptimWrapper。"""
        opt_func = getattr(self, "opt_func", self.opt.__class__)
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{"params": trainable_params(l), "lr": 0} for l in split_groups])
        return self.create(
            opt_func,
            self.lr,
            layer_groups,
            wd=self.wd,
            true_wd=self.true_wd,
            bn_wd=self.bn_wd,
        )

    def __repr__(self) -> str:
        return f"OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}"

    # PyTorch 优化器方法
    def step(self) -> None:
        """设置权重衰减并执行优化器更新。"""
        # 在优化器外部施加权重衰减（AdamW 风格）
        if self.true_wd:
            for lr, wd, pg1, pg2 in zip(
                self._lr,
                self._wd,
                self.opt.param_groups[::2],
                self.opt.param_groups[1::2],
            ):
                for p in pg1["params"]:
                    p.data.mul_(1 - wd * lr)
                if self.bn_wd:
                    for p in pg2["params"]:
                        p.data.mul_(1 - wd * lr)
            self.set_val("weight_decay", listify(0, self._wd))
        self.opt.step()

    def zero_grad(self) -> None:
        """清零优化器梯度。"""
        self.opt.zero_grad()

    # 转发给内部优化器
    def __getattr__(self, k: str):
        return getattr(self.opt, k, None)

    def clear(self):
        """重置内部优化器的状态。"""
        sd = self.state_dict()
        sd["state"] = {}
        self.load_state_dict(sd)

    # 超参数以属性形式暴露
    @property
    def lr(self) -> float:
        return self._lr[-1]

    @lr.setter
    def lr(self, val: float) -> None:
        self._lr = self.set_val("lr", listify(val, self._lr))

    @property
    def mom(self) -> float:
        return self._mom[-1]

    @mom.setter
    def mom(self, val: float) -> None:
        if "momentum" in self.opt_keys:
            self.set_val("momentum", listify(val, self._mom))
        elif "betas" in self.opt_keys:
            self.set_val("betas", (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self) -> float:
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val: float) -> None:
        """设置 beta（对某些优化器而言是 alpha）。"""
        if val is None:
            return
        if "betas" in self.opt_keys:
            self.set_val("betas", (self._mom, listify(val, self._beta)))
        elif "alpha" in self.opt_keys:
            self.set_val("alpha", listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self) -> float:
        return self._wd[-1]

    @wd.setter
    def wd(self, val: float) -> None:
        """设置权重衰减。"""
        if not self.true_wd:
            self.set_val("weight_decay", listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    # 辅助函数
    def read_defaults(self) -> None:
        """从优化器读取各超参数默认值。"""
        self._beta = None
        if "lr" in self.opt_keys:
            self._lr = self.read_val("lr")
        if "momentum" in self.opt_keys:
            self._mom = self.read_val("momentum")
        if "alpha" in self.opt_keys:
            self._beta = self.read_val("alpha")
        if "betas" in self.opt_keys:
            self._mom, self._beta = self.read_val("betas")
        if "weight_decay" in self.opt_keys:
            self._wd = self.read_val("weight_decay")

    def set_val(self, key: str, val, bn_groups: bool = True):
        """在优化器字典的 key 位置设置值 val。"""
        if is_tuple(val):
            val = [(v1, v2) for v1, v2 in zip(*val)]
        for v, pg1, pg2 in zip(
            val, self.opt.param_groups[::2], self.opt.param_groups[1::2]
        ):
            pg1[key] = v
            if bn_groups:
                pg2[key] = v
        return val

    def read_val(self, key: str):
        """读取优化器字典中的超参数 key。"""
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if is_tuple(val[0]):
            val = [o[0] for o in val], [o[1] for o in val]
        return val


class FastAIMixedOptim(OptimWrapper):
    """混合精度优化器：模型保留 FP16 权重，优化在 FP32 主参数上进行。"""

    @classmethod
    def create(
        cls,
        opt_func,
        lr,
        layer_groups,
        model,
        flat_master=False,
        loss_scale=512.0,
        **kwargs,
    ):
        """用 opt_func 创建混合精度优化器，并设置各层组学习率。"""
        opt = OptimWrapper.create(opt_func, lr, layer_groups, **kwargs)
        opt.model_params, opt.master_params = get_master(layer_groups, flat_master)
        opt.flat_master = flat_master
        opt.loss_scale = loss_scale
        opt.model = model
        # 重排优化器参数，使优化步在 FP32 上完成。
        # opt = self.learn.opt
        mom, wd, beta = opt.mom, opt.wd, opt.beta
        lrs = [lr for lr in opt._lr for _ in range(2)]
        opt_params = [
            {"params": mp, "lr": lr} for mp, lr in zip(opt.master_params, lrs)
        ]
        opt.opt = opt_func(opt_params)
        opt.mom, opt.wd, opt.beta = mom, wd, beta
        return opt

    def step(self):
        """执行一步混合精度优化。

        将模型梯度拷贝到主参数、按 loss_scale 缩放后更新，再把主参数写回模型。
        """
        model_g2master_g(self.model_params, self.master_params, self.flat_master)
        for group in self.master_params:
            for param in group:
                param.grad.div_(self.loss_scale)
        super(FastAIMixedOptim, self).step()
        self.model.zero_grad()
        # 将 FP32 主参数更新结果写回模型。
        master2model(self.model_params, self.master_params, self.flat_master)
