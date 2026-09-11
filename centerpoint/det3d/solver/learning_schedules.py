"""TensorFlow Object Detection API 学习率调度的 PyTorch 版本。

提供基于 torch.optim.Optimizer 的学习率调度器基类 _LRSchedulerStep 与
Constant、ManualStepping、指数衰减（含 warmup）、余弦衰减（含 warmup）等策略。

主要类：
    - _LRSchedulerStep: 学习率调度基类。
    - Constant: 恒定学习率。
    - ManualStepping: 手动分段学习率。
    - ExponentialDecayWithBurnin / ExponentialDecay: 指数衰减（前者带 warmup）。
    - CosineDecayWithWarmup: 余弦衰减（带 warmup）。
"""
import numpy as np
from torch.optim.optimizer import Optimizer


class _LRSchedulerStep(object):
    """学习率调度基类。

    记录各参数组初始学习率，并在 step() 时根据子类计算的新学习率更新优化器。

    Args:
        optimizer: torch.optim.Optimizer 实例。
        last_step: 上一次的步数，-1 表示首次初始化。
    """

    def __init__(self, optimizer, last_step=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_step == -1:
            # 首次初始化：把当前 lr 记录为 initial_lr。
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            # 恢复训练：要求参数组中已存在 initial_lr。
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.step(last_step + 1)
        self.last_step = last_step

    """
    def get_lr(self):
        raise NotImplementedError
    """

    def get_lr(self):
        ret = [self._get_lr_per_group(base_lr) for base_lr in self.base_lrs]
        return ret

    def _get_lr_per_group(self, base_lr):
        raise NotImplementedError

    def step(self, step=None):
        """推进调度器一步，为每个参数组设置新的学习率。"""
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class Constant(_LRSchedulerStep):
    """恒定学习率调度：始终保持 base_lr 不变。"""

    def __init__(self, optimizer, last_step=-1):
        super().__init__(optimizer, last_step)

    def _get_lr_per_group(self, base_lr):
        return base_lr


class ManualStepping(_LRSchedulerStep):
    """TensorFlow manual_stepping 的 PyTorch 版本。不支持参数组。"""

    def __init__(self, optimizer, boundaries, rates, last_step=-1):
        self._boundaries = boundaries
        self._num_boundaries = len(boundaries)
        self._learning_rates = rates

        if any([b < 0 for b in boundaries]) or any(
            [not isinstance(b, int) for b in boundaries]
        ):
            raise ValueError("boundaries must be a list of positive integers")
        if any([bnext <= b for bnext, b in zip(boundaries[1:], boundaries[:-1])]):
            raise ValueError("Entries in boundaries must be strictly increasing.")
        if any([not isinstance(r, float) for r in rates]):
            raise ValueError("Learning rates must be floats")
        if len(rates) != len(boundaries) + 1:
            raise ValueError(
                "Number of provided learning rates must exceed "
                "number of boundary points by exactly 1."
            )
        super().__init__(optimizer, last_step)

    def _get_lr_per_group(self, base_lr):
        step = self.last_step
        ret = None
        for i, bound in enumerate(self._boundaries):
            if step > bound:
                ret = self._learning_rates[i + 1]
        if ret is not None:
            return ret
        return self._learning_rates[0]


class ExponentialDecayWithBurnin(_LRSchedulerStep):
    """带 warmup（burnin）的指数衰减学习率调度。"""

    def __init__(
        self,
        optimizer,
        learning_rate_decay_steps,
        learning_rate_decay_factor,
        burnin_learning_rate,
        burnin_steps,
        last_step=-1,
    ):
        self._decay_steps = learning_rate_decay_steps
        self._decay_factor = learning_rate_decay_factor
        self._burnin_learning_rate = burnin_learning_rate
        self._burnin_steps = burnin_steps

        super().__init__(optimizer, last_step)

    def _get_lr_per_group(self, base_lr):
        if self._burnin_learning_rate == 0:
            burnin_learning_rate = base_lr
        step = self.last_step
        # 指数衰减后的学习率（注意：^ 为位异或，这里原样保留原实现）。
        post_burnin_learning_rate = base_lr * self._decay_factor ^ (
            step // self._decay_steps
        )
        if step < self._burnin_steps:
            return burnin_learning_rate
        else:
            return post_burnin_learning_rate


class ExponentialDecay(_LRSchedulerStep):
    """指数衰减学习率调度。

    Args:
        optimizer: torch.optim.Optimizer 实例。
        learning_rate_decay_steps: 衰减周期步数。
        learning_rate_decay_factor: 衰减因子。
        staircase: 是否阶梯式衰减。
        last_step: 上一次步数。
    """

    def __init__(
        self,
        optimizer,
        learning_rate_decay_steps,
        learning_rate_decay_factor,
        staircase=True,
        last_step=-1,
    ):
        self._decay_steps = learning_rate_decay_steps
        self._decay_factor = learning_rate_decay_factor
        self._staircase = staircase

        super().__init__(optimizer, last_step)

    def _get_lr_per_group(self, base_lr):
        step = self.last_step
        if self._staircase:
            # 阶梯式：每 decay_steps 步指数衰减一次。
            post_burnin_learning_rate = base_lr * pow(
                self._decay_factor, (step // self._decay_steps)
            )
        else:
            # 连续式：按步数比例连续衰减。
            post_burnin_learning_rate = base_lr * pow(
                self._decay_factor, (step / self._decay_steps)
            )

        return post_burnin_learning_rate


class CosineDecayWithWarmup(_LRSchedulerStep):
    """带 warmup 的余弦退火学习率调度。

    warmup 阶段从 warmup_learning_rate 线性升温到 base_lr，之后按余弦退火衰减。

    Args:
        optimizer: torch.optim.Optimizer 实例。
        total_steps: 调度总步数。
        warmup_learning_rate: warmup 起始学习率。
        warmup_steps: warmup 阶段的步数。
        last_step: 上一次步数。
    """

    def __init__(
        self, optimizer, total_steps, warmup_learning_rate, warmup_steps, last_step=-1
    ):
        if total_steps < warmup_steps:
            raise ValueError("total_steps must be larger or equal to " "warmup_steps.")
        self._total_steps = total_steps
        self._warmup_learning_rate = warmup_learning_rate
        self._warmup_steps = warmup_steps

        super().__init__(optimizer, last_step)

    def _get_lr_per_group(self, base_lr):
        if base_lr < self._warmup_learning_rate:
            raise ValueError(
                "learning_rate_base must be larger " "or equal to warmup_learning_rate."
            )

        step = self.last_step
        # 余弦退火阶段的学习率。
        learning_rate = (
            0.5
            * base_lr
            * (
                1
                + np.cos(
                    np.pi
                    * (float(step) - self._warmup_steps)
                    / float(self._total_steps - self._warmup_steps)
                )
            )
        )
        if self._warmup_steps > 0:
            # warmup 阶段：线性插值升温。
            slope = (base_lr - self._warmup_learning_rate) / self._warmup_steps
            pre_cosine_learning_rate = slope * float(step) + self._warmup_learning_rate
            if step < self._warmup_steps:
                return pre_cosine_learning_rate
            else:
                return learning_rate
