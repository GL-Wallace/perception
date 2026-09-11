"""学习率更新 Hook。

提供支持 warmup 的基类 LrUpdaterHook 及多种常见策略（固定、阶梯、指数、
多项式、逆时、余弦），在训练 epoch/iter 前调整优化器各参数组的学习率。

主要类：
    - LrUpdaterHook: 定义 warmup 与 by_epoch 机制的基类。
    - Fixed/Step/Exp/Poly/Inv/CosineLrUpdaterHook: 各种学习率策略子类。

依赖 det3d.solver 中的 fastai 学习率调度，由 Trainer.register_lr_hooks 注册。
"""

from __future__ import division

from math import cos, pi

from det3d.solver import learning_schedules_fastai as lsf

from .hook import Hook


class LrUpdaterHook(Hook):
    """学习率更新基类 Hook。

    支持按 epoch 或按 iter 更新学习率，并可在训练初期进行 warmup。

    Args:
        by_epoch (bool): 是否按 epoch 更新（False 则按 iter 更新）。
        warmup (str, 可选): warmup 类型，支持 constant / linear / exp。
        warmup_iters (int): warmup 持续的迭代数。
        warmup_ratio (float): warmup 起始学习率相对初始值的比例。
        **kwargs: 透传给子类的其余参数。
    """

    def __init__(
        self, by_epoch=True, warmup=None, warmup_iters=0, warmup_ratio=0.1, **kwargs
    ):
        if warmup is not None:
            if warmup not in ["constant", "linear", "exp"]:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warmup)
                )

        if warmup is not None:
            assert warmup_iters > 0, '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters

        # 各参数组的初始学习率
        self.base_lr = []  # 各参数组的初始学习率
        # 不做 warmup 时各参数组的期望学习率
        self.regular_lr = []  # 未进行 warmup 时各参数组应取的学习率

    def _set_lr(self, trainer, lr_groups):
        """按参数组写入新学习率。

        Args:
            trainer (Trainer): 训练器。
            lr_groups (list[float]): 与参数组一一对应的学习率。
        """
        for param_group, lr in zip(trainer.optimizer.param_groups, lr_groups):
            param_group["lr"] = lr

    def get_lr(self, runner, base_lr):
        """根据训练进度计算某个参数组的学习率，子类必须实现。

        Args:
            runner (Trainer): 训练器。
            base_lr (float): 该参数组的初始学习率。
        """
        raise NotImplementedError

    def get_regular_lr(self, trainer):
        """计算不做 warmup 时所有参数组的常规学习率。

        Returns:
            list[float]: 各参数组的学习率。
        """
        return [self.get_lr(trainer, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        """按 warmup 策略计算当前迭代的学习率。

        Args:
            cur_iters (int): 当前迭代数。

        Returns:
            list[float]: warmup 阶段的各参数组学习率。
        """
        if self.warmup == "constant":
            # 恒定 warmup：学习率固定为初始值乘以比例
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == "linear":
            # 线性 warmup：从 warmup_ratio*base 线性增长到 base
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == "exp":
            # 指数 warmup：从 warmup_ratio*base 指数逼近 base
            k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]

        return warmup_lr

    def before_run(self, trainer):
        # 记录每组的初始学习率，作为后续策略计算的基准
        for group in trainer.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lr = [group["initial_lr"] for group in trainer.optimizer.param_groups]

    def before_train_epoch(self, trainer):
        # 按 epoch 更新且无 warmup 场景，在 epoch 开始时一次性设置学习率
        if not self.by_epoch:
            return
        self.regular_lr = self.get_regular_lr(trainer)
        self._set_lr(trainer, self.regular_lr)

    def before_train_iter(self, trainer):
        cur_iter = trainer.iter
        if not self.by_epoch:
            # 按 iter 更新：warmup 结束后取常规学习率，否则取 warmup 学习率
            self.regular_lr = self.get_regular_lr(trainer)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(trainer, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(trainer, warmup_lr)
        elif self.by_epoch:
            # 按 epoch 更新时的 warmup 处理（epoch 内按 iter 推进 warmup）
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(trainer, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(trainer, warmup_lr)


class FixedLrUpdaterHook(LrUpdaterHook):
    """固定学习率策略：学习率始终保持初始值不变。"""

    def __init__(self, **kwargs):
        super(FixedLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        return base_lr


class StepLrUpdaterHook(LrUpdaterHook):
    """阶梯学习率策略：每经过 step 个 epoch/iter 乘以 gamma。

    Args:
        step (int 或 list[int]): 阶梯边界；提供列表时表示各下降阶梯。
        gamma (float): 每个阶梯的衰减因子。
        **kwargs: 透传给基类。
    """

    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else trainer.iter

        if isinstance(self.step, int):
            # 每隔 step 个进度衰减一次
            return base_lr * (self.gamma ** (progress // self.step))

        exp = len(self.step)
        # 找到第一个大于当前进度的阶梯边界，得到应衰减的次数
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break

        return base_lr * self.gamma ** exp


class ExpLrUpdaterHook(LrUpdaterHook):
    """指数衰减学习率策略：lr = base_lr * gamma^progress。

    Args:
        gamma (float): 衰减底数。
        **kwargs: 透传给基类。
    """

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = trainer.epoch if self.by_epoch else trainer.iter
        return base_lr * self.gamma ** progress


class PolyLrUpdaterHook(LrUpdaterHook):
    """多项式学习率策略：随进度按幂次从 base 衰减到 min_lr。

    Args:
        power (float): 指数。
        min_lr (float): 学习率下限。
        **kwargs: 透传给基类。
    """

    def __init__(self, power=1.0, min_lr=0.0, **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(PolyLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        if self.by_epoch:
            progress = trainer.epoch
            max_progress = trainer.max_epochs
        else:
            progress = trainer.iter
            max_progress = trainer.max_iters
        coeff = (1 - progress / max_progress) ** self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


class InvLrUpdaterHook(LrUpdaterHook):
    """逆时衰减学习率策略：lr = base_lr * (1 + gamma*progress)^(-power)。

    Args:
        gamma (float): 衰减系数。
        power (float): 指数。
        **kwargs: 透传给基类。
    """

    def __init__(self, gamma, power=1.0, **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        progress = trainer.epoch if self.by_epoch else trainer.iter
        return base_lr * (1 + self.gamma * progress) ** (-self.power)


class CosineLrUpdaterHook(LrUpdaterHook):
    """余弦退火学习率策略：从 base 余弦下降到 target_lr。

    Args:
        target_lr (float): 目标学习率下限。
        **kwargs: 透传给基类。
    """

    def __init__(self, target_lr=0, **kwargs):
        self.target_lr = target_lr
        super(CosineLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        if self.by_epoch:
            progress = trainer.epoch
            max_progress = trainer.max_epochs
        else:
            progress = trainer.iter
            max_progress = trainer.max_iters

        # 标准余弦退火：从 base_lr 平滑下降到 target_lr
        return self.target_lr + 0.5 * (base_lr - self.target_lr) * (
            1 + cos(pi * (progress / max_progress))
        )
