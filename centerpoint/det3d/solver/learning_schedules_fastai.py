"""fastai 风格的学习率调度器。

提供基于阶段（phase）的通用学习率/动量调度基类 LRSchedulerStep，以及 one-cycle、
指数衰减、手动分段等具体策略。这些调度器直接操作 FastAIMixedOptim / OptimWrapper
这类优化器包装对象的 lr / mom 属性。

主要类：
    - LRSchedulerStep: 阶段化学习率/动量调度基类。
    - OneCycle: one-cycle 学习率策略（先升后降）。
    - ExponentialDecay: 指数衰减策略（支持阶梯式）。
    - ManualStepping: 手动指定分段学习率。
"""

import math
from functools import partial

import numpy as np


class LRSchedulerStep(object):
    """阶段化学习率/动量调度基类。

    将整个训练过程按 [start, end) 划分为多个阶段，每个阶段用 lambda 函数
    根据阶段内进度计算学习率或动量。start 以总步数的比例表示。

    Args:
        fai_optimizer: 优化器包装对象（含 lr / mom 属性）。
        total_step: 总训练步数。
        lr_phases: 学习率阶段列表，元素为 (start, lambda_func)。
        mom_phases: 动量阶段列表，元素为 (start, lambda_func)。
    """

    def __init__(self, fai_optimizer, total_step, lr_phases, mom_phases):
        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < int(start * total_step)
            if isinstance(lambda_func, str):
                # 支持以字符串形式传入 lambda 表达式，此处解析为可调用对象。
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append(
                    (
                        int(start * total_step),
                        int(lr_phases[i + 1][0] * total_step),
                        lambda_func,
                    )
                )
            else:
                self.lr_phases.append(
                    (int(start * total_step), total_step, lambda_func)
                )
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append(
                    (
                        int(start * total_step),
                        int(mom_phases[i + 1][0] * total_step),
                        lambda_func,
                    )
                )
            else:
                self.mom_phases.append(
                    (int(start * total_step), total_step, lambda_func)
                )
        # assert self.mom_phases[0][0] == 0
        if len(mom_phases) > 0:
            assert self.mom_phases[0][0] == 0

    def step(self, step):
        """根据当前步数 step 更新优化器的学习率与动量。

        对每个覆盖当前步数的阶段，取最后一个生效阶段的计算结果作为新值。
        """
        lrs, moms = [], []

        for start, end, func in self.lr_phases:
            if step >= start:
                # self.optimizer.lr = func((step - start) / (end - start))
                lrs.append(func((step - start) / (end - start)))
        if len(lrs) > 0:
            self.optimizer.lr = lrs[-1]
        for start, end, func in self.mom_phases:
            if step >= start:
                moms.append(func((step - start) / (end - start)))
                self.optimizer.mom = func((step - start) / (end - start))
        if len(moms) > 0:
            self.optimizer.mom = moms[-1]


def annealing_cos(start, end, pct):
    """余弦退火：pct 从 0 到 1 时，从 start 平滑过渡到 end。"""
    # print(pct, start, end)
    # pct 从 0 到 1 时，余弦曲线从 start 平滑过渡到 end。
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    """one-cycle 学习率调度：前 pct_start 比例升温至 lr_max，之后余弦退火至近零。"""

    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor, pct_start):
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        a1 = int(total_step * self.pct_start)
        a2 = total_step - a1
        # 初始学习率为 lr_max / div_factor。
        low_lr = self.lr_max / self.div_factor
        lr_phases = (
            (0, partial(annealing_cos, low_lr, self.lr_max)),
            (self.pct_start, partial(annealing_cos, self.lr_max, low_lr / 1e4)),
        )
        # 动量在第一阶段从 moms[0] 到 moms[1]，第二阶段反向。
        mom_phases = (
            (0, partial(annealing_cos, *self.moms)),
            (self.pct_start, partial(annealing_cos, *self.moms[::-1])),
        )
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)


class ExponentialDecay(LRSchedulerStep):
    """指数衰减学习率调度。

    Args:
        fai_optimizer: 优化器包装对象。
        total_step: 总训练步数。
        initial_learning_rate: 初始学习率。
        decay_length: 每次衰减周期占总步数的比例，必须位于 (0, 1)。
        decay_factor: 每个周期学习率乘以的衰减因子。
        staircase: 是否阶梯式衰减（True 时每周期内保持不变）。
    """

    def __init__(
        self,
        fai_optimizer,
        total_step,
        initial_learning_rate,
        decay_length,
        decay_factor,
        staircase=True,
    ):
        """
        Args:
            decay_length: 必须位于 (0, 1) 区间。
        """
        assert decay_length > 0
        assert decay_length < 1
        self._decay_steps_unified = decay_length
        self._decay_factor = decay_factor
        self._staircase = staircase
        step = 0
        stage = 1
        lr_phases = []
        if staircase:
            # 每个阶梯阶段内学习率恒定，阶段间乘以衰减因子。
            while step <= total_step:
                func = lambda p, _d=initial_learning_rate * stage: _d
                lr_phases.append((step / total_step, func))
                stage *= decay_factor
                step += int(decay_length * total_step)
        else:
            # 连续衰减：按进度指数计算。
            func = lambda p: pow(decay_factor, (p / decay_length))
            lr_phases.append((0, func))
        super().__init__(fai_optimizer, total_step, lr_phases, [])


class ManualStepping(LRSchedulerStep):
    """手动分段学习率：在指定的 boundary 比例处分段使用给定的学习率。"""

    def __init__(self, fai_optimizer, total_step, boundaries, rates):
        assert all([b > 0 and b < 1 for b in boundaries])
        assert len(boundaries) + 1 == len(rates)
        boundaries.insert(0, 0.0)
        lr_phases = []
        for start, rate in zip(boundaries, rates):
            func = lambda p, _d=rate: _d
            lr_phases.append((start, func))
        super().__init__(fai_optimizer, total_step, lr_phases, [])


class FakeOptim:
    """带 lr / mom 属性的假优化器，供调度器调试与可视化使用。"""

    def __init__(self):
        self.lr = 0
        self.mom = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    opt = FakeOptim()  # 3e-3, wd=0.4, div_factor=10
    # schd = OneCycle(opt, 100, 3e-3, (0.95, 0.85), 10.0, 0.1)
    schd = ExponentialDecay(opt, 100, 3e-4, 0.1, 0.8, staircase=True)
    schd = ManualStepping(opt, 100, [0.8, 0.9], [0.001, 0.0001, 0.00005])

    lrs = []
    moms = []
    for i in range(100):
        schd.step(i)
        lrs.append(opt.lr)
        moms.append(opt.mom)
    plt.plot(lrs)
    # plt.plot(moms)
    # plt.show()
    # plt.plot(moms)
    plt.show()
