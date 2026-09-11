"""训练日志缓冲：累计并取平均各项指标。

LogBuffer 收集每个训练迭代产生的标量/数组指标及其样本计数，供 LoggerHook
按固定周期调用 average 计算加权平均后输出，是 Trainer 与日志 Hook 之间的
数据中转结构。

主要类：
    - LogBuffer: 保存各指标的历史值与样本数，支持按周期平均。
"""

from collections import OrderedDict

import numpy as np


class LogBuffer(object):
    """累积训练指标并按需取平均的缓冲区。

    维护每个指标的历史数值 val_history 与对应样本计数 n_history，
    调用 average 后把均值写入 output 并置 ready 标志。
    """

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        """清空全部历史与输出。"""
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        """仅清空输出缓冲，并把 ready 标志复位。"""
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        """追加一批新的指标值及其样本计数。

        Args:
            vars (dict): 键为指标名、值为数值或数组的字典。
            count (int): 本批对应的样本数，用于后续加权平均。
        """
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n=0):
        """对最近 n 个（n=0 表示全部）取值计算平均并写入输出。

        Args:
            n (int): 参与平均的历史条目数，0 表示全部。
        """
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            if values.shape == nums.shape:
                # 形状一致时按样本计数做加权平均
                avg = np.sum(values * nums) / np.sum(nums)
            else:
                # 形状不一致（如多任务返回值）时对第 0 维作简单平均
                avg = np.mean(values, axis=0).tolist()
            self.output[key] = avg
        self.ready = True
