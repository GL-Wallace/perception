"""日志 Hook 基类。

定义日志输出节流逻辑：按固定 interval 抽稀输出、在 epoch 边界兜底输出，
并通过 LogBuffer 的平均值与 ready 标志决定是否触发真正的 log 调用。

主要类：
    - LoggerHook: 抽象基类，统一训练/验证日志的输出时机。
"""

from abc import ABCMeta, abstractmethod

from ..hook import Hook


class LoggerHook(Hook):
    """日志 Hook 的抽象基类。

    Args:
        interval (int): 每多少次迭代输出一次日志。
        ignore_last (bool): 是否忽略每个 epoch 末尾不足以构成一个完整
            interval 的残余迭代。
        reset_flag (bool): 输出后是否清空 LogBuffer 的输出缓冲。
    """

    __metaclass__ = ABCMeta

    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag

    @abstractmethod
    def log(self, trainer):
        """执行真正的日志输出，子类必须实现。"""
        pass

    def before_run(self, trainer):
        # 多个日志 Hook 并存时，仅让列表中最后一个 LoggerHook 负责清空输出缓冲，
        # 避免前面的 Hook 输出后清空导致后面的 Hook 无法读取
        for hook in trainer.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, trainer):
        # 每个 epoch 开始前清空日志缓冲
        trainer.log_buffer.clear()

    def after_train_iter(self, trainer):
        # 每到固定周期对区间内指标取平均；否则在 epoch 末尾且不忽略残余时也取平均
        if self.every_n_inner_iters(trainer, self.interval):
            trainer.log_buffer.average(self.interval)
        elif self.end_of_epoch(trainer) and not self.ignore_last:
            # 不够精确但更稳定的兜底平均
            trainer.log_buffer.average(self.interval)

        if trainer.log_buffer.ready:
            self.log(trainer)
            if self.reset_flag:
                trainer.log_buffer.clear_output()

    def after_train_epoch(self, trainer):
        # epoch 结束时若仍有未输出的缓冲，则补一次输出
        if trainer.log_buffer.ready:
            self.log(trainer)
            if self.reset_flag:
                trainer.log_buffer.clear_output()

    def after_val_epoch(self, trainer):
        # 验证结束后对全部指标取平均并输出
        trainer.log_buffer.average()
        self.log(trainer)
        if self.reset_flag:
            trainer.log_buffer.clear_output()
