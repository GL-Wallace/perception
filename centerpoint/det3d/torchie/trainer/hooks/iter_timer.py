"""迭代计时 Hook。

在训练迭代的各个关键阶段记录时间戳，把数据加载、设备搬运、前向、损失解析
等耗时写入 LogBuffer，供日志 Hook 输出，帮助分析训练瓶颈。

主要类：
    - IterTimerHook: 在 before/after 各钩子触发点采样耗时。
"""

import time

from .hook import Hook


class IterTimerHook(Hook):
    """统计训练迭代各阶段的耗时。

    以 self.t 记录上一阶段结束时间，在对应钩子处计算相邻阶段的时间差，
    写入 runner.log_buffer（键名如 data_time / transfer_time / forward_time）。
    """

    def before_epoch(self, runner):
        # epoch 开始时记录起始时间戳
        self.t = time.time()

    def before_iter(self, runner):
        # before_iter 距上一阶段的时间即数据加载耗时 data_time
        runner.log_buffer.update({"data_time": time.time() - self.t})

    def after_iter(self, runner):
        # after_iter 距上一阶段的时间即单次迭代总耗时 time，并重置计时起点
        runner.log_buffer.update({"time": time.time() - self.t})
        self.t = time.time()

    def after_data_to_device(self, runner):
        # 数据搬运完成的额外耗时（相对 before_iter 起点）
        runner.log_buffer.update({"transfer_time": time.time() - self.t})

    def after_forward(self, runner):
        # 前向完成的耗时
        runner.log_buffer.update({"forward_time": time.time() - self.t})

    def after_parse_loss(self, runner):
        # 损失解析完成的耗时
        runner.log_buffer.update({"loss_parse_time": time.time() - self.t})
