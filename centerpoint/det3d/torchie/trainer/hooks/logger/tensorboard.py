"""TensorBoard 日志 Hook。

把 LogBuffer 中的标量指标写入 TensorBoard（时间序列以 iter 为横轴），
并跳过 time / data_time 等计时字段，仅记录真正关心的训练指标。

主要类：
    - TensorboardLoggerHook: 封装 SummaryWriter 的日志 Hook。
"""

import os.path as osp

import torch

from ...utils import master_only
from .base import LoggerHook


class TensorboardLoggerHook(LoggerHook):
    """把训练指标写入 TensorBoard 的日志 Hook。

    Args:
        log_dir (str, 可选): 事件文件目录，默认使用 trainer.work_dir/tf_logs。
        interval (int): 输出周期（继承自基类）。
        ignore_last (bool): 是否忽略 epoch 末尾残余迭代。
        reset_flag (bool): 输出后是否清空日志缓冲。
    """

    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.log_dir = log_dir

    @master_only
    def before_run(self, trainer):
        # 根据 PyTorch 版本选择 SummaryWriter 来源（1.1+ 使用 torch.utils.tensorboard）
        if torch.__version__ >= "1.1":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )
        else:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    "Please install tensorboardX to use " "TensorboardLoggerHook."
                )

        if self.log_dir is None:
            self.log_dir = osp.join(trainer.work_dir, "tf_logs")
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, trainer):
        # 遍历日志缓冲的每个指标，按 "指标名/阶段" 组织 tag 写入 writer
        for var in trainer.log_buffer.output:
            if var in ["time", "data_time"]:
                continue
            tag = "{}/{}".format(var, trainer.mode)
            record = trainer.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, trainer.iter)
            else:
                self.writer.add_scalar(
                    tag, trainer.log_buffer.output[var], trainer.iter
                )

    @master_only
    def after_run(self, trainer):
        # 训练结束关闭 writer 释放资源
        self.writer.close()
