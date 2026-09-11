"""checkpoint 保存 Hook。

在训练 epoch 结束后按固定间隔保存模型 checkpoint，仅主进程执行写文件操作。

主要类：
    - CheckpointHook: 周期性调用 trainer.save_checkpoint 的 Hook。
"""

from ..utils import master_only
from .hook import Hook


class CheckpointHook(Hook):
    """按 epoch 间隔保存 checkpoint。

    Args:
        interval (int): 每多少个 epoch 保存一次。
        save_optimizer (bool): 是否同时保存优化器状态。
        out_dir (str, 可选): 输出目录，默认使用 trainer.work_dir。
        **kwargs: 透传给 trainer.save_checkpoint 的其他参数。
    """

    def __init__(self, interval=1, save_optimizer=True, out_dir=None, **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_train_epoch(self, trainer):
        if not self.every_n_epochs(trainer, self.interval):
            return

        if not self.out_dir:
            self.out_dir = trainer.work_dir

        trainer.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args
        )
