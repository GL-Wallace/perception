"""显存清理 Hook。

在 epoch 前后或每次迭代后调用 torch.cuda.empty_cache()，
释放 CUDA 缓存中未再使用的显存块，减轻显存碎片化。

主要类：
    - EmptyCacheHook: 按配置在指定时机清理显存缓存。
"""

import torch

from .hook import Hook


class EmptyCacheHook(Hook):
    """按时机触发显存缓存清理的 Hook。

    Args:
        before_epoch (bool): 是否在每个 epoch 开始前清理。
        after_epoch (bool): 是否在每个 epoch 结束后清理。
        after_iter (bool): 是否在每次迭代结束后清理。
    """

    def __init__(self, before_epoch=False, after_epoch=True, after_iter=False):
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter

    def after_iter(self, trainer):
        if self._after_iter:
            torch.cuda.empty_cache()

    def before_epoch(self, trainer):
        if self._before_epoch:
            torch.cuda.empty_cache()

    def after_epoch(self, trainer):
        if self._after_epoch:
            torch.cuda.empty_cache()
