"""DataParallel 扩展。

在 PyTorch DataParallel 基础上重写 scatter，使其支持 DataContainer 与
含 kwargs 的输入分发，用于单机多卡并行训练。

主要类：
    - MegDataParallel: 重写 scatter 的 DataParallel 子类。
"""

from torch.nn.parallel import DataParallel

from .scatter_gather import scatter_kwargs


class MegDataParallel(DataParallel):
    """支持 DataContainer / kwargs 的 DataParallel。

    复用 DataParallel 的模型复制与梯度归约，仅把 scatter 替换为
    scatter_kwargs，从而按 DataContainer 的规则切分数据到各 GPU。
    """

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
