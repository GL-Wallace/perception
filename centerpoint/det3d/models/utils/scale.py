"""可学习的缩放模块（Scale）。

提供一个对输入逐元素乘以可学习标量因子的轻量模块，用于在网络中引入可训练的
尺度缩放。

主要类：
    - Scale: 可学习标量缩放层。
"""

import torch
import torch.nn as nn


class Scale(nn.Module):
    """可学习标量缩放层：将输入逐元素乘以一个可学习的标量参数。"""

    def __init__(self, scale=1.0):
        """构造缩放层。

        Args:
            scale (float): 缩放因子的初始值。
        """
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        """对输入做逐元素缩放。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: x * scale。
        """
        return x * self.scale
