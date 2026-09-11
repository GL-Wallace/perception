"""RPN（Region Proposal Network）风格的 BEV 特征转换 neck。

将 backbone 输出的稠密 BEV 特征经若干下采样卷积块后，对部分层级做反卷积上采样，
再把各尺度特征在通道维拼接，得到供检测头使用的多尺度 BEV 特征。这是
CenterPoint 论文 Sec 3 骨架中的 neck 部分（承担 BEV backbone 的角色）。

主要类：
    - RPN: SECOND / CenterPoint 常用的 BEV 特征金字塔 neck。
"""

import time
import numpy as np
import math

import torch

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer


@NECKS.register_module
class RPN(nn.Module):
    """BEV 特征转换 neck，输出多尺度上采样拼接后的特征。"""

    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        """构造 RPN：按层数与步长构建下采样卷积块，并为需上采样的层构建反卷积。

        Args:
            layer_nums (list): 每级卷积块内的卷积层数。
            ds_layer_strides (list): 各层级的下采样步长。
            ds_num_filters (list): 各层级输出通道数。
            us_layer_strides (list): 上采样步长（与靠后的若干层级对应）。
            us_num_filters (list): 各上采样分支的输出通道数。
            num_input_features (int): 输入特征通道数。
            norm_cfg (dict, optional): 归一化层配置。
            name (str): neck 名称。
            logger: 日志器。
            **kwargs: 额外参数。
        """
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        # 校验各上采样分支经各自累积下采样后的最终步长一致，保证拼接时空间对齐。
        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        # 每级卷积块的输入通道数：首级取输入通道，其后取上一级输出通道。
        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        """总体下采样倍率：各层步长之积除以上采样步长的末项。"""
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        """构建一个「下采样卷积 + 若干 3x3 卷积块」的 Sequential。

        Args:
            inplanes (int): 输入通道数。
            planes (int): 输出通道数。
            num_blocks (int): 后续 3x3 卷积块数量。
            stride (int): 首个卷积的步长（>1 时实现下采样）。

        Returns:
            Tuple[Sequential, int]: (卷积块序列, 输出通道数)。
        """
        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block.add(nn.ReLU())

        return block, planes

    # 对各卷积层执行默认的 msra（xavier）权重初始化。
    def init_weights(self):
        """对卷积层使用 xavier 均匀分布初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        """前向：逐级下采样并收集指定层级的反卷积特征，在通道维拼接。

        Args:
            x (Tensor): 输入 BEV 特征（NCHW）。

        Returns:
            Tensor: 多尺度上采样特征在通道维拼接后的 BEV 特征。
        """
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            # 从 _upsample_start_idx 起，对该层输出做反卷积上采样并收集。
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        return x

