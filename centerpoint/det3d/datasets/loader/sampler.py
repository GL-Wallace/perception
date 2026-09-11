"""数据加载的采样器（Sampler）集合。

提供分布式与按 group(长宽比/类别分组) 采样的 Sampler 实现，用于 DataLoader：
    - DistributedSampler / DistributedSamplerV2: 将数据集按 rank 切分，保证多卡
      各自消费互不重叠的子集。
    - GroupSampler / DistributedGroupSampler: 依据 dataset.flag 分组，使同一 batch
      内样本尽量来自同一组，减少 padding 浪费。

这些采样器通过 epoch 作为随机种子，保证同一 epoch 各进程的洗牌顺序一致。
"""
from __future__ import division
import math

import numpy as np
import torch
import math
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

from det3d.torchie.trainer import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler

# from torch.utils.data import Sampler


class DistributedSamplerV2(Sampler):
    """把数据加载限制到当前进程专属的子集（配合 DistributedDataParallel 使用）。

    每个进程拿到互不重叠的样本子集；dataset 规模视为固定。

    Args:
        dataset: 被采样的数据集。
        num_replicas (int, optional): 参与训练的进程数（默认取 world_size）。
        rank (int, optional): 当前进程 rank（默认取 get_rank()）。
        shuffle (bool): 是否打乱。
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        """按 epoch 确定性洗牌，补齐到可被 num_replicas 整除，再按 rank 步进取子集。"""
        if self.shuffle:
            # 以 epoch 为种子确定性洗牌，保证各进程同序
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # 头尾补样，保证总长度可被 num_replicas 整除
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # 按 rank 间隔取值，得到本进程子集
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """设置当前 epoch，用于控制洗牌随机种子。"""
        self.epoch = epoch


class DistributedSampler(_DistributedSampler):
    """继承 PyTorch DistributedSampler，但保留 shuffle 参数并重写 __iter__。"""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # 以 epoch 为种子确定性洗牌
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # 补齐到可整除
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # 按 rank 取子集
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):
    """按 dataset.flag 分组采样，使同一 batch 的样本尽量属于同一组。"""

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, "flag")
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += (
                int(np.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu
            )

    def __iter__(self):
        """每组内补齐到 samples_per_gpu 的整数倍，再按 group 打乱后拼接。"""
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(
                np.ceil(size / self.samples_per_gpu)
            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu : (i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """按 group 分组的分布式采样器。

    在 GroupSampler 基础上再按 rank 切分，使各进程消费互不重叠的、组内对齐的样本子集。

    Args:
        dataset: 被采样的数据集。
        samples_per_gpu (int): 每 GPU 的 batch 大小（组对齐粒度）。
        num_replicas (int, optional): 进程数。
        rank (int, optional): 当前进程 rank。
    """

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None, rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += (
                int(
                    math.ceil(
                        self.group_sizes[i]
                        * 1.0
                        / self.samples_per_gpu
                        / self.num_replicas
                    )
                )
                * self.samples_per_gpu
            )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # 以 epoch 为种子确定性洗牌
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size), generator=g))].tolist()
                extra = int(
                    math.ceil(size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        assert len(indices) == self.total_size

        indices = [
            indices[j]
            for i in list(
                torch.randperm(len(indices) // self.samples_per_gpu, generator=g)
            )
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]

        # 按 rank 偏移取本进程子集
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """设置当前 epoch，用于控制洗牌随机种子。"""
        self.epoch = epoch
