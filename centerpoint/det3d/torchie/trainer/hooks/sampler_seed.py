"""分布式采样器种子 Hook。

在分布式训练中，每个 epoch 开始前把当前 epoch 号写入 DistributedSampler，
使各 rank 在不同 epoch 使用不同的数据划分，保证训练数据充分打散。

主要类：
    - DistSamplerSeedHook: 在 before_epoch 同步 sampler 的 epoch 种子。
"""

from .hook import Hook


class DistSamplerSeedHook(Hook):
    """同步分布式采样器的 epoch 种子。

    在 epoch 开始前调用数据加载器 sampler 的 set_epoch，令各 rank 按 epoch
    重新划分数据，避免每个 epoch 抽取相同的子集。
    """

    def before_epoch(self, trainer):
        trainer.data_loader.sampler.set_epoch(trainer.epoch)
