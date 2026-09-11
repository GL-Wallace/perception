"""优化器 Hook。

在每次训练迭代后执行梯度清零、反向传播、可选梯度裁剪与参数更新，
是 Trainer 与优化器之间的桥梁。

主要类：
    - OptimizerHook: 在 after_train_iter 完成参数更新流程。
"""

from torch.nn.utils import clip_grad

from .hook import Hook


class OptimizerHook(Hook):
    """执行优化器更新步骤的 Hook。

    Args:
        grad_clip (dict, 可选): 梯度裁剪参数，将被透传给
            torch.nn.utils.clip_grad.clip_grad_norm_。
    """

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        """对需要梯度的参数做梯度范数裁剪。

        Args:
            params (iterable): 需要裁剪梯度的参数（可迭代对象）。
        """
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip
        )

    def after_train_iter(self, trainer):
        # 典型参数更新三步式：清零梯度 -> 反向传播 -> （裁剪）-> 更新参数
        trainer.optimizer.zero_grad()
        # print(trainer.outputs["loss"])
        trainer.outputs["loss"].backward()
        if self.grad_clip is not None:
            self.clip_grads(trainer.model.parameters())
        trainer.optimizer.step()
