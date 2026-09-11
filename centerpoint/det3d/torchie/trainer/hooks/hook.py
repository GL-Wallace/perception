"""Hook 基类。

定义训练/验证生命周期中可重载的钩子方法空实现，以及按 epoch/iter 判断
是否触发的辅助方法；所有具体 Hook 都继承自本类。

主要类：
    - Hook: 提供 before/after 系列钩子与周期判断工具。
"""


class Hook(object):
    """训练 Hook 基类。

    Trainer 在关键时机调用同名方法；默认均为空操作，具体 Hook 按需重载。
    带 train/val 前缀的方法会先调用对应的通用方法（如 before_train_epoch
    内部调用 before_epoch），子类只需重载通用方法即可同时作用于两种阶段。
    """

    def before_run(self, trainer):
        pass

    def after_run(self, trainer):
        pass

    def before_epoch(self, trainer):
        pass

    def after_epoch(self, trainer):
        pass

    def before_iter(self, trainer):
        pass

    def after_iter(self, trainer):
        pass

    def after_data_to_device(self, trainer):
        pass

    def after_forward(self, trainer):
        pass

    def after_parse_loss(self, trainer):
        pass

    def before_train_epoch(self, trainer):
        self.before_epoch(trainer)

    def before_val_epoch(self, trainer):
        self.before_epoch(trainer)

    def after_train_epoch(self, trainer):
        self.after_epoch(trainer)

    def after_val_epoch(self, trainer):
        self.after_epoch(trainer)

    def before_train_iter(self, trainer):
        self.before_iter(trainer)

    def before_val_iter(self, trainer):
        self.before_iter(trainer)

    def after_train_iter(self, trainer):
        self.after_iter(trainer)

    def after_val_iter(self, trainer):
        self.after_iter(trainer)

    def every_n_epochs(self, trainer, n):
        """是否处于每 n 个 epoch 的触发点（以 1 起算）。

        Args:
            trainer (Trainer): 训练器。
            n (int): 触发周期。

        Returns:
            bool: n<=0 时恒为 False。
        """
        return (trainer.epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, trainer, n):
        """是否处于每 n 个全局迭代的触发点。

        Args:
            trainer (Trainer): 训练器。
            n (int): 触发周期。

        Returns:
            bool: n<=0 时恒为 False。
        """
        return (trainer.iter + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, trainer, n):
        """是否处于当前 epoch 内每 n 个迭代的触发点。

        Args:
            trainer (Trainer): 训练器。
            n (int): 触发周期。

        Returns:
            bool: n<=0 时恒为 False。
        """
        return (trainer.inner_iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, trainer):
        """判断是否已走到当前 epoch 的最后一个迭代。

        Args:
            trainer (Trainer): 训练器。

        Returns:
            bool: 是否到达 epoch 末尾。
        """
        return trainer.inner_iter + 1 == len(trainer.data_loader)
