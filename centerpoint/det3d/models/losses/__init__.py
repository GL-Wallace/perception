"""损失函数子包。

提供 CenterPoint 训练所需的损失实现，具体见 centernet_loss 模块：
    - FastFocalLoss: 中心热图的 penalty-reduced focal loss（论文 Sec 3.3）。
    - RegLoss: 仅在 GT 中心位置计算的 L1 回归损失。
"""