"""模型各类组件的注册表定义。

为 reader / backbone / neck / head / loss / detector / second_stage / roi_head
分别创建独立的注册表，配合 @X.register_module 装饰器实现「按配置实例化组件」。

由 builder.py 使用这些注册表完成模块构建。
"""

from det3d.utils import Registry

# 各类网络组件的注册表，名称与各模块中的 @X.register_module 装饰器保持一致。
READERS = Registry("reader")
BACKBONES = Registry("backbone")
NECKS = Registry("neck")
HEADS = Registry("head")
LOSSES = Registry("loss")
DETECTORS = Registry("detector")
SECOND_STAGE = Registry("second_stage")
ROI_HEAD = Registry("roi_head")