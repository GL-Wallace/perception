"""torchie 训练框架基础设施包。

torchie 是 CenterPoint 复用的训练框架底层库，提供 2D 卷积 backbone、文件 IO、
配置解析、计时与进度条等与训练主循环、配置加载以及 checkpoint 保存/恢复密切相关的
通用能力。本包在导入时通过 ``import *`` 汇总并对外暴露各子模块的公共接口。

主要子模块：
    - cnn: 备用的 2D backbone（ResNet/VGG/AlexNet）与权重初始化工具。
    - fileio: json/yaml/pickle 等格式配置与数据的统一读写。
    - utils: 配置文件解析、计时器、进度条、路径与序列等通用工具。
    - trainer: 训练器与 checkpoint 的保存/恢复（load_checkpoint 等）。
    - parallel: 数据并行与分布式训练相关设施。
"""

# from .apis import *
from .cnn import *
from .fileio import *
from .parallel import *
from .trainer import *
from .utils import *
