"""backbone 模块定义与条件导入。

尝试检测 spconv 是否可用：可用时导入稀疏卷积骨干 SpMiddleResNetFHD，
否则打印提示并跳过稀疏卷积相关功能。
"""

import importlib
# 探测 spconv 是否安装，据此决定是否启用稀疏卷积骨干。
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None

if found:
    from .scn import SpMiddleResNetFHD
else:
    print("No spconv, sparse convolution disabled!")

