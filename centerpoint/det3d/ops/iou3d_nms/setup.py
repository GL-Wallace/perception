"""旋转 3D 框 IoU 与 NMS 的 CUDA 扩展编译脚本。

将 CPU/GPU 端源文件（iou3d_cpu.cpp、iou3d_nms_api.cpp、iou3d_nms.cpp、
iou3d_nms_kernel.cu）编译为 iou3d_nms_cuda 扩展模块，供 iou3d_nms_utils.py 调用。
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='iou3d_nms',
    ext_modules=[
        # 编译目标：iou3d_nms_cuda 扩展，提供 IoU 计算与 NMS 的 CPU/GPU 实现。
        CUDAExtension('iou3d_nms_cuda', [
            'src/iou3d_cpu.cpp',
            'src/iou3d_nms_api.cpp',
            'src/iou3d_nms.cpp',
            'src/iou3d_nms_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
