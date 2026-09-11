"""可变形卷积 CUDA 扩展的编译脚本。

将 src/deform_conv_cuda.cpp 与 src/deform_conv_cuda_kernel.cu 编译为
deform_conv_cuda 扩展模块（PyTorch CUDAExtension），供 deform_conv.py 调用。
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='masked_conv',
    ext_modules=[
        # 编译目标：deform_conv_cuda 扩展；禁用半精度相关算子以满足内核兼容性。
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ],
        define_macros=[('WITH_CUDA', None)],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
        ]})],
        cmdclass={'build_ext': BuildExtension})

