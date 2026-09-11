"""可变形卷积（DCN）与可调制可变形卷积（DCNv2）算子封装。

基于 CUDA 扩展 deform_conv_cuda 实现前反向计算，并提供与 nn.Conv2d 风格一致的
模块封装。deform_conv 学习采样点偏移；modulated_deform_conv 在此基础上额外
学习每个采样点的调制权重（mask）。

主要类/函数：
    - DeformConvFunction: 可变形卷积的 autograd.Function 实现。
    - ModulatedDeformConvFunction: 可调制可变形卷积的 autograd.Function 实现。
    - DeformConv / DeformConvPack: 可变形卷积模块及其带 offset 学习层的封装。
    - ModulatedDeformConv / ModulatedDeformConvPack: 可调制版本的模块封装。

依赖 .so 扩展 deform_conv_cuda（由 setup.py 编译生成）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

# from mmdet.utils import print_log
from . import deform_conv_cuda


class DeformConvFunction(Function):
    """可变形卷积的 autograd.Function 实现。

    前向调用 CUDA 内核 deform_conv_forward_cuda，反向计算输入/偏移/权重的梯度。
    """

    @staticmethod
    def forward(ctx,
                input,
                offset,
                weight,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1,
                im2col_step=64):
        """前向：利用 offset 对采样点做可变形卷积。

        仅支持 4D 输入与 CUDA 张量；im2col_step 需能整除 batch size。
        """
        if input is not None and input.dim() != 4:
            raise ValueError(
                'Expected 4D tensor as input, got {}D tensor instead.'.format(
                    input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset, weight)

        output = input.new_empty(
            DeformConvFunction._output_size(input, weight, ctx.padding,
                                            ctx.dilation, ctx.stride))

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(
                input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1],
                weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0],
                ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                ctx.dilation[0], ctx.groups, ctx.deformable_groups,
                cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """反向：计算 input、offset 与 weight 的梯度。

        返回的梯度数量与 forward 的参数数量一致，不需求梯度的项返回 None。
        """
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(
                    input, offset, grad_output, grad_input,
                    grad_offset, weight, ctx.bufs_[0], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.groups, ctx.deformable_groups,
                    cur_im2col_step)

            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(
                    input, offset, grad_output,
                    grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1,
                    cur_im2col_step)

        return (grad_input, grad_offset, grad_weight, None, None, None, None,
                None)

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        """根据输入尺寸、卷积核与参数计算输出空间尺寸。"""
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be {})'.format(
                    'x'.join(map(str, output_size))))
        return output_size


class ModulatedDeformConvFunction(Function):
    """可调制可变形卷积（DCNv2）的 autograd.Function 实现。

    相比普通可变形卷积，额外引入 mask 对每个采样点做调制加权。
    """

    @staticmethod
    def forward(ctx,
                input,
                offset,
                mask,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1):
        """前向：执行带调制权重的可变形卷积（仅支持 CUDA）。"""
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_cuda.modulated_deform_conv_cuda_forward(
            input, weight, bias, ctx._bufs[0], offset, mask, output,
            ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride,
            ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
            ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """反向：计算 input、offset、mask、weight 与 bias 的梯度。"""
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(
            input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1],
            grad_input, grad_weight, grad_bias, grad_offset, grad_mask,
            grad_output, weight.shape[2], weight.shape[3], ctx.stride,
            ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
            ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        """根据 2D 输入尺寸与卷积参数推断输出形状。"""
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


deform_conv = DeformConvFunction.apply
modulated_deform_conv = ModulatedDeformConvFunction.apply


class DeformConv(nn.Module):
    """可变形卷积模块，接口与 nn.Conv2d 保持一致。

    仅维护权重 weight，offset 由外部传入；不内置 bias。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        """初始化可变形卷积层。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int or tuple[int]): 卷积核尺寸。
            stride (int or tuple[int]): 步长。
            padding (int or tuple[int]): 填充。
            dilation (int or tuple[int]): 空洞率。
            groups (int): 分组卷积组数。
            deformable_groups (int): 可变形分组数。
            bias (bool): 是否使用 bias（当前实现强制为 False）。
        """
        super(DeformConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        # 与 nn.Conv2d 保持接口兼容
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重为 [-stdv, stdv] 的均匀分布。"""
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        """执行可变形卷积。

        当输入特征图小于卷积核时先补零再卷积，输出后裁剪回原尺寸，
        以避免 CUDA 内核断言失败。
        """
        # 修复 deform_conv_cuda.cpp:128 的断言错误：输入特征图小于卷积核时统一补零。
        input_pad = (
            x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant',
                           0).contiguous()
        out = deform_conv(x, offset, self.weight, self.stride, self.padding,
                          self.dilation, self.groups, self.deformable_groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        return out


class DeformConvPack(DeformConv):
    """带 offset 学习层的可变形卷积封装，可像普通卷积层一样使用。

    Args:
        in_channels (int): 同 nn.Conv2d。
        out_channels (int): 同 nn.Conv2d。
        kernel_size (int or tuple[int]): 同 nn.Conv2d。
        stride (int or tuple[int]): 同 nn.Conv2d。
        padding (int or tuple[int]): 同 nn.Conv2d。
        dilation (int or tuple[int]): 同 nn.Conv2d。
        groups (int): 同 nn.Conv2d。
        bias (bool or str): 若指定为 `auto`，则由 norm_cfg 决定；norm_cfg 为
            None 时 bias 置 True，否则置 False。
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)

        # 额外卷积层学习每个采样点的 (x, y) 偏移。
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 2 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        """将 offset 卷积层的权重与 bias 初始化为 0。"""
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups, self.deformable_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """加载状态字典，兼容旧版本 offset 参数的键名。"""
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # 旧版本模型的键名不同，需要做键名迁移。
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                'DeformConvPack {} is upgraded to version 2.'.format(
                    prefix.rstrip('.')),
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)


class ModulatedDeformConv(nn.Module):
    """可调制可变形卷积模块，offset 与 mask 均由外部传入。"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        """初始化可调制可变形卷积层。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int or tuple[int]): 卷积核尺寸。
            stride (int or tuple[int]): 步长。
            padding (int or tuple[int]): 填充。
            dilation (int or tuple[int]): 空洞率。
            groups (int): 分组卷积组数。
            deformable_groups (int): 可变形分组数。
            bias (bool): 是否使用 bias。
        """
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # 与 nn.Conv2d 保持接口兼容
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重（均匀分布）与 bias（置 0）。"""
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """带 offset/mask 学习层的可调制可变形卷积封装，可像普通卷积层一样使用。

    Args:
        in_channels (int): 同 nn.Conv2d。
        out_channels (int): 同 nn.Conv2d。
        kernel_size (int or tuple[int]): 同 nn.Conv2d。
        stride (int or tuple[int]): 同 nn.Conv2d。
        padding (int or tuple[int]): 同 nn.Conv2d。
        dilation (int or tuple[int]): 同 nn.Conv2d。
        groups (int): 同 nn.Conv2d。
        bias (bool or str): 若指定为 `auto`，则由 norm_cfg 决定；norm_cfg 为
            None 时 bias 置 True，否则置 False。
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        # 额外卷积层同时学习 offset（前 2/3）与 mask（后 1/3）。
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        """将 offset/mask 卷积层的权重与 bias 初始化为 0。"""
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        # 输出按通道均分为三份：前两份拼成 offset，第三份经 sigmoid 作为 mask。
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """加载状态字典，兼容旧版本 offset 参数的键名。"""
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # 旧版本模型的键名不同，需要做键名迁移。
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                'ModulatedDeformConvPack {} is upgraded to version 2.'.format(
                    prefix.rstrip('.')),
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
