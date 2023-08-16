"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class ComplexConv2D(nn.Module):
    def __init__(
        self, in_channel, out_channel, ksize=None, nobias=False, init=False, **kwargs
    ):
        super().__init__()
        in_channels_r = in_channel // 2
        in_channels_i = in_channel // 2
        if in_channels_r == 0:
            in_channels_r = in_channel
            in_channels_i = in_channel
        out_channels_r = out_channel // 2
        out_channels_i = out_channel // 2

        self.conv_r = nn.Conv2d(
            in_channels=in_channels_r,
            out_channels=out_channels_r,
            kernel_size=ksize,
            bias=not (nobias),
            **kwargs,
        )
        self.conv_i = nn.Conv2d(
            in_channels=in_channels_i,
            out_channels=out_channels_i,
            kernel_size=ksize,
            bias=not (nobias),
            **kwargs,
        )
        if init and not (nobias):
            nn.init.kaiming_normal_(self.conv_r.weight)
            nn.init.kaiming_normal_(self.conv_i.weight)
            nn.init.zeros_(self.conv_r.bias)
            nn.init.zeros_(self.conv_i.bias)
        elif init and nobias:
            nn.init.kaiming_normal_(self.conv_r.weight)
            nn.init.kaiming_normal_(self.conv_i.weight)

    def forward(self, x):
        x_r = x[:, : x.shape[1] // 2, ...]
        x_i = x[:, x.shape[1] // 2 :, ...]

        mr_kr = self.conv_r(x_r)
        mi_ki = self.conv_i(x_i)
        mi_kr = self.conv_r(x_i)
        mr_ki = self.conv_i(x_r)

        ret = th.cat(((mr_kr - mi_ki), (mr_ki + mi_kr)), dim=1)
        return ret


class ComplexDeconv2D(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        ksize=None,
        stride=1,
        pad=0,
        output_pad=0,
        nobias=False,
        outsize=None,
        init=False,
        **kwargs,
    ):
        super().__init__()
        in_channels_r = in_channel // 2
        in_channels_i = in_channel // 2
        out_channels_r = out_channel // 2
        out_channels_i = out_channel // 2
        if in_channels_r == 0:
            in_channels_r = in_channel
            in_channels_i = in_channel
        elif out_channels_r == 0:
            out_channels_r = out_channel
            out_channels_i = out_channel

        self.deconv_r = nn.ConvTranspose2d(
            in_channels=in_channels_r,
            out_channels=out_channels_r,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            output_padding=output_pad,
            bias=not (nobias),
        )
        self.deconv_i = nn.ConvTranspose2d(
            in_channels=in_channels_i,
            out_channels=out_channels_i,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            output_padding=output_pad,
            bias=not (nobias),
        )

        if init and not (nobias):
            nn.init.kaiming_normal_(self.deconv_r.weight)
            nn.init.kaiming_normal_(self.deconv_i.weight)
            nn.init.zeros_(self.deconv_r.bias)
            nn.init.zeros_(self.deconv_i.bias)
        elif init and nobias:
            nn.init.kaiming_normal_(self.deconv_r.weight)
            nn.init.kaiming_normal_(self.deconv_i.weight)

    def forward(self, x):
        x_r = x[:, : x.shape[1] // 2, ...]
        x_i = x[:, x.shape[1] // 2 :, ...]

        mr_kr = self.deconv_r(x_r)
        mi_ki = self.deconv_i(x_i)
        mi_kr = self.deconv_r(x_i)
        mr_ki = self.deconv_i(x_r)

        ret = th.cat(((mr_kr - mi_ki), (mr_ki + mi_kr)), dim=1)
        return ret


class ComplexBatchNormalization(nn.Module):
    def __init__(
        self,
        d_r,
        eps=1e-4,
        decay=0.9,
        initial_gamma_rr=None,
        initial_gamma_ii=None,
        initial_gamma_ri=None,
        initial_beta=None,
        initial_avg_mean=None,
        initial_avg_vrr=None,
        initial_avg_vii=None,
        initial_avg_vri=None,
    ):
        super().__init__()
        if d_r == 0:
            d_r = 1  # For Real-value U-net
        self._d_r = d_r  # The 1/2 number of dimensions due to [Real, Imag].
        self.eps = eps
        self.decay = decay

        if initial_beta is None:
            self.beta = Parameter(th.zeros(d_r * 2, dtype=th.float32))
        else:
            self.beta = Parameter(initial_beta)

        if initial_gamma_rr is None:
            self.gamma_rr = Parameter(th.ones(d_r, dtype=th.float32) / (2 ** 0.5))
            self.gamma_ii = Parameter(th.ones(d_r, dtype=th.float32) / (2 ** 0.5))
            self.gamma_ri = Parameter(th.ones(d_r, dtype=th.float32))
        else:
            raise
            self.gamma_rr = Parameter(initial_gamma_rr)
            self.gamma_ii = Parameter(initial_gamma_ii)
            self.gamma_ri = Parameter(initial_gamma_ri)

        if initial_avg_mean is None:
            self._avg_mean = th.zeros(2 * d_r, dtype=th.float32)
            self._avg_mean = self._avg_mean[None, :, None, None]
        else:
            self._avg_mean = th.tensor(initial_avg_mean)
        self.register_buffer("avg_mean", self._avg_mean)

        if initial_avg_vrr is None:
            self._avg_vrr = th.ones(d_r, dtype=th.float32) / (2 ** 0.5)
            self._avg_vii = th.ones(d_r, dtype=th.float32) / (2 ** 0.5)
            self._avg_vri = th.zeros(d_r, dtype=th.float32)
            self._avg_vrr = self._avg_vrr[None, :, None, None]
            self._avg_vii = self._avg_vii[None, :, None, None]
            self._avg_vri = self._avg_vri[None, :, None, None]
        else:
            self._avg_vrr = th.tensor(initial_avg_vrr)
            self._avg_vii = th.tensor(initial_avg_vii)
            self._avg_vri = th.tensor(initial_avg_vri)
        self.register_buffer("avg_vrr", self._avg_vrr)
        self.register_buffer("avg_vii", self._avg_vii)
        self.register_buffer("avg_vri", self._avg_vri)

    def forward(self, x, **kwargs):
        # assert x.shape[1] == self._d_r*2

        if self.training:
            # Calc. Statistic Values
            mean = th.mean(x, dim=(0, 2, 3), keepdims=True)
            x_centered = x - mean
            centered_squared = x_centered ** 2.0

            centered_real = x_centered[:, : self._d_r, Ellipsis]
            centered_imag = x_centered[:, self._d_r :, Ellipsis]
            centered_squared_real = centered_squared[:, : self._d_r, Ellipsis]
            centered_squared_imag = centered_squared[:, self._d_r :, Ellipsis]

            Vrr = (
                th.mean(centered_squared_real, dim=(0, 2, 3), keepdims=True) + self.eps
            )
            Vii = (
                th.mean(centered_squared_imag, dim=(0, 2, 3), keepdims=True) + self.eps
            )
            Vri = th.mean(centered_real * centered_imag, dim=(0, 2, 3), keepdims=True)

            # Saving Running Statistics
            self.avg_mean *= self.decay
            self.avg_mean += (1.0 - self.decay) * mean.data
            self.avg_vrr *= self.decay
            self.avg_vrr += (1.0 - self.decay) * Vrr.data
            self.avg_vii *= self.decay
            self.avg_vii += (1.0 - self.decay) * Vii.data
            self.avg_vri *= self.decay
            self.avg_vri += (1.0 - self.decay) * Vri.data
        else:
            # Calc. Statistic Values
            mean = self.avg_mean
            x_centered = x - mean
            centered_squared = x_centered ** 2.0

            centered_real = x_centered[:, : self._d_r, Ellipsis]
            centered_imag = x_centered[:, self._d_r :, Ellipsis]

            Vrr = self.avg_vrr
            Vii = self.avg_vii
            Vri = self.avg_vri

        # Inverse of Variance Matrix
        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2.0)
        s = th.sqrt(delta + np.finfo("float32").eps)
        t = th.sqrt(tau + 2.0 * s + np.finfo("float32").eps)
        inverse_st = 1.0 / ((s * t) + np.finfo("float32").eps)
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st

        # Complex Standardization
        cat_W_4_real = th.cat((Wrr, Wii), dim=1)
        cat_W_4_imag = th.cat((Wri, Wri), dim=1)
        rolled_x = th.cat((centered_imag, centered_real), dim=1)
        x_stdized = cat_W_4_real * x_centered + cat_W_4_imag * rolled_x

        # Re-scaling using Gamma, Beta
        x_stdized_r = x_stdized[:, : self._d_r, Ellipsis]
        x_stdized_i = x_stdized[:, self._d_r :, Ellipsis]
        rolled_x_stdized = th.cat((x_stdized_i, x_stdized_r), dim=1)
        broadcast_gamma_rr = th.broadcast_to(
            th.reshape(self.gamma_rr, (1, self._d_r, 1, 1)), x_stdized_r.shape
        )
        broadcast_gamma_ii = th.broadcast_to(
            th.reshape(self.gamma_ii, (1, self._d_r, 1, 1)), x_stdized_r.shape
        )
        broadcast_gamma_ri = th.broadcast_to(
            th.reshape(self.gamma_ri, (1, self._d_r, 1, 1)), x_stdized_r.shape
        )
        cat_gamma_4_real = th.cat((broadcast_gamma_rr, broadcast_gamma_ii), dim=1)
        cat_gamma_4_imag = th.cat((broadcast_gamma_ri, broadcast_gamma_ri), dim=1)

        out = cat_gamma_4_real * x_stdized + cat_gamma_4_imag * rolled_x_stdized
        out += th.reshape(self.beta, (1, self._d_r * 2, 1, 1))

        return out


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, complex_conv=False, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2 and complex_conv:
        return ComplexConv2D(
            in_channel=args[0], out_channel=args[1], ksize=args[2], **kwargs
        )
    elif dims == 2 and not (complex_conv):
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
