# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

from flag_gems.runtime.backend._spacemit.ops._dispatch import nested_dispatch

logger = logging.getLogger(__name__)

div_rn = tl_extra_shim.div_rn
_silu = tl_extra_shim.silu
exp = tl_extra_shim.exp

# Direct-kernel fast path for floating dtypes (campaign 1011night): wrapper costs
# an extra ~1.9-6.6 ms per call (1M: 5.16 -> 3.31 ms; 2048x2048: 19.62 -> 13.04 ms).
# Same numerics as the wrapper (maxdiff 4.77e-07 vs torch, same as wrapper's).
# campaign 1012am: mask-free variant (+7.9% f32, +2.5% f16 interleaved A/B,
# bit-identical) + nested dispatch (2.2x).
_BLOCK = 8192


@libentry()
@triton.jit
def silu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    x32 = x.to(tl.float32)
    y = x32 / (1.0 + exp(-x32))
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty), mask=m)


@libentry()
@triton.jit
def silu_kernel_nomask(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    x32 = x.to(tl.float32)
    y = x32 / (1.0 + exp(-x32))
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    # y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    y = _silu(x_fp32)
    return y


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_backward_pw(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


def silu_backward(grad_output, self):
    # aten::silu_backward(grad_output, self); nested dispatch lock-in for the
    # generic wrapper kernels (measured 6.6 -> 1.9 ms at 1M, +72%).
    logger.debug("GEMS_SPACEMIT SILU_BACKWARD")
    if _direct_ok(grad_output) and _direct_ok(self):
        with nested_dispatch(grad_output.numel()):
            return silu_backward_pw(self, grad_output)
    return silu_backward_pw(self, grad_output)


def _direct_ok(t):
    return (
        t.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and t.is_contiguous()
    )


def _launch_direct(x, out):
    n = x.numel()
    if n == 0:
        return out
    with torch_device_fn.device(x.device):
        with nested_dispatch(n):
            if n % _BLOCK == 0:
                silu_kernel_nomask[(n // _BLOCK,)](x, out, n, BLOCK=_BLOCK)
            else:
                silu_kernel[(triton.cdiv(n, _BLOCK),)](x, out, n, BLOCK=_BLOCK)
    return out


class Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("GEMS_SPACEMIT SILU_FORWARD")
        out = silu_forward(A)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_SPACEMIT SILU_BACKWARD")
        (inp,) = ctx.saved_tensors
        in_grad = silu_backward_pw(inp, out_grad)
        return in_grad


def silu(A):
    if _direct_ok(A) and not A.requires_grad:
        out = torch.empty_like(A)
        return _launch_direct(A, out)
    return Silu.apply(A)


class InplaceSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("GEMS_SPACEMIT SILU_FORWARD")
        ctx.save_for_backward(A.clone())
        ctx.mark_dirty(A)
        out = silu_forward(A, out0=A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_SPACEMIT SILU_BACKWARD")
        (inp,) = ctx.saved_tensors
        in_grad = silu_backward_pw(inp, out_grad)
        return in_grad


def silu_(A):
    if _direct_ok(A) and not A.requires_grad:
        return _launch_direct(A, A)
    InplaceSilu.apply(A)
    return A
