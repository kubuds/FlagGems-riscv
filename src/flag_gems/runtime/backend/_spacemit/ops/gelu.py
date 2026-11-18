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

logger = logging.getLogger(__name__)

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh
geluTanh = tl_extra_shim.gelu_tanh
geluNone = tl_extra_shim.gelu_none

# Direct-kernel fast path for floating dtypes (campaign 1011night). Wrapper costs
# an extra ~1.8-6.9 ms per call (none: 1M 7.51 -> 5.70 ms, 2048x2048 29.06 ->
# 22.16 ms; tanh: 1M 5.61 -> 3.69 ms, 2048x2048 21.48 -> 14.68 ms) because its
# generated kernel lowers worse. Same formulas as the wrapper (maxdiff ~2-5e-07
# vs torch, equal to the wrapper's own rounding).
_BLOCK = 8192


@libentry()
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    x32 = x.to(tl.float32)
    y = 0.5 * x32 * (1.0 + erf(x32 * 0.7071067811))
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty), mask=m)


@libentry()
@triton.jit
def gelu_tanh_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    x32 = x.to(tl.float32)
    y = 0.5 * x32 * (1.0 + tanh(0.79788456 * x32 * (1.0 + 0.044715 * x32 * x32)))
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty), mask=m)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    # scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    # output = 0.5 * x * (1 + erf(x * scale))
    output = geluNone(x.to(tl.float32))
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    output = (
        # 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * pow(x.to(tl.float32), 2))))
        geluTanh(x.to(tl.float32))
    )
    return output


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    dydx = (
        scale2 * x_fp32 * exp(-pow(scale1 * x_fp32, 2))
        + 0.5 * erf(scale1 * x_fp32)
        + 0.5
    )
    dx = dydx * dy
    return dx


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    # 0.79788456 = math.sqrt(2 / math.pi)
    tanh_out = tanh(0.79788456 * x_fp32 * (1 + 0.044715 * pow(x_fp32, 2)))
    dydx = 0.5 * x_fp32 * (
        (1 - pow(tanh_out, 2)) * (0.79788456 + 0.1070322243 * pow(x_fp32, 2))
    ) + 0.5 * (1 + tanh_out)
    dx = dydx * dy
    return dx


def _direct_ok(t):
    return (
        t.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and t.is_contiguous()
    )


def _launch_direct(x, out, approximate):
    n = x.numel()
    if n == 0:
        return out
    grid = (triton.cdiv(n, _BLOCK),)
    kernel = gelu_tanh_kernel if approximate == "tanh" else gelu_kernel
    with torch_device_fn.device(x.device):
        kernel[grid](x, out, n, BLOCK=_BLOCK)
    return out


def gelu(A, *, approximate="none"):
    logger.debug("GEMS_SPACEMIT GELU")
    if _direct_ok(A):
        out = torch.empty_like(A)
        return _launch_direct(A, out, approximate)
    if approximate == "tanh":
        out = gelu_tanh(A)
    else:
        out = gelu_none(A)
    return out


def gelu_backward(grad_output, self, *, approximate="none"):
    logger.debug("GEMS_SPACEMIT GELU_BACKWARD")
    if approximate == "tanh":
        in_grad = gelu_backward_tanh(self, grad_output)
    else:
        in_grad = gelu_backward_none(self, grad_output)
    return in_grad


def gelu_(A, *, approximate="none"):
    logger.debug("GEMS_SPACEMIT GELU_")
    if _direct_ok(A):
        return _launch_direct(A, A, approximate)
    if approximate == "tanh":
        out = gelu_tanh(A, out0=A)
    else:
        out = gelu_none(A, out0=A)
    return out
