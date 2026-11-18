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
_tanh = tl_extra_shim.tanh

# Direct-kernel fast path for floating dtypes (campaign 1011am). The generic
# pointwise wrapper costs ~1.9 ms extra per call on K3 (1M: 5.39 -> 3.48 ms;
# 2048x2048: 20.95 -> 13.93 ms); same lowering-quality issue as sigmoid/silu.
# Numerically equivalent (maxdiff 5.96e-08 vs torch on fp32 randn).
# campaign 1011pm: mask-free variants used when n divides BLOCK exactly.
# Mask handling is expensive on this backend's reductions/arithmetic kernels:
# tanh_backward at 2048^2 measured 3.85 ms masked vs 2.47 ms mask-free (1.5x;
# now faster than native torch's 3.72 ms).
_BLOCK = 8192


@libentry()
@triton.jit
def tanh_kernel_direct(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    y = _tanh(x.to(tl.float32))
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty), mask=m)


@libentry()
@triton.jit
def tanh_kernel_direct_nomask(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = _tanh(x.to(tl.float32))
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty))


@libentry()
@triton.jit
def tanh_backward_kernel_direct(y_ptr, dy_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    y = tl.load(y_ptr + offs, mask=m, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    o = dy * (1.0 - y * y)
    tl.store(out_ptr + offs, o.to(out_ptr.type.element_ty), mask=m)


@libentry()
@triton.jit
def tanh_backward_kernel_direct_nomask(y_ptr, dy_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(y_ptr + offs).to(tl.float32)
    dy = tl.load(dy_ptr + offs).to(tl.float32)
    o = dy * (1.0 - y * y)
    tl.store(out_ptr + offs, o.to(out_ptr.type.element_ty))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_forward(x):
    return _tanh(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward_generic(y, dy):
    y32 = y.to(tl.float32)
    return dy.to(tl.float32) * (1.0 - y32 * y32)


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
        if n % _BLOCK == 0:
            tanh_kernel_direct_nomask[(n // _BLOCK,)](x, out, n, BLOCK=_BLOCK)
        else:
            tanh_kernel_direct[(triton.cdiv(n, _BLOCK),)](x, out, n, BLOCK=_BLOCK)
    return out


def tanh(self):
    logger.debug("GEMS_SPACEMIT TANH")
    if _direct_ok(self):
        out = torch.empty_like(self)
        return _launch_direct(self, out)
    return tanh_forward(self)


def tanh_backward(grad_output, output):
    logger.debug("GEMS_SPACEMIT TANH_BACKWARD")
    if (
        _direct_ok(grad_output)
        and _direct_ok(output)
        and grad_output.shape == output.shape
        and grad_output.numel() > 0
    ):
        out = torch.empty_like(grad_output)
        n = grad_output.numel()
        with torch_device_fn.device(grad_output.device):
            if n % _BLOCK == 0:
                tanh_backward_kernel_direct_nomask[(n // _BLOCK,)](
                    output, grad_output, out, n, BLOCK=_BLOCK
                )
            else:
                tanh_backward_kernel_direct[(triton.cdiv(n, _BLOCK),)](
                    output, grad_output, out, n, BLOCK=_BLOCK
                )
        return out
    return tanh_backward_generic(output, grad_output)


def tanh_(A):
    logger.debug("GEMS_SPACEMIT TANH_")
    if _direct_ok(A):
        return _launch_direct(A, A)
    return tanh_forward(A, out0=A)
