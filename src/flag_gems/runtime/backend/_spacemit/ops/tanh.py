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
def tanh_backward_kernel_direct(y_ptr, dy_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    y = tl.load(y_ptr + offs, mask=m, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    o = dy * (1.0 - y * y)
    tl.store(out_ptr + offs, o.to(out_ptr.type.element_ty), mask=m)


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
    grid = (triton.cdiv(n, _BLOCK),)
    with torch_device_fn.device(x.device):
        tanh_kernel_direct[grid](x, out, n, BLOCK=_BLOCK)
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
        grid = (triton.cdiv(n, _BLOCK),)
        with torch_device_fn.device(grad_output.device):
            tanh_backward_kernel_direct[grid](
                output, grad_output, out, n, BLOCK=_BLOCK
            )
        return out
    return tanh_backward_generic(output, grad_output)


def tanh_(A):
    logger.debug("GEMS_SPACEMIT TANH_")
    if _direct_ok(A):
        return _launch_direct(A, A)
    return tanh_forward(A, out0=A)
