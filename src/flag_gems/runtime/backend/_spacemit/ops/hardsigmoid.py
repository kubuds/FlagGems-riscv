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
from flag_gems.utils import libentry
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Direct-kernel fast path for floating dtypes (campaign 1011am). The generic
# implementation is already a direct kernel but runs BLOCK=1024 (1M: 1.82 ms
# vs 0.63 ms at BLOCK=8192; 2048x2048: 8.70 -> 2.50 ms) plus per-call grid
# lambda overhead. Numerically equivalent (maxdiff 5.96e-08 on fp32 randn).
# campaign 1011pm: mask-free variant for n divisible by BLOCK (mask handling
# costs ~0.6 ms at 2048^2: 2.68 -> 2.09 ms measured); hardsigmoid_out added
# for the aten out= variant (was on the generic BLOCK=1024 kernel: 8.65 ms
# at 2048^2 vs 2.1 ms with the direct kernel).
_BLOCK = 8192


@libentry()
@triton.jit
def hardsigmoid_kernel_direct(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    x32 = x.to(tl.float32)
    y = x32 * (1.0 / 6.0) + 0.5
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    # torch propagates NaN through hardsigmoid; tl.minimum/maximum do not on
    # this backend (max(nan, 0) -> 0). Restore NaN positions explicitly.
    y = tl.where(x32 != x32, float("nan"), y)
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty), mask=m)


@libentry()
@triton.jit
def hardsigmoid_kernel_direct_nomask(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    x32 = x.to(tl.float32)
    y = x32 * (1.0 / 6.0) + 0.5
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    y = tl.where(x32 != x32, float("nan"), y)
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def hardsigmoid_forward(x):
    x32 = x.to(tl.float32)
    y = x32 * (1.0 / 6.0) + 0.5
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    y = tl.where(x32 != x32, float("nan"), y)
    return y


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
            hardsigmoid_kernel_direct_nomask[(n // _BLOCK,)](x, out, n, BLOCK=_BLOCK)
        else:
            grid = (triton.cdiv(n, _BLOCK),)
            hardsigmoid_kernel_direct[grid](x, out, n, BLOCK=_BLOCK)
    return out


def hardsigmoid(x):
    logger.debug("GEMS_SPACEMIT HARDSIGMOID")
    if _direct_ok(x):
        out = torch.empty_like(x)
        return _launch_direct(x, out)
    return hardsigmoid_forward(x)


def hardsigmoid_out(x, out):
    logger.debug("GEMS_SPACEMIT HARDSIGMOID_OUT")
    if (
        isinstance(out, torch.Tensor)
        and out.shape == x.shape
        and _direct_ok(x)
        and _direct_ok(out)
    ):
        return _launch_direct(x, out)
    return hardsigmoid_forward(x, out0=out)


def hardsigmoid_(A):
    logger.debug("GEMS_SPACEMIT HARDSIGMOID_")
    if _direct_ok(A):
        return _launch_direct(A, A)
    return hardsigmoid_forward(A, out0=A)
