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

# Direct-kernel fast path for floating dtypes. The generic pointwise wrapper
# measured ~1.3-4.7 ms slower per call than an equivalent direct launch on K3
# (2048x2048: 17.9 ms vs 13.2 ms; 1M: 4.6 ms vs 3.3 ms) because its generated
# kernel lowers worse; the direct kernel is bit-exact vs torch.rsqrt.
_BLOCK = 8192


@libentry()
@triton.jit
def rsqrt_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    tl.store(y_ptr + offs, tl.rsqrt(x.to(tl.float32)).to(y_ptr.type.element_ty), mask=m)


@libentry()
@triton.jit
def rsqrt_kernel_f64(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    tl.store(y_ptr + offs, tl.rsqrt(x), mask=m)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def rsqrt_func_generic(x):
    return tl.rsqrt(x.to(tl.float32))


def _launch_direct(x, out):
    n = x.numel()
    if n == 0:
        return out
    grid = (triton.cdiv(n, _BLOCK),)
    if x.dtype == torch.float64:
        rsqrt_kernel_f64[grid](x, out, n, BLOCK=_BLOCK)
    else:
        rsqrt_kernel[grid](x, out, n, BLOCK=_BLOCK)
    return out


def rsqrt(A):
    logger.debug("GEMS_SPACEMIT RSQRT")
    if A.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        x = A if A.is_contiguous() else A.contiguous()
        out = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            _launch_direct(x, out)
        return out
    return rsqrt_func_generic(A)


def rsqrt_(A):
    logger.debug("GEMS_SPACEMIT RSQRT_")
    if A.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        if A.is_contiguous():
            with torch_device_fn.device(A.device):
                _launch_direct(A, A)
            return A
        A.copy_(rsqrt(A))
        return A
    return rsqrt_func_generic(A, out0=A)
