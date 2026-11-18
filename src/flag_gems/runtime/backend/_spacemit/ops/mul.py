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

"""Direct-kernel mul fast path for K3 (campaign 2026-10-12 pm).

The common mul implementation dispatches a hand-written kernel with
BLOCK_SIZE=1024 and unconditional masking, which measures ~7.0-7.5 ms for
4M f32 elements on K3 (2.6-3.1 ns/elem).  A direct BLOCK=8192 kernel without
the mask (when n % 8192 == 0) measures ~2.6 ms for the same workload
(1.5x-2.9x faster depending on shape); the masked variant covers the
tail.  The spine-triton nested dispatch path is *slower* for this kernel
(4.2 vs 2.6 ms), so no nested_dispatch is used here.

Numerics: elementwise x*y is bit-identical to the common path (single IEEE
rounding); the nomask variant only removes predicate overhead.

Fast path conditions: both operands tensors, same shape, same dtype in
{float32, float16, int32}, both contiguous, no complex - everything else
falls back to the common implementation (broadcast / scalar / bool / out=).
"""

import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.mul import mul as _common_mul, mul_ as _common_mul_
from flag_gems.ops.multiply import multiply as _common_multiply
from flag_gems.ops.multiply_ import multiply_ as _common_multiply_

_BLOCK = 8192
_MIN_N = 1 << 16
_FAST_DTYPES = (torch.float32, torch.float16, torch.int32)


def _fast_enabled() -> bool:
    """Fast path kill switches: FLAGGEMS_SPACEMIT_MUL_FAST=0 (this kernel only)
    or FLAGGEMS_SPACEMIT_NESTED=0 (global campaign "pure official" mode)."""
    return (
        os.environ.get("FLAGGEMS_SPACEMIT_MUL_FAST", "1") != "0"
        and os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") != "0"
    )


@triton.jit
def mul_fast_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x_ptr + offs)
    b = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, a * b)


@triton.jit
def mul_fast_kernel_masked(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(x_ptr + offs, mask=m)
    b = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, a * b, mask=m)


def _mul_launch(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    n = out.numel()
    if n == 0:
        return
    grid = (triton.cdiv(n, _BLOCK),)
    if n % _BLOCK == 0:
        mul_fast_kernel[grid](x, y, out, BLOCK=_BLOCK)
    else:
        mul_fast_kernel_masked[grid](x, y, out, n, BLOCK=_BLOCK)


def _fast_ok(a, b) -> bool:
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.shape == b.shape
        and a.dtype == b.dtype
        and a.dtype in _FAST_DTYPES
        and a.is_contiguous()
        and b.is_contiguous()
        and a.numel() >= _MIN_N
        and a.device == b.device
    )


def mul(A, B, *, out=None):
    if out is None and _fast_ok(A, B) and _fast_enabled():
        o = torch.empty_like(A)
        _mul_launch(A, B, o)
        return o
    return _common_mul(A, B, out=out)


def mul_(A, B):
    if _fast_ok(A, B) and _fast_enabled():
        _mul_launch(A, B, A)
        return A
    return _common_mul_(A, B)


def multiply(A, B):
    """torch.multiply alias - route through the same fast path."""
    if _fast_ok(A, B) and _fast_enabled():
        o = torch.empty_like(A)
        _mul_launch(A, B, o)
        return o
    return _common_multiply(A, B)


def multiply_(A, B):
    if _fast_ok(A, B) and _fast_enabled():
        _mul_launch(A, B, A)
        return A
    return _common_multiply_(A, B)
