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

logger = logging.getLogger(__name__)

# Direct-kernel fast path for copy_ (campaign 1011night). The generic pointwise
# `_copy_kernel` runs at ~0.73 GB/s on K3 (2048x2048 fp32: 46 ms); the direct
# kernel reaches ~19.6 GB/s (1.7 ms) for same-dtype contiguous copies. The
# generic implementation is kept for every other case (dtype casts, broadcast,
# non-contiguous, alias handling, fill values, ...).
_BLOCK = 8192

_ALLOWED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.uint8,
    torch.bool,
)


@libentry()
@triton.jit
def copy_direct_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, v, mask=m)


def _fast_path_ok(dst, src):
    if not isinstance(src, torch.Tensor):
        return False
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.dtype != src.dtype or dst.dtype not in _ALLOWED_DTYPES:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if not (dst.is_contiguous() and src.is_contiguous()):
        return False
    if dst.shape != src.shape:
        return False
    n = dst.numel()
    if n == 0 or n > 2**31 - 1:
        return False
    try:
        if torch._C._is_alias_of(dst, src):
            return False
        if dst._is_zerotensor() or src._is_zerotensor():
            return False
    except Exception:
        return False
    return True


def copy_(dst, src, non_blocking=False):
    if _fast_path_ok(dst, src):
        n = dst.numel()
        with torch_device_fn.device(dst.device):
            copy_direct_kernel[(triton.cdiv(n, _BLOCK),)](src, dst, n, BLOCK=_BLOCK)
        return dst
    from flag_gems.ops.copy import copy_ as _generic_copy_

    return _generic_copy_(dst, src, non_blocking)
