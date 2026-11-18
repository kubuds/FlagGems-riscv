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

from flag_gems.utils import tl_extra_shim
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_TENSOR_")
    return pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


@triton.jit
def pow_by_mul_kernel(
    X_ptr,
    Out_ptr,
    n_elements,
    exponent: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X_ptr + offsets, mask=mask)
    result = x
    for _ in range(1, exponent):
        result = result * x
    tl.store(Out_ptr + offsets, result, mask=mask)


def _pow_by_mul(A, exponent):
    out = torch.empty_like(A)
    n_elements = A.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    pow_by_mul_kernel[grid](
        A,
        out,
        n_elements,
        exponent=exponent,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _pow_fastpath_exponent(A, exponent):
    """Integer exponent for the multiplication fast path, else None.

    Covers integer-valued positive exponents (int or float, e.g. 2 or 2.0)
    within [2, 64] on contiguous float32 tensors.  The multiplication path is
    faster than the generic pow path on this backend and is also correct for
    negative inputs, where the generic path evaluates log(x) and yields NaN.
    """
    if isinstance(exponent, bool):
        return None
    if isinstance(exponent, int):
        e = exponent
    elif isinstance(exponent, float) and exponent.is_integer():
        e = int(exponent)
    else:
        return None
    if 2 <= e <= 64 and A.dtype == torch.float32 and A.is_contiguous():
        return e
    return None


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_SCALAR")
    e = _pow_fastpath_exponent(A, exponent)
    if e is not None:
        return _pow_by_mul(A, e)
    if isinstance(exponent, int) and not isinstance(exponent, bool) and exponent > 0:
        has_neg = bool((A < 0).any().item())
        if has_neg:
            return _pow_by_mul(A, exponent)
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_SCALAR_")
    e = _pow_fastpath_exponent(A, exponent)
    if e is not None:
        result = _pow_by_mul(A, e)
        A.copy_(result)
        return A
    if isinstance(exponent, int) and not isinstance(exponent, bool) and exponent > 0:
        has_neg = bool((A < 0).any().item())
        if has_neg:
            result = _pow_by_mul(A, exponent)
            A.copy_(result)
            return A
    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent)


def pow_scalar(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
