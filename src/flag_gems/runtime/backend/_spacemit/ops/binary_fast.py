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

"""Direct-kernel binary fast paths for K3 (campaign 1012night, wave2b).

probe21 (1012night, f32, ms, gain = current-stack / direct):
  op           @1M (gain)   @2048^2 (gain)  @512x4096 (gain)   launch path
  add          0.635 (1.90) 2.649 (1.84)    1.165 (2.31)       default
  sub          0.583 (2.16) 2.473 (1.93)    1.241 (2.07)       default
  gt           1.047 (1.79) 4.211 (1.55)    1.999 (1.78)       nested
  lt           1.102 (1.70) 4.004 (1.63)    2.011 (1.74)       nested
  eq           0.981 (2.04) 4.224 (1.56)    1.973 (1.80)       nested
  clamp        0.471 (1.93) 1.729 (1.54)    0.841 (1.78)       default
  clamp_max    0.448 (2.01) 1.682 (1.59)    0.816 (1.79)       default
  clamp_min    0.442 (5.37) 1.837 (4.91)    0.822 (5.62)       default
  clamp_tensor 0.850 (2.14) 3.378 (1.97)    1.756 (2.12)       default
  where_self   1.254 (2.00) 5.254 (1.79)    2.610 (1.86)       nested
All rows bitwise-identical to the current stack (diff=0).

Key findings: pure memory-flow ops (add/sub/clamp family) saturate on the
default launcher path at BLOCK=8192 (nested adds overhead); comparison and
select kernels are compute/lowering-bound on this backend, so the nested
launcher path wins there (same kernel body, ~1.5-2x).  clamp_min was never
wired to nested before (default path; 5-9 ms / 4M) -> direct kernel is the
biggest single win of the batch (4.9-5.6x).

Fast-path gates: both operands tensors, same dtype in {f32, f16} (add/sub),
{f32, f16} (comparisons; scalar-compare variants NOT fast-pathed, they keep
the nested wrapper), clamp family f32/f16 with scalar bounds of int/float
type (clamp_tensor: f32/f16 all-tensor), where: bool condition + two tensors
same dtype.  alpha must be 1 for add/sub.  n >= 65536; contiguous only;
otherwise fall back to the elemwise_nested wrappers (previous behaviour).
Kill switches: FLAGGEMS_SPACEMIT_BINARY_FAST=0 or FLAGGEMS_SPACEMIT_NESTED=0.
"""

import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._spacemit.ops import elemwise_nested as _en
from flag_gems.runtime.backend._spacemit.ops import where as _vw
from flag_gems.runtime.backend._spacemit.ops._dispatch import nested_dispatch
from flag_gems.ops.clamp import clamp_min as _c_clamp_min, clamp_min_ as _c_clamp_min_

_BLOCK = 8192
_MIN_N = 1 << 16

_FLT2 = (torch.float32, torch.float16)


def _fast_enabled() -> bool:
    return (
        os.environ.get("FLAGGEMS_SPACEMIT_BINARY_FAST", "1") != "0"
        and os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") != "0"
    )


def _ok2(A, B, dtypes=_FLT2) -> bool:
    return (
        isinstance(A, torch.Tensor)
        and isinstance(B, torch.Tensor)
        and A.shape == B.shape
        and A.dtype == B.dtype
        and A.dtype in dtypes
        and A.is_contiguous()
        and B.is_contiguous()
        and A.numel() >= _MIN_N
    )


def _ok1(A, dtypes=_FLT2) -> bool:
    return (
        isinstance(A, torch.Tensor)
        and A.dtype in dtypes
        and not A.is_complex()
        and A.is_contiguous()
        and A.numel() >= _MIN_N
    )


def _scalar(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --- kernels -----------------------------------------------------------------


@triton.jit
def _add_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))


@triton.jit
def _add_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


@triton.jit
def _sub_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs) - tl.load(y_ptr + offs))


@triton.jit
def _sub_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) - tl.load(y_ptr + offs, mask=m), mask=m)


@triton.jit
def _gt_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, x.to(tl.float32) > y)


@triton.jit
def _gt_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x.to(tl.float32) > y, mask=m)


@triton.jit
def _lt_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, x.to(tl.float32) < y)


@triton.jit
def _lt_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x.to(tl.float32) < y, mask=m)


@triton.jit
def _eq_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, x.to(tl.float32) == y.to(tl.float32))


@triton.jit
def _eq_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x.to(tl.float32) == y.to(tl.float32), mask=m)


@triton.jit
def _clamp_kernel(x_ptr, o_ptr, lo, hi, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.minimum(hi, tl.maximum(lo, x.to(tl.float32)))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))


@triton.jit
def _clamp_kernel_m(x_ptr, o_ptr, lo, hi, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.minimum(hi, tl.maximum(lo, x.to(tl.float32)))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _clamp_max_kernel(x_ptr, o_ptr, hi, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.minimum(hi, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))


@triton.jit
def _clamp_max_kernel_m(x_ptr, o_ptr, hi, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.minimum(hi, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _clamp_min_kernel(x_ptr, o_ptr, lo, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.maximum(lo, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))


@triton.jit
def _clamp_min_kernel_m(x_ptr, o_ptr, lo, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.maximum(lo, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _clamp_t3_kernel(x_ptr, lo_ptr, hi_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    lo = tl.load(lo_ptr + offs)
    hi = tl.load(hi_ptr + offs)
    y = tl.minimum(hi, tl.maximum(lo, x.to(tl.float32)))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))


@triton.jit
def _clamp_t3_kernel_m(x_ptr, lo_ptr, hi_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    lo = tl.load(lo_ptr + offs, mask=m)
    hi = tl.load(hi_ptr + offs, mask=m)
    y = tl.minimum(hi, tl.maximum(lo, x.to(tl.float32)))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _clamp_t2_min_kernel(x_ptr, lo_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    lo = tl.load(lo_ptr + offs)
    y = tl.maximum(lo, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))


@triton.jit
def _clamp_t2_min_kernel_m(x_ptr, lo_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    lo = tl.load(lo_ptr + offs, mask=m)
    y = tl.maximum(lo, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _clamp_t2_max_kernel(x_ptr, hi_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    hi = tl.load(hi_ptr + offs)
    y = tl.minimum(hi, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))


@triton.jit
def _clamp_t2_max_kernel_m(x_ptr, hi_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    hi = tl.load(hi_ptr + offs, mask=m)
    y = tl.minimum(hi, x.to(tl.float32))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _where_kernel(c_ptr, x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.load(c_ptr + offs)
    a = tl.load(x_ptr + offs)
    b = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, tl.where(c, a, b))


@triton.jit
def _where_kernel_m(c_ptr, x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    c = tl.load(c_ptr + offs, mask=m)
    a = tl.load(x_ptr + offs, mask=m)
    b = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, tl.where(c, a, b), mask=m)


def _launch2(kern, kern_m, A, B, out, nest=False):
    n = out.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    if n % _BLOCK == 0:
        call = lambda: kern[grid](A, B, out, BLOCK=_BLOCK)
    else:
        call = lambda: kern_m[grid](A, B, out, n, BLOCK=_BLOCK)
    if nest:
        with nested_dispatch(n):
            call()
    else:
        call()
    return out


def _launch1(kern, kern_m, A, out, post=(), nest=False):
    n = out.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    if n % _BLOCK == 0:
        call = lambda: kern[grid](A, out, *post, BLOCK=_BLOCK)
    else:
        call = lambda: kern_m[grid](A, out, *post, n, BLOCK=_BLOCK)
    if nest:
        with nested_dispatch(n):
            call()
    else:
        call()
    return out


# --- add / sub ---------------------------------------------------------------


def add(A, B, *, alpha=1):
    if _fast_enabled() and alpha == 1 and _ok2(A, B):
        return _launch2(_add_kernel, _add_kernel_m, A, B, torch.empty_like(A))
    return _en.add(A, B, alpha=alpha)


def add_(A, B, *, alpha=1):
    if _fast_enabled() and alpha == 1 and _ok2(A, B):
        _launch2(_add_kernel, _add_kernel_m, A, B, A)
        return A
    return _en.add_(A, B, alpha=alpha)


def sub(A, B, *, alpha=1):
    if _fast_enabled() and alpha == 1 and _ok2(A, B):
        return _launch2(_sub_kernel, _sub_kernel_m, A, B, torch.empty_like(A))
    return _en.sub(A, B, alpha=alpha)


def sub_(A, B, *, alpha=1):
    if _fast_enabled() and alpha == 1 and _ok2(A, B):
        _launch2(_sub_kernel, _sub_kernel_m, A, B, A)
        return A
    return _en.sub_(A, B, alpha=alpha)


def subtract(A, B, *, alpha=1):
    if _fast_enabled() and alpha == 1 and _ok2(A, B):
        return _launch2(_sub_kernel, _sub_kernel_m, A, B, torch.empty_like(A))
    return _en.subtract(A, B, alpha=alpha)


def subtract_(A, B, *, alpha=1):
    if _fast_enabled() and alpha == 1 and _ok2(A, B):
        _launch2(_sub_kernel, _sub_kernel_m, A, B, A)
        return A
    return _en.subtract_(A, B, alpha=alpha)


# --- comparisons (nested launcher path wins for these kernels) ---------------


def gt(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch2(_gt_kernel, _gt_kernel_m, A, B, torch.empty_like(A, dtype=torch.bool), nest=True)
    return _en.gt(A, B)


def lt(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch2(_lt_kernel, _lt_kernel_m, A, B, torch.empty_like(A, dtype=torch.bool), nest=True)
    return _en.lt(A, B)


def eq(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch2(_eq_kernel, _eq_kernel_m, A, B, torch.empty_like(A, dtype=torch.bool), nest=True)
    return _en.eq(A, B)


# --- clamp family (scalar and tensor bounds) ---------------------------------


def _clamp_fast(A, lo, hi, out=None):
    """lo/hi scalar or None; at least one present."""
    o = out if out is not None else torch.empty_like(A)
    if hi is not None and lo is not None:
        _launch1(_clamp_kernel, _clamp_kernel_m, A, o, (lo, hi))
    elif hi is not None:
        _launch1(_clamp_max_kernel, _clamp_max_kernel_m, A, o, (hi,))
    else:
        _launch1(_clamp_min_kernel, _clamp_min_kernel_m, A, o, (lo,))
    return o


def _tensor_ok(A, t) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.shape == A.shape
        and t.dtype == A.dtype
        and t.is_contiguous()
    )


def _clamp_t3_fast(A, lo, hi, out=None):
    o = out if out is not None else torch.empty_like(A)
    n = A.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    if n % _BLOCK == 0:
        _clamp_t3_kernel[grid](A, lo, hi, o, BLOCK=_BLOCK)
    else:
        _clamp_t3_kernel_m[grid](A, lo, hi, o, n, BLOCK=_BLOCK)
    return o


def _clamp_t2_min_fast(A, lo, out=None):
    o = out if out is not None else torch.empty_like(A)
    n = A.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    if n % _BLOCK == 0:
        _clamp_t2_min_kernel[grid](A, lo, o, BLOCK=_BLOCK)
    else:
        _clamp_t2_min_kernel_m[grid](A, lo, o, n, BLOCK=_BLOCK)
    return o


def _clamp_t2_max_fast(A, hi, out=None):
    o = out if out is not None else torch.empty_like(A)
    n = A.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    if n % _BLOCK == 0:
        _clamp_t2_max_kernel[grid](A, hi, o, BLOCK=_BLOCK)
    else:
        _clamp_t2_max_kernel_m[grid](A, hi, o, n, BLOCK=_BLOCK)
    return o


def _clamp_route(A, mini, maxi, out=None):
    """Fast kernel for any supported (mini, maxi) combination.
    Returns the result tensor (or out) on success, None when no fast path
    applies (mixed scalar+tensor bounds, mismatched shapes/dtypes)."""
    if mini is None and maxi is None:
        return None
    both_scalar = (mini is None or _scalar(mini)) and (maxi is None or _scalar(maxi))
    if both_scalar:
        return _clamp_fast(A, mini, maxi, out=out)
    if isinstance(mini, torch.Tensor) and _tensor_ok(A, mini):
        if maxi is None:
            return _clamp_t2_min_fast(A, mini, out=out)
        if isinstance(maxi, torch.Tensor) and _tensor_ok(A, maxi):
            return _clamp_t3_fast(A, mini, maxi, out=out)
    if mini is None and isinstance(maxi, torch.Tensor) and _tensor_ok(A, maxi):
        return _clamp_t2_max_fast(A, maxi, out=out)
    return None


def clamp(A, mini=None, maxi=None):
    if _fast_enabled() and _ok1(A):
        o = _clamp_route(A, mini, maxi)
        if o is not None:
            return o
    return _en.clamp(A, mini=mini, maxi=maxi)


def clamp_(A, mini=None, maxi=None):
    if _fast_enabled() and _ok1(A):
        o = _clamp_route(A, mini, maxi, out=A)
        if o is not None:
            return A
    return _en.clamp_(A, mini=mini, maxi=maxi)


def clamp_max(A, max_value):
    if _fast_enabled() and _ok1(A):
        if _scalar(max_value):
            return _clamp_fast(A, None, max_value)
        if isinstance(max_value, torch.Tensor) and _tensor_ok(A, max_value):
            return _clamp_t2_max_fast(A, max_value)
    return _en.clamp_max(A, max_value)


def clamp_max_(A, max_value):
    if _fast_enabled() and _ok1(A):
        if _scalar(max_value):
            _clamp_fast(A, None, max_value, out=A)
            return A
        if isinstance(max_value, torch.Tensor) and _tensor_ok(A, max_value):
            _clamp_t2_max_fast(A, max_value, out=A)
            return A
    return _en.clamp_max_(A, max_value)


def clamp_min(A, min_value):
    if _fast_enabled() and _ok1(A):
        if _scalar(min_value):
            return _clamp_fast(A, min_value, None)
        if isinstance(min_value, torch.Tensor) and _tensor_ok(A, min_value):
            return _clamp_t2_min_fast(A, min_value)
    return _c_clamp_min(A, min_value)


def clamp_min_(A, min_value):
    if _fast_enabled() and _ok1(A):
        if _scalar(min_value):
            _clamp_fast(A, min_value, None, out=A)
            return A
        if isinstance(min_value, torch.Tensor) and _tensor_ok(A, min_value):
            _clamp_t2_min_fast(A, min_value, out=A)
            return A
    return _c_clamp_min_(A, min_value)


# --- clamp_tensor ------------------------------------------------------------


def clamp_tensor(A, mini=None, maxi=None):
    if _fast_enabled() and _ok1(A):
        o = _clamp_route(A, mini, maxi)
        if o is not None:
            return o
    return _en.clamp_tensor(A, mini=mini, maxi=maxi)


def clamp_tensor_(A, mini=None, maxi=None):
    if _fast_enabled() and _ok1(A):
        o = _clamp_route(A, mini, maxi, out=A)
        if o is not None:
            return A
    return _en.clamp_tensor_(A, mini=mini, maxi=maxi)


# --- where_self --------------------------------------------------------------


def _where_fast(condition, self_, other, out=None):
    o = out if out is not None else torch.empty_like(self_)
    n = o.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    if n % _BLOCK == 0:
        with nested_dispatch(n):
            _where_kernel[grid](condition, self_, other, o, BLOCK=_BLOCK)
    else:
        with nested_dispatch(n):
            _where_kernel_m[grid](condition, self_, other, o, n, BLOCK=_BLOCK)
    return o


def _where_ok(condition, self_, other, out) -> bool:
    return (
        isinstance(condition, torch.Tensor)
        and condition.dtype == torch.bool
        and isinstance(self_, torch.Tensor)
        and isinstance(other, torch.Tensor)
        and self_.dtype == other.dtype
        and self_.dtype in _FLT2
        and condition.shape == self_.shape == other.shape
        and condition.is_contiguous()
        and self_.is_contiguous()
        and other.is_contiguous()
        and (out is None or (out.is_contiguous() and out.dtype == self_.dtype and out.numel() == self_.numel()))
        and self_.numel() >= _MIN_N
    )


def where_self(condition, self_, other):
    if _fast_enabled() and _where_ok(condition, self_, other, None):
        return _where_fast(condition, self_, other)
    return _vw.where_self(condition, self_, other)


def where_self_out(condition, self_, other, out=None):
    if out is not None and _fast_enabled() and _where_ok(condition, self_, other, out):
        return _where_fast(condition, self_, other, out)
    return _vw.where_self_out(condition, self_, other, out=out)
