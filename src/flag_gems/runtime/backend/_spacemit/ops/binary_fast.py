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

Campaign 1013pm (wave2d) correction for the bool-output comparison family:
  * ALL bool-output comparisons (gt/lt/eq/ge/le/ne + scalar variants + the
    alias family) now write the predicate as a BYTE through a uint8 view of
    the bool buffer, and use the DEFAULT launcher at BLOCK 32768.  The
    previous form (direct i1 store, nested launch) was ~3x slower: spine
    PtrAnalysis cannot rewrite that predicate store ("PtrAnalysis: storeOp
    cannot be rewritten"), leaving a slow per-element path.  Measured f32:
    ge_scalar 2048^2 4.37 -> 1.01 ms (torch 1.93); ge 2048^2 5.09 -> 1.78
    (torch 3.28); 1M scalar 1.32 -> 0.31, 1M tensor 1.32 -> 0.47.  Inplace
    and float-output paths are unchanged by this batch.
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
_CMP_BLOCK = 1 << 15  # bool-output comparisons: default launch, byte stores

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
    tl.store(o_ptr + offs, (x.to(tl.float32) > y).to(o_ptr.dtype.element_ty))


@triton.jit
def _gt_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) > y).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _lt_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) < y).to(o_ptr.dtype.element_ty))


@triton.jit
def _lt_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) < y).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _eq_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) == y.to(tl.float32)).to(o_ptr.dtype.element_ty))


@triton.jit
def _eq_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) == y.to(tl.float32)).to(o_ptr.dtype.element_ty), mask=m)


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


def _launch_cmp2(kern, kern_m, A, B, out=None):
    """Bool-output comparison launch (campaign 1013pm wave2d).

    The spine backend does not vectorise a direct i1 store into the bool
    buffer, so the predicate is written as a byte via a uint8 view (see the
    module docstring).  Measured best with the DEFAULT launcher at
    BLOCK=32768; nested dispatch loses for byte stores.
    """
    if out is None:
        out = torch.empty_like(A, dtype=torch.bool)
    cell = out.view(torch.uint8)
    n = out.numel()
    grid = (triton.cdiv(n, _CMP_BLOCK),)
    if n % _CMP_BLOCK == 0:
        kern[grid](A, B, cell, BLOCK=_CMP_BLOCK)
    else:
        kern_m[grid](A, B, cell, n, BLOCK=_CMP_BLOCK)
    return out


def _launch_cmp1(kern, kern_m, A, v, out=None):
    """Scalar-comparison counterpart of _launch_cmp2."""
    if out is None:
        out = torch.empty_like(A, dtype=torch.bool)
    cell = out.view(torch.uint8)
    n = out.numel()
    grid = (triton.cdiv(n, _CMP_BLOCK),)
    if n % _CMP_BLOCK == 0:
        kern[grid](A, cell, v, BLOCK=_CMP_BLOCK)
    else:
        kern_m[grid](A, cell, v, n, BLOCK=_CMP_BLOCK)
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
        return _launch_cmp2(_gt_kernel, _gt_kernel_m, A, B)
    return _en.gt(A, B)


def lt(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_cmp2(_lt_kernel, _lt_kernel_m, A, B)
    return _en.lt(A, B)


def eq(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_cmp2(_eq_kernel, _eq_kernel_m, A, B)
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


# =============================================================================
# wave2c (campaign 1013am): comparison / select family completion.
#
# - ge / le / ne tensor kernels (nested launcher path, mirrors gt/lt/eq);
# - maximum / minimum (default launcher path; explicit select reproduces
#   torch's NaN propagation and signed-zero tie behaviour — tl.maximum /
#   tl.minimum do not propagate NaN on this backend);
# - in-place comparison kernels for gt_tensor_/lt_/eq_/ge_/le_/ne_ and the
#   greater_equal_ / less_ / less_equal_ aliases (bool result cast into the
#   float operand, matching aten in-place *.Tensor semantics);
# - scalar-comparison kernels for gt/ge/lt/le/eq/ne scalars; the scalar is
#   pre-rounded to the tensor dtype host-side (torch scalar-typing parity;
#   fixes f16 scalar mismatches seen with raw f32 constants);
# - alias wiring: aten less_ / less_equal* / greater* / not_equal* reach the
#   same kernels through the flag_gems op table.
# =============================================================================

from flag_gems.ops.greater import (  # noqa: E402
    greater as _c_greater,
    greater_scalar as _c_greater_scalar,
    greater_out as _c_greater_out,
    greater_scalar_out as _c_greater_scalar_out,
)
from flag_gems.ops.greater_equal import (  # noqa: E402
    greater_equal_ as _c_greater_equal_,
)
from flag_gems.ops.maximum import maximum as _c_maximum  # noqa: E402
from flag_gems.ops.minimum import minimum as _c_minimum  # noqa: E402
from flag_gems.ops.not_equal import (  # noqa: E402
    not_equal as _c_not_equal,
    not_equal_scalar as _c_not_equal_scalar,
)


def _scalar_cast(A, B):
    """Round the scalar to the tensor dtype first (torch scalar typing: half
    tensors compare against a half-rounded scalar)."""
    return float(torch.tensor(B, dtype=A.dtype).item())


# --- ge / le / ne (tensor) ---------------------------------------------------


@triton.jit
def _ge_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) >= y).to(o_ptr.dtype.element_ty))


@triton.jit
def _ge_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) >= y).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _le_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) <= y).to(o_ptr.dtype.element_ty))


@triton.jit
def _le_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) <= y).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _ne_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) != y.to(tl.float32)).to(o_ptr.dtype.element_ty))


@triton.jit
def _ne_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) != y.to(tl.float32)).to(o_ptr.dtype.element_ty), mask=m)


def ge(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_cmp2(_ge_kernel, _ge_kernel_m, A, B)
    return _en.ge(A, B)


def le(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_cmp2(_le_kernel, _le_kernel_m, A, B)
    return _en.le(A, B)


def ne(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_cmp2(_ne_kernel, _ne_kernel_m, A, B)
    return _en.ne(A, B)


# --- maximum / minimum (select-based, torch NaN / signed-zero parity) ---------


@triton.jit
def _maximum_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    r = tl.where(x != x, x, tl.where(y != y, y, tl.maximum(x, y)))
    xb = x.to(tl.int32, bitcast=True)
    yb = y.to(tl.int32, bitcast=True)
    tie = (x == y) & ((xb ^ yb) != 0)
    r = tl.where(tie, y, r)
    tl.store(o_ptr + offs, r)


@triton.jit
def _maximum_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=m).to(tl.float32)
    r = tl.where(x != x, x, tl.where(y != y, y, tl.maximum(x, y)))
    xb = x.to(tl.int32, bitcast=True)
    yb = y.to(tl.int32, bitcast=True)
    tie = (x == y) & ((xb ^ yb) != 0)
    r = tl.where(tie, y, r)
    tl.store(o_ptr + offs, r, mask=m)


@triton.jit
def _minimum_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    r = tl.where(x != x, x, tl.where(y != y, y, tl.minimum(x, y)))
    xb = x.to(tl.int32, bitcast=True)
    yb = y.to(tl.int32, bitcast=True)
    tie = (x == y) & ((xb ^ yb) != 0)
    r = tl.where(tie, y, r)
    tl.store(o_ptr + offs, r)


@triton.jit
def _minimum_kernel_m(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=m).to(tl.float32)
    r = tl.where(x != x, x, tl.where(y != y, y, tl.minimum(x, y)))
    xb = x.to(tl.int32, bitcast=True)
    yb = y.to(tl.int32, bitcast=True)
    tie = (x == y) & ((xb ^ yb) != 0)
    r = tl.where(tie, y, r)
    tl.store(o_ptr + offs, r, mask=m)


def maximum(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch2(_maximum_kernel, _maximum_kernel_m, A, B, torch.empty_like(A))
    return _c_maximum(A, B)


def minimum(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch2(_minimum_kernel, _minimum_kernel_m, A, B, torch.empty_like(A))
    return _c_minimum(A, B)


# --- in-place comparisons (bool result cast into the float operand) ----------


def _launch_ip(kern, kern_m, A, B):
    n = A.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    if n % _BLOCK == 0:
        kern[grid](A, B, BLOCK=_BLOCK)
    else:
        kern_m[grid](A, B, n, BLOCK=_BLOCK)
    return A


@triton.jit
def _gt_ip_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(x_ptr + offs, (x.to(tl.float32) > y).to(x_ptr.dtype.element_ty))


@triton.jit
def _gt_ip_kernel_m(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) > y).to(x_ptr.dtype.element_ty), mask=m)


@triton.jit
def _lt_ip_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(x_ptr + offs, (x.to(tl.float32) < y).to(x_ptr.dtype.element_ty))


@triton.jit
def _lt_ip_kernel_m(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) < y).to(x_ptr.dtype.element_ty), mask=m)


@triton.jit
def _eq_ip_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(x_ptr + offs, (x.to(tl.float32) == y.to(tl.float32)).to(x_ptr.dtype.element_ty))


@triton.jit
def _eq_ip_kernel_m(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) == y.to(tl.float32)).to(x_ptr.dtype.element_ty), mask=m)


@triton.jit
def _ge_ip_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(x_ptr + offs, (x.to(tl.float32) >= y).to(x_ptr.dtype.element_ty))


@triton.jit
def _ge_ip_kernel_m(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) >= y).to(x_ptr.dtype.element_ty), mask=m)


@triton.jit
def _le_ip_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(x_ptr + offs, (x.to(tl.float32) <= y).to(x_ptr.dtype.element_ty))


@triton.jit
def _le_ip_kernel_m(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) <= y).to(x_ptr.dtype.element_ty), mask=m)


@triton.jit
def _ne_ip_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(x_ptr + offs, (x.to(tl.float32) != y.to(tl.float32)).to(x_ptr.dtype.element_ty))


@triton.jit
def _ne_ip_kernel_m(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) != y.to(tl.float32)).to(x_ptr.dtype.element_ty), mask=m)


def gt_tensor_(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_ip(_gt_ip_kernel, _gt_ip_kernel_m, A, B)
    return _en.gt_tensor_(A, B)


def lt_(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_ip(_lt_ip_kernel, _lt_ip_kernel_m, A, B)
    return _en.lt_(A, B)


def eq_(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_ip(_eq_ip_kernel, _eq_ip_kernel_m, A, B)
    return _en.eq_(A, B)


def le_(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_ip(_le_ip_kernel, _le_ip_kernel_m, A, B)
    return _en.le_(A, B)


def ne_(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_ip(_ne_ip_kernel, _ne_ip_kernel_m, A, B)
    return _en.ne_(A, B)


def less_(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_ip(_lt_ip_kernel, _lt_ip_kernel_m, A, B)
    return _en.less_(A, B)


def greater_equal_(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_ip(_ge_ip_kernel, _ge_ip_kernel_m, A, B)
    return _c_greater_equal_(A, B)


def not_equal_(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_ip(_ne_ip_kernel, _ne_ip_kernel_m, A, B)
    return _en.not_equal_(A, B)


# --- scalar comparisons ------------------------------------------------------


@triton.jit
def _gt_scalar_kernel(x_ptr, o_ptr, v, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) > v).to(o_ptr.dtype.element_ty))


@triton.jit
def _gt_scalar_kernel_m(x_ptr, o_ptr, v, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) > v).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _ge_scalar_kernel(x_ptr, o_ptr, v, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) >= v).to(o_ptr.dtype.element_ty))


@triton.jit
def _ge_scalar_kernel_m(x_ptr, o_ptr, v, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) >= v).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _lt_scalar_kernel(x_ptr, o_ptr, v, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) < v).to(o_ptr.dtype.element_ty))


@triton.jit
def _lt_scalar_kernel_m(x_ptr, o_ptr, v, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) < v).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _le_scalar_kernel(x_ptr, o_ptr, v, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) <= v).to(o_ptr.dtype.element_ty))


@triton.jit
def _le_scalar_kernel_m(x_ptr, o_ptr, v, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) <= v).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _eq_scalar_kernel(x_ptr, o_ptr, v, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) == v).to(o_ptr.dtype.element_ty))


@triton.jit
def _eq_scalar_kernel_m(x_ptr, o_ptr, v, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) == v).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _ne_scalar_kernel(x_ptr, o_ptr, v, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, (x.to(tl.float32) != v).to(o_ptr.dtype.element_ty))


@triton.jit
def _ne_scalar_kernel_m(x_ptr, o_ptr, v, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, (x.to(tl.float32) != v).to(o_ptr.dtype.element_ty), mask=m)


def _sc_ok(A, B) -> bool:
    return _fast_enabled() and _ok1(A) and _scalar(B)


def gt_scalar(A, B):
    if _sc_ok(A, B):
        return _launch_cmp1(_gt_scalar_kernel, _gt_scalar_kernel_m, A, _scalar_cast(A, B))
    return _en.gt_scalar(A, B)


def ge_scalar(A, B):
    if _sc_ok(A, B):
        return _launch_cmp1(_ge_scalar_kernel, _ge_scalar_kernel_m, A, _scalar_cast(A, B))
    return _en.ge_scalar(A, B)


def lt_scalar(A, B):
    if _sc_ok(A, B):
        return _launch_cmp1(_lt_scalar_kernel, _lt_scalar_kernel_m, A, _scalar_cast(A, B))
    return _en.lt_scalar(A, B)


def le_scalar(A, B):
    if _sc_ok(A, B):
        return _launch_cmp1(_le_scalar_kernel, _le_scalar_kernel_m, A, _scalar_cast(A, B))
    return _en.le_scalar(A, B)


def eq_scalar(A, B):
    if _sc_ok(A, B):
        return _launch_cmp1(_eq_scalar_kernel, _eq_scalar_kernel_m, A, _scalar_cast(A, B))
    return _en.eq_scalar(A, B)


def ne_scalar(A, B):
    if _sc_ok(A, B):
        return _launch_cmp1(_ne_scalar_kernel, _ne_scalar_kernel_m, A, _scalar_cast(A, B))
    return _en.ne_scalar(A, B)


# --- aliases: greater* / less_equal* / not_equal* ride the same kernels -------


def greater(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_cmp2(_gt_kernel, _gt_kernel_m, A, B)
    return _c_greater(A, B)


def greater_out(A, B, *, out=None):
    if out is None:
        return greater(A, B)
    if (
        _fast_enabled()
        and _ok2(A, B)
        and isinstance(out, torch.Tensor)
        and out.dtype == torch.bool
        and out.is_contiguous()
        and out.shape == A.shape
    ):
        _launch_cmp2(_gt_kernel, _gt_kernel_m, A, B, out)
        return out
    return _c_greater_out(A, B, out=out)


def greater_scalar(A, B):
    if _sc_ok(A, B):
        return _launch_cmp1(_gt_scalar_kernel, _gt_scalar_kernel_m, A, _scalar_cast(A, B))
    return _c_greater_scalar(A, B)


def greater_scalar_out(A, B, *, out=None):
    if out is None:
        return greater_scalar(A, B)
    if (
        _fast_enabled()
        and _ok1(A)
        and _scalar(B)
        and isinstance(out, torch.Tensor)
        and out.dtype == torch.bool
        and out.is_contiguous()
        and out.shape == A.shape
    ):
        _launch_cmp1(_gt_scalar_kernel, _gt_scalar_kernel_m, A, _scalar_cast(A, B), out)
        return out
    return _c_greater_scalar_out(A, B, out=out)


def less_equal(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_cmp2(_le_kernel, _le_kernel_m, A, B)
    return _en.less_equal(A, B)


def less_equal_scalar(A, B):
    if _sc_ok(A, B):
        return _launch_cmp1(_le_scalar_kernel, _le_scalar_kernel_m, A, _scalar_cast(A, B))
    return _en.less_equal_scalar(A, B)


def less_equal_(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_ip(_le_ip_kernel, _le_ip_kernel_m, A, B)
    return _en.less_equal_(A, B)


def not_equal(A, B):
    if _fast_enabled() and _ok2(A, B):
        return _launch_cmp2(_ne_kernel, _ne_kernel_m, A, B)
    return _c_not_equal(A, B)


def not_equal_scalar(A, B):
    if _sc_ok(A, B):
        return _launch_cmp1(_ne_scalar_kernel, _ne_scalar_kernel_m, A, _scalar_cast(A, B))
    return _c_not_equal_scalar(A, B)
