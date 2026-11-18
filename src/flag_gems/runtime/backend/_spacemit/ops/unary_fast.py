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

"""Direct-kernel unary fast paths for K3 (campaign 1012night, wave2).

The nested-dispatch wrappers (elemwise_nested.py) parallelize the common
tile-512 ``pointwise_dynamic`` kernels (~2.6-3.2 ms / 4M f32).  A direct
BLOCK=8192 mask-free kernel on the *default* launcher path reaches the
memory-bound floor for the cheap ops and still wins for the others:

probe19 (1012night, f32, ms; gain = current-stack / direct):
  op        @1M (gain)   @2048^2 (gain)  @512x4096 (gain)
  abs       0.443 (1.96) 1.615 (1.64)    0.799 (1.83)
  neg       0.451 (1.96) 1.674 (1.58)    0.808 (1.84)
  relu      0.539 (1.71) 2.020 (1.34)    0.995 (1.56)
  square    0.456 (1.92) 1.639 (1.65)    0.800 (1.81)
  trunc     0.549 (1.64) 2.143 (1.35)    1.006 (1.53)
  sign      0.675 (2.29) 2.558 (3.48)    1.236 (3.03)
  elu(nst)  1.094 (4.12) 4.316 (4.01)    2.117 (4.20)
All rows bitwise-identical to the current stack (diff=0).  sqrt/exp/
reciprocal measured *slower* as raw kernels (libdevice-bound: keep nested
pointwise); ceil/floor marginal, not wired.

elu uses the *nested* launcher path inside its fast path (its exp is
compute-bound; nested 4.0x vs default 2.4x); the others stay on the default
path (nested is slower for memory-bound kernels).

Kill switches: FLAGGEMS_SPACEMIT_UNARY_FAST=0 (this module only) or
FLAGGEMS_SPACEMIT_NESTED=0 (global campaign "pure official" mode).  Fallback
(cache-ineligible shapes / small n / other dtypes) = the elemwise_nested
wrapper, i.e. exactly the previous shipped behaviour.
"""

import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._spacemit.ops import elemwise_nested as _en
from flag_gems.runtime.backend._spacemit.ops._dispatch import nested_dispatch
from flag_gems.ops.sign import sign as _common_sign, sign_out as _common_sign_out
from flag_gems.utils import tl_extra_shim

_BLOCK = 8192
_MIN_N = 1 << 16

_FLT2 = (torch.float32, torch.float16)
_FLT2I = (torch.float32, torch.float16, torch.int32)


def _fast_enabled() -> bool:
    return (
        os.environ.get("FLAGGEMS_SPACEMIT_UNARY_FAST", "1") != "0"
        and os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") != "0"
    )


def _ok(x, out=None, dtypes=_FLT2) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.dtype not in dtypes or x.is_complex():
        return False
    if not x.is_contiguous():
        return False
    if out is not None and (
        not isinstance(out, torch.Tensor)
        or not out.is_contiguous()
        or out.dtype != x.dtype
        or out.numel() != x.numel()
    ):
        return False
    return x.numel() >= _MIN_N


# --- kernels (nomask + masked tail) -----------------------------------------


@triton.jit
def _abs_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.abs(tl.load(x_ptr + offs)))


@triton.jit
def _abs_kernel_m(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.abs(tl.load(x_ptr + offs, mask=m)), mask=m)


@triton.jit
def _neg_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, -tl.load(x_ptr + offs))


@triton.jit
def _neg_kernel_m(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, -tl.load(x_ptr + offs, mask=m), mask=m)


@triton.jit
def _relu_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, tl.where(x > 0, x, 0))


@triton.jit
def _relu_kernel_m(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, tl.where(x > 0, x, 0), mask=m)


@triton.jit
def _square_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x * x)


@triton.jit
def _square_kernel_m(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x * x, mask=m)


@triton.jit
def _trunc_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl_extra_shim.trunc(x.to(tl.float32)).to(x.dtype)
    tl.store(o_ptr + offs, y)


@triton.jit
def _trunc_kernel_m(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl_extra_shim.trunc(x.to(tl.float32)).to(x.dtype)
    tl.store(o_ptr + offs, y, mask=m)


@triton.jit
def _sign_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    pos = (x > 0).to(x.dtype)
    neg = (x < 0).to(x.dtype)
    tl.store(o_ptr + offs, pos - neg)


@triton.jit
def _sign_kernel_m(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    pos = (x > 0).to(x.dtype)
    neg = (x < 0).to(x.dtype)
    tl.store(o_ptr + offs, pos - neg, mask=m)


@triton.jit
def _elu_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    x32 = x.to(tl.float32)
    y = tl.where(x > 0, x32, tl.exp(x32) - 1)
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty))


@triton.jit
def _elu_kernel_m(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    x32 = x.to(tl.float32)
    y = tl.where(x > 0, x32, tl.exp(x32) - 1)
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


def _launch(kern, kern_m, x, out, nest=False):
    n = x.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    if n % _BLOCK == 0:
        call = lambda: kern[grid](x, out, BLOCK=_BLOCK)
    else:
        call = lambda: kern_m[grid](x, out, n, BLOCK=_BLOCK)
    if nest:
        with nested_dispatch(n):
            call()
    else:
        call()
    return out


# --- abs / absolute ----------------------------------------------------------


def abs(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        return _launch(_abs_kernel, _abs_kernel_m, A, torch.empty_like(A))
    return _en.abs(A)


def abs_(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        _launch(_abs_kernel, _abs_kernel_m, A, A)
        return A
    return _en.abs_(A)


def absolute(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        return _launch(_abs_kernel, _abs_kernel_m, A, torch.empty_like(A))
    return _en.absolute(A)


def absolute_(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        _launch(_abs_kernel, _abs_kernel_m, A, A)
        return A
    return _en.absolute_(A)


# --- neg / negative ----------------------------------------------------------


def neg(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        return _launch(_neg_kernel, _neg_kernel_m, A, torch.empty_like(A))
    return _en.neg(A)


def neg_(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        _launch(_neg_kernel, _neg_kernel_m, A, A)
        return A
    return _en.neg_(A)


def negative(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        return _launch(_neg_kernel, _neg_kernel_m, A, torch.empty_like(A))
    return _en.negative(A)


def negative_(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        _launch(_neg_kernel, _neg_kernel_m, A, A)
        return A
    return _en.negative_(A)


# --- relu --------------------------------------------------------------------


def relu(A):
    if _fast_enabled() and _ok(A):
        return _launch(_relu_kernel, _relu_kernel_m, A, torch.empty_like(A))
    return _en.relu(A)


def relu_(A):
    if _fast_enabled() and _ok(A):
        _launch(_relu_kernel, _relu_kernel_m, A, A)
        return A
    return _en.relu_(A)


# --- square ------------------------------------------------------------------


def square(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        return _launch(_square_kernel, _square_kernel_m, A, torch.empty_like(A))
    return _en.square(A)


def square_(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        _launch(_square_kernel, _square_kernel_m, A, A)
        return A
    return _en.square_(A)


def square_out(A, *, out=None):
    if out is None:
        return square(A)
    if _fast_enabled() and _ok(A, out=out, dtypes=_FLT2I):
        return _launch(_square_kernel, _square_kernel_m, A, out)
    return _en.square_out(A, out=out)


# --- trunc -------------------------------------------------------------------


def trunc(A):
    if _fast_enabled() and _ok(A):
        return _launch(_trunc_kernel, _trunc_kernel_m, A, torch.empty_like(A))
    return _en.trunc(A)


def trunc_(A):
    if _fast_enabled() and _ok(A):
        _launch(_trunc_kernel, _trunc_kernel_m, A, A)
        return A
    return _en.trunc_(A)


# --- sign --------------------------------------------------------------------


def sign(A):
    if _fast_enabled() and _ok(A, dtypes=_FLT2I):
        return _launch(_sign_kernel, _sign_kernel_m, A, torch.empty_like(A))
    return _common_sign(A)


def sign_out(A, *, out):
    if _fast_enabled() and _ok(A, out=out, dtypes=_FLT2I):
        return _launch(_sign_kernel, _sign_kernel_m, A, out)
    return _common_sign_out(A, out=out)


# --- elu ---------------------------------------------------------------------


def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    if (
        _fast_enabled()
        and alpha == 1.0
        and scale == 1.0
        and input_scale == 1.0
        and _ok(A)
    ):
        return _launch(_elu_kernel, _elu_kernel_m, A, torch.empty_like(A), nest=True)
    return _en.elu(A, alpha=alpha, scale=scale, input_scale=input_scale)


def elu_(A, alpha=1.0, scale=1.0, input_scale=1.0):
    if (
        _fast_enabled()
        and alpha == 1.0
        and scale == 1.0
        and input_scale == 1.0
        and _ok(A)
    ):
        return _launch(_elu_kernel, _elu_kernel_m, A, A, nest=True)
    return _en.elu_(A, alpha=alpha, scale=scale, input_scale=input_scale)
