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

"""Direct-kernel log-family fast paths for K3 (campaign 1013pm).

The generic path lowers tl.log to the scalar libdevice extern (~31 ms/1M fp32)
and log2/log10/log1p run scalar-slow as well.  These kernels compute log2(x)
via exponent extraction + the degree-8 minimax poly already used by pow.py
(fp32 max abs err ~1.1e-7 on [sqrt(1/2), sqrt(2))) and scale to ln / log10.
Measured (f32, interleaved, ms, previous stack -> now):
  2048^2 (4M): log2 90.7 -> 3.25 | ln 34.9 -> 3.25 | log10 149.6 -> 3.36
  1M:          log2 22.9 -> 0.89 | ln  9.4 -> 0.93 | log10  37.8 -> 0.92
(torch f32 is 8-11x slower than this on the same shapes; log2/log10 were not
wired before this batch at all.)
Launch: nested dispatch (math-bound) + BLOCK 16384; nomask when divisible,
masked tail otherwise.  Gates: contiguous, dtype f32/f16, numel >= 65536.
Kill switch: FLAGGEMS_SPACEMIT_LOG_FAST=0 (falls back to the previous paths:
elemwise_nested for log/log_, the common ops for log2/log10).
Specials (vs torch): +/-0 -> -inf; negatives and -inf -> NaN; +inf -> +inf;
NaN -> NaN.  log10 specials land within ~2 ulp of torch at large magnitudes
(up to ~8e-6 abs at |result| >= 38, typical <= 2e-7); log2/ln bit-match on
the tested special sets.  f16: computed in f32 and rounded; matches an
fp64->f16 oracle except ~0.04% one-ulp cases.  NOTE: log1p is deliberately
NOT wired (log(1+x) loses precision near x=0; prototype only).
"""

import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.log10 import log10 as _c_log10, log10_ as _c_log10_, log10_out as _c_log10_out
from flag_gems.ops.log2 import log2 as _c_log2, log2_ as _c_log2_
from flag_gems.runtime.backend._spacemit.ops import elemwise_nested as _en
from flag_gems.runtime.backend._spacemit.ops._dispatch import nested_dispatch

_LN2 = 0.6931471805599453
_LOG10_2 = 0.3010299956639812
_BLOCK = 16384
_MIN_N = 1 << 16
_FLT2 = (torch.float32, torch.float16)


def _fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_LOG_FAST", "1") != "0"


def _ok(A) -> bool:
    return (
        isinstance(A, torch.Tensor)
        and A.dtype in _FLT2
        and A.is_contiguous()
        and A.numel() >= _MIN_N
    )


@triton.jit
def _logf_core(x):
    # log2(x): exponent extraction + deg-8 minimax poly + IEEE specials.
    # Log-family semantics differ from pow's copy in pow.py on one point:
    # -inf must map to NaN (pow wants +inf there).
    xi = x.to(tl.int32, bitcast=True)
    neg = xi < 0
    zero = (xi << 1) == 0
    e_raw = (xi >> 23) & 0xFF
    sub = (e_raw == 0) & (zero == 0)
    xs = tl.where(sub, x * 16777216.0, x)
    xi2 = xs.to(tl.int32, bitcast=True)
    e_u = ((xi2 >> 23) & 0xFF) - 127 - tl.where(sub, 24, 0)
    m = ((xi2 & 0x007FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    big = m > 1.4142135623730951
    m = tl.where(big, m * 0.5, m)
    e_u = e_u + tl.where(big, 1, 0)
    f = m - 1.0
    p = -1.386511922e-01
    p = p * f + 2.343947738e-01
    p = p * f - 2.515639663e-01
    p = p * f + 2.871227264e-01
    p = p * f - 3.599664867e-01
    p = p * f + 4.809199274e-01
    p = p * f - 7.213613391e-01
    p = p * f + 1.442695022e00
    p = p * f + 4.150201960e-08
    res = e_u.to(tl.float32) + p
    inf_cls = (xi & 0x7F800000) == 0x7F800000
    mant_nz = (xi & 0x007FFFFF) != 0
    res = tl.where(inf_cls & mant_nz, float("nan"), res)  # NaN payloads
    res = tl.where(inf_cls & ((xi & 0x007FFFFF) == 0), float("inf"), res)  # +/-inf
    res = tl.where(neg & ((xi << 1) != 0), float("nan"), res)  # negatives incl -inf
    res = tl.where(zero, float("-inf"), res)  # +/-0 -> -inf
    return res


@triton.jit
def _logf_out_nm(x_ptr, y_ptr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    r = (_logf_core(x.to(tl.float32)) * SCALE).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, r)


@triton.jit
def _logf_out_m(x_ptr, y_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    r = (_logf_core(x.to(tl.float32)) * SCALE).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, r, mask=m)


@triton.jit
def _logf_ip_nm(x_ptr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    r = (_logf_core(x.to(tl.float32)) * SCALE).to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + offs, r)


@triton.jit
def _logf_ip_m(x_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    r = (_logf_core(x.to(tl.float32)) * SCALE).to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + offs, r, mask=m)


def _launch_out(A, out, scale):
    n = out.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    with nested_dispatch(n):
        if n % _BLOCK == 0:
            _logf_out_nm[grid](A, out, SCALE=scale, BLOCK=_BLOCK)
        else:
            _logf_out_m[grid](A, out, n, SCALE=scale, BLOCK=_BLOCK)
    return out


def _launch_ip(A, scale):
    n = A.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    with nested_dispatch(n):
        if n % _BLOCK == 0:
            _logf_ip_nm[grid](A, SCALE=scale, BLOCK=_BLOCK)
        else:
            _logf_ip_m[grid](A, n, SCALE=scale, BLOCK=_BLOCK)
    return A


# --- exports (rebound onto flag_gems globals by SpecOpRegistrar) --------------


def log(A):
    if _fast_enabled() and _ok(A):
        return _launch_out(A, torch.empty_like(A), _LN2)
    return _en.log(A)


def log_(A):
    if _fast_enabled() and _ok(A):
        return _launch_ip(A, _LN2)
    return _en.log_(A)


def log2(A):
    if _fast_enabled() and _ok(A):
        return _launch_out(A, torch.empty_like(A), 1.0)
    return _c_log2(A)


def log2_(A):
    if _fast_enabled() and _ok(A):
        return _launch_ip(A, 1.0)
    return _c_log2_(A)


def log10(A):
    if _fast_enabled() and _ok(A):
        return _launch_out(A, torch.empty_like(A), _LOG10_2)
    return _c_log10(A)


def log10_(A):
    if _fast_enabled() and _ok(A):
        return _launch_ip(A, _LOG10_2)
    return _c_log10_(A)


def log10_out(A, *, out=None):
    if out is None:
        return log10(A)
    if (
        _fast_enabled()
        and _ok(A)
        and isinstance(out, torch.Tensor)
        and out.shape == A.shape
        and out.dtype == A.dtype
        and out.is_contiguous()
    ):
        _launch_out(A, out, _LOG10_2)
        return out
    return _c_log10_out(A, out=out)
