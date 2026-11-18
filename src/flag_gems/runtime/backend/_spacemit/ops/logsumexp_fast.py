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

"""wave18/19/20/21/22: logsumexp fast paths (row, dim=0, mid-dim).

The generic logsumexp keeps (TILE_N, TILE_K) / [TILE_N] masks and finishes with
tl.sum(z * tl.exp(m - m_reduced), axis=0) - a reduce fed by computed values,
which spine-triton lowers scalar.  Measured baseline (dim=1, f32):
6.92 / 104.29 / 52.21 / 108.02 ms at 512^2 / 2048^2 / 512x4096 / 8192x512
(0.45-0.48x torch).

wave20 added the mid-dim online fold.  probe183 cost breakdown on
1024x256x128 (33.5M elems): load+sum 12.6 ms, load+max 12.6 ms, load+one exp
61.3 ms => exp is the bottleneck at ~1.45 ns/elem (tl.exp lowers to a scalar
libm call on this backend), and the per-row online fold pays two of them.

wave21 added `_expf` (exp as 2^(x*log2e), degree-6 polynomial, integer-bitcast
scale; 1.15-1.38x over tl.exp) and the chunked mid-dim fold (RB rows per chunk,
one rescale and one exp per element): 1024x256x128 136.1 -> 65.2 ms.

wave22: the loop shape, not the arithmetic, was the remaining problem.
probe192b on the same tensor: unmasked load+sum is 0.273 ns/elem and the poly
exp 0.475 ns/elem, i.e. a two-pass fold should cost ~34 ms - the shipped kernel
was at 65.  probe193b/c found two causes:
  * masked loads everywhere (C % BN == 0 and B % RB == 0 are known at launch,
    so the masks can be dropped);
  * no unrolled load batch: a plain `for b in range(0, B)` loop issues one load
    per iteration and waits for it (two-pass form: 574 ms at BN=128!).  An
    RB-row static_range unroll supplies the memory-level parallelism.
Sweep result: mid-dim wants BN=64 (not BN=C) for C <= 128 and BN=256 above,
RB 8-16.  1024x256x128 65.2 -> 26.0 ms (0.78 ns/elem), 8x128x64x64 9.9 -> 9.0.
The row kernel had the same missing unroll: BLOCK=64 x U=4 unrolled, K=4 rows
per program gives 0.98 -> 0.67, 9.72 -> 6.82, 4.50 -> 3.18, 14.16 -> 9.56 ms
at 512^2 / 2048^2 / 512x4096 / 8192x512 (1.42-1.48x).

tl.range(num_stages=...) is a pathology on this build (identical work at
2-10x the plain-range cost, no benefit from stages 2/3) - never use it here.

Kill switches: FLAGGEMS_SPACEMIT_LOGSUMEXP_FAST=0, FLAGGEMS_SPACEMIT_LSE_POLY=0.
"""

import os

import triton
import triton.language as tl

from .fastmath import exp_fast as _expf
from .fastmath import exp_flag

_ROWS = 4
_ROWS_BLOCK = 64     # column tile of the unrolled row kernel (wave22, f32)
_ROWS_U = 4          # column blocks unrolled per step
_ROWS_BLOCK_F16 = 128  # f16: 64-wide blocks scalarize on this backend
_BLOCK_SMALL = 128   # N <= 1024 (masked fallback)
_BLOCK_LARGE = 256   # N > 1024    (masked fallback)
_MIN_N = 64
_DIM0_RB = 128
_DIM0_BN_SMALL = 256   # K < 2048
_DIM0_BN_LARGE = 512   # K >= 2048
_DIM0_MIN_M = 64
_MID_BN_SMALL = 64     # C <= 128: BN=C is 2.2x slower (wave22)
_MID_BN = 256
_MID_BN_LARGE = 512
_MID_RB = 8
_MID_RB_BIG = 16


@triton.jit
def _lse_rows(X, Out, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr, DEG: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    for i in tl.static_range(K):
        r = pid * K + i
        rmask = r < M
        base = r.to(tl.int64) * N
        m = tl.full([BLOCK_N], value=-float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            idx = off + cols
            mk = (idx < N) & rmask
            v = tl.load(X + base + idx, mask=mk, other=-float("inf")).to(tl.float32)
            m = tl.maximum(m, v)
        m_r = tl.max(m, axis=0)
        z = tl.zeros([BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            idx = off + cols
            mk = (idx < N) & rmask
            v = tl.load(X + base + idx, mask=mk, other=-float("inf")).to(tl.float32)
            z += _expf(v - m_r, DEG)
        res = m_r + tl.log(tl.sum(z, axis=0))
        # all -inf row: torch returns -inf (exp(-inf - -inf) is NaN)
        res = tl.where(m_r == -float("inf"), m_r, res)
        tl.store(Out + r, res.to(Out.dtype.element_ty), mask=rmask)


@triton.jit
def _lse_rows_u(X, Out, M, N, K: tl.constexpr, BLOCK: tl.constexpr, U: tl.constexpr,
                DEG: tl.constexpr, TAIL: tl.constexpr, RMASK: tl.constexpr):
    """wave22 row kernel: U column blocks unrolled per step, no masks when the
    row length is a multiple of BLOCK*U."""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    for i in tl.static_range(K):
        r = pid * K + i
        base = r.to(tl.int64) * N
        m = tl.full([BLOCK], value=-float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK * U):
            for u in tl.static_range(U):
                idx = off + u * BLOCK + cols
                if TAIL:
                    v = tl.load(X + base + idx, mask=idx < N, other=-float("inf")).to(tl.float32)
                else:
                    v = tl.load(X + base + idx).to(tl.float32)
                m = tl.maximum(m, v)
        m_r = tl.max(m, axis=0)
        z = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, N, BLOCK * U):
            for u in tl.static_range(U):
                idx = off + u * BLOCK + cols
                if TAIL:
                    v = tl.load(X + base + idx, mask=idx < N, other=-float("inf")).to(tl.float32)
                else:
                    v = tl.load(X + base + idx).to(tl.float32)
                z += _expf(v - m_r, DEG)
        res = m_r + tl.log(tl.sum(z, axis=0))
        res = tl.where(m_r == -float("inf"), m_r, res)
        if RMASK:
            tl.store(Out + r, res.to(Out.dtype.element_ty), mask=r < M)
        else:
            tl.store(Out + r, res.to(Out.dtype.element_ty))


@triton.jit
def _lse_dim0_s1(X, MidM, MidS, M, K, RB: tl.constexpr, BN: tl.constexpr, DEG: tl.constexpr):
    pk = tl.program_id(0)
    pr = tl.program_id(1)
    cols = pk * BN + tl.arange(0, BN)
    cmask = cols < K
    m = tl.full([BN], value=-float("inf"), dtype=tl.float32)
    s = tl.zeros([BN], dtype=tl.float32)
    for i in range(0, RB):
        r = pr * RB + i
        rm = r < M
        v = tl.load(X + r.to(tl.int64) * K + cols, mask=rm & cmask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, v)
        msafe = tl.where(m_new == -float("inf"), 0.0, m_new)
        s = s * _expf(m - msafe, DEG) + _expf(v - msafe, DEG)
        m = m_new
    tl.store(MidM + pr * K + cols, m, mask=cmask)
    tl.store(MidS + pr * K + cols, s, mask=cmask)


@triton.jit
def _lse_dim0_s2(MidM, MidS, Out, NB, K, BN: tl.constexpr, DEG: tl.constexpr):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    cmask = cols < K
    m = tl.full([BN], value=-float("inf"), dtype=tl.float32)
    s = tl.zeros([BN], dtype=tl.float32)
    for b in range(0, NB):
        mb = tl.load(MidM + b * K + cols, mask=cmask, other=-float("inf"))
        sb = tl.load(MidS + b * K + cols, mask=cmask, other=0.0)
        m_new = tl.maximum(m, mb)
        msafe = tl.where(m_new == -float("inf"), 0.0, m_new)
        s = s * _expf(m - msafe, DEG) + sb * _expf(mb - msafe, DEG)
        m = m_new
    res = m + tl.log(s)
    res = tl.where(m == -float("inf"), m, res)
    tl.store(Out + cols, res.to(Out.dtype.element_ty), mask=cmask)


@triton.jit
def _lse_mid(X, Out, B, C, BN: tl.constexpr, RB: tl.constexpr, DEG: tl.constexpr):
    """Reduce a middle dim. X is (A, B, C) with C contiguous; one program per
    (a, column block). Chunked online fold: a chunk max over RB rows, one
    rescale of the running sum, then one exp per element."""
    pa = tl.program_id(0)
    pc = tl.program_id(1)
    cols = pc * BN + tl.arange(0, BN)
    cmask = cols < C
    base = pa.to(tl.int64) * (B * C)
    m = tl.full([BN], value=-float("inf"), dtype=tl.float32)
    s = tl.zeros([BN], dtype=tl.float32)
    for b0 in range(0, B, RB):
        cm = tl.full([BN], value=-float("inf"), dtype=tl.float32)
        for i in tl.static_range(RB):
            b = b0 + i
            v = tl.load(X + base + b * C + cols, mask=cmask & (b < B), other=-float("inf")).to(tl.float32)
            cm = tl.maximum(cm, v)
        m_new = tl.maximum(m, cm)
        msafe = tl.where(m_new == -float("inf"), 0.0, m_new)
        s = s * _expf(m - msafe, DEG)
        acc = tl.zeros([BN], dtype=tl.float32)
        for i in tl.static_range(RB):
            b = b0 + i
            v = tl.load(X + base + b * C + cols, mask=cmask & (b < B), other=-float("inf")).to(tl.float32)
            acc += _expf(v - msafe, DEG)
        s += acc
        m = m_new
    res = m + tl.log(s)
    res = tl.where(m == -float("inf"), m, res)
    tl.store(Out + pa * C + cols, res.to(Out.dtype.element_ty), mask=cmask)


@triton.jit
def _lse_mid_u(X, Out, B, C, BN: tl.constexpr, RB: tl.constexpr, DEG: tl.constexpr):
    """wave22 unmasked mid-dim fold: requires C % BN == 0 and B % RB == 0."""
    pa = tl.program_id(0)
    pc = tl.program_id(1)
    cols = pc * BN + tl.arange(0, BN)
    base = pa.to(tl.int64) * (B * C)
    m = tl.full([BN], value=-float("inf"), dtype=tl.float32)
    s = tl.zeros([BN], dtype=tl.float32)
    for b0 in range(0, B, RB):
        cm = tl.full([BN], value=-float("inf"), dtype=tl.float32)
        for i in tl.static_range(RB):
            v = tl.load(X + base + (b0 + i) * C + cols).to(tl.float32)
            cm = tl.maximum(cm, v)
        m_new = tl.maximum(m, cm)
        msafe = tl.where(m_new == -float("inf"), 0.0, m_new)
        s = s * _expf(m - msafe, DEG)
        acc = tl.zeros([BN], dtype=tl.float32)
        for i in tl.static_range(RB):
            v = tl.load(X + base + (b0 + i) * C + cols).to(tl.float32)
            acc += _expf(v - msafe, DEG)
        s += acc
        m = m_new
    res = m + tl.log(s)
    res = tl.where(m == -float("inf"), m, res)
    tl.store(Out + pa * C + cols, res.to(Out.dtype.element_ty))


def logsumexp_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_LOGSUMEXP_FAST", "1") != "0"


def _poly_deg() -> int:
    if os.environ.get("FLAGGEMS_SPACEMIT_LSE_POLY", "1") == "0":
        return 0
    return exp_flag()


def logsumexp_mid_launch(x, out, A, B, C) -> bool:
    """True if the middle-dim fold (A outer x B reduced x C inner) was launched."""
    import torch

    if A <= 0 or B <= 1 or C < _MIN_N:
        return False
    f32 = x.dtype == torch.float32
    if f32:
        bn = _MID_BN_SMALL if C <= 128 else _MID_BN
    else:
        bn = _MID_BN_LARGE if C >= 2048 else _MID_BN
    # wave23: B=32 wants RB=32 (probe211: 64x32x1024 BN=256 RB=8 3.290 ->
    # RB=32 3.035 ms); B=16 wants 16, B=8 keeps 8.  B>=64 stays at 16: the
    # 32-row unroll only buys 2.5% on 8x128x64x64 and costs registers.
    rb = _MID_RB_BIG if B >= 64 else (32 if B >= 32 else (16 if B >= 16 else _MID_RB))
    if f32 and C % bn == 0 and B % rb == 0:
        _lse_mid_u[(A, C // bn)](x, out, B, C, BN=bn, RB=rb, DEG=_poly_deg())
        return True
    bn = min(bn, triton.next_power_of_2(C))
    rb = _MID_RB if B % _MID_RB == 0 else (4 if B % 4 == 0 else (2 if B % 2 == 0 else 1))
    _lse_mid[(A, triton.cdiv(C, bn))](x, out, B, C, BN=bn, RB=rb, DEG=_poly_deg())
    return True


def logsumexp_rows_launch(x, out, M, N) -> bool:
    """True if the (M rows x N cols) row logsumexp was launched."""
    import torch

    if M <= 0 or N < _MIN_N:
        return False
    # f16 loads with a 64-wide block fall into a scalar path on this backend
    # (probe196: 5.8-90 ms versus 0.71-10.5 ms at BLOCK=128/U=1), so only f32
    # gets the 64x4 unroll.
    if x.dtype == torch.float32:
        block, u = _ROWS_BLOCK, _ROWS_U
    else:
        block, u = _ROWS_BLOCK_F16, 1
    if N % (block * u) == 0 and M % _ROWS == 0:
        _lse_rows_u[(M // _ROWS,)](
            x, out, M, N, K=(_ROWS if M % _ROWS == 0 else 1), BLOCK=block, U=u, DEG=_poly_deg(),
            TAIL=False, RMASK=False,
        )
        return True
    block = _BLOCK_SMALL if N <= 1024 else _BLOCK_LARGE
    block = min(block, triton.next_power_of_2(N))
    _lse_rows[(triton.cdiv(M, _ROWS if M % _ROWS == 0 else 1),)](x, out, M, N, K=(_ROWS if M % _ROWS == 0 else 1), BLOCK_N=block, DEG=_poly_deg())
    return True


def logsumexp_dim0_launch(x, out, M, K) -> bool:
    """True if the dim=0 online-fold logsumexp (M rows x K cols) was launched."""
    import torch

    if M < _DIM0_MIN_M or K < _MIN_N:
        return False
    bn = _DIM0_BN_LARGE if K >= 2048 else _DIM0_BN_SMALL
    nrb = triton.cdiv(M, _DIM0_RB)
    ncb = triton.cdiv(K, bn)
    midm = torch.empty((nrb, K), dtype=torch.float32, device=x.device)
    mids = torch.empty((nrb, K), dtype=torch.float32, device=x.device)
    _lse_dim0_s1[(ncb, nrb)](x, midm, mids, M, K, RB=_DIM0_RB, BN=bn, DEG=_poly_deg())
    _lse_dim0_s2[(ncb,)](midm, mids, out, nrb, K, BN=bn, DEG=_poly_deg())
    return True
