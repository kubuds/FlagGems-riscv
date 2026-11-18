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

"""wave18/19: logsumexp row + dim=0 fast paths.

The generic logsumexp keeps (TILE_N, TILE_K) / [TILE_N] masks and finishes with
tl.sum(z * tl.exp(m - m_reduced), axis=0) - a reduce fed by computed values,
which spine-triton lowers scalar. Measured baseline (dim=1, f32):
6.92 / 104.29 / 52.21 / 108.02 ms at 512^2 / 2048^2 / 512x4096 / 8192x512
(0.45-0.48x torch).

The row kernel keeps the max and the exp-sum in narrow vector accumulators
(BLOCK_N lanes, one final reduce per row). probe161/169b:
f32 1.455/12.190/5.609/19.197 ms, f16 1.218/13.471/6.345/18.491 ms - 4.4-5.8x
faster than the generic path and 3.5-4.9x faster than torch (torch's f16
logsumexp runs 5.3-64.7 ms).

wave19 dim=0 carrier (probe170): online logsumexp fold per (row-block x
column-block), merged in stage 2 by the same max/rescale rule. generic ->
fast: 27.33->4.889 (2048^2), 11.76->2.709 (512x4096), 32.90->5.420 (4096x1024),
20.46->9.214 (8192x512); 4.2-9.6x faster than torch. Requires M >= 64 (an
8-row 3-D case regressed to 41.8 ms vs 27.6 generic, so small-M falls back).

Kill switch: FLAGGEMS_SPACEMIT_LOGSUMEXP_FAST=0.
"""

import os

import triton
import triton.language as tl

_ROWS = 4
_BLOCK_SMALL = 128   # N <= 1024
_BLOCK_LARGE = 256   # N > 1024
_MIN_N = 64
_DIM0_RB = 128
_DIM0_BN_SMALL = 256   # K < 2048
_DIM0_BN_LARGE = 512   # K >= 2048
_DIM0_MIN_M = 64


@triton.jit
def _lse_rows(X, Out, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
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
            z += tl.exp(v - m_r)
        res = m_r + tl.log(tl.sum(z, axis=0))
        # all -inf row: torch returns -inf (exp(-inf - -inf) is NaN)
        res = tl.where(m_r == -float("inf"), m_r, res)
        tl.store(Out + r, res.to(Out.dtype.element_ty), mask=rmask)


@triton.jit
def _lse_dim0_s1(X, MidM, MidS, M, K, RB: tl.constexpr, BN: tl.constexpr):
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
        s = s * tl.exp(m - msafe) + tl.exp(v - msafe)
        m = m_new
    tl.store(MidM + pr * K + cols, m, mask=cmask)
    tl.store(MidS + pr * K + cols, s, mask=cmask)


@triton.jit
def _lse_dim0_s2(MidM, MidS, Out, NB, K, BN: tl.constexpr):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    cmask = cols < K
    m = tl.full([BN], value=-float("inf"), dtype=tl.float32)
    s = tl.zeros([BN], dtype=tl.float32)
    for b in range(0, NB):
        mb = tl.load(MidM + b * K + cols, mask=cmask, other=-float("inf"))
        sb = tl.load(MidS + b * K + cols, mask=cmask, other=0.0)
        m_new = tl.maximum(m, mb)
        msafe = tl.where(m_new == -float("inf"), 0.0, m_new)
        s = s * tl.exp(m - msafe) + sb * tl.exp(mb - msafe)
        m = m_new
    res = m + tl.log(s)
    res = tl.where(m == -float("inf"), m, res)
    tl.store(Out + cols, res.to(Out.dtype.element_ty), mask=cmask)


def logsumexp_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_LOGSUMEXP_FAST", "1") != "0"


def logsumexp_rows_launch(x, out, M, N) -> bool:
    """True if the (M rows x N cols) row logsumexp was launched."""
    if M <= 0 or N < _MIN_N:
        return False
    block = _BLOCK_SMALL if N <= 1024 else _BLOCK_LARGE
    block = min(block, triton.next_power_of_2(N))
    _lse_rows[(triton.cdiv(M, _ROWS),)](x, out, M, N, K=_ROWS, BLOCK_N=block)
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
    _lse_dim0_s1[(ncb, nrb)](x, midm, mids, M, K, RB=_DIM0_RB, BN=bn)
    _lse_dim0_s2[(ncb,)](midm, mids, out, nrb, K, BN=bn)
    return True
