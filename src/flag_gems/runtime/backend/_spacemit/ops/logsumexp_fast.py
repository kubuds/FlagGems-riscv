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

"""wave18 (campaign 1019pm): logsumexp row fast path.

The generic logsumexp keeps (TILE_N, TILE_K) / [TILE_N] masks and finishes with
tl.sum(z * tl.exp(m - m_reduced), axis=0) - a reduce fed by computed values,
which spine-triton lowers scalar. Measured baseline (dim=1, f32):
6.92 / 104.29 / 52.21 / 108.02 ms at 512^2 / 2048^2 / 512x4096 / 8192x512
(0.45-0.48x torch).

This kernel keeps the max and the exp-sum in narrow vector accumulators
(BLOCK_N lanes, one final reduce per row). probe161:
1.08 / 11.22 / 5.20 / 16.01 ms - 6.4x-10.0x the generic path and 2.9x-4.6x
faster than torch.

Kill switch: FLAGGEMS_SPACEMIT_LOGSUMEXP_FAST=0. f32 + contiguous + trailing
reduce dim only; everything else falls back to the generic implementation.
"""

import os

import triton
import triton.language as tl

_ROWS = 4
_BLOCK_SMALL = 128   # N <= 1024
_BLOCK_LARGE = 256   # N > 1024
_MIN_N = 64


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
