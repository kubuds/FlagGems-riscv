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

"""wave17 (campaign 1019am): log_softmax last-dim row kernel.

The generic log_softmax inner kernel keeps a (BLOCK_M, BLOCK_N) online
max/sum tile and finishes with tl.max/tl.sum(axis=1); on spine-triton this
scalarizes: 9.1 / 100.6 / 378.4 ms at 512^2 / 2048^2 / 512x32768 (0.22-0.41x
torch).

Carrier: K=8 rows per program, three vector passes per row over BLOCK_N=256
chunks (max pass -> sum-exp pass -> store pass). The reduce inputs are plain
vector accumulators of width 256, never a computed expression; per probe156
the reduce width dominates the scalar cost (B256 is within 2% of the best,
B1024 4% worse, B4096 far worse).

Measured (same-session probe156): 1.31 / 13.21 / 42.88 ms -> 6.9x / 7.6x /
8.8x over the generic kernel and 2.0x / 3.1x / 3.8x over native torch.
maxabs err <= 2e-6 (512x32768).

Kill switch: FLAGGEMS_SPACEMIT_LOGSOFTMAX_FAST=0.
"""

import os

import torch
import triton
import triton.language as tl

_ROWS = 8
_BLOCK_MAX = 256


@triton.jit
def _logsm_rows(Out, X, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
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
        lz = tl.log(tl.sum(z, axis=0))
        for off in range(0, N, BLOCK_N):
            idx = off + cols
            mk = (idx < N) & rmask
            v = tl.load(X + base + idx, mask=mk, other=0.0).to(tl.float32)
            tl.store(Out + base + idx, (v - m_r - lz).to(Out.dtype.element_ty), mask=mk)


def logsm_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_LOGSOFTMAX_FAST", "1") != "0"


def logsm_launch(x, out, M, N) -> bool:
    if M <= 0 or N < 64:
        return False
    block = min(_BLOCK_MAX, triton.next_power_of_2(N))
    _logsm_rows[(triton.cdiv(M, _ROWS),)](out, x, M, N, K=_ROWS, BLOCK_N=block)
    return True
