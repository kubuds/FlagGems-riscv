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

"""wave17/27: log_softmax last-dim row kernel.

wave17 replaced the generic (BLOCK_M, BLOCK_N) online tile - whose axis=1
reductions scalarise on spine-triton (9.1 / 100.6 / 378.4 ms at 512^2 / 2048^2 /
512x32768) - with K rows per program, 256-wide chunks, three passes.

wave27 rewrote the carrier around what the probes actually showed (f32):

1. A whole-row-resident block - the fastest softmax shape here - is a trap for
   log_softmax.  Keeping the loaded row live across both reductions and using
   it afterwards costs about 1.4 ms per 256 KB extra: at 256^2, copy 0.091 ms,
   max 0.111, exp+sum+store 0.214, but the same kernel storing `x - m - 1` is
   1.609 (1.618 with a real tl.log, so libm is innocent; the spill is the whole
   story).  Softmax stores exp values, log_softmax needs the input, so the store
   pass re-reads the row.  Resident rows measured 0.461 at 128^2, 1.62 at
   256^2 and 24.5 ms at 1024^2 - never again.
2. Dynamic masks on every access cost 1.5-2x when the shape is exact.  The
   kernel is unmasked whenever n_cols % BN == 0 and only masks the tail chunk
   otherwise: 256^2 0.560 -> 0.281/0.297, 512^2 1.260 -> 0.761, 1024^2
   3.800 -> 2.543, 2048^2 12.099 -> 9.301 (probe270/271/272).
3. K is picked from (8, 4, 2, 1) so that M % K == 0 and the grid is exact -
   no row mask, ever.  Blocks are 64 wide at 64^2 (0.097 vs 0.171 for one
   resident row) and 256 wide for the big rows.

f16/bf16 used to fall through to the generic kernel (26 ms vs 2.6 ms at
1024^2, 1.0 ms vs 0.2 at 64^2); they now run the same f32-accumulating kernel
and cast on store.

wave28: like logsumexp, this row kernel had never been nested - the serialized
1-D launcher was the ceiling. Nested + more rows per program (probe277-280,
kernel-only f32): 256^2 0.29 -> 0.19, 512^2 0.40 -> 0.37, 1024^2 2.63 -> 0.96,
2048^2 9.48 -> 3.52 ms (R16 at N=2048, R32 below - wide rows lose a little at
32). f16 follows the same rule: 256^2 0.19, 1024^2 0.89, 2048^2 2.85 ms.

Kill switch: FLAGGEMS_SPACEMIT_LOGSOFTMAX_FAST=0.
"""

import os

import torch
import triton
import triton.language as tl

from ._dispatch import nested_dispatch
from .fastmath import exp_fast as _expf
from .fastmath import exp_flag

_ROWS_SMALL = 8      # M < 256: 4-8 programs are all the grid can carry
_ROWS_LARGE = 32     # wave28 (probe279: 1024^2 1.33 -> 0.96 against R8)
_ROWS_WIDE_N = 16    # N >= 2048 (probe280: 2048^2 3.79 at R32 vs 3.52 at R16)
_BLOCK_MAX = 256


@triton.jit
def _logsm_rows(Out, X, n_cols, K: tl.constexpr, BN: tl.constexpr, TAIL: tl.constexpr, FAST: tl.constexpr):
    """K rows per program, three vector passes per row.

    TAIL=False means n_cols % BN == 0 and every access is unmasked."""
    pid = tl.program_id(0)
    cols = tl.arange(0, BN)
    for i in range(K):
        base = (pid * K + i).to(tl.int64) * n_cols
        m = tl.full([BN], value=-float("inf"), dtype=tl.float32)
        if TAIL:
            for o in range(0, n_cols, BN):
                idx = o + cols
                mk = idx < n_cols
                m = tl.maximum(
                    m, tl.load(X + base + idx, mask=mk, other=-float("inf")).to(tl.float32)
                )
            mr = tl.max(m, axis=0)
            z = tl.zeros([BN], dtype=tl.float32)
            for o in range(0, n_cols, BN):
                idx = o + cols
                mk = idx < n_cols
                z += _expf(
                    tl.load(X + base + idx, mask=mk, other=-float("inf")).to(tl.float32) - mr,
                    FAST,
                )
            lz = tl.log(tl.sum(z, axis=0))
            for o in range(0, n_cols, BN):
                idx = o + cols
                mk = idx < n_cols
                v = tl.load(X + base + idx, mask=mk, other=0.0).to(tl.float32)
                tl.store(
                    Out + base + idx, (v - mr - lz).to(Out.dtype.element_ty), mask=mk
                )
        else:
            for o in range(0, n_cols, BN):
                m = tl.maximum(m, tl.load(X + base + o + cols).to(tl.float32))
            mr = tl.max(m, axis=0)
            z = tl.zeros([BN], dtype=tl.float32)
            for o in range(0, n_cols, BN):
                z += _expf(tl.load(X + base + o + cols).to(tl.float32) - mr, FAST)
            lz = tl.log(tl.sum(z, axis=0))
            for o in range(0, n_cols, BN):
                v = tl.load(X + base + o + cols).to(tl.float32)
                tl.store(Out + base + o + cols, (v - mr - lz).to(Out.dtype.element_ty))


def logsm_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_LOGSOFTMAX_FAST", "1") != "0"


def _row_split(M: int, N: int) -> int:
    """wave28: rows per program under nested dispatch. The nested launcher
    splits the grid across streams, so fewer/bigger programs win; 32 rows is
    the pick up to N=1024, 16 above (probe279/280). M < 256 keeps 8 so the
    grid still has work for every stream."""
    if M < 256:
        k = _ROWS_SMALL
    else:
        k = _ROWS_LARGE if N <= 1024 else _ROWS_WIDE_N
    while k > 1 and M % k:
        k //= 2
    return k


def _block_split(N: int, esize: int) -> int:
    # Exact-shape widths measured (probe270/271/274): f32 wants 64 at 256 cols,
    # 128 at 512, 256 above.  16-bit never goes below 128: a 64-wide f16 chunk
    # is 128 B per load and falls off a cliff (256^2 f16 1.727 vs 0.289 ms,
    # 128^2 0.585 vs ~0.25, tail shapes 0.89 vs 0.20).  Tail shapes keep 64 on
    # f32 (300 cols: 0.190 vs 0.205) and 128 on 16-bit (0.200 vs 0.891).
    if N < 128:
        return 64
    if esize < 4:
        target = _BLOCK_MAX if N >= _BLOCK_MAX else 128
    else:
        target = 64 if N <= 256 else (128 if N <= 512 else _BLOCK_MAX)
    for bn in (target, 128, 64):
        if N % bn == 0:
            return bn
    return 128 if esize < 4 else 64


def logsm_launch(x, out, M, N) -> bool:
    if M <= 0 or N < 64:
        return False
    k = _row_split(M, N)
    bn = _block_split(N, x.element_size())
    with nested_dispatch(M * N):
        _logsm_rows[(M // k,)](
            out, x, N, K=k, BN=bn, TAIL=(N % bn) != 0, FAST=exp_flag()
        )
    return True
