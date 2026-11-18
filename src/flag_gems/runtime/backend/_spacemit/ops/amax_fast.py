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

"""wave16 (campaign 1018pm): multi-row amax/amin row kernels.

The generic amax/amin kernel keeps a (BLOCK_M, BLOCK_N) accumulator tile and
finishes with tl.max(axis=1). On spine-triton that form lowers
elementwise-scalar: 4.2-4.7 ns/elem, i.e. 0.05-0.23x torch across shapes.

A per-row full-vector reduce with K rows per program (the mean_fast /
softmax-v3 form: vector accumulate, single final reduce, no reduce result fed
back into elementwise ops) runs at 0.3-0.5 ns/elem:

    shape        generic   fast(K=16)   torch     fast vs generic
    512x512        2.61      0.154      0.184        17.0x
    2048x2048     17.25      1.250      1.676        13.8x
    512x4096       4.31      0.606      0.898         7.1x
    8192x512      38.90      1.555      2.638        25.0x
    4096x1024     20.72      1.343      1.438        15.4x

K=16 is optimal on every pow2 shape (K8 within 3%, K32/K64 worse).
Non-pow2 or wide N takes the masked loop kernel (vector accumulator over
BLOCK_N tiles): 256x1000 0.303 vs torch 0.184 (generic ~2.6 ms),
512x32768 6.58 vs torch 6.11.

NaN: tl.max/tl.min do not propagate NaN while torch.amax/amin do. The generic
gems kernels have the same limitation, so this is not a regression (recorded
as a known semantic gap, same family as argmin/argmax).

Kill switches: FLAGGEMS_SPACEMIT_AMAX_FAST=0 / FLAGGEMS_SPACEMIT_AMIN_FAST=0.
"""

import os

import triton
import triton.language as tl

_ROWS_POW2 = 16        # rows per program for the full-row kernels
_ROWS_LOOP = 8         # rows per program for the masked loop kernel
_POW2_MAX_N = 4096     # largest row length compiled as a full-row kernel
_LOOP_BLOCK = 2048     # tile width of the masked loop kernel


@triton.jit
def _rows_pow2(X, Out, N: tl.constexpr, K: tl.constexpr, IS_MAX: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, N)
    for i in tl.static_range(K):
        r = pid * K + i
        base = r.to(tl.int64) * N
        x = tl.load(X + base + cols)
        if IS_MAX:
            tl.store(Out + r, tl.max(x, axis=0))
        else:
            tl.store(Out + r, tl.min(x, axis=0))


@triton.jit
def _rows_loop(X, Out, N, K: tl.constexpr, BLOCK_N: tl.constexpr, IS_MAX: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    for i in tl.static_range(K):
        r = pid * K + i
        base = r.to(tl.int64) * N
        if IS_MAX:
            acc = tl.full([BLOCK_N], value=-float("inf"), dtype=tl.float32)
        else:
            acc = tl.full([BLOCK_N], value=float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            idx = off + cols
            m = idx < N
            if IS_MAX:
                v = tl.load(X + base + idx, mask=m, other=-float("inf")).to(tl.float32)
                acc = tl.maximum(acc, v)
            else:
                v = tl.load(X + base + idx, mask=m, other=float("inf")).to(tl.float32)
                acc = tl.minimum(acc, v)
        if IS_MAX:
            tl.store(Out + r, tl.max(acc, axis=0))
        else:
            tl.store(Out + r, tl.min(acc, axis=0))


def amax_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_AMAX_FAST", "1") != "0"


def amin_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_AMIN_FAST", "1") != "0"


def _launch(x, out, M, N, is_max: bool) -> bool:
    """True if the fast kernel handled the (M rows x N cols) reduce."""
    if M <= 0 or N <= 0:
        return False
    if N <= _POW2_MAX_N and (N & (N - 1)) == 0 and M % _ROWS_POW2 == 0:
        _rows_pow2[(M // _ROWS_POW2,)](x, out, N, K=_ROWS_POW2, IS_MAX=is_max)
        return True
    if M % _ROWS_LOOP == 0:
        block = min(_LOOP_BLOCK, triton.next_power_of_2(N))
        _rows_loop[(M // _ROWS_LOOP,)](x, out, N, K=_ROWS_LOOP, BLOCK_N=block, IS_MAX=is_max)
        return True
    return False


def amax_fast_launch(x, out, M, N) -> bool:
    return _launch(x, out, M, N, True)


def amin_fast_launch(x, out, M, N) -> bool:
    return _launch(x, out, M, N, False)
