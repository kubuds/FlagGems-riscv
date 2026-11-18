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

"""wave17 (campaign 1019am): prod fast paths.

The generic prod keeps a (BLOCK_M, BLOCK_N) accumulator tile and finishes with
tl.reduce(axis=1, combine_fn=mul); on spine-triton that form lowers scalar
(18.1 ms at 2048x2048, 0.08x torch). Same root cause as amax/amin pre-wave16.

Key measurement (probe155/156): the scalar cost is proportional to the WIDTH of
the final reduce input, not to the data. A 64-lane vector accumulator fed by
BLOCK_N=64 chunks and one final 64-wide reduce costs ~0.7 ns/elem, while a
2048-wide reduce costs ~4.8 ns/elem:

    shape        generic   fast(B64,K8)   torch    fast vs generic
    512x512        1.42       0.45         0.13        3.2x
    2048x2048     18.06       2.90         1.59        6.2x
    512x4096       9.31       1.20         0.76        7.8x
    8192x512      22.87       6.14         1.29        3.7x

(Wider blocks B128/B256/B512/B2048 are 1.3x/1.5x/2.2x/6.7x slower; B32
degenerates, so N < 64 falls back to the generic kernel.)

Carriers:
  rows : K=8 rows per program, BLOCK_N=64 vector product + 64-wide final reduce
  flat : stage 1 per CHUNK elements with the same 64-lane carrier, stage 2 fold
  dim0 : stage 1 multiplies RB rows into a 64-lane column vector per row block,
         stage 2 folds row-block products (mid layout [nrb, K])

wave28: the row kernel was never nested either. Under nested dispatch the
row count wants to be much larger: 64 rows/program beats 16 by 1.5-1.9x
(probe285: 512^2 0.341 -> 0.221, 2048^2 2.039 -> 1.179, 512x4096 0.839 ->
0.492, 8192x512 4.530 -> 2.330, 4096x1024 2.896 -> 1.537 ms) and K128/K256
give it back (probe286). The same screen showed the rest of the reduce family
going the other way under nesting - amax/amin 0.57-0.96x, vector_norm
0.47-0.80x, var 0.48x, sum/mean 0.63-0.79x, argmax/argmin and layer_norm
neutral - so only prod was wired.

Kill switch: FLAGGEMS_SPACEMIT_PROD_FAST=0.
f32 only (f16 store in the reduce path aborts spine-opt; generic used there).
"""

import os

import torch
import triton
import triton.language as tl

from ._dispatch import nested_dispatch

_ROWS_LOOP = 8
_ROWS_NM = 64      # wave28: under nested dispatch 64 rows/program is the peak
#                    (probe285/286: 2048^2 1.18 vs 2.04 at K16; K128 regresses)
_LOOP_BLOCK = 64
_MIN_N = 64
_FLAT_CHUNK = 8192
_FLAT_BLOCK = 64
_DIM0_RB = 64
_DIM0_BN = 1024


@triton.jit
def _mul(a, b):
    return a * b


@triton.jit
def _prod_rows_loop(X, Out, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    for i in tl.static_range(K):
        r = pid * K + i
        rmask = r < M
        base = r.to(tl.int64) * N
        acc = tl.full([BLOCK_N], value=1.0, dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            idx = off + cols
            m = (idx < N) & rmask
            v = tl.load(X + base + idx, mask=m, other=1.0).to(tl.float32)
            acc = acc * v
        tl.store(Out + r, tl.reduce(acc, axis=0, combine_fn=_mul).to(Out.dtype.element_ty), mask=rmask)


@triton.jit
def _prod_rows_nm(X, Out, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr,
                  TO32: tl.constexpr):
    """No-mask row kernel (wave18): requires N % BLOCK_N == 0, M % K == 0.

    probe160: dropping the per-iteration row/column masks is worth 20-27% on
    the row-heavy shapes (8192x512 6.17 -> 4.85 ms, 4096x1024 4.05 -> 3.03).

    wave29: 16-bit inputs must be widened explicitly (TO32) - without the cast
    the multiply rounds to the input dtype each step and a 2048-long f16 row
    lands 15% off the f32 reference (probe289).  The f32 instantiation keeps
    the original body so its codegen is unchanged.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    for i in tl.static_range(K):
        r = pid * K + i
        base = r.to(tl.int64) * N
        acc = tl.full([BLOCK_N], value=1.0, dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            v = tl.load(X + base + off + cols)
            if TO32:
                v = v.to(tl.float32)
            acc = acc * v
        tl.store(Out + r, tl.reduce(acc, axis=0, combine_fn=_mul).to(Out.dtype.element_ty))


@triton.jit
def _prod_flat_stage1(X, Mid, M, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    acc = tl.full([BLOCK], value=1.0, dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK):
        idx = pid.to(tl.int64) * CHUNK + off + cols
        v = tl.load(X + idx, mask=idx < M, other=1.0).to(tl.float32)
        acc = acc * v
    tl.store(Mid + pid, tl.reduce(acc, axis=0, combine_fn=_mul))


@triton.jit
def _prod_flat_stage2(Mid, Out, NBLK, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    acc = tl.full([BLOCK], value=1.0, dtype=tl.float32)
    for off in range(0, NBLK, BLOCK):
        idx = off + cols
        v = tl.load(Mid + idx, mask=idx < NBLK, other=1.0)
        acc = acc * v
    tl.store(Out, tl.reduce(acc, axis=0, combine_fn=_mul).to(Out.dtype.element_ty))


@triton.jit
def _prod_dim0_stage1(X, Mid, M, K, RB: tl.constexpr, BN: tl.constexpr):
    pk = tl.program_id(0)
    pr = tl.program_id(1)
    cols = pk * BN + tl.arange(0, BN)
    cmask = cols < K
    acc = tl.full([BN], value=1.0, dtype=tl.float32)
    for i in range(0, RB):
        r = pr * RB + i
        m = (r < M) & cmask
        v = tl.load(X + r.to(tl.int64) * K + cols, mask=m, other=1.0).to(tl.float32)
        acc = acc * v
    tl.store(Mid + pr.to(tl.int64) * K + cols, acc, mask=cmask)


@triton.jit
def _prod_dim0_stage2(Mid, Out, NROW, K, BN: tl.constexpr):
    pk = tl.program_id(0)
    cols = pk * BN + tl.arange(0, BN)
    cmask = cols < K
    acc = tl.full([BN], value=1.0, dtype=tl.float32)
    for i in range(0, NROW):
        v = tl.load(Mid + i.to(tl.int64) * K + cols, mask=cmask, other=1.0)
        acc = acc * v
    tl.store(Out + cols, acc.to(Out.dtype.element_ty), mask=cmask)


def prod_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_PROD_FAST", "1") != "0"


def _pick_nm_rows(m: int) -> int:
    k = _ROWS_NM
    while k > 1 and m % k:
        k //= 2
    return k


def prod_rows_launch(x, out, M, N) -> bool:
    """True if the (M rows x N cols) row product was launched."""
    import torch

    if M <= 0 or N < _MIN_N:
        return False
    block = min(_LOOP_BLOCK, triton.next_power_of_2(N))
    if x.dtype != torch.float32 and block < 128 and N % 128 == 0:
        # 16-bit loads below 128 lanes fall into a scalar path (wave27 log_softmax
        # cliff; probe289 prod: 2048^2 B64 12.2 ms vs B128 2.7).
        block = 128
    if N % block == 0:
        # wave28: nested dispatch + 64 rows/program: 512^2 0.341 -> 0.221,
        # 2048^2 2.039 -> 1.179, 8192x512 4.530 -> 2.330 ms (probe285).
        k = _pick_nm_rows(M)
        with nested_dispatch(M * N):
            _prod_rows_nm[(M // k,)](x, out, M, N, K=k, BLOCK_N=block,
                                     TO32=(x.dtype != torch.float32))
        return True
    # The masked fallback stays on the default launcher: nesting it segfaults
    # spine-opt at 1001x1000 (wave28 probe, reproducible), and tail shapes are
    # not worth the risk.
    _prod_rows_loop[(triton.cdiv(M, _ROWS_LOOP),)](
        x, out, M, N, K=_ROWS_LOOP, BLOCK_N=block
    )
    return True


def prod_flat_launch(x, out, numel) -> bool:
    if numel < _MIN_N:
        return False
    import torch

    block = _FLAT_BLOCK
    if x.dtype != torch.float32:
        block = 128  # 16-bit loads below 128 lanes scalarize (probe289)
    nblk = triton.cdiv(numel, _FLAT_CHUNK)
    mid = torch.empty((nblk,), dtype=torch.float32, device=x.device)
    _prod_flat_stage1[(nblk,)](x, mid, numel, CHUNK=_FLAT_CHUNK, BLOCK=block)
    _prod_flat_stage2[(1,)](mid, out, nblk, BLOCK=max(32, triton.next_power_of_2(min(nblk, 1024))))
    return True


def prod_dim0_launch(x, out, M, K) -> bool:
    if M <= 0 or K < _MIN_N:
        return False
    bn = min(_DIM0_BN, triton.next_power_of_2(K))
    nrb = triton.cdiv(M, _DIM0_RB)
    mid = torch.empty((nrb, K), dtype=torch.float32, device=x.device)
    _prod_dim0_stage1[(triton.cdiv(K, bn), nrb)](x, mid, M, K, RB=_DIM0_RB, BN=bn)
    _prod_dim0_stage2[(triton.cdiv(K, bn),)](mid, out, nrb, K, BN=bn)
    return True
