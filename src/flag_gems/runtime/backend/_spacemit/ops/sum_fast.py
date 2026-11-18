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

"""wave6 (campaign 1015am): sum-family fast paths (spacemit backend).

Two fixes, both ported from the wave5 launch-amortization playbook:

1) flat sum: the legacy sqrt-rule stage-1 BLOCK lands at 1024 for 1M elems
   (scalar-lowered side of the tl.sum length cliff). Raising stage-1 BLOCK
   fixes it. Same-session f32 (probe59/60, interleaved re-check):
       1M flat: 1.85 -> 0.57 ms  (torch 0.32)   [BLOCK 16384]
   Rule: BLOCK = min(16384, next_pow2(max(4096, cdiv(M, 64)))) clamped to
   next_pow2(M).

2) sum_dim (dim = last, K == 1): K rows/program with a plain per-row
   load + tl.sum (same structure as mean_fast). Same-session off->on:
       (8192,512)  8.75 -> 1.79  (4.98x)   vs torch 1.28x
       (4096,1024) 8.13 -> 1.51  (5.47x)   vs torch 1.04x
       (2048,2048) 2.26 -> 1.40  (1.62x)   vs torch 1.17x
       (1024,4096) 1.97 -> 1.40  (1.41x)   vs torch 1.15x
       (512,32768) 7.34 -> 4.58  (1.60x)   vs torch 1.22x (chunked 8192x4)

   Shapes outside the table, non-divisible M and non-f32 inputs stay on the
   legacy route (this module never changes semantics, only speed).

Kill switch: FLAGGEMS_SPACEMIT_SUM_FAST=0 (both paths).
"""

import os

import torch
import triton
import triton.language as tl

from ._dispatch import nested_dispatch


def sum_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_SUM_FAST", "1") != "0"


# ---------------- flat two-stage ----------------
@triton.jit
def _sum_flat_k1(inp, mid, M, BLOCK: tl.constexpr, IS16: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(inp + off, mask=off < M, other=0.0)
    if IS16:
        v = v.to(tl.float32)
    tl.store(mid + pid, tl.sum(v, axis=0))


@triton.jit
def _sum_flat_k2(mid, out, MIDN, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    v = tl.load(mid + off, mask=off < MIDN, other=0.0)
    tl.store(out, tl.sum(v, axis=0))


@triton.jit
def _sum_flat_single(inp, out, M, BLOCK: tl.constexpr, IS16: tl.constexpr):
    # wave8: midn == 1 fast case (tiny/small M) - one launch, extent >= 32.
    off = tl.arange(0, BLOCK)
    v = tl.load(inp + off, mask=off < M, other=0.0)
    if IS16:
        v = v.to(tl.float32)
    tl.store(out, tl.sum(v, axis=0))


def _flat_block(M: int) -> int:
    b = triton.next_power_of_2(max(4096, -(-M // 64)))
    if b > 16384:
        b = 16384
    p2 = triton.next_power_of_2(M)
    if b > p2:
        b = p2
    if b < 32:
        b = 32  # wave8: tl.sum reduce extent < 32 lowers to NaN (spine backend)
    return b


def sum_flat_launch(x, out, M):
    """Two-stage flat sum into out (scalar f32 tensor). x contiguous, f32/16-bit.

    wave8: covers ALL M now (small numels used to fall to the generic
    sqrt-rule kernel whose stage-2 extent is <= 16 for M <= ~512 and for
    M == 65536 exactly -> tl.sum extent<32 NaN). Stage-2 reduce extent is
    floored at 32 and midn==1 short-circuits to a single launch.

    wave30: 16-bit inputs take the chunk-accumulate carrier (_sum_flat_k1_16);
    the f32 path below is byte-for-byte the wave8 one.
    """
    if x.dtype != torch.float32:
        _sum_flat_16(x, out, M)
        return
    block = _flat_block(M)
    midn = triton.cdiv(M, block)
    is16 = x.dtype != torch.float32
    if midn == 1:
        _sum_flat_single[(1,)](x, out, M, block, is16)
        return
    mid = torch.empty(midn, dtype=torch.float32, device=x.device)
    _sum_flat_k1[(midn,)](x, mid, M, block, is16)
    blk2 = triton.next_power_of_2(midn)
    if blk2 < 32:
        blk2 = 32  # wave8: extent<32 NaN floor (covers the old M=65536 edge)
    _sum_flat_k2[(1,)](mid, out, midn, blk2)


# ---------------- wave30: 16-bit carriers for sum / mean ----------------
# probe297/298: a 16-bit row sum built as "tl.load + .to(f32) + tl.sum(N)" runs
# 7-9x slower than the same shapes on f32 (2048^2 f16 13.9 ms in the shipped
# _sum_dim_mmk2048). Two things fix it: the accumulate loop must be a runtime
# `range(0, N, BN)` (a fully unrolled tl.static_range spills), and 8-16 rows
# per program amortize the per-row setup. Same structure as norm_fast._l2_rows,
# which is why the norm carrier was already fast on 16-bit.
_SUM16_BN = 128
_SUM16_CHUNK = 8192


@triton.jit
def _sum_rows_16(X, Out, M, N, K: tl.constexpr, BN: tl.constexpr, MASKED: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BN)
    for i in tl.static_range(K):
        r = pid * K + i
        rmask = r < M
        base = r.to(tl.int64) * N
        acc = tl.zeros([BN], dtype=tl.float32)
        for off in range(0, N, BN):
            idx = off + cols
            if MASKED:
                acc += tl.load(X + base + idx, mask=(idx < N) & rmask, other=0.0).to(tl.float32)
            else:
                acc += tl.load(X + base + idx).to(tl.float32)
        if MASKED:
            tl.store(Out + r, tl.sum(acc, axis=0), mask=rmask)
        else:
            tl.store(Out + r, tl.sum(acc, axis=0))


def _rows16_k(M: int) -> int:
    for cand in (16, 8, 4, 2):
        if M % cand == 0:
            return cand
    return 1


def sum_rows16_launch(x, out_flat, M, N) -> bool:
    """16-bit row sum into an f32 buffer (out_flat is f32, shape (M,))."""
    if x.dtype == torch.float32 or M <= 0 or N < 64:
        return False
    k = _rows16_k(M)
    bn = min(_SUM16_BN, triton.next_power_of_2(N))
    # probe300/301: the per-element mask costs 25-45% on these kernels (the
    # divisible shapes were never measured with it before); only pay it when
    # the row block or the row count actually exceeds the tensor.
    masked = (N % bn) != 0 or (M % k) != 0
    _sum_rows_16[(triton.cdiv(M, k),)](x, out_flat, M, N, K=k, BN=bn, MASKED=masked)
    return True


@triton.jit
def _sum_flat_k1_16(inp, mid, M, CHUNK: tl.constexpr, BN: tl.constexpr, MASKED: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    cols = tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for off in range(0, CHUNK, BN):
        idx = base + off + cols
        if MASKED:
            acc += tl.load(inp + idx, mask=idx < M, other=0.0).to(tl.float32)
        else:
            acc += tl.load(inp + idx).to(tl.float32)
    tl.store(mid + pid, tl.sum(acc, axis=0))


def _sum_flat_16(x, out, M):
    chunk = _SUM16_CHUNK
    midn = max(1, triton.cdiv(M, chunk))
    mid = torch.empty(midn, dtype=torch.float32, device=x.device)
    _sum_flat_k1_16[(midn,)](x, mid, M, CHUNK=chunk, BN=_SUM16_BN * 2,
                             MASKED=(M % chunk) != 0)
    blk2 = max(32, triton.next_power_of_2(midn))
    _sum_flat_k2[(1,)](mid, out, midn, blk2)


# ---------------- sum_dim multi-row (plain per-row reduce) ----------------
@triton.jit
def _sum_dim_mmk512(X, Out, N, K: tl.constexpr):
    p = tl.program_id(0)
    for i in tl.static_range(K):
        r = p * K + i
        xv = tl.load(X + r * N + tl.arange(0, 512)).to(tl.float32)
        tl.store(Out + r, tl.sum(xv, axis=0))


@triton.jit
def _sum_dim_mmk1024(X, Out, N, K: tl.constexpr):
    p = tl.program_id(0)
    for i in tl.static_range(K):
        r = p * K + i
        xv = tl.load(X + r * N + tl.arange(0, 1024)).to(tl.float32)
        tl.store(Out + r, tl.sum(xv, axis=0))


@triton.jit
def _sum_dim_mmk2048(X, Out, N, K: tl.constexpr):
    p = tl.program_id(0)
    for i in tl.static_range(K):
        r = p * K + i
        xv = tl.load(X + r * N + tl.arange(0, 2048)).to(tl.float32)
        tl.store(Out + r, tl.sum(xv, axis=0))


@triton.jit
def _sum_dim_mmk_chunked(
    X, Out, N, K: tl.constexpr, BLOCK: tl.constexpr, NCHUNK: tl.constexpr
):
    # per-row chunked accumulation (wide rows): runtime-N form only.
    p = tl.program_id(0)
    for i in tl.static_range(K):
        r = p * K + i
        acc = 0.0
        for c in tl.static_range(NCHUNK):
            offs = c * BLOCK + tl.arange(0, BLOCK)
            xv = tl.load(X + r * N + offs).to(tl.float32)
            acc += tl.sum(xv, axis=0)
        tl.store(Out + r, acc)


# (kernel, K) per row width; all shapes verified same-session in probe59/60.
_SUM_MMK = {
    512: (_sum_dim_mmk512, 16),
    1024: (_sum_dim_mmk1024, 16),
    2048: (_sum_dim_mmk2048, 8),
}


def sum_dim_mmk_launch(x, out_flat, M, N) -> bool:
    """Launch the multi-row kernel if (N, M) fit; return True if handled."""
    if x.dtype != torch.float32:
        # wave30: one 16-bit carrier covers every N (the per-width table below
        # is tuned for the f32 tl.sum lowering and loses 7-9x on 16-bit).
        return sum_rows16_launch(x, out_flat, M, N)
    ent = _SUM_MMK.get(N)
    if ent is not None:
        kern, k = ent
        if M % k:
            return False
        kern[(M // k,)](x, out_flat, N, k)
        return True
    if N == 4096:
        k = 4
        if M % k:
            return False
        _sum_dim_mmk_chunked[(M // k,)](x, out_flat, N, k, 2048, 2)
        return True
    if N == 32768:
        k = 4
        if M % k:
            return False
        # wave12: the chunked-32768 route wins under nested dispatch
        # (probe104 same-session 1.17x; mirrors the mean 32768 route).
        if os.environ.get("FLAGGEMS_SPACEMIT_SUM_NEST", "1") != "0":
            with nested_dispatch(M * N):
                _sum_dim_mmk_chunked[(M // k,)](x, out_flat, N, k, 8192, 4)
        else:
            _sum_dim_mmk_chunked[(M // k,)](x, out_flat, N, k, 8192, 4)
        return True
    return False


# ---------------- sum_dim dim0 (non-inner) row-fold (wave8) ----------------
# Grid: (cdiv(N, BN), cdiv(M, RB)); stage 1 accumulates RB rows per program
# into row-block partials (NO tl.sum - pure load+add, immune to the
# elementwise-feeding-tl.sum cliff; 2D-load+tl.sum(axis=0) measured 5-13x
# slower in probe78). Stage 2 combines the PRB partial rows (also pure
# accumulate). Same-session f32 (probe78/79, ms min-of-9):
#   (2048,2048)  12.46 generic -> 1.27  (torch 6.69)
#   (8192,512)    4.41         -> 1.96  (torch 3.83)
#   (512,32768)  14.33         -> 3.33  (torch 26.27)
#   (1024,4096)   6.12         -> 1.37  (torch 7.25)
#   (4096,1024)  17.32         -> 1.98  (torch 6.24)
@triton.jit
def _sum_dim0_fold(X, Mid, M, N, RB: tl.constexpr, BN: tl.constexpr):
    pc = tl.program_id(0)
    pr = tl.program_id(1)
    cols = pc * BN + tl.arange(0, BN)
    cmask = cols < N
    r_end = tl.minimum((pr + 1) * RB, M)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for r in range(pr * RB, r_end):
        acc += tl.load(X + r * N + cols, mask=cmask, other=0.0).to(tl.float32)
    tl.store(Mid + pr * N + cols, acc, mask=cmask)


@triton.jit
def _sum_dim0_comb(Mid, Out, PRB, N, BN: tl.constexpr):
    pc = tl.program_id(0)
    cols = pc * BN + tl.arange(0, BN)
    cmask = cols < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for i in range(0, PRB):
        acc += tl.load(Mid + i * N + cols, mask=cmask, other=0.0)
    tl.store(Out + cols, acc, mask=cmask)


# (M, N) -> (RB, BN); probed same-session (probe79 refinement).
_DIM0_ROUTES = {
    (2048, 2048): (256, 512),
    (8192, 512): (512, 512),
    (512, 32768): (128, 8192),
    (1024, 4096): (128, 2048),
    (4096, 1024): (256, 1024),
}


def sum_dim0_launch(x2d, out_flat, M, N) -> bool:
    """Row-fold dim0 reduce. x2d = (M, N) f32 contiguous view; out_flat = (N,).

    Any (M, N >= 1) is handled (masked edges); returns False only for empty."""
    if M <= 0 or N <= 0:
        return False
    rb, bn = _DIM0_ROUTES.get((M, N), (None, None))
    if rb is None:
        rb = 128 if M >= 4096 else 64
        bn = min(1024, triton.next_power_of_2(max(N, 32)))
    prb = triton.cdiv(M, rb)
    mid = torch.empty((prb, N), dtype=torch.float32, device=x2d.device)
    _sum_dim0_fold[(triton.cdiv(N, bn), prb)](x2d, mid, M, N, rb, bn)
    _sum_dim0_comb[(triton.cdiv(N, bn),)](mid, out_flat, prb, N, bn)
    return True
