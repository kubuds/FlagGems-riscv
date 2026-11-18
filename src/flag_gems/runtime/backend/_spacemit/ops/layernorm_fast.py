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

"""wave4 (campaign 1014am): layer_norm forward direct kernels ("hybrid fz3").

Structure (probe36-43, same-session f32):
- s1: ONE pure full-row load-sum (BLOCK=next_pow2(N) <= 8192). Measured
  probe41: pure load-sum 1.80 ms @2048^2 (acc-chain loop form costs 2.65).
- s2: mul-chain accumulator loop, single chain, BLOCK=512 (probe41 D/J:
  separate chains beat a merged two-chain loop by ~20-35%; 512-chunk beats
  2048-chunk).
- apply: separate loop, BLOCK=2048 (or next_pow2(N)).
- var = E[x2] - m*m (f32 throughout; accuracy vs torch dy<=2.9e-6 @2048^2).
Rejected structures: unrolled acc without a loop (probe40: 48 ms @2048^2 —
catastrophic codegen); direct tl.sum(x*x) (probe40: 23 ms); tl.join+huge
reduce (52 ms); padded BLOCK>N sum (no better rate, probe40); 2D multiline
(f3, 16.9 ms); register-resident single-pass (f1, 14.98 ms); app2d (7.5 ms).
Fused totals vs official baseline: 2048^2 15.47->~5.1-5.7 (2.7-3.0x);
512x2048 4.21->~1.31-1.44 (2.9-3.2x); 512^2 3.07->~0.53-0.56 (5.5-5.8x).

wave11 (campaign 1016night): multirow K + nested dispatch. K rows/program via
tl.static_range(K) (K=16 when N<=1024 else 8, divisibility fallback chain
K->K/2->...->1) launched under nested_dispatch(M*N) when K>=4. Same-session f32
min-of-9 (probe98/99b): 512^2 0.684 -> 0.343/0.383 (K16/K8 nest; torch 0.310);
2048^2 5.075 -> 2.76-2.92 (K8 nest; torch 4.74 -> 1.7x); 512x2048 1.467 ->
0.737-0.750 (K8 nest; torch 1.52 -> 2.0x). Backward dx gains the same pattern
(12.0 -> 4.85 @2048^2 K8 nest).
Kill switches: FLAGGEMS_SPACEMIT_LN_FAST=0, or NESTED=0 (two-pass A/B).
"""

import os

import torch
import triton
import triton.language as tl

from ._dispatch import nested_dispatch

_LN_MIN_N = 1 << 15
# wave35 probe357: the 16-bit carrier wins from 4096 elements up - official
# core shape (4,16,64,4) has numel 16384 and runs 0.472 ms on the common
# kernel against 0.107 at K2 (4.4x) - while f32 at the same sizes loses
# (M4/N4096 0.079 common vs 0.106 carrier, M1 0.042 vs 0.098), so the floor
# is dtype aware.
_LN_MIN_N_16 = 1 << 12
# wave36 probe366: the backward floor was _LN_MIN_N (32768) for every dtype,
# but (4,16,64,4) is an official core shape with 16384 elements and the common
# backward is pathological there - 99.6 ms f16 / 84.4 f32 against 1.11 / 1.24
# on the carrier (89x / 68x), with dx/dw/db within 3.2e-3 (f16) and 3.5e-6
# (f32) of a float64 oracle.  f16 (16,16,128) is 13.8x and (16,16,8,48) 32x.
_LNB_MIN_N = 1 << 12
_LN_MAX_N = 8192
# wave35 probe362 (flag_gems.use_gems A/B): the f32 fast kernel loses to the
# 16-bit carrier from 6144 columns on (M16/N6144 fast 0.581 vs carrier 0.379
# at N=11264, and 0.94x at 6144), so f32 switches to the carrier above 4096.
_F32_FAST_MAX_N = 4096
# 16-bit carrier is a 128-lane streaming loop with a masked tail, so it does
# not need the f32 block-size ceiling; the tests exercise N=40999.
# wave34 probe346: (16,8,128,128) normalizes over [8,128,128] = 131072
# columns; the generic f16 kernel takes 29.4 ms there, the carrier 2.22 ms
# (13x), so the ceiling moves to 262144.
# wave35 probe356/357: the last official core shape (20,6,65536) is 393216
# columns; the carrier runs it in 8.05 ms f16 (common kernel 8.5-9.4 ms), so
# the ceiling moves to 1M.  f32 takes the split-N path below instead.
_LN16_MAX_N = 1 << 20
# f32 above _LN_MAX_N also uses the streaming carrier (probe346: 5.14 -> 2.58
# ms at (16,8,128,128) f32) because the f32 fast kernel would allocate a
# next_power_of_2(n) accumulator.
_SB2 = 512
# wave15 probe133/134: AB=2048 loses 31% at 2048^2 (4.21 -> 2.90) and 28% at
# 512x2048 (1.12 -> 0.80); 1024 and 512 tie within 1%.
_AB = 1024
# per-shape SB2: n>=1024 with a small element count prefers 256 lanes
# (512x2048 0.746 vs 0.802), the 2048^2 class prefers 512 (2.905 vs 2.953).
_SB2_SMALL_TOTAL = 2_000_000


def ln_fast_enabled() -> bool:
    return (
        os.environ.get("FLAGGEMS_SPACEMIT_LN_FAST", "1") != "0"
        and os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") != "0"
    )


def _prod(seq) -> int:
    p = 1
    for s in seq:
        p *= int(s)
    return p


def ln_fast_ok(input, normalized_shape, weight, bias) -> bool:
    if not isinstance(input, torch.Tensor):
        return False
    # f32 only: f16 chain measured slower than the common path (26.7 vs 15.3 ms
    # @2048^2, probe45) — the f16 load/upcast lowering is the suspect.
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if not input.is_contiguous():
        return False
    numel = input.numel()
    try:
        n = _prod(normalized_shape)
    except (TypeError, ValueError):
        return False
    if n <= 0 or numel % n:
        return False
    m = numel // n
    # wave35 probe362 (flag_gems.use_gems A/B): fast on vs off on the same
    # build, 3 alternating passes - f16 (4,16,64,4) 1.22x, (16,16,128) 1.13x,
    # (20,6,65536) 1.90x; f16 (16,16,8,48)/(16,16,8,88) are ties within the
    # run-to-run spread (0.98/0.99), and f32 (16,16,8,48) prefers the carrier
    # over the old f32 fast kernel (0.94x), hence _F32_FAST_MAX_N.
    # f32 keeps the old numel floor: the fast kernel loses when a whole row
    # fits in one 4096-element block and the op is dispatch bound.
    if input.dtype == torch.float32:
        if n <= 4096 and numel < (1 << 16):
            return False
    floor = _LN_MIN_N if input.dtype == torch.float32 else _LN_MIN_N_16
    if numel < floor:
        return False
    # wave33: the 16-bit carrier is no longer capped at _LN_MAX_N - the
    # aligned kernel covers any N up to 65536 (official bench (16,8,128,128)
    # is N=16384) and the masked variant covers the rest.
    # wave34: the official benchmark's 4-D shape (16,8,128,128) normalizes over
    # [8,128,128] = 131072 columns, above the old 65536 ceiling; f32 above
    # _LN_MAX_N now takes the same streaming carrier.
    nmax = _LN16_MAX_N
    if n > nmax:
        return False
    if input.dim() < 1:
        return False
    for t in (weight, bias):
        if t is not None:
            if not isinstance(t, torch.Tensor) or t.numel() != n:
                return False
            if not t.is_contiguous():
                return False
    return True


@triton.jit
def _ln_fwd_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N,
    eps,
    K: tl.constexpr,
    SB1: tl.constexpr,
    NOM1: tl.constexpr,
    SB2: tl.constexpr,
    NOM2: tl.constexpr,
    AB: tl.constexpr,
    NOMA: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
):
    p = tl.program_id(0)
    for i in tl.static_range(K):
        row = p * K + i
        base = row * N
        offs1 = tl.arange(0, SB1)
        if NOM1:
            x0 = tl.load(X + base + offs1).to(tl.float32)
        else:
            x0 = tl.load(
                X + base + offs1, mask=offs1 < N, other=0.0
            ).to(tl.float32)
        s1 = tl.sum(x0, axis=0)
        acc2 = tl.zeros((SB2,), dtype=tl.float32)
        for c in range(0, N, SB2):
            offs = c + tl.arange(0, SB2)
            if NOM2:
                x = tl.load(X + base + offs).to(tl.float32)
            else:
                x = tl.load(X + base + offs, mask=offs < N, other=0.0).to(
                    tl.float32
                )
            acc2 += x * x
        s2 = tl.sum(acc2, axis=0)
        m = s1 / N
        var = s2 / N - m * m
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Mean + row, m.to(Mean.dtype.element_ty))
        tl.store(Rstd + row, rstd.to(Rstd.dtype.element_ty))
        for c in range(0, N, AB):
            offs = c + tl.arange(0, AB)
            if NOMA:
                x = tl.load(X + base + offs).to(tl.float32)
                if HAS_W and HAS_B:
                    w = tl.load(W + offs).to(tl.float32)
                    b = tl.load(B + offs).to(tl.float32)
                    y = (x - m) * rstd * w + b
                elif HAS_W:
                    w = tl.load(W + offs).to(tl.float32)
                    y = (x - m) * rstd * w
                elif HAS_B:
                    b = tl.load(B + offs).to(tl.float32)
                    y = (x - m) * rstd + b
                else:
                    y = (x - m) * rstd
                tl.store(Y + base + offs, y.to(Y.dtype.element_ty))
            else:
                mm = offs < N
                x = tl.load(X + base + offs, mask=mm, other=0.0).to(tl.float32)
                if HAS_W and HAS_B:
                    w = tl.load(W + offs, mask=mm, other=1.0).to(tl.float32)
                    b = tl.load(B + offs, mask=mm, other=0.0).to(tl.float32)
                    y = (x - m) * rstd * w + b
                elif HAS_W:
                    w = tl.load(W + offs, mask=mm, other=1.0).to(tl.float32)
                    y = (x - m) * rstd * w
                elif HAS_B:
                    b = tl.load(B + offs, mask=mm, other=0.0).to(tl.float32)
                    y = (x - m) * rstd + b
                else:
                    y = (x - m) * rstd
                tl.store(Y + base + offs, y.to(Y.dtype.element_ty), mask=mm)


# ---------------- wave31: 16-bit layer_norm carrier ----------------
# probe313b (same session, common vendor kernel -> carrier): 512^2 2.62 ->
# 0.66, 2048^2 15.08 -> 5.61, 512x4096 5.69 -> 2.49, 8192x512 36.22 -> 9.19,
# 4096x1024 21.94 -> 6.69 ms (2.3-4.1x). K=8 beats K=16 everywhere and BN=128
# beats 256; AB=512 ties AB=1024 and matches the f32 fast path.
_LN16_BN = 128


def _ln16_ab(n: int) -> int:
    for cand in (512, 256, 128):
        if n % cand == 0:
            return cand
    return 128


def _pick_k16(m: int, n: int) -> int:
    # wave32 probe322: K=32 wins at every 16-bit shape measured (512^2 0.712 ->
    # 0.422, 2048^2 3.04 -> 2.24, 8192x512 7.04 -> 4.88, 4096x1024 4.22 ->
    # 3.07 ms); more rows per program amortises the launch.
    # wave34 probe346/348: rows/program also sets the grid, and M rows with a
    # K=M launch leaves one program on eight cores - (16,8,128,128) N=131072
    # ran 4.55 ms at K=16 against 2.22 ms at K=2. Keep at least 8 programs.
    k = 32
    while k > 1 and m % k:
        k //= 2
    return min(k, max(1, m // 8))


def _ln16_tiles(n: int):
    """(BN, AB) for the 16-bit carrier; very wide rows prefer 256/1024."""
    # wave35 probe357: N=393216 f16 8.05 ms at BN256/AB1024 against 8.15 at
    # BN128/AB512 (1.2%), everything below 262144 keeps the old pair.
    if n >= (1 << 18):
        return 256, 1024
    # wave38 probe396: BN came out as next_pow2(n) below 128; at N=4/8/16 the
    # masked carrier then reduced a sub-32-lane block and returned NaN for
    # every element.  Floor BN at 32 (the alignment test below then sends those
    # widths to the masked kernel, which pads the extra lanes safely).
    return max(32, min(_LN16_BN, triton.next_power_of_2(n))), _ln16_ab(n)


@triton.jit
def _ln_fwd_kernel_16(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N,
    eps,
    K: tl.constexpr,
    BN: tl.constexpr,
    AB: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
):
    p = tl.program_id(0)
    cols = tl.arange(0, BN)
    for i in tl.static_range(K):
        row = p * K + i
        base = row.to(tl.int64) * N
        acc1 = tl.zeros([BN], dtype=tl.float32)
        acc2 = tl.zeros([BN], dtype=tl.float32)
        for off in range(0, N, BN):
            v = tl.load(X + base + off + cols).to(tl.float32)
            acc1 += v
            acc2 += v * v
        s1 = tl.sum(acc1, axis=0)
        s2 = tl.sum(acc2, axis=0)
        m = s1 / N
        var = s2 / N - m * m
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Mean + row, m.to(Mean.dtype.element_ty))
        tl.store(Rstd + row, rstd.to(Rstd.dtype.element_ty))
        idx = tl.arange(0, AB)
        for c in range(0, N, AB):
            x = tl.load(X + base + c + idx).to(tl.float32)
            y = (x - m) * rstd
            if HAS_W and HAS_B:
                w = tl.load(W + c + idx).to(tl.float32)
                b = tl.load(B + c + idx).to(tl.float32)
                y = y * w + b
            elif HAS_W:
                w = tl.load(W + c + idx).to(tl.float32)
                y = y * w
            elif HAS_B:
                b = tl.load(B + c + idx).to(tl.float32)
                y = y + b
            tl.store(Y + base + c + idx, y.to(Y.dtype.element_ty))


@triton.jit
def _ln_fwd_kernel_16m(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N,
    eps,
    K: tl.constexpr,
    BN: tl.constexpr,
    AB: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
):
    """wave33: masked 16-bit carrier for N that BN/AB do not divide.

    Same streaming two-loop structure as _ln_fwd_kernel_16, but every load and
    store carries a mask, so the 16-bit path now covers any N (official test
    shapes (16,16,4098), (100,40499), (1,40999)). Only used when the aligned
    kernel cannot be: the masked form is a few percent slower.
    """
    p = tl.program_id(0)
    cols = tl.arange(0, BN)
    for i in tl.static_range(K):
        row = p * K + i
        base = row.to(tl.int64) * N
        acc1 = tl.zeros([BN], dtype=tl.float32)
        acc2 = tl.zeros([BN], dtype=tl.float32)
        for off in range(0, N, BN):
            mm = off + cols < N
            v = tl.load(X + base + off + cols, mask=mm, other=0.0).to(tl.float32)
            acc1 += v
            acc2 += v * v
        s1 = tl.sum(acc1, axis=0)
        s2 = tl.sum(acc2, axis=0)
        mu = s1 / N
        var = s2 / N - mu * mu
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Mean + row, mu.to(Mean.dtype.element_ty))
        tl.store(Rstd + row, rstd.to(Rstd.dtype.element_ty))
        idx = tl.arange(0, AB)
        for c in range(0, N, AB):
            mm2 = c + idx < N
            x = tl.load(X + base + c + idx, mask=mm2, other=0.0).to(tl.float32)
            y = (x - mu) * rstd
            if HAS_W and HAS_B:
                w = tl.load(W + c + idx, mask=mm2, other=0.0).to(tl.float32)
                b = tl.load(B + c + idx, mask=mm2, other=0.0).to(tl.float32)
                y = y * w + b
            elif HAS_W:
                w = tl.load(W + c + idx, mask=mm2, other=0.0).to(tl.float32)
                y = y * w
            elif HAS_B:
                b = tl.load(B + c + idx, mask=mm2, other=0.0).to(tl.float32)
                y = y + b
            tl.store(Y + base + c + idx, y.to(Y.dtype.element_ty), mask=mm2)


# ---------------- wave35: split-N carrier for very wide f32 rows ----------------
# Official core shape (20,6,65536) normalizes over [6,65536] = 393216 columns
# with M=20. One row per program leaves ten programs for eight cores, so the
# second wave runs on two cores; f32 also pays a W/B load per row (two 786 KB
# vectors per row). Splitting the row into two column halves and applying
# column-major (tile outer, rows inner, W/B loaded once per tile) measured
# (probe356, same session): common 10.36 / carrier 13.8 / splitA 10.76 /
# splitB 9.17 ms. 16-bit keeps the plain carrier (8.05 ms) - its apply pass is
# already bandwidth bound and the extra pass only adds traffic.
_LN_SPLIT_MIN_N = 1 << 18
# wave37 probe385/386 (three interleaved rounds, min-of-9): S=2 7.12 ms was
# not the best split - S3/AB512 6.16, S6/AB1024 6.28, S4/AB1024 6.46,
# S4/AB512 6.73, S8/AB1024 6.75 at (20,6,65536) f32, error 1.9e-6 against
# torch (fp32 accumulation order only).  Keep the pick shape-aware because the
# split path must divide N: prefer 3, then 2/4/6/8.
_LN_SPLIT_S = 3
_LN_SPLIT_AB = 512


def _pick_split(n: int) -> int:
    for s in (_LN_SPLIT_S, 2, 4, 6, 8):
        if n % s == 0:
            return s
    return 1


@triton.jit
def _ln_split_part_kernel(
    X,
    P1,
    P2,
    N,
    SPLIT: tl.constexpr,
    CH: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // SPLIT
    s = pid % SPLIT
    base = row.to(tl.int64) * N
    lo = s * CH
    hi = tl.minimum(lo + CH, N)
    cols = tl.arange(0, BN)
    acc1 = tl.zeros([BN], dtype=tl.float32)
    acc2 = tl.zeros([BN], dtype=tl.float32)
    for off in range(lo, hi, BN):
        mm = off + cols < hi
        v = tl.load(X + base + off + cols, mask=mm, other=0.0).to(tl.float32)
        acc1 += v
        acc2 += v * v
    tl.store(P1 + pid, tl.sum(acc1, axis=0))
    tl.store(P2 + pid, tl.sum(acc2, axis=0))


@triton.jit
def _ln_split_combine_kernel(P1, P2, Mean, Rstd, N, eps, SPLIT: tl.constexpr):
    row = tl.program_id(0)
    s1 = 0.0
    s2 = 0.0
    for i in tl.static_range(SPLIT):
        s1 += tl.load(P1 + row * SPLIT + i)
        s2 += tl.load(P2 + row * SPLIT + i)
    mu = s1 / N
    var = s2 / N - mu * mu
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Mean + row, mu.to(Mean.dtype.element_ty))
    tl.store(Rstd + row, rstd.to(Rstd.dtype.element_ty))


@triton.jit
def _ln_split_apply_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    M,
    N,
    SPLIT: tl.constexpr,
    CH: tl.constexpr,
    AB: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
):
    s = tl.program_id(0)
    lo = s * CH
    hi = tl.minimum(lo + CH, N)
    idx = tl.arange(0, AB)
    for c in range(lo, hi, AB):
        mm = c + idx < hi
        if HAS_W:
            w = tl.load(W + c + idx, mask=mm, other=0.0).to(tl.float32)
        else:
            w = tl.zeros([AB], dtype=tl.float32)
        if HAS_B:
            b = tl.load(B + c + idx, mask=mm, other=0.0).to(tl.float32)
        else:
            b = tl.zeros([AB], dtype=tl.float32)
        for r in range(0, M):
            base = r.to(tl.int64) * N
            x = tl.load(X + base + c + idx, mask=mm, other=0.0).to(tl.float32)
            mu = tl.load(Mean + r).to(tl.float32)
            rs = tl.load(Rstd + r).to(tl.float32)
            y = (x - mu) * rs
            if HAS_W:
                y = y * w
            if HAS_B:
                y = y + b
            tl.store(Y + base + c + idx, y.to(Y.dtype.element_ty), mask=mm)


def ln_split_launch(x2d, y2d, w_arg, b_arg, mean, rstd, m, n, eps, has_w, has_b):
    split = _pick_split(n)
    ch = (n + split - 1) // split
    p1 = torch.empty(m * split, dtype=torch.float32, device=x2d.device)
    p2 = torch.empty(m * split, dtype=torch.float32, device=x2d.device)
    with nested_dispatch(m * n):
        _ln_split_part_kernel[(m * split,)](
            x2d, p1, p2, n, SPLIT=split, CH=ch, BN=128
        )
        _ln_split_combine_kernel[(m,)](p1, p2, mean, rstd, n, eps, SPLIT=split)
        _ln_split_apply_kernel[(split,)](
            x2d,
            y2d,
            w_arg,
            b_arg,
            mean,
            rstd,
            m,
            n,
            SPLIT=split,
            CH=ch,
            AB=_LN_SPLIT_AB,
            HAS_W=has_w,
            HAS_B=has_b,
        )


_LN_K_SMALL = 32  # N <= 1024 (wave13 probe109 P3: 512^2 0.404 -> 0.343)
_LN_K_LARGE = 16  # N > 1024 (wave13 probe112 P1: 2048^2 2.86 -> 2.50,
# 512x2048 0.79 -> 0.71 at K16)


def _pick_k(m: int, n: int) -> int:
    k = _LN_K_SMALL if n <= 1024 else _LN_K_LARGE
    while k > 1 and m % k:
        k //= 2
    return k


# -------------------- wave39: narrow-row forward tile --------------------
# probe421/422: below 32 lanes the f32 kernel's SB2/AB blocks (== N) are not
# only slow but unsafe.  The backend widens the row load and folds whatever
# sits past the row into the variance, so on the first call after allocation
# a handful of rows come back NaN - 8 of 16384 elements at (4096,4), 224 at
# (2048,16), 36 at (2048,12) - while the same tensors are clean on the next
# call and an over-allocated buffer is always clean.  Both blocks are floored
# at 32 (masked) now, exactly as wave38 fixed the backward; this tile is the
# fast path for the widths that were affected.
# probe420/425 same-session minima, routed -> tile:
#   (4096,4)  f32 2.61 -> 0.404 (6.5x, torch 0.468)  f16 8.14 -> 0.478 (17x)
#   (4096,8)      1.25 -> 0.347 (3.6x)               f16 2.61 -> 0.385 (6.8x)
#   (2048,16)     0.87 -> 0.332 (2.6x)               f16 1.45 -> 0.348 (4.2x)
#   (2048,32)     0.98 -> 0.417 (2.4x)               f16 1.44 -> 0.443 (3.3x)
# 64 columns was left to the routed kernel on wave39's evidence (0.55 against
# 1.02 at (1024,64)) - but that was measured before the tile got its own BM
# rule and the nested launch.  wave41 probe438, op level, gate 32 -> 64:
#   f16 (1024,64) 0.814 -> 0.501 (1.63x), (2048,64) 1.354 -> 0.715 (1.89x),
#       (4096,64) 2.202 -> 1.167 (1.89x)
#   f32 (1024,64) 0.525 -> 0.498 (1.05x), (2048,64) 0.759 -> 0.691 (1.10x),
#       (4096,64) 1.161 -> 1.084 (1.07x)
# The 16-bit side is the one that mattered: it had to mask (AB=128 > 64) and
# was 3-14% slower than the generic kernels there (wave40), and the tile now
# beats the generic path outright.  BM=128 is the knee at 64 columns for both
# dtypes (BM256 1.7-1.9x worse, BM512 1.5-2.9x worse), which is what the
# existing _fwd_tile_bm already picks from 256 rows up.
_LN_FWD_TILE_MAX_N = 64


def _fwd_tile_bm(m: int, bn: int) -> int:
    """Rows per tile program: ~8K elements, 64-row floor, 2+ programs."""
    bm = min(1024, max(64, 8192 // bn), max(64, m // 2))
    while bm > 8 and m % bm:
        bm //= 2
    return bm if m % bm == 0 else 0


@triton.jit
def _ln_fwd_tile_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N,
    eps,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
):
    p = tl.program_id(0)
    idx = p * BM * BN + tl.arange(0, BM * BN)
    x = tl.reshape(tl.load(X + idx).to(tl.float32), (BM, BN))
    cols = tl.arange(0, BN)
    mu = tl.sum(x, axis=1)[:, None] / N
    var = tl.sum(x * x, axis=1)[:, None] / N - mu * mu
    rs = 1.0 / tl.sqrt(var + eps)
    rows = p * BM + tl.arange(0, BM)
    tl.store(Mean + rows, tl.reshape(mu, (BM,)).to(Mean.dtype.element_ty))
    tl.store(Rstd + rows, tl.reshape(rs, (BM,)).to(Rstd.dtype.element_ty))
    y = (x - mu) * rs
    if HAS_W:
        y = y * tl.load(W + cols).to(tl.float32)[None, :]
    if HAS_B:
        y = y + tl.load(B + cols).to(tl.float32)[None, :]
    tl.store(Y + idx, tl.reshape(y, (BM * BN,)).to(Y.dtype.element_ty))


def ln_fast(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Return (y, mean_view, rstd_view) like aten.native_layer_norm."""
    inp = input.contiguous()
    n = _prod(normalized_shape)
    m = inp.numel() // n
    # wave45: the two 2-D views are only needed by the wide-row branches (the
    # tile addresses memory with p*BM*BN + arange), and each .view() costs
    # 7.5 us on this torch build, so they are created lazily below.
    x2d = None
    y2d = None
    y = torch.empty_like(inp)
    mean = torch.empty(m, dtype=inp.dtype, device=inp.device)
    rstd = torch.empty(m, dtype=inp.dtype, device=inp.device)
    sb1 = max(64, triton.next_power_of_2(n))
    # wave39 probe422: never hand this kernel a block narrower than 32 lanes -
    # with SB2/AB == N < 32 the widened row load picks up the tail page and a
    # few rows go NaN on the first call after allocation.
    sb2 = max(32, _SB2 if n >= _SB2 else triton.next_power_of_2(n))
    if n >= 1024 and m * n < _SB2_SMALL_TOTAL:
        sb2 = max(32, 256)
    ab = max(32, _AB if n >= _AB else triton.next_power_of_2(n))
    w_arg = weight if weight is not None else inp
    b_arg = bias if bias is not None else inp
    stats_shape = list(inp.shape[: inp.dim() - len(normalized_shape)]) + [1] * len(
        normalized_shape
    )
    bn_t = triton.next_power_of_2(n)
    bm_t = (
        _fwd_tile_bm(m, bn_t)
        if (
            os.environ.get("FLAGGEMS_SPACEMIT_LN_FWD_TILE", "1") != "0"
            and n <= _LN_FWD_TILE_MAX_N
            and n == bn_t
            and m >= 8
        )
        else 0
    )
    if bm_t:
        # wave39: the wide-row kernels re-read x for the apply pass and cannot
        # take a narrow block safely, so a whole narrow row goes through one
        # tile instead.  nested_dispatch self-gates at 32768 elements, which is
        # where probe425 puts the crossover (below it plain launch wins by 2%).
        # wave45: same host-path treatment the backward got in wave44 - one
        # cached CompiledKernel launch instead of the JIT's re-bind + key +
        # lookup (50-65 us), no x2d/y2d views (the tile only uses pointer +
        # arithmetic, 15 us for the pair), and the putenv-based nested switch.
        nprog = m // bm_t
        kern = _ln_fwd_tile_kernel
        has_w = weight is not None
        has_b = bias is not None
        targs = (inp, y, w_arg, b_arg, mean, rstd, n, eps, bm_t, bn_t, has_w, has_b)
        with _FastNested(m * n):
            _cached_launch(
                kern,
                (
                    kern,
                    "fwd_tile",
                    n,
                    m,
                    bm_t,
                    bn_t,
                    has_w,
                    has_b,
                    inp.dtype,
                    _align(inp, y, w_arg, b_arg, mean, rstd),
                ),
                (nprog,),
                targs,
            )
        return y, mean.view(stats_shape), rstd.view(stats_shape)

    # wide-row branches below need the 2-D views
    x2d = inp.view(m, n)
    y2d = y.view(m, n)
    if inp.dtype != torch.float32 or n > _F32_FAST_MAX_N:
        stats_shape = list(inp.shape[: inp.dim() - len(normalized_shape)]) + [
            1
        ] * len(normalized_shape)
        # wave35 probe356: f32 rows at or above 262144 columns run split-N
        # (9.17 vs 10.36 ms common at (20,6,65536)); every 16-bit width keeps
        # the single-pass carrier (8.05 ms there).
        if (
            inp.dtype == torch.float32
            and n >= _LN_SPLIT_MIN_N
            and _pick_split(n) > 1
            and m >= 2
        ):
            ln_split_launch(
                x2d,
                y2d,
                w_arg,
                b_arg,
                mean,
                rstd,
                m,
                n,
                eps,
                weight is not None,
                bias is not None,
            )
            return y, mean.view(stats_shape), rstd.view(stats_shape)
        k16 = _pick_k16(m, n)
        bn16, ab16 = _ln16_tiles(n)
        aligned = (n % bn16 == 0) and (n % ab16 == 0)
        kern16 = _ln_fwd_kernel_16 if aligned else _ln_fwd_kernel_16m
        # wave31 probe314: the serialized 1-D launcher costs the 16-bit carrier
        # almost half its time - nesting takes 2048^2 5.53 -> 2.81, 512x4096
        # 2.47 -> 1.05, 4096x1024 6.74 -> 4.12, 8192x512 9.14 -> 7.00 ms.
        with nested_dispatch(m * n):
            kern16[(m // k16,)](
                x2d,
                y2d,
                w_arg,
                b_arg,
                mean,
                rstd,
                n,
                eps,
                K=k16,
                BN=bn16,
                AB=ab16,
                HAS_W=(weight is not None),
                HAS_B=(bias is not None),
            )
        stats_shape = list(inp.shape[: inp.dim() - len(normalized_shape)]) + [
            1
        ] * len(normalized_shape)
        return y, mean.view(stats_shape), rstd.view(stats_shape)
    k = _pick_k(m, n)

    def _launch():
        _ln_fwd_kernel[(m // k,)](
            x2d,
            y2d,
            w_arg,
            b_arg,
            mean,
            rstd,
            n,
            eps,
            K=k,
            SB1=sb1,
            NOM1=(n == sb1),
            SB2=sb2,
            NOM2=(n % sb2 == 0),
            AB=ab,
            NOMA=(n % ab == 0),
            HAS_W=(weight is not None),
            HAS_B=(bias is not None),
        )

    if k >= 4:
        with nested_dispatch(m * n):
            _launch()
    else:
        _launch()
    stats_shape = list(inp.shape[: inp.dim() - len(normalized_shape)]) + [
        1
    ] * len(normalized_shape)
    return y, mean.view(stats_shape), rstd.view(stats_shape)


# ------------------------- backward (wave4) -------------------------
# Structure (probe46, same-session f32): dX = per-row kernel, two single-chain
# acc loops (dx2 = sum(g*w), dx3 = sum(g*w*x_hat)) + apply loop, BLOCK=512.
# dW/dB = row-fold split: stage1 accumulates column vectors over row blocks
# (grid (cdiv(M,64), cdiv(N,CB)), partials per row-block), stage2 combines.
# 2048^2: dx 12.08 (common+nested 89.7), wbdb 2.97; 512x2048: 3.08 + 0.87
# (common 36.17); 512^2: 1.19 + 0.36 (common 15.41). Accuracies vs torch:
# dX <= 5.7e-6, dW <= 6.1e-5 (scale ~sqrt(M)), dB <= 5e-5.

_DX_BLOCK = 512


def _dx_block(n: int) -> int:
    # wave36 probe370: a fixed BLOCK=512 spends 128 lanes on every useful
    # element when the row is narrow - at (4096,4) dx alone took 12.8 of the
    # 13.4 ms while dW/dB took 0.53.  Cap the pass at the row width.
    # wave38 probe395: the cap alone is wrong below 32 lanes - with BLOCK == N
    # in {4,8,16} every output element comes back NaN for both dtypes, while
    # BLOCK=32 on the same shape is exact (it is masked, so the padding lanes
    # are never read).  Floor at 32 so the un-tiled fallback stays correct.
    return max(32, min(_DX_BLOCK, triton.next_power_of_2(n)))
_DWDB_ROWS = 64
_DWDB_CB = 2048


def _pick_dx_k(m: int, n: int) -> int:
    # wave13 probe109 P5b: K16 wins at 2048^2 (4.68 vs 4.96 ms) and 512^2
    # (0.526 vs 0.603); 512x2048 keeps K8 (1.27 vs 1.31).
    k = 16 if (n <= 1024 or m >= 2048) else 8
    while k > 1 and m % k:
        k //= 2
    # wave36 probe372b/372c: a 4-column row cannot amortize 32 serial passes -
    # at (4096,4) f32 B4/K32 3.19 ms, K4 1.52, K2 1.64, K1 1.75 (the three
    # per-row loops are the cost, so narrow rows take a small K and leave the
    # parallelism to the grid).
    if n <= 8:
        k = min(k, 4)
    return k


def lnb_fast_enabled() -> bool:
    return (
        os.environ.get("FLAGGEMS_SPACEMIT_LNB_FAST", "1") != "0"
        and os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") != "0"
    )


def lnb_fast_ok(
    grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
) -> bool:
    if not (
        isinstance(grad_out, torch.Tensor) and isinstance(input, torch.Tensor)
    ):
        return False
    if (
        grad_out.shape != input.shape
        or input.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or grad_out.dtype != input.dtype
    ):
        return False
    if not input.is_contiguous() or not grad_out.is_contiguous():
        return False
    numel = input.numel()
    if numel < _LNB_MIN_N:
        return False
    try:
        n = _prod(normalized_shape)
    except (TypeError, ValueError):
        return False
    nmax = _LN16_MAX_N  # the backward kernels are masked streaming loops at any N
    if n <= 0 or n > nmax or numel % n:
        return False
    m = numel // n
    if not (isinstance(mean, torch.Tensor) and isinstance(rstd, torch.Tensor)):
        return False
    if mean.numel() != m or rstd.numel() != m:
        return False
    if mean.dtype != rstd.dtype or mean.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        return False
    if not mean.is_contiguous() or not rstd.is_contiguous():
        return False
    for t in (weight, bias):
        if t is not None:
            if (
                not isinstance(t, torch.Tensor)
                or t.numel() != n
                or t.dtype != input.dtype
                or not t.is_contiguous()
            ):
                return False
    if output_mask is None:
        output_mask = (True, True, True)
    if len(output_mask) != 3:
        return False
    if output_mask[1] and weight is None:
        return False
    if output_mask[2] and bias is None:
        return False
    return True


@triton.jit
def _ln_bwd_dx_kernel(
    G,
    X,
    W,
    Mean,
    Rstd,
    dX,
    N,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    NOMASK: tl.constexpr,
    HAS_W: tl.constexpr,
):
    p = tl.program_id(0)
    for i in tl.static_range(K):
        row = p * K + i
        base = row * N
        m = tl.load(Mean + row).to(tl.float32)
        rs = tl.load(Rstd + row).to(tl.float32)
        acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            if NOMASK:
                g = tl.load(G + base + offs).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs).to(tl.float32)
                    acc2 += g * w
                else:
                    acc2 += g
            else:
                mm = offs < N
                g = tl.load(G + base + offs, mask=mm, other=0.0).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs, mask=mm, other=0.0).to(tl.float32)
                    acc2 += g * w
                else:
                    acc2 += g
        dx2 = tl.sum(acc2, axis=0)
        acc3 = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            if NOMASK:
                g = tl.load(G + base + offs).to(tl.float32)
                x = tl.load(X + base + offs).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs).to(tl.float32)
                    dwh = g * w
                else:
                    dwh = g
                xh = (x - m) * rs
                acc3 += dwh * xh
            else:
                mm = offs < N
                g = tl.load(G + base + offs, mask=mm, other=0.0).to(tl.float32)
                x = tl.load(X + base + offs, mask=mm, other=0.0).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs, mask=mm, other=0.0).to(tl.float32)
                    dwh = g * w
                else:
                    dwh = g
                xh = (x - m) * rs
                acc3 += dwh * xh
        dx3 = tl.sum(acc3, axis=0)
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            if NOMASK:
                g = tl.load(G + base + offs).to(tl.float32)
                x = tl.load(X + base + offs).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs).to(tl.float32)
                    dwh = g * w
                else:
                    dwh = g
                xh = (x - m) * rs
                dx = rs * (dwh - (dx2 + xh * dx3) / N)
                tl.store(dX + base + offs, dx.to(dX.dtype.element_ty))
            else:
                mm = offs < N
                g = tl.load(G + base + offs, mask=mm, other=0.0).to(tl.float32)
                x = tl.load(X + base + offs, mask=mm, other=0.0).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs, mask=mm, other=0.0).to(tl.float32)
                    dwh = g * w
                else:
                    dwh = g
                xh = (x - m) * rs
                dx = rs * (dwh - (dx2 + xh * dx3) / N)
                tl.store(
                    dX + base + offs, dx.to(dX.dtype.element_ty), mask=mm
                )


# ------------------- wave38: fused narrow-row dx tile -------------------
# probe390-395: a row narrower than 32 lanes cannot use the three-pass per-row
# kernel at all - with BLOCK == N in {4,8,16} every element comes back NaN for
# both dtypes - and padding that kernel to BLOCK=32 costs 7.13 ms at (4096,4)
# against 1.51 for the broken form.  A (BM rows x BN cols) tile keeps the whole
# narrow row in one block, so both row sums come from one tl.sum(axis=1) each
# and the apply reuses the same registers.  Same-session minima, f32:
# (4096,4) 1.51 -> 0.53, (4096,8) 1.90 -> 0.94, (2048,16) 1.14 -> 0.73,
# (2048,32) 2.23 -> 1.35; f16 (4096,4) 1.65 -> 0.54.  Non-power-of-two widths
# lose badly (N=100: 12.9 against 5.98 ms), so the tile is gated on
# n == next_power_of_2(n).
# wave40 probe428/430: 64 columns was called a loss for the tile in wave38
# (1.42 vs 0.73 against the per-row kernel) but that compared against the
# per-row *dx* kernel alone; against the whole routed backward the tile wins at
# every M >= 128 rows - f32 1.16/0.98/1.32/1.26/1.29/1.23/1.19x and f16
# 1.61/1.53/2.04/2.08/2.30/2.37/2.39x at M = 128/256/512/1024/2048/4096/8192 -
# so the gate now stops at 64.  At 128 columns the tile is still a loss (0.87
# f32 / 0.77 f16 at 2048 rows), which is where the gate stays.
_LN_DX_TILE_MAX_N = 64


def _dx_tile_bm(m: int, bn: int) -> int:
    """Rows per tile program; the tile is ~4-16K elements, grid stays >16."""
    if bn >= 64:
        # wave40 probe430: at 64 lanes the tile is register-bound, not launch
        # bound - 128 rows/program is the knee at every M (8192 rows: BM128
        # 4.13 ms against BM256 4.90 and BM512 6.29), and at 128 rows the
        # single-program tile loses 12% so keep at least two programs.
        cap = 128
        while cap > 16 and (m % cap or m // cap < 2):
            cap //= 2
        return cap if m % cap == 0 else 0
    cap = 1024 if bn <= 8 else 512
    if bn >= 16:
        # wave42 probe453: at (2048,32) a 512-row tile leaves four programs on
        # eight cores and the launch-to-launch idle costs 1.35x - f32 1.135 ->
        # 0.724 and f16 1.279 -> 0.792 ms with BM=256 (eight programs).  Below
        # 16 lanes the tile is launch bound instead, so the cap stays put
        # (wave38: (4096,4) prefers a single 1024-row program).
        cap = min(cap, max(64, m // 8))
    bm = cap
    while bm > 16 and m % bm:
        bm //= 2
    return bm if m % bm == 0 else 0


@triton.jit
def _ln_bwd_dx_tile_kernel(
    G,
    X,
    W,
    Mean,
    Rstd,
    dX,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_W: tl.constexpr,
):
    p = tl.program_id(0)
    idx = p * BM * BN + tl.arange(0, BM * BN)
    g = tl.reshape(tl.load(G + idx).to(tl.float32), (BM, BN))
    x = tl.reshape(tl.load(X + idx).to(tl.float32), (BM, BN))
    cols = tl.arange(0, BN)
    if HAS_W:
        w = tl.load(W + cols).to(tl.float32)[None, :]
        dwh = g * w
    else:
        dwh = g
    rows = p * BM + tl.arange(0, BM)
    mu = tl.load(Mean + rows).to(tl.float32)[:, None]
    rs = tl.load(Rstd + rows).to(tl.float32)[:, None]
    xh = (x - mu) * rs
    s2 = tl.sum(dwh, axis=1)[:, None]
    s3 = tl.sum(dwh * xh, axis=1)[:, None]
    dx = rs * (dwh - (s2 + xh * s3) / N)
    tl.store(dX + idx, tl.reshape(dx, (BM * BN,)).to(dX.dtype.element_ty))


@triton.jit
def _ln_bwd_narrow_kernel(
    G,
    X,
    W,
    Mean,
    Rstd,
    dX,
    dWp,
    dBp,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_W: tl.constexpr,
):
    """wave38: the dx tile plus the dW/dB column partials in the same pass.

    g, x and x_hat are already in registers, so the per-program column sums
    cost one extra tl.sum(axis=0) each instead of a second kernel that re-reads
    both tensors (probe399 (4096,4) f32 1.12 -> 0.76 ms).
    """
    p = tl.program_id(0)
    idx = p * BM * BN + tl.arange(0, BM * BN)
    g = tl.reshape(tl.load(G + idx).to(tl.float32), (BM, BN))
    x = tl.reshape(tl.load(X + idx).to(tl.float32), (BM, BN))
    cols = tl.arange(0, BN)
    if HAS_W:
        w = tl.load(W + cols).to(tl.float32)[None, :]
        dwh = g * w
    else:
        dwh = g
    rows = p * BM + tl.arange(0, BM)
    mu = tl.load(Mean + rows).to(tl.float32)[:, None]
    rs = tl.load(Rstd + rows).to(tl.float32)[:, None]
    xh = (x - mu) * rs
    s2 = tl.sum(dwh, axis=1)[:, None]
    s3 = tl.sum(dwh * xh, axis=1)[:, None]
    dx = rs * (dwh - (s2 + xh * s3) / N)
    tl.store(dX + idx, tl.reshape(dx, (BM * BN,)).to(dX.dtype.element_ty))
    tl.store(dWp + p * BN + cols, tl.sum(g * xh, axis=0))
    tl.store(dBp + p * BN + cols, tl.sum(g, axis=0))


@triton.jit
def _ln_bwd_narrow_dot_kernel(
    G,
    X,
    W,
    Mean,
    Rstd,
    dX,
    dWp,
    dBp,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_W: tl.constexpr,
):
    """wave43: the same tile as _ln_bwd_narrow_kernel with dot-based reductions.

    probe462/463b: on this backend a tl.sum(axis=1) over 32/64 lanes costs
    0.15-0.28 ms per pass on a (BM,BN) tile - the whole elementwise body is
    0.23-0.29 - and a manual reshape tree is 1.4x worse than tl.sum.  tl.dot
    with an all-ones (BN,BN) block returns (BM,BN) with every element already
    holding its row sum, so the apply needs neither a horizontal reduce nor a
    broadcast: probe464 (2048,32) f32 0.530 -> 0.363, (2048,64) 0.880 -> 0.698,
    f16 0.588 -> 0.381 / 0.971 -> 0.743, bit-identical output.
    The column partials go through dot with (16,BM) ones; only row 0 of the
    (16,BN) result is stored (mask), so the combine kernel is unchanged.
    probe465 whole fused path: 1.19-1.50x over the tl.sum form at every
    official narrow shape, f32 and f16.
    """
    p = tl.program_id(0)
    idx = p * BM * BN + tl.arange(0, BM * BN)
    g = tl.reshape(tl.load(G + idx).to(tl.float32), (BM, BN))
    x = tl.reshape(tl.load(X + idx).to(tl.float32), (BM, BN))
    cols = tl.arange(0, BN)
    if HAS_W:
        w = tl.load(W + cols).to(tl.float32)[None, :]
        dwh = g * w
    else:
        dwh = g
    rows = p * BM + tl.arange(0, BM)
    mu = tl.load(Mean + rows).to(tl.float32)[:, None]
    rs = tl.load(Rstd + rows).to(tl.float32)[:, None]
    xh = (x - mu) * rs
    ones = tl.full((BN, BN), 1.0, tl.float32)
    s2 = tl.dot(dwh, ones, input_precision="ieee")
    s3 = tl.dot(dwh * xh, ones, input_precision="ieee")
    dx = rs * (dwh - (s2 + xh * s3) / N)
    tl.store(dX + idx, tl.reshape(dx, (BM * BN,)).to(dX.dtype.element_ty))
    rowsel = (tl.arange(0, 16) == 0)[:, None]
    ox = tl.full((16, BM), 1.0, tl.float32)
    zrow = tl.zeros((16, 1), tl.int32)
    tl.store(
        dWp + p * BN + zrow + cols[None, :],
        tl.dot(ox, g * xh, input_precision="ieee"),
        mask=rowsel,
    )
    tl.store(
        dBp + p * BN + zrow + cols[None, :],
        tl.dot(ox, g, input_precision="ieee"),
        mask=rowsel,
    )


@triton.jit
def _ln_bwd_narrow_dot16_kernel(
    G,
    X,
    W,
    Mean,
    Rstd,
    dX,
    dWp,
    dBp,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_W: tl.constexpr,
):
    """wave43 probe469: dot body with the full (16,BN) column block stored.

    The row-0 masked store in _ln_bwd_narrow_dot_kernel makes the compiler
    write 16 identical addresses; storing the whole block and reading row 0 of
    each block back in the combine is 3-6% faster at kernel level (2048,32
    0.534 -> 0.514; 2048,64 1.015 -> 0.957; 4096,64 1.734 -> 1.677 f32,
    similar f16).
    """
    p = tl.program_id(0)
    idx = p * BM * BN + tl.arange(0, BM * BN)
    g = tl.reshape(tl.load(G + idx).to(tl.float32), (BM, BN))
    x = tl.reshape(tl.load(X + idx).to(tl.float32), (BM, BN))
    cols = tl.arange(0, BN)
    if HAS_W:
        w = tl.load(W + cols).to(tl.float32)[None, :]
        dwh = g * w
    else:
        dwh = g
    rows = p * BM + tl.arange(0, BM)
    mu = tl.load(Mean + rows).to(tl.float32)[:, None]
    rs = tl.load(Rstd + rows).to(tl.float32)[:, None]
    xh = (x - mu) * rs
    ones = tl.full((BN, BN), 1.0, tl.float32)
    s2 = tl.dot(dwh, ones, input_precision="ieee")
    s3 = tl.dot(dwh * xh, ones, input_precision="ieee")
    dx = rs * (dwh - (s2 + xh * s3) / N)
    tl.store(dX + idx, tl.reshape(dx, (BM * BN,)).to(dX.dtype.element_ty))
    ox = tl.full((16, BM), 1.0, tl.float32)
    acc = tl.arange(0, 16)[:, None] * BN + cols[None, :]
    tl.store(
        dWp + p * 16 * BN + acc,
        tl.dot(ox, g * xh, input_precision="ieee"),
    )
    tl.store(
        dBp + p * 16 * BN + acc,
        tl.dot(ox, g, input_precision="ieee"),
    )


@triton.jit
def _ln_bwd_combine16_kernel(
    PW,
    PB,
    dW,
    dB,
    N,
    NRB: tl.constexpr,
    BLK: tl.constexpr,
):
    """Combine for _ln_bwd_narrow_dot16_kernel: row 0 of each (16,BLK) block."""
    cols = tl.arange(0, BLK)
    accW = tl.zeros((BLK,), dtype=tl.float32)
    accB = tl.zeros((BLK,), dtype=tl.float32)
    for s in range(0, NRB):
        accW += tl.load(PW + s * 16 * N + cols)
        accB += tl.load(PB + s * 16 * N + cols)
    tl.store(dW + cols, accW.to(dW.dtype.element_ty), mask=cols < N)
    tl.store(dB + cols, accB.to(dB.dtype.element_ty), mask=cols < N)


# ---------------- wave32: 16-bit backward dx carrier ----------------
# probe320 (same session, common vendor backward -> carrier): dx 512^2 11.22 ->
# 0.58, 2048^2 45.95 -> 4.25, 512x4096 20.68 -> 3.35, 8192x512 163.0 -> 7.23,
# 4096x1024 87.8 -> 5.55 ms (5-23x). BN=128 K=16 BLOCK=512; the three-pass
# structure (acc g*w, acc g*w*xh, apply) beats a fused two-accumulator loop by
# 2-8% on short rows and ties on long rows, so the f32 layout is reused with
# fp32 casts on every load. dW/dB reuse _ln_bwd_wb_kernel unchanged: it already
# accumulates in fp32 and casts on store; 16-bit only needed the gate lifted,
# plus ROWS=64 and a n/2 column block (probe320 wb: 6-43x over the common path).
_LN16_BWD_K = 16
_LN16_BWD_BN = 128


def _pick_dx_k16(m: int, n: int) -> int:
    # wave32 probe323: rows/program 32 for short rows, 16 for n>=2048
    # (512^2 0.575 -> 0.540, 8192x512 7.23 -> 6.96, 4096x1024 5.55 -> 5.25;
    # 512x4096 and 2048^2 tie within 1-3%).
    # wave34 probe345: same grid effect as the forward - at (16,16384) K=16
    # leaves one program and takes 1.453 ms against 0.752 ms at K=8.
    k = _LN16_BWD_K if n <= 1024 else 16
    while k > 1 and m % k:
        k //= 2
    # wave36: same narrow-row rule as the f32 picker (probe372b: K32 3.19 ms
    # against K4 1.52 at 4 columns on the f32 kernel; the 16-bit kernel has the
    # same three-pass body).
    if n <= 8:
        k = min(k, 4)
    return min(k, max(1, m // 8))


@triton.jit
def _ln_bwd_dx_kernel_16(
    G,
    X,
    W,
    Mean,
    Rstd,
    dX,
    N,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    NOMASK: tl.constexpr,
    HAS_W: tl.constexpr,
):
    p = tl.program_id(0)
    for i in tl.static_range(K):
        row = p * K + i
        base = row.to(tl.int64) * N
        m = tl.load(Mean + row).to(tl.float32)
        rs = tl.load(Rstd + row).to(tl.float32)
        acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            if NOMASK:
                g = tl.load(G + base + offs).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs).to(tl.float32)
                    acc2 += g * w
                else:
                    acc2 += g
            else:
                mm = offs < N
                g = tl.load(G + base + offs, mask=mm, other=0.0).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs, mask=mm, other=0.0).to(tl.float32)
                    acc2 += g * w
                else:
                    acc2 += g
        dx2 = tl.sum(acc2, axis=0)
        acc3 = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            if NOMASK:
                g = tl.load(G + base + offs).to(tl.float32)
                x = tl.load(X + base + offs).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs).to(tl.float32)
                    dwh = g * w
                else:
                    dwh = g
                xh = (x - m) * rs
                acc3 += dwh * xh
            else:
                mm = offs < N
                g = tl.load(G + base + offs, mask=mm, other=0.0).to(tl.float32)
                x = tl.load(X + base + offs, mask=mm, other=0.0).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs, mask=mm, other=0.0).to(tl.float32)
                    dwh = g * w
                else:
                    dwh = g
                xh = (x - m) * rs
                acc3 += dwh * xh
        dx3 = tl.sum(acc3, axis=0)
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            if NOMASK:
                g = tl.load(G + base + offs).to(tl.float32)
                x = tl.load(X + base + offs).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs).to(tl.float32)
                    dwh = g * w
                else:
                    dwh = g
                xh = (x - m) * rs
                dx = rs * (dwh - (dx2 + xh * dx3) / N)
                tl.store(dX + base + offs, dx.to(dX.dtype.element_ty))
            else:
                mm = offs < N
                g = tl.load(G + base + offs, mask=mm, other=0.0).to(tl.float32)
                x = tl.load(X + base + offs, mask=mm, other=0.0).to(tl.float32)
                if HAS_W:
                    w = tl.load(W + offs, mask=mm, other=0.0).to(tl.float32)
                    dwh = g * w
                else:
                    dwh = g
                xh = (x - m) * rs
                dx = rs * (dwh - (dx2 + xh * dx3) / N)
                tl.store(
                    dX + base + offs, dx.to(dX.dtype.element_ty), mask=mm
                )


@triton.jit
def _ln_bwd_wb_kernel(
    G,
    X,
    Mean,
    Rstd,
    PW,
    PB,
    M,
    N,
    ROWS: tl.constexpr,
    CB: tl.constexpr,
    NOMASK: tl.constexpr,
):
    rb = tl.program_id(0)
    cbb = tl.program_id(1)
    r0 = rb * ROWS
    r1 = tl.minimum(r0 + ROWS, M)
    cols = cbb * CB + tl.arange(0, CB)
    accW = tl.zeros((CB,), dtype=tl.float32)
    accB = tl.zeros((CB,), dtype=tl.float32)
    for r in range(r0, r1):
        off = r * N + cols
        if NOMASK:
            g = tl.load(G + off).to(tl.float32)
            x = tl.load(X + off).to(tl.float32)
        else:
            cm = cols < N
            g = tl.load(G + off, mask=cm, other=0.0).to(tl.float32)
            x = tl.load(X + off, mask=cm, other=0.0).to(tl.float32)
        m = tl.load(Mean + r).to(tl.float32)
        rs = tl.load(Rstd + r).to(tl.float32)
        xh = (x - m) * rs
        accW += g * xh
        accB += g
    if NOMASK:
        tl.store(PW + rb * N + cols, accW)
        tl.store(PB + rb * N + cols, accB)
    else:
        tl.store(PW + rb * N + cols, accW, mask=cols < N)
        tl.store(PB + rb * N + cols, accB, mask=cols < N)


@triton.jit
def _ln_bwd_combine_kernel(
    PW,
    PB,
    dW,
    dB,
    N,
    NRB: tl.constexpr,
    CB2: tl.constexpr,
    NOMASK: tl.constexpr,
):
    cb = tl.program_id(0)
    cols = cb * CB2 + tl.arange(0, CB2)
    accW = tl.zeros((CB2,), dtype=tl.float32)
    accB = tl.zeros((CB2,), dtype=tl.float32)
    for s in range(0, NRB):
        if NOMASK:
            accW += tl.load(PW + s * N + cols).to(tl.float32)
            accB += tl.load(PB + s * N + cols).to(tl.float32)
        else:
            cm = cols < N
            accW += tl.load(PW + s * N + cols, mask=cm, other=0.0).to(tl.float32)
            accB += tl.load(PB + s * N + cols, mask=cm, other=0.0).to(tl.float32)
    if NOMASK:
        tl.store(dW + cols, accW.to(dW.dtype.element_ty))
        tl.store(dB + cols, accB.to(dB.dtype.element_ty))
    else:
        tl.store(dW + cols, accW.to(dW.dtype.element_ty), mask=cols < N)
        tl.store(dB + cols, accB.to(dB.dtype.element_ty), mask=cols < N)


_KCACHE = {}
_LAUNCH = {"ok": True, "stream": None}
_ND_KEY = "SPINE_TRITON_FORCE_NESTED_DISPATCH_1D"


class _FastNested:
    """wave44: nested dispatch without the os.environ cost.

    _dispatch.nested_dispatch writes through os.environ, whose __setitem__/
    restore encode both strings (profile485: seven env touches per call, about
    0.02-0.05 ms at (2048,32)).  The spine launcher only reads the C environ,
    so os.putenv/unsetenv is enough and costs one call per switch.
    """

    __slots__ = ("active", "old")

    def __init__(self, numel):
        self.active = numel >= (1 << 15) and os.environ.get(
            "FLAGGEMS_SPACEMIT_NESTED", "1"
        ) != "0"
        self.old = None

    def __enter__(self):
        if self.active:
            self.old = os.environ.get(_ND_KEY)
            os.putenv(_ND_KEY, "1")
        return self

    def __exit__(self, *exc):
        if self.active:
            if self.old is None:
                os.unsetenv(_ND_KEY)
            else:
                os.putenv(_ND_KEY, self.old)
        return False


def _launch(kern, key, grid, args):
    """wave44: launch through a cached CompiledKernel.

    JITFunction.run re-binds every argument, rebuilds the specialization key
    and looks the kernel up on each call - probe482 measured 0.366 ms for the
    tile through the JIT against 0.317 through the cached CompiledKernel and
    0.301 through its raw .run() (the .run call also skips the stream lookup
    and the launch-metadata hook).  The key carries the constexprs and the
    16-byte alignment of every pointer, which is what the JIT specializes on,
    so a misaligned tensor compiles (and caches) its own entry.  Any failure
    falls back to the ordinary launch for the rest of the process.
    """
    ent = _KCACHE.get(key)
    if ent is None:
        ck = kern.warmup(*args, grid=grid)
        _KCACHE[key] = ck
        kern[grid](*args)
        return
    if ent is False or not _LAUNCH["ok"]:
        kern[grid](*args)
        return
    st = _LAUNCH["stream"]
    if st is None:
        from triton.runtime import driver

        st = driver.active.get_current_stream(driver.active.get_current_device())
        _LAUNCH["stream"] = st
    try:
        ent.run(
            grid[0], 1, 1, st, ent.function, ent.packed_metadata, None, None, None,
            *args,
        )
    except Exception:
        _LAUNCH["ok"] = False
        kern[grid](*args)


def _align(*ts):
    return tuple(t.data_ptr() & 15 == 0 for t in ts)


# wave45: ln_fast defines a local `_launch` closure for its wide-row branch,
# which shadows the module-level cached launcher inside that function; the
# tile branch uses this alias so the name resolution stays unambiguous.
_cached_launch = _launch


def ln_bwd_fast(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight=None,
    bias=None,
    output_mask=None,
    _checked=False,
):
    """Return (in_grad, weight_grad, bias_grad) like native_layer_norm_backward.

    wave44: the routed caller (`native_layer_norm.layer_norm_backward`) has
    already run lnb_fast_ok, which checks contiguity of all eight tensors, so
    it passes _checked=True and the six .contiguous() calls (0.021 ms measured)
    drop out of the hot path.
    """
    if _checked:
        g = grad_out
        x = input
        w = weight
        b = bias
    else:
        g = grad_out.contiguous()
        x = input.contiguous()
        w = weight.contiguous() if weight is not None else None
        b = bias.contiguous() if bias is not None else None
    n = _prod(normalized_shape)
    m = x.numel() // n
    if mean.dim() > 1:
        mean = mean.reshape(-1)
    if rstd.dim() > 1:
        rstd = rstd.reshape(-1)
    if not _checked:
        mean = mean.contiguous()
        rstd = rstd.contiguous()
    if output_mask is None:
        output_mask = (True, True, True)
    has_w = w is not None
    # wave44: the kernels address G/X/dX with p*BM*BN + arange, so only the
    # pointers matter - g.view(m,n) and friends cost 7 us each on this torch
    # build (probe489), and there are three of them on the hot path.
    g2 = g
    x2d = x
    in_grad = None
    fused_wb = False
    dW = None
    dB = None
    if output_mask[0]:
        in_grad = torch.empty_like(x)
        bn_t = triton.next_power_of_2(n)
        bm_t = (
            _dx_tile_bm(m, bn_t)
            if (
                os.environ.get("FLAGGEMS_SPACEMIT_LN_DX_TILE", "1") != "0"
                and n <= _LN_DX_TILE_MAX_N
                and n == bn_t
            )
            else 0
        )
        if bm_t:
            # wave38: narrow rows are both wrong (<32 lanes) and slow in the
            # per-row kernel; the tile is dtype agnostic.  When dW/dB are also
            # wanted the same pass emits the per-program column sums, which
            # saves a second read of g/x (probe399: (4096,4) 1.12 -> 0.76 ms).
            np_t = m // bm_t
            want_w = bool(output_mask[1]) and has_w
            want_b = bool(output_mask[2]) and b is not None
            if want_w or want_b:
                # wave43: dot-based reductions need 16 lanes in every dot
                # dimension, so only BN>=16 takes this body (4/8 lanes keep
                # the tl.sum tile).  LN_DOT=0 forces the old body, =2 selects
                # the full (16,BN) partial store + strided combine, which
                # probe469 measured 3-6% faster than the row-0 masked store.
                mode = os.environ.get("FLAGGEMS_SPACEMIT_LN_DOT", "1")
                dW = (
                    torch.empty_like(w)
                    if want_w
                    else torch.empty(n, dtype=x.dtype, device=x.device)
                )
                dB = (
                    torch.empty_like(b)
                    if want_b
                    else torch.empty(n, dtype=x.dtype, device=x.device)
                )
                if mode == "2" and bn_t >= 16:
                    pw = torch.empty(
                        np_t, 16, n, dtype=torch.float32, device=x.device
                    )
                    pb = torch.empty(
                        np_t, 16, n, dtype=torch.float32, device=x.device
                    )
                    with nested_dispatch(m * n):
                        _ln_bwd_narrow_dot16_kernel[(np_t,)](
                            g2,
                            x2d,
                            w if has_w else x2d,
                            mean,
                            rstd,
                            in_grad.view(m, n),
                            pw,
                            pb,
                            n,
                            BM=bm_t,
                            BN=bn_t,
                            HAS_W=has_w,
                        )
                        _ln_bwd_combine16_kernel[(1,)](
                            pw,
                            pb,
                            dW,
                            dB,
                            n,
                            NRB=np_t,
                            BLK=bn_t,
                        )
                else:
                    # wave44: one allocation for both column-partial buffers
                    # (torch.empty measures ~0.008 ms each) - part[1] is a
                    # contiguous view whose offset keeps 16-byte alignment.
                    part = torch.empty(
                        2, np_t, n, dtype=torch.float32, device=x.device
                    )
                    pw = part[0]
                    pb = part[1]
                    kern = (
                        _ln_bwd_narrow_dot_kernel
                        if (mode != "0" and bn_t >= 16)
                        else _ln_bwd_narrow_kernel
                    )
                    warg = w if has_w else x2d
                    out2d = in_grad.view(m, n)
                    targs = (
                        g2,
                        x2d,
                        warg,
                        mean,
                        rstd,
                        out2d,
                        pw,
                        pb,
                        n,
                        bm_t,
                        bn_t,
                        has_w,
                    )
                    with _FastNested(m * n):
                        _launch(
                            kern,
                            (
                                kern,
                                "tile",
                                n,
                                bm_t,
                                bn_t,
                                has_w,
                                x.dtype,
                                _align(g2, x2d, warg, mean, rstd, out2d, pw, pb),
                            ),
                            (np_t,),
                            targs,
                        )
                        # wave40 probe429: a plain launch that follows a nested
                        # one is not ordered against it, and the combine reads
                        # pw.  A nested launch inside the same block is
                        # (probe429: the (2048,32) f16 tile-then-plain-combine
                        # arrangement returned dw 4.5e2 on the first backward
                        # of a process, same-block nested 2.8e-2, four reps
                        # each).  Keep both launches under one dispatcher
                        # dialogue.
                        _launch(
                            _ln_bwd_combine_kernel,
                            (
                                _ln_bwd_combine_kernel,
                                "comb",
                                n,
                                np_t,
                                bn_t,
                                x.dtype,
                                _align(pw, pb, dW, dB),
                            ),
                            (1,),
                            (pw, pb, dW, dB, n, np_t, bn_t, True),
                        )
                if not want_w:
                    dW = None
                if not want_b:
                    dB = None
                fused_wb = True
            else:
                with nested_dispatch(m * n):
                    _ln_bwd_dx_tile_kernel[(np_t,)](
                        g2,
                        x2d,
                        w if has_w else x2d,
                        mean,
                        rstd,
                        in_grad.view(m, n),
                        n,
                        BM=bm_t,
                        BN=bn_t,
                        HAS_W=has_w,
                    )
        elif x.dtype == torch.float32:
            k = _pick_dx_k(m, n)
            blk = _dx_block(n)

            def _launch_dx():
                _ln_bwd_dx_kernel[(m // k,)](
                    g2,
                    x2d,
                    w if has_w else x2d,
                    mean,
                    rstd,
                    in_grad.view(m, n),
                    n,
                    K=k,
                    BLOCK=blk,
                    NOMASK=(n % blk == 0),
                    HAS_W=has_w,
                )

            if k >= 4:
                with nested_dispatch(m * n):
                    _launch_dx()
            else:
                _launch_dx()
        else:
            k16 = _pick_dx_k16(m, n)
            blk16 = _dx_block(n)

            def _launch_dx16():
                _ln_bwd_dx_kernel_16[(m // k16,)](
                    g2,
                    x2d,
                    w if has_w else x2d,
                    mean,
                    rstd,
                    in_grad.view(m, n),
                    n,
                    K=k16,
                    BLOCK=blk16,
                    NOMASK=(n % blk16 == 0),
                    HAS_W=has_w,
                )

            if k16 >= 4:
                with nested_dispatch(m * n):
                    _launch_dx16()
            else:
                _launch_dx16()
    if (output_mask[1] or output_mask[2]) and not fused_wb:
        # wave13 probe109 P5a: ROWS=32 wins for n>=2048 (2048^2 total
        # 3.95 vs 4.20; 512x2048 1.06 vs 1.07).
        if x.dtype == torch.float32:
            rows = 32 if n >= 2048 else _DWDB_ROWS
            cb = _DWDB_CB if n >= _DWDB_CB else triton.next_power_of_2(n)
        else:
            # probe320 wb scan: ROWS=64 wins at all five shapes; the column
            # block prefers n itself at 512 and about n/2 above (2048^2 1024
            # 2.14 vs 2048 2.87; 512x4096 2048 1.05 vs 1024 ~1.1).
            # wave34 probe345: above 8192 columns the block should keep
            # growing - (16,16384) CB 1024/2048/4096/8192 = 0.593/0.471/
            # 0.427/0.383 ms and (8,32768) 0.762/0.606/0.506/0.438 ms.
            rows = _DWDB_ROWS
            nn = triton.next_power_of_2(n)
            if n <= 512:
                cb = nn
            elif n >= 16384:
                cb = min(8192, nn // 2)
            else:
                cb = min(_DWDB_CB, nn // 2)
        nrb = triton.cdiv(m, rows)
        pw = torch.empty(nrb, n, dtype=torch.float32, device=x.device)
        pb = torch.empty(nrb, n, dtype=torch.float32, device=x.device)
        _ln_bwd_wb_kernel[(nrb, triton.cdiv(n, cb))](
            g2,
            x2d,
            mean,
            rstd,
            pw,
            pb,
            m,
            n,
            ROWS=rows,
            CB=cb,
            NOMASK=(n % cb == 0),
        )
        cb2 = cb
        dW = torch.empty_like(w) if output_mask[1] else pw[0]
        dB = torch.empty_like(b) if output_mask[2] else pb[0]
        _ln_bwd_combine_kernel[(triton.cdiv(n, cb2),)](
            pw,
            pb,
            dW,
            dB,
            n,
            NRB=nrb,
            CB2=cb2,
            NOMASK=(n % cb2 == 0),
        )
        if not output_mask[1]:
            dW = None
        if not output_mask[2]:
            dB = None
    return in_grad, dW, dB
