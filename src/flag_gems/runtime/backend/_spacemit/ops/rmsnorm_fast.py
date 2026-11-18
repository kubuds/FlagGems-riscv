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

"""wave6 (campaign 1015am): rms_norm forward direct kernels.

Structure ported from the layernorm_fast playbook (wave4): per-row
loop-carried mul-chain acc for sum(x^2) (elementwise feeding tl.sum would kill
the reduction) + a separate apply loop. Narrow-column rows additionally get
K rows/program (wave5 launch-amortization lesson).

wave11 (campaign 1016night): nest + per-shape K/TILE picks + f16 coverage.
- f32 routes now launch under nested_dispatch(M*N) for K>=4. Same-session
  min-of-9 (probe100/102): 512^2 0.579 -> 0.380 (K16 nest; torch 0.815);
  2048^2 5.19 -> 3.32 (K4 T256 nest); 512x4096 3.29 -> 1.28 (K4 T512 nest);
  8192x512 8.06 -> 4.31 (K16 nest, probe100).
- f16 forward now covered by _rms_row16/_rms_mmk16: loads cast to f32 INSIDE
  the loop-carried acc chain (wave10 lesson: an f16 cast feeding tl.sum is the
  feeding cliff), y stored back as x.dtype, inv in f32. Same-session f16
  numbers: 512^2 0.418 / 8192x512 4.25-5.14 / 2048^2 2.89 / 512x4096 1.04 ms
  (nest picks; legacy f16 path was 4.45-50.9 ms).
Accuracy vs f32 reference: maxabs 1.9e-6 (f32); f16 within ~2 ulp of oracle.

Backward is untouched (legacy kernels). Non-covered dtype / non-contiguous /
N not in the route table fall back to the generic forward.

Kill switch: FLAGGEMS_SPACEMIT_RMS_FAST=0.
"""

import os

import triton
import triton.language as tl

from ._dispatch import nested_dispatch


def rms_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_RMS_FAST", "1") != "0"


# ------------------- wave38: single-pass tile for N=64 -------------------
# probe401/402: the routed kernel reads a row twice (sum chain, then apply).
# At 64 columns the whole tile fits in registers, so one pass suffices: load x,
# tl.sum(x*x, axis=1), then y from the same registers.  Plain-launch numbers
# were a wash at 64 rows and only 1.1-1.4x above 256 rows, but the tile wants
# the nested dispatcher much more than the routed kernel does - at (4096,64)
# nesting takes it from 2.29 to 0.74 ms f16 (3.09x) and 1.90 to 0.86 f32
# (2.22x), against 3.03 / ~3.3 for the routed and generic paths.  At the
# official (64,64) core shape the element count is under the nested floor and
# the two forms tie, so the tile is gated to M >= 256.
_RMS_TILE_N = 64
_RMS_TILE_MIN_M = 256


def _rms_tile_bm(m: int) -> int:
    bm = 128
    while bm > 32 and m % bm:
        bm //= 2
    return bm if m % bm == 0 else 0


@triton.jit
def _rms_tile(X, W, Y, INV, N, eps, BM: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0)
    idx = p * BM * BN + tl.arange(0, BM * BN)
    x = tl.reshape(tl.load(X + idx).to(tl.float32), (BM, BN))
    s = tl.sum(x * x, axis=1) / N + eps
    rr = 1.0 / tl.sqrt(s)
    rows = p * BM + tl.arange(0, BM)
    tl.store(INV + rows, rr)
    cols = tl.arange(0, BN)
    w = tl.load(W + cols).to(tl.float32)[None, :]
    y = x * rr[:, None] * w
    tl.store(Y + idx, tl.reshape(y, (BM * BN,)).to(Y.dtype.element_ty))


@triton.jit
def _rms_row(X, W, Y, INV, N, eps, TILE: tl.constexpr):
    row = tl.program_id(0)
    base = X + row * N
    acc = tl.zeros((TILE,), dtype=tl.float32)
    num = tl.cdiv(N, TILE)
    for c in range(0, num):
        offs = c * TILE + tl.arange(0, TILE)
        xv = tl.load(base + offs, mask=offs < N, other=0.0)
        acc += xv * xv
    rr = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
    tl.store(INV + row, rr)
    for c in range(0, num):
        offs = c * TILE + tl.arange(0, TILE)
        m = offs < N
        xv = tl.load(base + offs, mask=m, other=0.0)
        wv = tl.load(W + offs, mask=m, other=0.0)
        tl.store(Y + row * N + offs, xv * rr * wv, mask=m)


@triton.jit
def _rms_row16(X, W, Y, INV, N, eps, TILE: tl.constexpr):
    row = tl.program_id(0)
    base = X + row * N
    acc = tl.zeros((TILE,), dtype=tl.float32)
    num = tl.cdiv(N, TILE)
    for c in range(0, num):
        offs = c * TILE + tl.arange(0, TILE)
        xv = tl.load(base + offs, mask=offs < N, other=0.0).to(tl.float32)
        acc += xv * xv
    rr = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
    tl.store(INV + row, rr)
    for c in range(0, num):
        offs = c * TILE + tl.arange(0, TILE)
        m = offs < N
        xv = tl.load(base + offs, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(
            Y + row * N + offs, (xv * rr * wv).to(Y.dtype.element_ty), mask=m
        )


@triton.jit
def _rms_mmk(X, W, Y, INV, N, eps, K: tl.constexpr, TILE: tl.constexpr):
    p = tl.program_id(0)
    num = tl.cdiv(N, TILE)
    for i in tl.static_range(K):
        row = p * K + i
        base = X + row * N
        acc = tl.zeros((TILE,), dtype=tl.float32)
        for c in range(0, num):
            offs = c * TILE + tl.arange(0, TILE)
            xv = tl.load(base + offs, mask=offs < N, other=0.0)
            acc += xv * xv
        rr = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
        tl.store(INV + row, rr)
        for c in range(0, num):
            offs = c * TILE + tl.arange(0, TILE)
            m = offs < N
            xv = tl.load(base + offs, mask=m, other=0.0)
            wv = tl.load(W + offs, mask=m, other=0.0)
            tl.store(Y + row * N + offs, xv * rr * wv, mask=m)


@triton.jit
def _rms_mmk16(X, W, Y, INV, N, eps, K: tl.constexpr, TILE: tl.constexpr):
    p = tl.program_id(0)
    num = tl.cdiv(N, TILE)
    for i in tl.static_range(K):
        row = p * K + i
        base = X + row * N
        acc = tl.zeros((TILE,), dtype=tl.float32)
        for c in range(0, num):
            offs = c * TILE + tl.arange(0, TILE)
            xv = tl.load(base + offs, mask=offs < N, other=0.0).to(tl.float32)
            acc += xv * xv
        rr = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
        tl.store(INV + row, rr)
        for c in range(0, num):
            offs = c * TILE + tl.arange(0, TILE)
            m = offs < N
            xv = tl.load(base + offs, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
            tl.store(
                Y + row * N + offs, (xv * rr * wv).to(Y.dtype.element_ty), mask=m
            )


@triton.jit
def _rms_mmk_m(X, W, Y, INV, M, N, eps, K: tl.constexpr, TILE: tl.constexpr):
    """wave13: masked multirow - odd M keeps K rows/program (probe110).

    1001x512 1.38 -> 0.55 ms, 1001x2048 2.64 -> 1.28, 513x4096 2.33 -> 1.17
    (same-session f32; bulk+tail two-launch variant measured slower).
    """
    p = tl.program_id(0)
    num = tl.cdiv(N, TILE)
    for i in tl.static_range(K):
        row = p * K + i
        if row < M:
            base = X + row * N
            acc = tl.zeros((TILE,), dtype=tl.float32)
            for c in range(0, num):
                offs = c * TILE + tl.arange(0, TILE)
                xv = tl.load(base + offs, mask=offs < N, other=0.0)
                acc += xv * xv
            rr = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
            tl.store(INV + row, rr)
            for c in range(0, num):
                offs = c * TILE + tl.arange(0, TILE)
                m = offs < N
                xv = tl.load(base + offs, mask=m, other=0.0)
                wv = tl.load(W + offs, mask=m, other=0.0)
                tl.store(Y + row * N + offs, xv * rr * wv, mask=m)


@triton.jit
def _rms_mmk16_m(X, W, Y, INV, M, N, eps, K: tl.constexpr, TILE: tl.constexpr):
    p = tl.program_id(0)
    num = tl.cdiv(N, TILE)
    for i in tl.static_range(K):
        row = p * K + i
        if row < M:
            base = X + row * N
            acc = tl.zeros((TILE,), dtype=tl.float32)
            for c in range(0, num):
                offs = c * TILE + tl.arange(0, TILE)
                xv = tl.load(base + offs, mask=offs < N, other=0.0).to(tl.float32)
                acc += xv * xv
            rr = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
            tl.store(INV + row, rr)
            for c in range(0, num):
                offs = c * TILE + tl.arange(0, TILE)
                m = offs < N
                xv = tl.load(base + offs, mask=m, other=0.0).to(tl.float32)
                wv = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
                tl.store(
                    Y + row * N + offs,
                    (xv * rr * wv).to(Y.dtype.element_ty),
                    mask=m,
                )


# N -> (K, TILE) mmk fast path; only verified shapes are wired.
# wave11 picks (probe100/102, min-of-9 same-session):
#   f32: 512 -> K16 T256 nest (0.380); 2048 -> K4 T256 nest (3.32);
#        4096 -> K4 T512 nest (1.28)
#   f16: 512 -> K16 T256 nest (4.25 @8192x512); 2048 -> K4 T512 nest (2.89);
#        4096 -> K4 T512 nest (1.04)
# wave12 (probe104b/105, interleaved same-session): f32 K raised per route —
# 512->K32 T256 (0.305/2.049/4.324 @512^2/4096x512/8192x512), 2048->K16 T256
# (0.636/1.238/4.677 @512/1024/4096 rows), 4096->K8 T512 (1.091 @512x4096).
_ROUTES_F32 = {
    512: (32, 256),
    2048: (16, 256),
    4096: (8, 512),
    # wave33 probe334: official benchmark N=8192 has no route, so every row
    # count ran the generic kernel. K8/T512 f32: 64 rows 0.330, 256 rows
    # 0.989, 1024 rows 4.090 ms (K4 3-11% slower). Same pair for bf16.
    8192: (8, 512),
    # wave33 probe337: campaign shape 4096x1024 (N=1024) had no route either.
    # K32/T256 f32 3.339 ms @4096 rows, 0.851 @1024 rows (K16 5% slower).
    1024: (32, 256),
    # wave33 probe340: official benchmark shapes (1,16384) and (64,16384).
    # K8/T512 f32 0.521 ms @64 rows, 1.901 @256 rows (K4 7-10% slower).
    16384: (8, 512),
    # wave34 probe342: (4096,256) is the official accuracy shape that had no
    # route; K32/T256 1.834 ms (K16 2.125, K8 2.813) at 4096 rows and 6.971 ms
    # at 16384 rows.
    256: (32, 256),
    # wave34 probe343: N=32768 K8/T512 0.615 ms @64 rows f16 / 0.966 f32
    # (K2 1.676 / 2.021); at 1 and 8 rows every K ties (0.13 / 0.28 ms).
    32768: (8, 512),
    # wave34 probe354: the official core shape (1024,65536) had no route -
    # K8/T512 17.203 ms f16 (K2 50.3) and 32.787 f32 (K4 33.7); 1.151 at 64
    # rows, 0.180 at 1 row (all K tie there).
    65536: (8, 512),
    # wave35 probe358: official core shape (64,64) had no route (N=64 is under
    # the wildcard floor); K4/T64 0.111 ms against 0.121 common at 64 rows
    # (T128 ties at 0.113).
    64: (4, 64),
}
_ROUTES_F16 = {
    512: (16, 256),
    2048: (4, 512),
    4096: (4, 512),
    # wave33 probe332 (16-bit path): K8/T512 wins at 1/64/256/1024 rows
    # (0.094 / 0.263 / 0.767 / 2.864 ms f16; bf16 within 1%) and at 16384
    # columns (64 rows 0.397 ms) - T1024 and K4/T2 all slower.
    8192: (8, 512),
    # wave33 probe337: N=1024 K16/T256 2.885 ms @4096 rows, 0.830 @1024 rows
    # (K8 24% slower, K4/T512 another 41%).
    1024: (16, 256),
    # wave33 probe340: N=16384 K8/T512 0.407 ms @64 rows, 1.234 @256 rows.
    16384: (8, 512),
    # wave34 probe342: N=256 K32/T256 1.722 ms f16 / 1.911 bf16 @4096 rows
    # (K16 1.99 / 1.97, K8 2.76 / 2.94); 6.53 ms at 16384 rows.
    256: (32, 256),
    # wave34 probe343: N=32768 K8/T512 0.615 ms @64 rows f16 (K2 1.676);
    # at 8 rows K2 0.279 vs K8 0.281 (tie).
    32768: (8, 512),
    # wave34 probe354: N=65536 K8/T512 17.203 ms at 1024 rows, 1.151 at 64,
    # 0.180 at 1 (K4/T512 ties at 64 rows, K2 is 3x slower).
    65536: (8, 512),
    # wave35 probe358: N=64 f16 K4/T128 0.115 ms against 0.223 common at 64
    # rows and 0.221 against 0.539 at 256 rows (T64 0.138).
    64: (4, 128),
}


def _pick_k(m: int, k0: int) -> int:
    k = k0
    while k > 1 and m % k:
        k //= 2
    return k


# wave34 probe347: any N missing from the tables now takes one masked route
# instead of the generic kernel. K16/T256 won at every odd width measured -
# (4096,4098) f16 7.383 vs 81.8 generic (11.1x), (2048,3000) 2.879 vs 32.1
# (11.1x), (2048,6000) 3.984 vs 61.4 (15.4x), (1024,12288) 3.717 vs 63.7
# (17.1x); f32 1.9-3.9x. Gated to the measured range and to M >= 32 so the
# multirow kernel is used (the row kernel was not measured for odd N).
_WILDCARD = (16, 256)
# wave37 probe376/377 (interleaved 3 rounds, use_gems): the f32 N=64 entry is
# only a win while the op is launch bound - 384 rows 0.441 vs 0.584 generic
# (1.32x), 512 rows a tie (0.682 vs 0.696), then the multirow kernel loses at
# 768 (0.977 vs 0.919), 1024 (1.208 vs 1.143) and 4096 (3.579 vs 3.314).  Cap
# the route by rows instead of dropping it.
_RMS_F32_MAX_M = {64: 512}
_WILD_MIN_N = 256
_WILD_MAX_N = 1 << 14
_WILD_MIN_M = 32


def _wildcard_entry(N: int, f16: bool):
    # wave35 probe359: above 8192 columns a wider tile wins on odd widths -
    # N=12288 f16 21.84 (K16/T256) -> 20.37 (K8/T1024) and f32 25.85 -> 23.51
    # (K4/T512); N=10240 agrees (9.37 -> 8.76 f16, 10.14 -> 9.78 f32).
    if N >= 8192:
        return (8, 1024) if f16 else (4, 512)
    # wave37 probe381: the band just above the 4096 route was still on T256;
    # T512 wins there at 1024 rows - N=4098 f16 2.026 vs 2.096 and f32 3.016
    # vs 3.129 (audit number), N=6000 f16 2.386 vs 2.388 and f32 3.207 vs
    # 3.253.  K8/T512 is within 1.4% of the best pair at both widths and both
    # dtypes, so one entry covers the band.
    if N > 4096:
        return (8, 512)
    return _WILDCARD


def rms_fast_route(M: int, N: int, f16: bool = False) -> bool:
    """True when rms_fast_launch would serve this shape (no allocation)."""
    if N == _RMS_TILE_N and M >= _RMS_TILE_MIN_M and _rms_tile_bm(M):
        return True
    routes = _ROUTES_F16 if f16 else _ROUTES_F32
    cap = None if f16 else _RMS_F32_MAX_M.get(N)
    if cap is not None and M > cap:
        return False
    if routes.get(N) is not None:
        return True
    return _WILD_MIN_N <= N <= _WILD_MAX_N and M >= _WILD_MIN_M


def rms_fast_launch(x, w, y, inv, M, N, eps, f16: bool = False) -> bool:
    """Launch the fast forward kernel; False = caller must fall back."""
    if N == _RMS_TILE_N and M >= _RMS_TILE_MIN_M:
        bm = _rms_tile_bm(M)
        if bm:
            # wave38: one pass over the tile; nesting is what makes it win
            # above 32K elements (probe402).
            with nested_dispatch(M * N):
                _rms_tile[(M // bm,)](x, w, y, inv, N, eps, BM=bm, BN=N)
            return True
    routes = _ROUTES_F16 if f16 else _ROUTES_F32
    cap = None if f16 else _RMS_F32_MAX_M.get(N)
    if cap is not None and M > cap:
        return False
    ent = routes.get(N)
    if ent is None:
        if _WILD_MIN_N <= N <= _WILD_MAX_N and M >= _WILD_MIN_M:
            ent = _wildcard_entry(N, f16)
        else:
            return False
    k0, tile = ent
    if M < 2 * k0:
        row = _rms_row16 if f16 else _rms_row
        row[(M,)](x, w, y, inv, N, eps, tile)
        return True
    if M % k0 == 0:
        kern = _rms_mmk16 if f16 else _rms_mmk
        grid = M // k0

        def _launch():
            kern[(grid,)](x, w, y, inv, N, eps, k0, tile)
    else:
        # wave13: masked multirow keeps K rows/program for odd M instead of
        # halving K down to 1 (probe110: 2.1-2.5x on odd-M shapes).
        kern = _rms_mmk16_m if f16 else _rms_mmk_m
        grid = triton.cdiv(M, k0)

        def _launch():
            kern[(grid,)](x, w, y, inv, M, N, eps, k0, tile)
    if k0 >= 4:
        with nested_dispatch(M * N):
            _launch()
    else:
        _launch()
    return True
