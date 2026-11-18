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


# ---------------- wave47: cached CompiledKernel launcher ----------------
# probe601/606: at the official (64,64) row the op spent 0.252 ms against
# torch's 0.127, and the kernel launched through the JIT alone was 0.116 ms
# while the same kernel through a cached CompiledKernel was 0.064 ms - the
# empty-launch floor on this board.  JITFunction.run re-binds every argument,
# rebuilds the specialization key and looks the kernel up on each call, which
# is 50 us of pure host tax on every launch.  Ported verbatim from the wave44
# layer_norm launcher; any failure falls back to the ordinary launch for the
# rest of the process.
_KCACHE = {}
_LAUNCH = {"ok": True, "stream": None}


def _align(*ts) -> tuple:
    return tuple(t.data_ptr() & 15 == 0 for t in ts)


def _launch(kern, key, grid, args):
    # wave49: the key must carry the kernel identity and the argument dtypes.
    # probe704 died with "double free or corruption (!prev)" at (4096,64) as
    # soon as the f16 call followed the f32 call (the key was ("tile", N, bm,
    # align) and the f16 launch reused the f32 CompiledKernel's runner), and
    # probe712 died with "corrupted double-linked list" when the masked tile and
    # the unmasked tile shared one key at the same (N, BM).  wave48 had papered
    # over the dtype half by disabling the cached path for the 33..255 band.
    key = (kern,) + key + (tuple(getattr(t, "dtype", None) for t in args),)
    ent = _KCACHE.get(key)
    if ent is None:
        ck = kern.warmup(*args, grid=grid)
        _KCACHE[key] = ck
        kern[grid](*args)
        return
    if ent is False or not _LAUNCH["ok"]:
        kern[grid](*args)
        return
    if len(grid) == 1:
        grid = (grid[0], 1, 1)
    try:
        # wave47: ent[grid] runs the CompiledKernel's own runner - it still
        # skips JITFunction.run's argument re-binding, specialization-key
        # rebuild and cache lookup.  Two traps measured on this build:
        #  - the runner indexes grid[1]/grid[2], so a 1-D grid raises
        #    IndexError; the fallback then costs more than the JIT (that alone
        #    made the official (64,64) row 0.278 -> 0.342 ms in the first A/B);
        #  - calling ent.run(...) positionally (the wave44 form) corrupts the
        #    heap as soon as a different dtype compiles afterwards.
        # probe609: jit 0.1095 / ent[grid] 0.0745 / raw .run 0.0616 ms on the
        # (64,64) kernel, so this form keeps most of the win without either.
        ent[grid](*args)
    except Exception:
        _LAUNCH["ok"] = False
        kern[grid](*args)


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


def _rms_tile_bm(m: int, bn: int = _RMS_TILE_N, f16: bool = False) -> int:
    """wave50 probe819: ~8K elements per program (16K on 16-bit), <= m//8 rows.

    The wave38 form was a flat 128 rows whenever M divided it, which leaves 4
    programs at M=512 (0.261 ms against 0.203 at BM=64) and, at N=128, 256 rows
    from the wave49 band picker's "ties to the larger BM" rule - that one costs
    1.41x at (2048,128) f32 (0.928 against 0.656) and 1.39x at (4096,128).
    Measured optima: N=64 f32 -> BM 128 at M>=1024, 64 at 512; N=128 f32 -> 64;
    16-bit prefers one size up (N=128 f16 128, N=64 f16 256).
    """
    elems = 16384 if f16 else 8192
    bm = min(elems // bn, max(32, m // 8), 256)
    bm = 1 << max(5, bm.bit_length() - 1)
    while bm > 32 and m % bm:
        bm //= 2
    return bm if m % bm == 0 and m >= 2 * bm else 0


# ---------------- wave39: narrow-row tile (N <= 32) ----------------
# probe423: these widths have no route at all today - the table starts at 64
# and the wildcard at 256 - so every N<=32 shape ran the generic kernel.  The
# wave38 tile is already generic over BN, so it only needed a BM pick and a
# gate.  Same-session minima, generic -> tile (f32 / f16):
#   (4096,4)  1.129 -> 0.227 (5.0x, torch 0.497)  / 1.274 -> 0.289 (4.4x)
#   (4096,8)  1.208 -> 0.373 (3.2x)               / 1.433 -> 0.487 (3.0x)
#   (2048,16) 0.844 -> 0.303 (2.8x)               / 0.937 -> 0.378 (2.5x)
#   (2048,32) 2.291 -> 0.495 (4.6x)               / 1.678 -> 0.656 (2.6x)
#   (256,32)  0.522 -> 0.133 (3.9x)               / 0.440 -> 0.151 (2.9x)
# BM follows the same rule as the layer_norm forward tile: ~8K elements, a
# 64-row floor and at least two programs (a single-program tile at (256,32)
# cost 40% and at (1024,16) 54%).
_RMS_TILE_MAX_N = 32

# ---------------- wave47: widen the tile to N = 128 ----------------
# probe602: N in 65..255 has NO route at all (the table starts at 64 with the
# wildcard floor at 256), so those widths run the generic FlagGems kernel.  At
# (2048,128) f32 that is 2.8-4.1 ms against 0.66-0.96 ms for the wave38 tile
# (BM = 8192/BN = 64), i.e. 3-4x, and f16 the same.  128 is the only power of
# two in that hole, so only it can use the unmasked tile.
#
# ---------------- wave48: masked tile fills the whole 33..255 band ----------
# probe610/612: the unmasked tile cannot serve a non-power-of-two width, so 33..63
# and every non-pow2 width up to 255 still ran the generic kernel.  A masked
# variant (BN = next_pow2(N)) wins there, same-session vendor -> tile medians:
#   (4096,33) 3.38 -> 0.68 (5.0x), (2048,48) 1.80 -> 0.42 (4.3x),
#   (2048,63) 1.91 -> 0.44 (4.3x), (2048,96) 2.59 -> 0.69 (3.8x),
#   (2048,160) 4.10 -> 1.21 (3.4x), (2048,255) 4.85 -> 1.30 (3.7x),
#   (4096,255) 9.35 -> 2.52 (3.7x), f16 tracks f32 within 5%.
# BM rule from the sweep: one program per core (M//8 rows), at most 32K
# elements and at most 256 rows; the picker still needs BM to divide M and at
# least two programs.
_RMS_BAND_MAX_N = 255


def _rms_narrow_bm(m: int, bn: int) -> int:
    bm = min(1024, max(64, 8192 // bn), max(64, m // 2))
    # wave49: this used to return m//2 verbatim, and _rms_tile needs BM*BN to be
    # a power of two - (1000,8) and (1000,16) therefore died with "arange's range
    # must be a power of 2" (probe710).  Snap down first.
    bm = 1 << (bm.bit_length() - 1)
    while bm > 8 and m % bm:
        bm //= 2
    return bm if m % bm == 0 else 0


def _rms_band_bm(m: int, n: int) -> int:
    """Picker for the masked 33..255 tile (wave48)."""
    bm = 8
    cap = max(16, m // 8)
    while bm * 2 <= 256 and bm * 2 <= cap and (bm * 2) * n <= 32768:
        bm *= 2
    while bm > 8 and m % bm:
        bm //= 2
    return bm if m % bm == 0 and m >= 2 * bm else 0


def rms_band_rowmask_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_RMS_BAND_ROWMASK", "1") != "0"


def _band_cache_enabled() -> bool:
    # wave49: default OFF.  The dtype/kern/M key fixes (probe704/707/712) remove
    # the deterministic wrong-runner crashes, but a 20-shape x 3-dtype battery in
    # one process still hits an occasional glibc heap assertion with the cached
    # band launches, and it hits it with plain JIT too - this build's compiler is
    # simply unstable when many variants compile in one process.  The cached
    # launch is worth 1-10% here, not worth the instability; probe7xx can turn it
    # back on to re-measure.
    return os.environ.get("FLAGGEMS_SPACEMIT_RMS_BAND_CACHE", "0") != "0"


def _rms_band_bm2(m: int, n: int) -> int:
    """Wave-row-masked picker (wave49).

    probe701b/702/703: the wave48 picker demanded BM | M, so 1000/3000-row
    tensors fell to BM=8 and 1023/777/4095-row tensors had no route at all.
    With a row mask the cost model is waves x tile rows: the kernel spends
    ceil(M/(8*BM)) waves of BM rows, so the picker minimises that product over
    BM in {32..256} with BM*n <= 32768 and at least two programs, breaking ties
    towards the larger BM (fewer launches).  Measured medians (f32, N=160):
    M=256 32 -> 0.275 (64: 0.400, 128: 0.632); M=512 64 -> 0.403 (32: 0.416,
    128: 0.650); M=768 32 -> 0.557 (64: 0.669, 128: 0.658); M=1024 128 -> 0.661
    (64: 0.671, 32: 0.699); M=2048 128 -> 1.205 (64: 1.246, 32: 1.300).
    """
    best = None
    bm = 32
    while bm <= 256:
        if bm * n <= 32768 and m >= 2 * bm:
            cost = ((m + 8 * bm - 1) // (8 * bm)) * bm
            key = (cost, -bm)
            if best is None or key < best[0]:
                best = (key, bm)
        bm *= 2
    return best[1] if best else 0


@triton.jit
def _rms_tile_m(X, W, Y, INV, N, eps, BM: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0)
    rows = p * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    m = cols < N
    off = rows[:, None] * N + cols[None, :]
    x = tl.load(X + off, mask=m[None, :], other=0.0).to(tl.float32)
    s = tl.sum(x * x, axis=1) / N + eps
    rr = 1.0 / tl.sqrt(s)
    tl.store(INV + rows, rr)
    w = tl.load(W + cols, mask=m, other=1.0).to(tl.float32)[None, :]
    tl.store(Y + off, (x * rr[:, None] * w).to(Y.dtype.element_ty),
             mask=m[None, :])


@triton.jit
def _rms_tile_mm(X, W, Y, INV, M, N, eps, BM: tl.constexpr, BN: tl.constexpr):
    """wave49: masked tile with a row mask, so BM no longer has to divide M.

    probe701b: 1000/3000-row tensors were pinned at BM=8 by the divisibility
    rule; with rows < M masked out, BM=128 cuts (1000,64) 0.689 -> 0.284 and
    (3000,255) 2.984 -> 1.882 ms.  Accuracy is unchanged (max abs vs float64
    reference 1e-6 f32 / 2e-3 f16, i.e. dtype ulp).
    """
    p = tl.program_id(0)
    rows = p * BM + tl.arange(0, BM)
    rm = rows < M
    cols = tl.arange(0, BN)
    cm = cols < N
    off = rows[:, None] * N + cols[None, :]
    mask = rm[:, None] & cm[None, :]
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x * x, axis=1) / N + eps
    rr = 1.0 / tl.sqrt(s)
    tl.store(INV + rows, rr, mask=rm)
    w = tl.load(W + cols, mask=cm, other=1.0).to(tl.float32)[None, :]
    tl.store(Y + off, (x * rr[:, None] * w).to(Y.dtype.element_ty), mask=mask)


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


def _rms_band_pick(m: int, n: int, f16: bool = False):
    """(BM, kernel, BN) for the 33..255 band, or (0, None, 0) (wave49/50)."""
    if n == triton.next_power_of_2(n):
        # wave50 probe819: powers of two take the element-count tile whenever
        # that BM divides M - at N=128 the wave49 "larger BM" tie-break was
        # 1.4x slower than BM=64 on f32.
        bm2 = _rms_tile_bm(m, n, f16)
        if bm2:
            return bm2, _rms_tile, n
    if rms_band_rowmask_enabled() and m >= 256:
        bm = _rms_band_bm2(m, n)
        if bm:
            return bm, _rms_tile_mm, triton.next_power_of_2(n)
    if n == triton.next_power_of_2(n):
        bm = _rms_narrow_bm(m, n)
        return (bm, _rms_tile, n) if bm else (0, None, 0)
    if m >= 256:
        bm = _rms_band_bm(m, n)
        if bm:
            return bm, _rms_tile_m, triton.next_power_of_2(n)
    return 0, None, 0


def rms_fast_route(M: int, N: int, f16: bool = False) -> bool:
    """True when rms_fast_launch would serve this shape (no allocation)."""
    if N <= _RMS_TILE_MAX_N and N == triton.next_power_of_2(N) and M >= 8:
        bm0 = _rms_narrow_bm(M, N)
        if bm0:
            return True
        if rms_band_rowmask_enabled() and M >= 256 and _rms_band_bm2(M, N):
            return True
    if N == _RMS_TILE_N and M >= _RMS_TILE_MIN_M:
        if _rms_tile_bm(M, N, f16) >= 32:
            return True
    # wave48: the masked tile covers every width in 33..255; only the two
    # powers of two in the band keep the unmasked kernel.  probe619: below 256
    # rows the masked tile is a 0.94-1.06x tie against the generic kernel (it
    # is launch bound), so the masked half starts at M=256 where it is 1.0-1.45x.
    # wave49: _rms_band_pick adds the row-masked kernel above 256 rows, which
    # lifts the BM|M constraint (and with it every odd row count).
    if _RMS_TILE_MAX_N < N <= _RMS_BAND_MAX_N and M >= 8:
        bm, _, _ = _rms_band_pick(M, N, f16)
        if bm and M >= 2 * bm:
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
    al = _align(x, w, y, inv)
    if N <= _RMS_TILE_MAX_N and N == triton.next_power_of_2(N) and M >= 8:
        # wave49: above 256 rows the row-masked picker takes over, because the
        # narrow rule caps BM at m//2 ((1000,8)/(1000,16) ran 2 or 125 programs,
        # and the old m//2 form did not even compile when it was not a power of
        # two).  Below 256 rows the shape is launch bound and stays as it was.
        kern, bn, bm = _rms_tile, N, _rms_narrow_bm(M, N)
        if rms_band_rowmask_enabled() and M >= 256 and (not bm or bm > 256):
            # probe709: for N<=32 the masked tile is 10-15% slower than the
            # narrow kernel while the narrow pick still has a sensible program
            # count, so the row mask only takes the shapes it alone can serve -
            # no route at all (777/1023 rows) or a >256-row program (2048,16).
            bm2 = _rms_band_bm2(M, N)
            if bm2:
                if M % bm2 == 0:
                    bm = bm2
                else:
                    bm, kern, bn = bm2, _rms_tile_mm, N
        if bm:
            with nested_dispatch(M * N):
                if kern is _rms_tile_mm:
                    # M is a runtime arg, but Triton specialises integer args on
                    # divisibility by 16, so it has to be in the key too: an
                    # M%16==0 compile reused for M=1000 is a silent overrun.
                    _launch(kern, ("tilemm", N, M, bm, al), (triton.cdiv(M, bm),),
                            (x, w, y, inv, M, N, eps, bm, bn))
                else:
                    _launch(kern, ("tile", N, bm, al), (M // bm,),
                            (x, w, y, inv, N, eps, bm, bn))
            return True
    if N == _RMS_TILE_N and M >= _RMS_TILE_MIN_M:
        bm = _rms_tile_bm(M, N, f16)
        if bm >= 32:
            # wave38: one pass over the tile; nesting is what makes it win
            # above 32K elements (probe402).
            with nested_dispatch(M * N):
                _launch(_rms_tile, ("tile", N, bm, al), (M // bm,),
                        (x, w, y, inv, N, eps, bm, N))
            return True
    if _RMS_TILE_MAX_N < N <= _RMS_BAND_MAX_N and M >= 8:
        # wave49: the pick now also covers the row-masked kernel, so odd row
        # counts (1000/3000/4095) keep a 32..256-row tile instead of BM=8 or
        # the generic kernel.  Powers of two stay on the unmasked kernel when
        # BM divides M.
        bm, kern, bn = _rms_band_pick(M, N, f16)
        # wave47: never a single-program tile (wave38: 40% at (256,32)).  The
        # official (64,64) row is exactly that case - bm lands on M and one
        # program serves 4096 elements, measured 0.286 -> 0.300 ms.
        if bm and M < 2 * bm:
            bm = 0
        if bm:
            with nested_dispatch(M * N):
                if kern is _rms_tile_mm:
                    key = ("tilemm", N, M, bm, al)
                    args = (x, w, y, inv, M, N, eps, bm, bn)
                    grid = (triton.cdiv(M, bm),)
                else:
                    key = ("tile", N, bm, al)
                    args = (x, w, y, inv, N, eps, bm, bn)
                    grid = (M // bm,)
                # wave49: the ~50 us JIT re-binding tax is back on the menu for
                # this band now that _launch keys on the argument dtypes - the
                # wave48 heap corruption was the stale f32 runner being fed f16
                # pointers, not the cached launch itself (probe707 survives a
                # f32->f16->bf16->f32 cycle at four shapes).
                if _band_cache_enabled():
                    _launch(kern, key, grid, args)
                else:
                    kern[grid](*args)
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
        _launch(row, ("row", N, tile, f16, al), (M,),
                (x, w, y, inv, N, eps, tile))
        return True
    if M % k0 == 0:
        kern = _rms_mmk16 if f16 else _rms_mmk
        grid = M // k0

        def _launch_one():
            _launch(kern, ("mmk", N, k0, tile, f16, al), (grid,),
                    (x, w, y, inv, N, eps, k0, tile))
    else:
        # wave13: masked multirow keeps K rows/program for odd M instead of
        # halving K down to 1 (probe110: 2.1-2.5x on odd-M shapes).
        kern = _rms_mmk16_m if f16 else _rms_mmk_m
        grid = triton.cdiv(M, k0)

        def _launch_one():
            _launch(kern, ("mmkm", N, k0, tile, f16, al, M), (grid,),
                    (x, w, y, inv, M, N, eps, k0, tile))
    if k0 >= 4:
        with nested_dispatch(M * N):
            _launch_one()
    else:
        _launch_one()
    return True
