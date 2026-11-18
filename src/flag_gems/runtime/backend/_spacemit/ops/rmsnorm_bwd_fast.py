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

"""wave9/wave10 (campaign 1016am/pm): rms_norm backward direct kernels.

dx: two-loop per-row kernel - pass1 loop-carried acc for
    c = sum(x*(dy*w)); pass2 apply (dx = r*(dy*w) - x*r^3*c/N); K rows/program.
    Same-session f32 (probe82/84; legacy = single-block per-row kernel):
        (512,512)   2.62 -> 0.46 ms  (K=4, TILE=256)
        (2048,2048) 33.7 -> 4.39 ms  (K=4, TILE=256)
        (512,4096)  16.6 -> 2.08 ms  (K=2, TILE=512)
        (8192,512)  46.3 -> 5.94 ms  (K=4, TILE=256)
dw: row-fold (grid (cdiv(N,BN), cdiv(M,RB)), pure acc, no tl.sum) + combine
    kernel. Legacy 2D-block (tl.sum over rows) + torch.sum combine:
        (512,512)   0.78 -> 0.40 + 0.08
        (2048,2048) 14.7 -> 2.34 + 0.15
        (512,4096)  6.02 -> 1.19 + 0.10
        (8192,512)  10.1 -> 3.80 + 0.15
Accuracy vs torch autograd.grad: dx <= ~7.5e-5, dw <= ~1.4e-3 (same order as
legacy; dw is an M-sum with large magnitude). Nested dispatch hurts these
kernels (probe82) - default dispatch only.

wave10 (1016pm): f16 support. Loads are cast to f32 inside the kernels (the
casts live in loop-carried accumulate chains, so no feeding cliff); dx/dw
are stored back in the input dtype. Same-session full-bwd (probe91b):
rebuild 1.06/6.84/10.98 ms @512^2/2048^2/8192x512 vs legacy 4.35/49.6/49.4
(4.1-7.3x) and vs torch 5.47/81.2/88.1 (5.2-11.9x); err dx 3.9e-3,
dw 1.6e-2 (f16 rounding, same order as legacy).

Coverage: f32/f16 + contiguous + N in {512, 2048, 4096} (+ K-divisibility
fallback chain); anything else returns None -> caller falls back to the
generic kernels.

Kill switch: FLAGGEMS_SPACEMIT_RMSB_FAST=0.
"""

import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from ._dispatch import nested_dispatch

_DX_ROUTES = {
    512: (4, 256),
    2048: (4, 256),
    4096: (2, 512),
    # wave33 probe333: dx is flat over K at N=8192 (K4/T512 1.134/3.391/12.06 ms
    # at 64/256/1024 rows, K8 within 3%); the dw column block matters much more
    # at 1024 rows (BN 512/1024/2048 = 12.45/11.06/9.92 ms), so BN=2048.
    8192: (4, 512),
    # wave33 probe337: N=1024 K4/T256 8.635 ms @4096 rows, 2.435 @1024 rows
    # (K8 ties, K2/T512 45% slower).
    1024: (4, 256),
    # wave33 probe340: N=16384 K2/T512 1.763 ms @64 rows, 5.331 @256 rows.
    16384: (2, 512),
    # wave34 probe347: (4096,256) is the official accuracy shape and had no
    # backward route; K8/T256 3.735 ms f16 / 3.684 f32 (K4 3.77/4.79).
    256: (8, 256),
    # wave34 probe343: N=32768 K2/T512 3.217 ms @64 rows f16 (K1 3.135, K4
    # 3.192 - flat); at 8 rows K4 0.726, K2 0.805, K1 0.751.
    32768: (2, 512),
    # wave34 probe354: N=65536 K is flat (K1 76.99, K2 79.56, K4 78.23 ms at
    # 1024 rows f16); dw BN 8192 is best (74.99 vs 80.14 at BN 2048).
    65536: (2, 512),
}
_DW_BN = {
    512: 512,
    1024: 1024,
    2048: 512,
    4096: 1024,
    8192: 2048,
    16384: 2048,
    # wave34 probe347: N=256 BN 256 beats 512 (3.735 vs 4.19 ms f16).
    256: 256,
    # wave34 probe343: at 64 rows BN 8192 is 3% ahead of 2048 (3.113 vs 3.218
    # for the whole backward).
    32768: 8192,
    # wave34 probe354: N=65536 BN 8192 74.99 ms (BN 4096 76.11, 2048 80.14).
    65536: 8192,
}
_DW_RB = 64

# wave34 probe351: any N missing from the tables takes one masked route instead
# of the generic kernels - (4096,4098) 147.5 -> 47.8 ms, (2048,3000) 76.2 ->
# 15.3 ms (K4/T512, dw BN 1024; K2/K8/K16 and BN 256/512/2048 are 3-10%
# worse). Same bounds as the forward wildcard: measured widths, M >= 32.
_WILD_DX = (4, 512)
_WILD_BN = 1024
_WILD_MIN_N = 256
_WILD_MAX_N = 1 << 14
_WILD_MIN_M = 32


def _routes(n: int, m: int, is16: bool = False):
    """(dx K, dx TILE, dw BN) or None when the shape must fall back."""
    ent = _DX_ROUTES.get(n)
    if ent is not None:
        return ent[0], ent[1], _DW_BN[n]
    # wave36 probe365: (64,64) is the official rms_norm core shape and had no
    # backward route (the wildcard starts at N=256) - dx at 64 rows 0.403 f16 /
    # 0.374 f32 against 0.596 / 0.524 on the generic kernels, and at 1024 rows
    # 1.28 / 0.862 against 3.72 / 2.82.  The tile pick flips with dtype: f16
    # T=128 (0.404 / 1.28) beats T=64 (0.464 / 2.22), f32 T=64 wins (0.374 /
    # 0.862 vs 0.407 / 1.28), so it cannot live in the dtype-blind table.
    if n == 64:
        if is16:
            return 4, 128, 128
        return 8, 64, 64
    if _WILD_MIN_N <= n <= _WILD_MAX_N and m >= _WILD_MIN_M:
        return _WILD_DX[0], _WILD_DX[1], _WILD_BN
    return None


def rmsb_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_RMSB_FAST", "1") != "0"


# ---------------- wave39: narrow-row fused tile (N <= 32) ----------------
# probe424: below 64 columns there is no route at all, so the whole backward
# ran the generic FlagGems kernels.  The tile keeps a whole narrow row in one
# block: c = sum(x*(dy*w)) per row, dx from those same registers, and the dw
# column partials from the same pass (x*r*dy summed down the tile) so the dw
# fold kernel never re-reads x.  Same-session minima, generic -> tile:
#   (4096,4)  f32 6.979 -> 0.615 (11.3x)  f16 7.328 -> 0.796 (9.2x)
#   (4096,8)      7.105 -> 0.987 (7.2x)       7.633 -> 1.350 (5.7x)
#   (2048,16)     3.966 -> 0.775 (5.1x)       4.191 -> 0.862 (4.9x)
#   (2048,32)     5.403 -> 1.180 (4.6x)       4.940 -> 1.504 (3.3x)
#   (256,32)      1.007 -> 0.367 (2.7x)       0.966 -> 0.421 (2.3x)
# dx error against a float64 oracle stays at the routed level (f32 6e-7..8e-7,
# f16 at output rounding); dw is an M-sum, so its f16 error is 1e-2..5e-2 on
# values of magnitude ~sqrt(M), the same relative level as the row route.
# 64 columns keeps its existing route (tile 0.357 against 0.533 generic there
# but the routed route is better), so the tile gate is power-of-two N <= 32.
_RMSB_TILE_MAX_N = 32


def _rmsb_tile_bm(m: int, bn: int) -> int:
    bm = min(1024, max(64, 8192 // bn), max(64, m // 2))
    while bm > 8 and m % bm:
        bm //= 2
    return bm if m % bm == 0 else 0


def _rmsb_tile_ok(n: int, m: int, f32: bool = True) -> bool:
    # f32 only, for two different reasons now.
    # (1) wave40 probe429: the garbage dw that wave39 saw on a cold process was
    # a launch-ordering bug, not accuracy - with a nested forward in front of
    # it, a *plain* combine launch is not ordered against the tile's nested
    # group, so it read pw before the tile's stores landed: (2048,32) f16 came
    # back dw 4.5e2 against |dw| 105 as tile-nested/combine-plain, while both
    # same-dispatcher arrangements (nested/nested, plain/plain) were exact on
    # all four reps.  torch_device_fn.synchronize() is `pass` on this driver,
    # which is why wave39's sync never helped.  rmsb_fast now keeps both
    # launches in one nested block (worth 6.5% at (2048,32) f16: 0.737 ->
    # 0.785 ms) and the direct-call battery is exact for f32/f16/bf16.
    # (2) probe433 still measures dw 6.107e3 on the *autograd* path for f16
    # (2048,32) with RMSB_FAST=1 against 2.9e-2 with RMSB_FAST=0, dx correct in
    # both, and instrumenting RmsNorm.backward and rmsb_fast shows neither is
    # entered - so that number comes from a path this gate does not control and
    # f16 stays out until it is explained.
    return (
        f32
        and n <= _RMSB_TILE_MAX_N
        and n == triton.next_power_of_2(n)
        and m >= 8
        and _rmsb_tile_bm(m, n) > 0
    )


@triton.jit
def _rmsb_dx_tile(X, DY, W, INV, DX, PW, N, BM: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0)
    idx = p * BM * BN + tl.arange(0, BM * BN)
    x = tl.reshape(tl.load(X + idx).to(tl.float32), (BM, BN))
    d = tl.reshape(tl.load(DY + idx).to(tl.float32), (BM, BN))
    w = tl.load(W + tl.arange(0, BN)).to(tl.float32)[None, :]
    rows = p * BM + tl.arange(0, BM)
    r = tl.load(INV + rows).to(tl.float32)[:, None]
    dwh = d * w
    c = tl.sum(x * dwh, axis=1)[:, None]
    dxv = r * dwh - x * (r * r * r * c / N)
    tl.store(DX + idx, tl.reshape(dxv, (BM * BN,)).to(DX.dtype.element_ty))
    tl.store(PW + p * BN + tl.arange(0, BN), tl.sum((x * r) * d, axis=0))


def rmsb_fast_ok(x, dy, inv_rms, weight) -> bool:
    if x.numel() == 0 or dy.numel() != x.numel():
        return False
    # wave33: bf16 admitted - the kernels load through .to(tl.float32) and store
    # with .to(DX.dtype.element_ty), so only the gate was keeping the whole
    # bf16 family on the generic kernels (25.8 -> 3.9 ms fwd at 2048^2).
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if dy.dtype != x.dtype or weight.dtype != x.dtype:
        return False
    if not x.is_contiguous() or not dy.is_contiguous() or not weight.is_contiguous():
        return False
    n = weight.numel()
    if x.numel() % n:
        return False
    if _rmsb_tile_ok(n, x.numel() // n, x.dtype == torch.float32):
        return inv_rms is not None and inv_rms.numel() >= x.numel() // n
    if _routes(n, x.numel() // n, x.dtype != torch.float32) is None:
        return False
    if inv_rms is None or inv_rms.numel() < x.numel() // n:
        return False
    return True


@triton.jit
def _rmsb_dx(
    X, DY, W, INV, DX, N, K: tl.constexpr, TILE: tl.constexpr, NOMASK: tl.constexpr
):
    pid = tl.program_id(0)
    for i in tl.static_range(K):
        row = pid * K + i
        base = row * N
        r = tl.load(INV + row).to(tl.float32)
        acc = tl.zeros((TILE,), dtype=tl.float32)
        for c in range(0, N, TILE):
            offs = c + tl.arange(0, TILE)
            if NOMASK:
                xv = tl.load(X + base + offs).to(tl.float32)
                dv = tl.load(DY + base + offs).to(tl.float32)
                wv = tl.load(W + offs).to(tl.float32)
                acc += xv * (dv * wv)
            else:
                mm = offs < N
                xv = tl.load(X + base + offs, mask=mm, other=0.0).to(tl.float32)
                dv = tl.load(DY + base + offs, mask=mm, other=0.0).to(tl.float32)
                wv = tl.load(W + offs, mask=mm, other=0.0).to(tl.float32)
                acc += xv * (dv * wv)
        csum = tl.sum(acc, axis=0)
        r3c = (r * r * r * csum) / N
        for c in range(0, N, TILE):
            offs = c + tl.arange(0, TILE)
            if NOMASK:
                xv = tl.load(X + base + offs).to(tl.float32)
                dv = tl.load(DY + base + offs).to(tl.float32)
                wv = tl.load(W + offs).to(tl.float32)
                dxv = r * (dv * wv) - xv * r3c
                tl.store(DX + base + offs, dxv.to(DX.dtype.element_ty))
            else:
                mm = offs < N
                xv = tl.load(X + base + offs, mask=mm, other=0.0).to(tl.float32)
                dv = tl.load(DY + base + offs, mask=mm, other=0.0).to(tl.float32)
                wv = tl.load(W + offs, mask=mm, other=0.0).to(tl.float32)
                dxv = r * (dv * wv) - xv * r3c
                tl.store(DX + base + offs, dxv.to(DX.dtype.element_ty), mask=mm)


@triton.jit
def _rmsb_dw_fold(
    X, DY, INV, PW, M, N, RB: tl.constexpr, BN: tl.constexpr, NOMASK: tl.constexpr
):
    pc = tl.program_id(0)
    pr = tl.program_id(1)
    cols = pc * BN + tl.arange(0, BN)
    cmask = cols < N
    r_end = tl.minimum((pr + 1) * RB, M)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for r in range(pr * RB, r_end):
        rv = tl.load(INV + r).to(tl.float32)
        if NOMASK:
            xv = tl.load(X + r * N + cols).to(tl.float32)
            dv = tl.load(DY + r * N + cols).to(tl.float32)
            acc += (xv * rv) * dv
        else:
            xv = tl.load(X + r * N + cols, mask=cmask, other=0.0).to(tl.float32)
            dv = tl.load(DY + r * N + cols, mask=cmask, other=0.0).to(tl.float32)
            acc += (xv * rv) * dv
    if NOMASK:
        tl.store(PW + pr * N + cols, acc)
    else:
        tl.store(PW + pr * N + cols, acc, mask=cmask)


@triton.jit
def _rmsb_dw_comb(PW, DW, N, NRB, BN: tl.constexpr, NOMASK: tl.constexpr):
    pc = tl.program_id(0)
    cols = pc * BN + tl.arange(0, BN)
    cmask = cols < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for s in range(0, NRB):
        if NOMASK:
            acc += tl.load(PW + s * N + cols)
        else:
            acc += tl.load(PW + s * N + cols, mask=cmask, other=0.0)
    if NOMASK:
        tl.store(DW + cols, acc.to(DW.dtype.element_ty))
    else:
        tl.store(DW + cols, acc.to(DW.dtype.element_ty), mask=cmask)


def rmsb_fast(x, dy, inv_rms, weight, n, m):
    """Launch the fast backward; returns (dx, dw). Caller must have checked ok()."""
    dx = torch.empty_like(x)
    inv = inv_rms.contiguous().reshape(-1)
    if _rmsb_tile_ok(n, m, x.dtype == torch.float32):
        # wave39: one pass over a narrow tile does dx and the dw partials.
        # wave40 probe429: the tile and the combine must sit in the same nested
        # block - a plain launch that follows a nested one is not ordered
        # against it, and on a cold process the combine then read pw before the
        # tile's stores landed (dw 4.5e2 against |dw| 105 at (2048,32) f16).
        # The sync that wave39 added here did nothing: torch_device_fn
        # .synchronize() is `pass` on this driver.
        bm = _rmsb_tile_bm(m, n)
        prb = m // bm
        pw = torch.empty((prb, n), dtype=torch.float32, device=x.device)
        cb = min(2048, triton.next_power_of_2(n))
        dw = torch.empty(n, dtype=x.dtype, device=x.device)
        with nested_dispatch(m * n):
            with torch_device_fn.device(x.device):
                _rmsb_dx_tile[(prb,)](x, dy, weight, inv, dx, pw, n, BM=bm, BN=n)
                _rmsb_dw_comb[(triton.cdiv(n, cb),)](pw, dw, n, prb, cb, True)
        return dx, dw
    k, tile, bn = _routes(n, m, x.dtype != torch.float32)
    while k > 1 and m % k:
        k //= 2
    nomask_dx = (n % tile) == 0
    grid = (m // k,) if k > 1 else (m,)
    with torch_device_fn.device(x.device):
        # wave12: dx wins under nested dispatch for N=2048 at large M
        # (probe105: 4096/8192 rows -> 1.41x; M<=2048 neutral; other N lose).
        if (
            k >= 4
            and n == 2048
            and m >= 4096
            and os.environ.get("FLAGGEMS_SPACEMIT_RMSB_NEST", "1") != "0"
        ):
            with nested_dispatch(m * n):
                _rmsb_dx[grid](x, dy, weight, inv, dx, n, k, tile, nomask_dx)
        else:
            _rmsb_dx[grid](x, dy, weight, inv, dx, n, k, tile, nomask_dx)
        prb = triton.cdiv(m, _DW_RB)
        pw = torch.empty((prb, n), dtype=torch.float32, device=x.device)
        _rmsb_dw_fold[(triton.cdiv(n, bn), prb)](
            x, dy, inv, pw, m, n, _DW_RB, bn, (n % bn) == 0
        )
        dw = torch.empty(n, dtype=x.dtype, device=x.device)
        cb = min(2048, triton.next_power_of_2(n))
        _rmsb_dw_comb[(triton.cdiv(n, cb),)](pw, dw, n, prb, cb, (n % cb) == 0)
    return dx, dw
