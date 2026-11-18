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

_DX_ROUTES = {512: (4, 256), 2048: (4, 256), 4096: (2, 512)}
_DW_BN = {512: 512, 2048: 512, 4096: 1024}
_DW_RB = 64


def rmsb_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_RMSB_FAST", "1") != "0"


def rmsb_fast_ok(x, dy, inv_rms, weight) -> bool:
    if x.numel() == 0 or dy.numel() != x.numel():
        return False
    if x.dtype not in (torch.float32, torch.float16):
        return False
    if dy.dtype != x.dtype or weight.dtype != x.dtype:
        return False
    if not x.is_contiguous() or not dy.is_contiguous() or not weight.is_contiguous():
        return False
    n = weight.numel()
    if n not in _DX_ROUTES or x.numel() % n:
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
    k, tile = _DX_ROUTES[n]
    while k > 1 and m % k:
        k //= 2
    nomask_dx = (n % tile) == 0
    grid = (m // k,) if k > 1 else (m,)
    with torch_device_fn.device(x.device):
        _rmsb_dx[grid](x, dy, weight, inv, dx, n, k, tile, nomask_dx)
        bn = _DW_BN[n]
        prb = triton.cdiv(m, _DW_RB)
        pw = torch.empty((prb, n), dtype=torch.float32, device=x.device)
        _rmsb_dw_fold[(triton.cdiv(n, bn), prb)](
            x, dy, inv, pw, m, n, _DW_RB, bn, (n % bn) == 0
        )
        dw = torch.empty(n, dtype=x.dtype, device=x.device)
        cb = min(2048, triton.next_power_of_2(n))
        _rmsb_dw_comb[(triton.cdiv(n, cb),)](pw, dw, n, prb, cb, (n % cb) == 0)
    return dx, dw
