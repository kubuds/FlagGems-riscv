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
}
_ROUTES_F16 = {
    512: (16, 256),
    2048: (4, 512),
    4096: (4, 512),
}


def _pick_k(m: int, k0: int) -> int:
    k = k0
    while k > 1 and m % k:
        k //= 2
    return k


def rms_fast_launch(x, w, y, inv, M, N, eps, f16: bool = False) -> bool:
    """Launch the fast forward kernel; False = caller must fall back."""
    routes = _ROUTES_F16 if f16 else _ROUTES_F32
    ent = routes.get(N)
    if ent is None:
        return False
    k0, tile = ent
    k = _pick_k(M, k0)
    if k > 1:
        kern = _rms_mmk16 if f16 else _rms_mmk
        if k >= 4:
            with nested_dispatch(M * N):
                kern[(M // k,)](x, w, y, inv, N, eps, k, tile)
        else:
            kern[(M // k,)](x, w, y, inv, N, eps, k, tile)
        return True
    row = _rms_row16 if f16 else _rms_row
    row[(M,)](x, w, y, inv, N, eps, tile)
    return True
