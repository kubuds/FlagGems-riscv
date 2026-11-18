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

"""wave6 (campaign 1015am): rms_norm forward direct kernels (f32).

Structure ported from the layernorm_fast playbook (wave4): per-row
loop-carried mul-chain acc for sum(x^2) (elementwise feeding tl.sum would kill
the reduction) + a separate apply loop. Narrow-column rows additionally get
K rows/program (wave5 launch-amortization lesson).

Same-session f32 numbers (probe59/60; legacy route = single-row tl.sum(x*x)):
    (512,512)   2.03 -> 0.59 ms  (K=8, TILE=256)   vs torch 0.88x
    (8192,512)  31.4 -> 8.05 ms  (K=8, TILE=256)   vs torch 1.11x
    (2048,2048) 25.0 -> 5.31 ms  (row, TILE=512)   vs torch 3.68x
    (512,4096)  12.4 -> 2.38 ms  (row, TILE=1024)  vs torch 1.89x
Accuracy vs f32 reference: maxabs 1.9e-6 @2048^2 (better than legacy).

Backward is untouched (legacy kernels). Non-f32 / non-contiguous / N not in
the route table fall back to the generic forward.

Kill switch: FLAGGEMS_SPACEMIT_RMS_FAST=0.
"""

import os

import triton
import triton.language as tl


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


# N -> ("mmk", K, TILE) or ("row", TILE); only verified shapes are wired.
_ROUTES = {
    512: ("mmk", 8, 256),
    2048: ("row", 512),
    4096: ("row", 1024),
}


def rms_fast_launch(x, w, y, inv, M, N, eps) -> bool:
    """Launch the fast forward kernel; False = caller must fall back."""
    ent = _ROUTES.get(N)
    if ent is None:
        return False
    if ent[0] == "mmk":
        k, tile = ent[1], ent[2]
        if M % k == 0:
            _rms_mmk[(M // k,)](x, w, y, inv, N, eps, k, tile)
            return True
        # conservative fallback chain for non-divisible M
        if M % 4 == 0:
            _rms_mmk[(M // 4,)](x, w, y, inv, N, eps, 4, 256)
            return True
        _rms_row[(M,)](x, w, y, inv, N, eps, 256)
        return True
    tile = ent[1]
    _rms_row[(M,)](x, w, y, inv, N, eps, tile)
    return True
