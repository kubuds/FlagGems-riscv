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


def sum_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_SUM_FAST", "1") != "0"


# ---------------- flat two-stage ----------------
@triton.jit
def _sum_flat_k1(inp, mid, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(inp + off, mask=off < M, other=0.0)
    tl.store(mid + pid, tl.sum(v, axis=0))


@triton.jit
def _sum_flat_k2(mid, out, MIDN, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    v = tl.load(mid + off, mask=off < MIDN, other=0.0)
    tl.store(out, tl.sum(v, axis=0))


def _flat_block(M: int) -> int:
    b = triton.next_power_of_2(max(4096, -(-M // 64)))
    if b > 16384:
        b = 16384
    p2 = triton.next_power_of_2(M)
    if b > p2:
        b = p2
    return b


def sum_flat_launch(x, out, M):
    """Two-stage flat sum into out (scalar tensor). x must be f32/contiguous."""
    block = _flat_block(M)
    midn = triton.cdiv(M, block)
    mid = torch.empty(midn, dtype=torch.float32, device=x.device)
    _sum_flat_k1[(midn,)](x, mid, M, block)
    _sum_flat_k2[(1,)](mid, out, midn, triton.next_power_of_2(midn))


# ---------------- sum_dim multi-row (plain per-row reduce) ----------------
@triton.jit
def _sum_dim_mmk512(X, Out, N, K: tl.constexpr):
    p = tl.program_id(0)
    for i in tl.static_range(K):
        r = p * K + i
        xv = tl.load(X + r * N + tl.arange(0, 512))
        tl.store(Out + r, tl.sum(xv, axis=0))


@triton.jit
def _sum_dim_mmk1024(X, Out, N, K: tl.constexpr):
    p = tl.program_id(0)
    for i in tl.static_range(K):
        r = p * K + i
        xv = tl.load(X + r * N + tl.arange(0, 1024))
        tl.store(Out + r, tl.sum(xv, axis=0))


@triton.jit
def _sum_dim_mmk2048(X, Out, N, K: tl.constexpr):
    p = tl.program_id(0)
    for i in tl.static_range(K):
        r = p * K + i
        xv = tl.load(X + r * N + tl.arange(0, 2048))
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
            xv = tl.load(X + r * N + offs)
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
        _sum_dim_mmk_chunked[(M // k,)](x, out_flat, N, k, 8192, 4)
        return True
    return False
