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

"""wave5 (campaign 1014pm): short-row mean_dim direct kernels, multi-row per program.

Root cause found in wave5 (probe50/51, same-session f32, 4M elems):
the "row penalty" of short-row mean_dim was NOT per-element scalar reduce
cost — a 512/1024-length tl.sum runs at ~0.4 ns/elem once the program
amortizes launch/setup across several rows. One-row-per-program kernels pay
~0.9-1.9 us/row no matter the reduce length (bp512 7.5-8.0 ms @4M; plain
8-ish), while K rows per program collapse to the same rate as wide rows:

    (8192,512)   7.46 -> 1.72 ms   K=16  (5.0x vs current bp512 route)
    (4096,1024)  8.00 -> 1.43 ms   K=16  (5.8x)
    (2048,2048)  2.10 -> 1.36 ms   K=8   (1.56x vs plain TILE=2048)
    (1024,4096)  ~1.70-> 1.28 ms   K=4
    (512,32768)  4.78 -> 4.42 ms   K=4 chunk-loop (vs plain TILE=2048)

vs torch (same session): (8192,512) 1.72 vs 1.87 (1.09x), (4096,1024) 1.43
vs 1.71 (1.20x) — the row shapes now BEAT torch-native mean.
nested dispatch measured WORSER for these kernels (off=1.93 on=4.35 @8192x512
probe51): keep default launcher. Compile-stability rechecked in fresh
processes (probe51b/51c: two rounds each, <2% drift).
Gate: contiguous x; N in {512,1024,2048,4096,32768}; M % K == 0.
Kill switch: FLAGGEMS_SPACEMIT_MEAN_MMK=0 (falls back to the legacy routes).
"""

import os

import triton
import triton.language as tl

from ._dispatch import nested_dispatch


@triton.jit
def mean_dim_kernel_mmk512(X, Out, N):
    p = tl.program_id(0)
    r0 = p * 16 + 0
    x0 = tl.load(X + r0 * N + tl.arange(0, 512))
    tl.store(Out + r0, tl.sum(x0, axis=0) / N)
    r1 = p * 16 + 1
    x1 = tl.load(X + r1 * N + tl.arange(0, 512))
    tl.store(Out + r1, tl.sum(x1, axis=0) / N)
    r2 = p * 16 + 2
    x2 = tl.load(X + r2 * N + tl.arange(0, 512))
    tl.store(Out + r2, tl.sum(x2, axis=0) / N)
    r3 = p * 16 + 3
    x3 = tl.load(X + r3 * N + tl.arange(0, 512))
    tl.store(Out + r3, tl.sum(x3, axis=0) / N)
    r4 = p * 16 + 4
    x4 = tl.load(X + r4 * N + tl.arange(0, 512))
    tl.store(Out + r4, tl.sum(x4, axis=0) / N)
    r5 = p * 16 + 5
    x5 = tl.load(X + r5 * N + tl.arange(0, 512))
    tl.store(Out + r5, tl.sum(x5, axis=0) / N)
    r6 = p * 16 + 6
    x6 = tl.load(X + r6 * N + tl.arange(0, 512))
    tl.store(Out + r6, tl.sum(x6, axis=0) / N)
    r7 = p * 16 + 7
    x7 = tl.load(X + r7 * N + tl.arange(0, 512))
    tl.store(Out + r7, tl.sum(x7, axis=0) / N)
    r8 = p * 16 + 8
    x8 = tl.load(X + r8 * N + tl.arange(0, 512))
    tl.store(Out + r8, tl.sum(x8, axis=0) / N)
    r9 = p * 16 + 9
    x9 = tl.load(X + r9 * N + tl.arange(0, 512))
    tl.store(Out + r9, tl.sum(x9, axis=0) / N)
    r10 = p * 16 + 10
    x10 = tl.load(X + r10 * N + tl.arange(0, 512))
    tl.store(Out + r10, tl.sum(x10, axis=0) / N)
    r11 = p * 16 + 11
    x11 = tl.load(X + r11 * N + tl.arange(0, 512))
    tl.store(Out + r11, tl.sum(x11, axis=0) / N)
    r12 = p * 16 + 12
    x12 = tl.load(X + r12 * N + tl.arange(0, 512))
    tl.store(Out + r12, tl.sum(x12, axis=0) / N)
    r13 = p * 16 + 13
    x13 = tl.load(X + r13 * N + tl.arange(0, 512))
    tl.store(Out + r13, tl.sum(x13, axis=0) / N)
    r14 = p * 16 + 14
    x14 = tl.load(X + r14 * N + tl.arange(0, 512))
    tl.store(Out + r14, tl.sum(x14, axis=0) / N)
    r15 = p * 16 + 15
    x15 = tl.load(X + r15 * N + tl.arange(0, 512))
    tl.store(Out + r15, tl.sum(x15, axis=0) / N)

@triton.jit
def mean_dim_kernel_mmk1024(X, Out, N):
    p = tl.program_id(0)
    r0 = p * 16 + 0
    x0 = tl.load(X + r0 * N + tl.arange(0, 1024))
    tl.store(Out + r0, tl.sum(x0, axis=0) / N)
    r1 = p * 16 + 1
    x1 = tl.load(X + r1 * N + tl.arange(0, 1024))
    tl.store(Out + r1, tl.sum(x1, axis=0) / N)
    r2 = p * 16 + 2
    x2 = tl.load(X + r2 * N + tl.arange(0, 1024))
    tl.store(Out + r2, tl.sum(x2, axis=0) / N)
    r3 = p * 16 + 3
    x3 = tl.load(X + r3 * N + tl.arange(0, 1024))
    tl.store(Out + r3, tl.sum(x3, axis=0) / N)
    r4 = p * 16 + 4
    x4 = tl.load(X + r4 * N + tl.arange(0, 1024))
    tl.store(Out + r4, tl.sum(x4, axis=0) / N)
    r5 = p * 16 + 5
    x5 = tl.load(X + r5 * N + tl.arange(0, 1024))
    tl.store(Out + r5, tl.sum(x5, axis=0) / N)
    r6 = p * 16 + 6
    x6 = tl.load(X + r6 * N + tl.arange(0, 1024))
    tl.store(Out + r6, tl.sum(x6, axis=0) / N)
    r7 = p * 16 + 7
    x7 = tl.load(X + r7 * N + tl.arange(0, 1024))
    tl.store(Out + r7, tl.sum(x7, axis=0) / N)
    r8 = p * 16 + 8
    x8 = tl.load(X + r8 * N + tl.arange(0, 1024))
    tl.store(Out + r8, tl.sum(x8, axis=0) / N)
    r9 = p * 16 + 9
    x9 = tl.load(X + r9 * N + tl.arange(0, 1024))
    tl.store(Out + r9, tl.sum(x9, axis=0) / N)
    r10 = p * 16 + 10
    x10 = tl.load(X + r10 * N + tl.arange(0, 1024))
    tl.store(Out + r10, tl.sum(x10, axis=0) / N)
    r11 = p * 16 + 11
    x11 = tl.load(X + r11 * N + tl.arange(0, 1024))
    tl.store(Out + r11, tl.sum(x11, axis=0) / N)
    r12 = p * 16 + 12
    x12 = tl.load(X + r12 * N + tl.arange(0, 1024))
    tl.store(Out + r12, tl.sum(x12, axis=0) / N)
    r13 = p * 16 + 13
    x13 = tl.load(X + r13 * N + tl.arange(0, 1024))
    tl.store(Out + r13, tl.sum(x13, axis=0) / N)
    r14 = p * 16 + 14
    x14 = tl.load(X + r14 * N + tl.arange(0, 1024))
    tl.store(Out + r14, tl.sum(x14, axis=0) / N)
    r15 = p * 16 + 15
    x15 = tl.load(X + r15 * N + tl.arange(0, 1024))
    tl.store(Out + r15, tl.sum(x15, axis=0) / N)

@triton.jit
def mean_dim_kernel_mmk2048(X, Out, N):
    p = tl.program_id(0)
    r0 = p * 8 + 0
    x0 = tl.load(X + r0 * N + tl.arange(0, 2048))
    tl.store(Out + r0, tl.sum(x0, axis=0) / N)
    r1 = p * 8 + 1
    x1 = tl.load(X + r1 * N + tl.arange(0, 2048))
    tl.store(Out + r1, tl.sum(x1, axis=0) / N)
    r2 = p * 8 + 2
    x2 = tl.load(X + r2 * N + tl.arange(0, 2048))
    tl.store(Out + r2, tl.sum(x2, axis=0) / N)
    r3 = p * 8 + 3
    x3 = tl.load(X + r3 * N + tl.arange(0, 2048))
    tl.store(Out + r3, tl.sum(x3, axis=0) / N)
    r4 = p * 8 + 4
    x4 = tl.load(X + r4 * N + tl.arange(0, 2048))
    tl.store(Out + r4, tl.sum(x4, axis=0) / N)
    r5 = p * 8 + 5
    x5 = tl.load(X + r5 * N + tl.arange(0, 2048))
    tl.store(Out + r5, tl.sum(x5, axis=0) / N)
    r6 = p * 8 + 6
    x6 = tl.load(X + r6 * N + tl.arange(0, 2048))
    tl.store(Out + r6, tl.sum(x6, axis=0) / N)
    r7 = p * 8 + 7
    x7 = tl.load(X + r7 * N + tl.arange(0, 2048))
    tl.store(Out + r7, tl.sum(x7, axis=0) / N)

@triton.jit
def mean_dim_kernel_mmk4096(X, Out, N):
    p = tl.program_id(0)
    r0 = p * 4 + 0
    x0 = tl.load(X + r0 * N + tl.arange(0, 4096))
    tl.store(Out + r0, tl.sum(x0, axis=0) / N)
    r1 = p * 4 + 1
    x1 = tl.load(X + r1 * N + tl.arange(0, 4096))
    tl.store(Out + r1, tl.sum(x1, axis=0) / N)
    r2 = p * 4 + 2
    x2 = tl.load(X + r2 * N + tl.arange(0, 4096))
    tl.store(Out + r2, tl.sum(x2, axis=0) / N)
    r3 = p * 4 + 3
    x3 = tl.load(X + r3 * N + tl.arange(0, 4096))
    tl.store(Out + r3, tl.sum(x3, axis=0) / N)

@triton.jit
def mean_dim_kernel_mmk32768(X, Out, N):
    p = tl.program_id(0)
    r0 = p * 4 + 0
    acc0 = 0.0
    for c in range(0, 32768, 2048):
        acc0 += tl.sum(tl.load(X + r0 * N + c + tl.arange(0, 2048)))
    tl.store(Out + r0, acc0 / N)
    r1 = p * 4 + 1
    acc1 = 0.0
    for c in range(0, 32768, 2048):
        acc1 += tl.sum(tl.load(X + r1 * N + c + tl.arange(0, 2048)))
    tl.store(Out + r1, acc1 / N)
    r2 = p * 4 + 2
    acc2 = 0.0
    for c in range(0, 32768, 2048):
        acc2 += tl.sum(tl.load(X + r2 * N + c + tl.arange(0, 2048)))
    tl.store(Out + r2, acc2 / N)
    r3 = p * 4 + 3
    acc3 = 0.0
    for c in range(0, 32768, 2048):
        acc3 += tl.sum(tl.load(X + r3 * N + c + tl.arange(0, 2048)))
    tl.store(Out + r3, acc3 / N)

_MMK_K = {512: 16, 1024: 16, 2048: 8, 4096: 4, 32768: 4}
_MMK_KERNELS = {
    512: mean_dim_kernel_mmk512,
    1024: mean_dim_kernel_mmk1024,
    2048: mean_dim_kernel_mmk2048,
    4096: mean_dim_kernel_mmk4096,
    32768: mean_dim_kernel_mmk32768,
}


def mean_mmk_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_MEAN_MMK", "1") != "0"


def mean_mmk_launch(x, out, M, N) -> bool:
    """Launch the multi-row kernel if (N, M) fits; return True if handled."""
    kern = _MMK_KERNELS.get(N)
    if kern is None:
        return False
    k = _MMK_K[N]
    if M % k:
        return False
    # wave12: only the N=32768 chunked route wins under nested dispatch
    # (probe104/105: 1.19/1.22x); all other routes lose 0.65-0.85.
    if N == 32768 and os.environ.get("FLAGGEMS_SPACEMIT_MEAN_NEST", "1") != "0":
        with nested_dispatch(M * N):
            kern[(M // k,)](x, out, N)
    else:
        kern[(M // k,)](x, out, N)
    return True
