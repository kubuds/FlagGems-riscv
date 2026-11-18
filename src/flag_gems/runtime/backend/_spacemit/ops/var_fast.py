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

"""wave7 (campaign 1015pm): var / std / var_mean fast paths (spacemit backend).

Two independent fixes:

1) Registration bug (probe62/63): aten.var.dim's schema is
   ``var.dim(Tensor self, int[1]? dim, bool unbiased, bool keepdim)`` - all
   positional - so torch calls the python impl as ``var_dim(x, dim, unbiased)``
   whenever unbiased is spelled out. The generic FlagGems signature
   ``var_dim(x, dim=None, *, correction=None, keepdim=False)`` raised TypeError
   ("takes from 1 to 2 positional arguments but 3 were given") for every
   ``torch.var(x, dim=..., unbiased=False)`` under use_gems. Fixed in
   ops/var.py (unconditional).

2) Speed: the generic trailing-dim route is one-row-per-program Welford -
   the launch/setup-amortization penalty wave5 found for mean, PLUS the
   elementwise-reduce cliff wave4 found for layer_norm.

   Structure (probe65-67, same-session f32, (8192,512) unless noted):
       generic (official baseline)      41.09 ms
       two-pass tl.sum(d*d) direct      21.72 ms   (probe65 v1 - REJECTED)
       single-pass tl.sum(x*x) direct   17.02 ms   (probe65 v2 - REJECTED)
       wave4 mul-chain acc, two-pass    4.66 ms    (probe67 A - CHOSEN)
   The poison is tl.sum() over a freshly computed elementwise vector
   (wave4 probe40 saw the same: 23 ms); the fix is a BLOCK=256 mul-chain
   accumulator loop + one tl.sum over the accumulator.

   Accuracy: variant A computes mean first, then sum((x-m)^2) in a second
   pass - cancellation-free. Variant B (one-pass E[x2]-m*m) ties on speed
   but FAILS a large-mean check (mean=1e3: abs err 0.57) - rejected.
   A's max abs error vs torch: <= 3.6e-7 on every shape (probe67).

   Routes (probe67 A, SB=256 unless noted):
       N=512    K=4   -> 4.66 ms  (official 41.09, torch 19.42)
       N=1024   K=8   -> 3.64 ms  (official 25.70, torch 15.94)
       N=2048   K=8   -> 3.05 ms  (official 21.92, torch 14.53)
       N=4096   K=4   -> 2.68 ms  (official 20.77, torch 17.51)
       N=32768  K=2, SB=512 -> 9.16 ms (official 77.63, torch 53.40)
       N=512, M=512 K=4 -> 0.38 ms (official 2.73, torch 1.05)

Coverage: f32, contiguous, single trailing dim, numel >= 65536, row width
in {512,1024,2048,4096,32768}, M divisible by the route's K (halving K is
tried before falling back), N > correction. Everything else falls through
to the generic implementations unchanged.

Kill switch: FLAGGEMS_SPACEMIT_VAR_FAST=0 (fast paths only; the var_dim
signature fix is always active).
"""

import os

import triton
import triton.language as tl

# N -> (K, SB, s1_chunk_width, s1_chunks); probe67 A sweep.
_ROUTES = {
    512: (4, 256, 512, 1),
    1024: (8, 256, 1024, 1),
    2048: (8, 256, 2048, 1),
    4096: (4, 256, 4096, 1),
    32768: (2, 512, 8192, 4),
}


def var_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_VAR_FAST", "1") != "0"


@triton.jit
def _var_mmk(
    X, Var, Mean, N, CORR,
    K: tl.constexpr, W: tl.constexpr, SB: tl.constexpr,
    NS1: tl.constexpr, OUT_MEAN: tl.constexpr, SQRT: tl.constexpr,
):
    """Two-pass wave4-style var per row (K rows/program).

    Pass 1: NS1 load-sums (single full-row tl.sum when NS1=1) -> mean.
    Pass 2: BLOCK=SB mul-chain accumulator of (x-m)^2 + one tl.sum over the
    accumulator (tl.sum over a fresh elementwise vector is the cliff).
    """
    p = tl.program_id(0)
    for i in tl.static_range(K):
        r = p * K + i
        base = X + r * N
        s1 = 0.0
        for c in tl.static_range(NS1):
            xv = tl.load(base + c * W + tl.arange(0, W))
            s1 += tl.sum(xv, axis=0)
        m = s1 / N
        acc = tl.zeros((SB,), dtype=tl.float32)
        for c in range(0, N, SB):
            x = tl.load(base + c + tl.arange(0, SB))
            d = x - m
            acc += d * d
        s2 = tl.sum(acc, axis=0)
        v = s2 / (N - CORR)
        if SQRT:
            v = tl.sqrt(v)
        tl.store(Var + r, v)
        if OUT_MEAN:
            tl.store(Mean + r, m)


def var_dim_mmk_launch(
    x, var_out, mean_out, M, N, corr, want_mean, sqrt
) -> bool:
    """Launch the multi-row kernel if (N, M) fit; return True if handled.

    mean_out is written only when want_mean; callers may pass var_out as the
    placeholder in that case (OUT_MEAN=False never stores to it).
    Tries the swept K first, then K/2 down to 1 for non-divisible M.
    """
    ent = _ROUTES.get(N)
    if ent is None:
        return False
    if N - float(corr) <= 0:
        return False
    k0, sb, w1, ns1 = ent
    k = None
    for cand in (k0, k0 // 2, k0 // 4, 1):
        if cand >= 1 and M % cand == 0:
            k = cand
            break
    if k is None:
        return False
    _var_mmk[(M // k,)](
        x, var_out, mean_out, N, float(corr),
        K=k, W=w1, SB=sb, NS1=ns1,
        OUT_MEAN=bool(want_mean), SQRT=bool(sqrt),
    )
    return True
