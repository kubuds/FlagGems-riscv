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

"""wave7/wave10 (campaign 1015pm/1016pm): var / std / var_mean fast paths.

wave7: the generic trailing-dim route is one-row-per-program Welford - the
launch/setup-amortization penalty (wave5) PLUS the elementwise-reduce cliff
(wave4). Structure (probe65-67, same-session f32, (8192,512) unless noted):

    generic (official baseline)      41.09 ms
    two-pass tl.sum(d*d) direct      21.72 ms   (probe65 v1 - REJECTED)
    single-pass tl.sum(x*x) direct   17.02 ms   (probe65 v2 - REJECTED)
    wave4 mul-chain acc, two-pass    4.66 ms    (probe67 A - CHOSEN)

Accuracy: variant A computes mean first, then sum((x-m)^2) in a second pass
- cancellation-free. Variant B (one-pass E[x2]-m*m) ties on speed but FAILS
a large-mean check (mean=1e3: abs err 0.57) - rejected. A's max abs error vs
torch: <= 3.6e-7 on every shape (probe67).

wave10 (1016pm):
  a) f16 support for the trailing-dim route. A dtype cast feeding tl.sum
     directly hits the spine "feeding cliff" (probe91b: 20.7 vs 4.9 ms f32
     @8192x512), so the f16 pass-1 is a chunked LOOP-carried accumulator
     (W2=256; probe93 scan). Same-session ms (8192x512 / 2048x2048 / 512x512):
     7.65/3.43/0.48 vs torch 19.64/10.84/0.80, vs pre-wave10 generic
     46.6/28.2/3.05; err 0.00 vs torch on every probe93 shape (incl. 1001x512
     K-chain, 1000x1024, 128x32768).
  b) flat (all-elements) two-stage path. Stage-1 partial sums (BLOCK from
     the sum-flat rule) + host-side f64 combine (math.fsum); pass 2 =
     squared-deviation loop with the mean as a runtime arg. Same-session 1M:
     f32 8.64 -> 1.06 ms (torch 1.07); f16 8.50 -> 0.99 ms (torch 4.40);
     err 0.00e+00 vs torch (large-mean and M=700000 checks included).

Coverage: f32/f16, contiguous, single trailing dim OR fully-flat reduction,
numel >= 65536, row width in {512,1024,2048,4096,32768}, M divisible by the
route's K (halving K is tried first). Everything else falls through to the
generic implementations unchanged.

Kill switch: FLAGGEMS_SPACEMIT_VAR_FAST=0 (fast paths only; the var_dim
signature fix is always active).
"""

import math
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from ._dispatch import nested_dispatch

# N -> (K, SB, s1_chunk_width, s1_chunks); probe67 A sweep (f32 form).
_ROUTES = {
    512: (4, 256, 512, 1),
    1024: (8, 256, 1024, 1),
    2048: (8, 256, 2048, 1),
    4096: (4, 256, 4096, 1),
    32768: (2, 512, 8192, 4),
}

# wave10: f16 pass-1 loop width (W2). 256 wins the probe93 scan for the
# covered widths; 32768 gets 2048 (not swept; exactness verified probe93).
_F16_W2 = {512: 256, 1024: 256, 2048: 256, 4096: 512, 32768: 1024}

# wave12: routes whose mmk launch runs under nested_dispatch (N=512 loses).
_NEST_ROUTES = (1024, 2048, 4096, 32768)

# wave10 flat path: stage-1 squared-deviation chunk widths, and the midn cap
# beyond which the flat path declines (host fsum over more than ~8192
# partials stops being free; that means >128M elements).
_FLAT_SB = 4096
_FLAT_SB_F16 = 2048
_FLAT_HOST_MAX = 8192


def var_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_VAR_FAST", "1") != "0"


def _fsum(vals):
    """Exact f64 sum of the f32 stage-1 partials. -inf + inf makes math.fsum
    raise ValueError while the torch semantic is NaN - map it to NaN."""
    try:
        return math.fsum(vals)
    except ValueError:  # "-inf + inf in fsum"
        return float("nan")


@triton.jit
def _var_mmk(
    X, Var, Mean, N, CORR,
    K: tl.constexpr, W: tl.constexpr, SB: tl.constexpr,
    NS1: tl.constexpr, OUT_MEAN: tl.constexpr, SQRT: tl.constexpr,
    LOOP1: tl.constexpr, W2: tl.constexpr,
):
    """Two-pass wave4-style var per row (K rows/program).

    Pass 1 computes the row mean:
      LOOP1=False (f32): NS1 direct load-sums (a single full-row tl.sum when
        NS1=1). A cast-free load feeding tl.sum is fine on this backend.
      LOOP1=True (f16): chunked loop-carried accumulator at width W2 - a
        dtype cast feeding tl.sum directly is the spine feeding cliff.
    Pass 2: BLOCK=SB mul-chain accumulator of (x-m)^2 + one tl.sum over the
    accumulator (tl.sum over a fresh elementwise vector is the cliff).
    """
    p = tl.program_id(0)
    for i in tl.static_range(K):
        r = p * K + i
        base = X + r * N
        s1 = 0.0
        if LOOP1:
            acc1 = tl.zeros((W2,), dtype=tl.float32)
            for c in range(0, N, W2):
                xv1 = tl.load(base + c + tl.arange(0, W2)).to(tl.float32)
                acc1 += xv1
            s1 = tl.sum(acc1, axis=0)
        else:
            for c in tl.static_range(NS1):
                xv = tl.load(base + c * W + tl.arange(0, W))
                s1 += tl.sum(xv, axis=0)
        m = s1 / N
        acc = tl.zeros((SB,), dtype=tl.float32)
        for c in range(0, N, SB):
            x = tl.load(base + c + tl.arange(0, SB)).to(tl.float32)
            d = x - m
            acc += d * d
        s2 = tl.sum(acc, axis=0)
        v = s2 / (N - CORR)
        if SQRT:
            v = tl.sqrt(v)
        tl.store(Var + r, v.to(Var.dtype.element_ty))
        if OUT_MEAN:
            tl.store(Mean + r, m.to(Mean.dtype.element_ty))


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
    is16 = x.dtype in (torch.float16, torch.bfloat16)
    # wave12: nest gate — every probed route except N=512 wins (probe104/105:
    # 1.16-1.69x; N=512 loses 0.55-0.62). 16-bit stays plain (untested form).
    nest = (
        not is16
        and k >= 2
        and N in _NEST_ROUTES
        and os.environ.get("FLAGGEMS_SPACEMIT_VAR_NEST", "1") != "0"
    )
    if nest:
        with nested_dispatch(M * N):
            _var_mmk[(M // k,)](
                x, var_out, mean_out, N, float(corr),
                K=k, W=w1, SB=sb, NS1=ns1,
                OUT_MEAN=bool(want_mean), SQRT=bool(sqrt),
                LOOP1=is16, W2=(_F16_W2.get(N, 256) if is16 else 0),
            )
    else:
        _var_mmk[(M // k,)](
            x, var_out, mean_out, N, float(corr),
            K=k, W=w1, SB=sb, NS1=ns1,
            OUT_MEAN=bool(want_mean), SQRT=bool(sqrt),
            LOOP1=is16, W2=(_F16_W2.get(N, 256) if is16 else 0),
        )
    return True


# ---------------- wave13: var over dim=0 (non-trailing) ----------------
# Row-fold two-stage, two passes (cancellation-free like wave7): stage1 sums
# per row block, combine -> mean, stage1b sums squared deviations, combine ->
# var. probe111 same-session vs gems generic / native torch:
#   2048^2 4.18 (357.2 / 54.8), 8192x512 3.82 (77.5 / 57.8),
#   512x32768 18.7 (2995 / 236), 1024x4096 4.24 (388 / 43),
#   4096x1024 3.44 (356 / 51); maxabs <= 8.4e-7 vs f64 incl. large-mean.
_V0_RB = 256
_V0_BN = 1024


@triton.jit
def _var0_stage1(X, PS, M, N, RB: tl.constexpr, BN: tl.constexpr):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    r0 = rb * RB
    r1 = tl.minimum(r0 + RB, M)
    cols = cb * BN + tl.arange(0, BN)
    cm = cols < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for r in range(r0, r1):
        off = r * N + cols
        v = tl.load(X + off, mask=cm, other=0.0).to(tl.float32)
        acc += v
    tl.store(PS + rb * N + cols, acc, mask=cm)


@triton.jit
def _var0_stage1b(X, MU, PQ, M, N, RB: tl.constexpr, BN: tl.constexpr):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    r0 = rb * RB
    r1 = tl.minimum(r0 + RB, M)
    cols = cb * BN + tl.arange(0, BN)
    cm = cols < N
    mu = tl.load(MU + cols, mask=cm, other=0.0)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for r in range(r0, r1):
        off = r * N + cols
        v = tl.load(X + off, mask=cm, other=0.0).to(tl.float32)
        d = v - mu
        acc += d * d
    tl.store(PQ + rb * N + cols, acc, mask=cm)


@triton.jit
def _var0_comb(P, OUT, N, NRB, DEN, BN: tl.constexpr, SQRT: tl.constexpr):
    cb = tl.program_id(0)
    cols = cb * BN + tl.arange(0, BN)
    cm = cols < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for s in range(0, NRB):
        acc += tl.load(P + s * N + cols, mask=cm, other=0.0)
    r = acc / DEN
    if SQRT:
        r = tl.sqrt(r)
    tl.store(OUT + cols, r.to(OUT.dtype.element_ty), mask=cm)


def var_dim0_launch(
    x, var_out, mean_out, M, N, corr, want_mean, sqrt
) -> bool:
    """dim=0 fast path (2D contiguous input). Returns False when out of scope."""
    if M - float(corr) <= 0:
        return False
    rb = _V0_RB
    if rb > M:
        rb = triton.next_power_of_2(M)
    bn = _V0_BN if N >= _V0_BN else triton.next_power_of_2(N)
    nrb = triton.cdiv(M, rb)
    nblk = triton.cdiv(N, bn)
    ps = torch.empty(nrb, N, dtype=torch.float32, device=x.device)
    pq = torch.empty(nrb, N, dtype=torch.float32, device=x.device)
    mu = torch.empty(N, dtype=torch.float32, device=x.device)
    with torch_device_fn.device(x.device):
        _var0_stage1[(nrb, nblk)](x, ps, M, N, rb, bn)
        _var0_comb[(nblk,)](ps, mu, N, nrb, float(M), bn, False)
        _var0_stage1b[(nrb, nblk)](x, mu, pq, M, N, rb, bn)
        _var0_comb[(nblk,)](
            pq, var_out, N, nrb, float(M) - float(corr), bn, bool(sqrt)
        )
    if want_mean:
        mean_out.copy_(mu)
    return True


# ---------------- wave10: flat (all-elements) two-stage ----------------
def _flat_block(M: int) -> int:
    """Stage-1 BLOCK rule (same shape as sum_fast._flat_block; probe60)."""
    b = triton.next_power_of_2(max(4096, -(-M // 64)))
    if b > 16384:
        b = 16384
    p2 = triton.next_power_of_2(M)
    if b > p2:
        b = p2
    if b < 32:
        b = 32  # tl.sum reduce extent < 32 lowers to NaN on this backend
    return b


@triton.jit
def _var_flat_sum_k1d(X, MID, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + off, mask=off < M, other=0.0)
    tl.store(MID + pid, tl.sum(v, axis=0))


@triton.jit
def _var_flat_sum_k1l(X, MID, M, BLOCK: tl.constexpr, SB: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    acc = tl.zeros((SB,), dtype=tl.float32)
    for c in range(0, BLOCK, SB):
        off = base + c + tl.arange(0, SB)
        v = tl.load(X + off, mask=off < M, other=0.0)
        acc += v.to(tl.float32)
    tl.store(MID + pid, tl.sum(acc, axis=0))


@triton.jit
def _var_flat_sq_k1(
    X, MID, m, M, BLOCK: tl.constexpr, SB: tl.constexpr, NOMASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    acc = tl.zeros((SB,), dtype=tl.float32)
    for c in range(0, BLOCK, SB):
        off = base + c + tl.arange(0, SB)
        if NOMASK:
            v = tl.load(X + off).to(tl.float32)
            d = v - m
        else:
            mm = off < M
            v = tl.load(X + off, mask=mm, other=0.0).to(tl.float32)
            d = tl.where(mm, v - m, 0.0)
        acc += d * d
    tl.store(MID + pid, tl.sum(acc, axis=0))


def var_flat_launch(x, corr, want_mean=False, sqrt=False):
    """Flat var/std/var_mean over ALL elements (two kernels + f64 combine).

    Returns the result tensor (and the mean tensor when want_mean), or None
    when out of scope. Two-pass (mean first, then squared deviations) so the
    result is cancellation-free; the combine is exact f64 (math.fsum) - every
    probe check matched torch (probe91a/92/93). Same-session 1M: f32 8.64 ->
    1.06 ms (torch 1.07); f16 8.50 -> 0.99 ms (torch 4.40).
    """
    n = x.numel()
    if n <= 0:
        return None
    block = _flat_block(n)
    midn = triton.cdiv(n, block)
    if midn > _FLAT_HOST_MAX:
        return None
    is16 = x.dtype in (torch.float16, torch.bfloat16)
    sb = _FLAT_SB_F16 if is16 else _FLAT_SB
    mid = torch.empty(midn, dtype=torch.float32, device=x.device)
    with torch_device_fn.device(x.device):
        if is16:
            _var_flat_sum_k1l[(midn,)](x, mid, n, block, sb)
        else:
            _var_flat_sum_k1d[(midn,)](x, mid, n, block)
        m = _fsum(mid.tolist()) / n
        _var_flat_sq_k1[(midn,)](x, mid, m, n, block, sb, n % block == 0)
        sq = _fsum(mid.tolist())
    val = sq / (n - float(corr))
    if sqrt:
        val = math.sqrt(val)
    var_t = torch.tensor(val, dtype=x.dtype, device=x.device)
    if want_mean:
        return var_t, torch.tensor(m, dtype=x.dtype, device=x.device)
    return var_t
