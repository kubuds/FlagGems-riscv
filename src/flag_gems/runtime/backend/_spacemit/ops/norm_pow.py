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

"""wave22/23: generic p-norm carriers for ord in {4, -1, -2, -3, -4}.

wave22 added the carriers with a plain `1.0 / x`, which lowers to a scalar
division at ~5.5 ns/element inside a reduction loop (probe202) and kept the
negative orders at 1-4x torch.

wave23 replaces that division with a bit-trick seed plus two Newton steps
(`fastmath.recip_fast`, probe205: 1.60 -> 0.51 ns/element standalone, max
relative error 6.7e-6).  Exact zeros still return +inf (-> 1/inf = 0, matching
torch) and masked lanes still load +inf.
"""

import os

import torch
import triton
import triton.language as tl

from ._dispatch import nested_dispatch
from .fastmath import recip_fast

_POW_ROWS_K = 16
_POW_BLOCK = 2048
_POW_FLAT_CHUNK = 1 << 16
_POW_FLAT_BLOCK = 2048


@triton.jit
def _pw(v, P: tl.constexpr):
    a = tl.abs(v)
    if P == 1:
        return a
    if P == 2:
        return a * a
    if P == 3:
        return a * a * a
    a2 = a * a
    return a2 * a2


@triton.jit
def _root(s, P: tl.constexpr):
    if P == 1:
        return s
    if P == 2:
        return tl.sqrt(s)
    if P == 3:
        return tl.exp(tl.log(s) * (1.0 / 3.0))
    return tl.sqrt(tl.sqrt(s))


@triton.jit
def _lp_rows(X, Out, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr,
             P: tl.constexpr, NEG: tl.constexpr, RF: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    for i in tl.static_range(K):
        r = pid * K + i
        rmask = r < M
        base = r.to(tl.int64) * N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            idx = off + cols
            mk = (idx < N) & rmask
            if NEG:
                v = tl.load(X + base + idx, mask=mk, other=float("inf")).to(tl.float32)
                acc += recip_fast(_pw(v, P), RF)
            else:
                v = tl.load(X + base + idx, mask=mk, other=0.0).to(tl.float32)
                acc += _pw(v, P)
        s = tl.sum(acc, axis=0)
        if NEG:
            res = 1.0 / _root(s, P)
        else:
            res = _root(s, P)
        tl.store(Out + r, res.to(Out.dtype.element_ty), mask=rmask)


@triton.jit
def _lp_flat_stage1(X, Mid, M, CHUNK: tl.constexpr, BLOCK: tl.constexpr,
                    P: tl.constexpr, NEG: tl.constexpr, RF: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK):
        idx = pid.to(tl.int64) * CHUNK + off + cols
        if NEG:
            v = tl.load(X + idx, mask=idx < M, other=float("inf")).to(tl.float32)
            acc += recip_fast(_pw(v, P), RF)
        else:
            v = tl.load(X + idx, mask=idx < M, other=0.0).to(tl.float32)
            acc += _pw(v, P)
    tl.store(Mid + pid, tl.sum(acc, axis=0))


@triton.jit
def _lp_flat_stage2(Mid, Out, NB, BLOCK: tl.constexpr, P: tl.constexpr,
                    NEG: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, NB, BLOCK):
        idx = off + cols
        v = tl.load(Mid + idx, mask=idx < NB, other=0.0)
        acc += v
    s = tl.sum(acc, axis=0)
    if NEG:
        res = 1.0 / _root(s, P)
    else:
        res = _root(s, P)
    tl.store(Out, res.to(Out.dtype.element_ty))


def lp_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_NORM_FAST", "1") != "0"


def _recip_flag() -> int:
    return 0 if os.environ.get("FLAGGEMS_SPACEMIT_FASTMATH_RECIP", "1") == "0" else 1


def _row_block(N):
    if N >= 4096:
        return 2048
    return min(_POW_BLOCK, triton.next_power_of_2(N))


def lp_rows_launch(x, out, M, N, P, NEG) -> bool:
    if M <= 0 or N <= 0 or not lp_fast_enabled():
        return False
    k = _POW_ROWS_K
    while k > 1 and M % k:
        k //= 2
    bn = _row_block(N)
    # wave29: nested dispatch + 16 rows/program (probe293: ord=4 2048^2
    # 3.15 -> 1.46, 8192x512 5.16 -> 3.38 ms; K8 under nesting leaves 1.5x
    # on the table, K16 without nesting is a wash).
    with nested_dispatch(M * N):
        _lp_rows[(M // k,)](x, out, M, N, K=k, BLOCK_N=bn, P=P, NEG=NEG,
                            RF=_recip_flag())
    return True


def lp_flat_launch(x, out, numel, P, NEG) -> bool:
    if numel <= 0 or not lp_fast_enabled():
        return False
    nb = triton.cdiv(numel, _POW_FLAT_CHUNK)
    mid = torch.empty((nb,), dtype=torch.float32, device=x.device)
    _lp_flat_stage1[(nb,)](x, mid, numel, CHUNK=_POW_FLAT_CHUNK, BLOCK=_POW_FLAT_BLOCK,
                           P=P, NEG=NEG, RF=_recip_flag())
    _lp_flat_stage2[(1,)](mid, out, nb, BLOCK=min(_POW_FLAT_BLOCK, triton.next_power_of_2(nb)),
                          P=P, NEG=NEG)
    return True
