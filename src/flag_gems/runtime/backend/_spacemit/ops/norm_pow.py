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

"""wave22: generic p-norm carriers for ord in {4, -1, -2, -3, -4}.

The wave18/19 carriers cover ord 0/1/2/3/inf/-inf only; every other ord falls
into the generic flag_gems kernel, which lowers its (BLOCK_M, BLOCK_N) tile
reduce elementwise-scalar (0.06-0.16x torch on the row shapes, probe162).

Same narrow-carrier structure as norm_fast: one program per K rows, a vector
accumulator, a single final reduce per row, then the p-th root in scalar form.
Negative orders accumulate 1/|v|^p and invert at the end, so a zero in the row
gives 1/0 = inf -> sum inf -> root inf -> 1/inf = 0, which is what torch
returns.  Masked lanes load +inf for the negative orders (1/inf = 0) and 0 for
the positive ones, so padding never contributes.
"""

import os

import torch
import triton
import triton.language as tl

_POW_ROWS_K = 8
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
             P: tl.constexpr, NEG: tl.constexpr):
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
                acc += 1.0 / _pw(v, P)
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
                    P: tl.constexpr, NEG: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK):
        idx = pid.to(tl.int64) * CHUNK + off + cols
        if NEG:
            v = tl.load(X + idx, mask=idx < M, other=float("inf")).to(tl.float32)
            acc += 1.0 / _pw(v, P)
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


def _row_block(N):
    if N >= 4096:
        return 2048
    return min(_POW_BLOCK, triton.next_power_of_2(N))


def lp_rows_launch(x, out, M, N, P, NEG) -> bool:
    if M <= 0 or N <= 0 or not lp_fast_enabled():
        return False
    k = _POW_ROWS_K if M % _POW_ROWS_K == 0 else 1
    bn = _row_block(N)
    _lp_rows[(triton.cdiv(M, k),)](x, out, M, N, K=k, BLOCK_N=bn, P=P, NEG=NEG)
    return True


def lp_flat_launch(x, out, numel, P, NEG) -> bool:
    if numel <= 0 or not lp_fast_enabled():
        return False
    nb = triton.cdiv(numel, _POW_FLAT_CHUNK)
    mid = torch.empty((nb,), dtype=torch.float32, device=x.device)
    _lp_flat_stage1[(nb,)](x, mid, numel, CHUNK=_POW_FLAT_CHUNK, BLOCK=_POW_FLAT_BLOCK,
                           P=P, NEG=NEG)
    _lp_flat_stage2[(1,)](mid, out, nb, BLOCK=min(_POW_FLAT_BLOCK, triton.next_power_of_2(nb)),
                          P=P, NEG=NEG)
    return True
