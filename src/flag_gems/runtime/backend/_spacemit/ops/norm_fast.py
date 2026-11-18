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

"""wave18/19: p=2, p=1 and p=inf norm fast paths (rows + flat).

Generic norm keeps a (BLOCK_M, BLOCK_N) tile and ends with tl.sum(_sum, axis=1);
on spine-triton that lowers elementwise-scalar (2.63 / 16.00 / 4.20 / 39.47 ms
at 512^2 / 2048^2 / 512x4096 / 8192x512 for dim=-1 rows = 0.06-0.16x torch).

Same carrier as prod/log_softmax: narrow vector accumulator, one final reduce
per row. probe162/162b (p=2, f32):

    shape        generic   fast rows      torch
    512x512        2.63      0.352         0.166
    2048x2048     16.00      2.449         1.841
    512x4096       4.20      1.061         0.734
    8192x512      39.47      4.493         2.286

Flat (dim=None) two-stage probe162b: 0.232 / 1.771 / 0.930 ms for
512^2 / 2048^2 / 512x4096 - 1.76-2.04x FASTER than torch.norm on the two large
shapes, 0.93x at 512^2.

wave19 adds ord=1 / ord=inf carriers (probe169b, f32 rows):

    shape        ord1 generic  ord1 fast   ordinf generic  ordinf fast
    512x512          23.56      0.347          2.86         0.350
    2048x2048       193.23      2.419         13.11         2.400
    512x4096         91.98      1.038          4.26         1.044
    8192x512        368.23      4.356         39.24         4.231

flat: ord=1 0.260/0.535/1.772 ms and ord=inf 0.259/0.559/1.788 ms at
512^2 / 1024^2 / 2048^2 (generic 11.3/44.0/172.5 and 1.9/6.5/23.0; torch
0.161/0.499/1.887 and 0.279/0.929/3.658).

Also fixes the generic dim=None crash (out.squeeze(dim=None) raises on this
torch build: "Please look up dimensions by name, got: name = None").

Kill switch: FLAGGEMS_SPACEMIT_NORM_FAST=0. f32 + contiguous only.

wave19 negative result: a 4-accumulator ILP row kernel never enters the top-5
on any of the four shapes - the row kernel is final-reduce bound, not
load-latency bound, so extra accumulators only cost registers.
"""

import os

import torch
import triton
import triton.language as tl

_ROWS = 4
_BLOCK_SMALL = 128   # N <= 2048
_BLOCK_LARGE = 256   # N > 2048
_MIN_N = 64
_FLAT_CHUNK = 32768
_FLAT_CHUNK_SMALL = 8192
_FLAT_BLOCK = 256


@triton.jit
def _l2_rows(X, Out, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
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
            v = tl.load(X + base + idx, mask=mk, other=0.0).to(tl.float32)
            acc += v * v
        tl.store(Out + r, tl.sqrt(tl.sum(acc, axis=0)).to(Out.dtype.element_ty), mask=rmask)


@triton.jit
def _l1_rows(X, Out, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
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
            v = tl.load(X + base + idx, mask=mk, other=0.0).to(tl.float32)
            acc += tl.abs(v)
        tl.store(Out + r, tl.sum(acc, axis=0).to(Out.dtype.element_ty), mask=rmask)


@triton.jit
def _linf_rows(X, Out, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
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
            v = tl.load(X + base + idx, mask=mk, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(v))
        tl.store(Out + r, tl.max(acc, axis=0).to(Out.dtype.element_ty), mask=rmask)


@triton.jit
def _l2_flat_stage1(X, Mid, M, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK):
        idx = pid.to(tl.int64) * CHUNK + off + cols
        v = tl.load(X + idx, mask=idx < M, other=0.0).to(tl.float32)
        acc += v * v
    tl.store(Mid + pid, tl.sum(acc, axis=0))


@triton.jit
def _l2_flat_stage2(Mid, Out, NB, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, NB, BLOCK):
        idx = off + cols
        v = tl.load(Mid + idx, mask=idx < NB, other=0.0)
        acc += v
    tl.store(Out, tl.sqrt(tl.sum(acc, axis=0)).to(Out.dtype.element_ty))


@triton.jit
def _l1_flat_stage1(X, Mid, M, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK):
        idx = pid.to(tl.int64) * CHUNK + off + cols
        v = tl.load(X + idx, mask=idx < M, other=0.0).to(tl.float32)
        acc += tl.abs(v)
    tl.store(Mid + pid, tl.sum(acc, axis=0))


@triton.jit
def _l1_flat_stage2(Mid, Out, NB, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, NB, BLOCK):
        idx = off + cols
        v = tl.load(Mid + idx, mask=idx < NB, other=0.0)
        acc += v
    tl.store(Out, tl.sum(acc, axis=0).to(Out.dtype.element_ty))


@triton.jit
def _linf_flat_stage1(X, Mid, M, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK):
        idx = pid.to(tl.int64) * CHUNK + off + cols
        v = tl.load(X + idx, mask=idx < M, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(v))
    tl.store(Mid + pid, tl.max(acc, axis=0))


@triton.jit
def _linf_flat_stage2(Mid, Out, NB, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, NB, BLOCK):
        idx = off + cols
        v = tl.load(Mid + idx, mask=idx < NB, other=0.0)
        acc = tl.maximum(acc, v)
    tl.store(Out, tl.max(acc, axis=0).to(Out.dtype.element_ty))


def norm_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_NORM_FAST", "1") != "0"


def _row_block(N):
    block = _BLOCK_SMALL if N <= 2048 else _BLOCK_LARGE
    return min(block, triton.next_power_of_2(N))


def norm_rows_launch(x, out, M, N) -> bool:
    if M <= 0 or N < _MIN_N:
        return False
    _l2_rows[(triton.cdiv(M, _ROWS),)](x, out, M, N, K=_ROWS, BLOCK_N=_row_block(N))
    return True


def l1_rows_launch(x, out, M, N) -> bool:
    if M <= 0 or N < _MIN_N:
        return False
    _l1_rows[(triton.cdiv(M, _ROWS),)](x, out, M, N, K=_ROWS, BLOCK_N=_row_block(N))
    return True


def linf_rows_launch(x, out, M, N) -> bool:
    if M <= 0 or N < _MIN_N:
        return False
    _linf_rows[(triton.cdiv(M, _ROWS),)](x, out, M, N, K=_ROWS, BLOCK_N=_row_block(N))
    return True


def _flat_blocks(numel):
    chunk = _FLAT_CHUNK_SMALL if numel <= (1 << 20) else _FLAT_CHUNK
    nblk = triton.cdiv(numel, chunk)
    return chunk, nblk, max(32, triton.next_power_of_2(min(nblk, 1024)))


def norm_flat_launch(x, out, numel) -> bool:
    if numel < _MIN_N:
        return False
    chunk, nblk, b2 = _flat_blocks(numel)
    mid = torch.empty((nblk,), dtype=torch.float32, device=x.device)
    _l2_flat_stage1[(nblk,)](x, mid, numel, CHUNK=chunk, BLOCK=_FLAT_BLOCK)
    _l2_flat_stage2[(1,)](mid, out, nblk, BLOCK=b2)
    return True


def l1_flat_launch(x, out, numel) -> bool:
    if numel < _MIN_N:
        return False
    chunk, nblk, b2 = _flat_blocks(numel)
    mid = torch.empty((nblk,), dtype=torch.float32, device=x.device)
    _l1_flat_stage1[(nblk,)](x, mid, numel, CHUNK=chunk, BLOCK=_FLAT_BLOCK)
    _l1_flat_stage2[(1,)](mid, out, nblk, BLOCK=b2)
    return True


def linf_flat_launch(x, out, numel) -> bool:
    if numel < _MIN_N:
        return False
    chunk, nblk, b2 = _flat_blocks(numel)
    mid = torch.empty((nblk,), dtype=torch.float32, device=x.device)
    _linf_flat_stage1[(nblk,)](x, mid, numel, CHUNK=chunk, BLOCK=_FLAT_BLOCK)
    _linf_flat_stage2[(1,)](mid, out, nblk, BLOCK=b2)
    return True
