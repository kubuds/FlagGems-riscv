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

"""wave3 (campaign 1014night): softmax_backward direct kernels.

Design ("two-loop acc-vector", probe29/30/31 measurements, f32 same-session):
- one program per row; loop1 accumulates t = g*y into a register VECTOR
  (elementwise, vectorized); a SINGLE tl.sum of that vector follows (acc
  segment >= 2048 lowers vectorized; below that scalar-lowered); loop2
  recomputes g,y and stores dx = y*(g - s).
- 2048^2: 5.07ms (common+nested: 10.7ms; torch: 3.96-4.65ms); 512^2: 0.55ms
  (common: 0.85ms; torch: 0.20-0.31ms). f32 maxabs vs torch ~ 4e-9.
- Rejected structures (all slower): single-pass register-resident multi-row
  blocks (sp_BM1..8: 2x-20x slower; the product feeding tl.sum in one
  expression kills the reduce), padded-BLOCK reduce for narrow rows
  (0.55 -> 1.39ms), split reduce/write kernels (v1: 38ms @2048^2).
- Gates: last-dim only, contiguous-izable, same-dtype f32/f16 chain,
  numel >= 65536, rows >= 2.
- Kill switches: FLAGGEMS_SPACEMIT_SB_FAST=0, or NESTED=0 (two-pass A/B).
"""

import os

import torch
import triton
import triton.language as tl

_SB_MIN_N = 1 << 16
_SB_MAX_BLOCK = 1 << 13
# wave5 (campaign 1014pm): K rows per program. Runtime-n form only - the
# constexpr-n form of the same kernel compiles 3-5x slower on spine (probe55).
_SB_ROWS_K = 4


def sb_fast_enabled() -> bool:
    return (
        os.environ.get("FLAGGEMS_SPACEMIT_SB_FAST", "1") != "0"
        and os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") != "0"
    )


def sb_ok(grad_output, output, dim, input_dtype) -> bool:
    if not (
        isinstance(grad_output, torch.Tensor) and isinstance(output, torch.Tensor)
    ):
        return False
    if grad_output.shape != output.shape:
        return False
    if grad_output.dtype != output.dtype or grad_output.dtype not in (
        torch.float32,
        torch.float16,
    ):
        return False
    if input_dtype != grad_output.dtype:
        return False
    numel = output.numel()
    if numel < _SB_MIN_N or output.ndim < 2:
        return False
    if (dim % output.ndim) != output.ndim - 1:
        return False
    if numel // output.shape[-1] < 2:
        return False
    return True


@triton.jit
def _sb_row_kernel(
    g_ptr, y_ptr, dx_ptr, n_cols, BLOCK: tl.constexpr, NOMASK: tl.constexpr
):
    row = tl.program_id(0)
    base = row * n_cols
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in range(0, n_cols, BLOCK):
        offs = c + tl.arange(0, BLOCK)
        if NOMASK:
            g = tl.load(g_ptr + base + offs).to(tl.float32)
            y = tl.load(y_ptr + base + offs).to(tl.float32)
            acc += g * y
        else:
            mm = offs < n_cols
            g = tl.load(g_ptr + base + offs, mask=mm, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + base + offs, mask=mm, other=0.0).to(tl.float32)
            acc += g * y
    total = tl.sum(acc, axis=0)
    for c in range(0, n_cols, BLOCK):
        offs = c + tl.arange(0, BLOCK)
        if NOMASK:
            g = tl.load(g_ptr + base + offs).to(tl.float32)
            y = tl.load(y_ptr + base + offs).to(tl.float32)
            tl.store(
                dx_ptr + base + offs, (y * (g - total)).to(dx_ptr.dtype.element_ty)
            )
        else:
            mm = offs < n_cols
            g = tl.load(g_ptr + base + offs, mask=mm, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + base + offs, mask=mm, other=0.0).to(tl.float32)
            tl.store(
                dx_ptr + base + offs,
                (y * (g - total)).to(dx_ptr.dtype.element_ty),
                mask=mm,
            )


@triton.jit
def _sb_rows_kernel(
    g_ptr, y_ptr, dx_ptr, n_cols, BLOCK: tl.constexpr, NOMASK: tl.constexpr,
    K: tl.constexpr,
):
    # campaign 1014pm (wave5): K rows per program to amortize the launcher cost.
    # Structure mirrors _sb_row_kernel exactly (runtime n_cols!). Measured
    # (probe56, f32): 512^2 0.539->0.405 ms (1.33x), 2048^2 5.14->4.77 (1.08x)
    # with K=4; K=8/16 slightly worse. Kill switch: FLAGGEMS_SPACEMIT_SB_ROWS=0.
    p = tl.program_id(0)
    for r in tl.static_range(K):
        row = p * K + r
        base = row * n_cols
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(0, n_cols, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            if NOMASK:
                g = tl.load(g_ptr + base + offs).to(tl.float32)
                y = tl.load(y_ptr + base + offs).to(tl.float32)
                acc += g * y
            else:
                mm = offs < n_cols
                g = tl.load(g_ptr + base + offs, mask=mm, other=0.0).to(tl.float32)
                y = tl.load(y_ptr + base + offs, mask=mm, other=0.0).to(tl.float32)
                acc += g * y
        total = tl.sum(acc, axis=0)
        for c in range(0, n_cols, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            if NOMASK:
                g = tl.load(g_ptr + base + offs).to(tl.float32)
                y = tl.load(y_ptr + base + offs).to(tl.float32)
                tl.store(
                    dx_ptr + base + offs,
                    (y * (g - total)).to(dx_ptr.dtype.element_ty),
                )
            else:
                mm = offs < n_cols
                g = tl.load(g_ptr + base + offs, mask=mm, other=0.0).to(tl.float32)
                y = tl.load(y_ptr + base + offs, mask=mm, other=0.0).to(tl.float32)
                tl.store(
                    dx_ptr + base + offs,
                    (y * (g - total)).to(dx_ptr.dtype.element_ty),
                    mask=mm,
                )


def _sb_block(ncols: int) -> int:
    b = 1
    while b < ncols and b < _SB_MAX_BLOCK:
        b <<= 1
    return b


def sb_fast(grad_output, output, dim, input_dtype, grad_input=None):
    g = grad_output.contiguous()
    y = output.contiguous()
    ncols = y.shape[-1]
    rows = y.numel() // ncols
    block = _sb_block(ncols)
    nomask = (ncols % block) == 0
    dx = grad_input if grad_input is not None else torch.empty_like(y, dtype=input_dtype)
    k = _SB_ROWS_K
    if k > 1 and rows % k == 0 and os.environ.get("FLAGGEMS_SPACEMIT_SB_ROWS", "1") != "0":
        _sb_rows_kernel[(rows // k,)](g, y, dx, ncols, BLOCK=block, NOMASK=nomask, K=k)
    else:
        _sb_row_kernel[(rows,)](g, y, dx, ncols, BLOCK=block, NOMASK=nomask)
    return dx
