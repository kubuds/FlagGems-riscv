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

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.softmax import softmax_backward as common_softmax_backward
from flag_gems.ops.softmax import softmax_out as common_softmax_out
from flag_gems.ops.softmax import (
    softmax_backward_out as common_softmax_backward_out,
)
from flag_gems.runtime.backend._spacemit.ops.softmax_backward_fast import (
    sb_fast as _sb_fast,
    sb_fast_enabled as _sb_fast_enabled,
    sb_ok as _sb_ok,
)
from flag_gems.utils import tl_extra_shim

from flag_gems.runtime.backend._spacemit.ops._dispatch import nested_dispatch
from flag_gems.runtime.backend._spacemit.ops.fastmath import exp_fast
from flag_gems.runtime.backend._spacemit.ops.memhint import hugepage_hint

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp


@triton.jit
def softmax_kernel_spacemit(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    ROW_SIZE: tl.constexpr,
    COL_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0) * ROW_SIZE
    element_ty = output_ptr.type.element_ty

    for row_idx in range(row_start, row_start + ROW_SIZE):
        if row_idx < n_rows:
            denominator = tl.zeros((1,), dtype=tl.float32)
            row_max = tl.full((COL_SIZE,), value=-float("inf"), dtype=tl.float32)

            for col_idx in range(0, n_cols, COL_SIZE):
                input_block_ptr = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(col_idx,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                row = tl.load(
                    input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                ).to(tl.float32)
                row_max = tl.maximum(row, row_max)

            row_max_total = tl.max(row_max, axis=0)

            for col_idx in range(0, n_cols, COL_SIZE):
                input_block_ptr = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(col_idx,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                output_block_ptr = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(col_idx,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                row = tl.load(
                    input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                ).to(tl.float32)
                numerator = exp(row - row_max_total)
                denominator += tl.sum(numerator, axis=0)
                tl.store(
                    output_block_ptr, numerator.to(element_ty), boundary_check=(0,)
                )

            inv_denom = 1.0 / denominator
            for col_idx in range(0, n_cols, COL_SIZE):
                output_block_ptr = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(col_idx,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                exp_out = tl.load(output_block_ptr, boundary_check=(0,)).to(tl.float32)
                tl.store(
                    output_block_ptr,
                    (exp_out * inv_denom).to(element_ty),
                    boundary_check=(0,),
                )


@triton.jit
def softmax_kernel_block(input_ptr, output_ptr, n_cols, COL: tl.constexpr):
    """wave22: whole row resident in one block (f16, COL == n_cols)."""
    row = tl.program_id(0)
    base = row.to(tl.int64) * n_cols
    offs = tl.arange(0, COL)
    x = tl.load(input_ptr + base + offs).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    # wave26: multiply by one reciprocal instead of dividing every lane.  The
    # per-element divide is not hoisted by this backend: probe262 priced a 1G
    # f32 row-resident softmax at 1133.9 ms with `e / s` and 832.0 ms with
    # `e * (1 / s)` - 27% of the whole kernel at 1024^3, 34% at 4096^2.
    inv = 1.0 / s
    tl.store(output_ptr + base + offs, (e * inv).to(output_ptr.dtype.element_ty))


# wave24: the resident row is not just a f16 trick, and the column loop is what
# costs the time.  probe228 priced one 1M f32 row: copy 0.46 ns/elem, element-wise
# max 0.29, exp+store 0.53, but a tl.sum over a block *inside* the loop 5.35 - the
# block reduction is scalar-lowered, so every 256-column chunk of the three-pass
# path pays it twice per row.  Keeping the row (or ROWS rows) in one block leaves
# exactly two reductions per row and one load plus one store:
# 4096^2 29.1 -> 17.0, 32768x512 60.9 -> 17.4, 8192x512 15.1 -> 5.8,
# 1024^2 2.8 -> 1.6 ms (probe229/230b/231).  ROWS is a plain dynamic loop on
# purpose: static_range unrolls it (128 rows = 62 s of compile, and slower).
@triton.jit
def softmax_kernel_rowres(X, Y, n_cols, ROWS: tl.constexpr, COL: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, COL)
    for i in range(ROWS):
        r = pid * ROWS + i
        base = r.to(tl.int64) * n_cols
        x = tl.load(X + base + offs).to(tl.float32)
        m = tl.max(x, axis=0)
        e = exp_fast(x - m, 1)
        s = tl.sum(e, axis=0)
        inv = 1.0 / s          # wave26: one scalar reciprocal per row, see below
        tl.store(Y + base + offs, (e * inv).to(Y.dtype.element_ty))


@triton.jit
def softmax_kernel_rowres_m(X, Y, n_cols, ROWS: tl.constexpr, COL: tl.constexpr):
    """Row resident, masked: n_cols is not a power of two (COL = next_pow2)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, COL)
    mo = offs < n_cols
    for i in range(ROWS):
        r = pid * ROWS + i
        base = r.to(tl.int64) * n_cols
        x = tl.load(X + base + offs, mask=mo, other=-float("inf")).to(tl.float32)
        m = tl.max(x, axis=0)
        e = exp_fast(x - m, 1)
        s = tl.sum(e, axis=0)
        inv = 1.0 / s
        tl.store(Y + base + offs, (e * inv).to(Y.dtype.element_ty), mask=mo)


# wave24: a single row wider than any block (official shape 1048576 is one row).
# The live wrapper launches grid=1 and walks 4096 x 256 columns on one core.
# Two levels: per-chunk max and lane-wise sum of exp, one fold, one rescale pass.
_FLAT_CHUNK = 32768
_FLAT_BLOCK = 1024


@triton.jit
def _sm_flat_s1(X, MB, SB, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    offs = tl.arange(0, BLOCK)
    m = tl.full((BLOCK,), -float("inf"), dtype=tl.float32)
    for o in range(0, CHUNK, BLOCK):
        x = tl.load(X + base + o + offs).to(tl.float32)
        m = tl.maximum(m, x)
    mt = tl.max(m, axis=0)
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for o in range(0, CHUNK, BLOCK):
        x = tl.load(X + base + o + offs).to(tl.float32)
        s = s + exp_fast(x - mt, 1)
    tl.store(MB + pid, mt)
    tl.store(SB + pid, tl.sum(s, axis=0))


@triton.jit
def _sm_flat_s2(MB, SB, GM, GS, nblk, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mk = offs < nblk
    m = tl.load(MB + offs, mask=mk, other=-float("inf"))
    s = tl.load(SB + offs, mask=mk, other=0.0)
    gm = tl.max(m, axis=0)
    gs = tl.sum(s * exp_fast(m - gm, 1), axis=0)
    tl.store(GM, gm)
    tl.store(GS, gs)


@triton.jit
def _sm_flat_s3(X, Y, GM, GS, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    offs = tl.arange(0, BLOCK)
    gm = tl.load(GM)
    inv = 1.0 / tl.load(GS)
    for o in range(0, CHUNK, BLOCK):
        x = tl.load(X + base + o + offs).to(tl.float32)
        y = exp_fast(x - gm, 1) * inv
        tl.store(Y + base + o + offs, y.to(Y.dtype.element_ty))


def _softmax_flat(inp, out):
    """Two-level softmax for one very wide row (10.66 -> 1.58 ms f32, probe229)."""
    n = inp.numel()
    nblk = n // _FLAT_CHUNK
    mb = torch.empty(nblk, dtype=torch.float32)
    sb = torch.empty(nblk, dtype=torch.float32)
    gm = torch.empty(1, dtype=torch.float32)
    gs = torch.empty(1, dtype=torch.float32)
    with nested_dispatch(n):
        _sm_flat_s1[(nblk,)](inp, mb, sb, CHUNK=_FLAT_CHUNK, BLOCK=_FLAT_BLOCK)
        _sm_flat_s2[(1,)](mb, sb, gm, gs, nblk,
                          BLOCK=max(64, triton.next_power_of_2(nblk)))
        _sm_flat_s3[(nblk,)](inp, out, gm, gs, CHUNK=_FLAT_CHUNK, BLOCK=_FLAT_BLOCK)
    return out


# wave25: softmax over a dim that is not the last one.  Until now the vendor
# path reshaped the tensor to (numel / last, last) and reduced the LAST axis no
# matter which dim the caller asked for, so every non-last-dim call returned the
# softmax of the wrong axis (w24-dim-ignored: 2.1e-2 abs error, row sums off by
# 0.24 at (8,512,512) dim=1).  Turning dim into the last dim needs a transpose
# and every transpose on this board measures badly - torch's contiguous() copy
# 2.3 GB/s, a tiled Triton transpose 2.3 GB/s and wrong - so walk the reduce
# axis in place instead: one program owns a CB-wide slice of the contiguous tail
# and all n_cols values of the reduce axis, with 1-D state only (2-D tile
# reduces scalarise, wave21).
#
# Structure priced at (64,512,512) f32, probe244/247/248:
#   online (m,s), 2 accumulators + x = 6 KB state  98.0 ms  (register spills)
#   three passes, CB=512                           71.0 ms
#   three passes with exp cached in the output     56.5 ms
# and then probe250/251 showed the launch shape matters far more than the width:
# with nested dispatch on, CB = the whole contiguous tail is best (one wide
# stream per outer) - 19.2 ms at (64,512,512), 72.3 ms at (64,1024,1024),
# 66.3 ms at (32,1024,2048) - while splitting the tail in two halves costs 1.9x
# (134 vs 66 ms at inner=2048).  So CB = next_pow2(inner), capped at 2048.
_SM_COL_CB = 2048


@triton.jit
def softmax_col_max(X, MB, n_cols, inner, n_cb, CB: tl.constexpr):
    pid = tl.program_id(0)
    o = pid // n_cb
    cb = pid % n_cb
    offs = cb * CB + tl.arange(0, CB)
    base = o.to(tl.int64) * (n_cols * inner) + offs
    m = tl.full((CB,), -float("inf"), dtype=tl.float32)
    for j in range(n_cols):
        m = tl.maximum(m, tl.load(X + base + j * inner).to(tl.float32))
    tl.store(MB + pid.to(tl.int64) * CB + tl.arange(0, CB), m)


@triton.jit
def softmax_col_max_m(X, MB, n_cols, inner, n_cb, CB: tl.constexpr):
    pid = tl.program_id(0)
    o = pid // n_cb
    cb = pid % n_cb
    offs = cb * CB + tl.arange(0, CB)
    mask = offs < inner
    base = o.to(tl.int64) * (n_cols * inner) + offs
    m = tl.full((CB,), -float("inf"), dtype=tl.float32)
    for j in range(n_cols):
        x = tl.load(X + base + j * inner, mask=mask, other=-float("inf"))
        m = tl.maximum(m, x.to(tl.float32))
    tl.store(MB + pid.to(tl.int64) * CB + tl.arange(0, CB), m)


@triton.jit
def softmax_col_exp(X, Y, MB, SB, n_cols, inner, n_cb, CB: tl.constexpr):
    """Pass 2: write exp(x - max) into the output and accumulate its sum."""
    pid = tl.program_id(0)
    o = pid // n_cb
    cb = pid % n_cb
    r = tl.arange(0, CB)
    base = o.to(tl.int64) * (n_cols * inner) + cb * CB + r
    m = tl.load(MB + pid.to(tl.int64) * CB + r)
    s = tl.zeros((CB,), dtype=tl.float32)
    for j in range(n_cols):
        y = exp_fast(tl.load(X + base + j * inner).to(tl.float32) - m, 1)
        s += y
        tl.store(Y + base + j * inner, y.to(Y.dtype.element_ty))
    tl.store(SB + pid.to(tl.int64) * CB + r, 1.0 / s)


@triton.jit
def softmax_col_exp_m(X, Y, MB, SB, n_cols, inner, n_cb, CB: tl.constexpr):
    pid = tl.program_id(0)
    o = pid // n_cb
    cb = pid % n_cb
    r = tl.arange(0, CB)
    offs = cb * CB + r
    mask = offs < inner
    base = o.to(tl.int64) * (n_cols * inner) + offs
    m = tl.load(MB + pid.to(tl.int64) * CB + r)
    s = tl.zeros((CB,), dtype=tl.float32)
    for j in range(n_cols):
        x = tl.load(X + base + j * inner, mask=mask, other=-float("inf")).to(tl.float32)
        y = exp_fast(x - m, 1)
        s += tl.where(mask, y, 0.0)
        tl.store(Y + base + j * inner, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(SB + pid.to(tl.int64) * CB + r, 1.0 / s)


@triton.jit
def softmax_col_norm(Y, SB, n_cols, inner, n_cb, CB: tl.constexpr):
    """Pass 3: rescale the cached exp values in place."""
    pid = tl.program_id(0)
    o = pid // n_cb
    cb = pid % n_cb
    r = tl.arange(0, CB)
    base = o.to(tl.int64) * (n_cols * inner) + cb * CB + r
    inv = tl.load(SB + pid.to(tl.int64) * CB + r)
    for j in range(n_cols):
        y = tl.load(Y + base + j * inner).to(tl.float32)
        tl.store(Y + base + j * inner, (y * inv).to(Y.dtype.element_ty))


@triton.jit
def softmax_col_norm_m(Y, SB, n_cols, inner, n_cb, CB: tl.constexpr):
    pid = tl.program_id(0)
    o = pid // n_cb
    cb = pid % n_cb
    r = tl.arange(0, CB)
    offs = cb * CB + r
    mask = offs < inner
    base = o.to(tl.int64) * (n_cols * inner) + offs
    inv = tl.load(SB + pid.to(tl.int64) * CB + r)
    for j in range(n_cols):
        y = tl.load(Y + base + j * inner, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + base + j * inner, (y * inv).to(Y.dtype.element_ty), mask=mask)


def _softmax_nonlast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_SOFTMAX_NONLAST", "1") != "0"


def _softmax_col_cb() -> int:
    """Tunable column width for the non-last-dim walk (default 512)."""
    try:
        return max(32, int(os.environ.get("FLAGGEMS_SPACEMIT_SOFTMAX_COL_CB", _SM_COL_CB)))
    except ValueError:
        return _SM_COL_CB


def _softmax_nonlast(inp, out, dim):
    """Softmax over dim != last, walked in place over the reduce axis."""
    n_cols = inp.shape[dim]
    inner = 1
    for s in inp.shape[dim + 1:]:
        inner *= s
    outer = inp.numel() // (n_cols * inner)
    if inp.numel() == 0:
        return out
    cb = _softmax_col_cb()
    if inner < cb:
        cb = max(32, triton.next_power_of_2(inner))
    n_cb = (inner + cb - 1) // cb
    grid = outer * n_cb
    exact = (inner % cb) == 0
    mb = torch.empty(grid * cb, dtype=torch.float32)
    sb = torch.empty(grid * cb, dtype=torch.float32)
    with nested_dispatch(inp.numel()):
        if exact:
            softmax_col_max[(grid,)](inp, mb, n_cols, inner, n_cb, CB=cb)
            softmax_col_exp[(grid,)](inp, out, mb, sb, n_cols, inner, n_cb, CB=cb)
            softmax_col_norm[(grid,)](out, sb, n_cols, inner, n_cb, CB=cb)
        else:
            softmax_col_max_m[(grid,)](inp, mb, n_cols, inner, n_cb, CB=cb)
            softmax_col_exp_m[(grid,)](inp, out, mb, sb, n_cols, inner, n_cb, CB=cb)
            softmax_col_norm_m[(grid,)](out, sb, n_cols, inner, n_cb, CB=cb)
    return out


def _sm_rows(n_rows: int) -> int:
    """Rows per resident-row program (wave48).

    probe613: below 128 rows the old rule always launched one row per program,
    so a 16-row 4096-column tensor became 16 single-row programs and left the
    three-pass kernel within 1.4x.  M//8 rows per program (power of two, 1..8)
    was within 2% of the best measured ROWS on every shape in the band and is
    1.3-1.5x over one-row launches at 16-64 rows.  Above 128 rows the wave24
    rule is unchanged.
    """
    if n_rows >= 128:
        return max(1, min(128, n_rows // 128))
    r = 1
    while r * 2 <= n_rows // 8 and r < 8:
        r *= 2
    return r


def _spacemit_softmax_lastdim(inp, out):
    n_rows, n_cols = inp.shape
    n = inp.numel()
    # wave26: small tensors were left on the three-pass block_ptr kernel and paid
    # 2.5-3x the floor.  probe266 at 64x64 f32: current 257 us against 82 us for
    # one resident-row launch with ROWS=8 (a null Triton launch is 58 us), and the
    # same 3.1x at f16 (317 -> 102 us).  256x256 f32 is 619 -> 212 us.  No nested
    # dispatch here: below 1<<16 it only adds overhead.
    # wave48: the band 2049..8192 columns with fewer than 65536 elements had no
    # resident route (the old guard stopped at 2048) and paid 1.1-3.4x to the
    # three-pass kernel.  probe611/613: keep the row resident there too.
    if n < (1 << 16) and n_cols <= 8192:
        col = triton.next_power_of_2(n_cols)
        if n_cols <= 1024:
            rows = 8
        elif n_cols <= 2048:
            rows = 1
        else:
            rows = _sm_rows(n_rows)
        while rows > 1 and n_rows % rows:
            rows //= 2
        if col == n_cols:
            softmax_kernel_rowres[(n_rows // rows,)](
                inp, out, n_cols, ROWS=rows, COL=col
            )
        else:
            softmax_kernel_rowres_m[(n_rows // rows,)](
                inp, out, n_cols, ROWS=rows, COL=col
            )
        return
    # wave24: a single very wide row is a grid=1 launch on the live path (one core
    # walking 4096 column blocks).  Two levels over 32768-element chunks:
    # 10.66 -> 1.58 ms f32, 14.34 -> 1.29 f16 at 1048576 (probe229).
    if n_rows == 1 and n >= (1 << 16) and n % _FLAT_CHUNK == 0:
        _softmax_flat(inp, out)
        return
    # wave24: rows that fit in one block stay resident, ROWS rows per program.
    # The three-pass path below pays a scalar-lowered tl.sum for every 256-wide
    # chunk and three passes over the row, which is 1.7-3.5x slower (probe231).
    if n_cols <= 8192 and n >= (1 << 16):
        col = triton.next_power_of_2(n_cols)
        # wave48: _sm_rows keeps the same rule above 128 rows and gives 16-64
        # row tensors 2-8 rows per program (1.3-1.5x, probe613).
        rows = _sm_rows(n_rows)
        if n_rows % rows:
            rows = 1  # keep every launch exact; a masked row store is not safe here
        with nested_dispatch(n):
            if col == n_cols:
                softmax_kernel_rowres[(n_rows // rows,)](
                    inp, out, n_cols, ROWS=rows, COL=col
                )
            else:
                softmax_kernel_rowres_m[(n_rows // rows,)](
                    inp, out, n_cols, ROWS=rows, COL=col
                )
        return
    row_size = 1 if n_rows < 2 else (2 if n_rows < 8 else 4)
    # wave22: 256-wide column blocks are 6-13% faster than 64-wide on every
    # official shape (probe197b), and f16 rows of length 256-1024 are better
    # held in one resident block (probe197/199b: 512^2 2.17 -> 1.10 ms,
    # 8192x512 30.4 -> 17.9; at 2048 the block spills and loses 1.5x).
    if (
        inp.dtype == torch.float16
        and 256 <= n_cols <= 1024
        and (n_cols & (n_cols - 1)) == 0
    ):
        # no nested_dispatch here: the resident-row kernel is memory bound and
        # the nested path costs it 1.9x (probe199b: 1.10 -> 2.05 ms at 512^2)
        softmax_kernel_block[(n_rows,)](inp, out, n_cols, COL=n_cols)
        return
    col_size = 64 if n_cols < 256 else 256
    grid = lambda meta: (triton.cdiv(n_rows, meta["ROW_SIZE"]),)
    with nested_dispatch(inp.numel()):
        softmax_kernel_spacemit[grid](
            out,
            inp,
            inp.stride(0),
            out.stride(0),
            n_rows,
            n_cols,
            ROW_SIZE=row_size,
            COL_SIZE=col_size,
        )


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_SPACEMIT SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim

    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype

    inp = self.contiguous()

    # wave25: a non-last dim now gets its own walk instead of being silently
    # reduced along the last axis (pre-existing vendor bug, see block comment).
    if dim != inp.ndim - 1 and _softmax_nonlast_enabled():
        out = torch.empty_like(inp, dtype=dtype)
        hugepage_hint(out)
        _softmax_nonlast(inp, out, dim)
        return out

    n_cols = inp.shape[-1]
    n_rows = inp.numel() // n_cols
    inp_2d = inp.view(n_rows, n_cols)
    out_2d = torch.empty_like(inp_2d, dtype=dtype)
    hugepage_hint(out_2d)
    _spacemit_softmax_lastdim(inp_2d, out_2d)
    return out_2d.view_as(inp)


def softmax_backward(grad_output, output, dim, input_dtype):
    # wave3 (campaign 1014night): direct two-loop kernels for the last-dim case.
    if _sb_fast_enabled() and _sb_ok(grad_output, output, dim, input_dtype):
        return _sb_fast(grad_output, output, dim, input_dtype)
    logger.debug("GEMS_SPACEMIT SOFTMAX_VJP")
    with nested_dispatch(grad_output.numel()):
        return common_softmax_backward(grad_output, output, dim, input_dtype)


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    if (
        _sb_fast_enabled()
        and _sb_ok(grad_output, output, dim, input_dtype)
        and isinstance(grad_input, torch.Tensor)
        and grad_input.shape == output.shape
        and grad_input.dtype == input_dtype
        and grad_input.is_contiguous()
    ):
        _sb_fast(grad_output, output, dim, input_dtype, grad_input=grad_input)
        return grad_input
    logger.debug("GEMS_SPACEMIT SOFTMAX_BACKWARD_OUT")
    return common_softmax_backward_out(
        grad_output, output, dim, input_dtype, grad_input=grad_input
    )


# wave23: vendor softmax_out.  The generic implementation only knows the
# row-per-program inner kernel (single core at 1024^3: 12 min, probe207) and a
# non-inner kernel that parallelises ~1 core as well.  Route the last-dim case
# through the same fast kernel as softmax(), and wrap the non-inner fallback in
# nested dispatch (probe207: 64x512x512 376.7 -> 95.0 ms, 3.96x; gain appears
# only when K/TILE_K == 1, i.e. TILE_K covers the contiguous dim, which is what
# the heuristic already picks for these shapes).
def softmax_out(self, dim, half_to_float=False, *, out):
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    if self.numel() == 0:
        if tuple(out.shape) != tuple(self.shape):
            out.resize_(self.shape)
        out.zero_()
        return out
    dim = dim % self.ndim
    dtype = torch.float32 if half_to_float else self.dtype
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)
    if out.dtype != dtype:
        raise RuntimeError(f"_softmax.out: expected out dtype {dtype}, got {out.dtype}")
    if out.device != self.device:
        raise RuntimeError("_softmax.out: out must be on the input device")
    inp = self.contiguous()
    hugepage_hint(out)
    if dim == inp.ndim - 1 and out.is_contiguous():
        n_cols = inp.shape[-1]
        n_rows = inp.numel() // n_cols
        _spacemit_softmax_lastdim(inp.view(n_rows, n_cols), out.view(n_rows, n_cols))
        return out
    logger.debug("GEMS_SPACEMIT SOFTMAX_OUT (non-inner)")
    # wave25: dim != last is a real walk now, not the last axis in disguise.
    if dim != inp.ndim - 1 and _softmax_nonlast_enabled() and out.is_contiguous():
        _softmax_nonlast(inp, out, dim)
        return out
    with nested_dispatch(self.numel()):
        return common_softmax_out(self, dim, half_to_float, out=out)
