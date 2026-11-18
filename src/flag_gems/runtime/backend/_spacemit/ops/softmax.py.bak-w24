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
    tl.store(output_ptr + base + offs, (e / s).to(output_ptr.dtype.element_ty))


def _spacemit_softmax_lastdim(inp, out):
    n_rows, n_cols = inp.shape
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

    n_cols = inp.shape[-1]
    n_rows = inp.numel() // n_cols
    inp_2d = inp.view(n_rows, n_cols)
    out_2d = torch.empty_like(inp_2d, dtype=dtype)
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
    if dim == inp.ndim - 1 and out.is_contiguous():
        n_cols = inp.shape[-1]
        n_rows = inp.numel() // n_cols
        _spacemit_softmax_lastdim(inp.view(n_rows, n_cols), out.view(n_rows, n_cols))
        return out
    logger.debug("GEMS_SPACEMIT SOFTMAX_OUT (non-inner)")
    with nested_dispatch(self.numel()):
        return common_softmax_out(self, dim, half_to_float, out=out)
