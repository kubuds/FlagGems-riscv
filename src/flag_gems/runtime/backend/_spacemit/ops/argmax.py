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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

from ._dispatch import nested_dispatch

logger = logging.getLogger(__name__)


@triton.jit
def _argmax_dim0_s1(X, PV, PI, N, K, RB: tl.constexpr, BN: tl.constexpr):
    """wave13: first-index argmax over axis 0, row-block partials."""
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    r0 = rb * RB
    r1 = tl.minimum(r0 + RB, N)
    cols = cb * BN + tl.arange(0, BN)
    cm = cols < K
    acc = tl.full((BN,), float("-inf"), tl.float32)
    idx = tl.zeros((BN,), tl.int32)
    for r in range(r0, r1):
        v = tl.load(X + r * K + cols, mask=cm, other=float("-inf")).to(tl.float32)
        # NaN propagates like torch (earlier NaN wins); strict > keeps the
        # first occurrence on ties.
        upd = (acc == acc) & ((v != v) | (v > acc))
        acc = tl.where(upd, v, acc)
        idx = tl.where(upd, r, idx)
    tl.store(PV + rb * K + cols, acc, mask=cm)
    tl.store(PI + rb * K + cols, idx, mask=cm)


@triton.jit
def _argmax_dim0_s2(PV, PI, OUT, K, NRB, BN: tl.constexpr):
    cb = tl.program_id(0)
    cols = cb * BN + tl.arange(0, BN)
    cm = cols < K
    acc = tl.full((BN,), float("-inf"), tl.float32)
    idx = tl.zeros((BN,), tl.int32)
    for s in range(0, NRB):
        v = tl.load(PV + s * K + cols, mask=cm, other=float("-inf"))
        i = tl.load(PI + s * K + cols, mask=cm, other=0)
        upd = (acc == acc) & ((v != v) | (v > acc))
        acc = tl.where(upd, v, acc)
        idx = tl.where(upd, i, idx)
    tl.store(OUT + cols, idx.to(tl.int64), mask=cm)


_ARGM0_RB = 128


def _argmax_dim0(inp, N, K):
    """Reduce axis 0 (M==1) of a contiguous 2-D view via the fold table."""
    if K <= 512:
        bn = 256
    elif K <= 2048:
        bn = 512
    else:
        bn = 1024
    nrb = triton.cdiv(N, _ARGM0_RB)
    nblk = triton.cdiv(K, bn)
    pv = torch.empty(nrb, K, dtype=torch.float32, device=inp.device)
    pi = torch.empty(nrb, K, dtype=torch.int32, device=inp.device)
    out = torch.empty(K, dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        _argmax_dim0_s1[(nrb, nblk)](inp, pv, pi, N, K, _ARGM0_RB, bn)
        _argmax_dim0_s2[(nblk,)](pv, pi, out, K, nrb, bn)
    return out

@libentry()
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


@libentry()
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    min_value = get_dtype_min(mid_value.type.element_ty)
    mid_val = tl.load(mid_value + offset, mask=mask, other=min_value)
    # First chunk achieving the maximum (deterministic min-select; also avoids
    # sentinel index output for degenerate all-sentinel inputs, clamped below).
    m = tl.max(mid_val, axis=0)
    eq = (mid_val == m) & mask
    big = tl.full((BLOCK_MID,), BLOCK_MID, dtype=tl.int32)
    sel = tl.min(tl.where(eq, offset, big), axis=0)
    out_val = tl.load(mid_index + sel)
    out_val = tl.where(out_val < 0, 0, out_val)
    tl.store(out, out_val)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax"))
@triton.jit
def argmax_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tle.program_id(1)
    start_row = pid_m * BLOCK_M
    row_offsets = start_row + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M
    dtype = inp_ptr.dtype.element_ty
    min_value = get_dtype_min(dtype)
    row_max = tl.full((BLOCK_M,), min_value, dtype=dtype)
    row_argmax = tl.full((BLOCK_M,), -1, dtype=tl.int32)

    for block_start in range(0, N, BLOCK_N):
        col_offsets = block_start + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        mask = row_mask[:, None] & col_mask[None, :]
        input_ptrs = (
            inp_ptr + row_offsets[:, None] * N * K + col_offsets[None, :] * K + pid_k
        )
        current_block = tl.load(input_ptrs, mask=mask, other=min_value)

        block_max = tl.max(current_block, axis=1)
        block_argmax = tl.argmax(current_block, axis=1).to(tl.int32) + block_start

        update_mask = block_max > row_max
        tie_mask = (block_max == row_max) & (
            (row_argmax < 0) | (block_argmax < row_argmax)
        )
        choose_new = update_mask | tie_mask

        row_argmax = tl.where(choose_new, block_argmax, row_argmax)
        row_max = tl.where(update_mask, block_max, row_max)

    out_offsets = row_offsets * K + pid_k
    out_ptrs = out_ptr + out_offsets
    # Rows never updated by the reduction (all elements below the dtype
    # sentinel, e.g. all -inf) keep the init -1; report 0 like torch.
    row_argmax = tl.where(row_argmax < 0, 0, row_argmax)
    tl.store(out_ptrs, row_argmax.to(out_ptr.dtype.element_ty), mask=row_mask)


# --- Two-stage row reduction (K == 1 fast path) -----------------------------
# The single-stage row kernel above lowers its reductions to scalar code on the
# spacemit backend (~54 ms for 2048x2048, on par with the official stack).
# Splitting the row reduction into a per-(row-block, chunk) partial stage plus a
# cross-chunk combine stage lets the backend emit the faster return_indices
# reduction path (measured ~10 ms for 2048x2048, ~5x vs the single-stage form).


@libentry()
@triton.jit
def argmax_kernel_row_s1(
    inp,
    mid_value,
    mid_index,
    M,
    N,
    NB: tl.constexpr,
    BN: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * NB + tl.arange(0, NB)
    cols = pid_c * BN + tl.arange(0, BN)
    ptrs = inp + rows[:, None] * N + cols[None, :]
    if EVEN:
        x = tl.load(ptrs)
    else:
        # Non-divisible N: mask the (last) partial chunk with the dtype
        # sentinel so it never wins the max.
        min_value = get_dtype_min(inp.type.element_ty)
        x = tl.load(ptrs, mask=cols[None, :] < N, other=min_value)
    max_val, max_idx = tl.max(x, axis=1, return_indices=True)
    # Degenerate rows (all elements == dtype sentinel) are never replaced in
    # the index reduction, which yields its init value (-1). Clamp to the
    # first column of the chunk, matching torch's first-occurrence semantics.
    max_idx = tl.where(max_idx < 0, 0, max_idx)
    max_idx = max_idx.to(tl.int32) + pid_c * BN
    tl.store(mid_value + pid_c * M + rows, max_val)
    tl.store(mid_index + pid_c * M + rows, max_idx)


@libentry()
@triton.jit
def argmax_kernel_row_s2(
    mid_value,
    mid_index,
    out,
    M,
    C,
    C_P2: tl.constexpr,
    BLOCK_M2: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M2 + tl.arange(0, BLOCK_M2)
    rmask = rows < M
    c = tl.arange(0, C_P2)
    cmask = c < C
    min_value = get_dtype_min(mid_value.type.element_ty)
    mv = tl.load(
        mid_value + c[:, None] * M + rows[None, :],
        mask=cmask[:, None] & rmask[None, :],
        other=min_value,
    )
    m = tl.max(mv, axis=0)
    eq = mv == m[None, :]
    # Padded chunk rows (c >= C) hold the sentinel; the true max is attained
    # by a real chunk, so eq can only match padded rows when m == sentinel,
    # in which case chunk 0 (a real chunk) also matches and min-select is
    # unaffected.
    big = tl.full((C_P2, BLOCK_M2), 2147483647, dtype=tl.int32)
    cbr = tl.broadcast_to(c[:, None], (C_P2, BLOCK_M2))
    sel = tl.min(tl.where(eq, cbr, big), axis=0)
    idx = tl.load(mid_index + sel * M + rows, mask=rmask)
    tl.store(out + rows, idx.to(tl.int64), mask=rmask)


# Measured per-shape best (NB, BN) for the two-stage row kernel
# (from campaign session work1010pm/argmax_tune on K3, fp32):
#   N=512/1024   -> (64, 256);  N>=2048 -> (16, 512)
_ROW_CFG_BY_N = {
    512: (64, 256),
    1024: (64, 256),
    2048: (16, 512),
    4096: (16, 512),
    8192: (16, 512),
    16384: (16, 512),
    32768: (16, 512),
}


def _choose_row_cfg(M, N):
    """Return (NB, BN, EVEN) for the two-stage row kernel, or None to use the
    single-stage kernel. Non-divisible N uses a masked tail chunk (EVEN=False)."""
    if N < 128 or M < 16:
        return None
    cfg = _ROW_CFG_BY_N.get(N)
    if cfg is not None:
        nb, bn = cfg
        even = True
    else:
        if N % 512 == 0:
            bn = 512
        elif N % 256 == 0:
            bn = 256
        elif N % 128 == 0:
            bn = 128
        else:
            # masked-tail path: cap the chunk at 512; the last chunk is partial
            bn = 512 if N > 512 else triton.next_power_of_2(N)
        even = (N % bn) == 0
        c = triton.cdiv(N, bn)
        nb = 16 if c >= 4 else 64
    # NB must divide M; shrink (never grow) otherwise
    while nb > 4 and M % nb != 0:
        nb //= 2
    if M % nb != 0:
        return None
    if M * N < 65536:
        return None
    return nb, bn, even


def _argmax_flat(inp):
    """Argmax over the whole (already contiguous) tensor via the two-stage path."""
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    mid_value = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
    mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
    out = torch.empty([], dtype=torch.int64, device=inp.device)
    # wave13 probe116: the flat two-stage path is math-heavy (index carry);
    # nested dispatch is 1.76x @1M / 2.25x @4M (bit-identical results).
    with torch_device_fn.device(inp.device), nested_dispatch(M):
        argmax_kernel_1[(mid_size, 1, 1)](inp, mid_value, mid_index, M, block_size)
        argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
    return out


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_SPACEMIT ARGMAX")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        with torch_device_fn.device(inp.device), nested_dispatch(M):
            argmax_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
            )
            argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
        return out
    else:
        if dim < -inp.ndim or dim >= inp.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
            )
        shape = inp.shape
        dim = dim % inp.ndim
        if inp.numel() == 0:
            out_shape = list(shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return torch.empty(out_shape, dtype=torch.int64, device=inp.device)

        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()

        if (
            M == 1
            and K > 1
            and inp.dtype in (torch.float32, torch.float16)
            and inp.numel() >= 65536
        ):
            # wave13: dim=0 reduce (M==1) - the generic path walks the reduce
            # axis with stride K (147 ms @512x2048 dim0); fold table is 1.2-2.8x
            # faster than native torch and 4-36x faster than the old path.
            out = _argmax_dim0(inp, N, K)
            base = tuple(shape[1:])
            if keepdim:
                out = out.reshape((1,) + base)
            else:
                out = out.reshape(base)
            return out

        if M == 1 and K == 1:
            # Single reduction group (e.g. last-dim argmax on a 1-D tensor): the
            # row kernel would launch a single program, so reuse the parallel
            # two-stage reduction instead.
            out = _argmax_flat(inp)
            if keepdim:
                out = out.reshape([1] * inp.dim())
            else:
                out = out.reshape(shape[:dim])
            return out

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        if K == 1:
            cfg = _choose_row_cfg(M, N)
            if cfg is not None:
                nb, bn, even = cfg
                c = triton.cdiv(N, bn)
                mid_value = torch.empty(
                    (c, M), dtype=inp.dtype, device=inp.device
                )
                mid_index = torch.empty((c, M), dtype=torch.int32, device=inp.device)
                with torch_device_fn.device(inp.device):
                    argmax_kernel_row_s1[(c, M // nb)](
                        inp, mid_value, mid_index, M, N, NB=nb, BN=bn, EVEN=even
                    )
                    blk2 = 256
                    argmax_kernel_row_s2[(triton.cdiv(M, blk2),)](
                        mid_value,
                        mid_index,
                        out_index.reshape(-1),
                        M,
                        c,
                        C_P2=triton.next_power_of_2(c),
                        BLOCK_M2=blk2,
                    )
                return out_index

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)
        with torch_device_fn.device(inp.device):
            argmax_kernel[grid](
                inp,
                out_index,
                M,
                N,
                K,
            )

        return out_index
