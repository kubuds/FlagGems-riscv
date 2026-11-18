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
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)


# ---------------- wave14: fast paths (row / flat / dim=0) ----------------
# Ported from the argmax family (wave13), where the same three structures are
# 1.5-40x faster than the generic row kernel (probe120: argmin's generic path
# is 0.04-0.09x of native torch). torch semantics: first occurrence wins on
# ties; NaN propagates as the minimum (first NaN wins).

from ._dispatch import nested_dispatch


@triton.jit
def _argmin_dim0_s1(X, PV, PI, N, K, RB: tl.constexpr, BN: tl.constexpr):
    """First-index argmin over axis 0, row-block partials."""
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    r0 = rb * RB
    r1 = tl.minimum(r0 + RB, N)
    cols = cb * BN + tl.arange(0, BN)
    cm = cols < K
    acc = tl.full((BN,), float("inf"), tl.float32)
    idx = tl.zeros((BN,), tl.int32)
    for r in range(r0, r1):
        v = tl.load(X + r * K + cols, mask=cm, other=float("inf")).to(tl.float32)
        upd = (acc == acc) & ((v != v) | (v < acc))
        acc = tl.where(upd, v, acc)
        idx = tl.where(upd, r, idx)
    tl.store(PV + rb * K + cols, acc, mask=cm)
    tl.store(PI + rb * K + cols, idx, mask=cm)


@triton.jit
def _argmin_dim0_s2(PV, PI, OUT, K, NRB, BN: tl.constexpr):
    cb = tl.program_id(0)
    cols = cb * BN + tl.arange(0, BN)
    cm = cols < K
    acc = tl.full((BN,), float("inf"), tl.float32)
    idx = tl.zeros((BN,), tl.int32)
    for s in range(0, NRB):
        v = tl.load(PV + s * K + cols, mask=cm, other=float("inf"))
        i = tl.load(PI + s * K + cols, mask=cm, other=0)
        upd = (acc == acc) & ((v != v) | (v < acc))
        acc = tl.where(upd, v, acc)
        idx = tl.where(upd, i, idx)
    tl.store(OUT + cols, idx.to(tl.int64), mask=cm)


_ARGMN0_RB = 128  # default; per-shape value from _dim0_rb(N)


def _dim0_rb(N: int) -> int:
    """wave14: row-block size for the dim=0 fold table. probe125 scan:
    N=256 -> 64 (3.92 ms vs 7.22 at 128), N=512 -> 128, N>=1024 -> 256
    (14.36 vs 14.67 @1024x4096; 15.19 vs 15.69 @8192x512)."""
    rb = N // 4
    if rb < 64:
        rb = 64
    if rb > 256:
        rb = 256
    return rb


def _argmin_dim0(inp, N, K):
    """Reduce axis 0 (M==1) of a contiguous 2-D view via the fold table."""
    if K <= 512:
        bn = 256
    elif K <= 2048:
        bn = 512
    else:
        bn = 1024
    rb = _dim0_rb(N)
    nrb = triton.cdiv(N, rb)
    nblk = triton.cdiv(K, bn)
    pv = torch.empty(nrb, K, dtype=torch.float32, device=inp.device)
    pi = torch.empty(nrb, K, dtype=torch.int32, device=inp.device)
    out = torch.empty(K, dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        _argmin_dim0_s1[(nrb, nblk)](inp, pv, pi, N, K, rb, bn)
        _argmin_dim0_s2[(nblk,)](pv, pi, out, K, nrb, bn)
    return out


@triton.jit
def _argmin_row_s1(
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
        x = tl.load(ptrs, mask=cols[None, :] < N, other=float("inf"))
    min_val, min_idx = tl.min(x, axis=1, return_indices=True)
    min_idx = tl.where(min_idx < 0, 0, min_idx)
    min_idx = min_idx.to(tl.int32) + pid_c * BN
    tl.store(mid_value + pid_c * M + rows, min_val)
    tl.store(mid_index + pid_c * M + rows, min_idx)


@triton.jit
def _argmin_row_s2(
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
    mv = tl.load(
        mid_value + c[:, None] * M + rows[None, :],
        mask=cmask[:, None] & rmask[None, :],
        other=float("inf"),
    )
    # NaN in any chunk must win (torch argmin returns the first NaN index):
    # the plain min/eq select would drop it because NaN != NaN.
    nan = mv != mv
    has_nan = tl.max(nan.to(tl.int32), axis=0) > 0
    big = tl.full((C_P2, BLOCK_M2), 2147483647, dtype=tl.int32)
    cbr = tl.broadcast_to(c[:, None], (C_P2, BLOCK_M2))
    first_nan = tl.min(tl.where(nan, cbr, big), axis=0)
    m = tl.min(mv, axis=0)
    eq = mv == m[None, :]
    first_eq = tl.min(tl.where(eq, cbr, big), axis=0)
    sel = tl.where(has_nan, first_nan, first_eq)
    idx = tl.load(mid_index + sel * M + rows, mask=rmask)
    tl.store(out + rows, idx.to(tl.int64), mask=rmask)


def _amin_flat_block(M: int) -> int:
    """Stage-1 block for the flat two-stage reduction (probe120b: the sqrt
    rule's 1024 is far off the vector side; 16384-class wins 1.6-2.2x)."""
    p2 = triton.next_power_of_2(M)
    b = triton.next_power_of_2(max(64, M // 8)) if M > 64 else 64
    if b > 131072:
        b = 131072
    if b < 1024:
        b = min(1024, p2)
    if b > p2:
        b = p2
    return b


def _argmin_flat(inp):
    """Argmin over the whole (already contiguous) tensor, two-stage + nested."""
    M = inp.numel()
    block = _amin_flat_block(M)
    mid_size = triton.cdiv(M, block)
    block_mid = triton.next_power_of_2(mid_size)
    mid_value = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
    mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
    out = torch.empty([], dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device), nested_dispatch(M):
        argmin_kernel_1[(mid_size, 1, 1)](inp, mid_value, mid_index, M, block)
        argmin_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
    return out


# per-shape (NB, BN) for the two-stage row kernel, mirrors argmax
_ROW_CFG_BY_N = {
    512: (64, 256),
    1024: (64, 256),
    2048: (16, 512),
    4096: (16, 1024),
    8192: (16, 512),
    16384: (16, 512),
    32768: (16, 512),
}


def _choose_row_cfg(M, N):
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
            bn = 512 if N > 512 else triton.next_power_of_2(N)
        even = (N % bn) == 0
        c = triton.cdiv(N, bn)
        nb = 16 if c >= 4 else 64
    while nb > 4 and M % nb != 0:
        nb //= 2
    if M % nb != 0:
        return None
    if M * N < 65536:
        return None
    return nb, bn, even


@libentry()
@triton.jit
def argmin_kernel_1(
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
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
    min_val, min_index = tl.min(inp_val, axis=0, return_indices=True)
    min_index = min_index + pid * BLOCK_SIZE
    tl.store(mid_value + pid, min_val)
    tl.store(mid_index + pid, min_index)


@libentry()
@triton.jit
def argmin_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    index_val = tl.argmin(mid_val, axis=0)
    out_val = tl.load(mid_index + index_val)
    tl.store(out, out_val)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmin"))
@triton.jit
def argmin_kernel(
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
    acc_type = (
        tl.float32
        if (dtype is tl.bfloat16 or dtype is tl.float16)
        else (
            tl.int32
            if (dtype is tl.int16 or dtype is tl.int8 or dtype is tl.uint8)
            else dtype
        )
    )
    max_value = get_dtype_max(dtype)
    max_value_acc = get_dtype_max(acc_type)
    row_min = tl.full((BLOCK_M,), max_value_acc, dtype=acc_type)
    row_argmin = tl.full((BLOCK_M,), -1, dtype=tl.int32)

    for block_start in range(0, N, BLOCK_N):
        col_offsets = block_start + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        mask = row_mask[:, None] & col_mask[None, :]
        input_ptrs = (
            inp_ptr + row_offsets[:, None] * N * K + col_offsets[None, :] * K + pid_k
        )
        current_block = tl.load(input_ptrs, mask=mask, other=max_value).to(acc_type)

        block_min = tl.min(current_block, axis=1)
        block_argmin = tl.argmin(current_block, axis=1).to(tl.int32) + block_start

        update_mask = block_min < row_min
        tie_mask = (block_min == row_min) & (
            (row_argmin < 0) | (block_argmin < row_argmin)
        )
        choose_new = update_mask | tie_mask

        row_argmin = tl.where(choose_new, block_argmin, row_argmin)
        row_min = tl.where(update_mask, block_min, row_min)

    out_offsets = row_offsets * K + pid_k
    tl.store(
        out_ptr + out_offsets, row_argmin.to(out_ptr.dtype.element_ty), mask=row_mask
    )



def _argmin_flat_legacy(inp):
    """Pre-wave14 flat path (sqrt-block, no nested dispatch) - A/B reference."""
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    mid_value = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
    mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
    out = torch.empty([], dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        argmin_kernel_1[(mid_size, 1, 1)](inp, mid_value, mid_index, M, block_size)
        argmin_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
    return out


def _amin_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_ARGMIN_FAST", "1") != "0"


def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_SPACEMIT ARGMIN")
    fast = _amin_fast_enabled()
    if dim is None:
        if inp.numel() == 0:
            return torch.empty([], dtype=torch.int64, device=inp.device)
        out = _argmin_flat(inp) if fast else _argmin_flat_legacy(inp)
        if keepdim:
            return out.reshape([1] * inp.dim())
        return out
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
    fdt = inp.dtype in (torch.float32, torch.float16)

    if fast and M == 1 and K > 1 and fdt and inp.numel() >= 65536:
        # wave14: non-trailing dim=0 reduce; the generic path walks the axis
        # with stride K (131 ms @512x2048, 577 ms @1024x4096).
        out = _argmin_dim0(inp, N, K)
        base = tuple(shape[1:])
        if keepdim:
            return out.reshape((1,) + base)
        return out.reshape(base)

    if fast and M == 1 and K == 1:
        # single reduction group (flat call with dim given): reuse the
        # two-stage reduction instead of a one-program row kernel
        out = _argmin_flat(inp) if fast else _argmin_flat_legacy(inp)
        if keepdim:
            return out.reshape([1] * inp.dim())
        return out.reshape(shape[:dim])

    shape_list = list(shape)
    shape_list[dim] = 1
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
    if not keepdim:
        out_index = torch.squeeze(out_index, dim)

    if fast and K == 1 and fdt:
        cfg = _choose_row_cfg(M, N)
        if cfg is not None:
            nb, bn, even = cfg
            c = triton.cdiv(N, bn)
            mid_value = torch.empty((c, M), dtype=inp.dtype, device=inp.device)
            mid_index = torch.empty((c, M), dtype=torch.int32, device=inp.device)
            with torch_device_fn.device(inp.device):
                _argmin_row_s1[(c, M // nb)](
                    inp, mid_value, mid_index, M, N, NB=nb, BN=bn, EVEN=even
                )
                blk2 = 256
                _argmin_row_s2[(triton.cdiv(M, blk2),)](
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
        argmin_kernel[grid](
            inp,
            out_index,
            M,
            N,
            K,
        )

    return out_index
