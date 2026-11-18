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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Campaign 1011night: replaced the block-ptr/acc structure (32x32 acc + one
# final scalar reduce, 41.7 ms at 2048^2) with a row-parallel kernel that keeps
# per-row partials in registers across the K loop and reduces once at the end
# (17.7-20.4 ms measured; ~2.1-2.4x). Two variants: fully unmasked when N and M
# are multiples of 512 (all configs' tiles divide), masked otherwise.


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("mv_even"), key=["M", "N"])
@triton.jit
def mv_kernel_even(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, M, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + rows[:, None] * stride_an + ks[None, :] * stride_am)
        b = tl.load(B + ks * stride_bm)
        acc += a.to(tl.float32) * b[None, :]
    y = tl.sum(acc, axis=1)
    tl.store(C + rows * stride_cn, y.to(C.dtype.element_ty))


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("mv"), key=["M", "N"])
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, M, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        kmask = ks < M
        a = tl.load(
            A + rows[:, None] * stride_an + ks[None, :] * stride_am,
            mask=rmask[:, None] & kmask[None, :],
            other=0.0,
        )
        b = tl.load(B + ks * stride_bm, mask=kmask, other=0.0)
        acc += a.to(tl.float32) * b[None, :]
    y = tl.sum(acc, axis=1)
    tl.store(C + rows * stride_cn, y.to(C.dtype.element_ty), mask=rmask)


def mv(inp, vec):
    logger.debug("GEMS_SPACEMIT MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        if N % 512 == 0 and M % 512 == 0:
            mv_kernel_even[grid](
                inp,
                vec,
                out,
                N,
                M,
                inp.stride(0),
                inp.stride(1),
                vec.stride(0),
                out.stride(0),
            )
        else:
            mv_kernel[grid](
                inp,
                vec,
                out,
                N,
                M,
                inp.stride(0),
                inp.stride(1),
                vec.stride(0),
                out.stride(0),
            )
    return out
