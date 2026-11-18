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
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

from .mean_fast import mean_mmk_enabled, mean_mmk_launch
from .sum_fast import sum_flat_launch, sum_rows16_launch

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(inp_val, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val)


def mean(inp, *, dtype=None):
    logger.debug("GEMS_SPACEMIT MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    # wave30: 16-bit inputs ride the flat sum carrier (f32 accumulator) and are
    # divided in f32 before the cast - the legacy sqrt-rule path kept f16
    # partials and ran at 5.7 ms per 1M elements (probe295).
    if (
        inp.is_contiguous()
        and inp.dtype in (torch.float16, torch.bfloat16)
        and M >= 65536
        and mean_mmk_enabled()
    ):
        acc = torch.empty([], dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            sum_flat_launch(inp, acc, M)
        res = acc / M
        return res if dtype == torch.float32 else res.to(dtype)
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("mean"),
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, TILE_N: tl.constexpr):
    row = tl.program_id(0)
    X = X + row * N
    Mean = Mean + row
    _mean = 0.0

    num_pid_n = tl.cdiv(N, TILE_N)

    x_ptr_desc = tl.make_block_ptr(
        base=X,
        shape=[N],
        strides=[1],
        offsets=[0],
        block_shape=[TILE_N],
        order=[0],
    )

    for off_n in range(0, num_pid_n):
        a = tl.load(
            x_ptr_desc,
            boundary_check=[0],
        )

        _mean += tl.sum(a)

        x_ptr_desc = tl.advance(x_ptr_desc, [TILE_N])

    mean = _mean / N

    tl.store(Mean, mean)


@libentry()
@triton.jit
def mean_dim_kernel_bp(X, Mean, N, TILE_N: tl.constexpr):
    # campaign 1011pm: non-autotuned block-ptr copy for a deterministic
    # short-row route. The autotuner's per-process picks fluctuated on
    # (4096,1024) between ~5.7 and ~8.0 ms; explicit TILE_N=512 measured
    # 5.68 ms vs 8.04 ms for the autotuned path in the same session.
    row = tl.program_id(0)
    X = X + row * N
    Mean = Mean + row
    _mean = 0.0
    num_pid_n = tl.cdiv(N, TILE_N)
    x_ptr_desc = tl.make_block_ptr(
        base=X,
        shape=[N],
        strides=[1],
        offsets=[0],
        block_shape=[TILE_N],
        order=[0],
    )
    for off_n in range(0, num_pid_n):
        a = tl.load(
            x_ptr_desc,
            boundary_check=[0],
        )
        _mean += tl.sum(a)
        x_ptr_desc = tl.advance(x_ptr_desc, [TILE_N])
    mean = _mean / N
    tl.store(Mean, mean)


@libentry()
@triton.jit
def mean_dim_kernel_plain(X, Mean, N, TILE_N: tl.constexpr):
    # campaign 1011am: plain-pointer row kernel; no block_ptr, fixed TILE_N.
    row = tl.program_id(0)
    base = X + row * N
    acc = 0.0
    for c in range(0, N // TILE_N):
        offs = c * TILE_N + tl.arange(0, TILE_N)
        a = tl.load(base + offs)
        acc += tl.sum(a)
    tl.store(Mean + row, acc / N)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS_SPACEMIT MEAN_DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    # wave30: 16-bit mean_dim rides the sum row carrier (f32 accumulator) and
    # divides by N in f32. The f16 rows stay on the wave5 _MMK kernels: their
    # raw-f16 tl.sum runs at 5.3 GB/s (1.50 ms at 2048^2) while the widening
    # carrier costs 3.0 ms there; bf16 has no native reduce and falls to 13.6 ms
    # on _MMK, so bf16 takes the carrier. M == 1 (flat reduce through
    # mean_dim(dim=[0])) always takes the flat carrier, f16 included.
    if (
        x.dtype in (torch.float16, torch.bfloat16)
        and x.is_contiguous()
        and x.numel() >= 65536
        and mean_mmk_enabled()
    ):
        acc = torch.empty(shape, dtype=torch.float32, device=x.device)
        with torch_device_fn.device(x.device):
            if M == 1:
                sum_flat_launch(x, acc.reshape(1), x.numel())
                handled = True
            else:
                handled = x.dtype == torch.bfloat16 and sum_rows16_launch(
                    x, acc.reshape(M), M, N
                )
            if handled:
                acc = acc / N
                out = acc if dtype == torch.float32 else acc.to(dtype)
                if not keepdim:
                    out = out.squeeze(dim)
                return out
    out = torch.empty(shape, dtype=dtype, device=x.device)
    grid = (M,)
    with torch_device_fn.device(x.device):
        if (
            x.is_contiguous()
            and mean_mmk_enabled()
            and mean_mmk_launch(x, out, M, N)
        ):
            # campaign 1014pm (wave5): multi-row-per-program kernels - the
            # short-row penalty was launch/setup amortization, not the reduce
            # itself; K rows per program gives 1.1x-5.8x over the legacy
            # routes (see mean_fast.py docstring). Kill: MEAN_MMK=0.
            pass
        elif x.is_contiguous() and N >= 2048 and N % 2048 == 0:
            # campaign 1011am: 2048-wide tiles keep tl.sum on the vectorized
            # side of the length-2048 cliff. Measured 1.2-1.3x over the
            # block_ptr loop on long-row shapes; short rows (N <= 1024) stay
            # on the old path (plain was slower there).
            mean_dim_kernel_plain[grid](x, out, N, TILE_N=2048)
        elif x.is_contiguous() and 512 <= N <= 1024 and N % 512 == 0:
            # campaign 1011pm: deterministic TILE_N=512 block_ptr route for
            # short rows (autotuner picks fluctuated, see kernel note).
            mean_dim_kernel_bp[grid](x, out, N, TILE_N=512)
        else:
            mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out


def global_avg_pool(x, _output_size=None):
    return mean_dim(x, dim=[2, 3], keepdim=True)
