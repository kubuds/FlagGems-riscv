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
import triton.language.extra.smt as smt

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fused_im2col_bmm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    im2col_buf_ptr,
    N,
    C,
    IH,
    IW,
    KH,
    KW,
    OC,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    OH,
    OW,
    GEMM_M,
    GEMM_K,
    KK,
    input_stride_n,
    input_stride_h,
    input_stride_w,
    input_stride_c,
    im2col_stride_n,
    im2col_stride_m,
    im2col_stride_k,
    weight_stride_oc,
    weight_stride_k,
    output_stride_n,
    output_stride_oc,
    output_stride_m,
    NUM_IM2COL_BLOCKS: tl.constexpr,
    NUM_BMM_TILES_PER_BATCH: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SUB_BLK_M: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_im2col = pid // (OH * OW)
    ohow = pid % (OH * OW)
    oh = ohow // OW
    ow = ohow % OW
    window_h = oh * stride_h - pad_h
    window_w = ow * stride_w - pad_w
    bmm_pid = tl.maximum(pid - NUM_IM2COL_BLOCKS, 0)
    pid_b = bmm_pid // NUM_BMM_TILES_PER_BATCH
    local_tile = bmm_pid % NUM_BMM_TILES_PER_BATCH
    pid_m = local_tile // NUM_TILES_N
    pid_n = local_tile % NUM_TILES_N
    block_m = pid_m * TILE_M
    block_n = pid_n * TILE_N
    bar = smt.global_mbarrier(0)
    is_im2col = pid < NUM_IM2COL_BLOCKS

    if is_im2col:
        input_block_ptr = tl.make_block_ptr(
            base=input_ptr,
            shape=(N, IH, IW, C),
            strides=(input_stride_n, input_stride_h, input_stride_w, input_stride_c),
            offsets=(n_im2col, 0, 0, 0),
            block_shape=(1, 1, 1, BLOCK_SIZE_C),
            order=(3, 2, 1, 0),
        )
        output_col_base_ptr = tl.make_block_ptr(
            base=im2col_buf_ptr,
            shape=(N, GEMM_M, GEMM_K),
            strides=(im2col_stride_n, im2col_stride_m, im2col_stride_k),
            offsets=(n_im2col, ohow, 0),
            block_shape=(1, 1, BLOCK_SIZE_C),
            order=(2, 1, 0),
        )

        for kh in range(KH):
            for kw in range(KW):
                h = window_h + kh * dilation_h
                w = window_w + kw * dilation_w
                valid_h = (h >= 0) & (h < IH)
                valid_w = (w >= 0) & (w < IW)
                valid = valid_h & valid_w
                for c_start in range(0, C, BLOCK_SIZE_C):
                    if valid:
                        input_ptr_cur = tl.advance(input_block_ptr, (0, h, w, c_start))
                        vals = tl.load(input_ptr_cur, boundary_check=(0, 1, 2, 3))
                        vals = tl.reshape(vals, (1, 1, BLOCK_SIZE_C))
                    else:
                        vals = tl.zeros(
                            (1, 1, BLOCK_SIZE_C), dtype=input_ptr.dtype.element_ty
                        )
                    col_idx = c_start * KK + kh * KW + kw
                    output_ptr_cur = tl.advance(output_col_base_ptr, (0, 0, col_idx))
                    tl.store(output_ptr_cur, vals, boundary_check=(0, 1, 2))
        smt.barrier_arrive(bar)

    else:
        if pid == NUM_IM2COL_BLOCKS:
            smt.barrier_set_expect(bar, NUM_IM2COL_BLOCKS)

        smt.barrier_wait(bar)
        a_ptr = tl.make_block_ptr(
            base=im2col_buf_ptr,
            shape=(N, GEMM_M, GEMM_K),
            strides=(im2col_stride_n, im2col_stride_m, im2col_stride_k),
            offsets=(pid_b, block_m, 0),
            block_shape=(1, TILE_M, TILE_K),
            order=(2, 1, 0),
        )

        b_ptr = tl.make_block_ptr(
            base=weight_ptr,
            shape=(OC, GEMM_K),
            strides=(weight_stride_oc, weight_stride_k),
            offsets=(block_n, 0),
            block_shape=(TILE_N, TILE_K),
            order=(1, 0),
        )

        if HAS_BIAS:
            bias_block_ptr = tl.make_block_ptr(
                base=bias_ptr,
                shape=(OC,),
                strides=(1,),
                offsets=(block_n,),
                block_shape=(TILE_N,),
                order=(0,),
            )
            bias_vals = tl.load(bias_block_ptr, boundary_check=(0,))
        output_ptr = output_ptr + pid_b * output_stride_n

        a_tile = tl.load(a_ptr, boundary_check=(0, 1, 2))
        a_tile = tl.trans(tl.reshape(a_tile, (TILE_M, TILE_K)))
        b_descriptor_load = smt.descriptor_load(b_ptr, (0, 0))
        b = smt.view(b_descriptor_load, (0, 0), (TILE_N, TILE_K), (MICRO_N, MICRO_K))
        sub_num = (min(TILE_M, GEMM_M - TILE_M * pid_m) + SUB_BLK_M - 1) // SUB_BLK_M
        for s in smt.parallel(0, sub_num):
            a = smt.view(
                a_tile, (0, s * SUB_BLK_M), (TILE_K, SUB_BLK_M), (MICRO_K, MICRO_M)
            )
            acc = smt.dot(b, a)
            acc = smt.view(acc, (0, 0), (TILE_N, SUB_BLK_M), (1, 1))
            if HAS_BIAS:
                acc += bias_vals[:, None]
            acc = acc.to(output_ptr.dtype.element_ty)
            o_ptr = tl.make_block_ptr(
                base=output_ptr,
                shape=(OC, GEMM_M),
                strides=(output_stride_oc, output_stride_m),
                offsets=(block_n, block_m + s * SUB_BLK_M),
                block_shape=(TILE_N, SUB_BLK_M),
                order=(1, 0),
            )
            tl.store(o_ptr, acc, boundary_check=(0, 1))


def conv2d(input, weight, bias=None, padding=0, stride=1, dilation=1, groups=1):
    logger.debug("GEMS_SPACEMIT CONV2D")

    N, C, H, W = input.shape
    OC, _, KH, KW = weight.shape

    str_h, str_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dil_h, dil_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // str_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // str_w + 1

    # wave54: a direct output-stationary convolution is 4.3-8.9x faster than the
    # fused im2col+bmm path on stride-1 f32 (bit-identical f32 accumulate order:
    # exact vs the vendor path on every shape measured).  Opt out with
    # FLAGGEMS_SPACEMIT_CONV_FAST=0.
    # wave69: conv_fast_applies() still rejects dilation != 1 for the shared
    # kernels, but the shifted-tap route handles it (the tap base becomes
    # kh*DIL*IW + kw*DIL), so a dilated square 3x3/5x5 stride-1 conv enters
    # conv2d_direct through its own gate.  FLAGGEMS_SPACEMIT_CONV_TAP9=0 or
    # FLAGGEMS_SPACEMIT_CONV_DIL=0 keeps the vendor path.
    from .conv2d_fast import (conv2d_direct, conv_fast_applies,
                              conv_fast_enabled, conv_phase_applies)

    dil_tap = (dil_h == dil_w and dil_h > 1 and groups == 1
               and KH == KW and KH in (3, 5, 7) and str_h == 1 and str_w == 1
               and 3 <= C <= 512 and 32 <= OC <= 512 and OH * OW >= 16
               and (KH == 3 or dil_h > 1
                    or os.environ.get("FLAGGEMS_SPACEMIT_CONV_TAP25", "0") == "1")
               and os.environ.get("FLAGGEMS_SPACEMIT_CONV_DIL", "1") != "0")
    # wave70: dilated depthwise has no fast route at all - conv_fast_applies()
    # rejects dilation, so it falls to the official im2col+bmm path, which at
    # (1,64,56,56) 5x5 d2 measures ~280 ms.  The depthwise kernel only needed
    # the tap offset to carry the dilation (kh*DILH / kw*DILW); both strides and
    # any square or non-square dilation stay expressible.  Floor as the f32
    # depthwise gate.
    dil_dwise = (groups == C == OC and (dil_h != 1 or dil_w != 1)
                 and N * OC * OH * OW * KH * KW >= (1 << 14) and OW >= 4
                 and os.environ.get("FLAGGEMS_SPACEMIT_CONV_DW_DIL", "1") != "0")
    # wave71: dilated and strided at once - the phase planes keep m affine
    dil_phase = (dil_h == dil_w and dil_h > 1 and groups == 1
                 and (str_h > 1 or str_w > 1)
                 and conv_phase_applies(N, C, OC, OH, OW, KH, KW, str_h, str_w))
    if conv_fast_enabled() and (dil_tap or dil_phase or dil_dwise
                                or conv_fast_applies(input, weight, stride, dilation, groups)):
        return conv2d_direct(input, weight, bias, pad_h, pad_w, OH, OW,
                             SH=str_h, SW=str_w, groups=groups,
                             dil_h=dil_h, dil_w=dil_w)

    GEMM_M = OH * OW
    KK = KH * KW
    GEMM_K = C * KK

    im2col_buf = torch.empty(
        (N, GEMM_M, GEMM_K), dtype=input.dtype, device=input.device
    )

    output = torch.empty((N, OC, OH, OW), dtype=input.dtype, device=input.device)

    input_nhwc = input.permute(0, 2, 3, 1).contiguous()
    weight_flat = weight.view(OC, -1).contiguous()

    NUM_IM2COL_BLOCKS = N * OH * OW

    TILE_M = 128
    TILE_N = 128
    TILE_K = triton.next_power_of_2(GEMM_K)
    BLOCK_SIZE_C = 32
    SUB_BLK_M = 32
    MICRO_M = 8
    MICRO_K = 8
    MICRO_N = 16

    num_tiles_m = triton.cdiv(GEMM_M, TILE_M)
    num_tiles_n = triton.cdiv(OC, TILE_N)
    NUM_BMM_TILES_PER_BATCH = num_tiles_m * num_tiles_n
    NUM_BMM_BLOCKS = N * NUM_BMM_TILES_PER_BATCH

    total_blocks = NUM_IM2COL_BLOCKS + NUM_BMM_BLOCKS
    grid = (total_blocks,)

    if bias is not None:
        bias_ptr = bias.contiguous()
    else:
        bias_ptr = torch.empty(0, device=input.device, dtype=input.dtype)

    output_3d = output.view(N, OC, GEMM_M)

    with torch_device_fn.device(input.device):
        fused_im2col_bmm_kernel[grid](
            input_nhwc,
            weight_flat,
            bias_ptr,
            output_3d,
            im2col_buf,
            N,
            C,
            H,
            W,
            KH,
            KW,
            OC,
            str_h,
            str_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            OH,
            OW,
            GEMM_M,
            GEMM_K,
            KK,
            input_nhwc.stride(0),
            input_nhwc.stride(1),
            input_nhwc.stride(2),
            input_nhwc.stride(3),
            im2col_buf.stride(0),
            im2col_buf.stride(1),
            im2col_buf.stride(2),
            weight_flat.stride(0),
            weight_flat.stride(1),
            output_3d.stride(0),
            output_3d.stride(1),
            output_3d.stride(2),
            NUM_IM2COL_BLOCKS=NUM_IM2COL_BLOCKS,
            NUM_BMM_TILES_PER_BATCH=NUM_BMM_TILES_PER_BATCH,
            NUM_TILES_N=num_tiles_n,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            HAS_BIAS=(bias is not None),
            SUB_BLK_M=SUB_BLK_M,
            MICRO_M=MICRO_M,
            MICRO_K=MICRO_K,
            MICRO_N=MICRO_N,
        )

    return output
