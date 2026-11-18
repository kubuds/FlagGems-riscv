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
"""SpacemiT (A100 / K3) flash-attention forward kernel.

Upstream port of the flashed attention tutorial onto ``smt`` descriptors.  The
version that was commented out of the backend registry could not compile; this
one fixes three blockers and folds in the two variants that measured positive
on the board (see ``notes/riscv/flagos/attention-next-2026-11-14.md``):

1. ``_attn_fwd_inner`` was called with one argument too many
   (``BLOCK_SIZE_K // MICRO_N`` -- the callee derives the same value from its
   own constexprs).  Dropped.
2. The launch never supplied ``MICRO_M`` / ``MICRO_K`` / ``MICRO_N``: the shared
   ``attention`` tuned-config section only produces ``block_m`` / ``block_n`` /
   ``num_warps`` / ``num_stages`` (and ``PRE_LOAD_V``, which this kernel does not
   take).  The micro-block is now chosen on the host from the element type and
   passed explicitly, so the kernel no longer depends on that config section.
3. ``scale=None`` reached the kernel as ``None``; PyTorch's default
   ``1/sqrt(E)`` is now resolved in the wrapper.
4. The causal mask was aligned bottom-right (``KV_CTX - Q_CTX`` shift); torch
   aligns it upper-left, so every non-square causal call disagreed with torch
   by O(1).  Fixed to match torch, which is what a drop-in has to do.

Backend note: spine-triton sizes its packed-descriptor staging buffer for the
kernel that is about to run but keeps whatever the previous one left there, so
the first launch after the packed footprint (BLOCK_SIZE_K, BLOCK_M, BLOCK_N)
changes returns garbage (nan at 32->64 and at a 64-tile -> 128-tile switch,
0.7-1.3 maxdiff at 64->128; measured, deterministic, ``libentry`` irrelevant).
The buffer shrinks again for a narrower kernel, so the same kernel launched
again after a narrower one breaks a second time.  The launcher therefore runs the
first launch of each configuration, and any launch whose footprint differs from
the previous one, twice.  It costs one launch on those transitions and
disappears once the backend reallocates properly.

Variants folded in:

* scores are unpacked to a 2D ``row_stride = BLOCK_N`` view and softmax runs one
  row reduction instead of the two-stage inner-block (axis=3 then axis=1)
  reduction: 1.15-1.26x on four shapes.
* causal (STAGE != 1) task order alternates front and back blocks so a program
  gets one cheap and one expensive block instead of all of one kind:
  1.07-1.18x.  Only enabled when ``NUM_BLOCKS % num_ctas == 0`` so the
  permutation stays an exact bijection (a partial trailing pair would duplicate
  one task and drop another).
* dtype-tiered outer block: f16/bf16 uses 128/128 once the query is at least
  256 rows and the KV at least 256 (causal: 512), and 64/64 otherwise.  A wide
  tile halves the block count and wastes half of it on short queries (causal
  tail 64x64 0.48x, decode 0.59x) while winning 1.03-1.17x on the long ones.
* the M tile also follows the query length for decode/short prompts (8/16/32
  rows at q <= 8/16/32): a 32-row tile with one real row spends 31/32 of both
  dots on padding, so 2.5-4.1x on decode and 1.8-2.7x on q16-32.  On
  non-square shapes the N tile is 256 (1.06-1.26x, fewer tile iterations);
  square shapes keep BM == BN because the causal prefix pass cannot overrun
  its KV range without a mask.
* the grid is halved when the block count is at most 32 so every program gets
  two tasks, which the front/back alternation then balances (1x1x1024 1.06x,
  f32 1x4x256 1.18x); larger grids keep 16 programs.
* the KV bound mask is skipped when ``BLOCK_N`` divides ``KV_CTX``: the compare
  plus select over the BLOCK_M x BLOCK_N score tile was 1.07-1.20x of pure
  overhead on the large cells.
* Q is packed once per task instead of once per stage pass (bit-exact, 0.1-0.4%
  on causal shapes).

Environment knobs (perf experiments only; defaults are the shipped choice):
``FLAGGEMS_SPACEMIT_ATTN_FAST=0`` forces the generic path,
``..._ATTN_BM`` / ``..._ATTN_BN`` / ``..._ATTN_CTAS`` override the launch (0 or
unset means the tiered default), ``..._ATTN_ALT=0`` disables the causal task
alternation.
"""

import math
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems.utils import libentry

# (MICRO_M, MICRO_K, MICRO_N) -- the f16 group matches the backend's
# config_pre_hook f16 micro blocks; the f32 group is what the fp32 probe used.
_MICROS_F16 = (16, 8, 32)
_MICROS_F32 = (8, 32, 32)

_SUPPORTED_HEAD_DIM = (32, 64, 128)

# spine-triton leaves the packed-descriptor staging buffer sized for whichever
# kernel ran last: the first launch of a kernel whose packed footprint
# (BLOCK_SIZE_K, BLOCK_M, BLOCK_N) is larger than the previous one reads stale
# staging data and returns garbage (observed: nan at 32->64 and at a 64-tile ->
# 128-tile switch, 0.7-1.3 maxdiff at 64->128; the second launch of the same
# kernel is exact, and a cold process that starts at 128 is exact).  The buffer
# also shrinks again when a narrower kernel runs, so 128-tile -> 64-tile ->
# 128-tile breaks a second time; comparing against the footprint of the previous
# launch (in either direction) covers both.  Every launch of a new footprint is
# therefore run twice; the second launch overwrites the staging with real
# numbers.  Re-check when spine-triton resizes its staging buffers; then this
# can go.
_WARMED = set()
_LAST_FOOTPRINT = None



def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _micros_for(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return _MICROS_F16
    return _MICROS_F32


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i_2d,
    m_i_2d,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    num_m_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
):
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, KV_CTX

    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # q arrives already packed: causal splits the KV range into a prefix pass
    # and a diagonal pass, and the old code re-loaded + re-packed Q in each.

    # torch's is_causal mask is upper-left aligned when Q_CTX != KV_CTX (its
    # math path builds tril(nq, nk) with diagonal 0), so row i attends to
    # columns <= i; there is no KV_CTX - Q_CTX shift here.
    causal_offset = 0
    # per-row causal bound, built once per task instead of once per KV block
    row_end_2d = offs_m[:, None] + causal_offset

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_desc = smt.descriptor_load(K_block_ptr, (0, 0))
        k = smt.view(k_desc, (0, 0), (BLOCK_N, BLOCK_SIZE_K), (MICRO_N, MICRO_K))
        trans_k = tl.permute(k, (1, 0, 3, 2))

        qk = smt.dot(q, trans_k) * qk_scale

        # unpack the inner-block compact scores to contiguous rows: row_stride
        # is BLOCK_N, so max/exp/sum are single axis=1 reductions
        qk_rows = smt.view(qk, (0, 0), (BLOCK_M, BLOCK_N), (1, 1))

        # drop the KV bound whenever BLOCK_N divides KV_CTX: the compare plus
        # select over the BLOCK_M x BLOCK_N score tile is then pure overhead
        if KV_CTX % BLOCK_N == 0:
            if (STAGE == 2) or (STAGE == 4):
                qk_rows = tl.where(
                    row_end_2d >= (start_n + offs_n)[None, :], qk_rows, -1.0e6
                )
        else:
            mask_n = (start_n + offs_n) < KV_CTX
            mask_2d = mask_n[None, :]
            if (STAGE == 2) or (STAGE == 4):
                mask_2d = mask_2d & (row_end_2d >= (start_n + offs_n)[None, :])
            qk_rows = tl.where(mask_2d, qk_rows, -1.0e6)

        row_max = tl.max(qk_rows, axis=1)
        m_i_flat = tl.reshape(m_i_2d, (BLOCK_M,))
        m_ij = tl.maximum(m_i_flat, row_max)

        qk_rows = qk_rows - m_ij[:, None]
        p_rows = tl.math.exp(qk_rows)
        l_ij = tl.sum(p_rows, axis=1)

        alpha = tl.math.exp(m_i_flat - m_ij)
        l_i_flat = tl.reshape(l_i_2d, (BLOCK_M,))
        l_i_flat = l_i_flat * alpha + l_ij

        m_i_2d = tl.reshape(m_ij, (num_m_tiles, MICRO_M))
        l_i_2d = tl.reshape(l_i_flat, (num_m_tiles, MICRO_M))

        alpha_bc = tl.reshape(alpha, (num_m_tiles, 1, MICRO_M, 1))
        acc = acc * alpha_bc

        v_desc = smt.descriptor_load(V_block_ptr, (0, 0))
        v = smt.view(v_desc, (0, 0), (BLOCK_N, BLOCK_SIZE_K), (MICRO_K, MICRO_N))

        p_cast = p_rows.to(v.dtype)
        p_pack = smt.view(p_cast, (0, 0), (BLOCK_M, BLOCK_N), (MICRO_M, MICRO_K))
        acc += smt.dot(p_pack, v)

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    return acc, l_i_2d, m_i_2d


@libentry()
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    M,
    Out,
    acc_buffer,
    sm_scale,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    Z,
    H_Q,
    H_KV,
    GROUP_SIZE,
    Q_CTX,
    KV_CTX,
    HEAD_DIM,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    num_ctas: tl.constexpr,
    ALT: tl.constexpr,
):
    NUM_BLOCKS_M = tl.cdiv(Q_CTX, BLOCK_M)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H_Q

    pid = tl.program_id(0)
    sub_num = tl.cdiv(max(NUM_BLOCKS - pid, 0), num_ctas)

    num_m_tiles: tl.constexpr = BLOCK_M // MICRO_M
    num_k_tiles: tl.constexpr = BLOCK_SIZE_K // MICRO_N

    for block_idx in tl.range(0, sub_num):
        t = pid + num_ctas * block_idx
        if ALT:
            # even pairs walk the cheap front blocks, odd pairs the expensive
            # back blocks, so every program sees both kinds
            pair = t // num_ctas
            pos = t % num_ctas
            front_base = (pair // 2) * num_ctas + pos
            if pair % 2 == 0:
                task = front_base
            else:
                task = NUM_BLOCKS - 1 - front_base
        else:
            task = t
        task_hz_idx = task // NUM_BLOCKS_M
        task_m_idx = task % NUM_BLOCKS_M

        off_z = task_hz_idx // H_Q
        off_hq = task_hz_idx % H_Q
        off_hkv = off_hq // GROUP_SIZE

        q_offset = off_z.to(tl.int64) * stride_qz + off_hq.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_hkv.to(tl.int64) * stride_vh
        o_offset = off_z.to(tl.int64) * stride_oz + off_hq.to(tl.int64) * stride_oh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(Q_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        # pack Q once per task: the causal path splits the KV range in two and
        # both passes need the same packed block
        q_desc = smt.descriptor_load(Q_block_ptr, (0, 0))
        q = smt.view(q_desc, (0, 0), (BLOCK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(KV_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_SIZE_K),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(KV_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_SIZE_K),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + o_offset,
            shape=(Q_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_SIZE_K),
            order=(1, 0),
        )

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=tl.float32) - float("inf")
        l_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=tl.float32) + 1.0
        acc_4d = tl.zeros(
            [num_m_tiles, num_k_tiles, MICRO_M, MICRO_N], dtype=tl.float32
        )

        if STAGE == 4:
            acc_4d, l_i_2d, m_i_2d = _attn_fwd_inner(
                acc_4d,
                l_i_2d,
                m_i_2d,
                q,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M,
                BLOCK_SIZE_K,
                BLOCK_N,
                4,
                offs_m,
                offs_n,
                Q_CTX,
                KV_CTX,
                MICRO_M,
                MICRO_K,
                MICRO_N,
                BLOCK_M // MICRO_M,
                BLOCK_N // MICRO_N,
            )
        else:
            if STAGE & 1:
                acc_4d, l_i_2d, m_i_2d = _attn_fwd_inner(
                    acc_4d,
                    l_i_2d,
                    m_i_2d,
                    q,
                    K_block_ptr,
                    V_block_ptr,
                    task_m_idx,
                    sm_scale,
                    BLOCK_M,
                    BLOCK_SIZE_K,
                    BLOCK_N,
                    4 - STAGE,
                    offs_m,
                    offs_n,
                    Q_CTX,
                    KV_CTX,
                    MICRO_M,
                    MICRO_K,
                    MICRO_N,
                    BLOCK_M // MICRO_M,
                    BLOCK_N // MICRO_N,
                )
            if STAGE & 2:
                acc_4d, l_i_2d, m_i_2d = _attn_fwd_inner(
                    acc_4d,
                    l_i_2d,
                    m_i_2d,
                    q,
                    K_block_ptr,
                    V_block_ptr,
                    task_m_idx,
                    sm_scale,
                    BLOCK_M,
                    BLOCK_SIZE_K,
                    BLOCK_N,
                    2,
                    offs_m,
                    offs_n,
                    Q_CTX,
                    KV_CTX,
                    MICRO_M,
                    MICRO_K,
                    MICRO_N,
                    BLOCK_M // MICRO_M,
                    BLOCK_N // MICRO_N,
                )

        acc_2d = smt.view(acc_4d, (0, 0), (BLOCK_M, BLOCK_SIZE_K), (1, 1))
        m_i = tl.reshape(m_i_2d, (BLOCK_M,))
        l_i = tl.reshape(l_i_2d, (BLOCK_M,))

        m_i = m_i + tl.math.log(l_i)
        accumulator = acc_2d / l_i[:, None]

        mask_m = offs_m < Q_CTX
        m_ptrs = M + task_hz_idx * Q_CTX + offs_m
        tl.store(m_ptrs, m_i.to(M.type.element_ty), mask=mask_m)

        tl.store(
            O_block_ptr, accumulator.to(Out.type.element_ty), boundary_check=(0, 1)
        )


def _launch(q, k, v, sm_scale, is_causal, group_size):
    """Mirror of Attention.forward without autograd; returns the output tensor."""
    head_dim = k.shape[-1]
    block_size_k = triton.next_power_of_2(head_dim)
    q_ctx, kv_ctx = q.shape[2], k.shape[2]
    h_q, h_kv = q.shape[1], k.shape[1]

    # dtype-tiered outer block: f16/bf16 likes 128/128 on shapes with at least
    # two query blocks of 128, but a 128-wide block halves the block count and
    # wastes half the tile when the query is shorter (tail 64x64 is 0.48x,
    # decode 0.59x), so those stay at 64/64.
    block_m = _env_int("FLAGGEMS_SPACEMIT_ATTN_BM", 0)
    block_n = _env_int("FLAGGEMS_SPACEMIT_ATTN_BN", 0)
    if block_m <= 0 or block_n <= 0:
        # outer M tile follows the query length: the kernel computes the whole
        # BLOCK_M x BLOCK_N score tile whatever the query holds, so a 32-row
        # tile with one real row (decode) spends 31/32 of both dots on padding.
        # 8/16/32 by query length is 2.5-4.1x over the old 32/32 on decode and
        # short prompts (q1 kv512 d64 4.67 -> 1.14 ms, q1 kv2048 d128 22.8 ->
        # 4.4 ms); q48 would still pad to 64, so the last tier is 32.
        # causal 256 keeps 64: two query blocks leave too little work per
        # program for a 128-wide tile (0.99x), while causal 512 gains 1.15x.
        wide = (
            q.dtype in (torch.float16, torch.bfloat16)
            and q_ctx >= 256
            and kv_ctx >= 256
            and (not is_causal or kv_ctx >= 512)
        )
        if q_ctx <= 8:
            bm = 8
        elif q_ctx <= 16:
            bm = 16
        elif q_ctx <= 32:
            bm = 32
        else:
            bm = 128 if wide else 64
        # N tile: the old tier always ran BM == BN.  Scanned apart, a 256-wide
        # score tile is worth 1.06-1.26x on non-square shapes (q64 kv512 1.14x,
        # q128 kv512 d128 1.26x, q256 kv512 1.19x) and 1.09-1.26x on plain
        # short prompts, while square plain is neutral to 0.99x and square
        # causal cannot use BM < BN: the prefix pass (STAGE 1) carries no KV
        # bound mask, so its range has to be a whole number of N tiles.
        if max(q_ctx, bm) == kv_ctx:
            bm = max(bm, 32)
            block_m = bm
            block_n = bm
        else:
            block_m = bm
            block_n = min(256, max(32, triton.next_power_of_2(kv_ctx)))
    num_ctas = _env_int("FLAGGEMS_SPACEMIT_ATTN_CTAS", 0)
    micro_m, micro_k, micro_n = _micros_for(q.dtype)
    # the f16 micro group is 16 rows; an 8-row outer tile (decode) uses the
    # 8-row micro tile instead, which the backend accepts for f16 as well
    if block_m < micro_m:
        micro_m = block_m

    # a Q block shorter than BLOCK_M makes the Q descriptor a 1 x ? tensor and
    # the smt.view into micro tiles fails to lower (decode: q_len 1).  Pad the
    # query rows up to BLOCK_M and slice the result; the padding rows cost one
    # block of work and are dropped.
    padded = q_ctx < block_m
    if padded:
        q_run = torch.zeros(
            (q.shape[0], h_q, block_m, head_dim), dtype=q.dtype, device=q.device
        )
        q_run[:, :, :q_ctx] = q
        o_run = torch.empty_like(q_run)
        m_run = torch.empty(
            (q.shape[0], h_q, block_m), dtype=torch.float32, device=q.device
        )
        q_ctx_run = block_m
    else:
        q_run = q
        o_run = torch.empty_like(q)
        m_run = torch.empty(
            (q.shape[0], h_q, q_ctx), dtype=torch.float32, device=q.device
        )
        q_ctx_run = q_ctx

    acc = torch.empty(
        (q.shape[0], h_q, q_ctx_run, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    stage = 3 if (q_ctx_run == kv_ctx) else 4
    if not is_causal:
        stage = 1

    num_blocks = triton.cdiv(q_ctx_run, block_m) * q.shape[0] * h_q
    if num_ctas <= 0:
        # give every program two tasks when the grid is small: with the causal
        # front/back alternation that pairs one cheap and one expensive block in
        # the same program (1x1x1024 1.06x, f32 1x4x256 1.18x), and any grid
        # that already covers 8 cores twice keeps 16.
        num_ctas = 16 if num_blocks > 32 else max(1, num_blocks // 2)
    num_ctas = max(1, min(num_ctas, num_blocks))
    alt = (
        stage != 1
        and _env_int("FLAGGEMS_SPACEMIT_ATTN_ALT", 1) != 0
        and num_blocks % num_ctas == 0
        and num_blocks > num_ctas
    )

    def _run():
        _attn_fwd[(num_ctas,)](
            q_run,
            k,
            v,
            m_run,
            o_run,
            acc,
            sm_scale,
            q_run.stride(0),
            q_run.stride(1),
            q_run.stride(2),
            q_run.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o_run.stride(0),
            o_run.stride(1),
            o_run.stride(2),
            o_run.stride(3),
            q_run.shape[0],
            h_q,
            h_kv,
            group_size,
            q_ctx_run,
            kv_ctx,
            head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            STAGE=stage,
            BLOCK_SIZE_K=block_size_k,
            MICRO_M=micro_m,
            MICRO_K=micro_k,
            MICRO_N=micro_n,
            num_ctas=num_ctas,
            ALT=alt,
            num_warps=8,
            num_stages=3,
        )

    _run()
    global _LAST_FOOTPRINT
    key = (
        block_size_k,
        stage,
        block_m,
        block_n,
        micro_m,
        micro_k,
        micro_n,
        num_ctas,
        alt,
    )
    # two triggers: a configuration that has never run, and a launch whose packed
    # footprint differs from the previous launch in this process in either
    # direction (128-tile -> 64-tile -> 128-tile breaks a second time).  When the
    # K tile widens the first launch has been seen to stay wrong even after one
    # repeat (one unreproduced 2.8 maxdiff in an A/B run, 56/56 transition pairs
    # clean afterwards), so that path gets a third launch; it costs one launch on
    # a head-dim change only.
    foot = (block_size_k, block_m, block_n)
    foot_changed = _LAST_FOOTPRINT is not None and foot != _LAST_FOOTPRINT
    widen = _LAST_FOOTPRINT is not None and foot[0] > _LAST_FOOTPRINT[0]
    if (key not in _WARMED or foot_changed) and _env_int(
        "FLAGGEMS_SPACEMIT_ATTN_WARMUP", 1
    ) != 0:
        # see _WARMED: the first launch after the staging footprint changed reads
        # stale data, the second one is exact
        _run()
        if widen:
            _run()
    _WARMED.add(key)
    _LAST_FOOTPRINT = foot
    if padded:
        return o_run[:, :, :q_ctx].contiguous()
    return o_run

class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, is_causal, enable_gqa: bool):
        head_dim_q, head_dim_k = q.shape[-1], k.shape[-1]
        assert head_dim_q == head_dim_k == v.shape[-1]

        h_q = q.shape[1]
        h_kv = k.shape[1]
        if h_q != h_kv:
            enable_gqa = True
        if enable_gqa:
            assert (
                h_q % h_kv == 0
            ), f"GQA requires H_Q % H_KV == 0, got H_Q={h_q}, H_KV={h_kv}"
            group_size = h_q // h_kv
        else:
            assert h_q == h_kv
            group_size = 1

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim_k)

        return _launch(q, k, v, float(sm_scale), is_causal, group_size)


def _fast_applies(query, key, value, attn_mask, dropout_p):
    if os.environ.get("FLAGGEMS_SPACEMIT_ATTN_FAST", "1") == "0":
        return False
    if attn_mask is not None or dropout_p not in (0, 0.0):
        return False
    if query.dtype not in (torch.float16, torch.float32):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return False
    head_dim = query.shape[-1]
    if head_dim != key.shape[-1] or head_dim != value.shape[-1]:
        return False
    if head_dim not in _SUPPORTED_HEAD_DIM:
        return False
    if query.shape[2] == 0 or key.shape[2] == 0:
        return False
    return True


def _generic_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
    from flag_gems.ops.attention import scaled_dot_product_attention as _impl

    return _impl(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def flash_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if not _fast_applies(query, key, value, attn_mask, dropout_p):
        return _generic_sdpa(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    return Attention.apply(query, key, value, scale, is_causal, enable_gqa)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if not _fast_applies(query, key, value, attn_mask, dropout_p):
        return _generic_sdpa(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    return Attention.apply(query, key, value, scale, is_causal, enable_gqa)
