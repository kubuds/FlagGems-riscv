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
* plain shapes whose single query block still carries at least 2 MiB of K/V
  rows (long decode, wide prompts) run as two KV halves in one launch
  (flash-decoding style) merged by ``_attn_combine``: 1.10-1.77x, largest on
  the very shapes where the one-program scan was fixed-cost bound
  (q128 kv1024 d128 7.2 -> 4.1 ms, q1 kv8192 d128 4.1 -> 2.8 ms).  Two splits
  is the measured optimum; four and eight lose to bandwidth and prologue cost.
* non-square causal shapes stop scanning at the last query row of the tile
  instead of walking the whole KV axis (torch aligns the mask upper-left, so
  row i never reads column > i): q32 kv3000 20.5 -> 2.2 ms, q128 kv1024
  14.5 -> 4.2 ms, q1 kv2048 1.59 -> 0.68 ms, bit-identical to the full scan.

Environment knobs (perf experiments only; defaults are the shipped choice):
``FLAGGEMS_SPACEMIT_ATTN_FAST=0`` forces the generic path,
``..._ATTN_BM`` / ``..._ATTN_BN`` / ``..._ATTN_CTAS`` override the launch (0 or
unset means the tiered default), ``..._ATTN_ALT=0`` disables the causal task
alternation, ``..._ATTN_SPLIT=0`` disables the KV split and ``..._ATTN_NS``
forces its split count.
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
    LO,
    HI,
    ROW_MOD: tl.constexpr,
    EXACT_BOUND: tl.constexpr,
):
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    elif STAGE == 5 or STAGE == 6:
        # explicit [LO, HI) range: the KV-split pass hands each program one
        # chunk of the KV axis (STAGE 5 plain, 6 causal)
        lo, hi = LO, HI
    elif STAGE == 7:
        # non-square causal: torch aligns the mask upper-left, so row i only
        # attends columns <= i and a block of rows starting at start_m*BLOCK_M
        # never reads past (start_m + 1) * BLOCK_M.  Scanning KV_CTX keys for a
        # one-row decode was 99% masked work (q1 kv2048: 2048 keys scanned to
        # attend to key 0).
        if ROW_MOD > 0:
            # wave97: a packed block holds whole query groups, so every row's
            # query position is < ROW_MOD and the block never reads past
            # ROW_MOD keys however many rows it packs (q8 kv2048 used to scan
            # 64 keys for 8 queries; now it scans 8).
            # keep the bound a runtime value: folding it to a constant 1
            # aborts spine-opt on a KV_CTX == 1 descriptor (expected
            # tensor<?x1x32x8>, got 4x4x32x8), so clamp the old runtime
            # expression instead of using ROW_MOD directly
            hi = tl.minimum((start_m + 1) * BLOCK_M, KV_CTX)
            if EXACT_BOUND:
                # wave99: row r is query r % ROW_MOD, so a block starting at
                # start_m * BLOCK_M spans query positions s .. s + BLOCK_M - 1
                # (mod ROW_MOD), s = start_m * BLOCK_M % ROW_MOD.  When the
                # block covers only part of a group (BLOCK_M < ROW_MOD) the
                # last live query in it is s + BLOCK_M - 1, not ROW_MOD - 1:
                # g2 q128 at a 64-row tile scanned all 128 keys for 64 live
                # rows.  Clamp to s + BLOCK_M (exclusive) instead.
                s0 = (start_m * BLOCK_M) % ROW_MOD
                hi = tl.minimum(hi, tl.minimum(ROW_MOD, s0 + BLOCK_M))
            else:
                hi = tl.minimum(hi, ROW_MOD)
            lo = 0
        else:
            lo, hi = 0, tl.minimum((start_m + 1) * BLOCK_M, KV_CTX)
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
    if ROW_MOD > 0:
        # wave96/97: a packed GQA block puts whole (head, query) groups on the
        # M axis, so row r is query position r % nq, not row r.  nq == 1 (the
        # decode case) collapses to all-zero, which is exactly torch's causal
        # mask for a one-token query.  Reading the packed row index as the
        # query position would let row r see keys <= r.
        row_end_2d = (offs_m % ROW_MOD)[:, None] + causal_offset
    else:
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
            if (STAGE == 2) or (STAGE == 4) or (STAGE == 6) or (STAGE == 7):
                qk_rows = tl.where(
                    row_end_2d >= (start_n + offs_n)[None, :], qk_rows, -1.0e6
                )
        else:
            mask_n = (start_n + offs_n) < KV_CTX
            mask_2d = mask_n[None, :]
            if (STAGE == 2) or (STAGE == 4) or (STAGE == 6) or (STAGE == 7):
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
    ROW_MOD: tl.constexpr,
    EXACT_BOUND: tl.constexpr,
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

        if STAGE == 4 or STAGE == 7:
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
                4 if STAGE == 4 else 7,
                offs_m,
                offs_n,
                Q_CTX,
                KV_CTX,
                MICRO_M,
                MICRO_K,
                MICRO_N,
                BLOCK_M // MICRO_M,
                BLOCK_N // MICRO_N,
                0,
                0,
                ROW_MOD,
                EXACT_BOUND,
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
                    0,
                    0,
                    ROW_MOD,
                    EXACT_BOUND,
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
                    0,
                    0,
                    ROW_MOD,
                    EXACT_BOUND,
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



@libentry()
@triton.jit
def _attn_fwd_split(
    Q,
    K,
    V,
    MMAX,
    LSUM,
    ACC,
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
    Z,
    H_Q,
    H_KV,
    GROUP_SIZE,
    Q_CTX,
    KV_CTX,
    HEAD_DIM,
    CHUNK,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    NS: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """One KV chunk of a plain (non-causal) attention task.

    A decode call leaves seven of eight cores idle in the single-pass kernel;
    each program here owns a CHUNK-wide slice of the KV axis and writes the
    unnormalized accumulator plus its running max / sum so _attn_combine can
    merge the slices with the exact online-softmax correction.
    """
    NUM_BLOCKS_M = tl.cdiv(Q_CTX, BLOCK_M)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H_Q

    pid = tl.program_id(0)
    split = tl.program_id(1)
    sub_num = tl.cdiv(max(NUM_BLOCKS - pid, 0), num_ctas)

    lo = split * CHUNK
    hi = tl.minimum(lo + CHUNK, KV_CTX)

    num_m_tiles: tl.constexpr = BLOCK_M // MICRO_M
    num_k_tiles: tl.constexpr = BLOCK_SIZE_K // MICRO_N

    for block_idx in tl.range(0, sub_num):
        t = pid + num_ctas * block_idx
        task_hz_idx = t // NUM_BLOCKS_M
        task_m_idx = t % NUM_BLOCKS_M

        off_z = task_hz_idx // H_Q
        off_hq = task_hz_idx % H_Q
        off_hkv = off_hq // GROUP_SIZE

        q_offset = off_z.to(tl.int64) * stride_qz + off_hq.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_hkv.to(tl.int64) * stride_vh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(Q_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
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

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        # l starts at 0 (not 1) so the combine is an exact weighted sum of the
        # partial normalizers
        m_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=tl.float32) - float("inf")
        l_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=tl.float32)
        acc_4d = tl.zeros(
            [num_m_tiles, num_k_tiles, MICRO_M, MICRO_N], dtype=tl.float32
        )

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
            5,
            offs_m,
            offs_n,
            Q_CTX,
            KV_CTX,
            MICRO_M,
            MICRO_K,
            MICRO_N,
            num_m_tiles,
            num_k_tiles,
            lo,
            hi,
            False,
            False,
        )

        acc_2d = smt.view(acc_4d, (0, 0), (BLOCK_M, BLOCK_SIZE_K), (1, 1))
        m_i = tl.reshape(m_i_2d, (BLOCK_M,))
        l_i = tl.reshape(l_i_2d, (BLOCK_M,))

        rows = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_SIZE_K)
        rmask = rows < Q_CTX
        acc_off = (split * Z * H_Q + task_hz_idx) * Q_CTX
        ml_off = (split * Z * H_Q + task_hz_idx) * Q_CTX
        tl.store(
            ACC + acc_off * HEAD_DIM + rows[:, None] * HEAD_DIM + cols[None, :],
            acc_2d,
            mask=rmask[:, None],
        )
        tl.store(MMAX + ml_off + rows, m_i, mask=rmask)
        tl.store(LSUM + ml_off + rows, l_i, mask=rmask)


@triton.jit
def _attn_combine(
    MMAX,
    LSUM,
    ACC,
    Out,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    Z,
    H,
    Q_CTX,
    HEAD_DIM,
    NS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    zh = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < Q_CTX
    cols = tl.arange(0, BLOCK_K)
    ml_base = zh * Q_CTX
    acc_base = zh * Q_CTX * HEAD_DIM
    ml_stride = Z * H * Q_CTX
    acc_stride = Z * H * Q_CTX * HEAD_DIM

    m_best = tl.full([BLOCK_M], float("-inf"), tl.float32)
    for s in tl.static_range(NS):
        m_best = tl.maximum(
            m_best,
            tl.load(
                MMAX + s * ml_stride + ml_base + rows,
                mask=rmask,
                other=float("-inf"),
            ),
        )

    denom = tl.zeros([BLOCK_M], tl.float32)
    num = tl.zeros([BLOCK_M, BLOCK_K], tl.float32)
    for s in tl.static_range(NS):
        m_s = tl.load(
            MMAX + s * ml_stride + ml_base + rows, mask=rmask, other=float("-inf")
        )
        l_s = tl.load(LSUM + s * ml_stride + ml_base + rows, mask=rmask, other=0.0)
        w = tl.math.exp(m_s - m_best)
        denom += l_s * w
        a = tl.load(
            ACC + s * acc_stride + acc_base + rows[:, None] * HEAD_DIM + cols[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        num += a * w[:, None]

    out = num / denom[:, None]
    o_base = (zh // H) * stride_oz + (zh % H) * stride_oh
    tl.store(
        Out + o_base + rows[:, None] * stride_om + cols[None, :] * stride_on,
        out.to(Out.type.element_ty),
        mask=rmask[:, None] & (cols[None, :] < HEAD_DIM),
    )


_PLAN_CACHE = {}
_BUF_POOL = {}


def _scratch(shape, dtype, device, kind):
    """Reusable f32 scratch for one attention call.

    Called tens or hundreds of times per second on a decode loop; three
    torch.empty calls are ~27 us of a 0.65 ms call.  The buffers are never
    returned to the caller, so a per-(kind, device, dtype, shape) buffer is
    safe: kernels on one stream are ordered, the next call overwrites the
    previous numbers after the previous kernel has consumed them.
    """
    key = (kind, str(device), dtype, shape)
    if _env_int("FLAGGEMS_SPACEMIT_ATTN_HOSTFAST", 1) == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    t = _BUF_POOL.get(key)
    if t is None:
        t = torch.empty(shape, dtype=dtype, device=device)
        _BUF_POOL[key] = t
    return t

_PLAN_KEYS = (
    "FLAGGEMS_SPACEMIT_ATTN_BM",
    "FLAGGEMS_SPACEMIT_ATTN_BN",
    "FLAGGEMS_SPACEMIT_ATTN_CTAS",
    "FLAGGEMS_SPACEMIT_ATTN_NS",
    "FLAGGEMS_SPACEMIT_ATTN_ALT",
    "FLAGGEMS_SPACEMIT_ATTN_SPLIT",
    "FLAGGEMS_SPACEMIT_ATTN_PACK_BN",
    "FLAGGEMS_SPACEMIT_ATTN_F32TILE",
    "FLAGGEMS_SPACEMIT_ATTN_PACKBOUND",
)


def _plan(q, k, v, is_causal, group_size, row_mod=0):
    """Tiles, grid, stage and split for one attention call.

    Pure function of the call signature; the decode path runs it on every
    token, where the ~11 us of env lookups and branches is 2% of the call, so
    hits are memoized.  Any experiment override in the environment bypasses
    the cache (the knobs are read live).
    """
    key = (
        q.dtype,
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        k.shape[2],
        k.shape[-1],
        bool(is_causal),
        group_size,
        row_mod,
    )
    live = any(os.environ.get(n) for n in _PLAN_KEYS) or _env_int(
        "FLAGGEMS_SPACEMIT_ATTN_HOSTFAST", 1
    ) == 0
    if not live:
        hit = _PLAN_CACHE.get(key)
        if hit is not None:
            return hit

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
    if (
        q.dtype == torch.float32
        and block_m <= 8
        and q_ctx < kv_ctx
        and q_ctx <= 8
        and _env_int("FLAGGEMS_SPACEMIT_ATTN_F32TILE", 1) != 0
    ):
        # wave98: f32 has no matrix engine, so an 8-row tile with one real row
        # is pure padding -- the dot runs on the vector unit and three empty
        # rows cost as much as the real one.  next_power_of_2(q_ctx) rows
        # (1/2/4/8) measured 1.45-1.89x on q1..q4 (q1 kv512 d64 0.785 -> 0.502,
        # q1 kv2048 d64 1.60 -> 0.85, q1 kv2048 d128 1.77 -> 1.38,
        # q4 kv1024 1.08 -> 0.76) and neutral at q8.  A causal pass also wants
        # the N tile at 128 (its bound is KV-independent, so a 256-wide tile
        # spends most of the dot masked: q1 kv512 0.454 vs 0.432, q8 kv512
        # 0.573 -> 0.496); plain keeps the tier's 256, which is 1.6% better at
        # kv2048.  f16 keeps the 8-row floor: BLOCK_M 1 fails spine-opt
        # (vector size 0) and its 16-row micro group already pays the padding.
        block_m = max(1, min(8, triton.next_power_of_2(q_ctx)))
        if is_causal:
            block_n = min(block_n, 128)
    if row_mod > 0 and is_causal:
        # wave97: the packed causal scan stops at nq keys, so the default
        # 256-wide N tile spends 8x the dot on masked columns; 32 is the
        # narrowest tile the backend accepts (BN 16 aborts, wave80)
        pack_bn = _env_int("FLAGGEMS_SPACEMIT_ATTN_PACK_BN", 32)
        if pack_bn > 0:
            block_n = min(block_n, pack_bn)
    # wave99: exact per-block causal bound for packed rows (row r -> query
    # r % nq, so a block only needs keys up to start%nq + BLOCK_M).  Off
    # restores the wave98 "every block scans nq keys" bound for A/B.
    exact = row_mod > 0 and _env_int("FLAGGEMS_SPACEMIT_ATTN_PACKBOUND", 1) != 0

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
    q_ctx_run = block_m if padded else q_ctx

    # causal: square shapes split the KV range into a prefix and a diagonal pass
    # (STAGE 3 = both); non-square shapes keep one masked pass, but it stops at
    # the last query row of the block instead of scanning the whole KV (STAGE 7)
    # a packed causal block maps row -> query position itself, so it can never
    # take the square prefix+diagonal split (that split assumes row i is query
    # i); it always uses the masked pass, whose bound is ROW_MOD
    stage = 3 if (q_ctx_run == kv_ctx and row_mod == 0) else 7
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

    # KV split (flash-decoding): one decode block leaves seven of eight cores
    # idle, and the single-program scan is a fixed ~0.25 ms plus ~0.44 us/key.
    # Hand each program half of the KV axis, then merge the two partial
    # (m, l, acc) triples.  Measured: two splits is the sweet spot on this
    # backend (4/8 splits cost more than they save -- K/V bandwidth and the
    # per-program prologue dominate before the cores do), and the win only
    # appears once each program still has >= 2 MiB of K/V rows to chew through
    # (q1 kv512 d64 is 0.26 MiB per program and 0.83x; q1 kv2048 d128 is
    # 2.1 MiB and 1.13x; q128 kv1024 d128 is 16 MiB and 1.74x).  Plain only:
    # a causal chunk past a row's key bound is all-masked work.
    split_ns = 0
    split_chunk = 0
    if (
        stage == 1
        and _env_int("FLAGGEMS_SPACEMIT_ATTN_SPLIT", 1) != 0
        and kv_ctx >= 512
        and num_blocks <= 8
        and block_m * kv_ctx * head_dim >= (2 << 20)
    ):
        ns = 2
        forced = _env_int("FLAGGEMS_SPACEMIT_ATTN_NS", 0)
        if forced:
            ns = max(1, min(16, forced))
        if ns >= 2:
            chunk = triton.cdiv(triton.cdiv(kv_ctx, ns), block_n) * block_n
            split_ns = triton.cdiv(kv_ctx, chunk)
            split_chunk = chunk

    plan = (
        head_dim,
        block_size_k,
        q_ctx,
        kv_ctx,
        h_q,
        h_kv,
        block_m,
        block_n,
        micro_m,
        micro_k,
        micro_n,
        num_ctas,
        alt,
        stage,
        padded,
        q_ctx_run,
        split_ns,
        split_chunk,
        exact,
    )
    if not live:
        _PLAN_CACHE[key] = plan
    return plan

def _launch(q, k, v, sm_scale, is_causal, group_size, row_mod=0):
    """Mirror of Attention.forward without autograd; returns the output tensor."""
    (
        head_dim,
        block_size_k,
        q_ctx,
        kv_ctx,
        h_q,
        h_kv,
        block_m,
        block_n,
        micro_m,
        micro_k,
        micro_n,
        num_ctas,
        alt,
        stage,
        padded,
        q_ctx_run,
        split_ns,
        split_chunk,
        exact,
    ) = _plan(q, k, v, is_causal, group_size, row_mod)

    if padded:
        q_run = torch.zeros(
            (q.shape[0], h_q, block_m, head_dim), dtype=q.dtype, device=q.device
        )
        q_run[:, :, :q_ctx] = q
        o_run = torch.empty_like(q_run)
    else:
        q_run = q
        o_run = torch.empty_like(q)

    m_run = _scratch(
        (q.shape[0], h_q, q_ctx_run), torch.float32, q.device, "m"
    )
    acc = _scratch(
        (q.shape[0], h_q, q_ctx_run, head_dim), torch.float32, q.device, "acc"
    )

    def _run_main():
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
            ROW_MOD=row_mod,
            EXACT_BOUND=exact,
            num_warps=8,
            num_stages=3,
        )

    def _run_split():
        acc_s = _scratch(
            (split_ns * q.shape[0] * h_q * q_ctx_run, head_dim),
            torch.float32,
            q.device,
            "acc_s",
        )
        mmax = _scratch(
            (split_ns * q.shape[0] * h_q * q_ctx_run,),
            torch.float32,
            q.device,
            "mmax",
        )
        lsum = _scratch(
            (split_ns * q.shape[0] * h_q * q_ctx_run,),
            torch.float32,
            q.device,
            "lsum",
        )
        _attn_fwd_split[(num_ctas, split_ns)](
            q_run,
            k,
            v,
            mmax,
            lsum,
            acc_s,
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
            q_run.shape[0],
            h_q,
            h_kv,
            group_size,
            q_ctx_run,
            kv_ctx,
            head_dim,
            split_chunk,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_SIZE_K=block_size_k,
            MICRO_M=micro_m,
            MICRO_K=micro_k,
            MICRO_N=micro_n,
            NS=split_ns,
            num_ctas=num_ctas,
            num_warps=8,
            num_stages=3,
        )
        _attn_combine[(triton.cdiv(q_ctx_run, 64), q_run.shape[0] * h_q)](
            mmax,
            lsum,
            acc_s,
            o_run,
            o_run.stride(0),
            o_run.stride(1),
            o_run.stride(2),
            o_run.stride(3),
            q_run.shape[0],
            h_q,
            q_ctx_run,
            head_dim,
            NS=split_ns,
            BLOCK_M=64,
            BLOCK_K=block_size_k,
            num_warps=4,
            num_stages=2,
        )

    if split_ns:
        _run_split()
    else:
        _run_main()
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
        split_ns,
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
        if split_ns:
            # the split path fires two kernels back to back (partials, combine);
            # the second launch of the pair in a process is the one that reads
            # a half-written staging buffer, so the split warmup burns one extra
            # pair -- otherwise the first real decode call of a footprint comes
            # back with one chunk's normalizer wrong (measured: 7.8e-02 on
            # q1 kv2048 d128, exact from the third pair on)
            _run_split()
            _run_split()
        else:
            _run_main()
        if widen:
            if split_ns:
                _run_split()
            else:
                _run_main()
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

        # GQA head packing: GROUP_SIZE query heads share one K/V head, and the
        # rows a short query leaves empty in the M tile (decode pads 1 real row
        # up to 8) are exactly the rows the other heads of the group can fill.
        # A (b, H_Q, nq, d) contiguous query is (b, H_KV, GROUP_SIZE*nq, d) with
        # the packed rows being (head, query) pairs, so this is a pure re-view:
        # each packed row keeps its own score row, only K/V is shared, which is
        # what GQA already guarantees.  Causal needs the kernel to know where
        # each packed row sits on the query axis: row r is query r % nq, which
        # ROW_MOD hands to the mask (wave97 extends this from decode-only to
        # nq <= 16; the packed bound stays a single 32-wide KV tile).
        if (
            group_size > 1
            and q.is_contiguous()
            # wave97/98 packing caps.  causal packing collapses the KV scan to
            # nq keys, so a 128-row block pays for itself: g2 q32 2.41x,
            # g2 q64 2.18x, g4 q32 1.73x, g8 q16 1.67x, g16 q8 1.65x,
            # g8 q16 b2 2.81x at 128 rows.  Plain packing only fills the
            # padding rows of a short query, so 32 rows stays the rule --
            # except when the per-head query itself is short (nq <= 4): there
            # an 8-row unpacked tile is mostly padding, and 128 packed rows
            # win 1.8-4.9x (g16 q4 2.10 -> 1.41, g32 q2 3.68 -> 1.40,
            # g64 q1 6.84 -> 1.41, g64 q2 6.98 -> 2.07, g64 q2 kv1024 d128
            # 15.39 -> 4.36, g32 q4 3.78 -> 2.07), while nq 8 at 64 rows
            # loses (g8 q8 0.91x) and at 128 rows is neutral (g16 q8 1.01x).
            and (
                (
                    is_causal
                    and q.shape[2] <= k.shape[2]
                    and (
                        q.shape[2] * group_size <= 128
                        # wave99: the row cap was set when only 64/128-row
                        # packings had been measured.  Packing more rows shares
                        # one K/V stream across more (head, query) groups and
                        # cuts the per-block prologue count; the scan itself is
                        # bounded by nq, so it stays cheap.  Measured against
                        # the unpacked entry: 256 rows g8 q32 2.00x, g16 q16
                        # 1.94x, g32 q8 2.19x, g64 q4 3.83x, f32 g16 q16 3.27x;
                        # 512 rows g8 q64 2.52x (b2 2.45x), g16 q32 2.73x,
                        # g32 q16 3.05x, g64 q8 3.45x, f32 g32 q16 3.61x;
                        # 1024 rows g8 q128 1.48x, g16 q64 2.26x, g32 q32
                        # 3.24x.  Boundaries: g2 at 256 rows and g4 at 768 rows
                        # lose (0.72x, 0.86x), so the cap grows with the group
                        # (128 / 512 / 1024 for g2 / g4 / g8+).
                        or (
                            group_size >= 4
                            and q.shape[2] * group_size <= 512
                        )
                        or (
                            group_size >= 8
                            and q.shape[2] * group_size <= 1024
                        )
                        # wave100: 2048 rows measured against 1024 for the
                        # bigger groups.  The per-block scan is bounded by nq,
                        # so extra rows only buy K/V reuse across groups; that
                        # holds while nq stays <= 64.  At 2048 rows: g64 q32
                        # kv512 4.25x, g64 q32 kv1024 d128 3.22x, g128 q16
                        # 3.68x, g32 q64 2.50x; but g16 q128 (nq 128) 1.38x and
                        # g8 q256 (nq 256) 0.98x, so nq > 64 stays out.  Grid
                        # sweep at 2048 rows (ctas 2/4/8/16/32) is flat within
                        # 2%, so the default grid rule is unchanged.
                        or (
                            group_size >= 16
                            and q.shape[2] <= 64
                            and q.shape[2] * group_size <= 2048
                        )
                        # wave101: 4096 rows for the big groups.  The trade is
                        # the same as at 2048 - the per-block scan is set by
                        # nq, so extra rows only buy K/V reuse across the
                        # (head, query) groups - and the nq <= 64 boundary
                        # still holds: g128 q32 kv512 4.37x, g256 q16 3.80x,
                        # g64 q64 2.64x, g128 q32 kv1024 d128 3.05x, while
                        # g16 q256 (nq 256) at 4096 rows is 1.06x.  g64 is the
                        # smallest group that measured a win at this size (the
                        # 2048 tier still covers g16..g32).
                        or (
                            group_size >= 64
                            and q.shape[2] <= 64
                            and q.shape[2] * group_size <= 4096
                        )
                        # wave102: past 4096 rows the ratio is flat - the scan
                        # is set by nq, so extra rows only buy K/V reuse - and
                        # the nq boundary moves up to 128: 8192 rows g128 q64
                        # kv512 2.57x, g256 q32 kv512 4.07x, g128 q64 kv1024
                        # d128 2.35x, g512 q16 3.88x, f32 g128 q64 2.88x;
                        # 16384 rows g256 q64 2.58x, g128 q128 1.47x; 32768
                        # rows g512 q64 2.58x, g256 q128 1.42x.  nq 128 wins
                        # at 4096 rows too (g32 q128 1.42x) while nq 256
                        # loses at both sizes (0.94x at 4096, 0.98x at 8192),
                        # so this tier is group >= 32, nq <= 128, rows <=
                        # 32768.  Plain stays out at this size (1.06x at 8192
                        # rows).
                        or (
                            group_size >= 32
                            and q.shape[2] <= 128
                            and q.shape[2] * group_size <= 32768
                        )
                    )
                )
                or (not is_causal and q.shape[2] * group_size <= 32)
                or (
                    not is_causal
                    and q.shape[2] <= 4
                    and q.shape[2] * group_size <= 128
                )
            )
            and _env_int("FLAGGEMS_SPACEMIT_ATTN_PACK", 1) != 0
        ):
            b, hq, nq, hd = q.shape
            out = _launch(
                q.view(b, h_kv, group_size * nq, hd),
                k,
                v,
                float(sm_scale),
                is_causal,
                1,
                row_mod=(nq if is_causal else 0),
            )
            return out.view(b, hq, nq, hd)

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
