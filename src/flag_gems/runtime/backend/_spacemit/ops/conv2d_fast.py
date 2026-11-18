# wave57 candidate: conv2d for the K3 backend.
#
# wave60 adds the 1x1 stride-1 route: in NCHW that convolution is already a
# (OC,C) @ (C,H*W) GEMM with an NCHW result, so it runs through a plain dot
# kernel at 43-53 GFLOPS instead of the tap kernel's 12-15.  Everything below
# is the tap path, which still serves every other kernel size.
#
# On top of wave56 (ROWS per program, BLOCK_OW following OW, depthwise kernel,
# NOMASK at ROWS == 1):
#   - grouped conv (1 < groups < C) is covered: each group is an independent
#     (CG -> OCG) convolution whose channel slabs sit back to back in x, w and
#     y, so it runs as one wave56 launch per group on a strided slice.  No
#     kernel change.  1.5-2.8x against the vendor path on 784 px/group and up,
#     and the gate keeps the small groups (196-256 px) on the vendor path
#     because there the dispatch dominates (0.72-0.76x).
#   - BLOCK_OC=128 whenever OC >= 128: bit-exact (it only changes which program
#     owns an output channel) and 1.08-1.15x on (1,128,28,28) and
#     (1,256,14,14).
#   - the tile picker never returns dot N < 32 (ROWS*BLOCK_OW), which aborted
#     spine-opt in LowerMMT4DPattern on 16-wide tiles at ROWS=1, and a 16-wide
#     tile always takes BLOCK_OC=64 (the 32x16 dot aborts).
#
# On top of wave55 (prepacked (KH,KW,OC,C) weights, stride support, tile rule):
#   - ROWS: one program computes ROWS consecutive output rows, stacked along
#     the dot's N dimension, so each weight tile feeds ROWS*BLOCK_OW columns
#     instead of BLOCK_OW.  Measured 1.3-1.4x at (1,64,56,56) r4 vs r1.
#   - BLOCK_OW follows the output width (next pow2, capped 64) instead of a
#     fixed 64: at (1,128,28,28) OW=28, w32 is 1.6x faster than w64 because
#     three quarters of a 64-wide tile were masked off.
#   - depthwise (groups == C == OC) gets its own elementwise kernel; the dot
#     path cannot serve it (K would be one lane) and the vendor im2col path is
#     far slower than a plain 9-tap FMA stream.
#   - NOMASK: at ROWS == 1 a fully divisible shape can drop every mask.  At
#     ROWS > 1 spine-opt rejects the unmasked 3D load ("different memory spaces
#     for base memref and view memref"), so the masked path stays.
#
# Taps must stay tl.static_range: a dynamic tap loop around the dot makes
# spine-opt abort with "scf.for not bufferizable".  The input is pre-padded
# because a scalar negative offset in the load pointer makes PtrAnalysis give
# up and the kernel goes scalar.
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:  # wave61: the smt descriptor/TCM path that flag_gems.mm uses
    import triton.language.extra.smt as smt
except Exception:  # a wheel without the spacemit extra keeps the tl path
    smt = None

# wave64: stem patch-matrix buffers, keyed (kpad, M, K, dtype, device); the pad
# tail is zeroed once at creation and the build rewrites every real row
_STEM_BUF = {}


def conv_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_CONV_FAST", "1") != "0"


def conv_fast_applies(x, w, stride, dilation, groups) -> bool:
    if x.dim() != 4 or w.dim() != 4:
        return False
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    if dh != 1 or dw != 1:
        return False
    n, c, h, ww = x.shape
    oc, _, kh, kw = w.shape
    if x.dtype != torch.float32 or w.dtype != torch.float32:
        # wave64: the 7x7 / C<=8 stem has no usable vendor path at all in half
        # precision - f16 measured 372 ms at (1,3,224,224)->64 against 13.8 ms
        # on the f32 stem route - so it takes an f32 carrier instead of falling
        # back.
        # wave65: bf16 conv has no working vendor kernel in this wheel at all
        # (spine-opt aborts with exit 134 on 3x3 c64, 1x1 c256 and C=16, with
        # the fast gate on or off), so every bf16 shape the f32 rules accept
        # takes the same carrier.  f16 turned out to be 1.5-54x slower on the
        # vendor path than on the f32 tap kernel (3x3 46->15 ms, depthwise
        # 521->9.7), so the carrier is the default there too;
        # FLAGGEMS_SPACEMIT_CONV_F16_CARRIER=0 keeps the vendor path.
        if w.dtype != x.dtype or x.dtype not in (torch.float16, torch.bfloat16):
            return False
        if x.dtype == torch.float16:
            if os.environ.get("FLAGGEMS_SPACEMIT_CONV_F16_CARRIER", "1") != "1":
                return (groups == 1 and kh == 7 and kw == 7 and c <= 8
                        and os.environ.get("FLAGGEMS_SPACEMIT_CONV_STEM_F16", "1") != "0")
        elif os.environ.get("FLAGGEMS_SPACEMIT_CONV_BF16_CARRIER", "1") == "0":
            return False
    dwise = (groups == c == oc)
    if groups != 1 and not dwise:
        # wave57: grouped conv ran as g independent (c/g -> oc/g) problems, one
        # launch each: 1.5-2.8x against the vendor at 784 px/group and up, but
        # 0.72-0.76x at 196-256 px/group, where the g-fold dispatch dominates,
        # so those were gated back to the vendor.
        # wave58 adds a single-launch kernel (group folded into the leading axis,
        # weight prepacked per group) and uses it exactly where the per-group
        # path had nothing to offer: more than 8 groups, or under 512 px of
        # output per group.  Measured with it: g8/196 px 1.35x, g4/196 px 3.52x,
        # g16/3136 px 2.43x, g8/256 px 1.17x over the vendor; the per-group path
        # wins only inside its old band (g4/784 px 0.50x for the single launch).
        if c % groups or oc % groups:
            return False
        if (oc // groups) < 2:
            return False
        sh0, sw0 = (stride, stride) if isinstance(stride, int) else stride
        area = (h // sh0) * (ww // sw0)
        if groups > 32:
            return False
        macs = n * oc * h * ww * (kh * kw) * (c // groups)
        if groups <= 8 and area >= 512:
            return ((c // groups) >= 8) and macs >= (1 << 20) and ww >= 8
        # single-launch band: K may run thin (g32 at CG=2 measured 3.73x over
        # the vendor, g16 at CG=4 2.43x) and the launch is one rather than g, so
        # both the lane floor and the MAC floor come down
        cg = c // groups
        return (cg >= (2 if groups > 8 else 4)) and macs >= (1 << 18) and ww >= 8
    macs = n * oc * h * ww * (kh * kw) * (c // groups)
    # depthwise has a much lower launch floor: the vendor path pads 1 input
    # channel to 16 and copies the whole tensor, so even a 512x7x7 depthwise
    # costs 86 ms there (wave56 probe933 measured 14-35x on every shape)
    # wave65: half precision has no vendor fallback at all (bf16 aborts, f16 is
    # 3-60x slower there), so its floor is the launch cost only - 0.78 ms of tap
    # kernel at (1,16,20,20) against 2.9 ms of vendor f32 and 4.6 of vendor f16.
    if macs < ((1 << 12) if x.dtype != torch.float32
               else (1 << 14) if dwise else (1 << 20)):
        return False
    # wave66: the 3x3 patch-matrix route is a replacement for the fallback path
    # in the deep stages, where the width drops to 7 and the old 8-pixel floor
    # used to hand the whole layer to the vendor kernel: (1,512,7,7) costs
    # 161.3 ms there against 14.6 ms through the 9C GEMM.  Only the 3x3
    # power-of-two-C class dips below 8.
    deep3 = (kh == 3 and kw == 3 and c >= 32 and oc >= 128
             and (c & (c - 1)) == 0)
    if ww < (4 if (dwise or x.dtype != torch.float32 or deep3) else 8):
        return False
    return True


@triton.jit
def _conv1x1_kernel(X, Wm, Bs, Y, C, M, OC, sxn, syn,
    BLOCK_OC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr, NOMASK: tl.constexpr, SWAP: tl.constexpr,
):
    """wave60: 1x1 stride-1 conv as the GEMM it already is in NCHW.

    y[oc, m] = sum_c w[oc, c] * x[c, m], m = h*W + w over the (C, M) slab, so
    both operands are already in the layout a GEMM wants and the result is
    already NCHW.  No im2col, no transpose, no tap loop: one dot per K block
    instead of one K=32 dot per (kh, kw, c-block).
    """
    if SWAP:
        pid_oc = tl.program_id(0)
        pid_m = tl.program_id(1)
    else:
        pid_m = tl.program_id(0)
        pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)
    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)
    xb = X + pid_n * sxn
    for k0 in range(0, C, BLOCK_K):
        cs = k0 + tl.arange(0, BLOCK_K)
        if NOMASK:
            wv = tl.load(Wm + oc[:, None] * C + cs[None, :])
            xv = tl.load(xb + cs[:, None] * M + m[None, :])
        else:
            wv = tl.load(Wm + oc[:, None] * C + cs[None, :],
                         mask=(oc[:, None] < OC) & (cs[None, :] < C), other=0.0)
            xv = tl.load(xb + cs[:, None] * M + m[None, :],
                         mask=(cs[:, None] < C) & (m[None, :] < M), other=0.0)
        acc += tl.dot(wv, xv)
    if HAS_BIAS:
        if NOMASK:
            acc += tl.load(Bs + oc)[:, None]
        else:
            acc += tl.load(Bs + oc, mask=oc < OC, other=0.0)[:, None]
    yb = Y + pid_n * syn
    if NOMASK:
        tl.store(yb + oc[:, None] * M + m[None, :], acc.to(Y.dtype.element_ty))
    else:
        tl.store(yb + oc[:, None] * M + m[None, :], acc.to(Y.dtype.element_ty),
                 mask=(oc[:, None] < OC) & (m[None, :] < M))


@triton.jit
def _conv1x1_smt_kernel(A, B, Bias, Y, M, N, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    MICRO_M: tl.constexpr, MICRO_N: tl.constexpr, MICRO_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """wave61: the 1x1 GEMM on the smt descriptor path (mm's own recipe).

    flag_gems.mm wins 1.3x over the tl.load/tl.dot kernel for one reason: it
    never loops over K.  It descriptor-loads the whole A block (BLOCK_M x K)
    and B block (K x BLOCK_N) through triton.language.extra.smt, so the K
    dimension is swept inside one smt.dot on the TCM path
    (mm_spacemit: BLOCK 128x64, MICRO 8/32/32).  Transplanted to the 1x1 conv
    operands (A = weight (OC, C), B = x (C, M)) it measures 1.30 ms at
    (1,128,28,28) OC=512 against mm's 1.46 and the tl kernel's 1.99.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_ptr = tl.make_block_ptr(base=A, shape=[M, K], strides=[K, 1],
                              offsets=[pid_m * BLOCK_M, 0],
                              block_shape=[BLOCK_M, K], order=[1, 0])
    b_ptr = tl.make_block_ptr(base=B, shape=[K, N], strides=[N, 1],
                              offsets=[0, pid_n * BLOCK_N],
                              block_shape=[K, BLOCK_N], order=[1, 0])
    a = smt.view(smt.descriptor_load(a_ptr, (0, 0)), (0, 0),
                 (BLOCK_M, K), (MICRO_M, MICRO_K))
    b = smt.view(smt.descriptor_load(b_ptr, (0, 0)), (0, 0),
                 (K, BLOCK_N), (MICRO_K, MICRO_N))
    acc = smt.dot(a, b)
    acc = smt.view(acc, (0, 0), (BLOCK_M, BLOCK_N), (1, 1))
    if HAS_BIAS:
        oc = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        acc += tl.load(Bias + oc, mask=oc < M, other=0.0)[:, None]
    c_ptr = tl.make_block_ptr(base=Y, shape=[M, N], strides=[N, 1],
                              offsets=[pid_m * BLOCK_M, pid_n * BLOCK_N],
                              block_shape=[BLOCK_M, BLOCK_N], order=[1, 0])
    tl.store(c_ptr, acc.to(Y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _conv1x1_smt2_kernel(A1, A2, B, Bias, Y, M, N, K1: tl.constexpr,
    K2: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    MICRO_M: tl.constexpr, MICRO_N: tl.constexpr, MICRO_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """wave64: the same GEMM, but K arrives as two power-of-two slabs.

    smt.view refuses any block shape that is not a power of two, so a K of 147
    cannot be one dot; it can be 128 + 32.  Two descriptor loads and two dots
    against one register accumulator are exact (err 0.0 against torch at both
    128+64 and 512+64, probe1206), and the recipe matters: M/N stay runtime
    args, the second B base is pointer arithmetic (B + K1 * N) rather than a
    sliced tensor, and the first dot's view is assigned, not accumulated into a
    zeros tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a1_ptr = tl.make_block_ptr(base=A1, shape=[M, K1], strides=[K1, 1],
                               offsets=[pid_m * BLOCK_M, 0],
                               block_shape=[BLOCK_M, K1], order=[1, 0])
    b1_ptr = tl.make_block_ptr(base=B, shape=[K1 + K2, N], strides=[N, 1],
                               offsets=[0, pid_n * BLOCK_N],
                               block_shape=[K1, BLOCK_N], order=[1, 0])
    a1 = smt.view(smt.descriptor_load(a1_ptr, (0, 0)), (0, 0), (BLOCK_M, K1),
                  (MICRO_M, MICRO_K))
    b1 = smt.view(smt.descriptor_load(b1_ptr, (0, 0)), (0, 0), (K1, BLOCK_N),
                  (MICRO_K, MICRO_N))
    acc = smt.view(smt.dot(a1, b1), (0, 0), (BLOCK_M, BLOCK_N), (1, 1))
    a2_ptr = tl.make_block_ptr(base=A2, shape=[M, K2], strides=[K2, 1],
                               offsets=[pid_m * BLOCK_M, 0],
                               block_shape=[BLOCK_M, K2], order=[1, 0])
    b2_ptr = tl.make_block_ptr(base=B + K1 * N, shape=[K2, N], strides=[N, 1],
                               offsets=[0, pid_n * BLOCK_N],
                               block_shape=[K2, BLOCK_N], order=[1, 0])
    a2 = smt.view(smt.descriptor_load(a2_ptr, (0, 0)), (0, 0), (BLOCK_M, K2),
                  (MICRO_M, MICRO_K))
    b2 = smt.view(smt.descriptor_load(b2_ptr, (0, 0)), (0, 0), (K2, BLOCK_N),
                  (MICRO_K, MICRO_N))
    acc += smt.view(smt.dot(a2, b2), (0, 0), (BLOCK_M, BLOCK_N), (1, 1))
    if HAS_BIAS:
        oc = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        acc += tl.load(Bias + oc, mask=oc < M, other=0.0)[:, None]
    c_ptr = tl.make_block_ptr(base=Y, shape=[M, N], strides=[N, 1],
                              offsets=[pid_m * BLOCK_M, pid_n * BLOCK_N],
                              block_shape=[BLOCK_M, BLOCK_N], order=[1, 0])
    tl.store(c_ptr, acc.to(Y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _conv1x1_s2_kernel(X, Wm, Bs, Y, C, IH, IW, OH, OW, OC, sxn,
    BLOCK_OC: tl.constexpr, ROWS: tl.constexpr, BLOCK_OW: tl.constexpr,
    BLOCK_K: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    HAS_BIAS: tl.constexpr, NOMASK: tl.constexpr,
):
    """wave61: 1x1 conv with a stride as a strided GEMM.

    Same y[oc, m] = sum_c w[oc, c] x[c, m] as _conv1x1_kernel, but m now walks
    the output lattice: the tile is ROWS whole output rows, so the output block
    is still contiguous in NCHW and the input block is one affine 3-D load with
    pixel strides (SH, SW).  A stride-2 1x1 is the ResNet downsample, which used
    to fall back to the tap kernel (4.2 ms -> 2.6 ms at (1,64,112,112)).
    """
    pid_oc = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_n = tl.program_id(2)
    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    r = pid_r * ROWS + tl.arange(0, ROWS)
    ow = tl.arange(0, BLOCK_OW)
    acc = tl.zeros((BLOCK_OC, ROWS * BLOCK_OW), dtype=tl.float32)
    xb = X + pid_n * sxn
    for k0 in range(0, C, BLOCK_K):
        cs = k0 + tl.arange(0, BLOCK_K)
        if NOMASK:
            wv = tl.load(Wm + oc[:, None] * C + cs[None, :])
            xv = tl.load(xb + cs[:, None, None] * (IH * IW)
                         + (r[None, :, None] * SH) * IW
                         + (ow[None, None, :] * SW))
        else:
            wv = tl.load(Wm + oc[:, None] * C + cs[None, :],
                         mask=(oc[:, None] < OC) & (cs[None, :] < C), other=0.0)
            xv = tl.load(xb + cs[:, None, None] * (IH * IW)
                         + (r[None, :, None] * SH) * IW
                         + (ow[None, None, :] * SW),
                         mask=(cs[:, None, None] < C)
                         & (r[None, :, None] < OH)
                         & (ow[None, None, :] < OW), other=0.0)
        xv = tl.reshape(xv, (BLOCK_K, ROWS * BLOCK_OW))
        acc += tl.dot(wv, xv)
    if HAS_BIAS:
        if NOMASK:
            acc += tl.load(Bs + oc)[:, None]
        else:
            acc += tl.load(Bs + oc, mask=oc < OC, other=0.0)[:, None]
    acc3 = tl.reshape(acc, (BLOCK_OC, ROWS, BLOCK_OW))
    yp = (Y + pid_n * (OC * OH * OW) + oc[:, None, None] * (OH * OW)
          + r[None, :, None] * OW + ow[None, None, :])
    if NOMASK:
        tl.store(yp, acc3.to(Y.dtype.element_ty))
    else:
        tl.store(yp, acc3.to(Y.dtype.element_ty),
                 mask=(oc[:, None, None] < OC) & (r[None, :, None] < OH)
                 & (ow[None, None, :] < OW))


@triton.jit
def _im2col_kernel(X, B, C, IH, IW, OH, OW, M,
                   SH: tl.constexpr, SW: tl.constexpr,
                   KH: tl.constexpr, KW: tl.constexpr,
                   BLOCK_C: tl.constexpr, BLOCK_OW: tl.constexpr):
    """wave62: build the (KH*KW*C, M) patch matrix for a small-C conv.

    Only worth it when C is tiny: at C=3 the tap kernel pads to 16 and runs
    49 dots whose 32 lanes carry 3 real values, while the buffer is 7 MB and
    the K=147 GEMM is the same arithmetic the 43-53 GFLOPS 1x1 route gets.
    """
    pid_ow = tl.program_id(0)
    oh = tl.program_id(1)
    pid_c = tl.program_id(2)
    ow0 = pid_ow * BLOCK_OW
    lo = tl.arange(0, BLOCK_OW)
    cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m = oh * OW + ow0 + lo
    for kh in tl.static_range(KH):
        ih = oh * SH + kh
        for kw in tl.static_range(KW):
            iw = (ow0 + lo) * SW + kw
            xp = X + cs[:, None] * (IH * IW) + ih * IW + iw[None, :]
            mm = (cs[:, None] < C) & (lo[None, :] < OW - ow0)
            xv = tl.load(xp, mask=mm, other=0.0)
            k = (kh * KW + kw) * C + cs
            tl.store(B + k[:, None] * M + m[None, :], xv, mask=mm)


@triton.jit
def _conv2d_direct_kernel(X, Wt, Bs, Y,
    C, IH, IW, OH, OW, OC,
    sxn, sxc, sxh, sxw,
    swkk, swkw, swc, swoc,
    syn, syoc, syoh, syow,
    SH: tl.constexpr, SW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OW: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_C: tl.constexpr,
    ROWS: tl.constexpr, NOMASK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_oh = tl.cdiv(OH, ROWS)
    n_ow = tl.cdiv(OW, BLOCK_OW)
    n_oc = tl.cdiv(OC, BLOCK_OC)
    pid_ow = pid % n_ow
    pid_oc = (pid // n_ow) % n_oc
    pid_nh = pid // (n_ow * n_oc)
    pid_n = pid_nh // n_oh
    oh0 = (pid_nh % n_oh) * ROWS
    ow0 = pid_ow * BLOCK_OW
    lo = tl.arange(0, BLOCK_OW)
    ro = tl.arange(0, ROWS)
    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    acc = tl.zeros((BLOCK_OC, ROWS * BLOCK_OW), dtype=tl.float32)
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            iw0 = ow0 * SW + kw
            for c0 in range(0, C, BLOCK_C):
                cs = c0 + tl.arange(0, BLOCK_C)
                xp = (X + pid_n * sxn
                      + ((oh0 + ro)[None, :, None] * SH + kh) * sxh
                      + (iw0 + lo[None, None, :] * SW) * sxw
                      + cs[:, None, None] * sxc)
                if NOMASK:
                    xv = tl.load(xp)
                else:
                    xv = tl.load(xp, mask=(cs[:, None, None] < C)
                                 & ((oh0 + ro)[None, :, None] < OH)
                                 & (lo[None, None, :] < OW - ow0), other=0.0)
                xv = tl.reshape(xv, (BLOCK_C, ROWS * BLOCK_OW))
                wpt = Wt + kh * swkk + kw * swkw + oc[:, None] * swoc + cs[None, :] * swc
                if NOMASK:
                    wv = tl.load(wpt)
                else:
                    wv = tl.load(wpt, mask=(oc[:, None] < OC) & (cs[None, :] < C), other=0.0)
                acc += tl.dot(wv, xv)
    if HAS_BIAS:
        if NOMASK:
            bias = tl.load(Bs + oc)
        else:
            bias = tl.load(Bs + oc, mask=oc < OC, other=0.0)
        acc += bias[:, None]
    yp = (Y + pid_n * syn
          + (oh0 + ro)[None, :, None] * syoh
          + (ow0 + lo)[None, None, :] * syow
          + oc[:, None, None] * syoc)
    acc3 = tl.reshape(acc, (BLOCK_OC, ROWS, BLOCK_OW))
    if NOMASK:
        tl.store(yp, acc3.to(Y.dtype.element_ty))
    else:
        tl.store(yp, acc3.to(Y.dtype.element_ty),
                 mask=(oc[:, None, None] < OC) & ((oh0 + ro)[None, :, None] < OH)
                 & ((ow0 + lo)[None, None, :] < OW))


@triton.jit
def _conv2d_group_kernel(X, Wt, Bs, Y,
    C, IH, IW, OH, OW, OC,
    sxn, sxc, sxh, sxw,
    swg, swkk, swkw, swc, swoc,
    syn, syoc, syoh, syow,
    SH: tl.constexpr, SW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OW: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_C: tl.constexpr,
    ROWS: tl.constexpr, NOMASK: tl.constexpr, NG: tl.constexpr,
):
    """wave58: one launch for 1 < groups < C.

    The group folds into the leading axis (x is viewed as (N*G, CG, H, W), y as
    (N*G, OCG, OH, OW)), so the image index already carries the group.  The only
    thing the per-group loop used to give the kernel was the weight base, and
    that is the one extra term here: swg is the stride of the prepacked
    (G, KH, KW, OCG, CG) weight tensor's group axis.  Bias is indexed per group.
    """
    pid = tl.program_id(0)
    n_oh = tl.cdiv(OH, ROWS)
    n_ow = tl.cdiv(OW, BLOCK_OW)
    n_oc = tl.cdiv(OC, BLOCK_OC)
    pid_ow = pid % n_ow
    pid_oc = (pid // n_ow) % n_oc
    pid_nh = pid // (n_ow * n_oc)
    pid_n = pid_nh // n_oh
    gid = pid_n % NG
    oh0 = (pid_nh % n_oh) * ROWS
    ow0 = pid_ow * BLOCK_OW
    lo = tl.arange(0, BLOCK_OW)
    ro = tl.arange(0, ROWS)
    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    wb = Wt + gid * swg
    acc = tl.zeros((BLOCK_OC, ROWS * BLOCK_OW), dtype=tl.float32)
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            iw0 = ow0 * SW + kw
            for c0 in range(0, C, BLOCK_C):
                cs = c0 + tl.arange(0, BLOCK_C)
                xp = (X + pid_n * sxn
                      + ((oh0 + ro)[None, :, None] * SH + kh) * sxh
                      + (iw0 + lo[None, None, :] * SW) * sxw
                      + cs[:, None, None] * sxc)
                if NOMASK:
                    xv = tl.load(xp)
                else:
                    xv = tl.load(xp, mask=(cs[:, None, None] < C)
                                 & ((oh0 + ro)[None, :, None] < OH)
                                 & (lo[None, None, :] < OW - ow0), other=0.0)
                xv = tl.reshape(xv, (BLOCK_C, ROWS * BLOCK_OW))
                wpt = wb + kh * swkk + kw * swkw + oc[:, None] * swoc + cs[None, :] * swc
                if NOMASK:
                    wv = tl.load(wpt)
                else:
                    wv = tl.load(wpt, mask=(oc[:, None] < OC) & (cs[None, :] < C), other=0.0)
                acc += tl.dot(wv, xv)
    if HAS_BIAS:
        if NOMASK:
            bias = tl.load(Bs + gid * OC + oc)
        else:
            bias = tl.load(Bs + gid * OC + oc, mask=oc < OC, other=0.0)
        acc += bias[:, None]
    yp = (Y + pid_n * syn
          + (oh0 + ro)[None, :, None] * syoh
          + (ow0 + lo)[None, None, :] * syow
          + oc[:, None, None] * syoc)
    acc3 = tl.reshape(acc, (BLOCK_OC, ROWS, BLOCK_OW))
    if NOMASK:
        tl.store(yp, acc3.to(Y.dtype.element_ty))
    else:
        tl.store(yp, acc3.to(Y.dtype.element_ty),
                 mask=(oc[:, None, None] < OC) & ((oh0 + ro)[None, :, None] < OH)
                 & ((ow0 + lo)[None, None, :] < OW))


@triton.jit
def _depthwise_kernel(X, Wt, Bs, Y,
    C, IH, IW, OH, OW,
    sxn, sxc, sxh, sxw,
    swc,
    syn, syoc, syoh, syow,
    SH: tl.constexpr, SW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OW: tl.constexpr, BLOCK_C: tl.constexpr,
    NOMASK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_ow = tl.cdiv(OW, BLOCK_OW)
    n_c = tl.cdiv(C, BLOCK_C)
    pid_ow = pid % n_ow
    pid_c = (pid // n_ow) % n_c
    pid_nh = pid // (n_ow * n_c)
    pid_n = pid_nh // OH
    oh = pid_nh % OH
    ow0 = pid_ow * BLOCK_OW
    lo = tl.arange(0, BLOCK_OW)
    cc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    acc = tl.zeros((BLOCK_C, BLOCK_OW), dtype=tl.float32)
    for kh in tl.static_range(KH):
        ih = oh * SH + kh
        for kw in tl.static_range(KW):
            iw = ow0 * SW + kw
            xp = (X + pid_n * sxn + ih * sxh + cc[:, None] * sxc
                  + (iw + lo[None, :] * SW) * sxw)
            if NOMASK:
                xv = tl.load(xp)
                wv = tl.load(Wt + cc * swc + kh * KW + kw)
            else:
                xv = tl.load(xp, mask=(cc[:, None] < C) & (lo[None, :] < OW - ow0), other=0.0)
                wv = tl.load(Wt + cc * swc + kh * KW + kw, mask=cc < C, other=0.0)
            acc += xv * wv[:, None]
    if HAS_BIAS:
        if NOMASK:
            acc += tl.load(Bs + cc)[:, None]
        else:
            acc += tl.load(Bs + cc, mask=cc < C, other=0.0)[:, None]
    yp = (Y + pid_n * syn + oh * syoh + cc[:, None] * syoc
          + (ow0 + lo[None, :]) * syow)
    if NOMASK:
        tl.store(yp, acc.to(Y.dtype.element_ty))
    else:
        tl.store(yp, acc.to(Y.dtype.element_ty),
                 mask=(cc[:, None] < C) & ((ow0 + lo)[None, :] < OW))


def _pick_tiles(n, oh, ow, oc, stride=(1, 1)):
    bow = max(16, min(64, triton.next_power_of_2(ow)))
    # wave57 probe950: BLOCK_OC=128 is a free 1.08-1.15x whenever OC >= 128 -
    # (1,128,28,28) 13.39 -> 11.77, (1,256,14,14) 13.87 -> 12.78 - and it stays
    # bit-exact because it only changes which program owns an output channel.
    # BLOCK_C stays 32: 64/128 both rewrite the K accumulation order (not
    # bit-exact) and are 1.1-1.5x slower.
    boc = 128 if oc >= 128 else 64 if oc >= 64 else 32 if oc >= 32 else 16
    # wave57 probe952: BLOCK_OW=16 with BLOCK_OC=32 aborts spine-opt inside
    # MMT4D (the same assertion wave56 saw at OW=14), and a grouped conv with
    # 32 channels per group lands there.  A 16-wide tile always takes the
    # 64-row output block instead.
    if bow == 16 and boc < 64:
        boc = 64
    # wave56 sweep (7 shapes x 16 tile combinations): every winner sits at
    # 8-28 programs, i.e. 1-3.5 waves of the 8 cores, so the rule grows ROWS
    # while the grid stays above one wave.  The two exceptions the sweep found:
    #   - stride 2 with a tile wider than 128 columns loses its row-to-row
    #     input reuse ((1,64,112,112) s2: r2 13.49 vs r4 16.00), so it caps at
    #     ROWS=2 there;
    #   - BLOCK_OW=16 only pays off when BLOCK_OC=64 (OW=14: 28.0 ms vs 45.5
    #     at w32); with BLOCK_OC=32 the 32x16 dot trips a spine-opt MMT4D
    #     assertion.
    max_rows = 8
    rows = 0
    for r in (8, 4, 2):
        if r > max_rows:
            continue
        # wave57 probe951: the dot's N is ROWS*BLOCK_OW, and a 16-wide tile at
        # ROWS=1 (N=16, half an f32 vector register) trips the same MMT4D
        # assertion from the other side - (1,256,14,14) g8 and (1,32,16,16) g2
        # both aborted until the picker stopped returning N < 32.
        if r * bow < 32:
            continue
        grid = n * triton.cdiv(oh, r) * triton.cdiv(ow, bow) * triton.cdiv(oc, boc)
        if grid < 8:
            continue
        if max(stride) > 1 and r * bow > 128:
            continue
        rows = r
        break
    if not rows:
        rows = max(1, 32 // bow)
    return bow, boc, rows


def _pick_1x1_tiles(C, M, OC):
    """wave60 probe976-980: BLOCK_K=32 wins at C=64/128/256 (43-53 GFLOPS
    against 35-40 at K=64), a 64x64 output tile wins whenever OC > 64, and at
    OC <= 64 the narrower 32-lane weight tile plus a 128-wide pixel tile is the
    only thing that fills the grid (1,64,112,112) OC=64: 3.65 ms vs 6.58 at
    64x64.  SWAP puts the oc axis on program_id(0) so the x tile a program just
    read is reused by the next oc tile: 2.41 -> 2.24 at (1,64,56,56) OC=256 and
    2.36 -> 1.93 at (1,128,28,28) OC=512.  With one oc tile it only costs
    ((1,64,112,112) 6.58 -> 9.66), hence the OC > 64 condition.
    """
    bk = 32 if C % 32 == 0 else (16 if C % 16 == 0 else triton.next_power_of_2(C))
    if OC <= 64:
        boc = 32 if OC >= 32 else triton.next_power_of_2(OC)
        bm = 128 if M >= 1024 else 64
        swap = False
    else:
        boc = 64
        bm = 64
        swap = True
    return boc, bm, bk, swap


def _conv1x1_gemm(input, weight, bias, OH, OW):
    N, C, H, W = input.shape
    OC = weight.shape[0]
    M = OH * OW
    x = input.contiguous()
    x2 = x.view(N, C, M)
    w2 = weight.view(OC, C).contiguous()
    y = torch.empty((N, OC, M), dtype=x.dtype, device=x.device)
    # wave61: mm's descriptor path beats our tl kernel at every 1x1 shape
    # measured so far (1.15-1.5x) because the whole K is swept inside one
    # smt.dot instead of four tl.dot calls.  Needs C 32-aligned (so K is a
    # multiple of MICRO_K) and a pixel count that still fills 8 cores.
    # FLAGGEMS_SPACEMIT_CONV_SMT=0 pins the tl kernel for A/B.
    if (smt is not None and C % 32 == 0 and C <= 256 and OC >= 128 and M >= 64
            and os.environ.get("FLAGGEMS_SPACEMIT_CONV_SMT", "1") != "0"):
        # probe987: BLOCK_M=128 only where it was measured to win - short K,
        # few pixels, many output channels ((1,128,28,28) OC=512: 1.30 vs
        # 1.99 tl).  At (1,64,56,56) OC=256 it loses 1.4x (3.31 vs 2.35) and at
        # K=256 the 128-wide A block no longer fits TCM at all (7.30 vs 4.67),
        # so everything else runs BLOCK_M=64.
        bm = 128 if (C >= 96 and M <= 1024 and OC >= 256) else 64
        bn = 64
        grid = (triton.cdiv(OC, bm), triton.cdiv(M, bn))
        for n in range(N):
            _conv1x1_smt_kernel[grid](
                w2, x2[n], bias if bias is not None else w2, y[n],
                OC, M, C, BLOCK_M=bm, BLOCK_N=bn,
                MICRO_M=8, MICRO_N=64, MICRO_K=32,
                HAS_BIAS=(bias is not None))
        return y.view(N, OC, OH, OW)
    boc, bm, bk, swap = _pick_1x1_tiles(C, M, OC)
    nomask = (C % bk == 0) and (M % bm == 0) and (OC % boc == 0)
    nm, no = triton.cdiv(M, bm), triton.cdiv(OC, boc)
    grid = (no, nm, N) if swap else (nm, no, N)
    _conv1x1_kernel[grid](
        x2, w2, bias if bias is not None else x2, y,
        C, M, OC, x2.stride(0), y.stride(0),
        BLOCK_OC=boc, BLOCK_M=bm, BLOCK_K=bk,
        HAS_BIAS=(bias is not None), NOMASK=nomask, SWAP=swap,
    )
    return y.view(N, OC, OH, OW)


def _pick_1x1_s2_tiles(C, OH, OW, OC, N):
    """wave61 probe983/984: rows follow the output width, not the accumulator.

    BLOCK_K=32 at every C again (K=64 loses 10-30% and, at ROWS>=4, currently
    miscompiles: err 3.7-7.8 on three shapes).  ROWS buys row-to-row reuse, but
    it turns bad at both ends of the width range: at OW=56 a 64x512 accumulator
    spills (r4 2.62 vs r8 3.63), and at OW=7 the 8-wide input runs stop
    coalescing (r4 0.51 vs r8 1.00 at (1,256,14,14)).  So rows=4 for bow>=64
    and bow<=8, rows=8 for bow=32/16; rows then halves until the grid reaches
    one wave (8 programs).  BLOCK_OC=128 only when the dot N collapsed to 32
    ((1,256,14,14) OC=512: 0.43 vs 0.51 at 64).
    """
    bk = 32 if C % 32 == 0 else (16 if C % 16 == 0 else triton.next_power_of_2(C))
    bow = triton.next_power_of_2(OW)
    boc = 64 if OC >= 64 else triton.next_power_of_2(OC)
    rows = 4 if (bow >= 64 or bow <= 8) else 8
    if rows * bow <= 32 and OC >= 128:
        boc = 128
    while (rows > 1 and (triton.cdiv(OC, boc) * triton.cdiv(OH, rows) * N) < 8
           and (rows // 2) * bow >= 32):
        # never halve below a 32-wide dot: a 16-wide tile at ROWS=1 trips the
        # spine-opt MMT4D assertion (wave57/wave60 landmine)
        rows //= 2
    return boc, rows, bow, bk


def _conv1x1_strided(input, weight, bias, OH, OW, SH, SW):
    """1x1 conv with a stride, as a strided GEMM over whole output rows."""
    N, C, H, W = input.shape
    OC = weight.shape[0]
    x = input.contiguous()
    w2 = weight.view(OC, C).contiguous()
    y = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)
    boc, rows, bow, bk = _pick_1x1_s2_tiles(C, OH, OW, OC, N)
    nomask = (C % bk == 0) and (OW == bow) and (OH % rows == 0) and (OC % boc == 0)
    grid = (triton.cdiv(OC, boc), triton.cdiv(OH, rows), N)
    _conv1x1_s2_kernel[grid](
        x, w2, bias if bias is not None else x, y,
        C, H, W, OH, OW, OC, x.stride(0),
        BLOCK_OC=boc, ROWS=rows, BLOCK_OW=bow, BLOCK_K=bk, SH=SH, SW=SW,
        HAS_BIAS=(bias is not None), NOMASK=nomask,
    )
    return y


def _stem_buf(kpad, M, K, dtype, device):
    """wave64: one buffer per (kpad, M, K, dtype) with the pad rows already zero.

    The build writes every row k < K on every call, so the zeroed pad tail only
    has to exist once.  This removes the per-forward buffer allocation, the
    per-forward memset of the tail (7.1 MB at kpad=256) and the page-fault tax
    that comes with a fresh 16.8 MB allocation each call.
    """
    key = (kpad, M, K, dtype, str(device))
    buf = _STEM_BUF.get(key)
    if buf is None:
        if len(_STEM_BUF) >= 6:            # FIFO, one eviction not a flush
            _STEM_BUF.pop(next(iter(_STEM_BUF)))
        buf = torch.zeros((kpad, M), dtype=dtype, device=device)
        _STEM_BUF[key] = buf
    return buf


def _conv_stem_im2col(input, weight, bias, OH, OW, SH, SW, pad_h, pad_w,
                      out_dtype=None):
    """wave62: a 7x7 stride-2 first layer with C<=8 through a buffered im2col.

    The tap kernel is the wrong shape for this layer: it pads C=3 to 16 and
    runs 49 dots whose 32 lanes carry 3 real values, which is why the stem
    costs 143 ms.  The patch matrix is 147 x OH*OW (7.4 MB at 112x112) and the
    GEMM on it is the ordinary 1x1 dot, so build + GEMM measures 17.0 ms at
    (1,3,224,224)->64 on the smt kernel (8.4x) and 19.0 ms on the tl kernel
    (7.5x) against the tap path.  K is padded to a power of two (147 -> 256 for
    smt, 160 for tl) so the GEMM carries no mask; only the tail rows are
    zeroed per forward.
    """
    N, C, H, W = input.shape
    OC = weight.shape[0]
    KH, KW = weight.shape[2], weight.shape[3]
    xp = F.pad(input, (pad_w, pad_w, pad_h, pad_h)) if (pad_h or pad_w) else input
    IH, IW = xp.shape[2], xp.shape[3]
    M = OH * OW
    K = C * KH * KW
    # wave64: K is 147 for the 7x7 stem, and smt.view rejects any block shape
    # that is not a power of two, so the old code padded to 256 and spent 43%
    # of both the buffer read and the dot on zeros.  160 = 128 + 32 is the
    # smallest legal K and measures 6.95 ms of GEMM against 9.75 at 256.
    # FLAGGEMS_SPACEMIT_CONV_STEM_K256=1 pins the old padded path for A/B.
    if os.environ.get("FLAGGEMS_SPACEMIT_CONV_STEM_K256", "0") == "1":
        kpad = 256 if (smt is not None and K <= 256) else max(
            32, triton.next_power_of_2(K))
        ksplit = 0
    elif smt is not None and K <= 160 and K > 128:
        kpad, ksplit = 160, 128
    elif smt is not None and K <= 192 and K > 128:
        kpad, ksplit = 192, 128
    else:
        kpad, ksplit = max(32, triton.next_power_of_2(K)), 0
    wp = weight.permute(0, 2, 3, 1).reshape(OC, K)
    if ksplit:
        w2a = torch.zeros((OC, ksplit), dtype=weight.dtype, device=weight.device)
        w2a[:, :ksplit] = wp[:, :ksplit]
        w2b = torch.zeros((OC, kpad - ksplit), dtype=weight.dtype,
                          device=weight.device)
        w2b[:, :K - ksplit] = wp[:, ksplit:]
        w2 = None
    else:
        w2a = w2b = None
        w2 = torch.zeros((OC, kpad), dtype=weight.dtype, device=weight.device)
        w2[:, :K] = wp
    # wave65: the half-precision carrier computes in f32 but stores its result
    # in the caller's dtype; writing 12.8 MB of f32 and copying it to half cost
    # ~1.5 ms of the stem's 15.7, and the GEMM already casts on store.
    y = torch.empty((N, OC, OH, OW), dtype=out_dtype or input.dtype,
                    device=input.device)
    bc = 4 if C <= 4 else (8 if C <= 8 else 16)
    bm = 32 if M >= 4096 else 64
    # probe1011: the GEMM is bound by the 12.8 MB buffer read, so the widest
    # pixel tile that divides M wins: 32x256 reads 9.74 ms at (1,3,224,224)
    # against 11.33 at 32x128 and 16.2 at 64x128.
    # wave63 probe1116: widening further keeps paying wherever it divides M,
    # because every extra column block re-loads the A tile (64x256 = 64 KB):
    # M=16384 1024 -> 9.99 ms against 11.65 at 256 (1.17x), M=9216 512 -> 6.05
    # against 6.40 (1.06x); at M=12544, which 512 does not divide, 1024 buys
    # nothing (8.65 vs 8.70).  So take the widest tile that divides M.
    bn = (1024 if M % 1024 == 0 else 512 if M % 512 == 0 else
          256 if M % 256 == 0 else 128 if M % 128 == 0 else 64)
    # the smt descriptor path has no element mask: it can only run when both
    # grid axes land exactly on the matrix edges
    use_smt = (smt is not None and (ksplit or kpad == 256) and OC % bm == 0
               and M % bn == 0)
    for n in range(N):
        buf = _stem_buf(kpad, M, K, input.dtype, input.device)
        grid = (triton.cdiv(OW, 64), OH, triton.cdiv(C, bc))
        _im2col_kernel[grid](xp[n], buf, C, IH, IW, OH, OW, M,
                             SH=SH, SW=SW, KH=KH, KW=KW,
                             BLOCK_C=bc, BLOCK_OW=64)
        if use_smt and ksplit:
            g = (triton.cdiv(OC, bm), triton.cdiv(M, bn))
            _conv1x1_smt2_kernel[g](w2a, w2b, buf,
                                    bias if bias is not None else w2a, y[n],
                                    OC, M, ksplit, kpad - ksplit,
                                    BLOCK_M=bm, BLOCK_N=bn,
                                    MICRO_M=8, MICRO_N=64, MICRO_K=32,
                                    HAS_BIAS=(bias is not None))
        elif use_smt:
            g = (triton.cdiv(OC, bm), triton.cdiv(M, bn))
            _conv1x1_smt_kernel[g](w2, buf, bias if bias is not None else w2, y[n],
                                   OC, M, kpad, BLOCK_M=bm, BLOCK_N=bn,
                                   MICRO_M=8, MICRO_N=64, MICRO_K=32,
                                   HAS_BIAS=(bias is not None))
        else:
            if w2 is None:
                w2 = torch.zeros((OC, kpad), dtype=weight.dtype,
                                 device=weight.device)
                w2[:, :K] = wp
            bmk = 128 if M % 128 == 0 else 64
            g = (triton.cdiv(OC, 64), triton.cdiv(M, bmk), 1)
            _conv1x1_kernel[g](buf, w2, bias if bias is not None else buf, y[n],
                               kpad, M, OC, 0, 0,
                               BLOCK_OC=64, BLOCK_M=bmk, BLOCK_K=32,
                               HAS_BIAS=(bias is not None),
                               NOMASK=(M % bmk == 0 and OC % 64 == 0
                                       and kpad % 32 == 0),
                               SWAP=True)
    return y


# wave66: 3x3 stride-1 patch matrices, keyed (KK, Mpad, dtype, device).  The
# build rewrites every real column of every row on every call, so the pad tail
# only has to be zero once.
_IM2COL3_BUF = {}


def _im2col3_buf(kk, mpad, dtype, device):
    key = (kk, mpad, dtype, str(device))
    buf = _IM2COL3_BUF.get(key)
    if buf is None:
        if len(_IM2COL3_BUF) >= 6:
            _IM2COL3_BUF.pop(next(iter(_IM2COL3_BUF)))
        buf = torch.zeros((kk, mpad), dtype=dtype, device=device)
        _IM2COL3_BUF[key] = buf
    return buf


def _conv3x3_im2col(input, weight, bias, OH, OW, pad_h, pad_w, out_dtype=None):
    """wave66: 3x3 stride-1 conv as a K = 9C patch-matrix GEMM.

    The tap kernel splits C into 32-lane dots and walks the input nine times;
    that collapses where M is small, because the program count shrinks with
    M while the per-program work stays at nine taps.  (1,512,7,7) is the
    extreme: 161.3 ms on the fallback path (the tap kernel never sees a
    7-pixel width) against 2.9 build + 11.6 GEMM here.
    Same-process A/B against the tap kernel (probe1330): 11.06x at
    (1,512,7,7), 1.80x at (1,256,14,14), 1.42x at (1,128,28,28), 1.03x at
    (1,64,56,56) - so the gate takes M <= 1024 and OC >= 128, the band that
    never measured below 1.4x.

    K = 9C splits into two power-of-two slabs only when C is a power of two
    (9 * 2^k = 2^(k+3) + 2^k), which is what the gate checks.
    """
    N, C, H, W = input.shape
    OC = weight.shape[0]
    x = input.contiguous()
    xp = F.pad(x, (pad_w, pad_w, pad_h, pad_h)) if (pad_h or pad_w) else x
    IH, IW = xp.shape[2], xp.shape[3]
    M = OH * OW
    KK = 9 * C
    k1 = 1 << (KK.bit_length() - 1)
    if k1 == KK:
        k1 = KK // 2
    k2 = KK - k1
    wp = weight.permute(0, 2, 3, 1).reshape(OC, KK)
    a1 = wp[:, :k1].contiguous()
    a2 = wp[:, k1:].contiguous()
    # probe1330: BLOCK_N=64 wins where 64 divides M or M is tiny, 128 wins in
    # the 196-384 band (196: 8.53 at 128 against 9.44 at 64); 256 never won.
    bn = 64 if (M <= 64 or M % 64 == 0) else (128 if M <= 384 else 64)
    mpad = ((M + bn - 1) // bn) * bn
    dt = out_dtype or input.dtype
    y = torch.empty((N, OC, OH, OW), dtype=dt, device=input.device)
    bow = min(64, triton.next_power_of_2(OW))
    for n in range(N):
        buf = _im2col3_buf(KK, mpad, x.dtype, x.device)
        grid = (triton.cdiv(OW, bow), OH, triton.cdiv(C, 32))
        _im2col_kernel[grid](xp[n], buf, C, IH, IW, OH, OW, mpad,
                             SH=1, SW=1, KH=3, KW=3,
                             BLOCK_C=32, BLOCK_OW=bow)
        exact = (mpad == M)
        yp = y[n].view(OC, M) if exact else torch.empty(
            (OC, mpad), dtype=dt, device=input.device)
        g = (triton.cdiv(OC, 32), triton.cdiv(mpad, bn))
        _conv1x1_smt2_kernel[g](a1, a2, buf, bias if bias is not None else a1,
                                yp, OC, mpad, k1, k2,
                                BLOCK_M=32, BLOCK_N=bn,
                                MICRO_M=8, MICRO_N=64, MICRO_K=32,
                                HAS_BIAS=(bias is not None))
        if not exact:
            y[n].view(OC, M).copy_(yp[:, :M])
    return y


def _depthwise_direct(input, weight, bias, pad_h, pad_w, OH, OW, SH, SW, C, KH, KW,
                      BLOCK_OW=None, BLOCK_C=None):
    N, _, H, W = input.shape
    x = input.contiguous()
    if pad_h or pad_w:
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
        H, W = x.shape[2], x.shape[3]
    wc = weight.reshape(C, KH * KW).contiguous()
    y = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
    # depthwise tile scan (probe940/941): BLOCK_C=16 beats 32/64 on every
    # stage - (1,64,56,56) 6.9 vs 15.1 at c64, (1,32,112,112) 15.3 vs 19.9 at
    # c32 - because a 64x64 accumulator is 16 KB and spills.  BLOCK_OW=128
    # only pays at OW=112 (12.6 vs 15.3); at 56 and 28 it costs 1.5x.
    block_c = BLOCK_C or min(16, triton.next_power_of_2(C))
    block_ow = BLOCK_OW or min(128, triton.next_power_of_2(OW))
    nomask = (C % block_c == 0) and (OW % block_ow == 0)
    grid = (N * OH * triton.cdiv(OW, block_ow) * triton.cdiv(C, block_c),)
    _depthwise_kernel[grid](
        x, wc, bias if bias is not None else x, y,
        C, H, W, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        wc.stride(0),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        SH=SH, SW=SW, HAS_BIAS=(bias is not None), KH=KH, KW=KW,
        BLOCK_OW=block_ow, BLOCK_C=block_c, NOMASK=nomask,
    )
    return y


def _conv_launch(x, weight, bias, y, OH, OW, SH, SW, BLOCK_OW, BLOCK_OC, ROWS, BLOCK_C):
    """One conv launch over a contiguous (N, C, H, W) slab."""
    N, C, H, W = x.shape
    OC, CG, KH, KW = weight.shape
    wp = weight.permute(2, 3, 0, 1).contiguous()   # (KH, KW, OC, CG)
    bow_pick, boc_pick, rows_pick = _pick_tiles(N, OH, OW, OC, stride=(SH, SW))
    BLOCK_OW = BLOCK_OW or bow_pick
    BLOCK_OC = BLOCK_OC or boc_pick
    ROWS = ROWS or rows_pick
    BLOCK_C = BLOCK_C or 32
    nomask = ((ROWS == 1) and (BLOCK_OW >= 32)
              and (CG % BLOCK_C == 0) and (OW % BLOCK_OW == 0) and (OC % BLOCK_OC == 0))
    grid = (N * triton.cdiv(OH, ROWS) * triton.cdiv(OW, BLOCK_OW) * triton.cdiv(OC, BLOCK_OC),)
    _conv2d_direct_kernel[grid](
        x, wp, bias if bias is not None else x, y,
        C, H, W, OH, OW, OC,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        wp.stride(0), wp.stride(1), wp.stride(3), wp.stride(2),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        SH=SH, SW=SW, HAS_BIAS=(bias is not None), KH=KH, KW=KW,
        BLOCK_OW=BLOCK_OW, BLOCK_OC=BLOCK_OC, BLOCK_C=BLOCK_C,
        ROWS=ROWS, NOMASK=nomask,
    )
    return y


def _conv_group_launch(x, weight, bias, y, OH, OW, SH, SW, groups,
                       BLOCK_OW=None, BLOCK_OC=None, ROWS=None, BLOCK_C=None):
    """One launch over all groups; x/y are (N, C, H, W)/(N, OC, OH, OW)."""
    N, C, H, W = x.shape
    OC, CG, KH, KW = weight.shape
    OCG = OC // groups
    wp = weight.view(groups, OCG, CG, KH, KW).permute(0, 3, 4, 1, 2).contiguous()
    xv = x.view(N * groups, CG, H, W)
    yv = y.view(N * groups, OCG, OH, OW)
    bow_pick, boc_pick, rows_pick = _pick_tiles(N * groups, OH, OW, OCG, stride=(SH, SW))
    BLOCK_OW = BLOCK_OW or bow_pick
    BLOCK_OC = BLOCK_OC or boc_pick
    ROWS = ROWS or rows_pick
    BLOCK_C = BLOCK_C or 32
    nomask = ((ROWS == 1) and (BLOCK_OW >= 32)
              and (CG % BLOCK_C == 0) and (OW % BLOCK_OW == 0) and (OCG % BLOCK_OC == 0))
    grid = (N * groups * triton.cdiv(OH, ROWS) * triton.cdiv(OW, BLOCK_OW)
            * triton.cdiv(OCG, BLOCK_OC),)
    _conv2d_group_kernel[grid](
        xv, wp, bias if bias is not None else xv, yv,
        CG, H, W, OH, OW, OCG,
        xv.stride(0), xv.stride(1), xv.stride(2), xv.stride(3),
        wp.stride(0), wp.stride(1), wp.stride(2), wp.stride(4), wp.stride(3),
        yv.stride(0), yv.stride(1), yv.stride(2), yv.stride(3),
        SH=SH, SW=SW, HAS_BIAS=(bias is not None), KH=KH, KW=KW,
        BLOCK_OW=BLOCK_OW, BLOCK_OC=BLOCK_OC, BLOCK_C=BLOCK_C,
        ROWS=ROWS, NOMASK=nomask, NG=groups,
    )
    return y


def conv2d_direct(input, weight, bias, pad_h, pad_w, OH, OW, SH=1, SW=1, groups=1,
                  BLOCK_OW=None, BLOCK_OC=None, ROWS=None, BLOCK_C=None,
                  _out_dtype=None):
    N, C, H, W = input.shape
    OC, CG, KH, KW = weight.shape
    if input.dtype != torch.float32:
        # wave65 carrier: half precision never enters the f32 kernels.  For
        # bf16 this is the only working route in the wheel (the vendor conv
        # aborts in spine-opt), and for the f16 stem it is 25x faster than the
        # vendor path.  Only the two boundaries convert: input 0.8 MB and
        # output 0.8 MB at (1,64,56,56), weight a few KB.
        if os.environ.get("FLAGGEMS_SPACEMIT_CONV_CARRIER", "1") != "0":
            y = conv2d_direct(input.float(), weight.float(),
                              bias.float() if bias is not None else None,
                              pad_h, pad_w, OH, OW, SH, SW, groups,
                              BLOCK_OW, BLOCK_OC, ROWS, BLOCK_C,
                              _out_dtype=(None if os.environ.get(
                                  "FLAGGEMS_SPACEMIT_CONV_STEM_OUT32", "0") == "1"
                                  else input.dtype))
            return y if y.dtype == input.dtype else y.to(input.dtype)
    if groups == C == OC:
        return _depthwise_direct(input, weight, bias, pad_h, pad_w, OH, OW, SH, SW, C, KH, KW)
    if C != CG * groups or OC % groups:
        raise NotImplementedError("channel counts do not divide by groups")
    # wave60: 1x1 stride-1 pad-0 conv is a GEMM in NCHW already - route it
    # before anything touches the input (no pad, no channel padding, no copy).
    # wave61 extends the same route to strides (the tile walks whole output
    # rows, so the input is one affine strided load and the output block stays
    # contiguous): the stride-2 downsample used to be the last 1x1 shape class
    # on the tap kernel.  FLAGGEMS_SPACEMIT_CONV_1X1=0 puts the tap kernel back.
    if (KH == 1 and KW == 1 and groups == 1
            and pad_h == 0 and pad_w == 0 and C >= 16 and OC >= 16
            and (OH - 1) * SH < H and (OW - 1) * SW < W
            and os.environ.get("FLAGGEMS_SPACEMIT_CONV_1X1", "1") != "0"):
        if SH == 1 and SW == 1:
            return _conv1x1_gemm(input, weight, bias, OH, OW)
        return _conv1x1_strided(input, weight, bias, OH, OW, SH, SW)
    # wave62: the 7x7 stride-2 stem is the one layer where 32-lane dots are
    # almost all padding (C=3 padded to 16 across 49 taps), so it goes through
    # a 147-row im2col buffer and the ordinary 1x1 GEMM: 143 -> 17.0 ms at
    # (1,3,224,224)->64.  FLAGGEMS_SPACEMIT_CONV_STEM=0 puts the tap kernel
    # back for A/B.
    if (groups == 1 and C <= 8 and KH == 7 and KW == 7
            and os.environ.get("FLAGGEMS_SPACEMIT_CONV_STEM", "1") != "0"):
        if input.dtype != torch.float32:
            # wave64 carrier, only reachable with the top-level carrier off:
            # half precision never enters the stem kernels, so the build and the
            # smt dot stay f32 and the output converts afterwards.
            y = _conv_stem_im2col(input.float(), weight.float(),
                                  bias.float() if bias is not None else None,
                                  OH, OW, SH, SW, pad_h, pad_w)
            return y.to(input.dtype)
        return _conv_stem_im2col(input, weight, bias, OH, OW, SH, SW, pad_h, pad_w,
                                 out_dtype=_out_dtype)
    # wave66: 3x3 stride-1 goes through a 9C patch matrix wherever M is small
    # and OC is not.  The tap kernel's nine passes over a 32-lane-split input
    # collapse there: (1,256,14,14) 15.4 vs 8.5 (1.80x), (1,128,28,28) 12.2 vs
    # 8.6 (1.42x), and the 7-pixel stages never reach the tap kernel at all
    # (161.3 ms on the fallback path against 14.6 here); at (1,64,56,56) it is
    # a wash (1.03x) and at 112 px it loses, hence the M <= 1024 / OC >= 128
    # gate.  C must be a power of two because 9C has to be two power-of-two
    # slabs.  FLAGGEMS_SPACEMIT_CONV_IM2COL3=0 pins the old route for A/B.
    if (groups == 1 and KH == 3 and KW == 3 and SH == 1 and SW == 1
            and C >= 32 and (C & (C - 1)) == 0 and OC >= 128
            and 16 <= OH * OW <= 1024
            and os.environ.get("FLAGGEMS_SPACEMIT_CONV_IM2COL3", "1") != "0"):
        return _conv3x3_im2col(input, weight, bias, OH, OW, pad_h, pad_w,
                               out_dtype=_out_dtype)
    x = input.contiguous()
    if pad_h or pad_w:
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
        H, W = x.shape[2], x.shape[3]
    if C < 16:
        if groups > 1:
            raise NotImplementedError("grouped conv with fewer than 16 channels")
        # keep the dot's K lanes useful: at C=3 (the 7x7 stem) the kernel would
        # run 3 real lanes of 32; the vendor path padded to 16 before its dot
        # and wave55 measured the stem on that padded tensor
        pad_c = 16 - C
        x = F.pad(x, (0, 0, 0, 0, 0, pad_c))
        weight = F.pad(weight, (0, 0, 0, 0, 0, pad_c))
        C = 16
        CG = 16
    y = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)
    if groups == 1:
        return _conv_launch(x, weight, bias, y, OH, OW, SH, SW, BLOCK_OW, BLOCK_OC, ROWS, BLOCK_C)
    # wave58: 1 < groups < C gets ONE launch (group folded into the leading axis,
    # weight prepacked per group) on the shapes the per-group path could not
    # serve - more than 8 groups, or under 512 px of output per group - and keeps
    # the per-group loop inside its old band, where it is the faster of the two
    # ((1,128,28,28) g4: 7.92 vs 15.86 ms).  FLAGGEMS_SPACEMIT_CONV_GROUP1=0
    # forces the per-group path for A/B.
    if ((groups > 8 or OH * OW < 512)
            and os.environ.get("FLAGGEMS_SPACEMIT_CONV_GROUP1", "1") != "0"):
        return _conv_group_launch(x, weight, bias, y, OH, OW, SH, SW, groups,
                                  BLOCK_OW, BLOCK_OC, ROWS, BLOCK_C)
    OCG = OC // groups
    xv = x.view(N, groups, CG, H, W)
    wv = weight.view(groups, OCG, CG, KH, KW)
    for gi in range(groups):
        bg = bias[gi * OCG:(gi + 1) * OCG] if bias is not None else None
        _conv_launch(xv[:, gi], wv[gi], bg, y[:, gi * OCG:(gi + 1) * OCG],
                     OH, OW, SH, SW, BLOCK_OW, BLOCK_OC, ROWS, BLOCK_C)
    return y
