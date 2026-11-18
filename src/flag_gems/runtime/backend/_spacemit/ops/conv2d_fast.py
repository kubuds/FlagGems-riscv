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


def conv_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_CONV_FAST", "1") != "0"


def conv_fast_applies(x, w, stride, dilation, groups) -> bool:
    if x.dim() != 4 or w.dim() != 4:
        return False
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    if dh != 1 or dw != 1:
        return False
    if x.dtype != torch.float32 or w.dtype != torch.float32:
        return False
    n, c, h, ww = x.shape
    oc, _, kh, kw = w.shape
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
    if macs < ((1 << 14) if dwise else (1 << 20)):
        return False
    if ww < (4 if dwise else 8):
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
                  BLOCK_OW=None, BLOCK_OC=None, ROWS=None, BLOCK_C=None):
    N, C, H, W = input.shape
    OC, CG, KH, KW = weight.shape
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
