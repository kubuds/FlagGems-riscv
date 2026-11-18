# wave57 candidate: conv2d for the K3 backend.
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
        # wave57: grouped conv runs as g independent (c/g -> oc/g) problems,
        # one launch each, so it needs enough area per group to amortise the
        # dispatch: (1,256,14,14) g8 (196 px/group) measured 0.76x and
        # (1,32,16,16) g2 (256 px) 0.72x, while 784 px/group and up wins 1.5x+.
        if c % groups or oc % groups:
            return False
        if (c // groups) < 8 or (oc // groups) < 16:
            return False
        sh0, sw0 = (stride, stride) if isinstance(stride, int) else stride
        if groups > 8 or (h // sh0) * (ww // sw0) < 512:
            return False
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


def conv2d_direct(input, weight, bias, pad_h, pad_w, OH, OW, SH=1, SW=1, groups=1,
                  BLOCK_OW=None, BLOCK_OC=None, ROWS=None, BLOCK_C=None):
    N, C, H, W = input.shape
    OC, CG, KH, KW = weight.shape
    if groups == C == OC:
        return _depthwise_direct(input, weight, bias, pad_h, pad_w, OH, OW, SH, SW, C, KH, KW)
    if C != CG * groups or OC % groups:
        raise NotImplementedError("channel counts do not divide by groups")
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
    # wave57: 1 < groups < C - the last conv case without coverage.  Each group
    # is an independent (CG -> OCG) convolution whose channel slabs sit back to
    # back in x, weight and y, so it is one wave56 launch per group on a strided
    # slice; the kernel itself is untouched.  (Folding the group into the batch
    # axis instead reads the wrong weight rows past group 0 - probe952.)
    OCG = OC // groups
    xv = x.view(N, groups, CG, H, W)
    wv = weight.view(groups, OCG, CG, KH, KW)
    for gi in range(groups):
        bg = bias[gi * OCG:(gi + 1) * OCG] if bias is not None else None
        _conv_launch(xv[:, gi], wv[gi], bg, y[:, gi * OCG:(gi + 1) * OCG],
                     OH, OW, SH, SW, BLOCK_OW, BLOCK_OC, ROWS, BLOCK_C)
    return y
