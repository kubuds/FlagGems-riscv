# wave54/wave55: direct (im2col-free) conv2d for the K3 backend.
#
# The vendor conv2d runs a fused im2col + bmm with one program per output
# pixel and a global mbarrier between the two stages.  On the K3 that path is
# 4-9x slower than a plain output-stationary direct convolution with the same
# f32 accumulate order, and at 1x3x224x224 (7x7, stride 2) it did not finish
# compiling in 9 minutes.
#
# wave55 changes on top of the wave54 kernel:
#   - the weight is prepacked to (KH, KW, OC, C) once per call, so the dot's
#     weight tile is contiguous instead of strided by KH*KW (1.4-1.5x on the
#     medium shapes; the wave54 layout loaded 64 rows of 32 floats at stride
#     9, which spine-opt cannot vectorise)
#   - stride is supported (the index is oh*SH + kh and the column step is
#     SW*sxw), which covers the stride-2 ResNet stages where torch does not
#     return and the vendor path takes 52 ms at (1,64,112,112)
#   - a tile picker keeps at least 32 programs before it grows a tile
#   - the narrow-input gate (W >= 32) is gone: prepacked weights make
#     W=17..31 shapes win too (0.8-2.2 ms vs 3.1-8.9 ms for the vendor)
#
# The kernel keeps the input pre-padded so the inner loop has no negative
# offsets (a scalar minus in the pointer expression makes spine-opt reject the
# load in PtrAnalysis and the whole kernel goes scalar - measured 2+ minutes
# for one 1x64x56x56 call).  Taps must be static_range: a dynamic tap loop
# around the dot makes spine-opt abort with "scf.for not bufferizable".
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def conv_fast_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_CONV_FAST", "1") != "0"


def conv_fast_applies(x, w, stride, dilation, groups) -> bool:
    if groups != 1 or x.dim() != 4 or w.dim() != 4:
        return False
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    if dh != 1 or dw != 1:
        return False
    if x.dtype != torch.float32 or w.dtype != torch.float32:
        return False
    n, c, h, ww = x.shape
    oc, _, kh, kw = w.shape
    # the launch floor and the pad copy dominate below this; the vendor path
    # is fine there (1x8x16x16: vendor 0.9 ms vs 2.3 ms here)
    if n * oc * h * ww * c * kh * kw < (1 << 20):
        return False
    if ww < 8:
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
):
    pid = tl.program_id(0)
    n_ow = tl.cdiv(OW, BLOCK_OW)
    n_oc = tl.cdiv(OC, BLOCK_OC)
    pid_ow = pid % n_ow
    pid_oc = (pid // n_ow) % n_oc
    pid_nh = pid // (n_ow * n_oc)
    pid_n = pid_nh // OH
    oh = pid_nh % OH
    ow0 = pid_ow * BLOCK_OW
    lo = tl.arange(0, BLOCK_OW)
    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)
    for kh in tl.static_range(KH):
        ih = oh * SH + kh
        for kw in tl.static_range(KW):
            iw0 = ow0 * SW + kw
            for c0 in range(0, C, BLOCK_C):
                cs = c0 + tl.arange(0, BLOCK_C)
                xp = X + pid_n * sxn + ih * sxh + iw0 * sxw + cs[:, None] * sxc + (lo[None, :] * SW) * sxw
                xv = tl.load(xp, mask=(cs[:, None] < C) & (lo[None, :] < OW - ow0), other=0.0)
                wpt = Wt + kh * swkk + kw * swkw + oc[:, None] * swoc + cs[None, :] * swc
                wv = tl.load(wpt, mask=(oc[:, None] < OC) & (cs[None, :] < C), other=0.0)
                acc += tl.dot(wv, xv)
    if HAS_BIAS:
        acc += tl.load(Bs + oc, mask=oc < OC, other=0.0)[:, None]
    yp = Y + pid_n * syn + oh * syoh + ow0 * syow + oc[:, None] * syoc + lo[None, :] * syow
    tl.store(yp, acc.to(Y.dtype.element_ty), mask=(oc[:, None] < OC) & ((ow0 + lo)[None, :] < OW))


def _pick_tiles(n, oh, ow, oc, min_prog=32):
    bow = 64
    boc = 64 if oc >= 64 else 32 if oc >= 32 else 16
    def nprog():
        return n * oh * triton.cdiv(ow, bow) * triton.cdiv(oc, boc)
    if nprog() < min_prog and bow > 32:
        bow = 32
    if nprog() < min_prog and boc > 16:
        boc = max(16, boc // 2)
    return bow, boc


def conv2d_direct(input, weight, bias, pad_h, pad_w, OH, OW, SH=1, SW=1, BLOCK_OW=None, BLOCK_OC=None):
    N, C, H, W = input.shape
    OC, _, KH, KW = weight.shape
    x = input.contiguous()
    if pad_h or pad_w:
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
        H, W = x.shape[2], x.shape[3]
    wp = weight.permute(2, 3, 0, 1).contiguous()   # (KH, KW, OC, C)
    if BLOCK_OW is None:
        BLOCK_OW, BLOCK_OC_ = _pick_tiles(N, OH, OW, OC)
        BLOCK_OC = BLOCK_OC if BLOCK_OC is not None else BLOCK_OC_
    BLOCK_C = 32
    y = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)
    grid = (N * OH * triton.cdiv(OW, BLOCK_OW) * triton.cdiv(OC, BLOCK_OC),)
    _conv2d_direct_kernel[grid](
        x, wp, bias if bias is not None else x, y,
        C, H, W, OH, OW, OC,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        wp.stride(0), wp.stride(1), wp.stride(3), wp.stride(2),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        SH=SH, SW=SW, HAS_BIAS=(bias is not None), KH=KH, KW=KW,
        BLOCK_OW=BLOCK_OW, BLOCK_OC=BLOCK_OC, BLOCK_C=BLOCK_C,
    )
    return y
