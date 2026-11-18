# wave54: direct (im2col-free) conv2d for the K3 backend.
#
# The vendor conv2d runs a fused im2col + bmm with one program per output
# pixel and a global mbarrier between the two stages.  On the K3 that path is
# 4-9x slower than a plain output-stationary direct convolution with the same
# f32 accumulate order, and at 1x3x224x224 (7x7, stride 2) it did not finish
# compiling in 9 minutes.
#
# This kernel: one program per (n, oh, ow-block, oc-block); the input is
# pre-padded so the inner loop has no negative offsets (a scalar minus in the
# pointer expression makes spine-opt reject the load in PtrAnalysis and the
# whole kernel goes scalar - measured 2+ minutes for one 1x64x56x56 call).
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
    if sh != 1 or sw != 1 or dh != 1 or dw != 1:
        return False
    if x.dtype != torch.float32 or w.dtype != torch.float32:
        return False
    n, c, h, ww = x.shape
    oc, _, kh, kw = w.shape
    # below this the launch floor and the pad copy dominate; the vendor path
    # is fine there (1x8x16x16: vendor 0.9 ms vs 2.3 ms here)
    if n * oc * h * ww * c * kh * kw < (1 << 20):
        return False
    if ww < 32:
        return False
    return True


@triton.jit
def _conv2d_pad0_kernel(X, Wt, Bs, Y,
    C, IH, IW, OH, OW, OC,
    sxn, sxc, sxh, sxw,
    swo, swc,
    syn, syoc, syoh, syow,
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
        ih = oh + kh
        for kw in tl.static_range(KW):
            iw0 = ow0 + kw
            for c0 in range(0, C, BLOCK_C):
                cs = c0 + tl.arange(0, BLOCK_C)
                xp = X + pid_n * sxn + ih * sxh + iw0 * sxw + cs[:, None] * sxc + lo[None, :] * sxw
                xv = tl.load(xp, mask=(cs[:, None] < C) & (lo[None, :] < OW - ow0), other=0.0)
                wpt = Wt + oc[:, None] * swo + cs[None, :] * swc + (kh * KW + kw)
                wv = tl.load(wpt, mask=(oc[:, None] < OC) & (cs[None, :] < C), other=0.0)
                acc += tl.dot(wv, xv)
    if HAS_BIAS:
        acc += tl.load(Bs + oc, mask=oc < OC, other=0.0)[:, None]
    yp = Y + pid_n * syn + oh * syoh + ow0 * syow + oc[:, None] * syoc + lo[None, :] * syow
    tl.store(yp, acc.to(Y.dtype.element_ty), mask=(oc[:, None] < OC) & ((ow0 + lo)[None, :] < OW))


def conv2d_direct(input, weight, bias, pad_h, pad_w, OH, OW):
    N, C, H, W = input.shape
    OC, _, KH, KW = weight.shape
    x = input.contiguous()
    if pad_h or pad_w:
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
        H, W = x.shape[2], x.shape[3]
    w = weight.contiguous()
    y = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)
    BLOCK_OW = 64
    BLOCK_OC = 64 if (OC % 64 == 0 and OC >= 64) else 32
    BLOCK_C = 32
    grid = (N * OH * triton.cdiv(OW, BLOCK_OW) * triton.cdiv(OC, BLOCK_OC),)
    _conv2d_pad0_kernel[grid](
        x, w, bias if bias is not None else x, y,
        C, H, W, OH, OW, OC,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        HAS_BIAS=(bias is not None), KH=KH, KW=KW,
        BLOCK_OW=BLOCK_OW, BLOCK_OC=BLOCK_OC, BLOCK_C=BLOCK_C,
    )
    return y
