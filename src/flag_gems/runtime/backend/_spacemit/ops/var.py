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

"""spacemit vendor override: var / var_dim / var_correction / var_mean / std
(campaign 1015pm wave7; wave10 additions).

1) Correctness fix (unconditional): aten.var.dim's schema is
   ``var.dim(Tensor self, int[1]? dim, bool unbiased, bool keepdim)`` - all
   positional - so torch calls the python impl as ``var_dim(x, dim, unbiased)``
   whenever ``unbiased`` is spelled out. The generic FlagGems signature
   ``var_dim(x, dim=None, *, correction=None, keepdim=False)`` raised
   TypeError ("takes from 1 to 2 positional arguments but 3 were given") for
   every ``torch.var(x, dim=..., unbiased=False)`` under use_gems
   (probe62/63). This wrapper accepts the positional form and maps
   unbiased -> correction (True -> 1, False -> 0).

2) Speed (fast paths, var_fast.py):
   - trailing-dim multi-row two-pass kernels; f32 (wave7, probe65-67) and
     f16 (wave10, loop-acc pass-1; probe92/93).
   - wave10 flat (all-elements) two-stage path: applies when dim is None or
     a single dim covers every element. Same-session 1M: f32 8.64 -> 1.06 ms
     (torch 1.07); f16 8.50 -> 0.99 ms (torch 4.40); every probe check exact
     vs torch (large-mean / M=700000 / M=700000-f16 included).

Anything not covered (non-f32/f16, non-contiguous, small numel, multi-dim
dims, odd widths) falls through to the generic implementations unchanged.

Kill switch: FLAGGEMS_SPACEMIT_VAR_FAST=0 (fast paths only; the var_dim
signature fix is always active).
"""

import logging

import torch

from flag_gems.ops.std import std as _orig_std
from flag_gems.ops.var import (
    var as _orig_var,
    var_correction as _orig_var_correction,
    var_dim as _orig_var_dim,
)
from flag_gems.ops.var_mean import var_mean as _orig_var_mean
from flag_gems.runtime import torch_device_fn

from .var_fast import var_dim_mmk_launch, var_fast_enabled, var_flat_launch

logger = logging.getLogger(__name__)

_FAST_MIN_NUMEL = 65536
_FAST_DTYPES = (torch.float32, torch.float16)


def _trailing_target(x, dim):
    """Return (M, N) if the multi-row fast path applies to this (x, dim)."""
    if not var_fast_enabled():
        return None
    if x.dtype not in _FAST_DTYPES or not x.is_contiguous():
        return None
    if dim is None or x.numel() < _FAST_MIN_NUMEL:
        return None
    dims = [dim] if isinstance(dim, int) else list(dim)
    if len(dims) != 1:
        return None
    if dims[0] % x.ndim != x.ndim - 1:
        return None
    N = x.shape[-1]
    M = x.numel() // N
    if M <= 0 or N <= 0:
        return None
    return M, N


def _fast_reduce(x, dim, corr, keepdim, want_mean=False, sqrt=False):
    """Return var (and mean) on the trailing-dim fast path, else None."""
    t = _trailing_target(x, dim)
    if t is None:
        return None
    M, N = t
    if N - float(corr) <= 0:
        return None
    final_shape = x.shape[:-1] + ((1,) if keepdim else ())
    var_out = torch.empty(final_shape, dtype=x.dtype, device=x.device)
    if want_mean:
        mean_out = torch.empty(final_shape, dtype=x.dtype, device=x.device)
        mean_flat = mean_out.reshape(M)
    else:
        mean_flat = var_out.reshape(M)  # placeholder, never stored
    with torch_device_fn.device(x.device):
        handled = var_dim_mmk_launch(
            x, var_out.reshape(M), mean_flat, M, N, corr, want_mean, sqrt
        )
    if not handled:
        return None
    if want_mean:
        return var_out, mean_out
    return var_out


def _flat_target(x, dim):
    """Return numel if the all-elements flat fast path applies, else None."""
    if not var_fast_enabled():
        return None
    if x.dtype not in _FAST_DTYPES or not x.is_contiguous():
        return None
    if x.numel() < _FAST_MIN_NUMEL:
        return None
    if dim is None:
        return x.numel()
    dims = [dim] if isinstance(dim, int) else list(dim)
    if len(dims) != 1 or x.ndim == 0:
        return None
    d = dims[0] % x.ndim
    if x.shape[d] != x.numel():
        return None
    return x.numel()


def _flat_reduce(x, dim, corr, keepdim, want_mean=False, sqrt=False):
    """Return var (and mean) on the flat fast path, else None."""
    n = _flat_target(x, dim)
    if n is None:
        return None
    if n - float(corr) <= 0:
        return None
    res = var_flat_launch(x, corr, want_mean, sqrt)
    if res is None:
        return None
    if dim is None:
        shape = ()
    else:
        dims = [dim] if isinstance(dim, int) else list(dim)
        d = dims[0] % x.ndim
        shp = list(x.shape)
        if keepdim:
            shp[d] = 1
        else:
            shp.pop(d)
        shape = tuple(shp)
    if want_mean:
        v, mm = res
        return v.reshape(shape), mm.reshape(shape)
    return res.reshape(shape)


def _corr_of(correction):
    return 1.0 if correction is None else float(correction)


def _grad_delegate_wanted(x) -> bool:
    """wave8: aten::var and aten::var.dim have NO Autograd formula on this
    spine-torch build (both composite-only, probe70 dispatch dump), so
    grad-mode calls used to detach (x.grad=None + 'autograd kernel was not
    registered' warning, probe68). Delegate grad-tracking calls to
    aten::var.correction, which does carry a formula (probe71/72:
    grads bit-close vs analytic; probe74: flat torch.var(x) already routes
    through var.correction natively)."""
    return bool(
        torch.is_grad_enabled()
        and isinstance(x, torch.Tensor)
        and x.requires_grad
    )


def var(x, dim=None, *args, correction=None, keepdim=False, **kwargs):
    """aten::var.default: (self, unbiased=True) - the legacy bool arrives
    POSITIONALLY: torch.var(x, unbiased=False) / torch.var(x, False) used to
    land in `dim` and crash the generic fallback (len(False), probe73).
    wave8 maps positional bools to correction, and grad-tracking calls
    delegate to var.correction (formula-carrying)."""
    logger.debug("GEMS_SPACEMIT VAR (fast override)")
    unbiased = None
    if isinstance(dim, bool):
        unbiased = bool(dim)
        dim = None
    else:
        for a in args:
            if isinstance(a, bool):
                unbiased = bool(a)
                break
    if correction is None and unbiased is not None:
        correction = 1.0 if unbiased else 0.0
    corr = _corr_of(correction)
    if _grad_delegate_wanted(x):
        return torch.ops.aten.var.correction(
            x, dim=dim, correction=corr, keepdim=keepdim
        )
    r = _fast_reduce(x, dim, corr, keepdim)
    if r is not None:
        return r
    r = _flat_reduce(x, dim, corr, keepdim)
    if r is not None:
        return r
    return _orig_var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT VAR_CORRECTION (fast override)")
    corr = _corr_of(correction)
    r = _fast_reduce(x, dim, corr, keepdim)
    if r is not None:
        return r
    r = _flat_reduce(x, dim, corr, keepdim)
    if r is not None:
        return r
    return _orig_var_correction(
        x, dim=dim, correction=correction, keepdim=keepdim
    )


def var_dim(x, dim=None, *args, correction=None, keepdim=False, **kwargs):
    """aten.var.dim: (self, dim, unbiased, keepdim) - unbiased arrives as a
    positional bool (this is the probe62 TypeError). Also tolerates a
    positional correction number and a positional keepdim."""
    logger.debug("GEMS_SPACEMIT VAR_DIM (fast override)")
    corr = None if correction is None else float(correction)
    kd = bool(keepdim)
    bools = [a for a in args if isinstance(a, bool)]
    others = [a for a in args if not isinstance(a, bool)]
    if bools:
        # schema order: unbiased first, keepdim second
        if corr is None:
            corr = 1.0 if bools[0] else 0.0
        if len(bools) > 1:
            kd = bools[1]
    elif others and corr is None:
        corr = float(others[0])
    if corr is None:
        corr = 1.0
    if _grad_delegate_wanted(x):
        return torch.ops.aten.var.correction(
            x, dim=dim, correction=corr, keepdim=kd
        )
    r = _fast_reduce(x, dim, corr, kd)
    if r is not None:
        return r
    r = _flat_reduce(x, dim, corr, kd)
    if r is not None:
        return r
    return _orig_var_dim(x, dim=dim, correction=corr, keepdim=kd)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT VAR_MEAN (fast override)")
    corr = _corr_of(correction)
    r = _fast_reduce(x, dim, corr, keepdim, want_mean=True)
    if r is not None:
        return r
    r = _flat_reduce(x, dim, corr, keepdim, want_mean=True)
    if r is not None:
        return r
    return _orig_var_mean(
        x, dim=dim, correction=correction, keepdim=keepdim
    )


def std(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT STD (fast override)")
    corr = _corr_of(correction)
    r = _fast_reduce(x, dim, corr, keepdim, sqrt=True)
    if r is not None:
        return r
    r = _flat_reduce(x, dim, corr, keepdim, sqrt=True)
    if r is not None:
        return r
    return _orig_std(x, dim=dim, correction=correction, keepdim=keepdim)
