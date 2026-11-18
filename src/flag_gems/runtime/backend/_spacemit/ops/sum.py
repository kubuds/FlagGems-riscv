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

"""spacemit vendor override: sum / sum_dim / sum_dim_out / sum_out (wave6 + wave30).

Fast paths live in sum_fast.py; everything not covered falls through to the
generic implementations (imported below by name from flag_gems.ops.sum, which
are module-level functions and therefore NOT affected by the vendor rebind of
the flag_gems globals).

Coverage:
- sum / sum_out: contiguous, numel >= 1 -> raised-BLOCK two-stage
  (1M f32: 1.85 -> 0.57 ms same-session).
- sum_dim / sum_dim_out: single trailing dim, K == 1, contiguous,
  numel >= 65536, row width in {512,1024,2048,4096,32768}, M divisible by the
  kernel's K -> multi-row per program.
- sum_dim over dim 0 -> row-fold two-stage.

wave30: f16 and bf16 ride an f32 carrier (mid buffer, accumulators and output
buffer are f32) and the result is cast to the requested dtype at the end.
torch.sum(f16) accumulates in float and returns half, so this matches torch
semantics; the generic route instead stored f16 partials and ran at 23 ms per
4M-element reduce.

Kill switch: FLAGGEMS_SPACEMIT_SUM_FAST=0.
"""

import logging

import torch

from flag_gems.ops.sum import sum as _orig_sum
from flag_gems.ops.sum import sum_dim as _orig_sum_dim
from flag_gems.ops.sum import sum_dim_out as _orig_sum_dim_out
from flag_gems.ops.sum import sum_out as _orig_sum_out
from flag_gems.runtime import torch_device_fn

from .sum_fast import (
    sum_dim0_launch,
    sum_dim_mmk_launch,
    sum_fast_enabled,
    sum_flat_launch,
)

logger = logging.getLogger(__name__)

_FAST_MIN_NUMEL = 65536
_FAST_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _want_dtype(inp, dtype):
    return inp.dtype if dtype is None else dtype


def _fast_dtype_ok(inp, dtype):
    """f32, or 16-bit input with either its own dtype or f32 as the result."""
    if inp.dtype not in _FAST_DTYPES:
        return False
    want = _want_dtype(inp, dtype)
    return want == inp.dtype or want == torch.float32


def _flat_fast_ok(inp, dtype):
    if not sum_fast_enabled():
        return False
    if not inp.is_contiguous():
        return False
    if not _fast_dtype_ok(inp, dtype):
        return False
    if inp.numel() < 1:
        return False
    return True


def _cast_out(out, want):
    return out if out.dtype == want else out.to(want)


def _dim_fast_target(inp, dim, keepdim):
    """Return (M, N) if the multi-row fast path applies, else None."""
    if not sum_fast_enabled():
        return None
    if not inp.is_contiguous() or inp.dtype not in _FAST_DTYPES:
        return None
    if inp.numel() < _FAST_MIN_NUMEL or inp.numel() == 0 or dim is None:
        return None
    if isinstance(dim, (list, tuple)):
        if len(dim) != 1:
            return None
        d = dim[0]
    else:
        d = dim
    d = d % inp.ndim
    if d != inp.ndim - 1:
        return None
    N = inp.shape[d]
    M = inp.numel() // N
    return M, N


def _dim0_fast_target(inp, dim, keepdim):
    """Return (M, N, final_shape) for the dim0 row-fold path, else None."""
    if not sum_fast_enabled():
        return None
    if not inp.is_contiguous() or inp.dtype not in _FAST_DTYPES:
        return None
    if inp.numel() < _FAST_MIN_NUMEL or inp.numel() == 0 or inp.ndim < 2:
        return None
    if dim is None:
        return None
    if isinstance(dim, (list, tuple)):
        if len(dim) != 1:
            return None
        d = dim[0]
    else:
        d = dim
    d = d % inp.ndim
    if d != 0:
        return None
    M = inp.shape[0]
    N = inp.numel() // M
    final_shape = ((1,) if keepdim else ()) + inp.shape[1:]
    return M, N, final_shape


def sum(inp, *, dtype=None):
    logger.debug("GEMS_SPACEMIT SUM (fast override)")
    if _flat_fast_ok(inp, dtype):
        M = inp.numel()
        out = torch.empty([], dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            sum_flat_launch(inp, out, M)
        return _cast_out(out, _want_dtype(inp, dtype))
    return _orig_sum(inp, dtype=dtype)


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_SPACEMIT SUM_OUT (fast override)")
    if _flat_fast_ok(inp, dtype) and out.dtype in (torch.float32, inp.dtype):
        M = inp.numel()
        if out.dtype == torch.float32:
            with torch_device_fn.device(inp.device):
                sum_flat_launch(inp, out, M)
            return out
        tmp = torch.empty([], dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            sum_flat_launch(inp, tmp, M)
        out.copy_(tmp)
        return out
    return _orig_sum_out(inp, dtype=dtype, out=out)


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_SPACEMIT SUM_DIM (fast override)")
    if _fast_dtype_ok(inp, dtype):
        t = _dim_fast_target(inp, dim, keepdim)
        if t is not None:
            M, N = t
            final_shape = inp.shape[:-1] + ((1,) if keepdim else ())
            out = torch.empty(final_shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                if sum_dim_mmk_launch(inp, out.reshape(M), M, N):
                    return _cast_out(out, _want_dtype(inp, dtype))
        t0 = _dim0_fast_target(inp, dim, keepdim)
        if t0 is not None:
            M, N, fs = t0
            out = torch.empty(fs, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                if sum_dim0_launch(inp.reshape(M, N), out.reshape(N), M, N):
                    return _cast_out(out, _want_dtype(inp, dtype))
    return _orig_sum_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_SPACEMIT SUM_DIM_OUT (fast override)")
    if _fast_dtype_ok(inp, dtype) and out.dtype in (torch.float32, inp.dtype):
        direct = out.dtype == torch.float32
        t = _dim_fast_target(inp, dim, keepdim)
        if t is not None:
            M, N = t
            final_shape = inp.shape[:-1] + ((1,) if keepdim else ())
            if direct:
                out.resize_(final_shape)
                with torch_device_fn.device(inp.device):
                    if sum_dim_mmk_launch(inp, out.reshape(M), M, N):
                        return out
            else:
                tmp = torch.empty(final_shape, dtype=torch.float32, device=inp.device)
                with torch_device_fn.device(inp.device):
                    if sum_dim_mmk_launch(inp, tmp.reshape(M), M, N):
                        out.resize_(final_shape)
                        out.copy_(tmp)
                        return out
        t0 = _dim0_fast_target(inp, dim, keepdim)
        if t0 is not None:
            M, N, fs = t0
            if direct:
                out.resize_(fs)
                with torch_device_fn.device(inp.device):
                    if sum_dim0_launch(inp.reshape(M, N), out.reshape(N), M, N):
                        return out
            else:
                tmp = torch.empty(fs, dtype=torch.float32, device=inp.device)
                with torch_device_fn.device(inp.device):
                    if sum_dim0_launch(inp.reshape(M, N), tmp.reshape(N), M, N):
                        out.resize_(fs)
                        out.copy_(tmp)
                        return out
    return _orig_sum_dim_out(inp, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
