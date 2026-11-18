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

"""spacemit vendor override: sum / sum_dim / sum_dim_out / sum_out (wave6).

Fast paths live in sum_fast.py; everything not covered falls through to the
generic implementations (imported below by name from flag_gems.ops.sum, which
are module-level functions and therefore NOT affected by the vendor rebind of
the flag_gems globals).

Coverage:
- sum / sum_out: f32, contiguous, numel >= 65536 -> raised-BLOCK two-stage
  (1M: 1.85 -> 0.57 ms same-session).
- sum_dim / sum_dim_out: single trailing dim, K == 1, f32 contiguous,
  numel >= 65536, row width in {512,1024,2048,4096,32768}, M divisible by the
  kernel's K -> multi-row per program.

Kill switch: FLAGGEMS_SPACEMIT_SUM_FAST=0.
"""

import logging

import torch

from flag_gems.ops.sum import sum as _orig_sum
from flag_gems.ops.sum import sum_dim as _orig_sum_dim
from flag_gems.ops.sum import sum_dim_out as _orig_sum_dim_out
from flag_gems.ops.sum import sum_out as _orig_sum_out
from flag_gems.runtime import torch_device_fn

from .sum_fast import sum_dim_mmk_launch, sum_fast_enabled, sum_flat_launch

logger = logging.getLogger(__name__)

_FAST_MIN_NUMEL = 65536


def _flat_fast_ok(inp, dtype):
    if not sum_fast_enabled():
        return False
    if inp.dtype != torch.float32 or not inp.is_contiguous():
        return False
    if dtype is not None and dtype != torch.float32:
        return False
    if inp.numel() < _FAST_MIN_NUMEL:
        return False
    return True


def _dim_fast_target(inp, dim, keepdim):
    """Return (M, N) if the multi-row fast path applies, else None."""
    if not sum_fast_enabled():
        return None
    if inp.dtype != torch.float32 or not inp.is_contiguous():
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


def sum(inp, *, dtype=None):
    logger.debug("GEMS_SPACEMIT SUM (fast override)")
    if _flat_fast_ok(inp, dtype):
        M = inp.numel()
        out = torch.empty([], dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            sum_flat_launch(inp, out, M)
        return out
    return _orig_sum(inp, dtype=dtype)


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_SPACEMIT SUM_OUT (fast override)")
    if _flat_fast_ok(inp, dtype):
        M = inp.numel()
        with torch_device_fn.device(inp.device):
            sum_flat_launch(inp, out, M)
        return out
    return _orig_sum_out(inp, dtype=dtype, out=out)


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_SPACEMIT SUM_DIM (fast override)")
    if dtype is None or dtype == torch.float32:
        t = _dim_fast_target(inp, dim, keepdim)
        if t is not None:
            M, N = t
            final_shape = inp.shape[:-1] + ((1,) if keepdim else ())
            out = torch.empty(final_shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                if sum_dim_mmk_launch(inp, out.reshape(M), M, N):
                    return out
    return _orig_sum_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_SPACEMIT SUM_DIM_OUT (fast override)")
    if dtype is None or dtype == torch.float32:
        t = _dim_fast_target(inp, dim, keepdim)
        if t is not None:
            M, N = t
            final_shape = inp.shape[:-1] + ((1,) if keepdim else ())
            out.resize_(final_shape)
            with torch_device_fn.device(inp.device):
                if sum_dim_mmk_launch(inp, out.reshape(M), M, N):
                    return out
    return _orig_sum_dim_out(inp, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
