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

"""vendor prod: row / flat / dim0 fast paths (wave17); everything else generic."""

import math
import os

import torch

from flag_gems.ops.prod import prod as _generic_prod
from flag_gems.ops.prod import prod_dim as _generic_prod_dim

from .prod_fast import (
    prod_dim0_launch,
    prod_fast_enabled,
    prod_flat_launch,
    prod_rows_launch,
)


def _fast_ok(inp) -> bool:
    return (
        prod_fast_enabled()
        and inp.is_contiguous()
        and inp.numel() > 0
        and inp.dtype == torch.float32
    )


def prod(inp, dim=None, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = inp.dtype
    if dim is not None:
        d = dim % inp.ndim if inp.ndim else 0
        if _fast_ok(inp) and inp.ndim >= 2:
            shape = list(inp.shape)
            N = shape[d]
            M = math.prod(shape[:d]) if d else 1
            K = math.prod(shape[d + 1 :]) if d + 1 < len(shape) else 1
            out_shape = list(shape)
            if keepdim:
                out_shape[d] = 1
            else:
                del out_shape[d]
            out = torch.empty(out_shape, dtype=dtype, device=inp.device)
            if K == 1 and N > 1 and prod_rows_launch(inp, out, M, N):
                return out
        return _generic_prod_dim(inp, dim, keepdim, dtype=dtype)
    if _fast_ok(inp):
        out = torch.empty([], dtype=dtype, device=inp.device)
        if prod_flat_launch(inp, out, inp.numel()):
            return out
    return _generic_prod(inp, dtype=dtype)


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    return prod(inp, dim, keepdim, dtype=dtype)
