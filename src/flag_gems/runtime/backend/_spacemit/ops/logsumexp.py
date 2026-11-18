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

"""vendor logsumexp: trailing-dim row + dim=0 fast paths (wave18/19).

f16/bf16 inputs run the same f32 kernels into an f32 scratch buffer and are
cast back in torch - the reduce path cannot store f16 directly on this
spine-opt build (SIGABRT, wave17 prod finding), and the cast is ~0.01 ms.
"""

import math

import torch

from flag_gems.ops.logsumexp import logsumexp as _generic_logsumexp

from .logsumexp_fast import (
    logsumexp_dim0_launch,
    logsumexp_fast_enabled,
    logsumexp_mid_launch,
    logsumexp_rows_launch,
)

_FAST_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _fast_ok(inp) -> bool:
    return (
        logsumexp_fast_enabled()
        and inp.is_contiguous()
        and inp.numel() > 0
        and inp.dtype in _FAST_DTYPES
    )


def _cast(out32, inp):
    return out32 if out32.dtype == inp.dtype else out32.to(inp.dtype)


def logsumexp(inp, dim, keepdim=False):
    # aten::logsumexp takes dim as int[1], so use_gems dispatches a list here.
    if isinstance(dim, (list, tuple)):
        if len(dim) != 1:
            return _generic_logsumexp(inp, dim, keepdim)
        dim = dim[0]
    if isinstance(dim, int) and _fast_ok(inp) and inp.ndim >= 1:
        d = dim % inp.ndim
        shape = list(inp.shape)
        N = shape[d]
        M = math.prod(shape[:d]) if d else 1
        K = math.prod(shape[d + 1 :]) if d + 1 < len(shape) else 1
        if K == 1 and N > 1:
            out_shape = list(shape)
            out_shape[d] = 1
            out32 = torch.empty(out_shape, dtype=torch.float32, device=inp.device)
            if logsumexp_rows_launch(inp, out32.view(-1), M, N):
                out = _cast(out32, inp)
                return out if keepdim else out.squeeze(d)
        elif d == 0 and inp.ndim >= 2 and N > 1:
            out_shape = shape[1:]
            out32 = torch.empty(out_shape, dtype=torch.float32, device=inp.device)
            if logsumexp_dim0_launch(inp, out32.view(-1), N, K):
                out = _cast(out32, inp)
                return out.unsqueeze(0) if keepdim else out
        elif d > 0 and N > 1 and K > 1:
            # wave20: middle-dim fold, X viewed as (M, N, K) with K contiguous.
            out_shape = list(shape)
            del out_shape[d]
            out32 = torch.empty(out_shape, dtype=torch.float32, device=inp.device)
            if logsumexp_mid_launch(inp, out32.view(-1), M, N, K):
                out = _cast(out32, inp)
                return out.unsqueeze(d) if keepdim else out
    return _generic_logsumexp(inp, dim, keepdim)
