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

"""vendor vector_norm: ord in {0, 1, 2, inf} fast paths; generic elsewhere.

torch.norm / torch.linalg.vector_norm dispatch to aten::linalg_vector_norm, so
this is the entry the norm benchmarks actually measure (probe168:
aten.norm.ScalarOpt_dim 2.70 ms vs linalg_vector_norm 12.95 ms at 2048^2).

wave18 wired ord=2. wave19 adds ord=1 and ord=inf with the same narrow-carrier
kernels (probe169b, rows f32): ord=1 generic 23.6/193.2/92.0/368.2 ->
0.347/2.419/1.038/4.356 ms; ord=inf generic 2.86/13.11/4.26/39.24 ->
0.350/2.400/1.044/4.231 ms at 512^2 / 2048^2 / 512x4096 / 8192x512.

wave20 adds ord=0 (non-zero count). This is also a bug fix: the generic
implementation does `len(dim)` on an int dim and raises
`TypeError: object of type 'int' has no len()` for every ord it does not
specialise (probe174 hit it on ord=0).
"""

import torch

from flag_gems.ops.vector_norm import vector_norm as _generic_vector_norm

from .norm_fast import (
    l0_flat_launch,
    l0_rows_launch,
    l1_flat_launch,
    l1_rows_launch,
    linf_flat_launch,
    linf_rows_launch,
    norm_fast_enabled,
    norm_flat_launch,
    norm_rows_launch,
)

_ROW_LAUNCH = {
    "l2": norm_rows_launch,
    "l1": l1_rows_launch,
    "linf": linf_rows_launch,
    "l0": l0_rows_launch,
}
_FLAT_LAUNCH = {
    "l2": norm_flat_launch,
    "l1": l1_flat_launch,
    "linf": linf_flat_launch,
    "l0": l0_flat_launch,
}


def _mode(ord):
    if ord == 2 or ord == 2.0:
        return "l2"
    if ord == 1 or ord == 1.0:
        return "l1"
    if ord == float("inf"):
        return "linf"
    if ord == 0 or ord == 0.0:
        return "l0"
    return None


def _fast_ok(x, dtype) -> bool:
    return (
        norm_fast_enabled()
        and isinstance(x, torch.Tensor)
        and x.is_contiguous()
        and x.numel() > 0
        and x.dtype == torch.float32
        and (dtype is None or dtype == x.dtype)
    )


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    mode = _mode(ord)
    if mode is not None and _fast_ok(x, dtype):
        nd = x.ndim
        if not dim or (isinstance(dim, (list, tuple)) and len(dim) == nd):
            out = torch.empty([], dtype=x.dtype, device=x.device)
            if _FLAT_LAUNCH[mode](x, out, x.numel()):
                return out.reshape([1] * nd) if keepdim else out
        else:
            dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
            if len(dims) == 1 and isinstance(dims[0], int) and (dims[0] % nd) == nd - 1:
                N = x.shape[nd - 1]
                M = x.numel() // N if N else 0
                shape = list(x.shape)
                shape[nd - 1] = 1
                out = torch.empty(shape, dtype=x.dtype, device=x.device)
                if _ROW_LAUNCH[mode](x, out.view(-1), M, N):
                    return out if keepdim else out.squeeze(nd - 1)
    return _generic_vector_norm(x, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)
