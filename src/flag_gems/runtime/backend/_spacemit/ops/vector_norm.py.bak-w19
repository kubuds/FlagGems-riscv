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

"""vendor vector_norm: ord==2 fast paths (wave18); generic elsewhere.

torch.norm / torch.linalg.vector_norm dispatch to aten::linalg_vector_norm, so
this is the entry the official norm benchmark actually measures (probe168:
aten.norm.ScalarOpt_dim 2.70 ms vs linalg_vector_norm 12.95 ms at 2048^2).
"""

import torch

from flag_gems.ops.vector_norm import vector_norm as _generic_vector_norm

from .norm_fast import norm_fast_enabled, norm_flat_launch, norm_rows_launch


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
    if (ord == 2 or ord == 2.0) and _fast_ok(x, dtype):
        nd = x.ndim
        if not dim or (isinstance(dim, (list, tuple)) and len(dim) == nd):
            out = torch.empty([], dtype=x.dtype, device=x.device)
            if norm_flat_launch(x, out, x.numel()):
                return out.reshape([1] * nd) if keepdim else out
        else:
            dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
            if len(dims) == 1 and isinstance(dims[0], int) and (dims[0] % nd) == nd - 1:
                N = x.shape[nd - 1]
                M = x.numel() // N if N else 0
                shape = list(x.shape)
                shape[nd - 1] = 1
                out = torch.empty(shape, dtype=x.dtype, device=x.device)
                if norm_rows_launch(x, out.view(-1), M, N):
                    return out if keepdim else out.squeeze(nd - 1)
    return _generic_vector_norm(x, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)
