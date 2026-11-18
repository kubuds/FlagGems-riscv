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

"""vendor norm: p=2 fast paths (wave18) + dim=None scalar fix; generic elsewhere."""

import math

import torch

from flag_gems.ops.norm import norm as _generic_norm

from .norm_fast import norm_fast_enabled, norm_flat_launch, norm_rows_launch


_FAST_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _fast_ok(x) -> bool:
    return (
        norm_fast_enabled()
        and isinstance(x, torch.Tensor)
        and x.is_contiguous()
        and x.numel() > 0
        and x.dtype in _FAST_DTYPES
    )


def _acc_dtype(x):
    return x.dtype if x.dtype == torch.float32 else torch.float32


def _cast(out, x):
    return out if out.dtype == x.dtype else out.to(x.dtype)


def norm(x, p=2, dim=None, keepdim=False):
    if p == 2 and _fast_ok(x):
        if dim is None:
            # torch.norm(x, 2) returns a 0-dim tensor
            out = torch.empty([], dtype=_acc_dtype(x), device=x.device)
            if norm_flat_launch(x, out, x.numel()):
                return _cast(out, x)
        else:
            # aten::norm.ScalarOpt_dim takes dim as int[1]
            dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
            nd = x.ndim
            if len(dims) == 1 and isinstance(dims[0], int) and nd >= 1 and (dims[0] % nd) == nd - 1:
                d = dims[0] % nd
                N = x.shape[nd - 1]
                M = x.numel() // N if N else 0
                shape = list(x.shape)
                shape[nd - 1] = 1
                out = torch.empty(shape, dtype=_acc_dtype(x), device=x.device)
                if norm_rows_launch(x, out.view(-1), M, N):
                    out = _cast(out, x)
                    return out if keepdim else out.squeeze(nd - 1)
    return _generic_norm(x, p=p, dim=dim, keepdim=keepdim)


def norm_scalar(x, p=2):
    return norm(x, p=p, dim=None, keepdim=False)


def norm_scalaropt_dim(x, p, dim, keepdim=False):
    return norm(x, p=p, dim=dim, keepdim=keepdim)
