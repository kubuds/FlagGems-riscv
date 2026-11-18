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

"""vendor amin: last-dim row fast path (wave16); everything else generic."""

import torch

from flag_gems.ops.amin import amin as _generic_amin

from .amax_fast import amin_fast_enabled, amin_fast_launch


def amin(inp, dim=None, keepdim=False):
    if dim is not None and amin_fast_enabled() and inp.is_contiguous():
        dims = [dim] if isinstance(dim, int) else list(dim)
        if len(dims) == 1 and inp.ndim >= 2:
            d = dims[0] % inp.ndim
            if d == inp.ndim - 1:
                N = inp.shape[-1]
                M = inp.numel() // N if N else 0
                shape = list(inp.shape)
                if keepdim:
                    shape[-1] = 1
                else:
                    shape = shape[:-1]
                out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
                if amin_fast_launch(inp, out, M, N):
                    return out
    return _generic_amin(inp, dim, keepdim)
