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

"""vendor max_dim: reuse the argmax row/dim0 kernels + a value gather (wave16).

Generic max_dim runs one program per row with tl.where-updated index state
(0.03-0.05x torch: 38.3 ms at 2048^2). The argmax fast paths already produce
first-occurrence indices at the backend's reduce limit; gathering the value at
that index is O(M). max_dim therefore inherits argmax's 3-4x over the generic
kernel.

Ties: argmax keeps the first occurrence, which is what torch.max returns.
NaN: the argmax kernels do not propagate NaN (known gap, same as argmin/argmax).
"""

import os
from collections import namedtuple

import torch

from flag_gems.ops.max import max_dim as _generic_max_dim

from .argmax import argmax as _fast_argmax

_MaxOut = namedtuple("max", ["values", "indices"])


def max_dim(inp, dim=None, keepdim=False):
    if os.environ.get("FLAGGEMS_SPACEMIT_MAXDIM_FAST", "1") == "0":
        return _generic_max_dim(inp, dim, keepdim)
    if dim is None or not isinstance(dim, int):
        return _generic_max_dim(inp, dim, keepdim)
    try:
        idx = _fast_argmax(inp, dim, True)
    except Exception:
        return _generic_max_dim(inp, dim, keepdim)
    val = torch.gather(inp, dim % inp.ndim, idx)
    if not keepdim:
        val = val.squeeze(dim)
        idx = idx.squeeze(dim)
    return _MaxOut(values=val, indices=idx)
