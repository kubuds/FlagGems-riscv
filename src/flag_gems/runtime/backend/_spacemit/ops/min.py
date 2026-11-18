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

"""vendor min_dim: reuse the argmin row/dim0 kernels + a value gather (wave16).

Same structure as vendor max_dim; generic min_dim measured 0.03-0.05x torch
(38.3 ms at 2048^2) because of the index-carrying scalar reduce.
"""

import os
from collections import namedtuple

import torch

from flag_gems.ops.min import min_dim as _generic_min_dim

from .argmin import argmin as _fast_argmin

_MinOut = namedtuple("min", ["values", "indices"])


def min_dim(inp, dim=None, keepdim=False):
    if os.environ.get("FLAGGEMS_SPACEMIT_MINDIM_FAST", "1") == "0":
        return _generic_min_dim(inp, dim, keepdim)
    if dim is None or not isinstance(dim, int):
        return _generic_min_dim(inp, dim, keepdim)
    try:
        idx = _fast_argmin(inp, dim, True)
    except Exception:
        return _generic_min_dim(inp, dim, keepdim)
    val = torch.gather(inp, dim % inp.ndim, idx)
    if not keepdim:
        val = val.squeeze(dim)
        idx = idx.squeeze(dim)
    return _MinOut(values=val, indices=idx)
