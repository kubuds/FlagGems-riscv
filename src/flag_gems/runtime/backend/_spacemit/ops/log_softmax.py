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

"""vendor log_softmax: last-dim row fast path (wave17); everything else generic.

wave27: f16/bf16 join the fast path (they used to fall through to the generic
kernel: 26 ms vs 2.5 ms at 1024^2), and the output gets the multi-GB
MADV_HUGEPAGE hint softmax already had.  `half_to_float` is honoured by
allocating an f32 output; the kernels cast on store either way.
"""

import torch

from flag_gems.ops.log_softmax import log_softmax as _generic_log_softmax
from flag_gems.ops.log_softmax import log_softmax_out as _generic_log_softmax_out

from .logsoftmax_fast import logsm_fast_enabled, logsm_launch
from .memhint import hugepage_hint

_FAST_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def log_softmax(self, dim, half_to_float=False):
    if (
        logsm_fast_enabled()
        and self.is_contiguous()
        and self.ndim >= 2
        and self.numel() > 0
        and self.dtype in _FAST_DTYPES
    ):
        d = dim % self.ndim
        if d == self.ndim - 1:
            N = self.shape[-1]
            M = self.numel() // N
            dtype = torch.float32 if half_to_float else self.dtype
            out = torch.empty_like(self, dtype=dtype)
            hugepage_hint(out)
            if logsm_launch(self, out, M, N):
                return out
    return _generic_log_softmax(self, dim, half_to_float)


def log_softmax_out(self, dim, half_to_float=False, *, out):
    if (
        logsm_fast_enabled()
        and self.is_contiguous()
        and self.ndim >= 2
        and self.numel() > 0
        and self.dtype in _FAST_DTYPES
        and out.dtype == (torch.float32 if half_to_float else self.dtype)
        and out.is_contiguous()
        and len(out.shape) == len(self.shape)
    ):
        d = dim % self.ndim
        if d == self.ndim - 1 and out.shape[-1] == self.shape[-1]:
            N = self.shape[-1]
            M = self.numel() // N
            hugepage_hint(out)
            if logsm_launch(self, out, M, N):
                return out
    return _generic_log_softmax_out(self, dim, half_to_float, out=out)
