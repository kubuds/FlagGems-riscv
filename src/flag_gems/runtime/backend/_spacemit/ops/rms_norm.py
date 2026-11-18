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

"""spacemit vendor override: rms_norm (wave6).

Only the forward is accelerated (f32, contiguous, N in {512, 2048, 4096});
the backward keeps using the generic kernels unchanged (autograd Function
subclass, backward inherited). Anything not covered falls back to the generic
forward.

Kill switch: FLAGGEMS_SPACEMIT_RMS_FAST=0.
"""

import logging
import math

import torch

from flag_gems.ops.rms_norm import RmsNorm as _BaseRmsNorm
from flag_gems.ops.rms_norm import rms_norm_forward as _orig_forward
from flag_gems.runtime import torch_device_fn

from .rmsnorm_fast import rms_fast_enabled, rms_fast_launch

logger = logging.getLogger(__name__)


def _forward_fast(x, normalized_shape, weight, eps):
    if not rms_fast_enabled():
        return None
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        return None
    if not x.is_contiguous() or not weight.is_contiguous():
        return None
    N = math.prod(normalized_shape)
    if N <= 0 or x.numel() % N:
        return None
    M = x.numel() // N
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)
    with torch_device_fn.device(x.device):
        if not rms_fast_launch(x, weight, y, inv_rms, M, N, eps):
            return None
    return y, inv_rms


class RmsNorm(_BaseRmsNorm):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS_SPACEMIT RMS_NORM FORWARD (fast override)")
        res = _forward_fast(x, normalized_shape, weight, eps)
        if res is None:
            y, inv_rms = _orig_forward(x, normalized_shape, weight, eps=eps)
        else:
            y, inv_rms = res
        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
