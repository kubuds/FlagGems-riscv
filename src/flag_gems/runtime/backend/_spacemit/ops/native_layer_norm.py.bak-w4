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

"""Vendor wrappers for the common layer-norm implementations.

campaign 1012am: the aten path for F.layer_norm / aten.native_layer_norm is the
common implementation in flag_gems.ops.layernorm (row-per-block kernels with 1D
grids). Under the default 1D dispatch those kernels run mostly serialised
(~25.7 ms at 2048^2); wrapping the call with nested dispatch brings it to
~14.5 ms (+43%, bit-identical outputs).

The wrappers keep the common signatures; only the dispatch toggle is local to
the call.
"""

import logging

from flag_gems.ops.layernorm import layer_norm_backward as _common_ln_backward
from flag_gems.ops.native_layer_norm import (
    native_layer_norm as _common_native_layer_norm,
)

from flag_gems.runtime.backend._spacemit.ops._dispatch import nested_dispatch

logger = logging.getLogger(__name__)


def native_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS_SPACEMIT NATIVE_LAYER_NORM")
    with nested_dispatch(input.numel()):
        return _common_native_layer_norm(input, normalized_shape, weight, bias, eps)


def layer_norm_backward(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight=None,
    bias=None,
    output_mask=None,
):
    logger.debug("GEMS_SPACEMIT LAYER_NORM_BACKWARD")
    with nested_dispatch(input.numel()):
        return _common_ln_backward(
            grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
        )
