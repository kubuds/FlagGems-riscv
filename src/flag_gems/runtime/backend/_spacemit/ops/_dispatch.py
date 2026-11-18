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

"""Nested (multi-stream) dispatch control for the K3 campaign.

The spine-triton CPU launcher has two dispatch paths for 1D-grid kernels:

- "safe" default: the whole grid is handed to ``spineMultiStreamDispatch`` as
  one job. Measured on K3, this parallelises weakly (cpu-time/wall ratio ~3)
  and leaves most of the 8 CPUs idle.
- nested (``SPINE_TRITON_FORCE_NESTED_DISPATCH_1D=1``): the grid is split into
  batches of ``stream_threads`` (4) sub-blocks per outer multi-stream block,
  which uses ~7.6 CPUs and is 2.2-2.8x faster for math-heavy kernels.
  Memory-bound kernels (copy_, hardsigmoid, mean, ...) are *slower* under the
  nested path, so only the ops that measured wins wrap their launches.

Campaign 1012am measurements (1M fp32, interleaved A/B, bitwise-identical):
sigmoid 3.34 -> 1.48 ms; silu 3.33 -> 1.49; gelu 5.78 -> 2.14; tanh 3.38 ->
1.49; rsqrt 3.29 -> 1.37; pow 5.74 -> 2.74; softmax 2048^2 25.5 -> 9.2;
layer_norm 2048^2 25.6 -> 14.2; where 1M 15.9 -> 4.7; mv 2048 16.4 -> 6.9.
Below MIN_N elements the fixed dispatch overhead (~13 us) can dominate, so
small launches keep the default path.

Set FLAGGEMS_SPACEMIT_NESTED=0 to disable this behaviour globally (debug).
"""

import os
from contextlib import contextmanager

_KEY = "SPINE_TRITON_FORCE_NESTED_DISPATCH_1D"
_MIN_N = 1 << 15


@contextmanager
def nested_dispatch(n):
    """Enable the nested dispatch path for one kernel launch (n = numel)."""
    if n < _MIN_N or os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") == "0":
        yield
        return
    old = os.environ.get(_KEY)
    os.environ[_KEY] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(_KEY, None)
        else:
            os.environ[_KEY] = old
