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

"""wave21: shared fast-math primitives for the _spacemit backend.

exp is the most expensive primitive in this backend's reduction/softmax family:
tl.exp lowers to a scalar libm call, measured at ~1.45 ns/element inside the
mid-dim logsumexp kernel (probe183: 1024x256x128 load+sum 12.6 ms versus
load+exp 61.3 ms, i.e. 33.5M exps cost ~48.7 ms).

`exp_fast` computes exp(y) = 2^(y*log2e) as a degree-6 polynomial plus an
integer-bitcast power-of-two scale. Measured 1.15-1.38x faster than tl.exp and
within ~1.5e-5 relative (max abs 1-2e-4 on N(0,2) sums); degree 7 is ~1.3e-6
but loses up to 20% on some row shapes, so degree 6 is the default.

NaN/inf semantics are kept: the small-argument clamp is a comparison plus
select (not fmax), so NaN stays NaN and -inf underflows to 2^-126 ~= 0, the
same way libm exp does. The one difference from libm is that exp(-inf) is
denormal-free (exactly 2^-126 rather than 0), which is below every tolerance
in use.

Kill switch: FLAGGEMS_SPACEMIT_FASTMATH_EXP=0 makes exp_fast call tl.exp.
"""

import os

import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


@triton.jit
def exp_fast(y, FAST: tl.constexpr):
    """exp(y) with a bitcast-assembled 2^n scale; FAST=0 falls back to libm."""
    if FAST == 0:
        return tl.exp(y)
    yc = y * 1.4426950408889634
    # select, not fmax: -inf -> 2^-126 (~0), NaN stays NaN
    yc = tl.where(yc < -126.0, -126.0, yc)
    n = tl.floor(yc)
    f = yc - n
    p = 1.535336188319500e-4
    p = p * f + 1.339887440266574e-3
    p = p * f + 9.618437357674640e-3
    p = p * f + 5.550332471162809e-2
    p = p * f + 2.402264791363012e-1
    p = p * f + 6.931472028550421e-1
    p = p * f + 1.0
    ni = n.to(tl.int32)
    ni = tl.where(ni > 127, 127, ni)
    scale = ((ni + 127) << 23).to(tl.float32, bitcast=True)
    return p * scale


def fastmath_exp_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_FASTMATH_EXP", "1") != "0"


def exp_flag() -> int:
    """constexpr FAST argument for exp_fast at launch time."""
    return 1 if fastmath_exp_enabled() else 0

# wave23: reciprocal without the slow scalar division.
# probe205 (1M f32 standalone): 1.0/x 1.60 ns/elem; bit-trick seed + 2 Newton
# 0.51 ns/elem with max relative error 6.7e-6 (3 Newton: 0.55 ns, 1.8e-7).
# The clamp is a select, not fmax, so NaN propagates; exact zero still yields
# +inf (a zero in a negative-order norm poisons the row to 0, as torch does)
# and +inf yields 0 (masked lanes load +inf).
@triton.jit
def recip_fast(x, FAST: tl.constexpr):
    if FAST == 0:
        return 1.0 / x
    xc = tl.where(x < 1.0e-38, 1.0e-38, x)
    r = (0x7EF311C3 - xc.to(tl.int32, bitcast=True)).to(tl.float32, bitcast=True)
    r = r * (2.0 - xc * r)
    r = r * (2.0 - xc * r)
    r = tl.where(x == 0.0, float("inf"), r)
    r = tl.where(x == float("inf"), 0.0, r)
    return r


def fastmath_recip_enabled() -> bool:
    return os.environ.get("FLAGGEMS_SPACEMIT_FASTMATH_RECIP", "1") != "0"


def recip_flag() -> int:
    """constexpr FAST argument for recip_fast at launch time."""
    return 1 if fastmath_recip_enabled() else 0
