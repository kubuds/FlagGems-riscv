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

"""Nested-dispatch wrappers for the common elementwise op family (K3 campaign, 2026-10-12).

Why: the spine-triton 1D launcher's default "safe" dispatch runs math-heavy 1D
kernels mostly serialized (cpu-time/wall ratio ~2.9, most cores idle); the
nested path (env ``SPINE_TRITON_FORCE_NESTED_DISPATCH_1D=1``) parallelizes them
(~7.6 threads busy).  The common ``pointwise_dynamic`` kernels (tile 512) gain
2.7-4.1x across the board on K3, bit-identical results.

Wrapped ops (A/B interleaved, 4 shapes each, all bitwise-identical): abs, abs_,
absolute, absolute_, add, add_, sub, sub_, subtract, subtract_, neg, neg_,
negative, negative_, gt/lt/eq/ge/le (+scalar+inplace variants), clamp family,
exp, log, sqrt, reciprocal, square, relu, floor, ceil, trunc, div
(true_divide / div_mode / out variants).  round/round_/round_out are NOT
wrapped (measured 0.94x: nested is slightly slower there), nor is
clamp_min/clamp_min_ (neutral 0.90-1.25x; default path already fast).

Two redirects fix ops whose hand-written kernels lose under nested:
absolute/absolute_ run the pointwise abs path, floor_ runs floor_func in-place.
All official-run regression rows (absolute 0.37-0.75x, floor_ 0.59-0.79x) are
fixed by these; switch=0 restores pristine behaviour for all of them.

NOT wrapped (nested is *slower* there on K3): mul family (direct-kernel fast
path in mul.py instead; 0.4-0.9x under nested), sign (0.4-0.5x), plus the
already-wrapped sigmoid/silu/gelu/tanh/rsqrt/pow/softmax/where/mv/norms
(see their modules).

The wrappers call the pristine common implementations
(``flag_gems.ops.*``) under :func:`nested_dispatch`, which is a no-op below
``MIN_N`` elements and can be disabled globally with ``FLAGGEMS_SPACEMIT_NESTED=0``
(used for official-baseline comparison runs).
"""

import os

import torch

from flag_gems.runtime.backend._spacemit.ops._dispatch import nested_dispatch


def _numel(args, kwargs):
    best = 0
    for a in args:
        if isinstance(a, torch.Tensor):
            n = a.numel()
            if n > best:
                best = n
    for a in kwargs.values():
        if isinstance(a, torch.Tensor):
            n = a.numel()
            if n > best:
                best = n
    return best


def _wrap(fn):
    """Wrap a common elementwise impl so its kernel launches use nested dispatch."""

    def wrapper(*args, **kwargs):
        with nested_dispatch(_numel(args, kwargs)):
            return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn
    return wrapper


# --- abs / absolute ---------------------------------------------------------
from flag_gems.ops.abs import abs as _c_abs, abs_ as _c_abs_  # noqa: E402
from flag_gems.ops.absolute import absolute as _c_absolute, absolute_ as _c_absolute_  # noqa: E402

abs = _wrap(_c_abs)
abs_ = _wrap(_c_abs_)


def absolute(A):
    """Route to the pointwise abs path (bit-identical |x|; the hand absolute
    kernel is 3x slower under nested dispatch).  Switch=0 restores the hand impl."""
    if os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") == "0" or not isinstance(
        A, torch.Tensor
    ):
        return _c_absolute(A)
    with nested_dispatch(_numel((A,), {})):
        return _c_abs(A)


def absolute_(A):
    if os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") == "0" or not isinstance(
        A, torch.Tensor
    ):
        return _c_absolute_(A)
    with nested_dispatch(_numel((A,), {})):
        return _c_abs_(A)

# --- add / sub / neg --------------------------------------------------------
from flag_gems.ops.add import add as _c_add, add_ as _c_add_  # noqa: E402
from flag_gems.ops.sub import sub as _c_sub, sub_ as _c_sub_  # noqa: E402
from flag_gems.ops.subtract_ import subtract as _c_subtract, subtract_ as _c_subtract_  # noqa: E402
from flag_gems.ops.neg import neg as _c_neg, neg_ as _c_neg_  # noqa: E402

add = _wrap(_c_add)
add_ = _wrap(_c_add_)
sub = _wrap(_c_sub)
sub_ = _wrap(_c_sub_)
subtract = _wrap(_c_subtract)
subtract_ = _wrap(_c_subtract_)
neg = _wrap(_c_neg)
neg_ = _wrap(_c_neg_)

# --- comparisons ------------------------------------------------------------
from flag_gems.ops.gt import (  # noqa: E402
    gt as _c_gt,
    gt_scalar as _c_gt_scalar,
    gt_scalar_ as _c_gt_scalar_,
    gt_tensor_ as _c_gt_tensor_,
)
from flag_gems.ops.lt import lt as _c_lt, lt_scalar as _c_lt_scalar  # noqa: E402
from flag_gems.ops.lt_ import lt_ as _c_lt_, lt_scalar_ as _c_lt_scalar_  # noqa: E402
from flag_gems.ops.eq import eq as _c_eq, eq_scalar as _c_eq_scalar  # noqa: E402
from flag_gems.ops.eq_ import eq_ as _c_eq_, eq_scalar_ as _c_eq_scalar_  # noqa: E402
from flag_gems.ops.ge import ge as _c_ge, ge_scalar as _c_ge_scalar  # noqa: E402
from flag_gems.ops.le import le as _c_le, le_scalar as _c_le_scalar  # noqa: E402

gt = _wrap(_c_gt)
gt_scalar = _wrap(_c_gt_scalar)
gt_scalar_ = _wrap(_c_gt_scalar_)
gt_tensor_ = _wrap(_c_gt_tensor_)
lt = _wrap(_c_lt)
lt_scalar = _wrap(_c_lt_scalar)
lt_ = _wrap(_c_lt_)
lt_scalar_ = _wrap(_c_lt_scalar_)
eq = _wrap(_c_eq)
eq_scalar = _wrap(_c_eq_scalar)
eq_ = _wrap(_c_eq_)
eq_scalar_ = _wrap(_c_eq_scalar_)
ge = _wrap(_c_ge)
ge_scalar = _wrap(_c_ge_scalar)
le = _wrap(_c_le)
le_scalar = _wrap(_c_le_scalar)

# --- clamp ------------------------------------------------------------------
from flag_gems.ops.clamp import (  # noqa: E402
    clamp as _c_clamp,
    clamp_ as _c_clamp_,
    clamp_tensor as _c_clamp_tensor,
    clamp_tensor_ as _c_clamp_tensor_,
)
from flag_gems.ops.clamp_max import clamp_max as _c_clamp_max, clamp_max_ as _c_clamp_max_  # noqa: E402

# NOTE: clamp_min/clamp_min_ intentionally NOT wrapped: measured ~neutral
# (0.90-1.25x) under nested; their default path is already fast (1.06ms/1M).

clamp = _wrap(_c_clamp)
clamp_ = _wrap(_c_clamp_)
clamp_tensor = _wrap(_c_clamp_tensor)
clamp_tensor_ = _wrap(_c_clamp_tensor_)
clamp_max = _wrap(_c_clamp_max)
clamp_max_ = _wrap(_c_clamp_max_)

# --- elementwise math -------------------------------------------------------
from flag_gems.ops.exp import exp as _c_exp, exp_ as _c_exp_, exp_out as _c_exp_out  # noqa: E402
from flag_gems.ops.log import log as _c_log  # noqa: E402
from flag_gems.ops.log_ import log_ as _c_log_  # noqa: E402
from flag_gems.ops.sqrt import sqrt as _c_sqrt, sqrt_ as _c_sqrt_  # noqa: E402
from flag_gems.ops.reciprocal import (  # noqa: E402
    reciprocal as _c_reciprocal,
    reciprocal_ as _c_reciprocal_,
)
from flag_gems.ops.square import square as _c_square, square_ as _c_square_, square_out as _c_square_out  # noqa: E402
from flag_gems.ops.relu import relu as _c_relu, relu_ as _c_relu_  # noqa: E402

exp = _wrap(_c_exp)
exp_ = _wrap(_c_exp_)
exp_out = _wrap(_c_exp_out)
log = _wrap(_c_log)
log_ = _wrap(_c_log_)
sqrt = _wrap(_c_sqrt)
sqrt_ = _wrap(_c_sqrt_)
reciprocal = _wrap(_c_reciprocal)
reciprocal_ = _wrap(_c_reciprocal_)
square = _wrap(_c_square)
square_ = _wrap(_c_square_)
square_out = _wrap(_c_square_out)
relu = _wrap(_c_relu)
relu_ = _wrap(_c_relu_)

# --- rounding ---------------------------------------------------------------
from flag_gems.ops.floor import (  # noqa: E402
    floor as _c_floor,
    floor_func as _c_floor_func,
    floor_out as _c_floor_out,
)
from flag_gems.ops.floor_ import floor_ as _c_floor_  # noqa: E402
from flag_gems.ops.ceil import ceil as _c_ceil, ceil_ as _c_ceil_, ceil_out as _c_ceil_out  # noqa: E402
from flag_gems.ops.trunc_ import trunc as _c_trunc, trunc_ as _c_trunc_  # noqa: E402

floor = _wrap(_c_floor)
floor_out = _wrap(_c_floor_out)


def floor_(input):
    """Pointwise in-place floor (the hand floor_ kernel loses 0.6-0.8x under
    nested dispatch).  Non-float inputs keep the hand path (integer no-op)."""
    if (
        os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") == "0"
        or not isinstance(input, torch.Tensor)
        or not torch.is_floating_point(input)
    ):
        return _c_floor_(input)
    with nested_dispatch(_numel((input,), {})):
        return _c_floor_func(input, out0=input)


ceil = _wrap(_c_ceil)
ceil_ = _wrap(_c_ceil_)
ceil_out = _wrap(_c_ceil_out)
trunc = _wrap(_c_trunc)
trunc_ = _wrap(_c_trunc_)

# --- negative / negative_ (redirect the slow hand kernel to the pointwise neg)-
from flag_gems.ops.negative import negative as _c_negative  # noqa: E402
from flag_gems.ops.negative_ import negative_ as _c_negative_  # noqa: E402


def negative(A):
    """Pointwise neg path (same bits as the common negative hand kernel, 3.4x faster
    on K3).  With FLAGGEMS_SPACEMIT_NESTED=0 falls back to the hand kernel so
    official comparison runs see pristine behaviour."""
    if os.environ.get("FLAGGEMS_SPACEMIT_NESTED", "1") == "0":
        return _c_negative(A)
    with nested_dispatch(_numel((A,), {})):
        return _c_neg(A)


negative_ = _wrap(_c_negative_)  # internal neg_ call inherits the nested env

# --- div --------------------------------------------------------------------
from flag_gems.ops.div import (  # noqa: E402
    true_divide as _c_true_divide,
    true_divide_ as _c_true_divide_,
    true_divide_out as _c_true_divide_out,
    div_mode as _c_div_mode,
    div_mode_ as _c_div_mode_,
)

true_divide = _wrap(_c_true_divide)
true_divide_ = _wrap(_c_true_divide_)
true_divide_out = _wrap(_c_true_divide_out)
div_mode = _wrap(_c_div_mode)
div_mode_ = _wrap(_c_div_mode_)
