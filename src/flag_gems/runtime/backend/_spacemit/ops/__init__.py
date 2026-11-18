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

# from .addmm import addmm
from .amax import amax
from .amin import amin
from .argmax import argmax
from .argmin import argmin
from .bmm import bmm, bmm_out
from .copy import copy_

# from .conv1d import conv1d
# from .conv2d import conv2d
# from .thnn_conv2d import thnn_conv2d
# from .conv_depthwise2d import _conv_depthwise2d
# from .flash_attention import flash_attention, scaled_dot_product_attention
from .gelu import gelu, gelu_, gelu_backward
from .hardsigmoid import hardsigmoid, hardsigmoid_, hardsigmoid_out
from .layernorm import layer_norm
from .max import max_dim
from .mean import global_avg_pool, mean, mean_dim
from .mm import mm, mm_out
from .min import min_dim
from .mv import mv
from .native_layer_norm import layer_norm_backward, native_layer_norm
from .log_softmax import log_softmax, log_softmax_out
from .prod import prod, prod_dim
from .logsumexp import logsumexp
from .norm import norm, norm_scalar, norm_scalaropt_dim
from .vector_norm import vector_norm
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .rsqrt import rsqrt, rsqrt_
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .silu import silu, silu_, silu_backward
from .tanh import tanh, tanh_, tanh_backward
from .softmax import softmax, softmax_backward, softmax_backward_out, softmax_out
from .where import where_scalar_other, where_scalar_self, where_self, where_self_out

# Nested-dispatch wrappers for the common elementwise family (campaign 1012pm).
from .elemwise_nested import (
    abs,
    abs_,
    absolute,
    absolute_,
    add,
    add_,
    sub,
    sub_,
    subtract,
    subtract_,
    neg,
    neg_,
    gt,
    gt_scalar,
    gt_scalar_,
    gt_tensor_,
    lt,
    lt_scalar,
    lt_,
    lt_scalar_,
    eq,
    eq_scalar,
    eq_,
    eq_scalar_,
    ge,
    ge_scalar,
    le,
    le_scalar,
    le_scalar_,
    ne_scalar_,
    less_scalar_,
    less_equal_scalar_,
    not_equal_scalar_,
    clamp,
    clamp_,
    clamp_tensor,
    clamp_tensor_,
    clamp_max,
    clamp_max_,
    exp,
    exp_,
    exp_out,
    log,
    log_,
    sqrt,
    sqrt_,
    reciprocal,
    reciprocal_,
    square,
    square_,
    square_out,
    relu,
    relu_,
    negative,
    negative_,
    floor,
    floor_,
    floor_out,
    ceil,
    ceil_,
    ceil_out,
    trunc,
    trunc_,
    true_divide,
    true_divide_,
    true_divide_out,
    div_mode,
    div_mode_,
    elu,
    elu_,
    elu_backward,
)

# Direct-kernel mul fast path (campaign 1012pm), incl. multiply aliases.
from .mul import mul, mul_, multiply, multiply_

# Direct-kernel unary fast paths (campaign 1012night, wave2). These load AFTER
# elemwise_nested so the fast versions win for the shared names; their
# fallback (small / non-contiguous / other dtypes) calls the elemwise_nested
# wrappers, i.e. the previously shipped behaviour.
from .unary_fast import (
    abs,
    abs_,
    absolute,
    absolute_,
    neg,
    neg_,
    negative,
    negative_,
    relu,
    relu_,
    square,
    square_,
    square_out,
    trunc,
    trunc_,
    sign,
    sign_out,
    elu,
    elu_,
)

# Direct-kernel binary fast paths (campaign 1012night, wave2b).
from .binary_fast import (
    add,
    add_,
    sub,
    sub_,
    subtract,
    subtract_,
    gt,
    lt,
    eq,
    ge,
    le,
    le_,
    ne,
    ne_scalar,
    ne_,
    gt_scalar,
    lt_scalar,
    eq_scalar,
    ge_scalar,
    le_scalar,
    gt_tensor_,
    lt_,
    eq_,
    maximum,
    minimum,
    less_,
    less_equal,
    less_equal_scalar,
    less_equal_,
    greater,
    greater_scalar,
    greater_out,
    greater_scalar_out,
    greater_equal_,
    not_equal,
    not_equal_scalar,
    not_equal_,
    clamp,
    clamp_,
    clamp_max,
    clamp_max_,
    clamp_min,
    clamp_min_,
    clamp_tensor,
    clamp_tensor_,
    where_self,
    where_self_out,
)

# Direct-kernel log-family fast paths (campaign 1013pm). Imported LAST so the
# poly-log2 kernels win over the elemwise_nested wrap for the shared names
# (log, log_).
from .log_fast import log, log10, log10_, log10_out, log2, log2_, log_

# sum family fast paths (campaign 1015am, wave6): raised-BLOCK flat
# sum + multi-row sum_dim; see sum_fast.py for measured numbers.
from .sum import sum, sum_dim, sum_dim_out, sum_out

# rms_norm fast paths: forward (campaign 1015am, wave6) and backward
# (campaign 1016am, wave9) direct kernels. See rmsnorm_fast.py /
# rmsnorm_bwd_fast.py.
from .rms_norm import rms_norm

# var/std/var_mean fast paths + aten.var.dim positional-signature fix
# (campaign 1015pm, wave7). See var.py.
from .var import std, var, var_correction, var_dim, var_mean

__all__ = [
    "_conv_depthwise2d",
    "addmm",
    "amax",
    "amin",
    "argmax",
    "argmin",
    "bmm",
    "bmm_out",
    "conv1d",
    "conv2d",
    "copy_",
    "flash_attention",
    "gelu",
    "gelu_",
    "gelu_backward",
    "hardsigmoid",
    "hardsigmoid_",
    "hardsigmoid_out",
    "global_avg_pool",
    "layer_norm",
    "mean",
    "mean_dim",
    "max_dim",
    "log_softmax",
    "log_softmax_out",
    "prod",
    "prod_dim",
    "logsumexp",
    "norm",
    "norm_scalar",
    "norm_scalaropt_dim",
    "vector_norm",
    "mm",
    "min_dim",
    "mm_out",
    "mv",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "rsqrt",
    "rsqrt_",
    "scaled_dot_product_attention",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "silu",
    "silu_",
    "silu_backward",
    "tanh",
    "tanh_",
    "tanh_backward",
    "softmax",
    "softmax_out",
    "softmax_backward",
    "softmax_backward_out",
    "native_layer_norm",
    "layer_norm_backward",
    "thnn_conv2d",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
    "abs",
    "abs_",
    "absolute",
    "absolute_",
    "add",
    "add_",
    "sub",
    "sub_",
    "subtract",
    "subtract_",
    "neg",
    "neg_",
    "gt",
    "gt_scalar",
    "gt_scalar_",
    "gt_tensor_",
    "lt",
    "lt_scalar",
    "lt_",
    "lt_scalar_",
    "eq",
    "eq_scalar",
    "eq_",
    "eq_scalar_",
    "ge",
    "ge_scalar",
    "le",
    "le_scalar",
    "le_",
    "le_scalar_",
    "ne",
    "ne_scalar",
    "ne_",
    "ne_scalar_",
    "maximum",
    "minimum",
    "less_",
    "less_scalar_",
    "less_equal",
    "less_equal_scalar",
    "less_equal_",
    "less_equal_scalar_",
    "greater",
    "greater_scalar",
    "greater_out",
    "greater_scalar_out",
    "greater_equal_",
    "not_equal",
    "not_equal_scalar",
    "not_equal_",
    "not_equal_scalar_",
    "clamp",
    "clamp_",
    "clamp_tensor",
    "clamp_tensor_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "exp",
    "exp_",
    "exp_out",
    "log",
    "log_",
    "log2",
    "log2_",
    "log10",
    "log10_",
    "log10_out",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "std",
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    "rms_norm",
    "sqrt",
    "sqrt_",
    "reciprocal",
    "reciprocal_",
    "square",
    "square_",
    "square_out",
    "relu",
    "relu_",
    "negative",
    "negative_",
    "floor",
    "floor_",
    "floor_out",
    "ceil",
    "ceil_",
    "ceil_out",
    "trunc",
    "trunc_",
    "sign",
    "sign_out",
    "elu",
    "elu_",
    "elu_backward",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "div_mode",
    "div_mode_",
    "mul",
    "mul_",
    "multiply",
    "multiply_",
]
