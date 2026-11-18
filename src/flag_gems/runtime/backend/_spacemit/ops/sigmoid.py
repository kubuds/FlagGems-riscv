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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

from flag_gems.runtime.backend._spacemit.ops._dispatch import nested_dispatch

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp

# Direct-kernel fast path for floating dtypes (campaign 1011night). The generic
# pointwise wrapper costs an extra 1.9-6.6 ms per call on K3 (1M: 5.15 -> 3.26 ms;
# 2048x2048: 19.62 -> 13.06 ms) because its generated kernel lowers worse.
# Numerically equivalent to the wrapper (same formula; maxdiff 1.19e-07 vs torch).
# campaign 1012am: mask-free variant for n % BLOCK == 0 (+7.8% f32, +3.6% f16
# interleaved A/B, bit-identical) + nested dispatch (2.3x).
_BLOCK = 8192


@libentry()
@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    y = 1.0 / (1.0 + exp(-x.to(tl.float32)))
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty), mask=m)


@libentry()
@triton.jit
def sigmoid_kernel_nomask(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = 1.0 / (1.0 + exp(-x.to(tl.float32)))
    tl.store(y_ptr + offs, y.to(y_ptr.type.element_ty))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_forward(x):
    # log2e: tl.constexpr = math.log2(math.e)
    # triton 3.0.0 disallow calling non-jitted function inside jitted function, even if it is in
    # the rhs of an assignment to a constexpr, so we use numeric literal instead to work around this.
    return 1 / (1 + exp(-x.to(tl.float32)))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward_kernel(dy, y):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


def _direct_ok(t):
    return (
        t.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and t.is_contiguous()
    )


def _launch_direct(x, out):
    n = x.numel()
    if n == 0:
        return out
    with torch_device_fn.device(x.device):
        with nested_dispatch(n):
            if n % _BLOCK == 0:
                sigmoid_kernel_nomask[(n // _BLOCK,)](x, out, n, BLOCK=_BLOCK)
            else:
                sigmoid_kernel[(triton.cdiv(n, _BLOCK),)](x, out, n, BLOCK=_BLOCK)
    return out


def sigmoid(self):
    logger.debug("GEMS_SPACEMIT SIGMOID")
    if _direct_ok(self):
        out = torch.empty_like(self)
        return _launch_direct(self, out)
    output = sigmoid_forward(self)
    return output


def sigmoid_backward(grad_output, output):
    logger.debug("GEMS_SPACEMIT SIGMOID_BACKWARD")
    with nested_dispatch(grad_output.numel()):
        grad_input = sigmoid_backward_kernel(grad_output, output)
    return grad_input


def sigmoid_(A):
    logger.debug("GEMS_SPACEMIT SIGMOID_")
    if _direct_ok(A):
        return _launch_direct(A, A)
    out = sigmoid_forward(A, out0=A)
    return out
