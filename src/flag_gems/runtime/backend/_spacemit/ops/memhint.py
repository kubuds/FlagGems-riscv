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
"""wave26: MADV_HUGEPAGE for multi-GB outputs.

A fresh 4.29 GB f32 output costs 1.22 s more than a warm one on this board
(2.32 s vs 1.10 s per softmax call at 1024^3, probe260): the block is handed
back to the OS when the caller drops it, so the writing kernel faults in every
4 KB page again.  With MADV_HUGEPAGE on the 2 MB-aligned span the same
allocation costs 0.22 s instead (2.32 -> 1.32 s per call, probe263: 1.75x at
f32, 1.65x at f16; the earlier fill-only probe measured 1.43 -> 0.45 s).

Below a few hundred MB the allocator reuses the block, so there is nothing to
win and the hint is skipped (67 MB shapes measured 17.13 vs 17.10 ms).
"""

import ctypes
import os

_MADV_HUGEPAGE = 14
_HUGE = 2 << 20

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_madvise = _libc.madvise
_madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_madvise.restype = ctypes.c_int


def hugepage_min_bytes() -> int:
    try:
        mb = int(os.environ.get("FLAGGEMS_SPACEMIT_HUGEPAGE_MIN_MB", "256"))
    except ValueError:
        mb = 256
    return max(0, mb) << 20


def hugepage_hint(t) -> None:
    """Best-effort MADV_HUGEPAGE over the 2 MB-aligned part of t's buffer."""
    if t is None or t.numel() == 0 or not t.is_contiguous():
        return
    nbytes = t.numel() * t.element_size()
    if nbytes < hugepage_min_bytes():
        return
    ptr = t.data_ptr()
    start = (ptr + _HUGE - 1) & ~(_HUGE - 1)
    end = (ptr + nbytes) & ~(_HUGE - 1)
    if end <= start:
        return
    # Failure is not an error: the hint only changes how the kernel backs pages.
    _madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), _MADV_HUGEPAGE)
