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

"""wave23: keep large operator outputs out of the mmap/munmap churn.

glibc caps its dynamic mmap threshold at 32 MB, so every output buffer above
that goes to mmap and is munmap'd on free.  The next operator call then pays
first-touch page faults again: on K3 a fresh 67 MB f32 buffer costs 26.3 ms to
allocate+fill versus 3.3 ms for a warm one (probe209), and that 23 ms is
exactly the gap between softmax at the operator entry (53.0 ms) and the same
kernel with a preallocated output (23.8 ms) at 512x32768.

Raising M_MMAP_THRESHOLD/M_TRIM_THRESHOLD makes malloc keep those blocks:
fresh alloc+fill 26.6 -> 3.9 ms, softmax entry 53.0 -> 24.1 ms, 4096^2 entry
28.9 ms against a 28.4 ms preallocated kernel (probe210).

Trade-off: the heap holds the high-water mark instead of returning it to the
OS, so RSS stays at the peak.  The threshold is capped at 1 GiB on purpose:
larger tensors still go through mmap so a 4 GB activation cannot pin 4 GB of
heap.  Set FLAGGEMS_SPACEMIT_MALLOC_TUNE=0 to restore the default allocator.
"""

import os

_MMAP_THRESHOLD = 1 << 30
_TRIM_THRESHOLD = 1 << 30
_M_MMAP_THRESHOLD = -3
_M_TRIM_THRESHOLD = -1

_applied = False


def malloc_tune_applied() -> bool:
    return _applied


def apply_malloc_tune() -> bool:
    """Idempotent; returns True when the tunables are in effect."""
    global _applied
    if _applied:
        return True
    if os.environ.get("FLAGGEMS_SPACEMIT_MALLOC_TUNE", "1") == "0":
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        ok = libc.mallopt(_M_MMAP_THRESHOLD, _MMAP_THRESHOLD)
        ok2 = libc.mallopt(_M_TRIM_THRESHOLD, _TRIM_THRESHOLD)
        _applied = bool(ok) and bool(ok2)
    except Exception:
        _applied = False
    return _applied
