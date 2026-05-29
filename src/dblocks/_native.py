"""Python bindings for the native C extension (libdblocks_native.dylib).

Provides fast, zero-copy access to:
  - macOS mach kernel process memory info (TASK_VM_INFO)
  - System info (total RAM, CPU counts including P-core count)
  - xoshiro256** batch index sampling
  - memcpy-based sequence extraction for dataset batching

Falls back gracefully to pure Python if the .dylib is not found.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path
from typing import Optional

import numpy as np


# -------------------------------------------------------------------------
# Load the shared library
# -------------------------------------------------------------------------

_LIB: Optional[ctypes.CDLL] = None
_LIB_AVAILABLE: bool = False


def _find_library() -> Optional[ctypes.CDLL]:
    """Find and load libdblocks_native.dylib."""
    # Look next to this file (installed via make install)
    candidates = [
        Path(__file__).parent / "libdblocks_native.dylib",
        Path(__file__).parent / "_csrc" / "libdblocks_native.dylib",
    ]
    for path in candidates:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                return lib
            except OSError:
                continue
    return None


def _init():
    global _LIB, _LIB_AVAILABLE
    _LIB = _find_library()
    _LIB_AVAILABLE = _LIB is not None

    if _LIB is None:
        return

    # -- Set up function signatures --

    # Memory info
    class MemInfo(ctypes.Structure):
        _fields_ = [
            ("resident_size", ctypes.c_uint64),
            ("resident_size_peak", ctypes.c_uint64),
            ("virtual_size", ctypes.c_uint64),
            ("internal", ctypes.c_uint64),
            ("compressed", ctypes.c_uint64),
            ("phys_footprint", ctypes.c_uint64),
        ]

    _LIB.dblocks_get_mem_info.argtypes = [ctypes.POINTER(MemInfo)]
    _LIB.dblocks_get_mem_info.restype = ctypes.c_int

    _LIB.dblocks_get_total_memory.argtypes = []
    _LIB.dblocks_get_total_memory.restype = ctypes.c_uint64

    _LIB.dblocks_get_cpu_count.argtypes = []
    _LIB.dblocks_get_cpu_count.restype = ctypes.c_int

    _LIB.dblocks_get_perf_cpu_count.argtypes = []
    _LIB.dblocks_get_perf_cpu_count.restype = ctypes.c_int

    # RNG
    _LIB.dblocks_seed_rng.argtypes = [ctypes.c_uint64]
    _LIB.dblocks_seed_rng.restype = None

    # Batch sampling
    _LIB.dblocks_sample_batch_indices.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int32,
    ]
    _LIB.dblocks_sample_batch_indices.restype = ctypes.c_int

    # Sequence extraction
    _LIB.dblocks_extract_sequences.argtypes = [
        ctypes.POINTER(ctypes.c_int32),  # data
        ctypes.c_int,                     # data_len
        ctypes.POINTER(ctypes.c_int32),  # starts
        ctypes.c_int,                     # batch_size
        ctypes.c_int,                     # seq_len
        ctypes.POINTER(ctypes.c_int32),  # out
    ]
    _LIB.dblocks_extract_sequences.restype = ctypes.c_int

    # Store the MemInfo struct type on the module
    _init.MemInfo = MemInfo


_init()


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def is_available() -> bool:
    """Check if the native C extension is loaded."""
    return _LIB_AVAILABLE


def get_process_memory() -> dict[str, int]:
    """Get detailed process memory info via mach TASK_VM_INFO.

    Returns dict with keys (all in bytes):
      - resident_size: current RSS
      - resident_size_peak: peak RSS (high water mark)
      - virtual_size: virtual address space
      - internal: anonymous/malloc memory
      - compressed: compressed memory
      - phys_footprint: actual physical memory footprint

    Falls back to resource.getrusage if native lib unavailable.
    """
    if _LIB is not None:
        info = _init.MemInfo()
        rc = _LIB.dblocks_get_mem_info(ctypes.byref(info))
        if rc == 0:
            return {
                "resident_size": info.resident_size,
                "resident_size_peak": info.resident_size_peak,
                "virtual_size": info.virtual_size,
                "internal": info.internal,
                "compressed": info.compressed,
                "phys_footprint": info.phys_footprint,
            }

    # Fallback
    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "resident_size": rusage.ru_maxrss,
        "resident_size_peak": rusage.ru_maxrss,
        "virtual_size": 0,
        "internal": 0,
        "compressed": 0,
        "phys_footprint": 0,
    }


def get_system_memory() -> int:
    """Get total physical memory in bytes."""
    if _LIB is not None:
        return _LIB.dblocks_get_total_memory()

    # Fallback via sysctl
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("System"))
        mib = (ctypes.c_int * 2)(6, 24)
        mem = ctypes.c_int64(0)
        size = ctypes.c_size_t(ctypes.sizeof(mem))
        libc.sysctl(mib, 2, ctypes.byref(mem), ctypes.byref(size), None, 0)
        return mem.value
    except Exception:
        return 0


def get_cpu_count() -> int:
    """Get total logical CPU count."""
    if _LIB is not None:
        return _LIB.dblocks_get_cpu_count()
    import os
    return os.cpu_count() or 1


def get_perf_cpu_count() -> int:
    """Get performance (P-core) CPU count on Apple Silicon."""
    if _LIB is not None:
        return _LIB.dblocks_get_perf_cpu_count()
    return get_cpu_count() // 2


def seed_rng(seed: int) -> None:
    """Seed the native xoshiro256** PRNG."""
    if _LIB is not None:
        _LIB.dblocks_seed_rng(ctypes.c_uint64(seed))


def sample_batch_indices(batch_size: int, max_start: int) -> np.ndarray:
    """Generate random batch indices in [0, max_start).

    Uses xoshiro256** PRNG in C — faster than numpy for small batches
    where Python/numpy overhead dominates.

    Falls back to numpy if native lib unavailable.
    """
    if _LIB is not None and batch_size > 0 and max_start > 0:
        out = np.empty(batch_size, dtype=np.int32)
        rc = _LIB.dblocks_sample_batch_indices(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            batch_size,
            max_start,
        )
        if rc == 0:
            return out

    return np.random.randint(0, max_start, size=batch_size, dtype=np.int32)


def extract_sequences(
    data: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """Extract batch of sequences from a flat token array.

    Given `data` (1D int32), `starts` (1D int32 indices), extracts
    `len(starts)` sequences of `seq_len` tokens into a contiguous
    (batch_size, seq_len) array using C memcpy.

    Falls back to numpy fancy indexing if native lib unavailable.
    """
    batch_size = len(starts)

    if _LIB is not None:
        data_c = np.ascontiguousarray(data, dtype=np.int32)
        starts_c = np.ascontiguousarray(starts, dtype=np.int32)
        out = np.empty((batch_size, seq_len), dtype=np.int32)

        rc = _LIB.dblocks_extract_sequences(
            data_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            len(data_c),
            starts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            batch_size,
            seq_len,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if rc == 0:
            return out

    # Fallback: numpy
    return np.stack([data[s:s + seq_len] for s in starts])
