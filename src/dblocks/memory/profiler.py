"""Apple Silicon memory profiler.

Uses the native C extension (libdblocks_native.dylib) for accurate
mach kernel memory reporting via TASK_VM_INFO.  Falls back to
resource/psutil if the native extension is unavailable.

MLX Metal memory stats are reported via MLX APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .._native import (
    get_process_memory,
    get_system_memory,
    is_available as native_available,
)


# --------------------------------------------------------------------------
# MLX memory reporting (new mx.* API with mx.metal.* fallback)
# --------------------------------------------------------------------------

def _get_metal_memory() -> dict[str, int]:
    """Get MLX Metal memory stats."""
    result = {"active": 0, "peak": 0, "cache": 0}
    for key in result:
        fn_name = f"get_{key}_memory"
        try:
            result[key] = getattr(mx, fn_name)()
        except (AttributeError, Exception):
            try:
                result[key] = getattr(mx.metal, fn_name)()
            except Exception:
                pass
    return result


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """A snapshot of system and GPU memory usage."""

    # System
    system_total_mb: float
    process_rss_mb: float
    process_rss_peak_mb: float
    process_virtual_mb: float
    process_phys_footprint_mb: float

    # MLX Metal GPU
    metal_active_mb: float
    metal_peak_mb: float
    metal_cache_mb: float

    # Backend info
    native_backend: bool

    def __str__(self) -> str:
        backend = "C/mach" if self.native_backend else "Python/resource"
        lines = [
            f"System RAM:     {self.system_total_mb:,.0f} MB total",
            f"Process RSS:    {self.process_rss_mb:,.1f} MB (peak {self.process_rss_peak_mb:,.1f} MB)",
            f"Phys footprint: {self.process_phys_footprint_mb:,.1f} MB",
            f"Metal active:   {self.metal_active_mb:,.1f} MB",
            f"Metal peak:     {self.metal_peak_mb:,.1f} MB",
            f"Metal cache:    {self.metal_cache_mb:,.1f} MB",
            f"Backend:        {backend}",
        ]
        return "\n".join(lines)


class MemoryProfiler:
    """Apple Silicon memory profiler.

    Uses native C extension for mach kernel APIs when available,
    falling back to Python resource/psutil otherwise.

    Example
    -------
    >>> prof = MemoryProfiler()
    >>> snap = prof.snapshot()
    >>> print(snap)
    """

    def __init__(self) -> None:
        self._system_total = get_system_memory()

    @property
    def system_total_mb(self) -> float:
        return self._system_total / (1024 ** 2)

    @property
    def using_native(self) -> bool:
        """Whether the native C extension is active."""
        return native_available()

    def snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        mem = get_process_memory()
        metal = _get_metal_memory()
        mb = 1024 ** 2
        return MemorySnapshot(
            system_total_mb=self._system_total / mb,
            process_rss_mb=mem["resident_size"] / mb,
            process_rss_peak_mb=mem["resident_size_peak"] / mb,
            process_virtual_mb=mem["virtual_size"] / mb,
            process_phys_footprint_mb=mem["phys_footprint"] / mb,
            metal_active_mb=metal["active"] / mb,
            metal_peak_mb=metal["peak"] / mb,
            metal_cache_mb=metal["cache"] / mb,
            native_backend=native_available(),
        )

    def reset_peak(self) -> None:
        """Reset the MLX Metal peak memory counter."""
        try:
            mx.reset_peak_memory()
        except (AttributeError, Exception):
            try:
                mx.metal.reset_peak_memory()
            except Exception:
                pass

    def set_memory_limit(self, limit_gb: float) -> None:
        """Set MLX Metal memory limit."""
        try:
            mx.set_memory_limit(int(limit_gb * 1024 ** 3))
        except (AttributeError, Exception):
            try:
                mx.metal.set_memory_limit(int(limit_gb * 1024 ** 3))
            except Exception:
                pass

    def set_cache_limit(self, limit_mb: float) -> None:
        """Set MLX Metal cache limit (lower = more aggressive freeing)."""
        try:
            mx.set_cache_limit(int(limit_mb * 1024 ** 2))
        except (AttributeError, Exception):
            try:
                mx.metal.set_cache_limit(int(limit_mb * 1024 ** 2))
            except Exception:
                pass


def memory_summary() -> str:
    """Quick one-line memory summary."""
    prof = MemoryProfiler()
    snap = prof.snapshot()
    native = " [C]" if snap.native_backend else " [py]"
    return (
        f"RSS={snap.process_rss_mb:.0f}MB "
        f"Phys={snap.process_phys_footprint_mb:.0f}MB "
        f"Metal={snap.metal_active_mb:.0f}MB/{snap.metal_peak_mb:.0f}MB(peak) "
        f"System={snap.system_total_mb:.0f}MB{native}"
    )
