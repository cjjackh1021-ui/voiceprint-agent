"""Dimension 5: end-to-end and per-step latency, embedder cache compute stats, and memory usage."""
import ctypes
import sys

import numpy as np

try:
    import resource
except ImportError:
    resource = None


def percentile_stats(values_ms: list[float]) -> dict:
    """Summarize millisecond latencies with count, mean, p50, p95, and total."""
    arr = np.asarray(list(values_ms), dtype=np.float64)
    if arr.size == 0:
        return {'n': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'total_ms': 0.0}
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError('latency values must be finite numbers')
    return {
        'n': int(arr.size),
        'mean_ms': float(np.mean(arr)),
        'p50_ms': float(np.quantile(arr, 0.50)),
        'p95_ms': float(np.quantile(arr, 0.95)),
        'total_ms': float(np.sum(arr)),
    }


def _windows_rss_mb() -> float | None:
    """Best-effort WorkingSetSize query on Windows without third-party dependencies."""
    if sys.platform != 'win32':
        return None
    try:
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_memory_info = psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
        get_memory_info.restype = wintypes.BOOL

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = get_current_process()
        ok = get_memory_info(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return None


def current_rss_mb() -> float | None:
    """Return process resident/max RSS in MiB across macOS, Linux, and Windows."""
    if resource is not None:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rss <= 0.0:
            return None
        if sys.platform == 'darwin':
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    return _windows_rss_mb()


def summarize_perf(
    case_timings_ms: list[float],
    step_timings_ms: dict[str, list[float]],
    cache_stats: dict | None,
    peak_tracemalloc_mb: float,
    rss_mb: float | None,
) -> dict:
    """Aggregate case, step, cold embedding compute, cache hit, and memory metrics."""
    case_stats = percentile_stats(case_timings_ms)
    steps = {
        str(step): percentile_stats(list(vals))
        for step, vals in sorted((step_timings_ms or {}).items())
    }
    if cache_stats is None:
        embed_compute = percentile_stats([])
        embed_lookup = percentile_stats([])
        cache_summary = {'enabled': False, 'hits': 0, 'misses': 0, 'total': 0, 'hit_rate': 0.0}
    else:
        compute_list = list(cache_stats.get('compute_ms_list') or [])
        lookup_list = list(cache_stats.get('lookup_ms_list') or [])
        embed_compute = percentile_stats(compute_list)
        embed_lookup = percentile_stats(lookup_list)
        hits = int(cache_stats.get('hits', 0))
        misses = int(cache_stats.get('misses', 0))
        total = int(cache_stats.get('total', hits + misses))
        hit_rate = float(cache_stats.get('hit_rate', (hits / total) if total else 0.0))
        cache_summary = {
            'enabled': bool(cache_stats.get('enabled', True)),
            'hits': hits,
            'misses': misses,
            'total': total,
            'hit_rate': hit_rate,
        }

    return {
        'case': case_stats,
        'total_seconds': float(case_stats['total_ms'] / 1000.0),
        'steps': steps,
        'embed_compute': embed_compute,
        'embed_lookup': embed_lookup,
        'cache': cache_summary,
        'peak_tracemalloc_mb': float(peak_tracemalloc_mb),
        'rss_mb': float(rss_mb) if rss_mb is not None else None,
    }
