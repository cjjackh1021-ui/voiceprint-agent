"""Dimension 5: end-to-end and per-step latency, embedder cache compute stats, Warm RTF, and memory usage."""
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

from voiceprint.audio import duration_seconds

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


def _rtf_from_pairs(
    compute_ms_list: list[float],
    audio_s_list: list[float],
    *,
    source: str,
) -> dict:
    n = min(len(compute_ms_list), len(audio_s_list))
    if n <= 0:
        return {
            'source': source,
            'n_samples': 0,
            'cold_embed_ms': None,
            'warm_embed': percentile_stats([]),
            'warm_fallback': False,
            'cold_rtf': None,
            'warm_rtf': None,
            'p50_warm_rtf': None,
            'p95_warm_rtf': None,
            'total_audio_seconds': 0.0,
            'warm_audio_seconds': 0.0,
        }

    ms_vals = [float(x) for x in compute_ms_list[:n]]
    dur_vals = [float(x) for x in audio_s_list[:n]]
    cold_ms = ms_vals[0]
    cold_dur = dur_vals[0]
    cold_rtf = (cold_ms / 1000.0) / cold_dur if cold_dur > 0.0 else None

    if n >= 2:
        warm_ms = ms_vals[1:]
        warm_durs = dur_vals[1:]
        warm_fallback = False
    else:
        warm_ms = ms_vals
        warm_durs = dur_vals
        warm_fallback = True

    warm_audio_s = float(sum(warm_durs))
    warm_compute_s = float(sum(warm_ms) / 1000.0)
    warm_rtf = (warm_compute_s / warm_audio_s) if warm_audio_s > 0.0 else None
    per_item_rtfs = [
        (m / 1000.0) / d for m, d in zip(warm_ms, warm_durs) if d > 0.0
    ]
    rtf_arr = np.asarray(per_item_rtfs, dtype=np.float64)
    p50_warm_rtf = float(np.quantile(rtf_arr, 0.50)) if rtf_arr.size else None
    p95_warm_rtf = float(np.quantile(rtf_arr, 0.95)) if rtf_arr.size else None

    return {
        'source': source,
        'n_samples': n,
        'cold_embed_ms': float(cold_ms),
        'warm_embed': percentile_stats(warm_ms),
        'warm_fallback': warm_fallback,
        'cold_rtf': float(cold_rtf) if cold_rtf is not None else None,
        'warm_rtf': float(warm_rtf) if warm_rtf is not None else None,
        'p50_warm_rtf': p50_warm_rtf,
        'p95_warm_rtf': p95_warm_rtf,
        'total_audio_seconds': float(sum(dur_vals)),
        'warm_audio_seconds': warm_audio_s,
    }


def measure_uncached_rtf(embedder, audio_paths: list[str], *, max_samples: int = 5) -> dict:
    """Measure uncached cold+warm Real-Time Factor on up to `max_samples` distinct readable audio files."""
    chosen: list[tuple[str, float]] = []
    seen: set[str] = set()
    limit = max(0, int(max_samples))
    for raw in audio_paths:
        if len(chosen) >= limit:
            break
        text = str(raw)
        if text in seen or not Path(text).is_file():
            continue
        try:
            dur = float(duration_seconds(text))
        except Exception:
            continue
        if not np.isfinite(dur) or dur <= 0.0:
            continue
        seen.add(text)
        chosen.append((text, dur))

    ms_list: list[float] = []
    dur_list: list[float] = []
    for path, dur in chosen:
        t0 = time.perf_counter()
        embedder._embed(path)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ms_list.append(float(elapsed_ms))
        dur_list.append(dur)

    return _rtf_from_pairs(ms_list, dur_list, source='micro_bench')


def summarize_perf(
    case_timings_ms: list[float],
    step_timings_ms: dict[str, list[float]],
    cache_stats: dict | None,
    peak_tracemalloc_mb: float,
    rss_mb: float | None,
    *,
    compute_audio_s_list: list[float] | None = None,
    rtf_bench: dict | None = None,
    worker_rss_mb: float | None = None,
) -> dict:
    """Aggregate case, step, cold/warm embedding compute, Warm RTF, cache hit, and memory metrics."""
    case_stats = percentile_stats(case_timings_ms)
    steps = {
        str(step): percentile_stats(list(vals))
        for step, vals in sorted((step_timings_ms or {}).items())
    }
    if cache_stats is None:
        compute_list: list[float] = []
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

    if rtf_bench is not None and int(rtf_bench.get('n_samples', 0)) > 0:
        rtf_summary = dict(rtf_bench)
        cold_embed_ms = rtf_bench.get('cold_embed_ms')
        warm_embed = dict(rtf_bench.get('warm_embed') or percentile_stats([]))
    else:
        rtf_summary = _rtf_from_pairs(compute_list, list(compute_audio_s_list or []), source='cache_miss')
        cold_embed_ms = rtf_summary.get('cold_embed_ms')
        warm_embed = dict(rtf_summary.get('warm_embed') or percentile_stats([]))

    parent_rss = float(rss_mb) if rss_mb is not None else None
    worker_rss = float(worker_rss_mb) if worker_rss_mb is not None else None
    if parent_rss is not None and worker_rss is not None:
        effective_rss = max(parent_rss, worker_rss)
    elif worker_rss is not None:
        effective_rss = worker_rss
    else:
        effective_rss = parent_rss

    return {
        'case': case_stats,
        'total_seconds': float(case_stats['total_ms'] / 1000.0),
        'steps': steps,
        'embed_compute': embed_compute,
        'embed_lookup': embed_lookup,
        'cold_embed_ms': cold_embed_ms,
        'warm_embed': warm_embed,
        'rtf': rtf_summary,
        'cache': cache_summary,
        'peak_tracemalloc_mb': float(peak_tracemalloc_mb),
        'parent_rss_mb': parent_rss,
        'worker_rss_mb': worker_rss,
        'rss_mb': effective_rss,
    }
