"""Content-hashed memory and disk cache for speaker embedding vectors."""
import os
import struct
import time
from pathlib import Path

import numpy as np

from voiceprint.audio import file_digest
from voiceprint.embed import CACHE_LIMIT, Embedder

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / 'cache'


class CachedEmbedder(Embedder):
    """Wrap an Embedder with content-hashed memory and atomic disk caching."""

    def __init__(self, inner: Embedder, cache_dir: Path | str | None = None, enabled: bool = True):
        super().__init__()
        self.inner = inner
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.enabled = bool(enabled)
        self._mem: dict[tuple[str, str, str], np.ndarray] = {}
        self._hits = 0
        self._misses = 0
        self._compute_ms: list[float] = []
        self._lookup_ms: list[float] = []

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def checkpoint(self):
        return self.inner.checkpoint

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def cohort_size(self) -> int:
        return self.inner.cohort_size

    def describe(self) -> dict:
        return self.inner.describe()

    def _embed(self, path: str) -> np.ndarray:
        return self.inner._embed(path)

    @staticmethod
    def _safe_segment(value: str, fallback: str) -> str:
        seg = value if value else fallback
        if '/' in seg or '\\' in seg or seg in {'.', '..'}:
            raise ValueError(f'invalid cache path segment: {seg!r}')
        return seg

    def _cache_path(self, digest: str) -> Path:
        name_seg = self._safe_segment(self.name, 'unknown')
        model_seg = self._safe_segment(self.model_id[:16], 'unknown')
        return self.cache_dir / name_seg / model_seg / f'{digest}.npy'

    def _is_valid_vec(self, arr: object) -> bool:
        return (
            isinstance(arr, np.ndarray)
            and arr.shape == (self.dim,)
            and arr.dtype == np.float32
            and bool(np.all(np.isfinite(arr)))
        )

    def _store_mem(self, key: tuple[str, str, str], vec: np.ndarray) -> None:
        if key not in self._mem and len(self._mem) >= CACHE_LIMIT:
            if CACHE_LIMIT <= 0:
                return
            self._mem.pop(next(iter(self._mem)))
        self._mem[key] = vec.copy()

    def _load_disk(self, cache_path: Path) -> np.ndarray | None:
        if not cache_path.is_file():
            return None
        loaded = None
        try:
            loaded = np.load(cache_path, allow_pickle=False)
            if self._is_valid_vec(loaded):
                return np.ascontiguousarray(loaded, dtype=np.float32)
        except (OSError, ValueError, EOFError, RuntimeError, struct.error):
            pass
        finally:
            # Guard against .npz archives masquerading as .npy holding an open ZipFile handle.
            if loaded is not None and hasattr(loaded, 'close'):
                loaded.close()
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    def _save_disk(self, cache_path: Path, digest: str, vec: np.ndarray) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Name must end in .npy or np.save appends another .npy, leaving orphan temp files.
        tmp_path = cache_path.with_name(f'.{digest}.{os.getpid()}.tmp.npy')
        try:
            np.save(tmp_path, vec, allow_pickle=False)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _timed_compute(self, path: str) -> tuple[np.ndarray, float]:
        c0 = time.perf_counter()
        raw_vec = self.inner.embed_file(path)
        compute_ms = (time.perf_counter() - c0) * 1000.0
        arr = np.ascontiguousarray(raw_vec, dtype=np.float32)
        if not self._is_valid_vec(arr):
            raise ValueError(
                f'{self.name} produced invalid embedding of shape {arr.shape}, dtype {arr.dtype}'
            )
        return arr, compute_ms

    def embed_file(self, path: str) -> np.ndarray:
        """Return embedding for audio at path, using memory/disk caches when enabled."""
        if not self.enabled:
            arr, compute_ms = self._timed_compute(path)
            self._misses += 1
            self._compute_ms.append(compute_ms)
            return arr.copy()

        t0 = time.perf_counter()
        digest = file_digest(path)
        key = (self.name, self.model_id, digest)
        cache_path = self._cache_path(digest)

        hit_vec = self._mem.get(key)
        if hit_vec is None:
            hit_vec = self._load_disk(cache_path)
            if hit_vec is not None:
                self._store_mem(key, hit_vec)
        if hit_vec is not None:
            self._hits += 1
            self._lookup_ms.append((time.perf_counter() - t0) * 1000.0)
            return hit_vec.copy()

        arr, compute_ms = self._timed_compute(path)
        self._save_disk(cache_path, digest, arr)
        self._store_mem(key, arr)
        self._misses += 1
        self._compute_ms.append(compute_ms)
        return arr.copy()

    def stats(self) -> dict:
        """Return cache hit/miss counters and per-call timing lists."""
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'total': total,
            'hit_rate': (self._hits / total) if total else 0.0,
            'compute_ms_list': list(self._compute_ms),
            'lookup_ms_list': list(self._lookup_ms),
        }

    def reset_stats(self) -> None:
        """Clear hit/miss counters and timing lists without evicting cached entries."""
        self._hits = 0
        self._misses = 0
        self._compute_ms.clear()
        self._lookup_ms.clear()
