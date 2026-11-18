"""Deterministic sandbox embedder, harmonic speaker synthesis, and hermetic cohort helpers."""
import functools
import hashlib
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voiceprint.audio import load_wav
from voiceprint.embed import Embedder, _normalize
from voiceprint.features import SR

DIM = 64
_FMIN = 200.0
_FMAX = 7600.0
_N_MODES = 56
_CENTERS = np.geomspace(_FMIN, _FMAX, DIM + 2)[1:-1]
_LOG_CENTERS = np.log(_CENTERS)
_ORTH = (
    np.cos(
        np.pi
        * np.arange(1, _N_MODES + 1, dtype=np.float64)[:, None]
        * ((np.arange(DIM, dtype=np.float64)[None, :] + 0.5) / DIM)
    )
    * np.sqrt(2.0 / DIM)
)
_SPEAKER_BASIS, _ = np.linalg.qr(np.random.default_rng(2025).standard_normal((_N_MODES, _N_MODES)))


def _orthogonal_projection(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


_PROJ = _orthogonal_projection(2026, DIM)


@functools.lru_cache(maxsize=8)
def _log_filters(n_freqs: int, sr: int = SR) -> np.ndarray:
    edges = np.geomspace(_FMIN, min(_FMAX, 0.48 * sr), DIM + 2)
    freqs = np.fft.rfftfreq((n_freqs - 1) * 2, d=1.0 / sr)
    fb = np.zeros((DIM, n_freqs), dtype=np.float64)
    for m in range(1, DIM + 1):
        lo, mid, hi = edges[m - 1], edges[m], edges[m + 1]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        dn = (hi - freqs) / max(hi - mid, 1e-9)
        weights = np.maximum(0.0, np.minimum(up, dn))
        area = weights.sum()
        if area == 0.0:
            fb[m - 1, int(np.argmin(np.abs(freqs - mid)))] = 1.0
        else:
            fb[m - 1] = weights / area
    fb.setflags(write=False)
    return fb


def _seed_words(*parts: object) -> list[int]:
    digest = hashlib.sha256('|'.join(str(p) for p in parts).encode('utf-8')).digest()
    return [int(word) for word in np.frombuffer(digest, dtype='<u4')]


def _speaker_profile(speaker_index: int) -> np.ndarray:
    if 0 <= speaker_index < 40:
        raw = _SPEAKER_BASIS[speaker_index] @ _ORTH
    elif 100 <= speaker_index < 116:
        raw = _SPEAKER_BASIS[40 + (speaker_index - 100)] @ _ORTH
    elif 40 <= speaker_index < _N_MODES:
        raw = _SPEAKER_BASIS[speaker_index] @ _ORTH
    else:
        rng = np.random.default_rng(_seed_words('spk_profile', speaker_index))
        weights = rng.standard_normal(_N_MODES)
        weights /= np.linalg.norm(weights) + 1e-8
        raw = weights @ _ORTH
    raw = raw - raw.mean()
    return raw / (np.std(raw) + 1e-8)


class ToneEmbedder(Embedder):
    """Deterministic spectral-bucket embedder for fast Mac unit tests and behavioral sandboxes."""

    name = 'tone'
    dim = DIM
    cohort_size = 16
    model_id = 'tone-v1-' + hashlib.sha256(b'tone-v1').hexdigest()[:56]

    def _embed_waveform(self, wav: np.ndarray) -> np.ndarray:
        win = np.hanning(wav.shape[0]).astype(np.float64)
        spec = np.abs(np.fft.rfft(wav.astype(np.float64) * win)) ** 2
        total = float(np.sum(spec)) / max(len(spec), 1) + 1e-12
        band = _log_filters(len(spec), SR) @ spec
        feat = np.log(band / total + 1e-2)
        feat = feat - np.mean(feat)
        vec = feat.astype(np.float32) @ _PROJ.T
        if float(np.linalg.norm(vec)) < 1e-8:
            return np.zeros(DIM, dtype=np.float32)
        return _normalize(vec)

    def _embed(self, path: str) -> np.ndarray:
        wav = load_wav(path, max_seconds=12.0)
        return self._embed_waveform(wav)


def generate_speaker_wav(
    path: Path,
    speaker_index: int,
    duration: float = 4.0,
    sr: int = SR,
    phase: float = 0.0,
    seed: int = 0,
) -> Path:
    """Synthesize a deterministic multi-harmonic formant-shaped speaker recording."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    profile = _speaker_profile(speaker_index)
    take_rng = np.random.default_rng(_seed_words('take', speaker_index, round(float(phase), 6), int(seed)))
    take_raw = take_rng.standard_normal(DIM)
    take_raw -= take_raw.mean()
    take_raw /= np.std(take_raw) + 1e-8
    band_log = profile + 0.18 * take_raw

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    log_f = np.log(np.maximum(freqs, _CENTERS[0]))
    env = np.exp(0.9 * np.interp(log_f, _LOG_CENTERS, band_log))

    f0 = 95.0 + (speaker_index % 17) * 8.5 + (speaker_index // 17) * 3.0
    f0_take = f0 * (1.0 + 0.015 * np.sin(phase + 0.7 * seed))
    k_max = max(1, int(min(_FMAX, 0.47 * sr) // f0_take))
    ks = np.arange(1, k_max + 1, dtype=np.float64)[:, None]
    phases = phase * (ks % 5 + 1) + 0.3 * seed * ks
    sig = np.sin(2.0 * np.pi * f0_take * ks * t[None, :] + phases).sum(axis=0)
    sig += 0.65 * np.std(sig) * take_rng.standard_normal(n)
    syllable = 0.65 + 0.35 * np.sin(2.0 * np.pi * (3.2 + 0.2 * (speaker_index % 5)) * t + phase)
    sig *= syllable

    shaped = np.fft.irfft(np.fft.rfft(sig) * env, n=n)
    shaped = shaped / (np.max(np.abs(shaped)) + 1e-8) * 0.7
    sf.write(str(out_path), shaped.astype(np.float32), sr)
    return out_path


def _build_sandbox_cohort(embedder: Embedder, n: int) -> tuple[np.ndarray, list[str]]:
    with TemporaryDirectory(prefix='vpcohort_') as tmp:
        root = Path(tmp)
        names = [f'sandbox_cohort_{i:02d}' for i in range(n)]
        rows = []
        for i, name in enumerate(names):
            wav_path = generate_speaker_wav(root / f'{name}.wav', speaker_index=100 + i, duration=2.5, seed=i)
            rows.append(embedder.embed_file(str(wav_path)))
        return np.stack(rows).astype(np.float32), names


@contextmanager
def sandbox_cohort(embedder: Embedder, n: int = 16):
    """Temporarily replace agent.service.cohort_matrix with an in-memory synthetic cohort."""
    import agent.service as service_mod

    memo_key = (embedder.name, embedder.model_id, n)
    memo_emb, memo_names = _build_sandbox_cohort(embedder, n)

    def _fake_cohort_matrix(req_embedder: Embedder, _train_root: Path, req_n: int | None = None):
        target_n = req_n if req_n is not None and req_n > 0 else (req_embedder.cohort_size or n)
        if (req_embedder.name, req_embedder.model_id, target_n) == memo_key:
            return memo_emb.copy(), list(memo_names)
        return _build_sandbox_cohort(req_embedder, target_n)

    with patch.object(service_mod, 'cohort_matrix', _fake_cohort_matrix):
        yield


def quote_paths(paths: Sequence[str | Path]) -> str:
    """Quote file paths as POSIX strings safe for shlex(posix=True) across platforms."""
    return ' '.join(f'"{p.as_posix() if isinstance(p, Path) else p}"' for p in paths)


def sandbox_enroll_takes(
    base_dir: Path,
    name: str,
    speaker_index: int,
    *,
    duration: float = 4.0,
) -> list[Path]:
    """Synthesize 3 deterministic cross-folder enrollment takes for a sandbox speaker."""
    return [
        generate_speaker_wav(base_dir / 's1' / f'{name}_0.wav', speaker_index, duration=duration, phase=0.0, seed=0),
        generate_speaker_wav(base_dir / 's1' / f'{name}_1.wav', speaker_index, duration=duration, phase=0.8, seed=1),
        generate_speaker_wav(base_dir / 's2' / f'{name}_2.wav', speaker_index, duration=duration, phase=1.6, seed=2),
    ]


def sandbox_enroll(
    agent,
    name: str,
    base_dir: Path,
    speaker_index: int,
    *,
    duration: float = 4.0,
) -> list[Path]:
    """Generate 3 enrollment takes, handle '注册 ... 本人同意 ...', and assert success."""
    takes = sandbox_enroll_takes(base_dir, name, speaker_index, duration=duration)
    out = agent.handle(f'注册 {name} 本人同意 {quote_paths(takes)}')
    if not out.get('ok'):
        raise RuntimeError(f'sandbox enrollment failed for {name}: {out}')
    return takes


def deterministic_separator(src: str, dest_dir: Path) -> dict:
    """Deterministic two-speaker splitter returning synthetic speaker-0 and speaker-1 waveforms."""
    _ = src
    dest_dir.mkdir(parents=True, exist_ok=True)
    p1 = generate_speaker_wav(dest_dir / 'det_spk1.wav', speaker_index=0, duration=2.0, phase=2.0, seed=11)
    p2 = generate_speaker_wav(dest_dir / 'det_spk2.wav', speaker_index=1, duration=2.0, phase=2.0, seed=12)
    return {'sample_rate': 8000, 'paths': [str(p1), str(p2)], 'checkpoint': 'deterministic'}
