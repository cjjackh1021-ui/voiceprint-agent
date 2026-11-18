"""Load utterances as 16 kHz mono float32, after checking that the path is a sane local recording."""
import hashlib
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceprint.features import SR

AUDIO_SUFFIXES = {'.wav', '.flac', '.ogg', '.mp3'}
MAX_BYTES = 512 * 1024 * 1024
MAX_SECONDS = 1800.0
MIN_SECONDS = 0.5


class AudioError(ValueError):
    """The recording is refused before any model sees it. The message is shown to the user."""


def check_audio(path: str, roots: list[Path] | None = None) -> Path:
    """Refuse network paths, non-audio files and oversized recordings. Returns the resolved path.

    The network check runs on the text of the path, before the file system is touched: probing
    \\\\host\\share on Windows already sends this account's credentials to that host.
    """
    text = str(path).strip()
    if not text:
        raise AudioError('录音路径是空的。')
    if text.startswith(('\\\\', '//')):
        raise AudioError(f'不接受网络路径：{text}')
    if Path(text).suffix.lower() not in AUDIO_SUFFIXES:
        raise AudioError(f'只接受 {"、".join(sorted(AUDIO_SUFFIXES))} 录音：{text}')
    if not os.path.isfile(text):
        raise AudioError(f'找不到录音：{text}')
    resolved = Path(text).resolve()
    if str(resolved).startswith('\\\\') and not str(resolved).startswith('\\\\?\\'):
        raise AudioError(f'不接受网络路径：{text}')
    if roots:
        allowed = [Path(root).resolve() for root in roots]
        if not any(resolved == root or root in resolved.parents for root in allowed):
            raise AudioError(f'录音不在允许的目录内：{text}')
    if resolved.stat().st_size > MAX_BYTES:
        raise AudioError(f'录音文件超过 {MAX_BYTES // 2**20} MB：{text}')
    try:
        info = sf.info(str(resolved))
    except Exception as exc:
        raise AudioError(f'不是可读取的录音：{text}（{type(exc).__name__}）') from None
    if info.duration > MAX_SECONDS:
        raise AudioError(f'录音长 {info.duration:.0f} 秒，超过 {MAX_SECONDS:.0f} 秒上限：{text}')
    if info.duration < MIN_SECONDS:
        raise AudioError(f'录音只有 {info.duration:.2f} 秒，至少需要 {MIN_SECONDS} 秒：{text}')
    return resolved


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def duration_seconds(path: str) -> float:
    return float(sf.info(path).duration)


def load_wav(path: str, max_seconds: float = 30.0) -> np.ndarray:
    # Only the part that is used is decoded, so a long file cannot fill the memory.
    native = sf.info(path).samplerate
    wav, sr = sf.read(path, frames=int(max_seconds * native), dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.shape[0] == 0:
        raise AudioError(f'录音里没有采样：{path}')
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        factor = gcd(int(sr), SR)
        wav = resample_poly(wav, SR // factor, int(sr) // factor).astype(np.float32)
    limit = int(max_seconds * SR)
    if wav.shape[0] > limit:
        wav = wav[:limit]
    if wav.shape[0] < SR:
        # Repeat instead of padding with zeros: silence the model never saw in training drags the embedding.
        wav = np.tile(wav, -(-SR // wav.shape[0]))[:SR]
    return np.ascontiguousarray(wav, dtype=np.float32)


def crop(wav: np.ndarray, seconds: float, start: int | None = None) -> np.ndarray:
    n = int(seconds * SR)
    if wav.shape[0] < n:
        wav = np.pad(wav, (0, n - wav.shape[0]))
    if start is None:
        start = 0
    start = max(0, min(start, wav.shape[0] - n))
    return wav[start:start + n]


def list_flac(root: str) -> dict[str, list[str]]:
    base = Path(root)
    speakers = {}
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        files = sorted(str(p) for p in folder.rglob('*.flac'))
        if files:
            speakers[folder.name] = files
    return speakers
