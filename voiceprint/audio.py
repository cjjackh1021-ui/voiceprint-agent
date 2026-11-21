"""Load utterances as 16 kHz mono float32, after checking that the path is a sane local recording."""
import contextlib
import hashlib
import os
import struct
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceprint.features import SR

AUDIO_SUFFIXES = {'.wav', '.flac', '.ogg', '.mp3'}
MAX_BYTES = 512 * 1024 * 1024
MAX_SECONDS = 1800.0
MIN_SECONDS = 0.5
_MAX_CHANNELS = 64
_MAX_SAMPLE_RATE = 768000


class AudioError(ValueError):
    """The recording is refused before any model sees it. The message is shown to the user."""


def sniff_audio_format(raw: bytes) -> str | None:
    """Identify supported audio container format from leading magic bytes."""
    if not raw or len(raw) < 4:
        return None
    if len(raw) >= 12 and raw[:4] == b'RIFF' and raw[8:12] == b'WAVE':
        return '.wav'
    if raw[:4] == b'fLaC':
        return '.flac'
    if raw[:4] == b'OggS':
        return '.ogg'
    if raw[:3] == b'ID3':
        return '.mp3'
    if raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0 and raw[1] != 0xFF:
        # MPEG audio frame sync: layer bits (bits 1..2) must not be 00, version bits (3..4) must not be 01.
        version_bits = (raw[1] >> 3) & 0x03
        layer_bits = (raw[1] >> 1) & 0x03
        if version_bits != 0x01 and layer_bits != 0x00:
            return '.mp3'
    return None


def _validate_wav_container(resolved: Path) -> None:
    """Walk RIFF/WAVE chunks to reject truncated or malformed WAV containers."""
    try:
        file_size = resolved.stat().st_size
    except OSError as exc:
        raise AudioError(f'无法读取录音文件：{resolved}（{type(exc).__name__}）') from None
    if file_size < 12:
        raise AudioError(f'WAV 文件不完整或已截断：{resolved}')
    with open(resolved, 'rb') as fh:
        header = fh.read(12)
        if len(header) < 12 or header[:4] != b'RIFF' or header[8:12] != b'WAVE':
            raise AudioError(f'不是有效的 WAV 容器：{resolved}')
        riff_size = struct.unpack('<I', header[4:8])[0]
        container_end = 8 + riff_size
        if container_end > file_size or container_end < 12:
            raise AudioError(f'WAV 容器长度与文件大小不符（可能已截断）：{resolved}')

        offset = 12
        fmt_info = None
        data_size = None
        while offset < container_end:
            if offset + 8 > container_end:
                raise AudioError(f'WAV 块头不完整（可能已截断）：{resolved}')
            fh.seek(offset)
            chunk_hdr = fh.read(8)
            if len(chunk_hdr) < 8:
                raise AudioError(f'WAV 块头读取中断：{resolved}')
            chunk_id, chunk_size = struct.unpack('<4sI', chunk_hdr)
            payload_start = offset + 8
            payload_end = payload_start + chunk_size
            if payload_end > container_end or payload_end > file_size:
                raise AudioError(f'WAV 数据块不完整或已截断：{resolved}')
            if chunk_id == b'fmt ' and chunk_size >= 16:
                fmt_bytes = fh.read(16)
                if len(fmt_bytes) == 16:
                    fmt_info = struct.unpack('<HHIIHH', fmt_bytes)
            elif chunk_id == b'data':
                data_size = chunk_size
            if payload_end == container_end:
                offset = container_end
                break
            offset = payload_end + (chunk_size & 1)

        if fmt_info is None or data_size is None or data_size <= 0:
            raise AudioError(f'WAV 缺少有效的 fmt 或 data 块：{resolved}')
        fmt_tag, channels, sample_rate, _byte_rate, block_align, _bits = fmt_info
        if not (0 < channels <= _MAX_CHANNELS) or not (0 < sample_rate <= _MAX_SAMPLE_RATE):
            raise AudioError(f'WAV 声道数或采样率超出允许范围：{resolved}')
        if fmt_tag in (1, 3) and block_align > 0 and data_size < block_align:
            raise AudioError(f'WAV 数据块不包含完整采样帧：{resolved}')


def check_audio(path: str, roots: list[Path] | None = None) -> Path:
    """Refuse network paths, non-audio files, symlinks, malformed containers and oversized recordings.

    The network check runs on the text of the path, before the file system is touched: probing
    \\\\host\\share on Windows already sends this account's credentials to that host.
    """
    text = str(path).strip()
    if not text:
        raise AudioError('录音路径是空的。')
    if text.startswith(('\\\\', '//')):
        raise AudioError(f'不接受网络路径：{text}')
    suffix = Path(text).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
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
    if Path(text).is_symlink():
        raise AudioError(f'不接受符号链接录音：{text}')
    if resolved.stat().st_size > MAX_BYTES:
        raise AudioError(f'录音文件超过 {MAX_BYTES // 2**20} MB：{text}')
    try:
        with open(resolved, 'rb') as fh:
            lead = fh.read(16)
    except OSError as exc:
        raise AudioError(f'无法读取录音：{text}（{type(exc).__name__}）') from None
    sniffed = sniff_audio_format(lead)
    if sniffed is None or sniffed != suffix:
        raise AudioError(f'录音文件头与扩展名不匹配或格式无效：{text}')
    if suffix == '.wav':
        _validate_wav_container(resolved)
    try:
        info = sf.info(str(resolved))
    except Exception as exc:
        raise AudioError(f'不是可读取的录音：{text}（{type(exc).__name__}）') from None
    if info.duration > MAX_SECONDS:
        raise AudioError(f'录音长 {info.duration:.0f} 秒，超过 {MAX_SECONDS:.0f} 秒上限：{text}')
    if info.duration < MIN_SECONDS:
        raise AudioError(f'录音只有 {info.duration:.2f} 秒，至少需要 {MIN_SECONDS} 秒：{text}')
    _assert_finite_audio(resolved, int(MAX_SECONDS * info.samplerate))
    return resolved


def _assert_finite_audio(resolved: Path, max_frames: int | None = None):
    """Scan decoded PCM frames in bounded blocks to reject NaN/Inf samples before modeling."""
    if max_frames is None:
        info = sf.info(str(resolved))
        max_frames = int(MAX_SECONDS * info.samplerate)
    frames_arg = max_frames if max_frames > 0 else -1
    try:
        with contextlib.closing(
            sf.blocks(str(resolved), blocksize=65536, frames=frames_arg, dtype='float32')
        ) as blocks:
            for block in blocks:
                if not np.all(np.isfinite(block)):
                    raise AudioError(f'录音包含无效采样（NaN/Inf）：{resolved}')
    except AudioError:
        raise
    except Exception as exc:
        raise AudioError(f'不是可读取的录音：{resolved}（{type(exc).__name__}）') from None


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
    if wav.shape[0] == 0:
        raise AudioError(f'录音里没有采样：{path}')
    if not np.all(np.isfinite(wav)):
        raise AudioError(f'录音包含无效采样（NaN/Inf）：{path}')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
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
    if not np.all(np.isfinite(wav)):
        raise AudioError(f'录音包含无效采样（NaN/Inf）：{path}')
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
