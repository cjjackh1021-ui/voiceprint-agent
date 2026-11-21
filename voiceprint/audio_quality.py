"""Model-free 100ms sliding-window DSP audio quality analyzer."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceprint.audio import MIN_SECONDS

MAX_ENROLL_CLIPPING_RATIO = 0.25
_MAX_CHANNELS = 64
_MAX_SAMPLE_RATE = 768000
_SILENCE_DBFS = -50.0
_CLIP_THRESHOLD = 0.999
_CLIP_MIN_SAMPLES = 3


@dataclass(frozen=True)
class AudioWarning:
    kind: str
    start: float
    end: float


@dataclass(frozen=True)
class AudioQualityReport:
    duration: float
    analyzed_seconds: float
    truncated: bool
    active_seconds: float
    silence_ratio: float
    clipping_ratio: float
    warnings: list[AudioWarning]
    passed_enroll_quality: bool
    reject_reasons: list[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        data['warnings'] = [asdict(w) for w in self.warnings]
        return data


def analyze_audio_quality(
    path: str | Path,
    *,
    max_seconds: float = 1800.0,
    max_warnings: int = 100,
) -> AudioQualityReport:
    """Scan 100ms windows for empty/invalid/clipping/silence/quiet/loud regions."""
    levels: list[float] = []
    clips: list[bool] = []
    invalids: list[bool] = []

    with sf.SoundFile(str(path)) as source:
        if source.channels <= 0 or source.channels > _MAX_CHANNELS or source.samplerate <= 0 or source.samplerate > _MAX_SAMPLE_RATE:
            raise ValueError('Unsupported audio dimensions')
        duration = float(len(source)) / float(source.samplerate)
        window = max(1, int(round(source.samplerate * 0.1)))
        limit = min(len(source), int(max(0.0, min(float(max_seconds), 7200.0)) * source.samplerate))
        read = 0
        while read < limit:
            data = source.read(min(window, limit - read), dtype='float32', always_2d=True)
            if len(data) == 0:
                break
            read += len(data)
            finite = np.isfinite(data)
            has_invalid = not bool(finite.all())
            invalids.append(has_invalid)
            clean = np.where(finite, data, 0.0).astype(np.float64)
            clips.append(bool(np.count_nonzero(np.abs(clean) >= _CLIP_THRESHOLD) >= _CLIP_MIN_SAMPLES))
            rms = float(np.sqrt(np.max(np.mean(clean * clean, axis=0))))
            levels.append(float(20.0 * np.log10(max(rms, 1e-12))))
        analyzed = float(read) / float(source.samplerate)
        step = float(window) / float(source.samplerate)

    truncated = analyzed + 1e-9 < duration
    if not levels:
        return AudioQualityReport(
            duration=duration,
            analyzed_seconds=analyzed,
            truncated=truncated,
            active_seconds=0.0,
            silence_ratio=1.0,
            clipping_ratio=0.0,
            warnings=[AudioWarning(kind='empty', start=0.0, end=0.0)],
            passed_enroll_quality=False,
            reject_reasons=['empty'],
        )

    values = np.asarray(levels, dtype=np.float64)
    clip_arr = np.asarray(clips, dtype=bool)
    invalid_arr = np.asarray(invalids, dtype=bool)
    active_mask = values > _SILENCE_DBFS
    active_vals = values[active_mask]
    baseline = float(np.median(active_vals)) if active_vals.size > 0 else _SILENCE_DBFS

    checks = (
        ('invalid', invalid_arr, 0.0),
        ('clipping', clip_arr, 0.0),
        ('silence', ~active_mask, 0.0 if active_vals.size == 0 else 1.0),
        ('quiet', active_mask & (values < baseline - 12.0), 0.5),
        ('loud', values > baseline + 12.0, 0.5),
    )

    warnings: list[AudioWarning] = []
    for kind, mask, min_dur in checks:
        padded = np.concatenate(([False], mask, [False])).astype(np.int8)
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for s_idx, e_idx in zip(starts, ends):
            begin = float(s_idx) * step
            finish = min(float(e_idx) * step, analyzed)
            if finish - begin + 1e-8 < min_dur:
                continue
            warnings.append(AudioWarning(kind=kind, start=round(begin, 3), end=round(finish, 3)))

    warnings.sort(key=lambda item: (item.start, item.kind))
    if len(warnings) > max_warnings:
        truncated = True
        warnings = warnings[:max_warnings]

    n_win = len(levels)
    active_seconds = min(analyzed, float(np.count_nonzero(active_mask)) * step)
    silence_ratio = float(np.count_nonzero(~active_mask)) / float(n_win)
    clipping_ratio = float(np.count_nonzero(clip_arr)) / float(n_win)

    reject_reasons: list[str] = []
    if bool(invalid_arr.any()):
        reject_reasons.append('invalid')
    if active_seconds + 1e-8 < MIN_SECONDS:
        reject_reasons.append('silence')
    if clipping_ratio > MAX_ENROLL_CLIPPING_RATIO:
        reject_reasons.append('clipping')

    return AudioQualityReport(
        duration=round(duration, 4),
        analyzed_seconds=round(analyzed, 4),
        truncated=truncated,
        active_seconds=round(active_seconds, 4),
        silence_ratio=round(silence_ratio, 4),
        clipping_ratio=round(clipping_ratio, 4),
        warnings=warnings,
        passed_enroll_quality=(len(reject_reasons) == 0),
        reject_reasons=reject_reasons,
    )
