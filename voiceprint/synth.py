"""Controllable voice synthesizer interface, zero-dependency acoustic morphing backend,
and isolated subprocess runner for Windows GPU neural voice cloning plugins.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
from math import gcd
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly
import soundfile as sf

from voiceprint.audio import MIN_SECONDS
from voiceprint.contained_subprocess import run_contained
from voiceprint.features import SR

MAX_SIM_TEXT_CHARS = 500
MAX_SIM_SECONDS = 60.0
_N_FFT = 512
_HOP = 128


class SynthesisError(RuntimeError):
    """Raised when a synthesis backend fails, times out, or returns invalid audio."""


@dataclass(frozen=True)
class SimulationRequest:
    target_name: str
    mode: Literal['audio', 'text']
    reference_audio_path: str
    reference_text: str | None = None
    source_audio_path: str | None = None
    text: str | None = None
    target_sr: int = SR
    formant_blend: float = 0.65
    pitch_pull: float = 0.70
    gain_scale: float = 0.85
    seed: int = 0
    attempt_index: int = 1


@dataclass(frozen=True)
class SimulationResult:
    wav: np.ndarray
    sr: int
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VoiceSynthesizer:
    """Abstract voice simulation backend."""

    name: str = 'base'
    supported_modes: tuple[str, ...] = ('audio', 'text')

    def synthesize(self, req: SimulationRequest) -> SimulationResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            'backend': self.name,
            'supported_modes': list(self.supported_modes),
        }


def validate_simulation_result(
    res: SimulationResult,
    *,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SIM_SECONDS,
) -> np.ndarray:
    """Validate and normalize synthesizer output into contiguous 16 kHz mono float32 in [-1, 1]."""
    if not isinstance(res, SimulationResult):
        raise SynthesisError('合成器返回类型无效。')
    sr = int(res.sr)
    if sr <= 0:
        raise SynthesisError(f'合成采样率无效：{sr}')
    arr = np.asarray(res.wav, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 0:
        raise SynthesisError('合成音频为空。')
    if arr.ndim > 1:
        if arr.ndim > 2:
            raise SynthesisError(f'合成音频维度异常：{arr.ndim}')
        # Support either (T, C) or rare (C, T) when small channel axis comes first
        axis = 1 if arr.shape[1] <= arr.shape[0] else 0
        arr = arr.mean(axis=axis)
    arr = arr.reshape(-1)
    if not bool(np.all(np.isfinite(arr))):
        raise SynthesisError('合成音频包含 NaN 或 Inf 无效采样。')

    if sr != SR:
        factor = gcd(sr, SR)
        arr = resample_poly(arr, SR // factor, sr // factor)
        if not bool(np.all(np.isfinite(arr))):
            raise SynthesisError('合成音频重采样后包含无效采样。')

    duration = float(arr.shape[0]) / float(SR)
    if duration + 1e-9 < float(min_seconds):
        raise SynthesisError(f'合成音频过短（{duration:.2f}s < {min_seconds:.2f}s）。')
    if duration - 1e-9 > float(max_seconds):
        raise SynthesisError(f'合成音频超长（{duration:.2f}s > {max_seconds:.2f}s）。')

    arr = np.clip(arr, -1.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(arr)


def _read_mono_16k(path: str, max_seconds: float = 30.0) -> np.ndarray:
    try:
        info = sf.info(str(path))
        max_frames = int(max(1.0, float(max_seconds)) * int(info.samplerate))
        wav, sr = sf.read(str(path), frames=max_frames, dtype='float32', always_2d=False)
    except (OSError, sf.LibsndfileError) as exc:
        raise SynthesisError(f'无法读取参考/源音频（{type(exc).__name__}）：{exc}') from None
    arr = np.asarray(wav, dtype=np.float64)
    if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        raise SynthesisError(f'参考/源音频无效或为空：{path}')
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.reshape(-1)
    if int(sr) != SR:
        factor = gcd(int(sr), SR)
        arr = resample_poly(arr, SR // factor, int(sr) // factor)
    return np.ascontiguousarray(arr, dtype=np.float64)


def _estimate_f0_hz(wav: np.ndarray, sr: int = SR) -> float:
    """Estimate median fundamental frequency in [70, 350] Hz via normalized autocorrelation."""
    frame_len = int(0.05 * sr)
    hop = int(0.025 * sr)
    lag_min = max(1, int(sr / 350.0))
    lag_max = max(lag_min + 2, int(sr / 70.0))
    if wav.shape[0] < frame_len:
        wav = np.pad(wav, (0, frame_len - wav.shape[0]))
    f0s: list[float] = []
    win = np.hanning(frame_len)
    rms_all = float(np.sqrt(np.mean(wav * wav))) + 1e-12
    for start in range(0, max(1, wav.shape[0] - frame_len + 1), hop):
        seg = wav[start:start + frame_len]
        if float(np.sqrt(np.mean(seg * seg))) < 0.15 * rms_all:
            continue
        x = (seg - np.mean(seg)) * win
        ac = np.correlate(x, x, mode='full')[frame_len - 1:]
        if ac[0] <= 1e-12:
            continue
        cand = ac[lag_min:min(lag_max + 1, ac.shape[0])] / ac[0]
        if cand.size == 0:
            continue
        idx = int(np.argmax(cand))
        if cand[idx] >= 0.20:
            f0s.append(float(sr) / float(lag_min + idx))
    return float(np.median(f0s)) if f0s else 145.0


def _stft(wav: np.ndarray, n_fft: int = _N_FFT, hop: int = _HOP) -> np.ndarray:
    win = np.hanning(n_fft)
    n_pad = max(n_fft, wav.shape[0] + n_fft)
    padded = np.pad(wav, (n_fft // 2, n_pad - wav.shape[0] - n_fft // 2))
    n_frames = 1 + max(0, (padded.shape[0] - n_fft) // hop)
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:n_frames]
    return np.fft.rfft(frames * win[None, :], axis=1)


def _istft(spec: np.ndarray, length: int, n_fft: int = _N_FFT, hop: int = _HOP) -> np.ndarray:
    win = np.hanning(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * win[None, :]
    total_len = (spec.shape[0] - 1) * hop + n_fft
    out = np.zeros(total_len, dtype=np.float64)
    wsum = np.zeros(total_len, dtype=np.float64)
    win_sq = win * win
    for i in range(spec.shape[0]):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        wsum[s:s + n_fft] += win_sq
    mask = wsum > 1e-8
    out[mask] /= wsum[mask]
    start = n_fft // 2
    end = start + length
    if out.shape[0] < end:
        out = np.pad(out, (0, end - out.shape[0]))
    return out[start:end]


def _smooth_log_envelope(wav: np.ndarray, n_fft: int = _N_FFT, hop: int = _HOP) -> np.ndarray:
    """Compute smoothed active-frame log spectral envelope over FFT bins."""
    spec = _stft(wav, n_fft=n_fft, hop=hop)
    mag_sq = np.abs(spec) ** 2
    frame_energy = np.mean(mag_sq, axis=1)
    thresh = float(np.percentile(frame_energy, 40)) if frame_energy.size > 1 else 0.0
    active = mag_sq[frame_energy >= thresh]
    if active.shape[0] == 0:
        active = mag_sq
    mean_pow = np.mean(active, axis=0)
    mean_pow = mean_pow / (float(np.mean(mean_pow)) + 1e-12)
    log_env = np.log(mean_pow + 1e-3)
    return gaussian_filter1d(log_env, sigma=1.0, mode='nearest')


def _pitch_shift_preserve_duration(wav: np.ndarray, ratio: float) -> np.ndarray:
    """Shift pitch by `ratio` (e.g., >1 raises pitch) while keeping output length constant."""
    ratio = float(np.clip(ratio, 0.65, 1.55))
    if abs(ratio - 1.0) < 1e-3 or wav.shape[0] < _N_FFT:
        return wav.copy()
    spec = _stft(wav, n_fft=_N_FFT, hop=_HOP)
    n_frames, n_bins = spec.shape
    time_steps = np.arange(0, n_frames, ratio, dtype=np.float64)
    if time_steps.size < 2:
        return wav.copy()
    omega = 2.0 * np.pi * _HOP * np.arange(n_bins, dtype=np.float64) / float(_N_FFT)
    phase_acc = np.angle(spec[0]).copy()
    stretched = np.zeros((time_steps.size, n_bins), dtype=np.complex128)
    for out_i, t in enumerate(time_steps):
        idx = min(int(np.floor(t)), n_frames - 2)
        frac = t - idx
        c0 = spec[idx]
        c1 = spec[idx + 1]
        mag = (1.0 - frac) * np.abs(c0) + frac * np.abs(c1)
        dphi = np.angle(c1) - np.angle(c0) - omega
        dphi = np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi
        phase_acc += omega + dphi
        stretched[out_i] = mag * np.exp(1j * phase_acc)
    stretched_wav = _istft(stretched, int(round(wav.shape[0] / ratio)))
    # Resample stretched waveform back to original sample count cleanly
    target_n = wav.shape[0]
    src_n = max(1, stretched_wav.shape[0])
    x_old = np.linspace(0.0, 1.0, src_n, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_n, dtype=np.float64)
    return np.interp(x_new, x_old, stretched_wav)


class AcousticFormantSynthesizer(VoiceSynthesizer):
    """Zero-dependency deterministic spectral envelope and F0 morphing synthesizer."""

    name = 'acoustic_formant'
    supported_modes = ('audio', 'text')

    def synthesize(self, req: SimulationRequest) -> SimulationResult:
        t0 = time.perf_counter()
        if req.mode not in self.supported_modes:
            raise SynthesisError(f'不支持的合成模式：{req.mode}')

        ref = _read_mono_16k(req.reference_audio_path)
        f0_ref = _estimate_f0_hz(ref, SR)
        log_env_ref = _smooth_log_envelope(ref)

        formant_blend = float(np.clip(req.formant_blend, 0.0, 1.0))
        pitch_pull = float(np.clip(req.pitch_pull, 0.0, 1.0))
        gain_scale = float(np.clip(req.gain_scale, 0.1, 1.25))

        freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / SR)
        voice_band = ((freqs >= 100.0) & (freqs <= 7800.0)).astype(np.float64)

        if req.mode == 'audio':
            if not req.source_audio_path:
                raise SynthesisError('源音频模式缺少 source_audio_path。')
            src = _read_mono_16k(req.source_audio_path)
            f0_src = _estimate_f0_hz(src, SR)
            ratio = (f0_ref / max(f0_src, 70.0)) ** pitch_pull
            shifted = _pitch_shift_preserve_duration(src, ratio)

            log_env_shifted = _smooth_log_envelope(shifted)
            delta_log = np.clip(log_env_ref - log_env_shifted, -5.0, 5.0) * formant_blend * voice_band
            filter_gain = np.exp(0.5 * delta_log)

            spec = _stft(shifted)
            morphed_spec = spec * filter_gain[None, :]
            # Normalize output spectral envelope once more in case phase-vocoder residual tilted slope
            out = _istft(morphed_spec, src.shape[0])
            if formant_blend > 0.4:
                log_env_out = _smooth_log_envelope(out)
                corr_gain = np.exp(0.5 * np.clip(log_env_ref - log_env_out, -3.5, 3.5) * formant_blend * voice_band)
                out = _istft(_stft(out) * corr_gain[None, :], src.shape[0])
            src_rms = float(np.sqrt(np.mean(src * src))) + 1e-12
            out_rms = float(np.sqrt(np.mean(out * out))) + 1e-12
            out = out * (src_rms * gain_scale / out_rms)
            extra = {'f0_ref_hz': round(f0_ref, 2), 'f0_src_hz': round(f0_src, 2), 'pitch_ratio': round(ratio, 3)}
        else:
            text_clean = (req.text or '').strip()
            if not text_clean or len(text_clean) > MAX_SIM_TEXT_CHARS:
                raise SynthesisError(f'文本模式要求非空且不超过 {MAX_SIM_TEXT_CHARS} 个字符。')
            out, f0_synth = self._synthesize_text(text_clean, f0_ref, log_env_ref, formant_blend, pitch_pull, gain_scale, req.seed)
            extra = {'f0_ref_hz': round(f0_ref, 2), 'f0_synth_hz': round(f0_synth, 2), 'text_len': len(text_clean)}

        out = np.tanh(out * 1.05) if np.max(np.abs(out)) > 1.0 else out
        wav_f32 = validate_simulation_result(
            SimulationResult(wav=out.astype(np.float32), sr=SR, backend=self.name, metadata=extra)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        meta = {
            **extra,
            'formant_blend': formant_blend,
            'pitch_pull': pitch_pull,
            'gain_scale': gain_scale,
            'elapsed_ms': round(elapsed_ms, 2),
        }
        return SimulationResult(wav=wav_f32, sr=SR, backend=self.name, metadata=meta)

    def _synthesize_text(
        self,
        text: str,
        f0_ref: float,
        log_env_ref: np.ndarray,
        formant_blend: float,
        pitch_pull: float,
        gain_scale: float,
        seed: int,
    ) -> tuple[np.ndarray, float]:
        digest = hashlib.sha256(f'{seed}|{text}'.encode('utf-8')).digest()
        rng_seed = int.from_bytes(digest[:8], 'little')
        rng = np.random.default_rng(rng_seed)

        # Deterministic syllable timeline: pauses at punctuation give temporal contrast under utterance CMVN.
        punc = set(',，.。!！?？;；:：、 ')
        segments: list[tuple[bool, float, float]] = []
        for ch in text:
            if ch in punc:
                segments.append((False, 0.08, 0.0))
            else:
                code = ord(ch)
                dur = 0.13 + (code % 5) * 0.018
                tone_mod = 1.0 + (((code % 11) - 5) * 0.015)
                segments.append((True, dur, tone_mod))
            if sum(s[1] for s in segments) >= 6.0:
                break
        while sum(s[1] for s in segments) < 1.4:
            segments.extend(segments[:max(1, min(len(segments), 4))])

        f0_base = (1.0 - pitch_pull) * 145.0 + pitch_pull * f0_ref
        total_n = int(round(sum(s[1] for s in segments) * SR))
        t_all = np.arange(total_n, dtype=np.float64) / SR
        inst_f0 = np.full(total_n, f0_base, dtype=np.float64)
        amp_env = np.zeros(total_n, dtype=np.float64)
        voiced_mask = np.zeros(total_n, dtype=np.float64)

        pos = 0
        for is_voiced, dur, tone_mod in segments:
            seg_n = max(1, int(round(dur * SR)))
            end = min(total_n, pos + seg_n)
            actual_n = end - pos
            if actual_n <= 0:
                break
            if is_voiced:
                local_t = np.linspace(0.0, 1.0, actual_n, dtype=np.float64)
                window = np.sin(np.pi * local_t) ** 0.75
                vibrato = 1.0 + 0.018 * np.sin(2.0 * np.pi * 5.2 * t_all[pos:end] + rng.uniform(0, 6.28))
                inst_f0[pos:end] = f0_base * tone_mod * vibrato
                amp_env[pos:end] = 0.25 + 0.75 * window
                voiced_mask[pos:end] = 1.0
            else:
                amp_env[pos:end] = 0.02
            pos = end

        # Build broadband harmonic excitation (randomized harmonic phases avoid impulsive Dirac crest)
        # plus shaped unvoiced breath noise so spread-spectrum watermark bins remain clean.
        phase = 2.0 * np.pi * np.cumsum(inst_f0) / float(SR)
        k_max = max(3, int(min(7500.0, 0.47 * SR) // max(f0_base, 70.0)))
        harm_phases = rng.uniform(0.0, 2.0 * np.pi, size=k_max)
        harm = np.zeros(total_n, dtype=np.float64)
        for idx_k in range(k_max):
            harm += np.sin((idx_k + 1) * phase + harm_phases[idx_k])
        harm = harm / max(float(np.std(harm)), 1e-6)
        noise = rng.standard_normal(total_n) * 0.65
        exc = (harm * voiced_mask + noise) * amp_env

        # Normalize excitation spectrum against target smoothed formant envelope directly
        freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / SR)
        voice_band = ((freqs >= 120.0) & (freqs <= 7800.0)).astype(np.float64)
        log_env_exc = _smooth_log_envelope(exc)
        delta_env = np.clip(log_env_ref - log_env_exc, -5.0, 5.0) * formant_blend * voice_band
        formant_filter = np.exp(0.5 * delta_env)

        spec = _stft(exc)
        shaped = _istft(spec * formant_filter[None, :], total_n)
        peak = float(np.max(np.abs(shaped))) + 1e-12
        out = shaped / peak * (0.68 * gain_scale)
        return out, float(f0_base)


def split_command_args(cmd: str | Sequence[str]) -> list[str]:
    """Split a command preserving quoted Windows paths with spaces and backslashes."""
    if not isinstance(cmd, str):
        parts = [str(item) for item in cmd if str(item) != '']
        if not parts:
            raise ValueError('外部合成器命令不能为空。')
        return parts

    text = cmd.strip()
    if not text:
        raise ValueError('外部合成器命令不能为空。')

    args: list[str] = []
    buf: list[str] = []
    in_token = False
    quote: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            if ch == '\\' and i + 1 < n and text[i + 1] == quote:
                buf.append(quote)
                i += 2
                continue
            if ch == quote:
                quote = None
            else:
                buf.append(ch)
        else:
            if ch.isspace():
                if in_token:
                    args.append(''.join(buf))
                    buf.clear()
                    in_token = False
            elif ch in ('"', "'"):
                quote = ch
                in_token = True
            else:
                buf.append(ch)
                in_token = True
        i += 1

    if quote is not None:
        raise ValueError('外部合成器命令存在未闭合的引号。')
    if in_token:
        args.append(''.join(buf))
    if not args:
        raise ValueError('外部合成器命令不能为空。')
    return args


class SubprocessNeuralSynthesizer(VoiceSynthesizer):
    """Isolated subprocess wrapper for Windows GPU Zero-Shot TTS/VC worker scripts."""

    name = 'neural_subprocess'

    def __init__(
        self,
        cmd: str | Sequence[str],
        *,
        supported_modes: tuple[str, ...] = ('text',),
        timeout_s: float = 120.0,
        cwd: str | Path | None = None,
        env_overrides: dict[str, str] | None = None,
    ):
        self.cmd = split_command_args(cmd)
        self.supported_modes = tuple(supported_modes)
        self.timeout_s = max(0.05, float(timeout_s))
        self.cwd = Path(cwd) if cwd is not None else None
        self.env_overrides = dict(env_overrides or {})

    def describe(self) -> dict[str, Any]:
        return {
            'backend': self.name,
            'supported_modes': list(self.supported_modes),
            'executable': Path(self.cmd[0]).name,
            'timeout_s': self.timeout_s,
        }

    def synthesize(self, req: SimulationRequest) -> SimulationResult:
        if req.mode not in self.supported_modes:
            raise SynthesisError(f'当前神经克隆后端不支持 {req.mode} 模式。')
        if req.mode == 'text':
            clean = (req.text or '').strip()
            if not clean or len(clean) > MAX_SIM_TEXT_CHARS:
                raise SynthesisError(f'合成文本必须非空且不超过 {MAX_SIM_TEXT_CHARS} 个字符。')

        child_env = os.environ.copy()
        child_env.update(self.env_overrides)
        child_env['PYTHONUTF8'] = '1'
        child_env['PYTHONIOENCODING'] = 'utf-8'

        t0 = time.perf_counter()
        with TemporaryDirectory(prefix='vpsyn_') as tmp:
            tmp_dir = Path(tmp)
            req_path = tmp_dir / 'request.json'
            out_wav = tmp_dir / 'output.wav'
            payload = {
                'target_name': req.target_name,
                'mode': req.mode,
                'reference_audio_path': str(Path(req.reference_audio_path).resolve()),
                'reference_text': req.reference_text,
                'source_audio_path': str(Path(req.source_audio_path).resolve()) if req.source_audio_path else None,
                'text': req.text,
                'target_sr': SR,
                'formant_blend': float(req.formant_blend),
                'pitch_pull': float(req.pitch_pull),
                'gain_scale': float(req.gain_scale),
                'seed': int(req.seed),
                'attempt_index': int(req.attempt_index),
                'output_wav_path': str(out_wav),
            }
            req_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

            try:
                res = run_contained(
                    self.cmd + ['--request', str(req_path)],
                    timeout=self.timeout_s,
                    cwd=self.cwd,
                    env=child_env,
                )
            except OSError as exc:
                raise SynthesisError(f'无法启动外部神经合成进程（{type(exc).__name__}）：{exc}') from None
            if res.timed_out:
                raise SynthesisError(f'外部神经合成子进程超时（>{self.timeout_s:.1f}s），已强杀进程树。')
            stdout_txt = res.stdout.decode('utf-8', errors='replace').strip()
            stderr_txt = res.stderr.decode('utf-8', errors='replace').strip()
            if res.returncode != 0:
                raise SynthesisError(f'外部神经合成失败（退出码 {res.returncode}）：{stderr_txt[:240] or stdout_txt[:240]}')
            if not out_wav.is_file():
                raise SynthesisError('外部神经合成未写入目标输出音频文件。')

            try:
                wav, sr = sf.read(str(out_wav), dtype='float32', always_2d=False)
            except Exception as exc:
                raise SynthesisError(f'无法读取外部合成输出音频（{type(exc).__name__}）。') from None

            worker_meta: dict[str, Any] = {}
            if stdout_txt:
                last_line = stdout_txt.splitlines()[-1].strip()
                if last_line.startswith('{'):
                    try:
                        parsed = json.loads(last_line)
                        if isinstance(parsed, dict):
                            worker_meta = parsed
                    except json.JSONDecodeError:
                        pass

            normalized = validate_simulation_result(
                SimulationResult(wav=wav, sr=int(sr), backend=self.name, metadata=worker_meta)
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            worker_meta.setdefault('elapsed_ms', round(elapsed_ms, 2))
            return SimulationResult(wav=normalized, sr=SR, backend=self.name, metadata=worker_meta)


def load_synthesizer(
    kind: str,
    *,
    neural_cmd: str | Sequence[str] | None = None,
    timeout_s: float = 120.0,
    supported_modes: tuple[str, ...] = ('text',),
) -> VoiceSynthesizer | None:
    """Instantiate synthesizer backend by name ('acoustic', 'neural_subprocess', 'none')."""
    if kind == 'none':
        return None
    if kind == 'acoustic':
        return AcousticFormantSynthesizer()
    if kind == 'neural_subprocess':
        if not neural_cmd:
            raise ValueError('使用 neural_subprocess 合成后端时必须提供 --neural-cmd。')
        return SubprocessNeuralSynthesizer(
            neural_cmd,
            supported_modes=supported_modes,
            timeout_s=timeout_s,
        )
    raise ValueError(f'未知的合成后端类型：{kind}')
