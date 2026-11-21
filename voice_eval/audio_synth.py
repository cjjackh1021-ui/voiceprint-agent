"""Deterministic audio distortion and mixture synthesis for v3 robustness evaluation."""
from math import gcd

import numpy as np
from scipy.signal import fftconvolve, resample_poly

SR = 16000


def _to_mono_f64(wav: np.ndarray) -> np.ndarray:
    arr = np.asarray(wav, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr.reshape(-1)


def _rms(wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or wav.size == 0:
        return wav
    factor = gcd(src_sr, dst_sr)
    return resample_poly(wav, dst_sr // factor, src_sr // factor)


def _mix_at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if sig.size == 0:
        return np.zeros(0, dtype=np.float32)
    sig_rms = _rms(sig)
    noise_rms = _rms(noise)
    if sig_rms <= 1e-12 or noise_rms <= 1e-12:
        return np.clip(sig, -1.0, 1.0).astype(np.float32)
    target_noise_rms = sig_rms / (10.0 ** (float(snr_db) / 20.0))
    mixed = sig + noise * (target_noise_rms / noise_rms)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def add_white_noise(wav: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add deterministic Gaussian white noise calibrated to target SNR (dB)."""
    sig = _to_mono_f64(wav)
    if sig.size == 0:
        return np.zeros(0, dtype=np.float32)
    noise = np.random.default_rng(seed).standard_normal(sig.shape[0])
    return _mix_at_snr(sig, noise, snr_db)


def add_pink_noise(wav: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add deterministic 1/f spectral-shaped pink noise calibrated to target SNR (dB)."""
    sig = _to_mono_f64(wav)
    n = sig.shape[0]
    if n <= 1:
        return np.clip(sig, -1.0, 1.0).astype(np.float32)
    white = np.random.default_rng(seed).standard_normal(n)
    spec = np.fft.rfft(white)
    bins = np.arange(spec.shape[0], dtype=np.float64)
    scale = np.zeros_like(bins)
    scale[1:] = 1.0 / np.sqrt(bins[1:])
    pink = np.fft.irfft(spec * scale, n=n)
    return _mix_at_snr(sig, pink, snr_db)


def add_babble_noise(
    wav: np.ndarray,
    interferers: list[np.ndarray],
    snr_db: float,
    seed: int,
) -> np.ndarray:
    """Overlay tiled/cropped interferer speech waveforms calibrated to target SNR (dB)."""
    sig = _to_mono_f64(wav)
    n = sig.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if not interferers:
        raise ValueError('add_babble_noise requires at least one interferer')
    rng = np.random.default_rng(seed)
    babble = np.zeros(n, dtype=np.float64)
    for raw in interferers:
        item = _to_mono_f64(raw)
        m = item.shape[0]
        if m == 0:
            continue
        if m <= n:
            reps = -(-n // m) + 1
            tiled = np.tile(item, reps)
            start = int(rng.integers(0, m))
            seg = tiled[start:start + n]
        else:
            start = int(rng.integers(0, m - n + 1))
            seg = item[start:start + n]
            if _rms(seg) <= 1e-12 and _rms(item) > 1e-12:
                peak_idx = int(np.argmax(np.abs(item)))
                centered = max(0, min(m - n, peak_idx - n // 2))
                seg = item[centered:centered + n]
        seg_rms = _rms(seg)
        if seg_rms > 1e-12:
            babble += seg / seg_rms
    return _mix_at_snr(sig, babble, snr_db)


def clip_duration(wav: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Crop or zero-pad waveform to exact `seconds` length at sample rate `sr`."""
    if sr <= 0 or seconds < 0:
        raise ValueError(f'invalid duration parameters: sr={sr}, seconds={seconds}')
    sig = _to_mono_f64(wav)
    target_n = int(round(float(seconds) * int(sr)))
    if sig.shape[0] >= target_n:
        out = sig[:target_n]
    else:
        out = np.pad(sig, (0, target_n - sig.shape[0]))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def apply_telephone_channel(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Simulate 8 kHz G.711 mu-law telephone companding, quantization, and expansion."""
    if sr <= 0:
        raise ValueError(f'invalid sample rate: {sr}')
    target_sr = 8000
    sig = _to_mono_f64(wav)
    if sig.size == 0:
        return np.zeros(0, dtype=np.float32), target_sr
    down = _resample(sig, int(sr), target_sr)
    x = np.clip(down, -1.0, 1.0)
    mu = 255.0
    companded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round(companded * 127.0) / 127.0
    expanded = np.sign(quantized) * (np.power(1.0 + mu, np.abs(quantized)) - 1.0) / mu
    return np.clip(expanded, -1.0, 1.0).astype(np.float32), target_sr


def apply_reverb(wav: np.ndarray, sr: int, rt60: float, seed: int) -> np.ndarray:
    """Convolve waveform with a deterministic exponential-decay RIR preserving RMS."""
    if sr <= 0:
        raise ValueError(f'invalid sample rate: {sr}')
    sig = _to_mono_f64(wav)
    if sig.size == 0 or rt60 <= 0.0:
        return np.clip(sig, -1.0, 1.0).astype(np.float32)
    rir_len = max(1, int(round(min(float(rt60), 0.8) * int(sr))))
    t = np.arange(rir_len, dtype=np.float64) / int(sr)
    tau = float(rt60) / (3.0 * np.log(10.0))
    carrier = np.random.default_rng(seed).standard_normal(rir_len)
    rir = carrier * np.exp(-t / tau)
    rir_norm = float(np.linalg.norm(rir))
    if rir_norm > 1e-12:
        rir = rir / rir_norm
    reverb = fftconvolve(sig, rir, mode='full')[:sig.shape[0]]
    in_rms = _rms(sig)
    out_rms = _rms(reverb)
    if in_rms > 1e-12 and out_rms > 1e-12:
        reverb = reverb * (in_rms / out_rms)
    return np.clip(reverb, -1.0, 1.0).astype(np.float32)


def mix_two_speakers(
    wav_a: np.ndarray,
    sr_a: int,
    wav_b: np.ndarray,
    sr_b: int,
    target_sr: int = 8000,
) -> tuple[np.ndarray, int]:
    """Resample two utterances to `target_sr`, normalize to equal RMS (0 dB), and sum."""
    if sr_a <= 0 or sr_b <= 0 or target_sr <= 0:
        raise ValueError('sample rates must be positive')
    a = _resample(_to_mono_f64(wav_a), int(sr_a), int(target_sr))
    b = _resample(_to_mono_f64(wav_b), int(sr_b), int(target_sr))
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float32), int(target_sr)
    a = a[:n]
    b = b[:n]
    rms_a = _rms(a)
    rms_b = _rms(b)
    if rms_a > 1e-12:
        a = a / rms_a
    if rms_b > 1e-12:
        b = b / rms_b
    mix = a + b
    peak = float(np.max(np.abs(mix)))
    if peak > 0.9:
        mix = mix * (0.9 / peak)
    return mix.astype(np.float32), int(target_sr)


def apply_watermark_spoof(
    wav: np.ndarray,
    sr: int,
    *,
    payload: int = 0x5653,
    strength: float = 0.02,
) -> np.ndarray:
    """Embed a deterministic 16-bit provenance watermark simulating watermarked TTS voice cloning."""
    from voiceprint.spoof import embed_spread_watermark

    return embed_spread_watermark(wav, int(sr), payload=payload, strength=strength)


def apply_vocoder_replay_spoof(
    wav: np.ndarray,
    sr: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Simulate deterministic band-limited replay + metallic comb/phase vocoder distortion."""
    from scipy.signal import butter, sosfiltfilt

    if sr <= 0:
        raise ValueError(f'invalid sample rate: {sr}')
    sig = _to_mono_f64(wav)
    n = sig.shape[0]
    if n < 32:
        return np.clip(sig, -1.0, 1.0).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    nyq = 0.5 * float(sr)
    low = min(300.0, nyq * 0.25)
    high = min(3400.0, nyq * 0.85)
    sos = butter(4, [low, high], btype='bandpass', fs=float(sr), output='sos')
    out = sosfiltfilt(sos, sig)

    # Short comb resonance + subtle spectral phase jitter.
    delay = int(rng.integers(19, 38))
    if delay < n:
        comb = out.copy()
        comb[delay:] += 0.35 * out[:-delay]
        out = comb

    spec = np.fft.rfft(out)
    jitter = rng.uniform(-0.35, 0.35, size=spec.shape[0])
    jitter[0] = 0.0
    if spec.shape[0] > 1:
        jitter[-1] = 0.0
    out = np.fft.irfft(spec * np.exp(1j * jitter), n=n)

    in_rms = _rms(sig)
    out_rms = _rms(out)
    if in_rms > 1e-12 and out_rms > 1e-12:
        out = out * (in_rms / out_rms)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
