"""Pluggable anti-spoofing interfaces and zero-dependency 16-bit spread-spectrum watermark detector."""
from dataclasses import dataclass, field
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from voiceprint.features import SR

DEFAULT_WATERMARK_PAYLOAD = 0x5653  # 'VS' 16-bit provenance tag
DEFAULT_WATERMARK_SEED = 20261121
_N_BITS = 16
_FRAME_LEN = SR  # 16,000 samples = 1.0s CDMA frame -> 1 Hz FFT resolution
_BAND_LOW_HZ = 2000.0
_BAND_HIGH_HZ = 7000.0
_N_CONTROL_CODES = 32

_CODE_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class SpoofResult:
    is_spoof: bool
    score: float
    reason: str
    detector: str
    metadata: dict = field(default_factory=dict)


class SpoofDetector:
    """Abstract interface for voiceprint anti-spoofing / watermark detectors."""

    name = 'base'

    def detect(self, wav: np.ndarray, sr: int) -> SpoofResult:
        raise NotImplementedError


def _prepare_mono_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    arr = np.asarray(wav, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.reshape(-1)
    if arr.size == 0:
        return arr
    if int(sr) != SR:
        factor = gcd(int(sr), SR)
        arr = resample_poly(arr, SR // factor, int(sr) // factor)
    return arr


def _in_band_indices() -> np.ndarray:
    freqs = np.fft.rfftfreq(_FRAME_LEN, d=1.0 / SR)
    return np.flatnonzero((freqs >= _BAND_LOW_HZ) & (freqs <= _BAND_HIGH_HZ))


def _code_banks(secret_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic unit-magnitude frequency-domain CDMA codes in [2000, 7000] Hz.

    Each of the 16 payload bits owns an interleaved disjoint set of FFT bins inside the band,
    so the 16 bits are strictly orthogonal with zero cross-bit interference; control codes
    use independent random signs on the same bins to estimate the empirical null correlation.
    """
    cached = _CODE_CACHE.get(secret_seed)
    if cached is not None:
        return cached
    rng = np.random.default_rng(secret_seed)
    bins = _in_band_indices()
    perm = rng.permutation(bins)
    per_bit = len(perm) // _N_BITS
    usable = perm[:per_bit * _N_BITS].reshape(_N_BITS, per_bit)

    bit_signs = rng.choice([-1.0, 1.0], size=(_N_BITS, per_bit))
    ctrl_rng = np.random.default_rng(secret_seed + 100003)
    ctrl_signs = ctrl_rng.choice([-1.0, 1.0], size=(_N_CONTROL_CODES, _N_BITS, per_bit))
    _CODE_CACHE[secret_seed] = (usable, bit_signs, ctrl_signs)
    return usable, bit_signs, ctrl_signs


def embed_spread_watermark(
    wav: np.ndarray,
    sr: int = SR,
    *,
    payload: int = DEFAULT_WATERMARK_PAYLOAD,
    strength: float = 0.02,
    secret_seed: int = DEFAULT_WATERMARK_SEED,
) -> np.ndarray:
    """Embed a deterministic 16-bit orthogonal frequency-domain spread watermark in 2–7 kHz."""
    arr = np.asarray(wav, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)

    usable, bit_signs, _ = _code_banks(int(secret_seed))
    spec = np.zeros(_FRAME_LEN // 2 + 1, dtype=np.complex128)
    bits = int(payload) & 0xFFFF
    for idx in range(_N_BITS):
        bit_val = 1.0 if ((bits >> (_N_BITS - 1 - idx)) & 1) else -1.0
        spec[usable[idx]] = bit_val * bit_signs[idx]

    frame = np.fft.irfft(spec, n=_FRAME_LEN)
    rms = float(np.sqrt(np.mean(frame * frame)))
    if rms > 1e-12:
        frame = frame / rms

    if int(sr) != SR:
        factor = gcd(SR, int(sr))
        frame_sr = resample_poly(frame, int(sr) // factor, SR // factor)
    else:
        frame_sr = frame

    reps = -(-arr.shape[0] // frame_sr.shape[0])
    wm_full = np.tile(frame_sr, reps)[:arr.shape[0]]
    sig_rms = float(np.sqrt(np.mean(arr * arr)))
    amp = float(strength) * max(sig_rms, 0.05)
    out = np.clip(arr + amp * wm_full, -1.0, 1.0)
    return out.astype(np.float32)


class SpreadSpectrumWatermarkDetector(SpoofDetector):
    """Deterministic 16-bit orthogonal frequency-domain spread-spectrum watermark detector."""

    name = 'spread_watermark_16bit'

    def __init__(
        self,
        expected_payload: int | None = None,
        *,
        secret_seed: int = DEFAULT_WATERMARK_SEED,
        z_threshold: float = 4.5,
    ):
        self.expected_payload = None if expected_payload is None else (int(expected_payload) & 0xFFFF)
        self.secret_seed = int(secret_seed)
        self.z_threshold = float(z_threshold)

    def detect(self, wav: np.ndarray, sr: int) -> SpoofResult:
        sig = _prepare_mono_16k(wav, sr)
        if sig.size < _FRAME_LEN // 2:
            return SpoofResult(
                is_spoof=False,
                score=0.0,
                reason='too_short',
                detector=self.name,
                metadata={'payload': None},
            )
        if not bool(np.all(np.isfinite(sig))):
            return SpoofResult(
                is_spoof=False,
                score=0.0,
                reason='non_finite',
                detector=self.name,
                metadata={'payload': None},
            )

        # Coherently fold complete/partial 1.0s frames aligned modulo _FRAME_LEN.
        folded = np.zeros(_FRAME_LEN, dtype=np.float64)
        counts = np.zeros(_FRAME_LEN, dtype=np.float64)
        pos = 0
        while pos < sig.size:
            take = min(_FRAME_LEN, sig.size - pos)
            folded[:take] += sig[pos:pos + take]
            counts[:take] += 1.0
            pos += take
        nonzero = counts > 0
        folded[nonzero] /= counts[nonzero]

        spec = np.fft.rfft(folded)
        usable, bit_signs, ctrl_signs = _code_banks(self.secret_seed)
        all_bins = usable.reshape(-1)
        band_mags = np.abs(spec[all_bins])
        floor = float(np.percentile(band_mags, 25)) * 0.05 + 1e-12

        # Phase-whitened real projection per bit bin cancels narrowband speech harmonics cleanly.
        real_proj = np.real(spec) / (np.abs(spec) + floor)

        bit_corrs = np.zeros(_N_BITS, dtype=np.float64)
        decoded_payload = 0
        for idx in range(_N_BITS):
            vals = real_proj[usable[idx]]
            corr = float(np.mean(vals * bit_signs[idx]))
            bit_corrs[idx] = corr
            if corr >= 0.0:
                decoded_payload |= 1 << (_N_BITS - 1 - idx)

        ctrl_scores = np.zeros(_N_CONTROL_CODES, dtype=np.float64)
        for k in range(_N_CONTROL_CODES):
            c_bits = np.array([
                float(np.mean(real_proj[usable[idx]] * ctrl_signs[k, idx]))
                for idx in range(_N_BITS)
            ])
            ctrl_scores[k] = float(np.mean(np.abs(c_bits)))

        abs_mean = float(np.mean(np.abs(bit_corrs)))
        ctrl_mean = float(np.mean(ctrl_scores))
        ctrl_std = float(np.std(ctrl_scores))
        z_score = float((abs_mean - ctrl_mean) / max(ctrl_std, 0.008))

        payload_match = self.expected_payload is None or decoded_payload == self.expected_payload
        hit = (z_score >= self.z_threshold) and payload_match
        return SpoofResult(
            is_spoof=hit,
            score=round(z_score, 4),
            reason='watermark_detected' if hit else 'clean',
            detector=self.name,
            metadata={
                'payload': decoded_payload,
                'bits_analyzed': _N_BITS,
                'abs_corr_mean': round(abs_mean, 4),
            },
        )


class CompositeSpoofDetector(SpoofDetector):
    """Run multiple SpoofDetectors in order, short-circuiting on the first spoof hit."""

    name = 'composite'

    def __init__(self, detectors: list[SpoofDetector]):
        self.detectors = list(detectors)

    def detect(self, wav: np.ndarray, sr: int) -> SpoofResult:
        last_res = SpoofResult(is_spoof=False, score=0.0, reason='clean', detector=self.name, metadata={})
        for det in self.detectors:
            res = det.detect(wav, sr)
            if res.is_spoof:
                return res
            last_res = res
        return last_res
