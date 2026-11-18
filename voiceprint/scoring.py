"""Cosine decisions with cohort score normalization and separate verify / identify thresholds.

Raw cosine is normalized by how the enrollment voice and the probe each sit against a cohort
of other speakers. The blend weight was selected on the validation speakers only.
"""
import numpy as np

# Validation-only settings. Locked speakers are not used to choose these.
# enroll_min: lowest cosine allowed between two enrollment recordings of one speaker. For ECAPA the
# validation speakers never go below 0.37 and 99% of different-speaker pairs stay under 0.34.
# VPNet and the stats baseline overlap too much for a tighter value (a validation speaker reaches -0.09).
PROFILES = {
    'ecapa': {'alpha': 0.25, 'identify_extra': 0.75, 'gap': 0.80, 'band': 0.35, 'enroll_min': 0.30},
    'vpnet': {'alpha': 0.45, 'identify_extra': 0.00, 'gap': 0.30, 'band': 0.20, 'enroll_min': 0.15},
    'stats': {'alpha': 0.45, 'identify_extra': 0.00, 'gap': 0.30, 'band': 0.20, 'enroll_min': 0.15},
}


def profile_for(name: str) -> dict:
    return PROFILES.get(name, PROFILES['vpnet'])


def l2(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim == 1:
        return v / max(np.linalg.norm(v), 1e-8)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-8, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2(a), l2(b)))


def theta_deg(score: float) -> float:
    return float(np.degrees(np.arccos(np.clip(score, -1.0, 1.0))))


def _z(value: float, reference: np.ndarray) -> float:
    return (value - float(reference.mean())) / max(float(reference.std()), 1e-3)


def decide(probe: np.ndarray, segments: np.ndarray, cohort: np.ndarray,
           bootstrap: int = 200, seed: int = 0, task: str = 'verify', profile: str = 'vpnet',
           alpha: float | None = None, band: float | None = None, identify_extra: float | None = None) -> dict:
    """alpha / band / identify_extra override the profile. voice_eval.tune uses them to search."""
    cfg = profile_for(profile)
    alpha = cfg['alpha'] if alpha is None else alpha
    band = cfg['band'] if band is None else band
    extra = cfg['identify_extra'] if identify_extra is None else identify_extra
    segments = l2(segments)
    probe = l2(probe)
    template = l2(segments.mean(axis=0))
    raw = float(np.dot(template, probe))
    genuine = np.asarray([
        float(np.dot(segments[i], segments[j]))
        for i in range(len(segments)) for j in range(i + 1, len(segments))
    ] or [0.5], dtype=np.float64)

    cohort_n = int(len(cohort))
    if cohort_n >= 8:
        cohort = l2(cohort)
        ref_template = cohort @ template
        ref_probe = cohort @ probe
        score = 0.5 * (_z(raw, ref_template) + _z(raw, ref_probe))
        genuine_norm = np.asarray([_z(float(item), ref_template) for item in genuine])
        negative_norm = (ref_template - ref_template.mean()) / max(float(ref_template.std()), 1e-3)
        negative_edge = float(np.quantile(negative_norm, 0.95))
        # Enrollment recordings that barely resemble each other must not pull the threshold
        # below the edge of the impostor scores.
        threshold = max((1.0 - alpha) * negative_edge + alpha * float(np.min(genuine_norm)), negative_edge)
    else:
        score = raw
        threshold = float(0.55 * 0.3 + 0.45 * np.min(genuine))

    if task == 'identify':
        threshold += extra

    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(segments), size=(bootstrap, len(segments)))
    samples = l2(segments[picked].mean(axis=1)) @ probe
    low, high = (float(x) for x in np.quantile(samples, [0.025, 0.975]))

    if abs(score - threshold) < band:
        decision = 'uncertain'
        reason = '分数贴近阈值，请再录一段更长、更安静的录音。'
    elif score >= threshold:
        decision = 'accept'
        reason = '归一化相似度高于验证阈值。' if task == 'verify' else '归一化相似度高于辨认阈值。'
    else:
        decision = 'reject'
        reason = '归一化相似度低于验证阈值。' if task == 'verify' else '归一化相似度低于辨认阈值。'
    return {
        'score': score,
        'raw_score': raw,
        'theta_deg': theta_deg(raw),
        'threshold': threshold,
        'low': low,
        'high': high,
        'decision': decision,
        'reason': reason,
        'cohort': cohort_n,
        'enrollment_segments': int(len(segments)),
        'task': task,
    }
