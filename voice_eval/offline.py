"""Strict offline execution environment and model fingerprint audit helpers."""
import os
from contextlib import contextmanager
from pathlib import Path

from voiceprint.audio import file_digest

_OFFLINE_ENV_KEYS = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE')


@contextmanager
def strict_offline_env():
    """Force HuggingFace/Transformers offline env vars for deterministic eval and restore on exit."""
    previous = {key: os.environ.get(key) for key in _OFFLINE_ENV_KEYS}
    try:
        for key in _OFFLINE_ENV_KEYS:
            os.environ[key] = '1'
        yield
    finally:
        for key, old_val in previous.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


def collect_model_fingerprint(embedder) -> dict:
    """Validate non-empty model_id and collect checkpoint SHA-256 for runtime_audit."""
    desc = dict(embedder.describe())
    model_id = str(desc.get('model_id') or getattr(embedder, 'model_id', '') or '').strip()
    if not model_id:
        raise ValueError(f'embedder {desc.get("embedder")!r} is missing a non-empty model_id')

    ckpt = desc.get('checkpoint')
    ckpt_sha256: str | None = None
    if ckpt:
        ckpt_path = Path(str(ckpt))
        if ckpt_path.is_file():
            ckpt_sha256 = file_digest(str(ckpt_path))

    return {
        'embedder': str(desc.get('embedder') or getattr(embedder, 'name', 'unknown')),
        'dim': int(desc.get('dim') or getattr(embedder, 'dim', 0)),
        'model_id': model_id,
        'checkpoint': str(ckpt) if ckpt is not None else None,
        'checkpoint_sha256': ckpt_sha256,
        'offline_enforced': True,
    }
