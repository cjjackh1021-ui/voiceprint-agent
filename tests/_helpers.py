"""Test-only DPAPI mock and synthetic LibriSpeech corpus builder."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import voiceprint.security
import voiceprint.store
from voice_eval.dims._sandbox import generate_speaker_wav

_MOCK_PREFIX = b'MOCKDPAPI1:'


def _mock_dpapi_protect(data: bytes) -> bytes:
    return _MOCK_PREFIX + bytes(data)


def _mock_dpapi_unprotect(data: bytes) -> bytes:
    if not data.startswith(_MOCK_PREFIX):
        raise ValueError('invalid mock DPAPI blob')
    return data[len(_MOCK_PREFIX):]


@contextmanager
def patched_dpapi():
    """Patch DPAPI protect/unprotect inside security and store for non-Windows test runs."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(voiceprint.security, 'dpapi_protect', _mock_dpapi_protect))
        stack.enter_context(patch.object(voiceprint.security, 'dpapi_unprotect', _mock_dpapi_unprotect))
        stack.enter_context(patch.object(voiceprint.store, 'dpapi_protect', _mock_dpapi_protect))
        stack.enter_context(patch.object(voiceprint.store, 'dpapi_unprotect', _mock_dpapi_unprotect))
        yield


def synthetic_librispeech_tree(root: Path, n_train: int = 12, n_test: int = 40) -> tuple[Path, Path]:
    """Create a deterministic LibriSpeech-shaped FLAC tree for offline pipeline tests."""
    base = Path(root)
    train_root = base / 'train-clean-100'
    test_root = base / 'test-clean'
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    for i in range(n_train):
        spk_id = f'{1000 + i:04d}'
        generate_speaker_wav(
            train_root / spk_id / '100' / f'{spk_id}-100-0000.flac',
            speaker_index=100 + i,
            duration=2.5,
            phase=0.0,
            seed=i,
        )

    for i in range(n_test):
        spk_id = f'{2000 + i:04d}'
        generate_speaker_wav(
            test_root / spk_id / '10' / f'{spk_id}-10-0000.flac',
            speaker_index=i,
            duration=4.5,
            phase=0.0,
            seed=0,
        )
        generate_speaker_wav(
            test_root / spk_id / '10' / f'{spk_id}-10-0001.flac',
            speaker_index=i,
            duration=4.5,
            phase=0.7,
            seed=1,
        )
        generate_speaker_wav(
            test_root / spk_id / '11' / f'{spk_id}-11-0000.flac',
            speaker_index=i,
            duration=4.5,
            phase=1.4,
            seed=2,
        )
        generate_speaker_wav(
            test_root / spk_id / '12' / f'{spk_id}-12-0000.flac',
            speaker_index=i,
            duration=5.0,
            phase=2.1,
            seed=3,
        )

    return train_root, test_root
