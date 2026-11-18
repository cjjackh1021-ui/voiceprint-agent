"""Log-mel features shared by training and verification.

The mel matrix is created once with librosa and reused, so both paths see the same filters.
Waveforms are 16 kHz. Features are utterance-mean normalized.
"""
from pathlib import Path

import numpy as np
import torch

SR = 16000
N_MELS = 80
N_FFT = 512
HOP = 160
WIN = 400


def mel_matrix() -> torch.Tensor:
    path = Path(__file__).with_name('mel80.npy')
    if not path.exists():
        import librosa
        basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
        np.save(path, basis)
    return torch.from_numpy(np.load(path))


def fbank_batch(wav: torch.Tensor, mel: torch.Tensor | None = None) -> torch.Tensor:
    """wav (B, T) -> log-mel (B, n_mels, frames), mean-normalized over time."""
    if mel is None:
        mel = mel_matrix()
    mel = mel.to(wav.device)
    window = torch.hann_window(WIN, device=wav.device)
    spec = torch.stft(wav, N_FFT, HOP, WIN, window, return_complex=True, center=True)
    power = spec.abs().square()
    feat = torch.einsum('mf,bft->bmt', mel, power)
    feat = torch.log(feat + 1e-6)
    return feat - feat.mean(dim=-1, keepdim=True)
