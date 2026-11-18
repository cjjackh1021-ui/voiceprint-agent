"""Separate a mixture with the replicated 8 kHz Conv-TasNet checkpoint."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from voiceprint.audio import AudioError, file_digest

ROOT = Path(__file__).resolve().parents[1]
MAX_SECONDS = 180.0
TASNET_DIR = ROOT / 'replicate' / 'conv_tasnet' / 'Conv_TasNet_Pytorch'
CKPT = TASNET_DIR / 'Conv-TasNet-libri2mix' / 'best.pt'


def separate_file(src: str, dest_dir: Path) -> dict:
    if not CKPT.exists():
        raise FileNotFoundError(f'找不到分离模型：{CKPT}')
    info = sf.info(src)
    if info.duration > MAX_SECONDS:
        # The whole mixture goes through the network in one piece on the GPU.
        raise AudioError(f'混合录音长 {info.duration:.0f} 秒，分离一次最多处理 {MAX_SECONDS:.0f} 秒。')
    wav, sr = sf.read(src, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 8000:
        from math import gcd
        from scipy.signal import resample_poly
        factor = gcd(int(sr), 8000)
        wav = resample_poly(wav, 8000 // factor, int(sr) // factor).astype(np.float32)
    if str(TASNET_DIR) not in sys.path:
        sys.path.insert(0, str(TASNET_DIR))
    from Conv_TasNet import ConvTasNet
    net = ConvTasNet(N=512, L=16, B=128, H=512, P=3, X=8, R=3, norm='gln', num_spks=2, activate='relu', causal=False)
    state = torch.load(CKPT, map_location='cpu', weights_only=True)
    net.load_state_dict(state['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device).eval()
    mixture = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).to(device)
    length = mixture.numel()
    padded = F.pad(mixture, (0, (-length) % net.encoder.stride[0]))
    with torch.inference_mode():
        outputs = [out.reshape(-1)[:length].cpu().numpy() for out in net(padded)]
    dest_dir.mkdir(parents=True, exist_ok=True)
    # The content hash keeps two different mixtures with the same file name apart.
    tag = file_digest(src)[:10]
    paths = []
    for i, audio in enumerate(outputs, start=1):
        path = dest_dir / f'{Path(src).stem}_{tag}_spk{i}.wav'
        sf.write(path, audio, 8000)
        paths.append(str(path))
    return {'sample_rate': 8000, 'paths': paths, 'checkpoint': str(CKPT)}
