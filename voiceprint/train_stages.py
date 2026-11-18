"""Three-stage VPNet training on LibriSpeech train-clean-100.

pretrain: contrastive NT-Xent on two augmented crops, no speaker labels.
finetune: load that backbone and train AAM-Softmax with speaker labels.
posttrain: large-margin fine-tuning, a short low-LR pass with a larger angular margin.

test-clean and dev-clean are never read. The previous from-scratch vpnet.pt is left in place.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from voiceprint.audio import SR, list_flac
from voiceprint.features import fbank_batch, mel_matrix
from voiceprint.train_vpnet import CropSet, build_items, closed_set_accuracy, save_atomic
from voiceprint.vpnet import AAMSoftmax, VPNet

CKPT_DIR = Path(__file__).resolve().parent / 'checkpoints'
PATHS = {
    'pretrain': CKPT_DIR / 'vpnet_pretrain.pt',
    'finetune': CKPT_DIR / 'vpnet_finetune.pt',
    'posttrain': CKPT_DIR / 'vpnet_posttrain.pt',
}


def read_clip(path: str, n: int, rng: random.Random, noise: np.random.Generator | None) -> np.ndarray:
    wav, sr = sf.read(path, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise RuntimeError(f'{path} is {sr} Hz, expected {SR}')
    if wav.shape[0] < n:
        wav = np.pad(wav, (0, n - wav.shape[0]))
    start = rng.randrange(0, wav.shape[0] - n + 1)
    clip = np.ascontiguousarray(wav[start:start + n], dtype=np.float32)
    if noise is not None and rng.random() < 0.8:
        clip = clip + np.float32(rng.uniform(0.002, 0.02)) * noise.standard_normal(clip.shape, dtype=np.float32)
    return clip


class TwoViewSet(Dataset):
    def __init__(self, items, seconds: float, seed: int):
        self.items = items
        self.n = int(seconds * SR)
        self.rng = random.Random(seed)
        self.noise = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        path = self.items[index][0]
        return (torch.from_numpy(read_clip(path, self.n, self.rng, self.noise)),
                torch.from_numpy(read_clip(path, self.n, self.rng, self.noise)))


def nt_xent(first: torch.Tensor, second: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z = torch.cat([F.normalize(first, dim=-1), F.normalize(second, dim=-1)], dim=0)
    logits = z @ z.T / temperature
    n = first.size(0)
    labels = torch.cat([torch.arange(n, device=z.device) + n, torch.arange(n, device=z.device)])
    logits = logits.masked_fill(torch.eye(2 * n, device=z.device, dtype=torch.bool), -1e9)
    return F.cross_entropy(logits, labels)


def save_model(path: Path, net, head, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': net.state_dict(),
        'n_mels': 80,
        'channels': 128,
        'emb': 192,
        'dilations': [1, 2, 4, 8, 16, 32, 64, 128],
        **meta,
    }
    if head is not None:
        payload['aam'] = head.state_dict()
        payload['margin'] = head.margin
        payload['speakers'] = meta.get('speakers', [])
    save_atomic(payload, path)
    print(f'saved {path}', flush=True)


def run_pretrain(items, device, mel, epochs, batch, seconds, lr):
    net = VPNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loader = DataLoader(TwoViewSet(items, seconds, 0), batch_size=batch, shuffle=True, drop_last=True)
    for epoch in range(epochs):
        net.train()
        loader.dataset.rng.seed(epoch + 1)
        total, seen, start = 0.0, 0, time.time()
        for left, right in loader:
            feats = fbank_batch(torch.cat([left, right], dim=0).to(device), mel)
            emb = net(feats)
            loss = nt_xent(emb[:left.size(0)], emb[left.size(0):])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * left.size(0)
            seen += left.size(0)
        print(f'pretrain {epoch:02d}  loss {total / seen:.3f}  {time.time() - start:.0f}s', flush=True)
    return net


def run_supervised(net, head, items, dev_items, device, mel, epochs, batch, seconds, lr, tag):
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=lr)
    loader = DataLoader(CropSet(items, seconds, 0), batch_size=batch, shuffle=True, drop_last=True)
    for epoch in range(epochs):
        net.train()
        head.train()
        loader.dataset.rng.seed(100 + epoch)
        total, correct, seen, start = 0.0, 0, 0, time.time()
        for wav, labels in loader:
            wav, labels = wav.to(device), labels.to(device)
            embedding = net(fbank_batch(wav, mel))
            loss = head(embedding, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                cosine = F.linear(embedding, F.normalize(head.weight, dim=-1))
                correct += int((cosine.argmax(dim=-1) == labels).sum())
            total += float(loss.detach()) * labels.size(0)
            seen += labels.size(0)
        print(f'{tag} {epoch:02d}  loss {total / max(seen, 1):.3f}  train-acc {correct / max(seen, 1):.3f}  '
              f'{time.time() - start:.0f}s', flush=True)
    acc = closed_set_accuracy(net, head, dev_items, mel, device, seconds)
    print(f'{tag} dev closed-set accuracy {acc:.3f}', flush=True)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-speaker', type=int, default=16)
    parser.add_argument('--batch', type=int, default=48)
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--pretrain-epochs', type=int, default=4)
    parser.add_argument('--finetune-epochs', type=int, default=6)
    parser.add_argument('--posttrain-epochs', type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_items, dev_items, names = build_items(args.per_speaker, 0)
    print(f'speakers {len(names)} train {len(train_items)} dev {len(dev_items)} device {device}', flush=True)
    mel = mel_matrix()

    net = run_pretrain(train_items, device, mel, args.pretrain_epochs, args.batch, args.seconds, 1e-3)
    save_model(PATHS['pretrain'], net, None, {'stage': 'pretrain', 'epochs': args.pretrain_epochs})

    head = AAMSoftmax(192, len(names), margin=0.2).to(device)
    dev_acc = run_supervised(net, head, train_items, dev_items, device, mel,
                             args.finetune_epochs, args.batch, args.seconds, 3e-4, 'finetune')
    save_model(PATHS['finetune'], net, head, {
        'stage': 'finetune', 'epochs': args.finetune_epochs, 'margin': 0.2,
        'speakers': names, 'dev_closed_set_accuracy': dev_acc,
    })

    head.margin = 0.4
    dev_acc = run_supervised(net, head, train_items, dev_items, device, mel,
                             args.posttrain_epochs, args.batch, args.seconds, 1e-5, 'posttrain')
    save_model(PATHS['posttrain'], net, head, {
        'stage': 'posttrain', 'epochs': args.posttrain_epochs, 'margin': 0.4,
        'speakers': names, 'dev_closed_set_accuracy': dev_acc,
    })


if __name__ == '__main__':
    main()
