"""Train VPNet on LibriSpeech train-clean-100.

test-clean and dev-clean are never read here. Verification cases use those speakers.
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from voiceprint.audio import SR, list_flac
from voiceprint.features import fbank_batch, mel_matrix
from voiceprint.vpnet import AAMSoftmax, VPNet

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / 'data' / 'LibriSpeech' / 'train-clean-100'
CKPT = Path(__file__).resolve().parent / 'checkpoints' / 'vpnet.pt'


class CropSet(Dataset):
    def __init__(self, items: list[tuple[str, int]], seconds: float, seed: int):
        self.items = items
        self.n = int(seconds * SR)
        self.rng = random.Random(seed)
        self.noise = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        wav, sr = sf.read(path, dtype='float32', always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            raise RuntimeError(f'{path} is {sr} Hz, expected {SR}')
        if wav.shape[0] < self.n:
            wav = np.pad(wav, (0, self.n - wav.shape[0]))
        start = self.rng.randrange(0, wav.shape[0] - self.n + 1)
        clip = np.ascontiguousarray(wav[start:start + self.n], dtype=np.float32)
        if self.rng.random() < 0.5:
            clip = clip + np.float32(self.rng.uniform(0.002, 0.01)) * self.noise.standard_normal(clip.shape, dtype=np.float32)
        return torch.from_numpy(clip), label


def save_atomic(payload: dict, path: Path):
    # A kill during the write leaves only the .tmp file broken, never the previous checkpoint.
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_items(per_speaker: int, seed: int):
    rng = random.Random(seed)
    speakers = list_flac(str(TRAIN_ROOT))
    train, dev = [], []
    for spk in sorted(speakers):
        files = speakers[spk]
        if len(files) < 8:
            continue
        chosen = rng.sample(files, min(per_speaker, len(files)))
        dev_n = 2
        dev.extend((p, spk) for p in chosen[:dev_n])
        train.extend((p, spk) for p in chosen[dev_n:])
    names = sorted({s for _, s in train})
    index = {name: i for i, name in enumerate(names)}
    return ([(p, index[s]) for p, s in train],
            [(p, index[s]) for p, s in dev],
            names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--per-speaker', type=int, default=8)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_items, dev_items, names = build_items(args.per_speaker, args.seed)
    print(f'speakers {len(names)}  train clips {len(train_items)}  dev clips {len(dev_items)}  device {device}', flush=True)

    net = VPNet().to(device)
    head = AAMSoftmax(192, len(names)).to(device)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=args.lr)
    mel = mel_matrix()
    loader = DataLoader(CropSet(train_items, args.seconds, args.seed), batch_size=args.batch, shuffle=True, drop_last=True)

    for epoch in range(args.epochs):
        net.train()
        head.train()
        loader.dataset.rng.seed(args.seed + epoch + 1)
        total, correct, seen = 0.0, 0, 0
        start = time.time()
        for wav, labels in loader:
            wav = wav.to(device)
            labels = labels.to(device)
            embedding = net(fbank_batch(wav, mel))
            loss = head(embedding, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                cosine = torch.nn.functional.linear(
                    embedding, torch.nn.functional.normalize(head.weight, dim=-1))
                correct += int((cosine.argmax(dim=-1) == labels).sum())
            total += float(loss.detach()) * labels.size(0)
            seen += labels.size(0)
        if epoch == args.epochs // 2:
            for group in opt.param_groups:
                group['lr'] = args.lr / 10
        print(f'epoch {epoch:02d}  loss {total / seen:.3f}  train-acc {correct / seen:.3f}  {time.time() - start:.0f}s', flush=True)

    dev_acc = closed_set_accuracy(net, head, dev_items, mel, device, args.seconds)
    print(f'dev closed-set accuracy {dev_acc:.3f}', flush=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    save_atomic({
        'model': net.state_dict(),
        'speakers': names,
        'dev_closed_set_accuracy': dev_acc,
        'epochs': args.epochs,
        'per_speaker': args.per_speaker,
        'n_mels': 80,
        'channels': 128,
        'emb': 192,
        'dilations': [1, 2, 4, 8, 16, 32, 64, 128],
    }, CKPT)
    print(f'saved {CKPT}', flush=True)


@torch.no_grad()
def closed_set_accuracy(net, head, items, mel, device, seconds):
    net.eval()
    head.eval()
    if not items:
        return float('nan')
    loader = DataLoader(CropSet(items, seconds, 123), batch_size=64, shuffle=False)
    correct = seen = 0
    for wav, labels in loader:
        embedding = net(fbank_batch(wav.to(device), mel))
        cosine = torch.nn.functional.linear(embedding, torch.nn.functional.normalize(head.weight, dim=-1))
        correct += int((cosine.argmax(dim=-1).cpu() == labels).sum())
        seen += labels.size(0)
    return correct / max(seen, 1)


if __name__ == '__main__':
    main()
