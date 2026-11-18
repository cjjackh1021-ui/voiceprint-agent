"""File-level speaker embeddings.

stats: MFCC statistics. No training, used as the baseline.
vpnet: the trained network. Its classifier head is not used at verification time.
ecapa: SpeechBrain ECAPA-TDNN, loaded only from local files whose hashes are pinned below.

Every embedder has a model_id (a hash of its weights). Templates and cohorts are tied to it,
so vectors from different models are never compared with each other.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from voiceprint.audio import file_digest, load_wav
from voiceprint.features import SR, fbank_batch, mel_matrix
from voiceprint.vpnet import VPNet

CKPT_DIR = Path(__file__).resolve().parent / 'checkpoints'
CKPT = CKPT_DIR / 'vpnet_posttrain.pt'
if not CKPT.exists():
    CKPT = CKPT_DIR / 'vpnet.pt'
COHORT_DIR = Path(__file__).resolve().parent / 'cohorts'
ECAPA_DIR = CKPT_DIR / 'ecapa'
ECAPA_REPO = 'speechbrain/spkrec-ecapa-voxceleb'
ECAPA_REVISION = '0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'
# sha256 of the files at that revision. The two large ones equal the LFS oids published on
# huggingface.co (checked 2026-11-10); hyperparams.yaml is pinned because loading it builds Python objects.
ECAPA_SHA256 = {
    'hyperparams.yaml': '6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41',
    'embedding_model.ckpt': '0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2',
    'mean_var_norm_emb.ckpt': 'cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33',
    'classifier.ckpt': 'fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535',
    'label_encoder.ckpt': 'e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9',
}
CACHE_LIMIT = 4096


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.clip(n, 1e-8, None)).astype(np.float32)


class Embedder:
    name = 'base'
    dim = 0
    checkpoint = None
    model_id = ''
    cohort_size = 0  # training speakers in the negative cohort; 0 means all of them

    def __init__(self):
        self._cache: dict[tuple, np.ndarray] = {}

    def embed_file(self, path: str) -> np.ndarray:
        # The key includes size and modification time: a file re-recorded at the same path is embedded again.
        stat = os.stat(path)
        key = (os.path.normcase(os.path.abspath(path)), stat.st_mtime_ns, stat.st_size)
        if key not in self._cache:
            if len(self._cache) >= CACHE_LIMIT:
                self._cache.clear()
            self._cache[key] = self._embed(path)
        return self._cache[key].copy()

    def _embed(self, path: str) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {'embedder': self.name, 'dim': self.dim, 'checkpoint': self.checkpoint, 'model_id': self.model_id}


class StatsEmbedder(Embedder):
    """MFCC mean, standard deviation and deltas, whitened on training speakers.

    Whole-utterance log-mel means are almost the same for every speaker, so cosine
    cannot separate them. Cepstra with the cohort mean removed are the statistical baseline.
    """

    name = 'stats'
    dim = 76
    cohort_size = 40  # the first 40 speakers; whitening is fit on speakers 40:120

    def __init__(self):
        super().__init__()
        self.mu, self.sd = self._whiten()
        self.model_id = hashlib.sha256(b'stats-mfcc19-v1' + self.mu.tobytes() + self.sd.tobytes()).hexdigest()

    def _embed(self, path: str) -> np.ndarray:
        return _normalize((self._raw(path) - self.mu) / self.sd)

    def _raw(self, path: str) -> np.ndarray:
        import librosa
        wav = load_wav(path, max_seconds=12)
        mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=20)[1:]
        delta = librosa.feature.delta(mfcc)
        return np.concatenate([mfcc.mean(1), mfcc.std(1), delta.mean(1), delta.std(1)]).astype(np.float64)

    def _whiten(self) -> tuple[np.ndarray, np.ndarray]:
        path = COHORT_DIR / 'stats_whiten.npz'
        if path.exists():
            data = np.load(path, allow_pickle=False)
            return data['mu'].astype(np.float64), data['sd'].astype(np.float64)
        from voiceprint.audio import list_flac
        train = Path(__file__).resolve().parents[1] / 'data' / 'LibriSpeech' / 'train-clean-100'
        speakers = list_flac(str(train))
        # Speakers 40:120. The negative cohort uses the first 40, so whitening is not fit on it.
        names = sorted(speakers)[40:120]
        rows = [self._raw(speakers[name][0]) for name in names]
        mu = np.mean(rows, axis=0).astype(np.float32)
        sd = np.clip(np.std(rows, axis=0), 1e-3, None).astype(np.float32)
        COHORT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, mu=mu, sd=sd)
        return mu.astype(np.float64), sd.astype(np.float64)


class VPNetEmbedder(Embedder):
    name = 'vpnet'
    dim = 192

    def __init__(self, checkpoint: Path = CKPT):
        super().__init__()
        # weights_only: a checkpoint is data. A modified file must not be able to run code.
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('emb') != 192 or state.get('n_mels') != 80:
            raise RuntimeError(f'unexpected VPNet checkpoint config in {checkpoint}')
        self.net = VPNet()
        self.net.load_state_dict(state['model'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device).eval()
        self.checkpoint = str(checkpoint)
        self.model_id = file_digest(str(checkpoint))
        self._mel = mel_matrix()

    def _embed(self, path: str) -> np.ndarray:
        wav = load_wav(path)
        win = 4 * SR
        if wav.shape[0] <= win:
            # A short recording goes through at its own length. Padding it to 4 s with zeros, which
            # training never produced, cut accepted 1.5 s genuine probes from 12 to 5 of 20 on validation.
            pieces = [wav]
        else:
            starts = [0, (wav.shape[0] - win) // 2, wav.shape[0] - win]
            pieces = [wav[s:s + win] for s in starts]
        vectors = [self._forward(piece) for piece in pieces]
        return _normalize(np.mean(vectors, axis=0))

    def _forward(self, wav: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            feat = fbank_batch(torch.from_numpy(wav).unsqueeze(0).to(self.device), self._mel)
            emb = self.net(feat)
        return emb.squeeze(0).cpu().numpy()


def ecapa_files_present() -> bool:
    return all((ECAPA_DIR / name).exists() for name in ECAPA_SHA256)


def _verify_ecapa_files():
    for name, expected in ECAPA_SHA256.items():
        found = file_digest(os.path.realpath(ECAPA_DIR / name))
        if found != expected:
            raise RuntimeError(
                f'ECAPA 文件 {name} 的 sha256 与固定值不一致（{found[:12]}… 应为 {expected[:12]}…），拒绝加载。')


class EcapaEmbedder(Embedder):
    """SpeechBrain ECAPA-TDNN trained on VoxCeleb.

    The files are fetched once at a pinned revision, checked against ECAPA_SHA256, and from then
    on loaded with the network switched off.
    """

    name = 'ecapa'
    dim = 192

    def __init__(self):
        super().__init__()
        if not ecapa_files_present():
            os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
            from huggingface_hub import snapshot_download
            ECAPA_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_download(ECAPA_REPO, revision=ECAPA_REVISION, local_dir=str(ECAPA_DIR),
                              allow_patterns=list(ECAPA_SHA256))
        # Checked before SpeechBrain reads hyperparams.yaml, and with no later download to replace them.
        _verify_ecapa_files()
        os.environ['HF_HUB_OFFLINE'] = '1'
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EncoderClassifier.from_hparams(
            source=str(ECAPA_DIR),
            savedir=str(ECAPA_DIR),
            overrides={'pretrained_path': str(ECAPA_DIR)},
            run_opts={'device': 'cuda:0' if self.device.type == 'cuda' else 'cpu'},
        )
        self.checkpoint = str(ECAPA_DIR)
        self.model_id = ECAPA_SHA256['embedding_model.ckpt']

    def _embed(self, path: str) -> np.ndarray:
        wav = torch.from_numpy(load_wav(path, max_seconds=20.0)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            emb = self.model.encode_batch(wav)
        return _normalize(emb.squeeze().detach().cpu().numpy())


def default_embedder_name() -> str:
    """ECAPA when its files are already on disk (it leads every locked suite), otherwise VPNet."""
    return 'ecapa' if ecapa_files_present() else 'vpnet'


def load_embedder(name: str) -> Embedder:
    if name == 'auto':
        name = default_embedder_name()
    if name == 'stats':
        return StatsEmbedder()
    if name == 'vpnet':
        if not CKPT.exists():
            raise FileNotFoundError(f'VPNet checkpoint missing: {CKPT}')
        return VPNetEmbedder(CKPT)
    if name == 'ecapa':
        return EcapaEmbedder()
    raise ValueError(f'unknown embedder {name}')


def cohort_matrix(embedder: Embedder, train_root: Path, n: int | None = None) -> tuple[np.ndarray, list[str]]:
    """One utterance from each training speaker. n defaults to the embedder's cohort_size.

    The file name carries the model_id, so a new checkpoint builds its own cohort instead of
    silently reusing vectors from the previous one.
    """
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    if n is None:
        n = embedder.cohort_size
    suffix = '' if n <= 0 else f'-n{n}'
    path = COHORT_DIR / f'{embedder.name}-{embedder.model_id[:16]}{suffix}.npz'
    if path.exists():
        data = np.load(path, allow_pickle=False)
        return data['emb'].astype(np.float32), [str(s) for s in data['speakers'].tolist()]
    from voiceprint.audio import list_flac
    speakers = list_flac(str(train_root))
    names = sorted(speakers)
    if n > 0:
        names = names[:n]
    rows = [embedder.embed_file(speakers[name][0]) for name in names]
    emb = np.stack(rows).astype(np.float32)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, emb=emb, speakers=np.array(names))
    os.replace(tmp, path)
    return emb, names
