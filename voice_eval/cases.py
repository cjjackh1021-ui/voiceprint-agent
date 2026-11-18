"""Build verification, identification, and multi-axis robustness cases from LibriSpeech speakers.

Public commands and hidden speaker labels are written to different files.
train-clean-100 is not used: those speakers train VPNet and the negative cohort.

Case set version 3 (2026-11-18): replaces v2 verify_short / verify_noise with deterministic
multi-axis robustness suites (noise, duration, channel, reverb, mixture) while keeping
verify, verify_impostor, identify, and enroll_quality protocol guarantees intact.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_eval.audio_synth import (
    add_babble_noise,
    add_pink_noise,
    add_white_noise,
    apply_reverb,
    apply_telephone_channel,
    clip_duration,
    mix_two_speakers,
)
from voiceprint.audio import list_flac

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / 'data' / 'LibriSpeech' / 'test-clean'
CASE_DIR = Path(__file__).resolve().parent / 'cases'
CASE_VERSION = 3

NOISE_TYPES = ('white', 'pink', 'babble')
SNR_LEVELS = (20, 10, 5, 0)
DURATIONS = (1.0, 1.5, 3.0, 5.0)
RT60_LEVELS = (0.3, 0.6)


def _quote(path: str) -> str:
    return '"' + path + '"'


def enroll_command(name: str, paths: list[str]) -> str:
    return '注册 ' + name + ' 本人同意 ' + ' '.join(_quote(p) for p in paths)


def _seed_for(*parts: object) -> int:
    digest = hashlib.sha256('|'.join(str(p) for p in parts).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


def _read_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def _write_wav(path: Path, wav: np.ndarray, sr: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(wav, dtype=np.float32), int(sr))
    return str(path)


def _gather(files: list[str], seconds: float = 30.0) -> list[str]:
    """At least three recordings and `seconds` in total, the minimum for enrolling from one folder."""
    picked, total = [], 0.0
    for path in files:
        picked.append(path)
        total += sf.info(path).duration
        if len(picked) >= 3 and total >= seconds:
            return picked
    return []


def _parts(files: list[str]) -> tuple[list[str], str, str]:
    """Enrollment recordings, a probe, and whether the probe shares a chapter with the enrollment."""
    by_chapter = defaultdict(list)
    for path in files:
        by_chapter[Path(path).parent.name].append(path)
    chapters = sorted(by_chapter)
    if len(chapters) >= 3 and len(by_chapter[chapters[0]]) >= 2:
        return by_chapter[chapters[0]][:2] + [by_chapter[chapters[1]][0]], by_chapter[chapters[2]][0], 'cross'
    if len(chapters) >= 2:
        for own, other in ((chapters[0], chapters[1]), (chapters[1], chapters[0])):
            enroll = _gather(by_chapter[own])
            if enroll:
                return enroll, by_chapter[other][0], 'cross'
    for chapter in chapters:
        enroll = _gather(by_chapter[chapter])
        if enroll and len(by_chapter[chapter]) > len(enroll):
            return enroll, by_chapter[chapter][len(enroll)], 'same'
    raise RuntimeError(f'not enough audio to build a case from {Path(files[0]).parent.parent}')


def _same_folder_short(out_dir: Path, split: str, name: str, source: str) -> list[str]:
    """Three two-second clips in one folder. Enrollment must refuse this."""
    folder = out_dir / 'audio' / split / f'{name}-one-take'
    folder.mkdir(parents=True, exist_ok=True)
    wav, sr = _read_mono(source)
    paths = []
    for index in range(3):
        piece = wav[index * sr:(index + 2) * sr]
        if piece.shape[0] < sr:
            piece = np.pad(piece, (0, sr - piece.shape[0]))
        path = folder / f'part{index}.wav'
        sf.write(path, piece, sr)
        paths.append(str(path))
    return paths


def _sample_babble_interferers(
    names: list[str],
    name: str,
    clean_probes: dict[str, tuple[np.ndarray, int]],
    split: str,
    snr: int,
) -> list[np.ndarray]:
    """Sample up to 3 interferers excluding both probe owner and paired impostor claimant."""
    idx = names.index(name)
    prev_name = names[(idx - 1) % len(names)]
    pool = [spk for spk in names if spk not in (name, prev_name)]
    if len(pool) < 3:
        pool = [spk for spk in names if spk != name] or list(names)
    rng = np.random.default_rng(_seed_for('babble_pick', split, name, snr))
    chosen = rng.choice(len(pool), size=min(3, len(pool)), replace=False)
    return [clean_probes[pool[int(i)]][0] for i in chosen]


def _synthesize_noise_probe(
    noise_type: str,
    snr: int,
    name: str,
    names: list[str],
    clean_probes: dict[str, tuple[np.ndarray, int]],
    split: str,
) -> tuple[np.ndarray, int]:
    wav, sr = clean_probes[name]
    seed = _seed_for('noise', split, noise_type, snr, name)
    if noise_type == 'white':
        return add_white_noise(wav, snr, seed), sr
    if noise_type == 'pink':
        return add_pink_noise(wav, snr, seed), sr
    if noise_type == 'babble':
        interferers = _sample_babble_interferers(names, name, clean_probes, split, snr)
        return add_babble_noise(wav, interferers, snr, seed), sr
    raise ValueError(f'unknown noise type: {noise_type}')


def _add_verify_condition(
    add_fn,
    names: list[str],
    prepared: dict[str, tuple[list[str], str, str]],
    probe_map: dict[str, str],
    suite: str,
    id_prefix: str,
    axis: str,
    variant: str,
) -> None:
    for idx, name in enumerate(names):
        enroll = prepared[name][0]
        other = names[(idx + 1) % len(names)]
        add_fn(
            f'{id_prefix}-accept-{name}',
            suite,
            [enroll_command(name, enroll), f'验证 {name} {_quote(probe_map[name])}'],
            'accept',
            name,
            name,
            axis=axis,
            variant=variant,
        )
        add_fn(
            f'{id_prefix}-reject-{name}',
            suite,
            [enroll_command(name, enroll), f'验证 {name} {_quote(probe_map[other])}'],
            'reject',
            other,
            name,
            axis=axis,
            variant=variant,
        )


def build_split(
    names: list[str],
    speakers: dict[str, list[str]],
    split: str,
    case_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    out_dir = Path(case_dir) if case_dir is not None else CASE_DIR
    audio_dir = out_dir / 'audio' / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    prepared = {name: _parts(speakers[name]) for name in names}
    clean_probes = {name: _read_mono(prepared[name][1]) for name in names}
    public, hidden = [], []

    def add(case_id, suite, steps, expect, true_speaker, claimed, **extra_hidden):
        public.append({'id': case_id, 'suite': suite, 'split': split, 'steps': steps})
        entry = {
            'id': case_id,
            'suite': suite,
            'split': split,
            'expect': expect,
            'true_speaker': true_speaker,
            'claimed': claimed,
            'probe_session': prepared[true_speaker][2],
        }
        entry.update(extra_hidden)
        hidden.append(entry)

    for idx, name in enumerate(names):
        enroll, probe, _session = prepared[name]
        other = names[(idx + 1) % len(names)]
        add(
            f'verify-accept-{name}',
            'verify',
            [enroll_command(name, enroll), f'验证 {name} {_quote(probe)}'],
            'accept',
            name,
            name,
        )
        add(
            f'verify-reject-{name}',
            'verify',
            [enroll_command(name, enroll), f'验证 {name} {_quote(prepared[other][1])}'],
            'reject',
            other,
            name,
        )
        for impostor in names:
            if impostor not in (name, other):
                add(
                    f'verify-impostor-{name}-{impostor}',
                    'verify_impostor',
                    [enroll_command(name, enroll), f'验证 {name} {_quote(prepared[impostor][1])}'],
                    'reject',
                    impostor,
                    name,
                )
        if idx < 5:
            weak = _same_folder_short(out_dir, split, name, enroll[0])
            add(f'enroll-same-session-{name}', 'enroll_quality', [enroll_command(name, weak)], 'reject', name, name)

    for start in range(min(10, max(0, len(names) - 6))):
        gallery = names[start:start + 6]
        target = gallery[0]
        probe = prepared[target][1]
        steps = [enroll_command(name, prepared[name][0]) for name in gallery]
        steps.append(f'辨认 {_quote(probe)}')
        add(f'identify-accept-{target}', 'identify', steps, 'accept', target, target)
        outsider = names[start + 6]
        steps = [enroll_command(name, prepared[name][0]) for name in gallery]
        steps.append(f'辨认 {_quote(prepared[outsider][1])}')
        add(f'identify-reject-{target}', 'identify', steps, 'unknown', outsider, None)

    for noise_type in NOISE_TYPES:
        for snr in SNR_LEVELS:
            variant = f'{noise_type}_snr{snr}'
            probe_map = {}
            for name in names:
                wav, sr = _synthesize_noise_probe(noise_type, snr, name, names, clean_probes, split)
                probe_map[name] = _write_wav(audio_dir / f'{name}-{variant}.wav', wav, sr)
            _add_verify_condition(
                add, names, prepared, probe_map,
                'robustness_noise', f'robustness-noise-{variant}', 'noise', variant,
            )

    for sec in DURATIONS:
        variant = f'dur_{sec:.1f}s'
        probe_map = {
            name: _write_wav(audio_dir / f'{name}-{variant}.wav', clip_duration(wav, sr, sec), sr)
            for name, (wav, sr) in clean_probes.items()
        }
        _add_verify_condition(
            add, names, prepared, probe_map,
            'robustness_duration', f'robustness-duration-{variant}', 'duration', variant,
        )

    channel_variant = 'tel8k_ulaw'
    channel_map = {}
    for name, (wav, sr) in clean_probes.items():
        tel_wav, tel_sr = apply_telephone_channel(wav, sr)
        channel_map[name] = _write_wav(audio_dir / f'{name}-{channel_variant}.wav', tel_wav, tel_sr)
    _add_verify_condition(
        add, names, prepared, channel_map,
        'robustness_channel', f'robustness-channel-{channel_variant}', 'channel', channel_variant,
    )

    for rt in RT60_LEVELS:
        variant = f'rt60_{rt:.1f}s'
        probe_map = {
            name: _write_wav(
                audio_dir / f'{name}-{variant}.wav',
                apply_reverb(wav, sr, rt, _seed_for('reverb', split, variant, name)),
                sr,
            )
            for name, (wav, sr) in clean_probes.items()
        }
        _add_verify_condition(
            add, names, prepared, probe_map,
            'robustness_reverb', f'robustness-reverb-{variant}', 'reverb', variant,
        )

    n_mix = min(5, max(0, len(names) - 5))
    for idx in range(n_mix):
        start = idx * 2 if idx * 2 + 6 <= len(names) else min(idx, len(names) - 6)
        gallery = names[start:start + 6]
        spk_a, spk_b = gallery[0], gallery[1]
        wav_a, sr_a = clean_probes[spk_a]
        wav_b, sr_b = clean_probes[spk_b]
        mix_wav, mix_sr = mix_two_speakers(wav_a, sr_a, wav_b, sr_b, target_sr=8000)
        mix_path = _write_wav(audio_dir / f'mix-{idx:02d}-{spk_a}-{spk_b}.wav', mix_wav, mix_sr)
        steps = [enroll_command(name, prepared[name][0]) for name in gallery]
        steps.append(f'分离再辨认 {_quote(mix_path)}')
        add(
            f'robustness-mixture-{idx:02d}-{spk_a}-{spk_b}',
            'robustness_mixture',
            steps,
            'accept',
            spk_a,
            spk_a,
            axis='mixture',
            variant='mix2_0db_8k',
            expect_speakers=[spk_a, spk_b],
            gallery_speakers=list(gallery),
        )

    return public, hidden


def build(test_root: Path | None = None, case_dir: Path | None = None) -> dict:
    root = Path(test_root) if test_root is not None else TEST_ROOT
    out_dir = Path(case_dir) if case_dir is not None else CASE_DIR
    speakers = list_flac(str(root))
    names = sorted(speakers)
    if len(names) < 40:
        raise RuntimeError(f'expected at least 40 test speakers, found {len(names)}')
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, group in (('validation', names[:20]), ('locked', names[20:40])):
        public, hidden = build_split(group, speakers, split, case_dir=out_dir)
        (out_dir / f'{split}.public.json').write_text(
            json.dumps(public, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        (out_dir / f'{split}.hidden.json').write_text(
            json.dumps(hidden, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        counts[split] = len(public)
        counts[f'{split}_same_session_probes'] = sum(
            row['probe_session'] == 'same' for row in hidden if row['id'].startswith('verify-accept-')
        )
    (out_dir / 'VERSION').write_text(f'{CASE_VERSION}\n', encoding='utf-8')
    return counts


def case_version(case_dir: Path | None = None) -> int:
    out_dir = Path(case_dir) if case_dir is not None else CASE_DIR
    path = out_dir / 'VERSION'
    return int(path.read_text(encoding='utf-8').strip()) if path.exists() else 1


def load(split: str, case_dir: Path | None = None) -> list[dict]:
    out_dir = Path(case_dir) if case_dir is not None else CASE_DIR
    public = json.loads((out_dir / f'{split}.public.json').read_text(encoding='utf-8'))
    hidden = {row['id']: row for row in json.loads((out_dir / f'{split}.hidden.json').read_text(encoding='utf-8'))}
    cases = []
    for row in public:
        item = dict(row)
        item['hidden'] = hidden[row['id']]
        cases.append(item)
    return cases
