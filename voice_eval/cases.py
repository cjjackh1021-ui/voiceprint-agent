"""Build verification cases from LibriSpeech test speakers.

Public commands and hidden speaker labels are written to different files.
train-clean-100 is not used: those speakers train VPNet and the negative cohort.

Case set version 2 (2026-11-10): probes come from a chapter that enrollment did not use, every
other speaker is tried as an impostor, and the short and noisy suites contain impostors too.
Reports made from version 1 cases are not comparable with reports made from these.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceprint.audio import list_flac

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / 'data' / 'LibriSpeech' / 'test-clean'
CASE_DIR = Path(__file__).resolve().parent / 'cases'
CASE_VERSION = 2


def _quote(path: str) -> str:
    return '"' + path + '"'


def enroll_command(name: str, paths: list[str]) -> str:
    return '注册 ' + name + ' 本人同意 ' + ' '.join(_quote(p) for p in paths)


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
    """Enrollment recordings, a probe, and whether the probe shares a chapter with the enrollment.

    A chapter is the closest thing to a recording session in LibriSpeech. The probe comes from a
    chapter that enrollment never touches whenever the speaker has one: with three chapters,
    enrollment takes two and the probe the third; with two, enrollment is gathered from one
    chapter and the probe comes from the other. Only a single-chapter speaker falls back to a
    probe from the enrollment chapter, and the case is marked 'same' so reports can separate it.
    """
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


def _variants(split: str, name: str, probe: str) -> tuple[str, str]:
    folder = CASE_DIR / 'audio' / split
    folder.mkdir(parents=True, exist_ok=True)
    wav, sr = sf.read(probe, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    short_n = int(1.5 * sr)
    short = wav[:short_n] if wav.shape[0] >= short_n else np.pad(wav, (0, short_n - wav.shape[0]))
    short_path = folder / f'{name}-short.wav'
    sf.write(short_path, short, sr)
    rms = float(np.sqrt(np.mean(np.square(wav))) + 1e-8)
    noise = np.random.default_rng(int(name) if name.isdigit() else 1).normal(0.0, rms / (10 ** 0.5), size=wav.shape[0])
    noisy = np.clip(wav + noise.astype(np.float32), -1.0, 1.0)
    noise_path = folder / f'{name}-noise.wav'
    sf.write(noise_path, noisy.astype(np.float32), sr)
    return str(short_path), str(noise_path)


def _same_folder_short(split: str, name: str, source: str) -> list[str]:
    """Three two-second clips in one folder. Enrollment must refuse this."""
    folder = CASE_DIR / 'audio' / split / f'{name}-one-take'
    folder.mkdir(parents=True, exist_ok=True)
    wav, sr = sf.read(source, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    paths = []
    for index in range(3):
        piece = wav[index * sr:(index + 2) * sr]
        if piece.shape[0] < sr:
            piece = np.pad(piece, (0, sr - piece.shape[0]))
        path = folder / f'part{index}.wav'
        sf.write(path, piece, sr)
        paths.append(str(path))
    return paths


def build_split(names: list[str], speakers: dict[str, list[str]], split: str) -> tuple[list[dict], list[dict]]:
    prepared = {name: _parts(speakers[name]) for name in names}
    variants = {name: _variants(split, name, prepared[name][1]) for name in names}
    public, hidden = [], []

    def add(case_id, suite, steps, expect, true_speaker, claimed):
        public.append({'id': case_id, 'suite': suite, 'split': split, 'steps': steps})
        hidden.append({'id': case_id, 'suite': suite, 'split': split, 'expect': expect,
                       'true_speaker': true_speaker, 'claimed': claimed,
                       'probe_session': prepared[true_speaker][2]})

    for name in names:
        enroll, probe, _session = prepared[name]
        other = names[(names.index(name) + 1) % len(names)]
        add(f'verify-accept-{name}', 'verify',
            [enroll_command(name, enroll), f'验证 {name} {_quote(probe)}'],
            'accept', name, name)
        add(f'verify-reject-{name}', 'verify',
            [enroll_command(name, enroll), f'验证 {name} {_quote(prepared[other][1])}'],
            'reject', other, name)
        # Every remaining speaker also claims this name. The balanced 'verify' suite above stays the
        # primary metric; this one exists to measure the false-accept rate on more than 20 trials.
        for impostor in names:
            if impostor not in (name, other):
                add(f'verify-impostor-{name}-{impostor}', 'verify_impostor',
                    [enroll_command(name, enroll), f'验证 {name} {_quote(prepared[impostor][1])}'],
                    'reject', impostor, name)
        short_path, noise_path = variants[name]
        other_short, other_noise = variants[other]
        add(f'verify-short-{name}', 'verify_short',
            [enroll_command(name, enroll), f'验证 {name} {_quote(short_path)}'],
            'accept', name, name)
        add(f'verify-short-reject-{name}', 'verify_short',
            [enroll_command(name, enroll), f'验证 {name} {_quote(other_short)}'],
            'reject', other, name)
        add(f'verify-noise-{name}', 'verify_noise',
            [enroll_command(name, enroll), f'验证 {name} {_quote(noise_path)}'],
            'accept', name, name)
        add(f'verify-noise-reject-{name}', 'verify_noise',
            [enroll_command(name, enroll), f'验证 {name} {_quote(other_noise)}'],
            'reject', other, name)
        if names.index(name) < 5:
            weak = _same_folder_short(split, name, enroll[0])
            add(f'enroll-same-session-{name}', 'enroll_quality',
                [enroll_command(name, weak)],
                'reject', name, name)

    for start in range(10):
        gallery = names[start:start + 6]
        target = gallery[0]
        enroll, probe, _session = prepared[target]
        steps = [enroll_command(name, prepared[name][0]) for name in gallery]
        steps.append(f'辨认 {_quote(probe)}')
        add(f'identify-accept-{target}', 'identify', steps, 'accept', target, target)
        outsider = names[start + 6]
        steps = [enroll_command(name, prepared[name][0]) for name in gallery]
        steps.append(f'辨认 {_quote(prepared[outsider][1])}')
        add(f'identify-reject-{target}', 'identify', steps, 'unknown', outsider, None)
    return public, hidden


def build() -> dict:
    speakers = list_flac(str(TEST_ROOT))
    names = sorted(speakers)
    if len(names) < 40:
        raise RuntimeError(f'expected at least 40 test speakers, found {len(names)}')
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, group in (('validation', names[:20]), ('locked', names[20:40])):
        public, hidden = build_split(group, speakers, split)
        (CASE_DIR / f'{split}.public.json').write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding='utf-8')
        (CASE_DIR / f'{split}.hidden.json').write_text(json.dumps(hidden, ensure_ascii=False, indent=2), encoding='utf-8')
        counts[split] = len(public)
        counts[f'{split}_same_session_probes'] = sum(
            row['probe_session'] == 'same' for row in hidden if row['id'].startswith('verify-accept-'))
    (CASE_DIR / 'VERSION').write_text(f'{CASE_VERSION}\n', encoding='utf-8')
    return counts


def case_version() -> int:
    path = CASE_DIR / 'VERSION'
    return int(path.read_text(encoding='utf-8').strip()) if path.exists() else 1


def load(split: str) -> list[dict]:
    public = json.loads((CASE_DIR / f'{split}.public.json').read_text(encoding='utf-8'))
    hidden = {row['id']: row for row in json.loads((CASE_DIR / f'{split}.hidden.json').read_text(encoding='utf-8'))}
    cases = []
    for row in public:
        item = dict(row)
        item['hidden'] = hidden[row['id']]
        cases.append(item)
    return cases
