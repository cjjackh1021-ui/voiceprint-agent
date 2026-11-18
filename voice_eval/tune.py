"""Search verify thresholds on the validation split only. Locked cases are never read.

    python -m voice_eval.tune --embedder vpnet

Prints one line per (alpha, band). Nothing is written: to adopt a setting, edit PROFILES in
voiceprint/scoring.py and rerun the validation report.
"""
import argparse
import shlex
from pathlib import Path

import numpy as np

from voice_eval.cases import load
from voiceprint.embed import cohort_matrix, load_embedder
from voiceprint.scoring import decide, profile_for

TRAIN = Path(__file__).resolve().parents[1] / 'data' / 'LibriSpeech' / 'train-clean-100'
SUITES = ('verify', 'verify_impostor')


def pairs(embedder):
    found = []
    for case in load('validation'):
        if case['suite'] not in SUITES:
            continue
        paths = shlex.split(case['steps'][0], posix=True)[3:]
        probe = shlex.split(case['steps'][1], posix=True)[2]
        vectors = np.stack([embedder.embed_file(path) for path in paths])
        found.append((case['hidden']['expect'], vectors, embedder.embed_file(probe)))
    return found


def evaluate(rows, cohort, profile, alpha, band):
    correct = fa = fr = review = n_acc = n_rej = 0
    for expect, segments, probe in rows:
        # The overrides are passed in: decide() reads its settings from PROFILES, so assigning
        # module globals (what this script used to do) changed nothing.
        decision = decide(probe, segments, cohort, task='verify', profile=profile, alpha=alpha, band=band)['decision']
        n_acc += expect == 'accept'
        n_rej += expect == 'reject'
        review += decision == 'uncertain'
        if decision == 'uncertain':
            continue
        correct += decision == expect
        fa += expect == 'reject' and decision == 'accept'
        fr += expect == 'accept' and decision == 'reject'
    decided = len(rows) - review
    auto = correct / decided if decided else 0
    return auto, correct / len(rows), fa / max(n_rej, 1), fr / max(n_acc, 1), review / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder', default='vpnet', choices=['stats', 'vpnet', 'ecapa'])
    args = parser.parse_args()
    embedder = load_embedder(args.embedder)
    cohort, _names = cohort_matrix(embedder, TRAIN)
    rows = pairs(embedder)
    current = profile_for(embedder.name)
    print(f'{embedder.name}: cohort {len(cohort)}, validation trials {len(rows)}, '
          f'current alpha {current["alpha"]} band {current["band"]}', flush=True)
    for alpha in (0.15, 0.25, 0.35, 0.45, 0.6):
        for band in (0.0, 0.2, 0.35):
            auto, strict, fa, fr, review = evaluate(rows, cohort, embedder.name, alpha, band)
            print(f'alpha {alpha:.2f} band {band:.2f} auto {auto:.3f} strict {strict:.3f} '
                  f'FA {fa:.3f} FR {fr:.3f} review {review:.3f}', flush=True)


if __name__ == '__main__':
    main()
