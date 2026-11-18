"""Dimension 7: multi-axis distortion degradation, fault-injection recovery, and repeat consistency."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from agent.service import VoiceAgent
from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav, sandbox_cohort
from voice_eval.dims.task import summarize_verify_group
from voice_eval.grader import aggregate

STANDARD_AXES = ('noise', 'duration', 'channel', 'reverb', 'mixture')


def _enroll_takes(base_dir: Path, name: str, speaker_index: int) -> list[Path]:
    return [
        generate_speaker_wav(base_dir / 's1' / f'{name}_0.wav', speaker_index, duration=4.0, phase=0.0, seed=0),
        generate_speaker_wav(base_dir / 's1' / f'{name}_1.wav', speaker_index, duration=4.0, phase=0.8, seed=1),
        generate_speaker_wav(base_dir / 's2' / f'{name}_2.wav', speaker_index, duration=4.0, phase=1.6, seed=2),
    ]


def evaluate_fault_recovery() -> dict:
    """Inject 4 fault classes into a live VoiceAgent and verify 100% clean post-fault recovery."""
    embedder = ToneEmbedder()
    checks: list[dict] = []

    with sandbox_cohort(embedder), TemporaryDirectory(prefix='vprob_fault_') as tmp:
        root = Path(tmp)
        audio_dir = root / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)

        takes = _enroll_takes(audio_dir, 'alice', speaker_index=0)
        genuine_probe = generate_speaker_wav(audio_dir / 'genuine.wav', speaker_index=0, duration=4.0, phase=2.4, seed=3)

        fake_txt = audio_dir / 'not_audio.txt'
        fake_txt.write_text('not an audio file', encoding='utf-8')
        corrupt_wav = audio_dir / 'corrupt.wav'
        corrupt_wav.write_bytes(b'RIFF\x00\x00\x00\x00WAVEcorrupt-payload')

        faults = [
            ('missing_path', f'验证 alice "{audio_dir / "missing.wav"}"'),
            ('non_audio_file', f'验证 alice "{fake_txt}"'),
            ('corrupt_wav', f'验证 alice "{corrupt_wav}"'),
            ('typo_command', '错别字命令 alice'),
        ]

        agent = VoiceAgent(root / 'db', embedder, detail=True, audio_roots=[audio_dir])
        try:
            enrolled = agent.handle('注册 alice 本人同意 ' + ' '.join(f'"{p}"' for p in takes))
            if not enrolled['ok']:
                raise RuntimeError(f'fault recovery setup failed: {enrolled}')
            seed_orig, seg_orig = agent.store.load_speaker('alice', *agent.model)

            for fault_id, fault_cmd in faults:
                fault_out = agent.handle(fault_cmd)
                fault_handled = (not fault_out.get('ok')) and (fault_out.get('tool') != 'error')

                rec_out = agent.handle(f'验证 alice "{genuine_probe}"')
                recovered = bool(rec_out.get('ok')) and (rec_out.get('result') or {}).get('decision') == 'accept'

                null_rows = agent.store.db.execute("select count(*) from events where score is null").fetchone()[0]
                if 'alice' in agent.store.names():
                    seed_now, seg_now = agent.store.load_speaker('alice', *agent.model)
                    store_intact = (
                        seed_now == seed_orig
                        and np.array_equal(seg_now, seg_orig)
                        and null_rows == 0
                    )
                else:
                    store_intact = False

                passed = fault_handled and recovered and store_intact
                checks.append({
                    'id': fault_id,
                    'fault_handled': fault_handled,
                    'recovered': recovered,
                    'store_intact': store_intact,
                    'passed': passed,
                })
        finally:
            agent.close()

    fault_total = len(checks)
    recovered_count = sum(1 for c in checks if c['passed'])
    recovery_rate = recovered_count / fault_total if fault_total else 0.0
    return {
        'passed': recovered_count == fault_total,
        'fault_total': fault_total,
        'recovered': recovered_count,
        'recovery_rate': recovery_rate,
        'checks': checks,
    }


def evaluate_consistency(repeats: int = 3) -> dict:
    """Re-run 5 representative verify trials K times from fresh enrollment and measure stability."""
    if repeats < 1:
        raise ValueError(f'repeats must be >= 1, got {repeats}')

    embedder = ToneEmbedder()
    checks: list[dict] = []

    with sandbox_cohort(embedder), TemporaryDirectory(prefix='vprob_cons_') as tmp:
        root = Path(tmp)
        audio_dir = root / 'audio'
        takes = {idx: _enroll_takes(audio_dir, f'spk_{idx}', idx) for idx in (0, 1, 2)}
        probes = {
            idx: generate_speaker_wav(audio_dir / f'probe_{idx}.wav', speaker_index=idx, duration=4.0, phase=2.4, seed=3)
            for idx in (0, 1, 2)
        }

        trials = [
            ('trial_0_genuine_spk0', takes[0], probes[0], 'accept'),
            ('trial_1_impostor_spk1_on_spk0', takes[0], probes[1], 'reject'),
            ('trial_2_genuine_spk1', takes[1], probes[1], 'accept'),
            ('trial_3_impostor_spk2_on_spk1', takes[1], probes[2], 'reject'),
            ('trial_4_genuine_spk2', takes[2], probes[2], 'accept'),
        ]

        for trial_id, enroll_paths, probe_path, expect_decision in trials:
            decisions: list[str] = []
            scores: list[float] = []
            enroll_cmd = '注册 target 本人同意 ' + ' '.join(f'"{p}"' for p in enroll_paths)
            verify_cmd = f'验证 target "{probe_path}"'

            for rep in range(repeats):
                agent = VoiceAgent(root / f'db_{trial_id}_{rep}', embedder, detail=True, audio_roots=[audio_dir])
                try:
                    enr = agent.handle(enroll_cmd)
                    if not enr['ok']:
                        raise RuntimeError(f'consistency setup failed: {enr}')
                    out = agent.handle(verify_cmd)
                    result = out.get('result') or {}
                    decisions.append(str(result.get('decision') or 'missing'))
                    if result.get('score') is not None:
                        scores.append(float(result['score']))
                finally:
                    agent.close()

            agreed = len(set(decisions)) == 1 and decisions[0] == expect_decision
            spread = float(max(scores) - min(scores)) if len(scores) >= 2 else 0.0
            passed = agreed and len(scores) == repeats and spread <= 1e-3
            checks.append({
                'id': trial_id,
                'expect_decision': expect_decision,
                'decisions': decisions,
                'agreed': agreed,
                'score_spread': spread,
                'passed': passed,
            })

    total_trials = len(checks)
    agreed_count = sum(1 for c in checks if c['agreed'])
    agreement_rate = agreed_count / total_trials if total_trials else 0.0
    max_score_spread = max((c['score_spread'] for c in checks), default=0.0)
    return {
        'passed': all(c['passed'] for c in checks),
        'trials': total_trials,
        'repeats': int(repeats),
        'agreement_rate': agreement_rate,
        'max_score_spread': max_score_spread,
        'checks': checks,
    }


def _extract_axis_variant(row: dict) -> tuple[str | None, str | None]:
    hidden = row.get('hidden') or {}
    axis = row.get('axis') or hidden.get('axis')
    if not axis:
        suite = str(row.get('suite') or '')
        if suite.startswith('robustness_'):
            axis = suite[len('robustness_'):]
    variant = row.get('variant') or hidden.get('variant')
    return (str(axis) if axis else None), (str(variant) if variant else None)


def _summarize_robustness_group(
    group_rows: list[dict],
    baseline_verify_accuracy: float | None,
    *,
    axis: str | None = None,
) -> dict:
    agg = aggregate(group_rows)
    verify_stats = summarize_verify_group(group_rows)
    eer = verify_stats['eer'] if verify_stats['valid_trials'] > 0 else None
    acc = agg['accuracy']
    acc_drop = (
        float(baseline_verify_accuracy) - float(acc)
        if baseline_verify_accuracy is not None and acc is not None
        else None
    )
    out = dict(agg)
    if axis is not None:
        out['axis'] = axis
    out['eer'] = eer
    out['acc_drop'] = acc_drop
    return out


def evaluate_robustness(
    rows: list[dict],
    baseline_verify_accuracy: float | None,
    *,
    fault_recovery: dict | None = None,
    consistency: dict | None = None,
) -> dict:
    """Group robustness rows by axis and variant, compute drops/EER, and run fault/consistency checks."""
    axis_groups: dict[str, list[dict]] = {}
    variant_groups: dict[str, tuple[str, list[dict]]] = {}

    for row in rows:
        axis, variant = _extract_axis_variant(row)
        if not axis:
            continue
        axis_groups.setdefault(axis, []).append(row)
        if variant:
            if variant not in variant_groups:
                variant_groups[variant] = (axis, [])
            variant_groups[variant][1].append(row)

    ordered_axes = [ax for ax in STANDARD_AXES if ax in axis_groups]
    for ax in axis_groups:
        if ax not in ordered_axes:
            ordered_axes.append(ax)

    by_axis = {
        ax: _summarize_robustness_group(axis_groups[ax], baseline_verify_accuracy)
        for ax in ordered_axes
    }
    by_variant = {
        variant: _summarize_robustness_group(v_rows, baseline_verify_accuracy, axis=v_axis)
        for variant, (v_axis, v_rows) in variant_groups.items()
    }
    drops = [item['acc_drop'] for item in by_variant.values() if item['acc_drop'] is not None]
    worst_drop = max(drops) if drops else None

    return {
        'by_axis': by_axis,
        'by_variant': by_variant,
        'worst_drop': worst_drop,
        'fault_recovery': fault_recovery if fault_recovery is not None else evaluate_fault_recovery(),
        'consistency': consistency if consistency is not None else evaluate_consistency(),
    }
