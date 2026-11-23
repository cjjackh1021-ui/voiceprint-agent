"""Dual-track closed-loop evaluation for controllable voice simulation, Self-Refine and watermark resilience."""
from contextlib import ExitStack, contextmanager
import ctypes
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import soundfile as sf

import agent.service as service_mod
from agent.service import VoiceAgent
from voice_eval.audio_synth import (
    add_white_noise,
    apply_gain_clip_attack,
    apply_lowpass_attack,
    apply_resample_attack,
    clip_duration,
)
from voice_eval.cases import TEST_ROOT, _seed_for
from voice_eval.dims._sandbox import (
    ToneEmbedder,
    generate_speaker_wav,
    quote_paths,
    sandbox_cohort,
    sandbox_enroll,
)
from voice_eval.dims.perf import current_rss_mb, percentile_stats
from voiceprint.audio import MIN_SECONDS, list_flac
from voiceprint.audio_quality import analyze_audio_quality
from voiceprint.features import SR
from voiceprint.spoof import DEFAULT_WATERMARK_PAYLOAD, SpreadSpectrumWatermarkDetector
from voiceprint.synth import VoiceSynthesizer

_NON_WIN_KEY_PREFIX = b'SBXDPAPI1:'


@contextmanager
def _ephemeral_sandbox_dpapi_if_needed():
    """Allow standalone non-Windows sim-bench CLI runs inside TemporaryDirectory without weakening security.py."""
    import voiceprint.security as sec_mod
    import voiceprint.store as store_mod

    if sys.platform == 'win32' or hasattr(ctypes, 'windll'):
        yield
        return
    if store_mod.dpapi_protect is not sec_mod.dpapi_protect:
        # Already patched by test harness (e.g. patched_dpapi); honor caller patch.
        yield
        return

    def _sbx_protect(data: bytes) -> bytes:
        return _NON_WIN_KEY_PREFIX + bytes(data)

    def _sbx_unprotect(blob: bytes) -> bytes:
        if not blob.startswith(_NON_WIN_KEY_PREFIX):
            raise ValueError('invalid sandbox DPAPI blob')
        return blob[len(_NON_WIN_KEY_PREFIX):]

    orig_sec_p, orig_sec_u = sec_mod.dpapi_protect, sec_mod.dpapi_unprotect
    orig_st_p, orig_st_u = store_mod.dpapi_protect, store_mod.dpapi_unprotect
    sec_mod.dpapi_protect = _sbx_protect
    sec_mod.dpapi_unprotect = _sbx_unprotect
    store_mod.dpapi_protect = _sbx_protect
    store_mod.dpapi_unprotect = _sbx_unprotect
    try:
        yield
    finally:
        sec_mod.dpapi_protect, sec_mod.dpapi_unprotect = orig_sec_p, orig_sec_u
        store_mod.dpapi_protect, store_mod.dpapi_unprotect = orig_st_p, orig_st_u


def _read_librispeech_transcript(audio_path: Path) -> str | None:
    stem = audio_path.stem
    for trans in audio_path.parent.glob('*.trans.txt'):
        for line in trans.read_text(encoding='utf-8', errors='ignore').splitlines():
            if line.startswith(stem + ' '):
                return line[len(stem) + 1:].strip()
    return None


def _run_attack_suite(wav: np.ndarray, sr: int, detector: SpreadSpectrumWatermarkDetector, seed_base: int) -> dict[str, dict]:
    attacks = {
        'white_noise_15db': add_white_noise(wav, snr_db=15.0, seed=seed_base + 11),
        'white_noise_10db': add_white_noise(wav, snr_db=10.0, seed=seed_base + 23),
        'duration_crop_0.6s': clip_duration(wav, sr, 0.6),
        'lowpass_4500hz': apply_lowpass_attack(wav, sr, cutoff_hz=4500.0),
        'lowpass_3600hz': apply_lowpass_attack(wav, sr, cutoff_hz=3600.0),
        'resample_8k_roundtrip': apply_resample_attack(wav, sr, mid_sr=8000),
        'gain_clip_1.8x': apply_gain_clip_attack(wav, sr, drive=1.8),
    }
    out: dict[str, dict] = {}
    for name, attacked in attacks.items():
        res = detector.detect(attacked, sr)
        out[name] = {'detected': bool(res.is_spoof), 'z_score': float(res.score)}
    return out


def evaluate_simulation(
    embedder,
    synthesizer: VoiceSynthesizer,
    *,
    max_attempts: int = 3,
    speakers_limit: int = 4,
) -> dict:
    """Run Track A (Self-Refine convergence & gates) and Track B (watermark & attack survival)."""
    if synthesizer is None:
        raise ValueError('evaluate_simulation requires an active VoiceSynthesizer')

    speakers_limit = max(2, int(speakers_limit))
    wm_detector = SpreadSpectrumWatermarkDetector()

    with ExitStack() as stack, TemporaryDirectory(prefix='vpsim_bench_') as tmp:
        stack.enter_context(_ephemeral_sandbox_dpapi_if_needed())
        if embedder.name == 'tone' and service_mod.cohort_matrix.__module__ == 'voiceprint.embed':
            stack.enter_context(sandbox_cohort(embedder))

        root = Path(tmp)
        audio_dir = root / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        loop_dir = root / 'loop_check'
        loop_dir.mkdir(parents=True, exist_ok=True)

        agent = VoiceAgent(
            root / 'db',
            embedder,
            detail=True,
            audio_roots=[audio_dir, loop_dir],
            spoof_detector=wm_detector,
            synthesizer=synthesizer,
            max_sim_attempts=max_attempts,
        )
        try:
            speaker_specs: list[dict] = []
            if embedder.name == 'tone':
                for idx in range(speakers_limit):
                    name = f'spk_{idx:02d}'
                    takes = sandbox_enroll(agent, name, audio_dir / name, speaker_index=idx, duration=4.0)
                    other_src = generate_speaker_wav(
                        audio_dir / f'src_for_{name}.wav',
                        speaker_index=(idx + 1) % max(2, speakers_limit),
                        duration=2.5,
                        phase=1.3,
                        seed=50 + idx,
                    )
                    speaker_specs.append({
                        'name': name,
                        'ref_path': str(takes[0]),
                        'ref_text': None,
                        'src_path': str(other_src),
                        'text': f'第 {idx + 1} 号说话人的授权声音模拟与溯源闭环验证文本。',
                    })
            else:
                if not TEST_ROOT.is_dir():
                    raise FileNotFoundError(f'缺少 LibriSpeech test-clean 数据集目录：{TEST_ROOT}')
                spk_map = list_flac(str(TEST_ROOT))
                eligible = [k for k in sorted(spk_map) if len(spk_map[k]) >= 3]
                if len(eligible) < 2:
                    raise FileNotFoundError(f'{TEST_ROOT} 中可用多段录音说话人不足 2 位')
                chosen = eligible[:speakers_limit]
                for idx, spk_id in enumerate(chosen):
                    takes = spk_map[spk_id][:3]
                    enr = agent.handle(f'注册 {spk_id} 本人同意 {quote_paths(takes)}')
                    if not enr['ok']:
                        # Fallback to full utterance list when 3 takes are short in same folder
                        enr = agent.handle(f'注册 {spk_id} 本人同意 {quote_paths(spk_map[spk_id][:6])}')
                    if not enr['ok']:
                        continue
                    other_id = chosen[(idx + 1) % len(chosen)]
                    speaker_specs.append({
                        'name': spk_id,
                        'ref_path': str(takes[0]),
                        'ref_text': _read_librispeech_transcript(Path(takes[0])),
                        'src_path': str(spk_map[other_id][0]),
                        'text': 'Controlled voice simulation verification with spread-spectrum watermark traceability.',
                    })
                if len(speaker_specs) < 2:
                    raise RuntimeError('未能成功注册足够数量的 LibriSpeech 测试说话人。')

            # Security gate sanity: missing consent + cross-speaker reference refusal rate
            gate_checks: list[bool] = []
            s0, s1 = speaker_specs[0], speaker_specs[1]
            g_no_consent, _ = agent._run_simulate_core(
                s0['name'], consent=False, ref_path=s0['ref_path'], mode='text', text='no consent check'
            )
            gate_checks.append(not g_no_consent['ok'] and (g_no_consent.get('result') or {}).get('reason') == 'consent_required')
            if 'audio' in synthesizer.supported_modes:
                g_mismatch, _ = agent._run_simulate_core(
                    s0['name'], consent=True, ref_path=s1['ref_path'], mode='audio', src_path=s0['src_path']
                )
            else:
                g_mismatch, _ = agent._run_simulate_core(
                    s0['name'], consent=True, ref_path=s1['ref_path'], mode='text', text='mismatch'
                )
            gate_checks.append(not g_mismatch['ok'] and (g_mismatch.get('result') or {}).get('reason') == 'reference_speaker_mismatch')
            security_gate_refusal_rate = sum(1 for x in gate_checks if x) / len(gate_checks)

            cases_out: list[dict] = []
            latencies_ms: list[float] = []
            attempt1_raws: list[float] = []
            final_raws: list[float] = []
            attempt1_norms: list[float] = []
            final_norms: list[float] = []
            attempts_used: list[int] = []
            attempt1_converged_flags: list[bool] = []
            converged_flags: list[bool] = []
            skipped_unsupported = 0

            track_b_quality: list[bool] = []
            track_b_wm: list[bool] = []
            track_b_loopback_blocked: list[bool] = []
            attack_buckets: dict[str, list[dict]] = {}

            for spk_idx, spec in enumerate(speaker_specs):
                for mode in ('audio', 'text'):
                    case_id = f"{spec['name']}_{mode}"
                    if mode not in synthesizer.supported_modes:
                        skipped_unsupported += 1
                        cases_out.append({
                            'id': case_id,
                            'speaker': spec['name'],
                            'mode': mode,
                            'skipped': True,
                            'reason': 'unsupported_mode',
                        })
                        continue

                    t0 = time.perf_counter()
                    outcome, best_wav = agent._run_simulate_core(
                        spec['name'],
                        consent=True,
                        ref_path=spec['ref_path'],
                        mode=mode,
                        src_path=spec['src_path'] if mode == 'audio' else None,
                        text=spec['text'] if mode == 'text' else None,
                        reference_text=spec['ref_text'],
                        max_attempts=max_attempts,
                        collect_best_candidate=True,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.append(float(elapsed_ms))

                    res = outcome.get('result') or {}
                    hist = [h for h in (res.get('attempt_history') or []) if 'raw_score' in h]
                    converged = bool(outcome['ok'] and res.get('converged'))
                    converged_flags.append(converged)
                    a1_conv = bool(
                        hist
                        and hist[0].get('quality_ok')
                        and hist[0].get('watermarked')
                        and hist[0].get('decision') == 'accept'
                    )
                    attempt1_converged_flags.append(a1_conv)
                    attempts_used.append(int(res.get('attempts') or len(hist) or 1))

                    if hist:
                        a1 = hist[0]
                        valid_hist = [h for h in hist if h.get('watermarked')]
                        best_h = max(
                            valid_hist or hist,
                            key=lambda h: (1 if h.get('quality_ok') else 0, h['score'] - h['threshold'], h['raw_score']),
                        )
                        attempt1_raws.append(float(a1['raw_score']))
                        final_raws.append(float(best_h['raw_score']))
                        attempt1_norms.append(float(a1['score']))
                        final_norms.append(float(best_h['score']))

                    cand_wav = best_wav
                    if outcome['ok'] and res.get('output_path'):
                        cand_wav, _ = sf.read(res['output_path'], dtype='float32')

                    wm_ok = False
                    q_ok = False
                    loop_blocked = False
                    atk_res: dict[str, dict] = {}
                    if cand_wav is not None:
                        cand_path = loop_dir / f'trackb_{spk_idx}_{mode}.wav'
                        sf.write(str(cand_path), cand_wav, SR)
                        q_rep = analyze_audio_quality(cand_path)
                        bad_q = [r for r in q_rep.reject_reasons if r in ('empty', 'silence', 'invalid', 'clipping')]
                        q_ok = (not bad_q) and (q_rep.clipping_ratio <= 0.05) and (q_rep.active_seconds + 1e-8 >= MIN_SECONDS)

                        wm_det = wm_detector.detect(cand_wav, SR)
                        wm_ok = bool(wm_det.is_spoof and wm_det.metadata.get('payload') == DEFAULT_WATERMARK_PAYLOAD)

                        ver = agent.handle(f'验证 {quote_paths([spec["name"]])[1:-1]} "{cand_path}"')
                        loop_blocked = (not ver['ok']) and ((ver.get('result') or {}).get('reason') == 'spoof_detected')

                        seed_base = _seed_for('simbench_atk', spk_idx, mode) % (2**31)
                        atk_res = _run_attack_suite(cand_wav, SR, wm_detector, seed_base)
                        for atk_name, atk_stat in atk_res.items():
                            attack_buckets.setdefault(atk_name, []).append(atk_stat)

                        track_b_quality.append(q_ok)
                        track_b_wm.append(wm_ok)
                        track_b_loopback_blocked.append(loop_blocked)

                    cases_out.append({
                        'id': case_id,
                        'speaker': spec['name'],
                        'mode': mode,
                        'skipped': False,
                        'converged': converged,
                        'attempts': int(res.get('attempts') or len(hist) or 0),
                        'attempt1_raw_sim': float(hist[0]['raw_score']) if hist else None,
                        'final_raw_sim': float(final_raws[-1]) if hist else None,
                        'attempt1_norm_score': float(hist[0]['score']) if hist else None,
                        'final_norm_score': float(final_norms[-1]) if hist else None,
                        'quality_ok': q_ok,
                        'watermark_ok': wm_ok,
                        'loopback_blocked': loop_blocked,
                        'latency_ms': float(elapsed_ms),
                        'attacks': atk_res,
                    })

            total_exec = len(converged_flags)
            a1_raw_mean = float(np.mean(attempt1_raws)) if attempt1_raws else 0.0
            fn_raw_mean = float(np.mean(final_raws)) if final_raws else 0.0
            a1_norm_mean = float(np.mean(attempt1_norms)) if attempt1_norms else 0.0
            fn_norm_mean = float(np.mean(final_norms)) if final_norms else 0.0

            attack_survival: dict[str, dict] = {}
            for atk_name, rows in attack_buckets.items():
                attack_survival[atk_name] = {
                    'n': len(rows),
                    'detection_rate': float(sum(1 for r in rows if r['detected']) / len(rows)),
                    'mean_z_score': float(np.mean([r['z_score'] for r in rows])),
                }

            a1_conv_rate = float(sum(1 for x in attempt1_converged_flags if x) / total_exec) if total_exec else 0.0
            final_conv_rate = float(sum(1 for x in converged_flags if x) / total_exec) if total_exec else 0.0

            return {
                'embedder': embedder.name,
                'model': embedder.describe(),
                'synthesizer': synthesizer.describe(),
                'max_attempts': int(max_attempts),
                'track_a': {
                    'total_cases': total_exec,
                    'skipped_unsupported_cases': skipped_unsupported,
                    'attempt1_mean_raw_sim': a1_raw_mean,
                    'final_mean_raw_sim': fn_raw_mean,
                    'delta_raw_sim': fn_raw_mean - a1_raw_mean,
                    'attempt1_mean_norm_score': a1_norm_mean,
                    'final_mean_norm_score': fn_norm_mean,
                    'delta_norm_score': fn_norm_mean - a1_norm_mean,
                    'attempt1_convergence_rate': a1_conv_rate,
                    'convergence_rate': final_conv_rate,
                    'delta_convergence_rate': final_conv_rate - a1_conv_rate,
                    'avg_attempts': float(np.mean(attempts_used)) if attempts_used else 0.0,
                    'security_gate_refusal_rate': float(security_gate_refusal_rate),
                },
                'track_b': {
                    'evaluated_samples': len(track_b_wm),
                    'quality_pass_rate': float(sum(1 for x in track_b_quality if x) / len(track_b_quality)) if track_b_quality else 0.0,
                    'watermark_readback_rate': float(sum(1 for x in track_b_wm if x) / len(track_b_wm)) if track_b_wm else 0.0,
                    'loopback_spoof_block_rate': float(sum(1 for x in track_b_loopback_blocked if x) / len(track_b_loopback_blocked)) if track_b_loopback_blocked else 0.0,
                    'attack_survival': attack_survival,
                },
                'latency_ms': percentile_stats(latencies_ms),
                'rss_mb': current_rss_mb(),
                'cases': cases_out,
            }
        finally:
            agent.close()


def write_simulation_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    """Persist traceable JSON report and human-readable Markdown summary."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    emb_name = str(report.get('embedder', 'unknown'))
    syn_name = str((report.get('synthesizer') or {}).get('backend', 'unknown'))
    stem = f'sim_bench_{emb_name}_{syn_name}'
    json_path = dest / f'{stem}.json'
    md_path = dest / f'{stem}.md'

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    ta = report['track_a']
    tb = report['track_b']
    lat = report['latency_ms']
    lines = [
        f'# Simulation Closed-Loop Benchmark ({emb_name} / {syn_name})',
        '',
        '## Track A — Self-Refine & Security Gate',
        f"- Cases executed: `{ta['total_cases']}` (skipped unsupported: `{ta['skipped_unsupported_cases']}`)",
        f"- Attempt-1 -> Final Cosine SIM: `{ta['attempt1_mean_raw_sim']:.4f}` -> `{ta['final_mean_raw_sim']:.4f}` (delta `{ta['delta_raw_sim']:+.4f}`)",
        f"- Attempt-1 -> Final Norm Score: `{ta['attempt1_mean_norm_score']:.3f}` -> `{ta['final_mean_norm_score']:.3f}` (delta `{ta['delta_norm_score']:+.3f}`)",
        f"- 1-Shot -> Self-Refine Convergence Rate: `{ta['attempt1_convergence_rate']:.1%}` -> `{ta['convergence_rate']:.1%}` (delta `{ta['delta_convergence_rate']:+.1%}`, avg attempts `{ta['avg_attempts']:.2f}`)",
        f"- Unauthorized / Cross-Speaker Refusal Rate: `{ta['security_gate_refusal_rate']:.1%}`",
        '',
        '## Track B — Provenance Watermark & Attack Resilience',
        f"- Samples evaluated: `{tb['evaluated_samples']}`",
        f"- 100 ms Quality Pass Rate: `{tb['quality_pass_rate']:.1%}`",
        f"- 16-bit Watermark Readback Rate: `{tb['watermark_readback_rate']:.1%}`",
        f"- Closed-Loop Spoof Block Rate: `{tb['loopback_spoof_block_rate']:.1%}`",
        f"- End-to-End Latency: P50 `{lat['p50_ms']:.2f} ms`, P95 `{lat['p95_ms']:.2f} ms`" if lat.get('p50_ms') is not None else '- End-to-End Latency: N/A',
        '',
        '| Removal Attack | Detection Rate | Mean z-score |',
        '| :--- | :---: | :---: |',
    ]
    for atk_name, stats in sorted(tb['attack_survival'].items()):
        lines.append(f"| `{atk_name}` | `{stats['detection_rate']:.1%}` | `{stats['mean_z_score']:.2f}` |")
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return json_path, md_path
