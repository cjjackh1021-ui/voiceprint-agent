"""End-to-end tests for Task 6: 7-dimension runner, 3-tier gate, CLI, and tune on v3 cases."""
import hashlib
import io
import json
import shlex
import shutil
import tempfile
import tracemalloc
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import agent.service as service_mod
import voiceprint.embed as embed_mod
from tests._helpers import patched_dpapi, synthetic_librispeech_tree
from voice_eval import __main__ as cli_mod
from voice_eval import audit as audit_mod
from voice_eval import tune as tune_mod
from voice_eval.cases import build, load
from voice_eval.dims._sandbox import ToneEmbedder
from voice_eval.dims.robustness import evaluate_robustness
from voice_eval.dims.task import evaluate_task
from voice_eval.gate import compare_reports, resolve_report_dir
from voice_eval.grader import aggregate
from voice_eval.runner import _decision_for_grade, run_case, run_split
from voiceprint.audio import file_digest, load_wav
from voiceprint.embed import Embedder, _normalize


class WeakEmbedder(Embedder):
    """Low-accuracy deterministic embedder that still enrolls cleanly for baseline comparison."""

    name = 'weak'
    dim = 64
    cohort_size = 16
    model_id = 'weak-v1-' + '0' * 56

    def _embed(self, path: str) -> np.ndarray:
        wav = load_wav(path, max_seconds=12.0)
        digest = hashlib.sha256(file_digest(path).encode('utf-8')).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        base = np.zeros(self.dim, dtype=np.float32)
        base[0] = 4.0
        base[1] = float(np.sqrt(np.mean(np.square(wav))))
        jitter = rng.normal(0.0, 0.45, size=self.dim).astype(np.float32)
        return _normalize(base + jitter)


def _copy_report(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _mutate_json(path: Path, mutator) -> None:
    data = json.loads(path.read_text(encoding='utf-8'))
    mutator(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


class EndToEndEvalTests(unittest.TestCase):
    def test_full_v3_pipeline_and_three_tier_gate_outcomes(self):
        with patched_dpapi(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_root, test_root = synthetic_librispeech_tree(root / 'LibriSpeech', n_train=16, n_test=40)
            case_dir = root / 'cases'
            reports_dir = root / 'reports'
            cache_dir = root / 'cache'
            cohort_dir = root / 'cohorts'

            counts = build(test_root=test_root, case_dir=case_dir)
            self.assertEqual(counts['validation'], 1190)
            self.assertEqual(counts['locked'], 1190)

            with ExitStack() as stack:
                stack.enter_context(patch.object(service_mod, 'TRAIN_ROOT', train_root))
                stack.enter_context(patch.object(embed_mod, 'COHORT_DIR', cohort_dir))

                audit_res = audit_mod.run_all()
                self.assertTrue(audit_res['ok'], audit_res)

                weak = WeakEmbedder()
                strong = ToneEmbedder()

                weak_val = run_split(
                    weak, 'validation', True,
                    case_dir=case_dir, reports_dir=reports_dir, cache_dir=cache_dir, skip_mixture=True,
                )
                strong_val = run_split(
                    strong, 'validation', True,
                    case_dir=case_dir, reports_dir=reports_dir, cache_dir=cache_dir, skip_mixture=True,
                )
                weak_locked = run_split(
                    weak, 'locked', False,
                    case_dir=case_dir, reports_dir=reports_dir, cache_dir=cache_dir, skip_mixture=True,
                )
                strong_locked = run_split(
                    strong, 'locked', False,
                    case_dir=case_dir, reports_dir=reports_dir, cache_dir=cache_dir, skip_mixture=True,
                )

                # Second warm run reuses disk cache with zero cold misses.
                warm_dir = reports_dir / 'warm_check'
                strong_warm = run_split(
                    ToneEmbedder(), 'validation', False,
                    case_dir=case_dir, reports_dir=warm_dir, cache_dir=cache_dir, skip_mixture=True,
                )

            # Verify cases.jsonl recomputation, all 7 dims, summary.md, and cache reuse.
            strong_summary = json.loads((strong_val / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(strong_summary['case_version'], 3)
            self.assertTrue(strong_summary['recomputed_from_jsonl'])
            self.assertEqual(strong_summary['skipped_suites'], ['robustness_mixture'])
            dims = strong_summary['dimensions']
            self.assertEqual(
                set(dims.keys()),
                {'task', 'tooling', 'response', 'flow', 'perf', 'redteam', 'robustness'},
            )
            reloaded = [
                json.loads(line)
                for line in (strong_val / 'cases.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            self.assertEqual(evaluate_task(reloaded), dims['task'])
            reloaded_rob = evaluate_robustness(
                reloaded,
                dims['task']['by_suite']['verify']['accuracy'],
                fault_recovery=dims['robustness']['fault_recovery'],
                consistency=dims['robustness']['consistency'],
            )
            self.assertEqual(reloaded_rob['by_variant']['white_snr20'], dims['robustness']['by_variant']['white_snr20'])
            self.assertTrue(dims['robustness']['by_axis']['mixture']['skipped'])
            self.assertGreater(dims['perf']['cache']['hits'], 0)
            self.assertGreater(dims['perf']['cache']['misses'], 0)
            self.assertGreater(dims['perf']['cache']['hit_rate'], 0.5)

            warm_summary = json.loads((strong_warm / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(warm_summary['dimensions']['perf']['cache']['misses'], 0)
            self.assertGreater(warm_summary['dimensions']['perf']['cache']['hits'], 0)

            summary_md = (strong_val / 'summary.md').read_text(encoding='utf-8')
            self.assertIn('七维', summary_md)
            self.assertIn('robustness_mixture', summary_md)

            # Tier 1 + Tier 2 valid win on validation -> adopt=True.
            win_cmp = compare_reports(weak_val, strong_val, out_dir=reports_dir)
            self.assertTrue(win_cmp['adopt'], win_cmp)
            self.assertTrue(win_cmp['adoption_allowed'])
            self.assertTrue(win_cmp['redline_passed'])
            self.assertTrue(win_cmp['primary_passed'])
            self.assertGreater(win_cmp['paired']['low'], 0.0)
            self.assertLessEqual(win_cmp['eer_delta'], 1e-6)
            self.assertTrue(Path(win_cmp['markdown_path']).is_file())
            self.assertTrue(Path(win_cmp['json_path']).is_file())
            cmp_md = Path(win_cmp['markdown_path']).read_text(encoding='utf-8')
            self.assertIn('采用', cmp_md)
            self.assertIn('Tier 1', cmp_md)

            # Locked split: never adopt despite high accuracy and clean redlines.
            locked_cmp = compare_reports(weak_locked, strong_locked, out_dir=reports_dir)
            self.assertFalse(locked_cmp['adoption_allowed'])
            self.assertFalse(locked_cmp['primary_passed'])
            self.assertFalse(locked_cmp['adopt'])
            self.assertIn('锁定', locked_cmp['reason'])

            # Tier 1 redline block: inject redteam.red = 1.
            cand_rt_fail = _copy_report(strong_val, root / 'mut_redteam_validation')
            _mutate_json(cand_rt_fail / 'summary.json', lambda d: d['dimensions']['redteam'].update({'red': 1, 'passed': False}))
            res_rt = compare_reports(weak_val, cand_rt_fail, out_dir=root / 'out_rt')
            self.assertFalse(res_rt['redline_passed'])
            self.assertFalse(res_rt['adopt'])

            # Tier 1 redline block: inject tooling/response/flow/security failures individually.
            for dim_key, mut in (
                ('tooling', lambda d: d['dimensions']['tooling'].update({'passed': False, 'accuracy': 0.9})),
                ('response', lambda d: d['dimensions']['response'].update({'passed': False, 'leak_count': 1})),
                ('flow', lambda d: d['dimensions']['flow'].update({'passed': False})),
                ('security', lambda d: d['by_suite']['security'].update({'accuracy': 0.5})),
            ):
                cand_mut = _copy_report(strong_val, root / f'mut_{dim_key}_validation')
                _mutate_json(cand_mut / 'summary.json', mut)
                res_mut = compare_reports(weak_val, cand_mut, out_dir=root / f'out_{dim_key}')
                self.assertFalse(res_mut['redline_passed'], dim_key)
                self.assertFalse(res_mut['adopt'], dim_key)

            # Tier 2 EER regression block (> +1e-6).
            cand_eer_bad = _copy_report(strong_val, root / 'mut_eer_validation')
            weak_summary = json.loads((weak_val / 'summary.json').read_text(encoding='utf-8'))
            base_eer = weak_summary['dimensions']['task']['verify_all_metrics']['eer']
            _mutate_json(
                cand_eer_bad / 'summary.json',
                lambda d: d['dimensions']['task']['verify_all_metrics'].update({'eer': base_eer + 0.01}),
            )
            res_eer = compare_reports(weak_val, cand_eer_bad, out_dir=root / 'out_eer')
            self.assertTrue(res_eer['redline_passed'])
            self.assertFalse(res_eer['primary_passed'])
            self.assertFalse(res_eer['adopt'])

            # Tier 2 FA delta > 0.02 block.
            cand_fa_bad = _copy_report(strong_val, root / 'mut_fa_validation')
            cand_rows = [json.loads(line) for line in (cand_fa_bad / 'cases.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            flipped = 0
            for r in cand_rows:
                if r['suite'] == 'verify' and r['expect'] == 'reject':
                    r['false_accept'] = True
                    r['correct'] = False
                    r['decision'] = 'accept'
                    flipped += 1
                    if flipped >= 15:
                        break
            (cand_fa_bad / 'cases.jsonl').write_text(
                ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in cand_rows), encoding='utf-8'
            )
            res_fa = compare_reports(weak_val, cand_fa_bad, out_dir=root / 'out_fa')
            self.assertFalse(res_fa['primary_passed'])
            self.assertFalse(res_fa['adopt'])

            # Tier 3 observational warnings ONLY -> adopt=True with non-empty warnings.
            cand_warn = _copy_report(strong_val, root / 'mut_warn_validation')

            def _inject_tier3(d):
                base_p95 = d['dimensions']['perf']['case']['p95_ms'] or 10.0
                d['dimensions']['perf']['case']['p95_ms'] = max(base_p95 * 2.0, base_p95 + 25.0)
                d['dimensions']['perf']['peak_tracemalloc_mb'] = max(
                    (weak_summary['dimensions']['perf']['peak_tracemalloc_mb'] or 1.0) * 3.0, 50.0
                )
                d['dimensions']['robustness']['by_variant']['white_snr20']['accuracy'] = 0.0
                d['dimensions']['robustness']['fault_recovery']['recovery_rate'] = 0.75
                d['dimensions']['robustness']['consistency']['agreement_rate'] = 0.8

            _mutate_json(cand_warn / 'summary.json', _inject_tier3)
            res_warn = compare_reports(weak_val, cand_warn, out_dir=root / 'out_warn')
            self.assertTrue(res_warn['redline_passed'])
            self.assertTrue(res_warn['primary_passed'])
            self.assertTrue(res_warn['adopt'])
            self.assertGreaterEqual(len(res_warn['warnings']), 4)

            # Cross-version (case_version=2 vs 3) and mismatched-split must raise ValueError.
            cand_v2 = _copy_report(strong_val, root / 'mut_v2_validation')
            _mutate_json(cand_v2 / 'summary.json', lambda d: d.update({'case_version': 2}))
            with self.assertRaises(ValueError):
                compare_reports(weak_val, cand_v2, out_dir=root / 'out_v2')

            with self.assertRaises(ValueError):
                compare_reports(weak_val, strong_locked, out_dir=root / 'out_split_mismatch')

            with self.assertRaises(ValueError):
                compare_reports(weak_locked, strong_locked, split='validation', out_dir=root / 'out_split_req')

    def test_real_detail_false_candidate_no_leak_mixture_cli_and_tune(self):
        with patched_dpapi(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_root, test_root = synthetic_librispeech_tree(root / 'LibriSpeech', n_train=16, n_test=40)
            case_dir = root / 'cases'
            reports_dir = root / 'reports'
            cache_dir = root / 'cache'
            cohort_dir = root / 'cohorts'
            build(test_root=test_root, case_dir=case_dir)

            # Build a compact subset of validation cases including 1 mixture case to test skip_mixture=False and spot checks fast.
            val_cases = load('validation', case_dir=case_dir)
            subset_public = []
            subset_hidden = []
            by_suite = {}
            for c in val_cases:
                by_suite.setdefault(c['suite'], []).append(c)
            picked = (
                by_suite['verify'][:6]
                + by_suite['verify_impostor'][:4]
                + by_suite['identify'][:2]
                + by_suite['robustness_noise'][:2]
                + by_suite['robustness_mixture'][:1]
            )
            for c in picked:
                hid = dict(c['hidden'])
                pub = {k: v for k, v in c.items() if k != 'hidden'}
                subset_public.append(pub)
                subset_hidden.append(hid)

            mini_case_dir = root / 'mini_cases'
            mini_case_dir.mkdir(parents=True, exist_ok=True)
            (mini_case_dir / 'VERSION').write_text('3\n', encoding='utf-8')
            (mini_case_dir / 'validation.public.json').write_text(json.dumps(subset_public), encoding='utf-8')
            (mini_case_dir / 'validation.hidden.json').write_text(json.dumps(subset_hidden), encoding='utf-8')

            # Verify _decision_for_grade handles robustness_mixture cleanly.
            mix_case = by_suite['robustness_mixture'][0]
            spk_a = mix_case['hidden']['true_speaker']
            self.assertEqual(_decision_for_grade(mix_case, {'paths': ['a.wav', 'b.wav'], 'names': [spk_a]}), ('accept', 'accept'))
            self.assertEqual(_decision_for_grade(mix_case, {'paths': ['a.wav', 'b.wav'], 'names': []}), ('accept', 'reject'))
            self.assertEqual(_decision_for_grade(mix_case, {}), ('accept', 'missing'))
            self.assertEqual(_decision_for_grade(mix_case, {'error': 'RuntimeError'}), ('accept', 'missing'))

            def _fake_separate(_src: str, dest_dir: Path) -> dict:
                dest_dir.mkdir(parents=True, exist_ok=True)
                # Copy clean probes for the two expected speakers into dest_dir to verify real end-to-end grading.
                spks = mix_case['hidden']['expect_speakers']
                out_paths = []
                for idx, spk in enumerate(spks):
                    src_wav = test_root / spk / '11' / f'{spk}-11-0000.flac'
                    dst_wav = dest_dir / f'sep_{idx}.wav'
                    import soundfile as sf
                    sig, sr = sf.read(str(src_wav), dtype='float32')
                    sf.write(str(dst_wav), sig, sr)
                    out_paths.append(str(dst_wav))
                return {'sample_rate': 8000, 'paths': out_paths, 'checkpoint': 'fake'}

            with ExitStack() as stack:
                stack.enter_context(patch.object(service_mod, 'TRAIN_ROOT', train_root))
                stack.enter_context(patch.object(embed_mod, 'COHORT_DIR', cohort_dir))
                stack.enter_context(patch('voiceprint.separate.separate_file', _fake_separate))

                # Ensure run_split respects an already-active outer tracemalloc tracer.
                was_tracing = tracemalloc.is_tracing()
                if not was_tracing:
                    tracemalloc.start()
                try:
                    strong_mini = run_split(
                        ToneEmbedder(), 'validation', True,
                        case_dir=mini_case_dir, reports_dir=reports_dir, cache_dir=cache_dir, skip_mixture=False,
                    )
                    self.assertTrue(tracemalloc.is_tracing())
                finally:
                    if not was_tracing and tracemalloc.is_tracing():
                        tracemalloc.stop()

                summary = json.loads((strong_mini / 'summary.json').read_text(encoding='utf-8'))
                resp_dim = summary['dimensions']['response']
                self.assertTrue(resp_dim['passed'], resp_dim)
                self.assertEqual(resp_dim['leak_count'], 0)
                real_ids = [c['id'] for c in resp_dim['checks'] if str(c['id']).startswith('real-')]
                self.assertGreaterEqual(len(real_ids), 5)
                self.assertIn('mixture', summary['dimensions']['robustness']['by_axis'])
                self.assertEqual(summary['skipped_suites'], [])
                self.assertEqual(summary['by_suite']['robustness_mixture']['accuracy'], 1.0)

                # Negative check: leaking score under detail=False must fail runner's response dimension.
                orig_verify = service_mod.VoiceAgent._verify

                def _leaking_verify(self_agent, rest):
                    out = orig_verify(self_agent, rest)
                    if not self_agent.detail and out.get('ok'):
                        out = dict(out)
                        out['reply'] = out['reply'] + ' 阈值 0.42'
                    return out

                with patch.object(service_mod.VoiceAgent, '_verify', _leaking_verify):
                    leak_dir = run_split(
                        ToneEmbedder(), 'validation', False,
                        case_dir=mini_case_dir, reports_dir=root / 'leak_reports', cache_dir=cache_dir, skip_mixture=True,
                    )
                leak_summary = json.loads((leak_dir / 'summary.json').read_text(encoding='utf-8'))
                self.assertFalse(leak_summary['dimensions']['response']['passed'])
                self.assertGreaterEqual(leak_summary['dimensions']['response']['leak_count'], 1)

                # CLI compare default invocation (stats vs vpnet -> writes both canonical and legacy names).
                weak_mini = run_split(
                    WeakEmbedder(), 'validation', True,
                    case_dir=mini_case_dir, reports_dir=reports_dir, cache_dir=cache_dir, skip_mixture=True,
                )
                cli_reports = root / 'cli_reports'
                _copy_report(weak_mini, cli_reports / 'stats_validation')
                _copy_report(strong_mini, cli_reports / 'vpnet_validation')
                _mutate_json(cli_reports / 'stats_validation' / 'summary.json', lambda d: d.update({'embedder': 'stats'}))
                _mutate_json(cli_reports / 'vpnet_validation' / 'summary.json', lambda d: d.update({'embedder': 'vpnet'}))

                with patch('voice_eval.gate.REPORTS', cli_reports), patch.object(cli_mod, 'REPORTS', cli_reports):
                    self.assertEqual(resolve_report_dir('stats', split='validation', reports_root=cli_reports), cli_reports / 'stats_validation')
                    self.assertEqual(resolve_report_dir('vpnet_validation', reports_root=cli_reports), cli_reports / 'vpnet_validation')

                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        cli_mod.main(['compare'])
                    self.assertTrue((cli_reports / 'compare_stats_vs_vpnet_validation.json').is_file())
                    self.assertTrue((cli_reports / 'compare_stats_vs_vpnet_validation.md').is_file())
                    self.assertTrue((cli_reports / 'compare_validation.json').is_file())
                    self.assertTrue((cli_reports / 'compare_validation.md').is_file())

                    # Explicit locked paths without --split auto-detect locked split cleanly.
                    _copy_report(weak_mini, cli_reports / 'stats_locked')
                    _copy_report(strong_mini, cli_reports / 'vpnet_locked')
                    _mutate_json(cli_reports / 'stats_locked' / 'summary.json', lambda d: d.update({'split': 'locked'}))
                    _mutate_json(cli_reports / 'vpnet_locked' / 'summary.json', lambda d: d.update({'split': 'locked'}))
                    with redirect_stdout(io.StringIO()):
                        cli_mod.main([
                            'compare',
                            '--base', str(cli_reports / 'stats_locked'),
                            '--cand', str(cli_reports / 'vpnet_locked'),
                        ])
                    locked_json = json.loads((cli_reports / 'compare_stats_vs_vpnet_locked.json').read_text(encoding='utf-8'))
                    self.assertEqual(locked_json['split'], 'locked')
                    self.assertFalse(locked_json['adopt'])

                # Verify CLI run supports --no-cache and --skip-mixture flags.
                with patch.object(cli_mod, 'run_all', return_value={'ok': True, 'errors': []}), \
                        patch.object(cli_mod, 'load_embedder', return_value=ToneEmbedder()), \
                        patch.object(cli_mod, 'run_split', return_value=strong_mini) as spy_run:
                    with redirect_stdout(io.StringIO()):
                        cli_mod.main(['run', '--embedder', 'vpnet', '--split', 'validation', '--no-cache', '--skip-mixture'])
                    spy_run.assert_called_once()
                    self.assertFalse(spy_run.call_args.kwargs['use_cache'])
                    self.assertTrue(spy_run.call_args.kwargs['skip_mixture'])

                # Verify tune.py compatibility on full v3 validation cases.
                tune_pairs = tune_mod.pairs(ToneEmbedder(), case_dir=case_dir)
                self.assertEqual(len(tune_pairs), 400)
                with patch.object(tune_mod, 'load_embedder', return_value=ToneEmbedder()):
                    tune_buf = io.StringIO()
                    with redirect_stdout(tune_buf):
                        tune_mod.main(['--embedder', 'vpnet', '--case-dir', str(case_dir), '--train-root', str(train_root)])
                    self.assertIn('validation trials 400', tune_buf.getvalue())


if __name__ == '__main__':
    unittest.main()
