"""Tests for Warm RTF uncached micro-benchmark, worker RSS aggregation, offline audit, and Tier 3 RTF gate."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests._helpers import patched_dpapi, synthetic_librispeech_tree
from voice_eval.cache import CachedEmbedder
from voice_eval.cases import build as build_cases
from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav, sandbox_cohort
from voice_eval.dims.perf import measure_uncached_rtf, summarize_perf
from voice_eval.gate import PERF_RTF_ABS_WARN, PERF_RTF_RATIO_WARN, _evaluate_tier3_warnings
from voice_eval.isolated_worker import WorkerCircuitOpenError
from voice_eval.runner import run_split


class _CircuitTrippingEmbedder(ToneEmbedder):
    is_isolated = True

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.circuit_open = False
        self.restart_count = 0
        self.timeout_count = 0
        self.peak_worker_rss_mb = 42.0

    def embed_file(self, path: str):
        self.calls += 1
        self.restart_count += 1
        self.circuit_open = True
        raise WorkerCircuitOpenError('simulated circuit open')


class TestPerfRtfAndOffline(unittest.TestCase):
    def test_cached_embedder_keeps_stats_keys_and_tracks_audio_seconds(self):
        with TemporaryDirectory(prefix='vprtf_cache_') as tmp:
            root = Path(tmp)
            w1 = generate_speaker_wav(root / 'w1.wav', speaker_index=0, duration=2.0, seed=1)
            w2 = generate_speaker_wav(root / 'w2.wav', speaker_index=1, duration=3.0, seed=2)
            cached = CachedEmbedder(ToneEmbedder(), cache_dir=root / 'cache')
            cached.embed_file(str(w1))
            cached.embed_file(str(w1))  # hit
            cached.embed_file(str(w2))

            st = cached.stats()
            self.assertEqual(
                set(st.keys()),
                {'enabled', 'hits', 'misses', 'total', 'hit_rate', 'compute_ms_list', 'lookup_ms_list'},
            )
            durs = cached.compute_audio_seconds()
            self.assertEqual(len(durs), 2)
            self.assertAlmostEqual(durs[0], 2.0, places=2)
            self.assertAlmostEqual(durs[1], 3.0, places=2)

            cached.reset_stats()
            self.assertEqual(cached.compute_audio_seconds(), [])

    def test_measure_uncached_rtf_and_summarize_perf_cold_warm_split(self):
        with TemporaryDirectory(prefix='vprtf_bench_') as tmp:
            root = Path(tmp)
            paths = [
                str(generate_speaker_wav(root / f's_{i}.wav', speaker_index=i, duration=2.0, seed=i))
                for i in range(6)
            ]
            bench = measure_uncached_rtf(ToneEmbedder(), paths, max_samples=5)
            self.assertEqual(bench['source'], 'micro_bench')
            self.assertEqual(bench['n_samples'], 5)
            self.assertGreater(bench['cold_rtf'], 0.0)
            self.assertGreater(bench['warm_rtf'], 0.0)
            self.assertGreater(bench['p50_warm_rtf'], 0.0)
            self.assertGreater(bench['p95_warm_rtf'], 0.0)
            self.assertAlmostEqual(bench['total_audio_seconds'], 10.0, places=1)
            self.assertAlmostEqual(bench['warm_audio_seconds'], 8.0, places=1)

            perf = summarize_perf(
                case_timings_ms=[10.0, 20.0],
                step_timings_ms={'verify': [5.0, 8.0]},
                cache_stats=None,
                peak_tracemalloc_mb=1.5,
                rss_mb=64.0,
                rtf_bench=bench,
                worker_rss_mb=256.0,
            )
            self.assertEqual(perf['rtf']['source'], 'micro_bench')
            self.assertEqual(perf['parent_rss_mb'], 64.0)
            self.assertEqual(perf['worker_rss_mb'], 256.0)
            self.assertEqual(perf['rss_mb'], 256.0)
            self.assertIsNotNone(perf['cold_embed_ms'])
            self.assertIsNotNone(perf['warm_embed']['mean_ms'])

    def test_gate_tier3_warns_on_warm_rtf_regression(self):
        self.assertEqual(PERF_RTF_RATIO_WARN, 1.5)
        self.assertEqual(PERF_RTF_ABS_WARN, 0.05)

        def _make_summary(warm_rtf: float) -> dict:
            return {
                'dimensions': {
                    'robustness': {'by_variant': {}},
                    'perf': {'case': {}, 'embed_compute': {}, 'rtf': {'warm_rtf': warm_rtf}},
                }
            }

        warns_bad, _ = _evaluate_tier3_warnings(_make_summary(0.10), _make_summary(0.30))
        self.assertTrue(any('Warm RTF' in w for w in warns_bad), f'expected Warm RTF warning, got {warns_bad}')

        warns_small, _ = _evaluate_tier3_warnings(_make_summary(0.01), _make_summary(0.02))
        self.assertFalse(any('Warm RTF' in w for w in warns_small))

    def test_run_split_emits_runtime_audit_and_uncached_rtf_even_on_100pct_cache_hit(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vprtf_run_') as tmp:
            root = Path(tmp)
            _train, test_root = synthetic_librispeech_tree(root / 'corpus', n_train=4, n_test=40)
            case_dir = root / 'cases'
            build_cases(test_root=test_root, case_dir=case_dir)
            # Slice validation to just 4 verify cases for fast E2E verification.
            pub = json.loads((case_dir / 'validation.public.json').read_text(encoding='utf-8'))
            hid = json.loads((case_dir / 'validation.hidden.json').read_text(encoding='utf-8'))
            sel_ids = {c['id'] for c in pub if c['suite'] == 'verify'}
            sel_ids = set(sorted(sel_ids)[:4])
            pub = [c for c in pub if c['id'] in sel_ids]
            hid = [c for c in hid if c['id'] in sel_ids]
            (case_dir / 'validation.public.json').write_text(json.dumps(pub, ensure_ascii=False), encoding='utf-8')
            (case_dir / 'validation.hidden.json').write_text(json.dumps(hid, ensure_ascii=False), encoding='utf-8')

            cache_dir = root / 'cache'
            reports_dir = root / 'reports'
            # Run 1 populates disk cache
            run_split(embedder, 'validation', with_security=False, case_dir=case_dir, reports_dir=reports_dir, cache_dir=cache_dir, skip_mixture=True)
            # Run 2 hits 100% disk cache, yet still produces uncached micro_bench Warm RTF & runtime_audit
            out_dir = run_split(embedder, 'validation', with_security=False, case_dir=case_dir, reports_dir=reports_dir, cache_dir=cache_dir, skip_mixture=True)
            summary = json.loads((out_dir / 'summary.json').read_text(encoding='utf-8'))
            perf = summary['dimensions']['perf']
            self.assertEqual(perf['cache']['hit_rate'], 1.0)
            self.assertEqual(perf['rtf']['source'], 'micro_bench')
            self.assertIsNotNone(perf['rtf']['warm_rtf'])
            self.assertGreater(perf['rtf']['warm_rtf'], 0.0)
            audit = summary['runtime_audit']
            self.assertTrue(audit['offline_enforced'])
            self.assertEqual(audit['model_fingerprint']['model_id'], embedder.model_id)
            self.assertFalse(audit['worker_isolated'])

    def test_run_split_aborts_immediately_when_isolated_worker_circuit_opens(self):
        embedder = _CircuitTrippingEmbedder()
        with patched_dpapi(), sandbox_cohort(ToneEmbedder()), TemporaryDirectory(prefix='vprtf_cb_') as tmp:
            root = Path(tmp)
            _train, test_root = synthetic_librispeech_tree(root / 'corpus', n_train=4, n_test=40)
            case_dir = root / 'cases'
            build_cases(test_root=test_root, case_dir=case_dir)
            pub = json.loads((case_dir / 'validation.public.json').read_text(encoding='utf-8'))
            hid = json.loads((case_dir / 'validation.hidden.json').read_text(encoding='utf-8'))
            sel_ids = set(sorted({c['id'] for c in pub if c['suite'] == 'verify'})[:6])
            pub = [c for c in pub if c['id'] in sel_ids]
            hid = [c for c in hid if c['id'] in sel_ids]
            (case_dir / 'validation.public.json').write_text(json.dumps(pub, ensure_ascii=False), encoding='utf-8')
            (case_dir / 'validation.hidden.json').write_text(json.dumps(hid, ensure_ascii=False), encoding='utf-8')

            with self.assertRaises(WorkerCircuitOpenError):
                run_split(
                    embedder,
                    'validation',
                    with_security=False,
                    case_dir=case_dir,
                    reports_dir=root / 'reports',
                    use_cache=False,
                    skip_mixture=True,
                )
            # Must abort right after the first failing case, not burn through all 6 cases.
            self.assertLessEqual(embedder.calls, 2)


if __name__ == '__main__':
    unittest.main()
