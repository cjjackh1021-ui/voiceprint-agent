"""Unit tests for voice_eval evaluation dimensions (Task 4: task math & aggregation)."""
import math
import unittest

from voice_eval.dims.task import (
    MINDCF_PARAMS,
    MISSING_NORM_SCORE,
    MISSING_RAW_SCORE,
    compute_eer,
    compute_mindcf,
    evaluate_task,
    score_distribution,
    summarize_verify_group,
)


def _make_verify_row(
    case_id: str,
    suite: str,
    expect: str,
    decision: str,
    score: float | None = None,
    raw_score: float | None = None,
) -> dict:
    result = {'decision': decision}
    if score is not None:
        result['score'] = score
    if raw_score is not None:
        result['raw_score'] = raw_score
    return {
        'id': case_id,
        'suite': suite,
        'split': 'validation',
        'expect': expect,
        'decision': decision,
        'correct': decision == expect,
        'false_accept': expect == 'reject' and decision == 'accept',
        'false_reject': expect == 'accept' and decision == 'reject',
        'reviewed': decision == 'uncertain',
        'delivered': decision in ('accept', 'reject', 'uncertain', 'unknown'),
        'trace': [{'command': '验证', 'tool': 'verify', 'ok': score is not None, 'result': result, 'reply': ''}],
    }


class TaskDimensionTests(unittest.TestCase):
    def test_constants_match_protocol(self):
        self.assertEqual(MISSING_NORM_SCORE, -10.0)
        self.assertEqual(MISSING_RAW_SCORE, -1.0)
        self.assertEqual(MINDCF_PARAMS, {'p_target': 0.01, 'c_miss': 1.0, 'c_fa': 1.0})

    def test_compute_eer_separable_identical_inverted_tied_and_degenerate(self):
        eer, thr = compute_eer([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
        self.assertEqual(eer, 0.0)
        self.assertIsNotNone(thr)
        self.assertGreaterEqual(thr, 0.2)
        self.assertLessEqual(thr, 0.8)

        eer_id, _ = compute_eer([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [1, 1, 1, 0, 0, 0])
        self.assertAlmostEqual(eer_id, 0.5, places=6)

        eer_tied, thr_tied = compute_eer([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0])
        self.assertAlmostEqual(eer_tied, 0.5, places=6)
        self.assertAlmostEqual(thr_tied, 0.5, places=6)

        eer_inv, _ = compute_eer([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
        self.assertAlmostEqual(eer_inv, 1.0, places=6)

        # Tied scores must jump atomically regardless of input ordering.
        eer_a, _ = compute_eer([0.9, 0.9, 0.8], [1, 0, 0])
        eer_b, _ = compute_eer([0.9, 0.8, 0.9], [0, 0, 1])
        self.assertAlmostEqual(eer_a, eer_b, places=9)
        self.assertAlmostEqual(eer_a, 1.0 / 3.0, places=6)

        # Empty or single-class inputs must return (None, None) rather than faking 0.0.
        self.assertEqual(compute_eer([], []), (None, None))
        self.assertEqual(compute_eer([0.5, 0.6], [1, 1]), (None, None))
        self.assertEqual(compute_eer([0.5, 0.6], [0, 0]), (None, None))

        with self.assertRaises(ValueError):
            compute_eer([0.5], [1, 0])

    def test_compute_mindcf_bounds_and_exact_values(self):
        sep = compute_mindcf([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
        self.assertEqual(sep['min_dcf_raw'], 0.0)
        self.assertEqual(sep['min_dcf_norm'], 0.0)
        self.assertEqual(sep['params'], MINDCF_PARAMS)

        # Trivial/random overlap is upper-bounded by 1.0 when normalized.
        tied = compute_mindcf([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0])
        self.assertAlmostEqual(tied['min_dcf_raw'], 0.01, places=6)
        self.assertAlmostEqual(tied['min_dcf_norm'], 1.0, places=6)

        # Known operating point: 2 pos [0.9, 0.2], 10 neg [0.8] + [0.1]*9 -> at t=0.9: FRR=0.5, FAR=0.0 => DCF=0.005
        partial = compute_mindcf([0.9, 0.2, 0.8] + [0.1] * 9, [1, 1] + [0] * 10)
        self.assertAlmostEqual(partial['min_dcf_raw'], 0.005, places=6)
        self.assertAlmostEqual(partial['min_dcf_norm'], 0.5, places=6)
        self.assertAlmostEqual(partial['threshold'], 0.9, places=6)

        empty = compute_mindcf([0.5], [1])
        self.assertIsNone(empty['min_dcf_raw'])
        self.assertIsNone(empty['min_dcf_norm'])
        self.assertIsNone(empty['threshold'])

    def test_score_distribution_and_d_prime(self):
        dist = score_distribution([2.0, 4.0, 0.0, 2.0], [1, 1, 0, 0])
        self.assertEqual(dist['genuine']['n'], 2)
        self.assertAlmostEqual(dist['genuine']['mean'], 3.0, places=6)
        self.assertAlmostEqual(dist['genuine']['std'], 1.0, places=6)
        self.assertAlmostEqual(dist['genuine']['p50'], 3.0, places=6)
        self.assertEqual(dist['impostor']['n'], 2)
        self.assertAlmostEqual(dist['impostor']['mean'], 1.0, places=6)
        expected_dp = (3.0 - 1.0) / math.sqrt(0.5 * (1.0 ** 2 + 1.0 ** 2) + 1e-8)
        self.assertAlmostEqual(dist['d_prime'], expected_dp, places=6)

        # Constant scores should remain finite thanks to 1e-8 denominator guard.
        const = score_distribution([1.0, 0.0], [1, 0])
        self.assertTrue(math.isfinite(const['d_prime']))
        self.assertAlmostEqual(const['d_prime'], 1.0 / math.sqrt(1e-8), places=4)

        missing_cls = score_distribution([1.0, 2.0], [1, 1])
        self.assertEqual(missing_cls['impostor']['n'], 0)
        self.assertIsNone(missing_cls['impostor']['mean'])
        self.assertIsNone(missing_cls['d_prime'])

    def test_summarize_verify_group_imputes_missing_scores_and_penalizes_weak_models(self):
        # Model A fails to enroll one genuine trial ('unknown', missing score), while scoring well on the easy one.
        rows = [
            _make_verify_row('v1', 'verify', 'accept', 'accept', score=3.0, raw_score=0.9),
            _make_verify_row('v2', 'verify', 'accept', 'unknown', score=None, raw_score=None),
            _make_verify_row('v3', 'verify', 'reject', 'reject', score=-1.0, raw_score=0.1),
            _make_verify_row('v4', 'verify', 'reject', 'reject', score=0.0, raw_score=0.2),
        ]
        summary = summarize_verify_group(rows)
        self.assertEqual(summary['n'], 4)
        self.assertEqual(summary['valid_trials'], 3)
        self.assertEqual(summary['imputed_missing_trials'], 1)
        self.assertEqual(summary['raw_eer'], 0.0)
        # Missing genuine score (-10.0) falls below both impostors (-1.0, 0.0) -> EER must be > 0!
        self.assertIsNotNone(summary['eer'])
        self.assertGreater(summary['eer'], 0.0)
        self.assertEqual(summary['score_stats']['genuine']['n'], 2)
        self.assertAlmostEqual(summary['score_stats']['genuine']['mean'], (3.0 + MISSING_NORM_SCORE) / 2.0, places=6)
        self.assertAlmostEqual(summary['raw_score_stats']['genuine']['mean'], (0.9 + MISSING_RAW_SCORE) / 2.0, places=6)

        # When all genuine trials are missing scores, raw_eer must be None while eer reflects imputed penalty.
        all_missing_pos = [
            _make_verify_row('v1', 'verify', 'accept', 'unknown', score=None, raw_score=None),
            _make_verify_row('v2', 'verify', 'reject', 'reject', score=0.0, raw_score=0.2),
        ]
        res_all_miss = summarize_verify_group(all_missing_pos)
        self.assertEqual(res_all_miss['valid_trials'], 1)
        self.assertEqual(res_all_miss['imputed_missing_trials'], 1)
        self.assertIsNone(res_all_miss['raw_eer'])
        self.assertAlmostEqual(res_all_miss['eer'], 1.0, places=6)

    def test_evaluate_task_combines_verify_and_verify_impostor_for_primary_eer(self):
        rows = [
            _make_verify_row('v-a1', 'verify', 'accept', 'accept', score=2.5, raw_score=0.85),
            _make_verify_row('v-r1', 'verify', 'reject', 'reject', score=-0.5, raw_score=0.15),
            _make_verify_row('vi-1', 'verify_impostor', 'reject', 'accept', score=3.0, raw_score=0.90),
            _make_verify_row('vi-2', 'verify_impostor', 'reject', 'reject', score=-1.0, raw_score=0.10),
            _make_verify_row('id-1', 'identify', 'accept', 'accept', score=2.0, raw_score=0.80),
        ]
        task_out = evaluate_task(rows)
        self.assertEqual(task_out['by_suite']['verify']['n'], 2)
        self.assertAlmostEqual(task_out['by_suite']['verify']['accuracy'], 1.0, places=6)
        self.assertEqual(task_out['by_suite']['verify_impostor']['n'], 2)
        self.assertAlmostEqual(task_out['by_suite']['verify_impostor']['false_accept_rate'], 0.5, places=6)
        self.assertEqual(task_out['verify_metrics']['n'], 2)
        self.assertEqual(task_out['verify_metrics']['eer'], 0.0)
        self.assertEqual(task_out['verify_all_metrics']['n'], 4)
        self.assertGreater(task_out['verify_all_metrics']['eer'], 0.0)
        self.assertEqual(task_out['eer_primary'], task_out['verify_all_metrics']['eer'])


class PerfDimensionTests(unittest.TestCase):
    def test_percentile_stats_empty_and_non_empty(self):
        import numpy as np

        from voice_eval.dims.perf import percentile_stats

        empty = percentile_stats([])
        self.assertEqual(empty, {'n': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'total_ms': 0.0})

        stats = percentile_stats(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        self.assertEqual(stats['n'], 5)
        self.assertAlmostEqual(stats['mean_ms'], 30.0, places=6)
        self.assertAlmostEqual(stats['p50_ms'], 30.0, places=6)
        self.assertAlmostEqual(stats['p95_ms'], 48.0, places=6)
        self.assertAlmostEqual(stats['total_ms'], 150.0, places=6)

        with self.assertRaises(ValueError):
            percentile_stats([1.0, float('nan')])

    def test_current_rss_mb_handles_darwin_linux_and_missing_resource(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from voice_eval.dims import perf

        real_rss = perf.current_rss_mb()
        if real_rss is not None:
            self.assertGreater(real_rss, 0.0)

        fake_usage = SimpleNamespace(ru_maxrss=104857600)  # 100 MiB if bytes
        with patch.object(perf, 'resource', SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _who: fake_usage)), \
                patch.object(perf.sys, 'platform', 'darwin'):
            self.assertAlmostEqual(perf.current_rss_mb(), 100.0, places=6)

        fake_linux_usage = SimpleNamespace(ru_maxrss=102400)  # 100 MiB if KiB
        with patch.object(perf, 'resource', SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _who: fake_linux_usage)), \
                patch.object(perf.sys, 'platform', 'linux'):
            self.assertAlmostEqual(perf.current_rss_mb(), 100.0, places=6)

        with patch.object(perf, 'resource', None), patch.object(perf.sys, 'platform', 'linux'):
            self.assertIsNone(perf.current_rss_mb())

        with patch.object(perf, 'resource', None), patch.object(perf.sys, 'platform', 'win32'), \
                patch.object(perf, '_windows_rss_mb', return_value=None):
            self.assertIsNone(perf.current_rss_mb())

    def test_windows_rss_mb_sets_64bit_argtypes_restype_and_reads_working_set(self):
        import ctypes
        from ctypes import wintypes
        from types import SimpleNamespace
        from unittest.mock import patch

        from voice_eval.dims import perf

        handle_64 = 0x00000123456789AB
        observed = {}

        def fake_get_current_process():
            return handle_64

        def fake_get_process_memory_info(handle, counters_ref, cb):
            observed['handle'] = handle
            observed['cb'] = cb
            counters_ref._obj.WorkingSetSize = 64 * 1024 * 1024
            return 1

        kernel32 = SimpleNamespace(GetCurrentProcess=fake_get_current_process)
        psapi = SimpleNamespace(GetProcessMemoryInfo=fake_get_process_memory_info)
        dll_calls = []

        def fake_windll(name, use_last_error=False):
            dll_calls.append((name, use_last_error))
            if name == 'kernel32':
                return kernel32
            if name == 'psapi':
                return psapi
            raise AssertionError(f'unexpected DLL {name!r}')

        with patch.object(perf, 'resource', None), patch.object(perf.sys, 'platform', 'win32'), \
                patch.object(perf.ctypes, 'WinDLL', side_effect=fake_windll, create=True):
            rss = perf.current_rss_mb()

        self.assertAlmostEqual(rss, 64.0, places=6)
        self.assertEqual(dll_calls, [('kernel32', True), ('psapi', True)])
        self.assertEqual(kernel32.GetCurrentProcess.argtypes, [])
        self.assertIs(kernel32.GetCurrentProcess.restype, wintypes.HANDLE)
        argtypes = psapi.GetProcessMemoryInfo.argtypes
        self.assertEqual(len(argtypes), 3)
        self.assertIs(argtypes[0], wintypes.HANDLE)
        self.assertTrue(issubclass(argtypes[1], ctypes._Pointer))
        self.assertEqual(argtypes[1]._type_.__name__, '_ProcessMemoryCounters')
        self.assertIs(argtypes[2], wintypes.DWORD)
        self.assertIs(psapi.GetProcessMemoryInfo.restype, wintypes.BOOL)
        self.assertEqual(observed['handle'], handle_64)
        self.assertGreater(observed['cb'], 0)

        psapi_fail = SimpleNamespace(GetProcessMemoryInfo=lambda _h, _c, _cb: 0)
        with patch.object(perf, 'resource', None), patch.object(perf.sys, 'platform', 'win32'), \
                patch.object(
                    perf.ctypes,
                    'WinDLL',
                    side_effect=lambda name, use_last_error=False: kernel32 if name == 'kernel32' else psapi_fail,
                    create=True,
                ):
            self.assertIsNone(perf.current_rss_mb())

    def test_summarize_perf_includes_case_steps_embed_compute_and_memory(self):
        from voice_eval.dims.perf import summarize_perf

        cache_stats = {
            'hits': 3,
            'misses': 2,
            'total': 5,
            'hit_rate': 0.6,
            'compute_ms_list': [10.0, 30.0],
            'lookup_ms_list': [0.2, 0.4, 0.6],
        }
        summary = summarize_perf(
            case_timings_ms=[12.0, 24.0, 36.0],
            step_timings_ms={'enroll': [15.0, 25.0], 'verify': [5.0, 7.0, 9.0]},
            cache_stats=cache_stats,
            peak_tracemalloc_mb=4.25,
            rss_mb=92.5,
        )
        self.assertEqual(summary['case']['n'], 3)
        self.assertAlmostEqual(summary['case']['p50_ms'], 24.0, places=6)
        self.assertAlmostEqual(summary['total_seconds'], 0.072, places=6)
        self.assertEqual(summary['steps']['enroll']['n'], 2)
        self.assertEqual(summary['steps']['verify']['n'], 3)
        self.assertEqual(summary['embed_compute']['n'], 2)
        self.assertAlmostEqual(summary['embed_compute']['mean_ms'], 20.0, places=6)
        self.assertEqual(summary['embed_lookup']['n'], 3)
        self.assertEqual(summary['cache']['hits'], 3)
        self.assertEqual(summary['cache']['misses'], 2)
        self.assertAlmostEqual(summary['cache']['hit_rate'], 0.6, places=6)
        self.assertAlmostEqual(summary['peak_tracemalloc_mb'], 4.25, places=6)
        self.assertAlmostEqual(summary['rss_mb'], 92.5, places=6)

        no_cache = summarize_perf([5.0], {}, None, 1.0, None)
        self.assertFalse(no_cache['cache']['enabled'])
        self.assertEqual(no_cache['embed_compute']['n'], 0)
        self.assertIsNone(no_cache['rss_mb'])

        disabled_stats = summarize_perf(
            [5.0],
            {},
            {
                'enabled': False,
                'hits': 0,
                'misses': 2,
                'total': 2,
                'hit_rate': 0.0,
                'compute_ms_list': [1.0, 2.0],
                'lookup_ms_list': [],
            },
            1.0,
            None,
        )
        self.assertFalse(disabled_stats['cache']['enabled'])
        self.assertEqual(disabled_stats['cache']['misses'], 2)

    def test_sandbox_log_filters_lru_bounded_and_readonly(self):
        from voice_eval.dims._sandbox import _log_filters

        _log_filters.cache_clear()
        self.assertEqual(_log_filters.cache_info().maxsize, 8)
        fb = _log_filters(257, 16000)
        self.assertFalse(fb.flags.writeable)
        with self.assertRaises(ValueError):
            fb[0, 0] = 1.0
        for k in range(12):
            _log_filters(128 + k, 16000)
        self.assertLessEqual(_log_filters.cache_info().currsize, 8)


class RobustnessDimensionTests(unittest.TestCase):
    def test_evaluate_robustness_groups_by_axis_and_variant_and_computes_worst_drop(self):
        from tests._helpers import patched_dpapi
        from voice_eval.dims.robustness import evaluate_robustness

        def _rob_row(cid: str, suite: str, axis: str, variant: str, expect: str, decision: str, score: float | None):
            row = _make_verify_row(cid, suite, expect, decision, score=score, raw_score=0.5 if score is not None else None)
            row['axis'] = axis
            row['variant'] = variant
            return row

        rows = [
            # Non-robustness rows must be ignored by by_axis/by_variant.
            _make_verify_row('v1', 'verify', 'accept', 'accept', score=3.0, raw_score=0.9),
            # Noise axis: 2 variants
            _rob_row('n1', 'robustness_noise', 'noise', 'white_snr20', 'accept', 'accept', 2.5),
            _rob_row('n2', 'robustness_noise', 'noise', 'white_snr20', 'reject', 'reject', -1.0),
            _rob_row('n3', 'robustness_noise', 'noise', 'babble_snr0', 'accept', 'reject', -0.5),
            _rob_row('n4', 'robustness_noise', 'noise', 'babble_snr0', 'reject', 'reject', -1.2),
            # Duration axis
            _rob_row('d1', 'robustness_duration', 'duration', 'dur_1.0s', 'accept', 'uncertain', 0.1),
            _rob_row('d2', 'robustness_duration', 'duration', 'dur_1.0s', 'reject', 'accept', 1.2),
            # Channel axis
            _rob_row('c1', 'robustness_channel', 'channel', 'tel8k_ulaw', 'accept', 'accept', 2.0),
            _rob_row('c2', 'robustness_channel', 'channel', 'tel8k_ulaw', 'reject', 'reject', -0.8),
            # Reverb axis
            _rob_row('r1', 'robustness_reverb', 'reverb', 'rt60_0.3s', 'accept', 'accept', 2.2),
            _rob_row('r2', 'robustness_reverb', 'reverb', 'rt60_0.3s', 'reject', 'reject', -0.9),
            # Mixture axis (accept-only identification style -> EER is None)
            _rob_row('m1', 'robustness_mixture', 'mixture', 'mix2_0db_8k', 'accept', 'accept', None),
        ]

        with patched_dpapi():
            res = evaluate_robustness(rows, baseline_verify_accuracy=1.0)

        self.assertEqual(set(res['by_axis'].keys()), {'noise', 'duration', 'channel', 'reverb', 'mixture'})
        self.assertEqual(
            set(res['by_variant'].keys()),
            {'white_snr20', 'babble_snr0', 'dur_1.0s', 'tel8k_ulaw', 'rt60_0.3s', 'mix2_0db_8k'},
        )
        self.assertAlmostEqual(res['by_variant']['white_snr20']['accuracy'], 1.0, places=6)
        self.assertAlmostEqual(res['by_variant']['white_snr20']['acc_drop'], 0.0, places=6)
        self.assertEqual(res['by_variant']['white_snr20']['eer'], 0.0)
        self.assertAlmostEqual(res['by_variant']['babble_snr0']['accuracy'], 0.5, places=6)
        self.assertAlmostEqual(res['by_variant']['babble_snr0']['acc_drop'], 0.5, places=6)
        self.assertAlmostEqual(res['by_variant']['dur_1.0s']['accuracy'], 0.0, places=6)
        self.assertGreater(res['by_variant']['dur_1.0s']['eer'], 0.5)
        self.assertIsNone(res['by_variant']['mix2_0db_8k']['eer'])
        self.assertAlmostEqual(res['by_axis']['noise']['accuracy'], 0.75, places=6)
        self.assertAlmostEqual(res['worst_drop'], 1.0, places=6)
        self.assertTrue(res['fault_recovery']['passed'])
        self.assertTrue(res['consistency']['passed'])

        with patched_dpapi():
            no_base = evaluate_robustness(rows, baseline_verify_accuracy=None)
        self.assertIsNone(no_base['worst_drop'])
        self.assertIsNone(no_base['by_variant']['white_snr20']['acc_drop'])


class AuditUpgradeTests(unittest.TestCase):
    def test_audit_run_all_passes_with_dim_checks_and_catches_corruptions(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from tests._helpers import patched_dpapi
        from voice_eval import audit

        with patched_dpapi():
            ok_res = audit.run_all()
            self.assertTrue(ok_res['ok'], ok_res)
            self.assertEqual(ok_res['errors'], [])
            self.assertEqual(audit.dim_checks(), [])

            # Negative subtest 1: corrupt EER calculation caught by dim_checks / run_all
            with patch('voice_eval.dims.task.compute_eer', return_value=(0.9, 0.5)):
                bad_eer = audit.run_all()
            self.assertFalse(bad_eer['ok'])
            self.assertTrue(bad_eer['errors'])

            # Negative subtest 2: corrupt protocol.json caught by dim_checks / run_all
            with tempfile.TemporaryDirectory() as tmp:
                bad_proto = Path(tmp) / 'protocol.json'
                bad_proto.write_text(json.dumps({'case_version': 2, 'dimensions': ['task']}), encoding='utf-8')
                with patch.object(audit, 'PROTOCOL_PATH', bad_proto):
                    bad_proto_res = audit.run_all()
            self.assertFalse(bad_proto_res['ok'])
            self.assertTrue(bad_proto_res['errors'])


class GateTier3Tests(unittest.TestCase):
    def test_consistency_score_drift_emits_tier3_warning_even_when_agreement_is_one(self):
        from voice_eval.gate import _evaluate_tier3_warnings

        base_summary = {'dimensions': {'perf': {}, 'robustness': {}}}
        cand_summary = {
            'dimensions': {
                'perf': {},
                'robustness': {
                    'fault_recovery': {'recovery_rate': 1.0},
                    'consistency': {
                        'passed': False,
                        'agreement_rate': 1.0,
                        'max_score_spread': 0.025,
                    },
                },
            }
        }
        warnings, _ = _evaluate_tier3_warnings(base_summary, cand_summary)
        self.assertEqual(len(warnings), 1)
        self.assertIn('0.025', warnings[0])


if __name__ == '__main__':
    unittest.main()
