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


if __name__ == '__main__':
    unittest.main()
