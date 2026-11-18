"""Dimension 1: task completion metrics (EER, minDCF, score distributions, and suite aggregates)."""
import math
from typing import Sequence

import numpy as np

from voice_eval.grader import aggregate

MISSING_NORM_SCORE = -10.0
MISSING_RAW_SCORE = -1.0
MINDCF_PARAMS = {'p_target': 0.01, 'c_miss': 1.0, 'c_fa': 1.0}
STANDARD_SUITES = ('verify', 'verify_impostor', 'identify', 'enroll_quality', 'security')


def _grouped_roc_counts(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[list[tuple[float, int, int]], int, int]:
    """Aggregate (score -> [neg_count, pos_count]) sorted descending by score."""
    if len(scores) != len(labels):
        raise ValueError(f'scores ({len(scores)}) and labels ({len(labels)}) must have equal length')
    grouped: dict[float, list[int]] = {}
    n_pos = 0
    n_neg = 0
    for raw_score, raw_label in zip(scores, labels):
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f'non-finite score encountered: {raw_score!r}')
        label = int(raw_label)
        if label not in (0, 1):
            raise ValueError(f'labels must be 0 or 1, got {raw_label!r}')
        bucket = grouped.setdefault(score, [0, 0])
        bucket[label] += 1
        if label == 1:
            n_pos += 1
        else:
            n_neg += 1
    ordered = [(score, counts[0], counts[1]) for score, counts in sorted(grouped.items(), key=lambda x: x[0], reverse=True)]
    return ordered, n_pos, n_neg


def compute_eer(scores: Sequence[float], labels: Sequence[int]) -> tuple[float | None, float | None]:
    """Compute Equal Error Rate via atomic tie-grouped ROC sweep and linear crossing interpolation."""
    ordered, n_pos, n_neg = _grouped_roc_counts(scores, labels)
    if n_pos == 0 or n_neg == 0:
        return None, None

    tp = 0
    fp = 0
    prev_thr = ordered[0][0]
    prev_far = 0.0
    prev_frr = 1.0
    prev_diff = prev_far - prev_frr

    for thr, neg_cnt, pos_cnt in ordered:
        tp += pos_cnt
        fp += neg_cnt
        far = fp / n_neg
        frr = (n_pos - tp) / n_pos
        diff = far - frr
        if diff == 0.0:
            return float(far), float(thr)
        if prev_diff < 0.0 < diff:
            weight = -prev_diff / (diff - prev_diff)
            eer = prev_far + weight * (far - prev_far)
            eer_thr = prev_thr + weight * (thr - prev_thr)
            return float(min(max(eer, 0.0), 1.0)), float(eer_thr)
        prev_thr, prev_far, prev_frr, prev_diff = thr, far, frr, diff

    return float(prev_far), float(prev_thr)


def compute_mindcf(
    scores: Sequence[float],
    labels: Sequence[int],
    p_target: float = MINDCF_PARAMS['p_target'],
    c_miss: float = MINDCF_PARAMS['c_miss'],
    c_fa: float = MINDCF_PARAMS['c_fa'],
) -> dict:
    """Compute NIST SRE raw and normalized minimum Detection Cost Function."""
    if not (0.0 < p_target < 1.0) or c_miss <= 0.0 or c_fa <= 0.0:
        raise ValueError(f'invalid DCF parameters: p_target={p_target}, c_miss={c_miss}, c_fa={c_fa}')
    params = {'p_target': float(p_target), 'c_miss': float(c_miss), 'c_fa': float(c_fa)}
    ordered, n_pos, n_neg = _grouped_roc_counts(scores, labels)
    if n_pos == 0 or n_neg == 0:
        return {'min_dcf_raw': None, 'min_dcf_norm': None, 'threshold': None, 'params': params}

    denom = min(c_miss * p_target, c_fa * (1.0 - p_target))
    best_dcf = c_miss * p_target * 1.0 + c_fa * (1.0 - p_target) * 0.0
    best_thr = float(ordered[0][0]) + 1e-6

    tp = 0
    fp = 0
    for thr, neg_cnt, pos_cnt in ordered:
        tp += pos_cnt
        fp += neg_cnt
        far = fp / n_neg
        frr = (n_pos - tp) / n_pos
        dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
        if dcf < best_dcf:
            best_dcf = dcf
            best_thr = float(thr)

    return {
        'min_dcf_raw': float(best_dcf),
        'min_dcf_norm': float(best_dcf / denom),
        'threshold': float(best_thr),
        'params': params,
    }


def _group_stats(values: list[float]) -> dict:
    if not values:
        return {'n': 0, 'mean': None, 'std': None, 'p05': None, 'p50': None, 'p95': None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        'n': int(arr.size),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=0)),
        'p05': float(np.quantile(arr, 0.05)),
        'p50': float(np.quantile(arr, 0.50)),
        'p95': float(np.quantile(arr, 0.95)),
    }


def score_distribution(scores: Sequence[float], labels: Sequence[int]) -> dict:
    """Summarize genuine/impostor score distributions and Cohen/d-prime separability."""
    if len(scores) != len(labels):
        raise ValueError(f'scores ({len(scores)}) and labels ({len(labels)}) must have equal length')
    genuine: list[float] = []
    impostor: list[float] = []
    for raw_score, raw_label in zip(scores, labels):
        score = float(raw_score)
        label = int(raw_label)
        if label == 1:
            genuine.append(score)
        elif label == 0:
            impostor.append(score)
        else:
            raise ValueError(f'labels must be 0 or 1, got {raw_label!r}')

    gen_stats = _group_stats(genuine)
    imp_stats = _group_stats(impostor)
    if gen_stats['n'] == 0 or imp_stats['n'] == 0:
        d_prime = None
    else:
        denom = math.sqrt(0.5 * (gen_stats['std'] ** 2 + imp_stats['std'] ** 2) + 1e-8)
        d_prime = float((gen_stats['mean'] - imp_stats['mean']) / denom)
    return {'genuine': gen_stats, 'impostor': imp_stats, 'd_prime': d_prime}


def _extract_last_result(row: dict) -> dict:
    trace = row.get('trace') or []
    if not trace:
        return {}
    return trace[-1].get('result') or {}


def summarize_verify_group(rows: list[dict]) -> dict:
    """Compute EER, minDCF, and score distributions over verify rows with missing-score imputation."""
    norm_scores: list[float] = []
    raw_scores: list[float] = []
    labels: list[int] = []
    valid_norm_scores: list[float] = []
    valid_labels: list[int] = []
    valid_trials = 0
    imputed_missing_trials = 0

    for row in rows:
        expect = row.get('expect')
        if expect not in ('accept', 'reject'):
            continue
        label = 1 if expect == 'accept' else 0
        result = _extract_last_result(row)
        score = result.get('score')
        raw_score = result.get('raw_score')

        if score is None:
            imputed_missing_trials += 1
            norm_scores.append(MISSING_NORM_SCORE)
            raw_scores.append(MISSING_RAW_SCORE)
        else:
            valid_trials += 1
            val = float(score)
            norm_scores.append(val)
            valid_norm_scores.append(val)
            valid_labels.append(label)
            raw_scores.append(float(raw_score) if raw_score is not None else MISSING_RAW_SCORE)
        labels.append(label)

    eer, eer_threshold = compute_eer(norm_scores, labels)
    raw_eer, _ = compute_eer(valid_norm_scores, valid_labels)
    mindcf = compute_mindcf(norm_scores, labels, **MINDCF_PARAMS)

    return {
        'n': len(labels),
        'valid_trials': valid_trials,
        'imputed_missing_trials': imputed_missing_trials,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'raw_eer': raw_eer,
        'min_dcf_norm': mindcf['min_dcf_norm'],
        'min_dcf_raw': mindcf['min_dcf_raw'],
        'mindcf_params': mindcf['params'],
        'score_stats': score_distribution(norm_scores, labels),
        'raw_score_stats': score_distribution(raw_scores, labels),
    }


def evaluate_task(rows: list[dict]) -> dict:
    """Evaluate Dimension 1 task metrics across per-suite aggregates, verify, and verify_all."""
    seen_suites = list(STANDARD_SUITES)
    for row in rows:
        suite = row.get('suite')
        if suite and suite not in seen_suites:
            seen_suites.append(suite)

    by_suite = {
        suite: aggregate([row for row in rows if row.get('suite') == suite])
        for suite in seen_suites
    }
    verify_rows = [row for row in rows if row.get('suite') == 'verify']
    verify_all_rows = [row for row in rows if row.get('suite') in ('verify', 'verify_impostor')]
    verify_metrics = summarize_verify_group(verify_rows)
    verify_all_metrics = summarize_verify_group(verify_all_rows)
    eer_primary = verify_all_metrics['eer'] if verify_all_metrics['eer'] is not None else verify_metrics['eer']

    return {
        'by_suite': by_suite,
        'verify_metrics': verify_metrics,
        'verify_all_metrics': verify_all_metrics,
        'eer_primary': eer_primary,
    }
