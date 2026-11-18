"""Multi-dimensional offline evaluation modules for voiceprint-agent."""
from pathlib import Path

from voice_eval.dims._sandbox import generate_speaker_wav
from voice_eval.dims.flow import evaluate_flow
from voice_eval.dims.perf import summarize_perf
from voice_eval.dims.redteam import evaluate_redteam
from voice_eval.dims.response import evaluate_response
from voice_eval.dims.robustness import evaluate_robustness
from voice_eval.dims.task import evaluate_task
from voice_eval.dims.tooling import evaluate_tooling

__all__ = ['evaluate_all_dimensions']


def _fast_flow_separator(_src: str, dest_dir: Path) -> dict:
    """Deterministic two-speaker splitter used when heavy Conv-TasNet flow execution is skipped."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    p1 = generate_speaker_wav(dest_dir / 'det_spk1.wav', speaker_index=0, duration=2.0, phase=2.0, seed=11)
    p2 = generate_speaker_wav(dest_dir / 'det_spk2.wav', speaker_index=1, duration=2.0, phase=2.0, seed=12)
    return {'sample_rate': 8000, 'paths': [str(p1), str(p2)], 'checkpoint': 'deterministic'}


def evaluate_all_dimensions(
    rows: list[dict],
    *,
    security_rows: list[dict] | None,
    real_non_detail_checks: list[dict] | None,
    perf_data: dict,
    skip_flow_separate: bool = False,
) -> dict:
    """Compute all 7 evaluation dimensions from executed rows and runtime telemetry."""
    task_res = evaluate_task(rows)
    tooling_res = evaluate_tooling()
    response_res = evaluate_response(real_non_detail_checks=real_non_detail_checks)
    flow_res = evaluate_flow(separator=_fast_flow_separator if skip_flow_separate else None)

    if 'case' in perf_data and 'cache' in perf_data and 'embed_compute' in perf_data:
        perf_res = dict(perf_data)
    else:
        perf_res = summarize_perf(
            case_timings_ms=list(perf_data.get('case_timings_ms') or []),
            step_timings_ms=dict(perf_data.get('step_timings_ms') or {}),
            cache_stats=perf_data.get('cache_stats'),
            peak_tracemalloc_mb=float(perf_data.get('peak_tracemalloc_mb') or 0.0),
            rss_mb=perf_data.get('rss_mb'),
        )

    redteam_res = evaluate_redteam(security_rows=security_rows)
    baseline_verify_acc = task_res['by_suite'].get('verify', {}).get('accuracy')
    robustness_res = evaluate_robustness(rows, baseline_verify_accuracy=baseline_verify_acc)

    return {
        'task': task_res,
        'tooling': tooling_res,
        'response': response_res,
        'flow': flow_res,
        'perf': perf_res,
        'redteam': redteam_res,
        'robustness': robustness_res,
    }
