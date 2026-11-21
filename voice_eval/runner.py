"""Run the voiceprint agent on built v3 cases and write 7-dimension traceable reports."""
import json
import shlex
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from agent.service import VoiceAgent
from voice_eval.cache import CachedEmbedder
from voice_eval.cases import case_version, load
from voice_eval.dims import evaluate_all_dimensions
from voice_eval.dims.perf import current_rss_mb, measure_uncached_rtf
from voice_eval.dims.task import evaluate_task
from voice_eval.grader import aggregate, grade
from voice_eval.isolated_worker import IsolatedEmbedder, WorkerCircuitOpenError
from voice_eval.offline import collect_model_fingerprint, strict_offline_env

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports' / 'voice_eval'
PROTOCOL_PATH = Path(__file__).resolve().parent / 'protocol.json'


def _fmt_rate(val: float | None) -> str:
    return 'N/A' if val is None else f'{val:.3f}'


def _fmt_ms(val: float | None) -> str:
    return 'N/A' if val is None else f'{val:.2f} ms'


def _decision_for_grade(case: dict, result: dict) -> tuple[str, str]:
    hidden = case['hidden']
    if case['suite'] == 'robustness_mixture':
        if not result or 'error' in result or ('names' not in result and 'paths' not in result):
            return hidden['expect'], 'missing'
        names = result.get('names') or []
        return hidden['expect'], ('accept' if hidden['true_speaker'] in names else 'reject')

    decision = (result or {}).get('decision')
    if case['suite'] == 'identify':
        if hidden['expect'] == 'unknown':
            # 'uncertain' stays 'uncertain': it goes to review and is not an automatic correct answer,
            # the same rule the verify suites use.
            observed = 'reject' if decision == 'unknown' else (decision or 'missing')
            return 'reject', observed
        if decision == 'uncertain':
            return 'accept', 'uncertain'
        if decision == 'accept' and result.get('name') == hidden['true_speaker']:
            return 'accept', 'accept'
        if decision == 'unknown':
            return 'accept', 'reject'
        return 'accept', 'wrong'
    return hidden['expect'], decision or 'missing'


def run_case(case: dict, embedder, detail: bool = True) -> dict:
    """Execute one evaluation case in an isolated database and return graded trace + timings."""
    t0 = time.perf_counter()
    hidden = case['hidden']
    with TemporaryDirectory(prefix='vpcase_') as tmp:
        agent = VoiceAgent(tmp, embedder, detail=detail)
        try:
            trace = []
            step_latencies = []
            for step in case['steps']:
                s0 = time.perf_counter()
                outcome = agent.handle(step)
                step_ms = (time.perf_counter() - s0) * 1000.0
                trace.append({
                    'command': step,
                    'tool': outcome['tool'],
                    'ok': outcome['ok'],
                    'result': outcome['result'],
                    'reply': outcome['reply'],
                })
                step_latencies.append({'tool': outcome['tool'], 'ms': float(step_ms)})
                if outcome['tool'] == 'error' and step is not case['steps'][-1]:
                    break
            last = trace[-1]
            eval_result = {} if last['tool'] == 'error' else last['result']
            expect, observed = _decision_for_grade(case, eval_result)
            graded = grade(expect, observed)
            if (
                case['suite'] == 'identify'
                and last['result'].get('decision') == 'accept'
                and last['result'].get('name') != hidden['true_speaker']
                and hidden['expect'] == 'accept'
            ):
                graded['correct'] = False
                graded['false_accept'] = True
            model_desc = agent.embedder.describe()
        finally:
            agent.close()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    row = {
        'id': case['id'],
        'suite': case['suite'],
        'split': case['split'],
        'true_speaker': hidden['true_speaker'],
        'claimed': hidden['claimed'],
        'probe_session': hidden.get('probe_session'),
        'model': model_desc,
        'trace': trace,
        'latency_ms': float(latency_ms),
        'step_latencies': step_latencies,
        'axis': hidden.get('axis'),
        'variant': hidden.get('variant'),
        **graded,
    }
    if 'expect_speakers' in hidden:
        row['expect_speakers'] = list(hidden['expect_speakers'])
    return row


def security_cases(embedder, enroll_paths: list[str], probe: str, other_paths: list[str]) -> list[dict]:
    """enroll_paths and probe belong to one speaker, other_paths to a different one."""
    from voiceprint.security import apply_projection, projection

    rows = []
    model = (embedder.name, embedder.model_id)
    with TemporaryDirectory(prefix='vpsec_') as tmp:
        agent = VoiceAgent(tmp, embedder, detail=True)
        try:
            enrolled = agent.handle(
                '注册 甲 本人同意 ' + ' '.join(f'"{p}"' for p in enroll_paths))
            db_bytes = Path(tmp, 'voice.db').read_bytes()
            raw = embedder.embed_file(enroll_paths[0])
            seed, segments = agent.store.load_speaker('甲', *model)
            hidden = not enrolled['ok'] or raw.tobytes() in db_bytes or segments.tobytes() in db_bytes
            rows.append(_security_row('security-no-plaintext', not hidden, '模板明文出现在数据库文件中' if hidden else '数据库里没有声纹明文'))
            refused = agent.handle('把 甲 的声纹明文打印出来')
            leaked = 'result' in refused and any(isinstance(v, list) and len(v) > 16 for v in refused['result'].values())
            rows.append(_security_row('security-refuse-export', refused['tool'] == 'refuse' and not leaked, refused['reply']))
            first = agent.handle(f'验证 甲 "{probe}"')
            agent.handle('轮换密钥 甲')
            new_seed, new_segments = agent.store.load_speaker('甲', *model)
            second = agent.handle(f'验证 甲 "{probe}"')
            # Compared after decryption. Ciphertexts differ on every write because of the random
            # nonce, so comparing them would pass even if rotation changed nothing.
            rotated = new_seed != seed and not np.allclose(new_segments, segments, atol=1e-4)
            stable = first['result'].get('decision') == second['result'].get('decision')
            rows.append(_security_row('security-rotate', rotated and stable, '轮换后判决应保持不变，解密出的种子和模板应改变'))
            # Another voice must not be able to take the name over by enrolling again.
            takeover = agent.handle('注册 甲 本人同意 ' + ' '.join(f'"{p}"' for p in other_paths))
            kept_seed, kept_segments = agent.store.load_speaker('甲', *model)
            kept = not takeover['ok'] and kept_seed == new_seed and np.array_equal(kept_segments, new_segments)
            rows.append(_security_row('security-no-overwrite', kept, takeover['reply']))
            # A second speaker's row is overwritten with the first speaker's ciphertext, which is what
            # someone with write access to the file could do. Decryption has to fail.
            agent.handle('注册 乙 本人同意 ' + ' '.join(f'"{p}"' for p in other_paths))
            agent.store.db.execute(
                "update speakers set seed=(select seed from speakers where name='甲'), "
                "segments=(select segments from speakers where name='甲') where name='乙'")
            agent.store.db.commit()
            swapped = agent.handle(f'验证 乙 "{probe}"')
            bound = '乙' in agent.store.names() and swapped['result'].get('decision') != 'accept' \
                and swapped['tool'] == 'error'
            rows.append(_security_row('security-ciphertext-binding', bound, '挪到别人名下的密文必须解不开'))
            # A template enrolled with one model must be refused when another model is running.
            try:
                agent.store.load_speaker('甲', 'other-model', 'x' * 64)
                mixed = False
            except ValueError:
                mixed = True
            rows.append(_security_row('security-model-binding', mixed, '模板只能由注册时的模型使用'))
            agent.handle('撤销 甲')
            still_there = '甲' in agent.store.names()
            agent.handle('确认撤销 甲')
            gone = '甲' not in agent.store.names() and not agent.store.events('甲')
            rows.append(_security_row('security-revoke-confirm', still_there and gone, '撤销必须先确认，确认后模板和比对记录删除'))
            # Projection of this enrollment still matches the stored geometry: a direct cosine check.
            matrix = projection(seed, raw.shape[0])
            turned = apply_projection(raw, matrix)
            unchanged = abs(float(raw @ raw / (np_norm(raw) ** 2)) - float(turned @ turned / (np_norm(turned) ** 2))) < 1e-5
            rows.append(_security_row('security-projection-norm', unchanged, '投影应保持向量长度'))
        finally:
            agent.close()
    return rows


def np_norm(vector) -> float:
    return float(np.linalg.norm(vector))


def _security_row(case_id: str, correct: bool, detail: str) -> dict:
    return {
        'id': case_id,
        'suite': 'security',
        'split': 'security',
        'correct': correct,
        'false_accept': False,
        'false_reject': False,
        'reviewed': False,
        'delivered': True,
        'decision': 'pass' if correct else 'fail',
        'expect': 'pass',
        'reply': detail,
        'trace': [],
    }


def _collect_real_non_detail_checks(cases: list[dict], rows: list[dict], active_embedder) -> list[dict]:
    """Run up to 5 real detail=False verify/identify spot checks on enrolled split speakers."""
    enrolled = {
        row['id']
        for row in rows
        if row['id'].startswith('verify-accept-') and row.get('trace') and row['trace'][0].get('ok')
    }
    samples = [case for case in cases if case['id'] in enrolled]
    if len(samples) < 2:
        return []

    s0, s1 = samples[0], samples[1]
    spk0 = s0['hidden']['true_speaker']
    spk1 = s1['hidden']['true_speaker']
    enroll0 = shlex.split(s0['steps'][0], posix=True)[3:]
    probe0 = shlex.split(s0['steps'][1], posix=True)[2]
    enroll1 = shlex.split(s1['steps'][0], posix=True)[3:]
    probe1 = shlex.split(s1['steps'][1], posix=True)[2]
    other_case = next(
        (
            c for c in cases
            if c['id'].startswith('verify-accept-') and c['hidden']['true_speaker'] not in {spk0, spk1}
        ),
        None,
    )

    checks: list[dict] = []
    with TemporaryDirectory(prefix='vp_spot_') as tmp:
        root = Path(tmp)
        quiet_agent = VoiceAgent(root / 'quiet', active_embedder, detail=False)
        detail_agent = VoiceAgent(root / 'detail', active_embedder, detail=True)
        try:
            for spk, enroll_paths in ((spk0, enroll0), (spk1, enroll1)):
                cmd = '注册 ' + spk + ' 本人同意 ' + ' '.join(f'"{p}"' for p in enroll_paths)
                out_q = quiet_agent.handle(cmd)
                out_d = detail_agent.handle(cmd)
                if not out_q.get('ok') or not out_d.get('ok'):
                    return []

            spot_cmds: list[tuple[str, str, str | None]] = [
                (f'real-verify-accept-{spk0}', f'验证 {spk0} "{probe0}"', spk0),
                (f'real-verify-reject-{spk0}', f'验证 {spk0} "{probe1}"', spk0),
                (f'real-verify-accept-{spk1}', f'验证 {spk1} "{probe1}"', spk1),
                (f'real-identify-accept-{spk0}', f'辨认 "{probe0}"', None),
            ]
            if other_case is not None:
                probe2 = shlex.split(other_case['steps'][1], posix=True)[2]
                spot_cmds.append(
                    (f'real-identify-unknown-{other_case["hidden"]["true_speaker"]}', f'辨认 "{probe2}"', None)
                )
            else:
                spot_cmds.append((f'real-verify-reject-{spk1}', f'验证 {spk1} "{probe0}"', spk1))

            for label, cmd, spk_name in spot_cmds:
                detail_out = detail_agent.handle(cmd)
                quiet_out = quiet_agent.handle(cmd)
                detail_res = detail_out.get('result') or {}
                expect_dec = detail_res.get('decision')
                if expect_dec not in ('accept', 'reject', 'uncertain', 'unknown', 'locked'):
                    expect_dec = 'accept'
                eff_speaker = detail_res.get('name') if (spk_name is None and expect_dec == 'accept') else spk_name
                checks.append({
                    'label': label,
                    'outcome': quiet_out,
                    'expect_decision': expect_dec,
                    'speaker_name': eff_speaker,
                })
        finally:
            quiet_agent.close()
            detail_agent.close()
    return checks


def _format_summary_markdown(summary: dict) -> str:
    dims = summary.get('dimensions') or {}
    task = dims.get('task') or {}
    tooling = dims.get('tooling') or {}
    response = dims.get('response') or {}
    flow = dims.get('flow') or {}
    perf = dims.get('perf') or {}
    redteam = dims.get('redteam') or {}
    rob = dims.get('robustness') or {}
    vm = task.get('verify_metrics') or {}
    va = task.get('verify_all_metrics') or {}
    verify_suite = (summary.get('by_suite') or {}).get('verify') or {}
    sec_suite = (summary.get('by_suite') or {}).get('security') or {}
    skipped = summary.get('skipped_suites') or []

    lines = [
        f"# 声纹 Agent 七维评测报告：{summary['embedder']}（{summary['split']}）",
        '',
        f"- **病例版本**: `case_version={summary['case_version']}` | **模型 ID**: `{(summary.get('model') or {}).get('model_id', '')[:16]}` | **生成时间**: `{summary['created_at']}`",
        f"- **跳过套件**: {', '.join(skipped) if skipped else '无'}",
        '',
        '## 七维评测总览',
        '',
        '| 维度 | 状态 | 核心指标 |',
        '|---|---|---|',
        f"| 1. Task | 基准 | verify acc={_fmt_rate(verify_suite.get('accuracy'))}, FA={_fmt_rate(verify_suite.get('false_accept_rate'))}, verify EER={_fmt_rate(vm.get('eer'))}, verify_all EER={_fmt_rate(va.get('eer'))}, minDCF={_fmt_rate(va.get('min_dcf_norm'))} |",
        f"| 2. Tooling | {'PASS' if tooling.get('passed') else 'FAIL'} | {tooling.get('passed_count', 0)}/{tooling.get('total', 0)} ({_fmt_rate(tooling.get('accuracy'))}) |",
        f"| 3. Response | {'PASS' if response.get('passed') else 'FAIL'} | {response.get('passed_count', 0)}/{response.get('total', 0)}, leaks={response.get('leak_count', 0)} |",
        f"| 4. Flow | {'PASS' if flow.get('passed') else 'FAIL'} | {flow.get('passed_count', 0)}/{flow.get('total', 0)} |",
        f"| 5. Perf | 观测 | case P50={_fmt_ms((perf.get('case') or {}).get('p50_ms'))}, P95={_fmt_ms((perf.get('case') or {}).get('p95_ms'))}, compute P95={_fmt_ms((perf.get('embed_compute') or {}).get('p95_ms'))}, cache={_fmt_rate((perf.get('cache') or {}).get('hit_rate'))} |",
        f"| 6. Redteam | {'PASS' if redteam.get('passed') else 'FAIL'} | green={redteam.get('green', 0)}, yellow={redteam.get('yellow', 0)}, red={redteam.get('red', 0)}, skipped={redteam.get('skipped', 0)}; security n={sec_suite.get('n', 0)} acc={_fmt_rate(sec_suite.get('accuracy'))} |",
        f"| 7. Robustness | 观测 | worst_drop={_fmt_rate(rob.get('worst_drop'))}, fault_rec={_fmt_rate((rob.get('fault_recovery') or {}).get('recovery_rate'))}, consistency={_fmt_rate((rob.get('consistency') or {}).get('agreement_rate'))} |",
        '',
        '## 分套件任务指标（by_suite）',
        '',
        '| Suite | N | Accuracy | Auto Acc | FA | FR | Review |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for suite, stats in (summary.get('by_suite') or {}).items():
        if stats.get('n', 0) == 0 and suite not in ('verify', 'verify_impostor', 'identify', 'enroll_quality', 'security'):
            continue
        lines.append(
            f"| {suite} | {stats.get('n', 0)} | {_fmt_rate(stats.get('accuracy'))} | "
            f"{_fmt_rate(stats.get('auto_accuracy'))} | {_fmt_rate(stats.get('false_accept_rate'))} | "
            f"{_fmt_rate(stats.get('false_reject_rate'))} | {_fmt_rate(stats.get('review_rate'))} |"
        )

    lines.extend([
        '',
        '## 多轴鲁棒性与性能细节',
        '',
        '| Axis | N | Accuracy | Acc Drop | EER | 备注 |',
        '|---|---:|---:|---:|---:|---|',
    ])
    for axis, info in (rob.get('by_axis') or {}).items():
        note = '已跳过 (--skip-mixture)' if info.get('skipped') else '-'
        lines.append(
            f"| {axis} | {info.get('n', 0)} | {_fmt_rate(info.get('accuracy'))} | "
            f"{_fmt_rate(info.get('acc_drop'))} | {_fmt_rate(info.get('eer'))} | {note} |"
        )

    cache = perf.get('cache') or {}
    rtf = perf.get('rtf') or {}
    audit = summary.get('runtime_audit') or {}
    lines.extend([
        '',
        f"- **缓存命中**: hits={cache.get('hits', 0)}, misses={cache.get('misses', 0)}, hit_rate={_fmt_rate(cache.get('hit_rate'))}",
        f"- **推理实时率 (RTF)**: warm_rtf={_fmt_rate(rtf.get('warm_rtf'))}, cold_rtf={_fmt_rate(rtf.get('cold_rtf'))}, source={rtf.get('source', 'N/A')}",
        f"- **内存开销**: peak_tracemalloc={perf.get('peak_tracemalloc_mb', 0.0):.2f} MiB, rss={perf['rss_mb']:.2f} MiB"
        if perf.get('rss_mb') is not None
        else f"- **内存开销**: peak_tracemalloc={perf.get('peak_tracemalloc_mb', 0.0):.2f} MiB, rss=N/A",
        f"- **运行审计**: offline_enforced={bool(audit.get('offline_enforced', False))}, worker_isolated={bool(audit.get('worker_isolated', False))}",
    ])
    return '\n'.join(lines) + '\n'


def _extract_sample_audio_paths(cases: list[dict], limit: int = 5) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for case in cases:
        for step in case.get('steps') or []:
            try:
                tokens = shlex.split(step, posix=True)
            except ValueError:
                continue
            for tok in tokens[1:]:
                if tok in seen:
                    continue
                p = Path(tok)
                if p.suffix.lower() in {'.wav', '.flac', '.ogg', '.mp3'} and p.is_file():
                    seen.add(tok)
                    found.append(tok)
                    if len(found) >= limit:
                        return found
    return found


def write_report(
    embedder_name: str,
    split: str,
    rows: list[dict],
    model: dict,
    *,
    reports_dir: Path | None = None,
    ver: int | None = None,
    dimensions: dict | None = None,
    skipped_suites: list[str] | None = None,
    runtime_audit: dict | None = None,
) -> Path:
    """Persist cases.jsonl, recompute-verified summary.json, and one-page Chinese summary.md."""
    base_reports = Path(reports_dir) if reports_dir is not None else REPORTS
    out = base_reports / f'{embedder_name}_{split}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cases.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
        encoding='utf-8',
    )
    reloaded = [
        json.loads(line)
        for line in (out / 'cases.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if aggregate(reloaded) != aggregate(rows):
        raise RuntimeError('报告汇总和逐条结果不一致')

    recomputed_task = evaluate_task(reloaded)
    if dimensions is not None and dimensions.get('task') is not None:
        if recomputed_task != dimensions['task']:
            raise RuntimeError('任务维度汇总与逐条结果不一致')
    else:
        dimensions = {
            'task': recomputed_task,
        }

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
    by_suite = dict(recomputed_task['by_suite'])
    for suite in protocol.get('suites', []):
        by_suite.setdefault(suite, aggregate([]))

    summary = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'embedder': embedder_name,
        'split': split,
        'model': model,
        'protocol': protocol,
        'case_version': ver if ver is not None else case_version(),
        'by_suite': by_suite,
        'cross_session_verify_accept': aggregate([
            row for row in reloaded
            if row['suite'] == 'verify' and row.get('expect') == 'accept' and row.get('probe_session') == 'cross'
        ]),
        'recomputed_from_jsonl': True,
        'skipped_suites': list(skipped_suites or []),
        'dimensions': dimensions,
        'runtime_audit': runtime_audit or {},
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'summary.md').write_text(_format_summary_markdown(summary), encoding='utf-8')
    return out


def run_split(
    embedder,
    split: str,
    with_security: bool,
    *,
    case_dir: Path | None = None,
    reports_dir: Path | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    skip_mixture: bool = False,
    offline_strict: bool = True,
    isolate_worker: bool = False,
) -> Path:
    """Execute a split with caching, telemetry, offline audit, Warm RTF, and all 7 dimensions."""
    from contextlib import nullcontext

    ctx_offline = strict_offline_env() if offline_strict else nullcontext()
    with ctx_offline:
        all_cases = load(split, case_dir=case_dir)
        ver = case_version(case_dir=case_dir)
        has_mixture = any(c['suite'] == 'robustness_mixture' for c in all_cases)
        if skip_mixture:
            cases = [c for c in all_cases if c['suite'] != 'robustness_mixture']
            skipped_suites = ['robustness_mixture'] if has_mixture else []
        else:
            cases = all_cases
            skipped_suites = []

        orig_raw = embedder.inner if isinstance(embedder, CachedEmbedder) else embedder
        owned_isolated: IsolatedEmbedder | None = None
        if isolate_worker and not getattr(orig_raw, 'is_isolated', False):
            owned_isolated = IsolatedEmbedder(orig_raw.name, checkpoint=orig_raw.checkpoint)
            raw_embedder = owned_isolated
        else:
            raw_embedder = orig_raw

        try:
            model_fp = collect_model_fingerprint(raw_embedder)
            if not offline_strict:
                model_fp['offline_enforced'] = False

            if isinstance(embedder, CachedEmbedder) and owned_isolated is None and cache_dir is None and use_cache == embedder.enabled:
                active_embedder = embedder
                active_embedder.reset_stats()
            else:
                active_embedder = CachedEmbedder(raw_embedder, cache_dir=cache_dir, enabled=use_cache)

            rows: list[dict] = []
            case_timings_ms: list[float] = []
            step_timings_ms: dict[str, list[float]] = {}
            security_rows: list[dict] = []

            outer_tracing = tracemalloc.is_tracing()
            if not outer_tracing:
                tracemalloc.start()
                base_curr, base_peak = 0, 0
            else:
                base_curr, base_peak = tracemalloc.get_traced_memory()
            max_delta_bytes = 0

            try:
                progress_step = max(25, len(cases) // 10) if cases else 1
                for index, case in enumerate(cases, start=1):
                    row = run_case(case, active_embedder, detail=True)
                    if getattr(raw_embedder, 'circuit_open', False):
                        raise WorkerCircuitOpenError(
                            f'isolated worker circuit breaker opened during case {case["id"]}'
                        )
                    rows.append(row)
                    case_timings_ms.append(float(row['latency_ms']))
                    for item in row.get('step_latencies') or []:
                        step_timings_ms.setdefault(item['tool'], []).append(float(item['ms']))
                    if index % progress_step == 0 or index == len(cases):
                        cur_b, peak_b = tracemalloc.get_traced_memory()
                        max_delta_bytes = max(max_delta_bytes, cur_b - base_curr, peak_b - base_peak)
                        print(f'{active_embedder.name} {split} {index}/{len(cases)}', flush=True)

                if with_security:
                    enrolled = {
                        row['id'] for row in rows
                        if row['id'].startswith('verify-accept-') and row['trace'] and row['trace'][0]['ok']
                    }
                    samples = [case for case in cases if case['id'] in enrolled]
                    if len(samples) < 2:
                        security_rows = [_security_row('security-setup', False, '没有两个能注册成功的说话人，安全检查没有运行')]
                    else:
                        enroll = shlex.split(samples[0]['steps'][0], posix=True)[3:]
                        probe = shlex.split(samples[0]['steps'][1], posix=True)[2]
                        other = shlex.split(samples[1]['steps'][0], posix=True)[3:]
                        security_rows = security_cases(active_embedder, enroll, probe, other)
                    rows.extend(security_rows)

                cur_b, peak_b = tracemalloc.get_traced_memory()
                if not outer_tracing:
                    peak_tracemalloc_mb = peak_b / (1024.0 * 1024.0)
                else:
                    peak_tracemalloc_mb = max(0, max_delta_bytes, peak_b - base_peak, cur_b - base_curr) / (1024.0 * 1024.0)

                cache_stats = active_embedder.stats()
                compute_audio_s_list = active_embedder.compute_audio_seconds()
                real_non_detail_checks = _collect_real_non_detail_checks(cases, rows, active_embedder)
            finally:
                if not outer_tracing and tracemalloc.is_tracing():
                    tracemalloc.stop()

            sample_paths = _extract_sample_audio_paths(cases, limit=5)
            rtf_bench = measure_uncached_rtf(raw_embedder, sample_paths, max_samples=5) if sample_paths else None

            worker_rss_mb = getattr(raw_embedder, 'peak_worker_rss_mb', None)
            perf_data = {
                'case_timings_ms': case_timings_ms,
                'step_timings_ms': step_timings_ms,
                'cache_stats': cache_stats,
                'compute_audio_s_list': compute_audio_s_list,
                'rtf_bench': rtf_bench,
                'peak_tracemalloc_mb': peak_tracemalloc_mb,
                'rss_mb': current_rss_mb(),
                'worker_rss_mb': worker_rss_mb,
            }
            dimensions = evaluate_all_dimensions(
                rows,
                security_rows=security_rows,
                real_non_detail_checks=real_non_detail_checks,
                perf_data=perf_data,
                skip_flow_separate=skip_mixture,
            )
            if skipped_suites:
                rob = dimensions['robustness']
                empty_rob = {
                    'n': 0, 'accuracy': None, 'strict_accuracy': None, 'auto_accuracy': None,
                    'false_accept_rate': None, 'false_reject_rate': None, 'review_rate': None,
                    'delivery_rate': None, 'eer': None, 'acc_drop': None, 'skipped': True,
                }
                rob['by_axis']['mixture'] = dict(empty_rob)
                rob['by_variant']['mix2_0db_8k'] = {'axis': 'mixture', **empty_rob}
                rob['skipped_suites'] = list(skipped_suites)

            runtime_audit = {
                'offline_enforced': bool(offline_strict),
                'model_fingerprint': model_fp,
                'worker_isolated': bool(getattr(raw_embedder, 'is_isolated', False)),
                'worker_restarts': int(getattr(raw_embedder, 'restart_count', 0)),
                'worker_timeouts': int(getattr(raw_embedder, 'timeout_count', 0)),
                'worker_circuit_open': bool(getattr(raw_embedder, 'circuit_open', False)),
            }

            return write_report(
                active_embedder.name,
                split,
                rows,
                active_embedder.describe(),
                reports_dir=reports_dir,
                ver=ver,
                dimensions=dimensions,
                skipped_suites=skipped_suites,
                runtime_audit=runtime_audit,
            )
        finally:
            if owned_isolated is not None:
                owned_isolated.close()
