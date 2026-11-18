"""Universal 3-tier adoption gate and one-page Chinese comparison report generator."""
import json
from pathlib import Path

from voice_eval.dims.task import evaluate_task
from voice_eval.grader import aggregate, paired_bootstrap
from voice_eval.runner import REPORTS

REQUIRED_CASE_VERSION = 3


def resolve_report_dir(
    ref: str | Path,
    split: str | None = None,
    reports_root: Path | None = None,
) -> Path:
    """Resolve an explicit directory path, report subdir name, or embedder name (+ split)."""
    cand = Path(ref)
    if cand.is_dir() and ((cand / 'summary.json').is_file() or (cand / 'cases.jsonl').is_file()):
        return cand
    if cand.is_file() and cand.name in ('summary.json', 'cases.jsonl'):
        return cand.parent

    ref_str = str(ref)
    root = Path(reports_root) if reports_root is not None else REPORTS
    direct = root / ref_str
    if direct.is_dir() and ((direct / 'summary.json').is_file() or (direct / 'cases.jsonl').is_file()):
        return direct
    if split:
        by_split = root / f'{ref_str}_{split}'
        if by_split.is_dir() and ((by_split / 'summary.json').is_file() or (by_split / 'cases.jsonl').is_file()):
            return by_split

    raise FileNotFoundError(f'找不到评测报告目录：{ref!r}（split={split!r}, root={root}）')


def _load_report(report_dir: Path) -> tuple[dict, list[dict]]:
    summary_path = report_dir / 'summary.json'
    cases_path = report_dir / 'cases.jsonl'
    if not summary_path.is_file():
        raise FileNotFoundError(f'报告缺少 summary.json：{summary_path}')
    if not cases_path.is_file():
        raise FileNotFoundError(f'报告缺少 cases.jsonl：{cases_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in cases_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return summary, rows


def _report_tag(report_dir: Path, summary: dict, split: str) -> str:
    name = report_dir.name
    suffix = f'_{split}'
    if name.endswith(suffix) and len(name) > len(suffix):
        return name[:-len(suffix)]
    return str(summary.get('embedder') or name)


def _pick_eer_and_dcf(task_dim: dict) -> tuple[float | None, float | None]:
    va = task_dim.get('verify_all_metrics') or {}
    vm = task_dim.get('verify_metrics') or {}
    eer = va.get('eer') if va.get('eer') is not None else vm.get('eer')
    dcf = va.get('min_dcf_norm') if va.get('min_dcf_norm') is not None else vm.get('min_dcf_norm')
    return eer, dcf


def _evaluate_tier1_redlines(cand_summary: dict, cand_rows: list[dict], cand_task: dict, split: str) -> dict[str, bool]:
    cand_dims = cand_summary.get('dimensions') or {}
    sec_suite = (cand_summary.get('by_suite') or {}).get('security') or (cand_task.get('by_suite') or {}).get('security') or {}
    sec_rows = [r for r in cand_rows if r.get('suite') == 'security']
    sec_n = max(len(sec_rows), int(sec_suite.get('n') or 0))
    if split == 'validation' or sec_n > 0:
        rows_ok = all(bool(r.get('correct')) for r in sec_rows) if sec_rows else True
        suite_ok = (sec_suite.get('accuracy') == 1.0) if int(sec_suite.get('n') or 0) > 0 else len(sec_rows) > 0
        security_ok = sec_n > 0 and rows_ok and suite_ok
    else:
        security_ok = True

    redteam = cand_dims.get('redteam') or {}
    redteam_ok = (
        bool(redteam.get('passed'))
        and int(redteam.get('yellow', 0)) == 0
        and int(redteam.get('red', 0)) == 0
    )

    tooling = cand_dims.get('tooling') or {}
    tooling_ok = bool(tooling.get('passed')) and (tooling.get('accuracy') == 1.0)

    response = cand_dims.get('response') or {}
    response_ok = bool(response.get('passed')) and int(response.get('leak_count', 0)) == 0

    flow = cand_dims.get('flow') or {}
    flow_ok = bool(flow.get('passed'))

    return {
        'security': security_ok,
        'redteam': redteam_ok,
        'tooling': tooling_ok,
        'response': response_ok,
        'flow': flow_ok,
    }


def _evaluate_tier3_warnings(base_summary: dict, cand_summary: dict) -> tuple[list[str], float | None]:
    warnings: list[str] = []
    base_dims = base_summary.get('dimensions') or {}
    cand_dims = cand_summary.get('dimensions') or {}

    base_rob = base_dims.get('robustness') or {}
    cand_rob = cand_dims.get('robustness') or {}
    base_vars = base_rob.get('by_variant') or {}
    cand_vars = cand_rob.get('by_variant') or {}

    rel_drops: list[tuple[float, str]] = []
    for variant, c_info in cand_vars.items():
        if c_info.get('skipped') or not c_info.get('n') or c_info.get('accuracy') is None:
            continue
        b_info = base_vars.get(variant) or {}
        if b_info.get('skipped') or not b_info.get('n') or b_info.get('accuracy') is None:
            continue
        drop = float(b_info['accuracy']) - float(c_info['accuracy'])
        rel_drops.append((drop, variant))

    worst_drop = max((d for d, _v in rel_drops), default=None)
    bad_vars = [(d, v) for d, v in rel_drops if d > 0.10]
    if bad_vars:
        worst_d, worst_v = max(bad_vars)
        warnings.append(
            f'鲁棒性退化告警：{len(bad_vars)} 个失真档位相对基线准确率下降超过 0.10，最差为 {worst_v}（下降 {worst_d:.3f}）。'
        )

    base_perf = base_dims.get('perf') or {}
    cand_perf = cand_dims.get('perf') or {}
    for metric_key, label in (('case', '端到端用例'), ('embed_compute', '冷特征计算')):
        b_p95 = (base_perf.get(metric_key) or {}).get('p95_ms')
        c_p95 = (cand_perf.get(metric_key) or {}).get('p95_ms')
        if b_p95 is not None and c_p95 is not None and b_p95 > 0.0:
            delta = float(c_p95) - float(b_p95)
            ratio = float(c_p95) / float(b_p95)
            if ratio > 1.5 and delta > 5.0:
                warnings.append(
                    f'性能告警：{label} P95 延迟从 {b_p95:.2f} ms 升至 {c_p95:.2f} ms（+{delta:.2f} ms，{ratio:.2f}x）。'
                )

    for mem_key, label in (('peak_tracemalloc_mb', '峰值 Tracemalloc 内存'), ('rss_mb', '进程常驻内存 RSS')):
        b_mem = base_perf.get(mem_key)
        c_mem = cand_perf.get(mem_key)
        if b_mem is not None and c_mem is not None and float(b_mem) > 0.0 and float(c_mem) >= 2.0 * float(b_mem):
            warnings.append(
                f'内存告警：{label}从 {float(b_mem):.2f} MiB 升至 {float(c_mem):.2f} MiB（{float(c_mem) / float(b_mem):.2f}x）。'
            )

    fault_rec = cand_rob.get('fault_recovery') or {}
    rec_rate = fault_rec.get('recovery_rate')
    if rec_rate is not None and float(rec_rate) < 1.0:
        warnings.append(f'故障恢复率告警：故障注入后恢复率为 {float(rec_rate):.3f}（低于 1.000）。')

    consistency = cand_rob.get('consistency') or {}
    agree_rate = consistency.get('agreement_rate')
    if agree_rate is not None and float(agree_rate) < 1.0:
        warnings.append(f'重复一致性告警：多次重复运行判决一致率为 {float(agree_rate):.3f}（低于 1.000）。')
    elif consistency.get('passed') is False:
        spread = consistency.get('max_score_spread')
        spread_str = 'N/A' if spread is None else f'{float(spread):.6f}'
        warnings.append(f'重复一致性告警：多次重复运行分数存在漂移（max_score_spread={spread_str}）。')

    return warnings, worst_drop


def _reason_text(
    split: str,
    base_tag: str,
    cand_tag: str,
    paired: dict,
    fa_left: float | None,
    fa_right: float | None,
    base_eer: float | None,
    cand_eer: float | None,
    redline_checks: dict[str, bool],
    primary_passed: bool,
    adopt: bool,
    warnings: list[str],
) -> str:
    fa_l = 'N/A' if fa_left is None else f'{fa_left:.3f}'
    fa_r = 'N/A' if fa_right is None else f'{fa_right:.3f}'
    eer_l = 'N/A' if base_eer is None else f'{base_eer:.4f}'
    eer_r = 'N/A' if cand_eer is None else f'{cand_eer:.4f}'
    head = (
        f'{split} 上 {cand_tag} 比 {base_tag} 的验证准确率高 {paired["diff"]:.3f}，'
        f'95% 区间 [{paired["low"]:.3f}, {paired["high"]:.3f}]。'
        f'误接受率 {fa_l} -> {fa_r}，EER {eer_l} -> {eer_r}。'
    )
    if split != 'validation':
        return head + '这是锁定测试，只报告，不据此更换模型。'
    if not all(redline_checks.values()):
        failed = [k for k, ok in redline_checks.items() if not ok]
        return head + f'Tier 1 红线未通过（{", ".join(failed)}），禁止采用。'
    if adopt:
        if warnings:
            return head + f'Tier 1 与 Tier 2 门禁全部通过，可以采用（附 {len(warnings)} 条 Tier 3 观察告警）。'
        return head + '按预定三级门禁规则可以通过，建议采用。'
    if not primary_passed:
        return head + 'Tier 2 主指标门禁未通过（Bootstrap 下界、误接受护栏或 EER 退化约束不满足），保持基线不变。'
    return head + '未满足采用门禁，保持基线不变。'


def _fmt(val: float | None, digits: int = 3) -> str:
    return 'N/A' if val is None else f'{val:.{digits}f}'


def _format_compare_markdown(report: dict, base_summary: dict, cand_summary: dict) -> str:
    base_info = report['base']
    cand_info = report['cand']
    base_dims = base_summary.get('dimensions') or {}
    cand_dims = cand_summary.get('dimensions') or {}
    tier1 = report['gate']['tier1_redlines']
    tier2 = report['gate']['tier2_primary']
    warnings = report['warnings']

    verdict_icon = '✅ 建议采用' if report['adopt'] else '❌ 不采用'
    lines = [
        f"# 声纹模型对比门禁报告：{base_info['tag']} vs {cand_info['tag']}（{report['split']}）",
        '',
        f"> **最终结论：{verdict_icon}** — {report['reason']}",
        '',
        '## 三级门禁明细（3-Tier Gate）',
        '',
        '| 层级 | 检查项 | 结果 | 详情 |',
        '|---|---|---|---|',
        f"| Tier 1 红线 | security / redteam / tooling / response / flow | {'PASS' if report['redline_passed'] else 'FAIL'} | "
        + ', '.join(f"{k}={'✓' if v else '✗'}" for k, v in tier1.items()) + ' |',
        f"| Tier 2 主指标 | split={report['split']}, bootstrap.low > 0, FA<=+0.02, EER<=base+1e-6 | "
        f"{'PASS' if report['primary_passed'] else 'FAIL'} | "
        f"allowed={tier2['adoption_allowed']}, paired_low={report['paired']['low']:.3f}, "
        f"fa_delta={_fmt(report['fa_delta'])}, eer_delta={_fmt(report['eer_delta'], 4)} |",
        f"| Tier 3 观察 | robustness / perf / mem / fault / consistency | {len(warnings)} 条告警 | "
        f"worst_rob_drop_vs_base={_fmt(report['gate']['tier3_observational']['worst_robustness_drop_vs_base'])} |",
        '',
        '## 七维指标对照表',
        '',
        f"| 维度 | {base_info['tag']} (Base) | {cand_info['tag']} (Cand) |",
        '|---|---|---|',
        f"| Task (verify acc / FA / EER / minDCF) | "
        f"{_fmt(base_info['accuracy'])} / {_fmt(base_info['false_accept_rate'])} / {_fmt(base_info['eer'], 4)} / {_fmt(base_info['min_dcf_norm'], 4)} | "
        f"{_fmt(cand_info['accuracy'])} / {_fmt(cand_info['false_accept_rate'])} / {_fmt(cand_info['eer'], 4)} / {_fmt(cand_info['min_dcf_norm'], 4)} |",
        f"| Tooling | {_fmt((base_dims.get('tooling') or {}).get('accuracy'))} | {_fmt((cand_dims.get('tooling') or {}).get('accuracy'))} |",
        f"| Response | passed={bool((base_dims.get('response') or {}).get('passed'))}, leaks={(base_dims.get('response') or {}).get('leak_count', 0)} | passed={bool((cand_dims.get('response') or {}).get('passed'))}, leaks={(cand_dims.get('response') or {}).get('leak_count', 0)} |",
        f"| Flow | passed={bool((base_dims.get('flow') or {}).get('passed'))} ({(base_dims.get('flow') or {}).get('passed_count', 0)}/{(base_dims.get('flow') or {}).get('total', 0)}) | passed={bool((cand_dims.get('flow') or {}).get('passed'))} ({(cand_dims.get('flow') or {}).get('passed_count', 0)}/{(cand_dims.get('flow') or {}).get('total', 0)}) |",
        f"| Redteam | green={(base_dims.get('redteam') or {}).get('green', 0)}, yellow={(base_dims.get('redteam') or {}).get('yellow', 0)}, red={(base_dims.get('redteam') or {}).get('red', 0)} | green={(cand_dims.get('redteam') or {}).get('green', 0)}, yellow={(cand_dims.get('redteam') or {}).get('yellow', 0)}, red={(cand_dims.get('redteam') or {}).get('red', 0)} |",
        f"| Perf (case P50/P95 ms) | {_fmt(((base_dims.get('perf') or {}).get('case') or {}).get('p50_ms'), 2)} / {_fmt(((base_dims.get('perf') or {}).get('case') or {}).get('p95_ms'), 2)} | {_fmt(((cand_dims.get('perf') or {}).get('case') or {}).get('p50_ms'), 2)} / {_fmt(((cand_dims.get('perf') or {}).get('case') or {}).get('p95_ms'), 2)} |",
        f"| Robustness (worst_drop / fault / cons) | {_fmt((base_dims.get('robustness') or {}).get('worst_drop'))} / {_fmt(((base_dims.get('robustness') or {}).get('fault_recovery') or {}).get('recovery_rate'))} / {_fmt(((base_dims.get('robustness') or {}).get('consistency') or {}).get('agreement_rate'))} | {_fmt((cand_dims.get('robustness') or {}).get('worst_drop'))} / {_fmt(((cand_dims.get('robustness') or {}).get('fault_recovery') or {}).get('recovery_rate'))} / {_fmt(((cand_dims.get('robustness') or {}).get('consistency') or {}).get('agreement_rate'))} |",
        '',
        '## 多轴鲁棒性对比（by_axis）',
        '',
        f"| Axis | {base_info['tag']} Acc | {cand_info['tag']} Acc | Delta |",
        '|---|---:|---:|---:|',
    ]
    base_axes = (base_dims.get('robustness') or {}).get('by_axis') or {}
    cand_axes = (cand_dims.get('robustness') or {}).get('by_axis') or {}
    seen_axes = list(base_axes.keys()) + [ax for ax in cand_axes if ax not in base_axes]
    for ax in seen_axes:
        b_ax = base_axes.get(ax) or {}
        c_ax = cand_axes.get(ax) or {}
        b_acc = b_ax.get('accuracy')
        c_acc = c_ax.get('accuracy')
        delta = (c_acc - b_acc) if (b_acc is not None and c_acc is not None) else None
        lines.append(f"| {ax} | {_fmt(b_acc)} | {_fmt(c_acc)} | {_fmt(delta)} |")

    lines.extend(['', '## Tier 3 观察告警（Warnings）', ''])
    if warnings:
        for w in warnings:
            lines.append(f'- {w}')
    else:
        lines.append('- 无告警。')

    return '\n'.join(lines) + '\n'


def compare_reports(
    base_ref: str | Path,
    cand_ref: str | Path,
    *,
    split: str | None = None,
    out_dir: Path | None = None,
    write_legacy_name: bool = False,
    reports_root: Path | None = None,
) -> dict:
    """Compare two v3 report directories under the universal 3-tier gate and write JSON + Markdown."""
    base_dir = resolve_report_dir(base_ref, split=split, reports_root=reports_root)
    cand_dir = resolve_report_dir(cand_ref, split=split, reports_root=reports_root)
    base_summary, base_rows = _load_report(base_dir)
    cand_summary, cand_rows = _load_report(cand_dir)

    base_ver = base_summary.get('case_version')
    cand_ver = cand_summary.get('case_version')
    if base_ver != REQUIRED_CASE_VERSION or cand_ver != REQUIRED_CASE_VERSION:
        raise ValueError(
            f'仅支持 case_version={REQUIRED_CASE_VERSION} 报告对比，禁止跨版本对比（base={base_ver!r}, cand={cand_ver!r}）'
        )

    base_split = base_summary.get('split')
    cand_split = cand_summary.get('split')
    if not base_split or base_split != cand_split:
        raise ValueError(f'两份报告的 split 不一致（base={base_split!r}, cand={cand_split!r}）')
    if split is not None and base_split != split:
        raise ValueError(f'报告 split={base_split!r} 与请求的 split={split!r} 不匹配')
    eff_split = str(base_split)

    base_verify = [r for r in base_rows if r.get('suite') == 'verify']
    cand_verify_by_id = {r['id']: r for r in cand_rows if r.get('suite') == 'verify'}
    aligned_base = [r for r in base_verify if r['id'] in cand_verify_by_id]
    aligned_cand = [cand_verify_by_id[r['id']] for r in aligned_base]
    if not aligned_base:
        raise ValueError('两份报告没有可配对的 verify 试次')

    paired = paired_bootstrap(
        [bool(r['correct']) for r in aligned_base],
        [bool(r['correct']) for r in aligned_cand],
    )
    base_agg = aggregate(aligned_base)
    cand_agg = aggregate(aligned_cand)
    base_acc = base_agg['accuracy']
    cand_acc = cand_agg['accuracy']
    base_fa = base_agg['false_accept_rate']
    cand_fa = cand_agg['false_accept_rate']
    fa_delta = (cand_fa - base_fa) if (cand_fa is not None and base_fa is not None) else None

    base_task = (base_summary.get('dimensions') or {}).get('task') or evaluate_task(base_rows)
    cand_task = (cand_summary.get('dimensions') or {}).get('task') or evaluate_task(cand_rows)
    base_eer, base_mindcf = _pick_eer_and_dcf(base_task)
    cand_eer, cand_mindcf = _pick_eer_and_dcf(cand_task)
    eer_delta = (cand_eer - base_eer) if (cand_eer is not None and base_eer is not None) else None
    mindcf_delta = (cand_mindcf - base_mindcf) if (cand_mindcf is not None and base_mindcf is not None) else None

    redline_checks = _evaluate_tier1_redlines(cand_summary, cand_rows, cand_task, eff_split)
    redline_passed = all(redline_checks.values())

    adoption_allowed = (eff_split == 'validation')
    paired_passed = bool(paired['low'] > 0.0)
    fa_passed = (fa_delta is not None) and (fa_delta <= 0.02)
    eer_passed = (eer_delta is not None) and (cand_eer <= base_eer + 1e-6)
    primary_passed = bool(adoption_allowed and paired_passed and fa_passed and eer_passed)
    adopt = bool(redline_passed and primary_passed)

    warnings, worst_rob_drop = _evaluate_tier3_warnings(base_summary, cand_summary)

    base_tag = _report_tag(base_dir, base_summary, eff_split)
    cand_tag = _report_tag(cand_dir, cand_summary, eff_split)
    reason = _reason_text(
        eff_split, base_tag, cand_tag, paired, fa_left=base_fa, fa_right=cand_fa,
        base_eer=base_eer, cand_eer=cand_eer, redline_checks=redline_checks,
        primary_passed=primary_passed, adopt=adopt, warnings=warnings,
    )

    out_root = Path(out_dir) if out_dir is not None else cand_dir.parent
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f'compare_{base_tag}_vs_{cand_tag}_{eff_split}.json'
    md_path = out_root / f'compare_{base_tag}_vs_{cand_tag}_{eff_split}.md'

    report = {
        'split': eff_split,
        'case_version': REQUIRED_CASE_VERSION,
        'base': {
            'tag': base_tag,
            'embedder': base_summary.get('embedder'),
            'dir': str(base_dir),
            'accuracy': base_acc,
            'false_accept_rate': base_fa,
            'eer': base_eer,
            'min_dcf_norm': base_mindcf,
            'model': base_summary.get('model'),
        },
        'cand': {
            'tag': cand_tag,
            'embedder': cand_summary.get('embedder'),
            'dir': str(cand_dir),
            'accuracy': cand_acc,
            'false_accept_rate': cand_fa,
            'eer': cand_eer,
            'min_dcf_norm': cand_mindcf,
            'model': cand_summary.get('model'),
        },
        'adopt': adopt,
        'adoption_allowed': adoption_allowed,
        'redline_passed': redline_passed,
        'primary_passed': primary_passed,
        'fa_delta': fa_delta,
        'eer_delta': eer_delta,
        'mindcf_delta': mindcf_delta,
        'paired': paired,
        'warnings': warnings,
        'gate': {
            'tier1_redlines': redline_checks,
            'tier2_primary': {
                'adoption_allowed': adoption_allowed,
                'paired_passed': paired_passed,
                'fa_passed': fa_passed,
                'eer_passed': eer_passed,
            },
            'tier3_observational': {
                'warnings': warnings,
                'worst_robustness_drop_vs_base': worst_rob_drop,
            },
        },
        'reason': reason,
        'json_path': str(json_path),
        'markdown_path': str(md_path),
    }

    md_text = _format_compare_markdown(report, base_summary, cand_summary)
    json_text = json.dumps(report, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding='utf-8')
    md_path.write_text(md_text, encoding='utf-8')
    if write_legacy_name:
        (out_root / f'compare_{eff_split}.json').write_text(json_text, encoding='utf-8')
        (out_root / f'compare_{eff_split}.md').write_text(md_text, encoding='utf-8')
    return report
