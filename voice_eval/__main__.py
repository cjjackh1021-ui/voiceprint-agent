"""Voiceprint evaluation commands.

    python -m voice_eval audit
    python -m voice_eval build
    python -m voice_eval run --embedder stats --split validation
    python -m voice_eval compare
"""
import argparse
import json
from pathlib import Path

from voice_eval.audit import run_all
from voice_eval.cases import build
from voice_eval.grader import aggregate, paired_bootstrap
from voice_eval.runner import REPORTS, run_split
from voiceprint.embed import load_embedder


def _verify_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    return [row for row in rows if row['suite'] == 'verify']


def compare(split: str) -> dict:
    left_path = REPORTS / f'stats_{split}' / 'cases.jsonl'
    right_path = REPORTS / f'vpnet_{split}' / 'cases.jsonl'
    left, right = _verify_rows(left_path), _verify_rows(right_path)
    by_id_right = {row['id']: row for row in right}
    paired_left, paired_right = [], []
    for row in left:
        if row['id'] in by_id_right:
            paired_left.append(row)
            paired_right.append(by_id_right[row['id']])
    diff = paired_bootstrap([row['correct'] for row in paired_left], [row['correct'] for row in paired_right])
    fa_left = aggregate(paired_left)['false_accept_rate']
    fa_right = aggregate(paired_right)['false_accept_rate']
    security = []
    for name in ('stats', 'vpnet'):
        sec_path = REPORTS / f'{name}_validation' / 'cases.jsonl'
        if sec_path.exists():
            rows = [json.loads(line) for line in sec_path.read_text(encoding='utf-8').splitlines()]
            security.append(all(row['correct'] for row in rows if row['suite'] == 'security'))
    adoption_allowed = split == 'validation'
    adopt = bool(adoption_allowed and diff['low'] > 0 and fa_right is not None and fa_left is not None
                 and fa_right - fa_left <= 0.02 and all(security))
    report = {
        'split': split,
        'adoption_allowed': adoption_allowed,
        'stats_accuracy': aggregate(paired_left)['accuracy'],
        'vpnet_accuracy': aggregate(paired_right)['accuracy'],
        'stats_false_accept_rate': fa_left,
        'vpnet_false_accept_rate': fa_right,
        'paired': diff,
        'security_all_passed': all(security) if security else False,
        'adopt_vpnet': adopt,
        'reason': _reason(split, diff, fa_left, fa_right, all(security) if security else False, adopt),
    }
    out = REPORTS / f'compare_{split}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report['reason'])
    return report


def _reason(split, diff, fa_left, fa_right, security_ok, adopt) -> str:
    text = (f'{split} 上 VPNet 比统计基线的验证准确率高 {diff["diff"]:.3f}，'
            f'95% 区间 [{diff["low"]:.3f}, {diff["high"]:.3f}]。'
            f'误接受率 {fa_left:.3f} -> {fa_right:.3f}。')
    if split != 'validation':
        return text + ' 这是锁定测试，只报告，不据此更换模型。'
    if adopt:
        return text + ' 按预定规则可以采用 VPNet。'
    if not security_ok:
        return text + ' 安全检查未全部通过，不采用。'
    return text + ' 提升没有超过预定的区间或误接受护栏，保留统计基线为对照，不把这次结果说成已胜出。'


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('audit')
    sub.add_parser('build')
    run = sub.add_parser('run')
    run.add_argument('--embedder', required=True, choices=['stats', 'vpnet', 'ecapa'])
    run.add_argument('--split', default='validation', choices=['validation', 'locked', 'both'])
    cmp = sub.add_parser('compare')
    cmp.add_argument('--split', default='validation', choices=['validation', 'locked'])
    args = parser.parse_args()
    if args.cmd == 'audit':
        result = run_all()
        print('评测健康检查通过' if result['ok'] else '评测健康检查失败: ' + '；'.join(result['errors']))
        raise SystemExit(0 if result['ok'] else 1)
    if args.cmd == 'build':
        print(build())
        return
    if args.cmd == 'compare':
        compare(args.split)
        return
    audit = run_all()
    if not audit['ok']:
        raise SystemExit('评测健康检查失败: ' + '；'.join(audit['errors']))
    embedder = load_embedder(args.embedder)
    splits = ['validation', 'locked'] if args.split == 'both' else [args.split]
    for split in splits:
        path = run_split(embedder, split, with_security=(split == 'validation'))
        print(path)


if __name__ == '__main__':
    main()
