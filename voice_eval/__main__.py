"""Voiceprint evaluation commands.

    python -m voice_eval audit
    python -m voice_eval build
    python -m voice_eval run --embedder stats --split validation
    python -m voice_eval compare
"""
import argparse
from pathlib import Path

from voice_eval.audit import run_all
from voice_eval.cases import build
from voice_eval.gate import compare_reports
from voice_eval.isolated_worker import IsolatedEmbedder
from voice_eval.offline import strict_offline_env
from voice_eval.runner import REPORTS, run_split
from voiceprint.embed import load_embedder

_DEFAULT_REPORTS = REPORTS


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('audit')
    sub.add_parser('build')

    run = sub.add_parser('run')
    run.add_argument('--embedder', required=True, choices=['stats', 'vpnet', 'ecapa'])
    run.add_argument('--split', default='validation', choices=['validation', 'locked', 'both'])
    run.add_argument('--no-cache', action='store_true')
    run.add_argument('--skip-mixture', action='store_true')
    run.add_argument('--isolate-worker', action='store_true')
    run.add_argument('--worker-timeout', type=float, default=30.0)
    run.add_argument('--max-worker-failures', type=int, default=3)

    cmp = sub.add_parser('compare')
    cmp.add_argument('--base', default=None)
    cmp.add_argument('--cand', default=None)
    cmp.add_argument('--out', default=None, type=Path)
    cmp.add_argument('--split', default=None, choices=['validation', 'locked'])

    args = parser.parse_args(argv)
    if args.cmd == 'audit':
        result = run_all()
        print('评测健康检查通过' if result['ok'] else '评测健康检查失败: ' + '；'.join(result['errors']))
        raise SystemExit(0 if result['ok'] else 1)
    if args.cmd == 'build':
        print(build())
        return
    if args.cmd == 'compare':
        default_pair = args.base is None and args.cand is None
        base_ref = args.base if args.base is not None else 'stats'
        cand_ref = args.cand if args.cand is not None else 'vpnet'
        if args.split is not None:
            eff_split = args.split
        elif Path(base_ref).exists() or (REPORTS / str(base_ref)).exists():
            eff_split = None
        else:
            eff_split = 'validation'
        reports_root = REPORTS if REPORTS != _DEFAULT_REPORTS else None
        report = compare_reports(
            base_ref,
            cand_ref,
            split=eff_split,
            out_dir=args.out,
            write_legacy_name=default_pair,
            reports_root=reports_root,
        )
        print(report['reason'])
        return

    audit = run_all()
    if not audit['ok']:
        raise SystemExit('评测健康检查失败: ' + '；'.join(audit['errors']))
    splits = ['validation', 'locked'] if args.split == 'both' else [args.split]
    with strict_offline_env():
        if args.isolate_worker:
            embedder = IsolatedEmbedder(
                args.embedder,
                timeout_s=args.worker_timeout,
                max_consecutive_failures=args.max_worker_failures,
            )
        else:
            embedder = load_embedder(args.embedder)
        try:
            for split in splits:
                path = run_split(
                    embedder,
                    split,
                    with_security=(split == 'validation'),
                    use_cache=not args.no_cache,
                    skip_mixture=args.skip_mixture,
                    isolate_worker=args.isolate_worker,
                )
                print(path)
        finally:
            if hasattr(embedder, 'close'):
                embedder.close()


if __name__ == '__main__':
    main()
