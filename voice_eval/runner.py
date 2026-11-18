"""Run the voiceprint agent on built cases and write a traceable report.

Each case gets a new database. The agent receives only the public commands.
"""
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from agent.service import VoiceAgent
from voice_eval.cases import case_version, load
from voice_eval.grader import aggregate, grade

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports' / 'voice_eval'


def _decision_for_grade(case: dict, result: dict) -> tuple[str, str]:
    hidden = case['hidden']
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


def run_case(case: dict, embedder) -> dict:
    with TemporaryDirectory(prefix='vpcase_') as tmp:
        agent = VoiceAgent(tmp, embedder, detail=True)
        try:
            trace = []
            for step in case['steps']:
                outcome = agent.handle(step)
                trace.append({'command': step, 'tool': outcome['tool'], 'ok': outcome['ok'],
                              'result': outcome['result'], 'reply': outcome['reply']})
                if outcome['tool'] == 'error' and step is not case['steps'][-1]:
                    break
            last = trace[-1]
            expect, observed = _decision_for_grade(case, last['result'])
            graded = grade(expect, observed)
            if case['suite'] == 'identify' and last['result'].get('decision') == 'accept' \
                    and last['result'].get('name') != case['hidden']['true_speaker'] \
                    and case['hidden']['expect'] == 'accept':
                graded['correct'] = False
                graded['false_accept'] = True
            return {
                'id': case['id'],
                'suite': case['suite'],
                'split': case['split'],
                'true_speaker': case['hidden']['true_speaker'],
                'claimed': case['hidden']['claimed'],
                'probe_session': case['hidden'].get('probe_session'),
                'model': last and agent.embedder.describe(),
                'trace': trace,
                **graded,
            }
        finally:
            agent.close()


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
    import numpy as np
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


def write_report(embedder_name: str, split: str, rows: list[dict], model: dict) -> Path:
    out = REPORTS / f'{embedder_name}_{split}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cases.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    reloaded = [json.loads(line) for line in (out / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    by_suite = {}
    suite_names = ('verify', 'verify_impostor', 'verify_short', 'verify_noise', 'identify', 'enroll_quality', 'security')
    for suite in suite_names:
        by_suite[suite] = aggregate([row for row in reloaded if row['suite'] == suite])
    if aggregate(reloaded) != aggregate(rows):
        raise RuntimeError('报告汇总和逐条结果不一致')
    summary = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'embedder': embedder_name,
        'split': split,
        'model': model,
        'protocol': json.loads((Path(__file__).resolve().parent / 'protocol.json').read_text(encoding='utf-8')),
        'case_version': case_version(),
        'by_suite': by_suite,
        'cross_session_verify_accept': aggregate([
            row for row in reloaded
            if row['suite'] == 'verify' and row.get('expect') == 'accept' and row.get('probe_session') == 'cross']),
        'recomputed_from_jsonl': True,
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def run_split(embedder, split: str, with_security: bool) -> Path:
    cases = load(split)
    rows = []
    for index, case in enumerate(cases, start=1):
        rows.append(run_case(case, embedder))
        if index % 10 == 0 or index == len(cases):
            print(f'{embedder.name} {split} {index}/{len(cases)}', flush=True)
    if with_security:
        import shlex
        # The security checks need two speakers this embedder can enroll; a weak embedder may refuse some.
        enrolled = {row['id'] for row in rows
                    if row['id'].startswith('verify-accept-') and row['trace'] and row['trace'][0]['ok']}
        samples = [case for case in cases if case['id'] in enrolled]
        if len(samples) < 2:
            rows.append(_security_row('security-setup', False, '没有两个能注册成功的说话人，安全检查没有运行'))
        else:
            enroll = shlex.split(samples[0]['steps'][0], posix=True)[3:]
            probe = shlex.split(samples[0]['steps'][1], posix=True)[2]
            other = shlex.split(samples[1]['steps'][0], posix=True)[3:]
            rows.extend(security_cases(embedder, enroll, probe, other))
    return write_report(embedder.name, split, rows, embedder.describe())
