"""Check the grader and the security primitives before any model result is trusted."""
import json
import tempfile
from pathlib import Path

import numpy as np

from voice_eval.grader import aggregate, grade


def _policy_rows(decisions: list[str], expects: list[str]) -> list[dict]:
    return [grade(expect, decision) for expect, decision in zip(expects, decisions)]


def grader_checks() -> list[str]:
    errors = []
    expects = ['accept', 'accept', 'reject', 'reject']
    oracle = _policy_rows(expects, expects)
    if aggregate(oracle)['accuracy'] != 1:
        errors.append('正确对照没有被评为全部正确')
    always_accept = _policy_rows(['accept'] * 4, expects)
    if aggregate(always_accept)['false_accept_rate'] != 1:
        errors.append('一律接受没有在冒充样本上被抓住')
    always_reject = _policy_rows(['reject'] * 4, expects)
    if aggregate(always_reject)['false_reject_rate'] != 1:
        errors.append('一律拒绝没有在本人样本上被抓住')
    always_review = _policy_rows(['uncertain'] * 4, expects)
    stats = aggregate(always_review)
    if stats['accuracy'] != 0 or stats['review_rate'] != 1:
        errors.append('一律转人工仍被算作自动判对')
    again = grade('reject', 'accept')
    if again != grade('reject', 'accept'):
        errors.append('裁判不稳定')
    return errors


def security_checks() -> list[str]:
    from voiceprint.security import (
        apply_projection, decrypt, dpapi_protect, dpapi_unprotect, encrypt, projection)

    errors = []
    secret = b'voiceprint-master-key-check'
    if dpapi_unprotect(dpapi_protect(secret)) != secret:
        errors.append('DPAPI 不能还原主密钥')
    key = b'k' * 32
    if decrypt(key, encrypt(key, secret)) != secret:
        errors.append('AES-GCM 不能还原模板')
    try:
        decrypt(key, encrypt(key, secret, b'owner-a'), b'owner-b')
        errors.append('换了归属的密文仍能解开')
    except Exception:
        pass
    rng = np.random.default_rng(0)
    left, right = rng.normal(size=(2, 32)).astype(np.float32)
    matrix = projection(b'seed', 32)
    raw = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))
    turned_l = apply_projection(left, matrix)
    turned_r = apply_projection(right, matrix)
    projected = float(turned_l @ turned_r / (np.linalg.norm(turned_l) * np.linalg.norm(turned_r)))
    if abs(raw - projected) > 1e-5:
        errors.append('正交投影改变了余弦相似度')
    return errors


def recompute_check() -> list[str]:
    rows = [grade('accept', 'accept'), grade('reject', 'uncertain')]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'cases.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        loaded = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if aggregate(loaded) != aggregate(rows):
        return ['汇总不能由逐条结果重算']
    return []


def run_all() -> dict:
    errors = grader_checks() + security_checks() + recompute_check()
    return {'ok': not errors, 'errors': errors}
