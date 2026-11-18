"""Check the grader and the security primitives before any model result is trusted."""
import json
import tempfile
from pathlib import Path

import numpy as np

from voice_eval.grader import aggregate, grade

PROTOCOL_PATH = Path(__file__).resolve().parent / 'protocol.json'
EXPECTED_DIMENSIONS = ['task', 'tooling', 'response', 'flow', 'perf', 'redteam', 'robustness']


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


def dim_checks() -> list[str]:
    from voice_eval.cache import CachedEmbedder
    from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav
    from voice_eval.dims.task import MINDCF_PARAMS, compute_eer, compute_mindcf
    from voiceprint.audio import file_digest

    errors = []

    eer_sep, thr_sep = compute_eer([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    if eer_sep != 0.0 or thr_sep is None:
        errors.append('EER 在完全可分分数上未返回 0')
    eer_inv, _ = compute_eer([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
    if eer_inv is None or abs(eer_inv - 1.0) > 1e-6:
        errors.append('EER 在完全反转分数上未返回 1')
    eer_tie, _ = compute_eer([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0])
    if eer_tie is None or abs(eer_tie - 0.5) > 1e-6:
        errors.append('EER 在同分随机对照上未返回 0.5')
    if compute_eer([], []) != (None, None) or compute_eer([0.5], [1]) != (None, None):
        errors.append('EER 在退化单类/空输入上未返回 None')

    dcf_sep = compute_mindcf([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], **MINDCF_PARAMS)
    if dcf_sep['min_dcf_norm'] != 0.0:
        errors.append('minDCF 在完全可分分数上未返回 0')
    dcf_tie = compute_mindcf([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0], **MINDCF_PARAMS)
    if dcf_tie['min_dcf_norm'] is None or abs(dcf_tie['min_dcf_norm'] - 1.0) > 1e-6:
        errors.append('minDCF 在同分对照上归一化值未返回 1.0')

    try:
        with tempfile.TemporaryDirectory(prefix='vpaudit_cache_') as tmp:
            tmp_path = Path(tmp)
            wav = generate_speaker_wav(tmp_path / 'sample.wav', speaker_index=0, duration=2.0, phase=0.0, seed=0)
            cache_dir = tmp_path / 'cache'
            c1 = CachedEmbedder(ToneEmbedder(), cache_dir=cache_dir)
            v1 = c1.embed_file(str(wav))
            if v1.shape != (c1.dim,) or not bool(np.all(np.isfinite(v1))) or c1.stats()['misses'] != 1:
                errors.append('ToneEmbedder + CachedEmbedder 首次计算校验失败')
            c2 = CachedEmbedder(ToneEmbedder(), cache_dir=cache_dir)
            v2 = c2.embed_file(str(wav))
            if c2.stats()['hits'] != 1 or not np.allclose(v1, v2, atol=1e-6):
                errors.append('CachedEmbedder 磁盘命中往返结果不一致')
            cache_file = c2._cache_path(file_digest(str(wav)))
            cache_file.write_bytes(b'corrupted-npy-payload')
            c3 = CachedEmbedder(ToneEmbedder(), cache_dir=cache_dir)
            v3 = c3.embed_file(str(wav))
            if c3.stats()['misses'] != 1 or not np.allclose(v1, v3, atol=1e-6):
                errors.append('CachedEmbedder 损坏缓存自愈重算失败')
    except Exception as exc:
        errors.append(f'嵌入缓存往返自检异常：{exc}')

    try:
        proto = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
        if proto.get('case_version') != 3:
            errors.append(f'protocol.json case_version 必须为 3，当前为 {proto.get("case_version")!r}')
        if proto.get('dimensions') != EXPECTED_DIMENSIONS:
            errors.append('protocol.json dimensions 与七维协议不匹配')
        gate_tiers = proto.get('gate_tiers')
        required_tiers = {'tier1_redlines', 'tier2_primary', 'tier3_observational'}
        if not isinstance(gate_tiers, dict) or not required_tiers.issubset(gate_tiers):
            errors.append('protocol.json 缺少完整的 gate_tiers 三级门禁定义')
        if proto.get('mindcf_params') != MINDCF_PARAMS:
            errors.append('protocol.json mindcf_params 与任务维度参数不一致')
    except Exception as exc:
        errors.append(f'读取或校验 protocol.json 失败：{exc}')

    return errors


def run_all() -> dict:
    errors = grader_checks() + security_checks() + recompute_check() + dim_checks()
    return {'ok': not errors, 'errors': errors}
