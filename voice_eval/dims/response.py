"""Dimension 3: reply-decision consistency, detail=False score-leak guards, and vector-export protection."""
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from agent.service import VoiceAgent
from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav, sandbox_cohort

FORBIDDEN_NON_DETAIL_KEYS = frozenset({'score', 'raw_score', 'threshold', 'theta_deg', 'low', 'high', 'gap'})
FORBIDDEN_NON_DETAIL_TERMS = ('归一化相似度', '原始余弦', '夹角', '验证阈值', '相差')
_VECTOR_SEQ_RE = re.compile(r'\[(?:\s*-?\d+(?:\.\d+)?\s*,){4,}')
_FLOAT_RE = re.compile(r'-?\d+\.\d+')
_NUMERIC_THRESHOLD_RE = re.compile(r'阈值\s*[:：=]?\s*-?\d+(?:\.\d+)?')
_SPLIT_WINNER_REPLY_RE = re.compile(
    r'^没有足够把握。(?P<other>[^\s。，]+) 过了自己的阈值，但分数最高的是 (?P<best>[^\s。，]+)。请再录一段更长的录音。$'
)


def _contains_large_numeric_vector(obj: object) -> bool:
    if isinstance(obj, np.ndarray):
        return obj.size > 16
    if isinstance(obj, dict):
        return any(_contains_large_numeric_vector(val) for val in obj.values())
    if isinstance(obj, (list, tuple)):
        numeric_count = sum(1 for item in obj if isinstance(item, (int, float)) and not isinstance(item, bool))
        if numeric_count > 16:
            return True
        return any(_contains_large_numeric_vector(item) for item in obj)
    return False


def _scrub_names(
    reply: str,
    result: dict,
    speaker_name: str | None,
    extra_names: list[str] | None = None,
) -> str:
    scrubbed = reply
    candidates = [speaker_name, result.get('name'), result.get('candidate'), *(extra_names or ())]
    for token in sorted({str(c) for c in candidates if c}, key=len, reverse=True):
        scrubbed = scrubbed.replace(token, '')
    return scrubbed


def _split_winner_match(
    reply: str,
    result: dict,
    tool: str | None,
    expect_decision: str,
) -> re.Match[str] | None:
    if tool != 'identify' or expect_decision != 'uncertain':
        return None
    match = _SPLIT_WINNER_REPLY_RE.fullmatch(reply)
    if match is None:
        return None
    candidate = result.get('candidate')
    if candidate is not None and match.group('other') != str(candidate):
        return None
    return match


def _inspect_reply_payload(
    outcome: dict,
    expect_decision: str,
    detail: bool,
    speaker_name: str | None = None,
) -> tuple[bool, bool, str]:
    """Return (passed, is_leak, reason) for an agent verify/identify outcome."""
    result = outcome.get('result') or {}
    reply = str(outcome.get('reply') or '')
    tool = outcome.get('tool')

    if _contains_large_numeric_vector(result) or _VECTOR_SEQ_RE.search(reply):
        return False, True, 'plaintext embedding vector leaked in payload or reply'

    if not detail:
        leaked_keys = sorted(FORBIDDEN_NON_DETAIL_KEYS.intersection(result.keys()))
        if leaked_keys:
            return False, True, f'forbidden detail keys exposed when detail=False: {leaked_keys}'
        for term in FORBIDDEN_NON_DETAIL_TERMS:
            if term in reply:
                return False, True, f'forbidden score term {term!r} leaked when detail=False'
        if _NUMERIC_THRESHOLD_RE.search(reply):
            return False, True, f'numeric threshold leaked in reply: {reply!r}'
        split_match = _split_winner_match(reply, result, tool, expect_decision)
        allow_unknown_threshold = expect_decision == 'unknown' and tool != 'verify' and '未达到辨认阈值' in reply
        if '阈值' in reply and not allow_unknown_threshold and split_match is None:
            return False, True, f'threshold keyword leaked in reply: {reply!r}'
        scrubbed_base = _scrub_names(reply, result, speaker_name)
        if _FLOAT_RE.search(scrubbed_base):
            return False, True, f'floating-point score leaked in reply: {reply!r}'
        extra = [split_match.group('best')] if split_match is not None else None
        scrubbed = _scrub_names(reply, result, speaker_name, extra_names=extra)
        if expect_decision != 'locked' and re.search(r'\d', scrubbed):
            return False, True, f'unexpected numeric digit leaked in reply: {reply!r}'

    actual_decision = result.get('decision')
    if actual_decision != expect_decision:
        return False, False, f'decision mismatch: expected {expect_decision!r}, got {actual_decision!r}'

    if expect_decision == 'accept':
        if tool == 'identify':
            if '最匹配的是' not in reply or (speaker_name and speaker_name not in reply):
                return False, False, f'identify accept reply missing expected speaker match: {reply!r}'
        elif tool == 'verify':
            if '接受' not in reply or (speaker_name and speaker_name not in reply):
                return False, False, f'verify accept reply missing acceptance phrase: {reply!r}'
        elif '接受' not in reply and '最匹配的是' not in reply:
            return False, False, f'accept reply missing acceptance phrase: {reply!r}'
    elif expect_decision == 'reject':
        if '拒绝' not in reply:
            return False, False, f'reject reply missing 拒绝: {reply!r}'
    elif expect_decision == 'uncertain':
        if tool == 'identify':
            if '没有足够把握' not in reply:
                return False, False, f'identify uncertain reply missing 没有足够把握: {reply!r}'
        elif tool == 'verify':
            if '不确定' not in reply:
                return False, False, f'verify uncertain reply missing 不确定: {reply!r}'
        elif '不确定' not in reply and '没有足够把握' not in reply:
            return False, False, f'uncertain reply missing uncertainty phrase: {reply!r}'
    elif expect_decision == 'unknown':
        if not any(tok in reply for tok in ('没有', '还没有注册', '未达到辨认阈值')):
            return False, False, f'unknown reply missing expected phrase: {reply!r}'
    elif expect_decision == 'locked':
        if '暂停验证' not in reply:
            return False, False, f'locked reply missing 暂停验证: {reply!r}'

    return True, False, 'ok'


def check_reply_payload(
    outcome: dict,
    expect_decision: str,
    detail: bool,
    speaker_name: str | None = None,
) -> tuple[bool, str]:
    """Validate reply consistency, detail=False non-leakage, and vector export guardrails."""
    passed, _is_leak, reason = _inspect_reply_payload(outcome, expect_decision, detail, speaker_name)
    return passed, reason


def _enroll(agent: VoiceAgent, name: str, base_dir: Path, speaker_index: int) -> None:
    takes = [
        generate_speaker_wav(base_dir / 's1' / f'{name}_0.wav', speaker_index, duration=4.0, phase=0.0, seed=0),
        generate_speaker_wav(base_dir / 's1' / f'{name}_1.wav', speaker_index, duration=4.0, phase=0.8, seed=1),
        generate_speaker_wav(base_dir / 's2' / f'{name}_2.wav', speaker_index, duration=4.0, phase=1.6, seed=2),
    ]
    out = agent.handle(f'注册 {name} 本人同意 ' + ' '.join(f'"{p}"' for p in takes))
    if not out['ok']:
        raise RuntimeError(f'sandbox enrollment failed for {name}: {out}')


def _calibrate_uncertain_blend(
    root: Path,
    embedder: ToneEmbedder,
    wav_a: Path,
    wav_b: Path,
    command_prefix: str,
    tag: str,
) -> Path:
    """Search blend alpha using unique filenames to hit decision='uncertain' cleanly without cache/lock interference."""
    sig_a, sr = sf.read(str(wav_a), dtype='float32')
    sig_b, _ = sf.read(str(wav_b), dtype='float32')
    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    blend_dir = root / f'blends_{tag}'
    blend_dir.mkdir(parents=True, exist_ok=True)
    last_accept = 0.0
    first_reject = 1.0

    def _probe_grid(alphas: np.ndarray, prefix: str) -> tuple[Path | None, float, float]:
        lo, hi = last_accept, first_reject
        calib = VoiceAgent(root / f'calib_{tag}_{prefix}', embedder, detail=True, audio_roots=[root])
        try:
            _enroll(calib, 'alice', root / 'calib_audio', 0)
            _enroll(calib, 'bob', root / 'calib_audio', 1)
            for idx, alpha in enumerate(alphas):
                candidate = blend_dir / f'{prefix}_{idx:03d}.wav'
                mixed = ((1.0 - float(alpha)) * sig_a + float(alpha) * sig_b).astype(np.float32)
                sf.write(str(candidate), mixed, sr)
                out = calib.handle(f'{command_prefix} "{candidate}"')
                dec = (out.get('result') or {}).get('decision')
                if dec == 'uncertain':
                    return candidate, lo, hi
                if dec == 'accept':
                    lo = float(alpha)
                elif dec in ('reject', 'unknown'):
                    hi = float(alpha)
                    break
                elif dec == 'locked':
                    break
        finally:
            calib.close()
        return None, lo, hi

    found, last_accept, first_reject = _probe_grid(np.linspace(0.0, 1.0, 41), 'coarse')
    if found is not None:
        return found
    found, _, _ = _probe_grid(np.linspace(last_accept, first_reject, 61), 'fine')
    if found is not None:
        return found
    raise RuntimeError(f'failed to synthesize uncertain blend for {command_prefix}')


def evaluate_response(real_non_detail_checks: list[dict] | None = None) -> dict:
    """Validate reply consistency and anti-leak rules across ToneEmbedder branches plus real_non_detail_checks."""
    embedder = ToneEmbedder()
    checks: list[dict] = []
    leak_count = 0

    def _record(label: str, outcome: dict, expect_decision: str, detail: bool, speaker_name: str | None) -> None:
        nonlocal leak_count
        passed, is_leak, reason = _inspect_reply_payload(outcome, expect_decision, detail, speaker_name)
        if is_leak:
            leak_count += 1
        checks.append({
            'id': label,
            'detail': detail,
            'expect_decision': expect_decision,
            'actual_decision': (outcome.get('result') or {}).get('decision'),
            'passed': passed,
            'leak': is_leak,
            'reason': reason,
        })

    with sandbox_cohort(embedder), TemporaryDirectory(prefix='vpresp_') as tmp:
        root = Path(tmp)
        audio_dir = root / 'audio'
        alice_probe = generate_speaker_wav(audio_dir / 'alice_probe.wav', 0, duration=4.0, phase=2.4, seed=3)
        bob_probe = generate_speaker_wav(audio_dir / 'bob_probe.wav', 1, duration=4.0, phase=2.4, seed=3)
        stranger_probe = generate_speaker_wav(audio_dir / 'stranger_probe.wav', 2, duration=4.0, phase=0.0, seed=0)

        uncertain_verify_wav = _calibrate_uncertain_blend(
            root, embedder, alice_probe, bob_probe, '验证 alice', 'verify'
        )
        uncertain_identify_wav = _calibrate_uncertain_blend(
            root, embedder, alice_probe, bob_probe, '辨认', 'identify'
        )

        for detail_flag in (False, True):
            tag = 'detail' if detail_flag else 'quiet'
            agent = VoiceAgent(root / f'db_{tag}', embedder, detail=detail_flag, audio_roots=[root])
            try:
                _enroll(agent, 'alice', audio_dir / f'enroll_{tag}', 0)
                _enroll(agent, 'bob', audio_dir / f'enroll_{tag}', 1)

                _record(f'{tag}-verify-accept', agent.handle(f'验证 alice "{alice_probe}"'), 'accept', detail_flag, 'alice')
                _record(f'{tag}-verify-uncertain', agent.handle(f'验证 alice "{uncertain_verify_wav}"'), 'uncertain', detail_flag, 'alice')
                _record(f'{tag}-verify-unknown', agent.handle(f'验证 nobody "{alice_probe}"'), 'unknown', detail_flag, 'nobody')
                _record(f'{tag}-identify-accept', agent.handle(f'辨认 "{alice_probe}"'), 'accept', detail_flag, 'alice')
                _record(f'{tag}-identify-uncertain', agent.handle(f'辨认 "{uncertain_identify_wav}"'), 'uncertain', detail_flag, None)
                _record(f'{tag}-identify-unknown', agent.handle(f'辨认 "{stranger_probe}"'), 'unknown', detail_flag, None)

                # Reset consecutive verify failures via genuine accept before driving verify-reject -> locked.
                agent.handle(f'验证 alice "{alice_probe}"')
                first_reject = agent.handle(f'验证 alice "{bob_probe}"')
                _record(f'{tag}-verify-reject', first_reject, 'reject', detail_flag, 'alice')
                for _ in range(4):
                    agent.handle(f'验证 alice "{bob_probe}"')
                locked_out = agent.handle(f'验证 alice "{alice_probe}"')
                _record(f'{tag}-verify-locked', locked_out, 'locked', detail_flag, 'alice')
            finally:
                agent.close()

    for item in real_non_detail_checks or []:
        _record(
            item['label'],
            item['outcome'],
            item['expect_decision'],
            False,
            item.get('speaker_name'),
        )

    total = len(checks)
    passed_count = sum(1 for c in checks if c['passed'])
    return {
        'passed': passed_count == total and leak_count == 0,
        'total': total,
        'passed_count': passed_count,
        'leak_count': leak_count,
        'checks': checks,
    }
