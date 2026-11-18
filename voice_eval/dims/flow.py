"""Dimension 4: multi-step state machine and workflow invariant verification."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf

from agent.service import VoiceAgent
from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav, sandbox_cohort
from voiceprint.separate import CKPT as SEPARATE_CKPT


def _enroll(agent: VoiceAgent, name: str, base_dir: Path, speaker_index: int) -> list[Path]:
    takes = [
        generate_speaker_wav(base_dir / 's1' / f'{name}_0.wav', speaker_index, duration=4.0, phase=0.0, seed=0),
        generate_speaker_wav(base_dir / 's1' / f'{name}_1.wav', speaker_index, duration=4.0, phase=0.8, seed=1),
        generate_speaker_wav(base_dir / 's2' / f'{name}_2.wav', speaker_index, duration=4.0, phase=1.6, seed=2),
    ]
    out = agent.handle(f'注册 {name} 本人同意 ' + ' '.join(f'"{p}"' for p in takes))
    if not out['ok']:
        raise RuntimeError(f'flow setup enrollment failed for {name}: {out}')
    return takes


class _AltToneEmbedder(ToneEmbedder):
    model_id = 'tone-v2-alt-' + 'b' * 52


def _flow_revoke_confirm(embedder: ToneEmbedder) -> tuple[bool, str]:
    with TemporaryDirectory(prefix='vpflow_rev_') as tmp:
        root = Path(tmp)
        agent = VoiceAgent(root / 'db', embedder, detail=True, audio_roots=[root])
        try:
            _enroll(agent, 'alice', root / 'audio', 0)
            probe = generate_speaker_wav(root / 'probe.wav', 0, duration=4.0, phase=2.4, seed=3)
            for interrupt in ('名单', '帮助', '退出', '未知指令', f'验证 alice "{probe}"'):
                ask = agent.handle('撤销 alice')
                if not ask['ok'] or agent.pending != 'alice':
                    return False, f'revoke ask failed before {interrupt!r}'
                agent.handle(interrupt)
                if agent.pending is not None:
                    return False, f'pending not cleared by {interrupt!r}'
                bad_confirm = agent.handle('确认撤销 alice')
                if bad_confirm['ok'] or 'alice' not in agent.store.names():
                    return False, f'confirm succeeded after interrupt {interrupt!r}'

            agent.handle('撤销 alice')
            mismatch = agent.handle('确认撤销 bob')
            if mismatch['ok'] or 'alice' not in agent.store.names():
                return False, 'confirm succeeded on mismatched speaker name'

            agent.handle('撤销 alice')
            done = agent.handle('确认撤销 alice')
            if not done['ok'] or 'alice' in agent.store.names() or agent.store.events('alice'):
                return False, f'confirm revoke did not clean speaker and events: {done}'
            return True, 'ok'
        finally:
            agent.close()


def _flow_lockout_and_reset(embedder: ToneEmbedder) -> tuple[bool, str]:
    with TemporaryDirectory(prefix='vpflow_lock_') as tmp:
        root = Path(tmp)
        genuine = generate_speaker_wav(root / 'genuine.wav', 0, duration=4.0, phase=2.4, seed=3)
        impostor = generate_speaker_wav(root / 'impostor.wav', 1, duration=4.0, phase=0.0, seed=0)

        agent_a = VoiceAgent(root / 'db_a', embedder, detail=True, audio_roots=[root])
        try:
            _enroll(agent_a, 'alice', root / 'audio_a', 0)
            for idx in range(5):
                res = agent_a.handle(f'验证 alice "{impostor}"')
                if (res.get('result') or {}).get('decision') != 'reject':
                    return False, f'expected reject on attempt {idx + 1}, got {res}'
            sixth = agent_a.handle(f'验证 alice "{genuine}"')
            if sixth['ok'] or (sixth.get('result') or {}).get('decision') != 'locked':
                return False, f'6th attempt after 5 failures must be locked: {sixth}'
        finally:
            agent_a.close()

        agent_b = VoiceAgent(root / 'db_b', embedder, detail=True, audio_roots=[root])
        try:
            _enroll(agent_b, 'alice', root / 'audio_b', 0)
            for _ in range(4):
                agent_b.handle(f'验证 alice "{impostor}"')
            mid_ok = agent_b.handle(f'验证 alice "{genuine}"')
            if (mid_ok.get('result') or {}).get('decision') != 'accept':
                return False, f'expected mid-streak accept before lock: {mid_ok}'
            for _ in range(2):
                agent_b.handle(f'验证 alice "{impostor}"')
            after = agent_b.handle(f'验证 alice "{genuine}"')
            if not after['ok'] or (after.get('result') or {}).get('decision') != 'accept':
                return False, f'accept did not reset consecutive failure streak: {after}'
            return True, 'ok'
        finally:
            agent_b.close()


def _flow_enroll_quality(embedder: ToneEmbedder) -> tuple[bool, str]:
    with TemporaryDirectory(prefix='vpflow_qual_') as tmp:
        root = Path(tmp)
        agent = VoiceAgent(root / 'db', embedder, detail=True, audio_roots=[root])
        try:
            w0 = generate_speaker_wav(root / 's1' / 'a0.wav', 0, duration=4.0, phase=0.0, seed=0)
            w1 = generate_speaker_wav(root / 's1' / 'a1.wav', 0, duration=4.0, phase=0.8, seed=1)
            w2 = generate_speaker_wav(root / 's2' / 'a2.wav', 0, duration=4.0, phase=1.6, seed=2)
            other = generate_speaker_wav(root / 's2' / 'b0.wav', 1, duration=4.0, phase=0.0, seed=0)

            dup_copy = root / 's2' / 'a0_dup.wav'
            dup_copy.write_bytes(w0.read_bytes())
            res_dup = agent.handle(f'注册 dup_u 本人同意 "{w0}" "{dup_copy}" "{w2}"')
            if res_dup['ok'] or (res_dup.get('result') or {}).get('reason') != 'duplicate':
                return False, f'exact duplicate was not rejected: {res_dup}'

            sig0, sr = sf.read(str(w0), dtype='float32')
            near_dup = root / 's2' / 'a0_near.wav'
            rng = np.random.default_rng(123)
            sf.write(str(near_dup), (sig0 + 1e-4 * rng.normal(size=sig0.shape)).astype(np.float32), sr)
            res_near = agent.handle(f'注册 near_u 本人同意 "{w0}" "{near_dup}" "{w2}"')
            if res_near['ok'] or (res_near.get('result') or {}).get('decision') != 'reject':
                return False, f'near-duplicate copy was not rejected: {res_near}'

            res_mix = agent.handle(f'注册 mix_u 本人同意 "{w0}" "{w1}" "{other}"')
            if res_mix['ok'] or (res_mix.get('result') or {}).get('decision') != 'reject':
                return False, f'mixed speaker enroll was not rejected: {res_mix}'

            s_only = generate_speaker_wav(root / 's1' / 'a3.wav', 0, duration=4.0, phase=2.4, seed=3)
            res_short_same = agent.handle(f'注册 short_u 本人同意 "{w0}" "{w1}" "{s_only}"')
            if res_short_same['ok'] or (res_short_same.get('result') or {}).get('decision') != 'reject':
                return False, f'same-folder <30s enroll was not rejected: {res_short_same}'

            l0 = generate_speaker_wav(root / 'long' / 'l0.wav', 0, duration=10.5, phase=0.0, seed=10)
            l1 = generate_speaker_wav(root / 'long' / 'l1.wav', 0, duration=10.5, phase=0.8, seed=11)
            l2 = generate_speaker_wav(root / 'long' / 'l2.wav', 0, duration=10.5, phase=1.6, seed=12)
            res_long_same = agent.handle(f'注册 long_u 本人同意 "{l0}" "{l1}" "{l2}"')
            if not res_long_same['ok']:
                return False, f'same-folder >=30s enroll failed: {res_long_same}'

            res_cross = agent.handle(f'注册 cross_u 本人同意 "{w0}" "{w1}" "{w2}"')
            if not res_cross['ok']:
                return False, f'cross-folder <30s enroll failed: {res_cross}'

            res_over = agent.handle(f'注册 cross_u 本人同意 "{w0}" "{w1}" "{w2}"')
            if res_over['ok'] or (res_over.get('result') or {}).get('reason') != 'exists':
                return False, f'overwrite existing speaker was not blocked: {res_over}'

            return True, 'ok'
        finally:
            agent.close()


def _flow_key_rotation(embedder: ToneEmbedder) -> tuple[bool, str]:
    with TemporaryDirectory(prefix='vpflow_rot_') as tmp:
        root = Path(tmp)
        agent = VoiceAgent(root / 'db', embedder, detail=True, audio_roots=[root])
        try:
            _enroll(agent, 'alice', root / 'audio', 0)
            probe = generate_speaker_wav(root / 'probe.wav', 0, duration=4.0, phase=2.4, seed=3)
            first = agent.handle(f'验证 alice "{probe}"')
            seed_before, seg_before = agent.store.load_speaker('alice', *agent.model)
            rot = agent.handle('轮换密钥 alice')
            if not rot['ok']:
                return False, f'rotate failed: {rot}'
            seed_after, seg_after = agent.store.load_speaker('alice', *agent.model)
            second = agent.handle(f'验证 alice "{probe}"')
            if seed_before == seed_after or np.allclose(seg_before, seg_after, atol=1e-4):
                return False, 'key rotation did not update decrypted seed/segments'
            if first['result'].get('decision') != second['result'].get('decision') or second['result'].get('decision') != 'accept':
                return False, f'decision changed across rotation: {first} vs {second}'
            return True, 'ok'
        finally:
            agent.close()


def _flow_drift_detection(embedder: ToneEmbedder) -> tuple[bool, str]:
    with TemporaryDirectory(prefix='vpflow_drift_') as tmp:
        root = Path(tmp)
        p1 = generate_speaker_wav(root / 'p1.wav', 0, duration=4.0, phase=2.4, seed=3)
        p2 = generate_speaker_wav(root / 'p2.wav', 0, duration=4.0, phase=3.1, seed=4)
        imp = generate_speaker_wav(root / 'imp.wav', 1, duration=4.0, phase=0.0, seed=0)

        agent_ok = VoiceAgent(root / 'db_ok', embedder, detail=True, audio_roots=[root])
        try:
            _enroll(agent_ok, 'alice', root / 'audio_ok', 0)
            insufficient = agent_ok.handle('漂移 alice')
            if not insufficient['ok'] or insufficient['result'].get('n', -1) != 0 or '记录不足' not in insufficient['reply']:
                return False, f'expected insufficient drift message: {insufficient}'
            agent_ok.handle(f'验证 alice "{p1}"')
            agent_ok.handle(f'验证 alice "{p2}"')
            stable = agent_ok.handle('漂移 alice')
            if not stable['ok'] or stable['result'].get('drift') is not False:
                return False, f'consistent genuine verifies triggered false drift: {stable}'
        finally:
            agent_ok.close()

        agent_drop = VoiceAgent(root / 'db_drop', embedder, detail=True, audio_roots=[root])
        try:
            _enroll(agent_drop, 'alice', root / 'audio_drop', 0)
            agent_drop.handle(f'验证 alice "{p1}"')
            agent_drop.handle(f'验证 alice "{p2}"')
            agent_drop.handle(f'验证 alice "{imp}"')
            drifted = agent_drop.handle('漂移 alice')
            if not drifted['ok'] or drifted['result'].get('drift') is not True or '重新注册' not in drifted['reply']:
                return False, f'score drop failed to trigger drift=True: {drifted}'
            return True, 'ok'
        finally:
            agent_drop.close()


def _flow_cross_model_isolation(embedder: ToneEmbedder) -> tuple[bool, str]:
    alt_embedder = _AltToneEmbedder()
    with sandbox_cohort(alt_embedder), TemporaryDirectory(prefix='vpflow_mod_') as tmp:
        root = Path(tmp)
        db_dir = root / 'db'
        probe = generate_speaker_wav(root / 'probe.wav', 0, duration=4.0, phase=2.4, seed=3)

        agent_a = VoiceAgent(db_dir, embedder, detail=True, audio_roots=[root])
        try:
            _enroll(agent_a, 'alice', root / 'audio', 0)
        finally:
            agent_a.close()

        agent_b = VoiceAgent(db_dir, alt_embedder, detail=True, audio_roots=[root])
        try:
            id_out = agent_b.handle(f'辨认 "{probe}"')
            if id_out['ok'] or (id_out.get('result') or {}).get('decision') != 'unknown':
                return False, f'cross-model identify should report unknown: {id_out}'
            ver_out = agent_b.handle(f'验证 alice "{probe}"')
            if ver_out['ok'] or ver_out['tool'] != 'error' or (ver_out.get('result') or {}).get('error') != 'ModelMismatch':
                return False, f'cross-model verify did not fail safely with ModelMismatch: {ver_out}'
            return True, 'ok'
        finally:
            agent_b.close()


def _run_separate_once(embedder: ToneEmbedder, custom_separator) -> tuple[bool, str, list[str]]:
    with TemporaryDirectory(prefix='vpflow_sep_') as tmp:
        root = Path(tmp)
        audio_dir = root / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        mix_path = audio_dir / 'mix_08s.wav'
        t = np.arange(int(0.8 * 8000), dtype=np.float32) / 8000.0
        mix = 0.35 * np.sin(2.0 * np.pi * 190.0 * t) + 0.35 * np.sin(2.0 * np.pi * 340.0 * t)
        sf.write(str(mix_path), mix.astype(np.float32), 8000)

        agent = VoiceAgent(root / 'db', embedder, detail=True, audio_roots=[audio_dir])
        try:
            _enroll(agent, 'alice', audio_dir / 'enroll', 0)
            _enroll(agent, 'bob', audio_dir / 'enroll', 1)
            if custom_separator is not None:
                with patch('voiceprint.separate.separate_file', custom_separator):
                    out = agent.handle(f'分离再辨认 "{mix_path}"')
            else:
                out = agent.handle(f'分离再辨认 "{mix_path}"')
            if not out['ok'] or out['tool'] != 'separate':
                return False, f'separate_and_identify failed: {out}', []
            paths = (out.get('result') or {}).get('paths', [])
            if len(paths) != 2:
                return False, f'expected 2 separated paths, got {paths}', []
            sep_dir = agent.separated_dir.resolve()
            if not all(Path(p).resolve().parent == sep_dir for p in paths):
                return False, f'separated paths not inside separated_dir: {paths}', []
            if len(out.get('reply', '').splitlines()) != 3:
                return False, f'expected 3 reply lines (separate + 2 identifies), got {out["reply"]!r}', []
            return True, 'ok', list((out.get('result') or {}).get('names', []))
        finally:
            agent.close()


def _flow_separate_and_identify(embedder: ToneEmbedder, separator=None) -> tuple[bool, str]:
    def _deterministic_splitter(_src: str, dest_dir: Path) -> dict:
        dest_dir.mkdir(parents=True, exist_ok=True)
        p1 = generate_speaker_wav(dest_dir / 'det_spk1.wav', speaker_index=0, duration=2.0, phase=2.0, seed=11)
        p2 = generate_speaker_wav(dest_dir / 'det_spk2.wav', speaker_index=1, duration=2.0, phase=2.0, seed=12)
        return {'sample_rate': 8000, 'paths': [str(p1), str(p2)], 'checkpoint': 'deterministic'}

    if separator is not None:
        ok, msg, _names = _run_separate_once(embedder, separator)
        return ok, msg

    if SEPARATE_CKPT.exists():
        ok_real, msg_real, _ = _run_separate_once(embedder, None)
        if not ok_real:
            return False, f'real Conv-TasNet path failed: {msg_real}'

    ok_det, msg_det, names = _run_separate_once(embedder, _deterministic_splitter)
    if not ok_det:
        return False, msg_det
    if set(names) != {'alice', 'bob'}:
        return False, f'expected identified names {{"alice", "bob"}}, got {names}'
    return True, 'ok'


def evaluate_flow(separator=None) -> dict:
    """Verify all 7 multi-step VoiceAgent state machines under sandbox_cohort."""
    embedder = ToneEmbedder()
    checks: list[dict] = []
    flows = [
        ('revoke_confirm_and_interrupts', lambda: _flow_revoke_confirm(embedder)),
        ('lockout_and_reset', lambda: _flow_lockout_and_reset(embedder)),
        ('enroll_quality_gates', lambda: _flow_enroll_quality(embedder)),
        ('key_rotation_invariance', lambda: _flow_key_rotation(embedder)),
        ('drift_detection', lambda: _flow_drift_detection(embedder)),
        ('cross_model_isolation', lambda: _flow_cross_model_isolation(embedder)),
        ('separate_and_identify_flow', lambda: _flow_separate_and_identify(embedder, separator=separator)),
    ]

    with sandbox_cohort(embedder):
        for check_id, fn in flows:
            passed, note = fn()
            checks.append({'id': check_id, 'passed': bool(passed), 'note': note})

    total = len(checks)
    passed_count = sum(1 for c in checks if c['passed'])
    return {
        'passed': passed_count == total,
        'total': total,
        'passed_count': passed_count,
        'checks': checks,
    }
