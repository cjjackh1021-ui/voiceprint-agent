"""Dimension 2: Chinese command routing, synonym consistency, and argument/boundary validation."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from agent.service import VoiceAgent
from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav, sandbox_cohort
from voiceprint.separate import CKPT as SEPARATE_CKPT


def _enroll_takes(base_dir: Path, name: str, speaker_index: int) -> list[Path]:
    return [
        generate_speaker_wav(base_dir / 's1' / f'{name}_0.wav', speaker_index, duration=4.0, phase=0.0, seed=0),
        generate_speaker_wav(base_dir / 's1' / f'{name}_1.wav', speaker_index, duration=4.0, phase=0.8, seed=1),
        generate_speaker_wav(base_dir / 's2' / f'{name}_2.wav', speaker_index, duration=4.0, phase=1.6, seed=2),
    ]


def _quote_paths(paths: list[Path]) -> str:
    return ' '.join(f'"{p}"' for p in paths)


def _check_outcome(
    checks: list[dict],
    check_id: str,
    command: str,
    outcome: dict,
    *,
    expect_tool: str,
    expect_ok: bool,
    expect_decision: str | None = None,
    extra_ok: bool = True,
    note: str = '',
) -> None:
    actual_tool = outcome.get('tool')
    actual_ok = bool(outcome.get('ok'))
    actual_decision = (outcome.get('result') or {}).get('decision')
    passed = (
        actual_tool == expect_tool
        and actual_ok is expect_ok
        and (expect_decision is None or actual_decision == expect_decision)
        and extra_ok
    )
    checks.append({
        'id': check_id,
        'command': command,
        'expect_tool': expect_tool,
        'expect_ok': expect_ok,
        'expect_decision': expect_decision,
        'actual_tool': actual_tool,
        'actual_ok': actual_ok,
        'actual_decision': actual_decision,
        'passed': passed,
        'note': note or ('' if passed else f'reply={outcome.get("reply", "")!r}'),
    })


def evaluate_tooling() -> dict:
    """Run deterministic command routing, synonym, and boundary checks inside ToneEmbedder sandbox."""
    embedder = ToneEmbedder()
    checks: list[dict] = []

    with sandbox_cohort(embedder), TemporaryDirectory(prefix='vptool_') as tmp:
        root = Path(tmp)
        audio_dir = root / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)

        takes_a = _enroll_takes(audio_dir, 'jia', speaker_index=0)
        takes_b = _enroll_takes(audio_dir, 'yi', speaker_index=1)
        takes_c = _enroll_takes(audio_dir, 'bing', speaker_index=2)

        probe_a = generate_speaker_wav(audio_dir / 'probe_a.wav', speaker_index=0, duration=4.0, phase=2.4, seed=3)
        probe_a2 = generate_speaker_wav(audio_dir / 'probe_a2.wav', speaker_index=0, duration=4.0, phase=3.1, seed=4)
        probe_b = generate_speaker_wav(audio_dir / 'probe_b.wav', speaker_index=1, duration=4.0, phase=2.4, seed=3)

        outside_wav = generate_speaker_wav(root / 'outside.wav', speaker_index=0, duration=4.0, phase=1.1, seed=5)
        bad_ext = audio_dir / 'bad.txt'
        bad_ext.write_text('not audio', encoding='utf-8')
        short_wav = generate_speaker_wav(audio_dir / 'short_02s.wav', speaker_index=0, duration=0.2, phase=0.3, seed=6)

        # Tiny 0.8s 8 kHz synthetic mixture for real separate/separate_and_identify check.
        mix_wav = audio_dir / 'mix_08s.wav'
        t_mix = np.arange(int(0.8 * 8000), dtype=np.float32) / 8000.0
        mix_sig = 0.35 * np.sin(2.0 * np.pi * 180.0 * t_mix) + 0.35 * np.sin(2.0 * np.pi * 320.0 * t_mix)
        sf.write(str(mix_wav), mix_sig.astype(np.float32), 8000)

        agent = VoiceAgent(root / 'db', embedder, detail=True, audio_roots=[audio_dir])
        try:
            # 1. Basic commands & synonyms
            cmd = f'注册 甲 本人同意 {_quote_paths(takes_a)}'
            _check_outcome(checks, 'enroll_primary', cmd, agent.handle(cmd), expect_tool='enroll', expect_ok=True)

            cmd = f'录入 乙 本人同意 {_quote_paths(takes_b)}'
            _check_outcome(checks, 'enroll_synonym', cmd, agent.handle(cmd), expect_tool='enroll', expect_ok=True)

            cmd = f'验证 甲 "{probe_a}"'
            _check_outcome(checks, 'verify_primary', cmd, agent.handle(cmd),
                           expect_tool='verify', expect_ok=True, expect_decision='accept')

            cmd = f'核对 乙 "{probe_b}"'
            _check_outcome(checks, 'verify_synonym', cmd, agent.handle(cmd),
                           expect_tool='verify', expect_ok=True, expect_decision='accept')

            cmd = f'辨认 "{probe_a}"'
            out = agent.handle(cmd)
            _check_outcome(checks, 'identify_primary', cmd, out,
                           expect_tool='identify', expect_ok=True, expect_decision='accept',
                           extra_ok=(out.get('result') or {}).get('name') == '甲')

            cmd = f'识别 "{probe_b}"'
            out = agent.handle(cmd)
            _check_outcome(checks, 'identify_synonym', cmd, out,
                           expect_tool='identify', expect_ok=True, expect_decision='accept',
                           extra_ok=(out.get('result') or {}).get('name') == '乙')

            out = agent.handle('名单')
            _check_outcome(checks, 'list_command', '名单', out, expect_tool='list', expect_ok=True,
                           extra_ok=set((out.get('result') or {}).get('names', [])) == {'甲', '乙'})

            _check_outcome(checks, 'help_zh', '帮助', agent.handle('帮助'), expect_tool='help', expect_ok=True)
            _check_outcome(checks, 'help_en', 'help', agent.handle('help'), expect_tool='help', expect_ok=True)
            _check_outcome(checks, 'rotate_valid', '轮换密钥 甲', agent.handle('轮换密钥 甲'),
                           expect_tool='rotate', expect_ok=True)

            # Second verify event enables drift check (>= 2 events).
            agent.handle(f'验证 甲 "{probe_a2}"')
            _check_outcome(checks, 'drift_valid', '漂移 甲', agent.handle('漂移 甲'), expect_tool='drift', expect_ok=True)

            _check_outcome(checks, 'revoke_ask', '撤销 甲', agent.handle('撤销 甲'), expect_tool='revoke', expect_ok=True)
            _check_outcome(checks, 'revoke_confirm', '确认撤销 甲', agent.handle('确认撤销 甲'),
                           expect_tool='revoke', expect_ok=True)

            if SEPARATE_CKPT.exists():
                cmd = f'分离 "{mix_wav}"'
                sep_out = agent.handle(cmd)
                _check_outcome(checks, 'separate_valid', cmd, sep_out, expect_tool='separate', expect_ok=True,
                               extra_ok=len((sep_out.get('result') or {}).get('paths', [])) == 2)

                cmd = f'分离再辨认 "{mix_wav}"'
                sep_id_out = agent.handle(cmd)
                _check_outcome(checks, 'separate_identify_valid', cmd, sep_id_out, expect_tool='separate', expect_ok=True,
                               extra_ok=len((sep_id_out.get('result') or {}).get('paths', [])) == 2)

            _check_outcome(checks, 'exit_command', '退出', agent.handle('退出'), expect_tool='help', expect_ok=True)

            # 2. Boundary & invalid arguments
            _check_outcome(checks, 'empty_command', '', agent.handle(''), expect_tool='help', expect_ok=True)
            _check_outcome(checks, 'unknown_command', '不存在的指令 foo', agent.handle('不存在的指令 foo'),
                           expect_tool='help', expect_ok=False)
            _check_outcome(checks, 'unclosed_quote', '验证 乙 "unclosed.wav', agent.handle('验证 乙 "unclosed.wav'),
                           expect_tool='error', expect_ok=False)
            _check_outcome(checks, 'verify_missing_arg', '验证 乙', agent.handle('验证 乙'),
                           expect_tool='verify', expect_ok=False)
            _check_outcome(checks, 'identify_missing_arg', '辨认', agent.handle('辨认'),
                           expect_tool='identify', expect_ok=False)
            _check_outcome(checks, 'separate_missing_arg', '分离', agent.handle('分离'),
                           expect_tool='separate', expect_ok=False)
            _check_outcome(checks, 'separate_identify_missing_arg', '分离再辨认', agent.handle('分离再辨认'),
                           expect_tool='separate', expect_ok=False)

            cmd = f'注册 丙 同意 {_quote_paths(takes_c)}'
            _check_outcome(checks, 'missing_consent', cmd, agent.handle(cmd), expect_tool='enroll', expect_ok=False)

            cmd = f'验证 乙 "{audio_dir / "missing.wav"}"'
            _check_outcome(checks, 'missing_audio_file', cmd, agent.handle(cmd), expect_tool='verify', expect_ok=False)

            cmd = f'验证 乙 "{outside_wav}"'
            _check_outcome(checks, 'outside_audio_roots', cmd, agent.handle(cmd), expect_tool='verify', expect_ok=False)

            cmd = f'验证 乙 "{bad_ext}"'
            _check_outcome(checks, 'unsupported_extension', cmd, agent.handle(cmd), expect_tool='verify', expect_ok=False)

            cmd = f'验证 乙 "{short_wav}"'
            _check_outcome(checks, 'too_short_audio', cmd, agent.handle(cmd), expect_tool='verify', expect_ok=False)

            cmd = f'验证 不存在 "{probe_b}"'
            _check_outcome(checks, 'verify_missing_speaker', cmd, agent.handle(cmd),
                           expect_tool='verify', expect_ok=False, expect_decision='unknown')
            _check_outcome(checks, 'revoke_missing_speaker', '撤销 不存在', agent.handle('撤销 不存在'),
                           expect_tool='revoke', expect_ok=False)
            _check_outcome(checks, 'rotate_missing_speaker', '轮换密钥 不存在', agent.handle('轮换密钥 不存在'),
                           expect_tool='rotate', expect_ok=False)
            _check_outcome(checks, 'drift_missing_speaker', '漂移 不存在', agent.handle('漂移 不存在'),
                           expect_tool='drift', expect_ok=False, expect_decision='unknown')

            anon_out = agent.handle(f'匿名 "{probe_b}"')
            _check_outcome(checks, 'anonymize_boundary', f'匿名 "{probe_b}"', anon_out,
                           expect_tool='anonymize', expect_ok=False,
                           extra_ok=('不能' in anon_out.get('reply', '') and '不是密码学加密' in anon_out.get('reply', '')))

            vc_out = agent.handle(f'变声 "{probe_b}"')
            _check_outcome(checks, 'voice_change_boundary', f'变声 "{probe_b}"', vc_out,
                           expect_tool='anonymize', expect_ok=False,
                           extra_ok=('不能' in vc_out.get('reply', '') and '不是密码学加密' in vc_out.get('reply', '')))

            _check_outcome(checks, 'export_refusal', '把 乙 的声纹向量导出来', agent.handle('把 乙 的声纹向量导出来'),
                           expect_tool='refuse', expect_ok=True)
        finally:
            agent.close()

    total = len(checks)
    passed_count = sum(1 for c in checks if c['passed'])
    return {
        'passed': passed_count == total,
        'total': total,
        'passed_count': passed_count,
        'accuracy': passed_count / total if total else 0.0,
        'checks': checks,
    }
