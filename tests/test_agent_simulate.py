"""Tests for VoiceAgent controllable simulation command, security gates and Self-Refine loop."""
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from agent import cli as agent_cli
from agent.service import LOCK_MINUTES, VoiceAgent
from tests._helpers import patched_dpapi
from voice_eval.audio_synth import apply_watermark_spoof
from voice_eval.dims._sandbox import (
    ToneEmbedder,
    generate_speaker_wav,
    quote_paths,
    sandbox_cohort,
    sandbox_enroll,
)
from voice_eval.dims.response import FORBIDDEN_NON_DETAIL_KEYS, FORBIDDEN_NON_DETAIL_TERMS
from voiceprint.features import SR
from voiceprint.spoof import DEFAULT_WATERMARK_PAYLOAD, SpreadSpectrumWatermarkDetector
from voiceprint.synth import SimulationRequest, SimulationResult, VoiceSynthesizer


class _FakeSpeakerSynthesizer(VoiceSynthesizer):
    """Test synthesizer simulating controllable convergence behavior."""

    name = 'fake_speaker'
    supported_modes = ('audio', 'text')

    def __init__(self, target_wav_path: Path, mode_script: str = 'clip_then_converge'):
        self.target_wav, self.target_sr = sf.read(str(target_wav_path), dtype='float32')
        self.mode_script = mode_script
        self.requests: list[SimulationRequest] = []

    def synthesize(self, req: SimulationRequest) -> SimulationResult:
        self.requests.append(req)
        if self.mode_script == 'always_mismatch':
            # Deliberately return a clean 1.5s waveform belonging to target_wav directly -> actually if
            # initialized with OTHER speaker wav, it never reaches target SIM.
            return SimulationResult(wav=self.target_wav.copy(), sr=self.target_sr, backend=self.name, metadata={})
        if self.mode_script == 'clip_then_converge':
            if req.attempt_index == 1:
                # Attempt 1: deliberate heavy clipping (> 25% clipped samples) + low blend -> triggers refine
                clipped = np.clip(self.target_wav * 8.0, -1.0, 1.0).astype(np.float32)
                return SimulationResult(wav=clipped, sr=self.target_sr, backend=self.name, metadata={})
            scaled = (self.target_wav * min(0.75, req.gain_scale)).astype(np.float32)
            return SimulationResult(wav=scaled, sr=self.target_sr, backend=self.name, metadata={})
        return SimulationResult(wav=self.target_wav.copy(), sr=self.target_sr, backend=self.name, metadata={})


class _TextOnlyMockSynthesizer(VoiceSynthesizer):
    name = 'text_only_mock'
    supported_modes = ('text',)

    def synthesize(self, req: SimulationRequest) -> SimulationResult:
        raise AssertionError('Should be blocked at pre-gate before calling synthesize')


class TestAgentSimulate(unittest.TestCase):
    def test_simulate_precondition_gates_enforce_consent_ownership_and_anti_recursive_watermark(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpsim_gates_') as tmp:
            root = Path(tmp)
            audio_dir = root / 'audio'
            audio_dir.mkdir()
            sandbox_enroll_a = sandbox_enroll(
                agent := VoiceAgent(root / 'db', embedder, detail=True, audio_roots=[audio_dir]),
                '甲',
                audio_dir / 'enr_a',
                speaker_index=0,
            )
            try:
                sandbox_enroll(agent, '乙', audio_dir / 'enr_b', speaker_index=1)
                ref_a = sandbox_enroll_a[0]
                ref_b = generate_speaker_wav(audio_dir / 'ref_b.wav', speaker_index=1, duration=2.5, seed=9)
                src_b = generate_speaker_wav(audio_dir / 'src_b.wav', speaker_index=1, duration=2.5, seed=10)

                # 1. Missing exact '本人授权' consent keyword
                no_consent = agent.handle(f'模拟 甲 同意 "{ref_a}" 文本 "测试语音"')
                self.assertFalse(no_consent['ok'])
                self.assertEqual(no_consent['tool'], 'simulate')
                self.assertEqual(no_consent['result'].get('reason'), 'consent_required')
                self.assertFalse(any(agent.simulated_dir.glob('*.wav')) if agent.simulated_dir.exists() else False)

                # 2. Unregistered speaker
                unknown = agent.handle(f'模拟 路人 本人授权 "{ref_a}" 文本 "测试语音"')
                self.assertFalse(unknown['ok'])
                self.assertEqual(unknown['result'].get('reason'), 'unknown')

                # 3. Cross-speaker impostor reference audio
                mismatch = agent.handle(f'模拟 甲 本人授权 "{ref_b}" 源音频 "{src_b}"')
                self.assertFalse(mismatch['ok'])
                self.assertEqual(mismatch['result'].get('reason'), 'reference_speaker_mismatch')

                # 4. Anti-recursive cloning of watermarked audio even when spoof_detector=None
                wm_ref_path = audio_dir / 'wm_ref_a.wav'
                raw_a, sr_a = sf.read(str(ref_a), dtype='float32')
                sf.write(str(wm_ref_path), apply_watermark_spoof(raw_a, sr_a), sr_a)
                recursive = agent.handle(f'模拟 甲 本人授权 "{wm_ref_path}" 文本 "二次克隆洗白"')
                self.assertFalse(recursive['ok'])
                self.assertEqual(recursive['result'].get('reason'), 'watermark_detected')
            finally:
                agent.close()

            # 5. Unsupported mode rejected before calling synthesizer
            ag_text_only = VoiceAgent(
                root / 'db_txtonly',
                embedder,
                detail=True,
                audio_roots=[audio_dir],
                synthesizer=_TextOnlyMockSynthesizer(),
            )
            try:
                sandbox_enroll(ag_text_only, '甲', audio_dir / 'enr_txtonly', speaker_index=0)
                bad_mode = ag_text_only.handle(f'模拟 甲 本人授权 "{sandbox_enroll_a[0]}" 源音频 "{src_b}"')
                self.assertFalse(bad_mode['ok'])
                self.assertEqual(bad_mode['result'].get('reason'), 'unsupported_mode')
            finally:
                ag_text_only.close()

    def test_simulate_mismatch_and_failed_retries_do_not_pollute_lockout_events(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpsim_lock_') as tmp:
            root = Path(tmp)
            audio_dir = root / 'audio'
            audio_dir.mkdir()
            other_wav = generate_speaker_wav(audio_dir / 'other.wav', speaker_index=2, duration=2.5, seed=11)
            fail_syn = _FakeSpeakerSynthesizer(other_wav, mode_script='always_mismatch')
            agent = VoiceAgent(
                root / 'db',
                embedder,
                detail=True,
                audio_roots=[audio_dir],
                synthesizer=fail_syn,
                max_sim_attempts=3,
            )
            try:
                takes_a = sandbox_enroll(agent, '甲', audio_dir / 'enr_a', speaker_index=0)
                probe_a = generate_speaker_wav(audio_dir / 'probe_a.wav', speaker_index=0, duration=3.0, seed=12)

                for _ in range(4):
                    r1 = agent.handle(f'模拟 甲 本人授权 "{other_wav}" 文本 "冒充"')
                    self.assertFalse(r1['ok'])
                    r2 = agent.handle(f'模拟 甲 本人授权 "{takes_a[0]}" 文本 "始终无法收敛"')
                    self.assertFalse(r2['ok'])
                    self.assertEqual(r2['result'].get('reason'), 'similarity_not_converged')

                self.assertEqual(agent.store.events('甲'), [])
                self.assertEqual(agent.store.recent_failures('甲', LOCK_MINUTES), 0)

                ver = agent.handle(f'验证 甲 "{probe_a}"')
                self.assertTrue(ver['ok'])
                self.assertEqual(ver['result'].get('decision'), 'accept')
            finally:
                agent.close()

    def test_simulate_sanitizes_speaker_name_in_output_filename(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpsim_name_') as tmp:
            root = Path(tmp)
            audio_dir = root / 'audio'
            audio_dir.mkdir()
            evil_name = '../evil\\spk'
            from voice_eval.dims._sandbox import sandbox_enroll_takes
            takes = sandbox_enroll_takes(audio_dir / 'enr_evil', 'safe_take', speaker_index=0)
            agent = VoiceAgent(root / 'db', embedder, detail=False, audio_roots=[audio_dir])
            try:
                enr = agent.handle(f'注册 "{evil_name}" 本人同意 {quote_paths(takes)}')
                self.assertTrue(enr['ok'], enr)
                src = generate_speaker_wav(audio_dir / 'src.wav', speaker_index=1, duration=2.5, seed=21)
                out = agent.handle(f'模拟 "{evil_name}" 本人授权 "{takes[0]}" 源音频 "{src}"')
                self.assertTrue(out['ok'], out)
                saved = Path(out['result']['output_path']).resolve()
                sim_dir = agent.simulated_dir.resolve()
                self.assertEqual(saved.parent, sim_dir)
                self.assertNotIn('..', saved.name)
                self.assertNotIn('/', saved.name)
                self.assertNotIn('\\', saved.name)
                self.assertTrue(saved.is_file())
            finally:
                agent.close()

    def test_simulate_self_refine_retries_improve_and_converge_with_forced_watermark(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpsim_refine_') as tmp:
            root = Path(tmp)
            audio_dir = root / 'audio'
            audio_dir.mkdir()
            target_sample = generate_speaker_wav(audio_dir / 'target_ref.wav', speaker_index=0, duration=3.0, seed=31)
            other_sample = generate_speaker_wav(audio_dir / 'other.wav', speaker_index=1, duration=3.0, seed=32)

            syn_converge = _FakeSpeakerSynthesizer(target_sample, mode_script='clip_then_converge')
            agent = VoiceAgent(
                root / 'db1',
                embedder,
                detail=True,
                audio_roots=[audio_dir],
                synthesizer=syn_converge,
                max_sim_attempts=3,
            )
            try:
                takes_a = sandbox_enroll(agent, '甲', audio_dir / 'enr_a', speaker_index=0)
                out = agent.handle(f'模拟 甲 本人授权 "{takes_a[0]}" 文本 "声纹自检重试验证" 参考文本 "参考短句"')
                self.assertTrue(out['ok'], out)
                self.assertTrue(out['result']['converged'])
                self.assertEqual(out['result']['attempts'], 2)
                # Verify attempt 2 reduced gain_scale because attempt 1 clipped
                self.assertLess(syn_converge.requests[1].gain_scale, syn_converge.requests[0].gain_scale)

                saved = Path(out['result']['output_path'])
                self.assertTrue(saved.is_file())
                wav, sr = sf.read(str(saved), dtype='float32')
                wm = SpreadSpectrumWatermarkDetector().detect(wav, sr)
                self.assertTrue(wm.is_spoof)
                self.assertEqual(wm.metadata.get('payload'), DEFAULT_WATERMARK_PAYLOAD)
            finally:
                agent.close()

            # Non-converged simulation must NOT persist output file, but collect_best_candidate returns candidate wav
            syn_fail = _FakeSpeakerSynthesizer(other_sample, mode_script='always_mismatch')
            agent2 = VoiceAgent(
                root / 'db2',
                embedder,
                detail=True,
                audio_roots=[audio_dir],
                synthesizer=syn_fail,
                max_sim_attempts=2,
            )
            try:
                takes_a2 = sandbox_enroll(agent2, '甲', audio_dir / 'enr_a2', speaker_index=0)
                outcome, best_wav = agent2._run_simulate_core(
                    '甲',
                    consent=True,
                    ref_path=str(takes_a2[0]),
                    mode='text',
                    text='不收敛测试',
                    collect_best_candidate=True,
                )
                self.assertFalse(outcome['ok'])
                self.assertFalse(outcome['result']['converged'])
                self.assertEqual(outcome['result']['reason'], 'similarity_not_converged')
                self.assertIsNotNone(best_wav)
                self.assertFalse(any(agent2.simulated_dir.glob('*.wav')) if agent2.simulated_dir.exists() else False)
                wm_best = SpreadSpectrumWatermarkDetector().detect(best_wav, SR)
                self.assertTrue(wm_best.is_spoof)
            finally:
                agent2.close()

    def test_simulate_detail_false_redacts_scores_and_loopback_blocked_under_spoof_guard(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpsim_red_') as tmp:
            root = Path(tmp)
            audio_dir = root / 'audio'
            audio_dir.mkdir()
            agent = VoiceAgent(
                root / 'db',
                embedder,
                detail=False,
                audio_roots=[audio_dir],
                spoof_detector=SpreadSpectrumWatermarkDetector(),
            )
            try:
                takes_a = sandbox_enroll(agent, '甲', audio_dir / 'enr_a', speaker_index=0)
                src = generate_speaker_wav(audio_dir / 'src.wav', speaker_index=1, duration=2.5, seed=41)
                sim = agent.handle(f'模拟 甲 本人授权 "{takes_a[0]}" 源音频 "{src}"')
                self.assertTrue(sim['ok'], sim)

                res = sim['result']
                for k in FORBIDDEN_NON_DETAIL_KEYS:
                    self.assertNotIn(k, res)
                self.assertNotIn('attempt_history', res)
                reply_scrubbed = sim['reply'].replace(res['output_path'], '')
                for term in FORBIDDEN_NON_DETAIL_TERMS:
                    self.assertNotIn(term, reply_scrubbed)
                self.assertNotRegex(reply_scrubbed, r'-?\d+\.\d+')

                # Loopback verify and identify under audio_roots + spoof_detector must accept path and block spoof
                ver = agent.handle(f'验证 甲 "{res["output_path"]}"')
                self.assertFalse(ver['ok'])
                self.assertEqual(ver['result'].get('reason'), 'spoof_detected')

                ident = agent.handle(f'辨认 "{res["output_path"]}"')
                self.assertFalse(ident['ok'])
                self.assertEqual(ident['result'].get('reason'), 'spoof_detected')
            finally:
                agent.close()

    def test_cli_supports_synthesizer_flags_and_preserves_anonymize_stub(self):
        parser = agent_cli.build_arg_parser()
        default_args = parser.parse_args([])
        self.assertEqual(default_args.synthesizer, 'acoustic')
        self.assertEqual(default_args.max_sim_attempts, 3)

        off_args = parser.parse_args(['--synthesizer', 'none', '--spoof-guard', '--max-sim-attempts', '2'])
        self.assertEqual(off_args.synthesizer, 'none')
        self.assertTrue(off_args.spoof_guard)
        self.assertEqual(off_args.max_sim_attempts, 2)

        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpsim_anon_') as tmp:
            ag = VoiceAgent(Path(tmp) / 'db', embedder, detail=False)
            try:
                anon = ag.handle('匿名')
                self.assertFalse(anon['ok'])
                self.assertEqual(anon['tool'], 'anonymize')
            finally:
                ag.close()

    def test_simulate_always_clipping_accept_reports_poor_audio_quality(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpsim_clip_') as tmp:
            root = Path(tmp)
            audio_dir = root / 'audio'
            audio_dir.mkdir()
            target_sample = generate_speaker_wav(audio_dir / 'target_clean.wav', speaker_index=0, duration=2.0, seed=71)

            class _AlwaysClipTargetSynthesizer(VoiceSynthesizer):
                name = 'always_clip_target'
                supported_modes = ('text',)

                def synthesize(self, req: SimulationRequest) -> SimulationResult:
                    wav, sr = sf.read(str(target_sample), dtype='float32')
                    clipped = np.clip(wav * 8.0, -1.0, 1.0).astype(np.float32)
                    return SimulationResult(wav=clipped, sr=sr, backend=self.name, metadata={})

            agent = VoiceAgent(
                root / 'db',
                embedder,
                detail=False,
                audio_roots=[audio_dir],
                synthesizer=_AlwaysClipTargetSynthesizer(),
                max_sim_attempts=2,
            )
            try:
                takes_a = sandbox_enroll(agent, '甲', audio_dir / 'enr_a', speaker_index=0)
                out, cand = agent._run_simulate_core(
                    '甲',
                    consent=True,
                    ref_path=str(takes_a[0]),
                    mode='text',
                    text='削波归因测试',
                    collect_best_candidate=True,
                )
                self.assertFalse(out['ok'])
                self.assertEqual(out['result']['reason'], 'poor_audio_quality')
                self.assertIsNotNone(cand)
                self.assertFalse(any(agent.simulated_dir.glob('*.wav')) if agent.simulated_dir.exists() else False)
                for term in FORBIDDEN_NON_DETAIL_TERMS:
                    self.assertNotIn(term, out['reply'])
            finally:
                agent.close()


if __name__ == '__main__':
    unittest.main()
