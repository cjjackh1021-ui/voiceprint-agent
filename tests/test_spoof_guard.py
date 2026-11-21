"""Tests for pluggable anti-spoofing watermark detector, synthetic spoof generators, and VoiceAgent guardrail."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from agent.service import VoiceAgent
from tests._helpers import patched_dpapi
from voice_eval.audio_synth import add_white_noise, apply_vocoder_replay_spoof, apply_watermark_spoof
from voice_eval.dims._sandbox import (
    ToneEmbedder,
    generate_speaker_wav,
    quote_paths,
    sandbox_cohort,
    sandbox_enroll_takes,
)
from voice_eval.dims.response import check_reply_payload
from voiceprint.spoof import (
    CompositeSpoofDetector,
    DEFAULT_WATERMARK_PAYLOAD,
    SpoofDetector,
    SpoofResult,
    SpreadSpectrumWatermarkDetector,
    embed_spread_watermark,
)


class _AlwaysHitDetector(SpoofDetector):
    def detect(self, wav: np.ndarray, sr: int) -> SpoofResult:
        return SpoofResult(is_spoof=True, score=1.0, reason='custom_hit', detector='stub', metadata={})


class TestSpoofGuard(unittest.TestCase):
    def test_spread_watermark_roundtrip_and_clean_zero_false_positive(self):
        detector = SpreadSpectrumWatermarkDetector()
        with TemporaryDirectory(prefix='vpspoof_wm_') as tmp:
            root = Path(tmp)
            for idx in range(5):
                p = generate_speaker_wav(root / f'spk_{idx}.wav', speaker_index=idx, duration=2.5, seed=idx)
                wav, sr = sf.read(str(p), dtype='float32')
                clean_res = detector.detect(wav, sr)
                self.assertFalse(clean_res.is_spoof, f'false positive on clean speaker {idx}: {clean_res}')

                wm = embed_spread_watermark(wav, sr, payload=DEFAULT_WATERMARK_PAYLOAD, strength=0.02)
                wm_res = detector.detect(wm, sr)
                self.assertTrue(wm_res.is_spoof, f'missed watermark on speaker {idx}: {wm_res}')
                self.assertEqual(wm_res.reason, 'watermark_detected')
                self.assertEqual(wm_res.metadata.get('payload'), DEFAULT_WATERMARK_PAYLOAD)
                self.assertGreaterEqual(wm_res.score, 4.5)

                # Mild 20 dB noise should still preserve detection.
                wm_noisy = add_white_noise(wm, snr_db=20.0, seed=100 + idx)
                noisy_res = detector.detect(wm_noisy, sr)
                self.assertTrue(noisy_res.is_spoof, f'missed watermark under 20 dB noise: {noisy_res}')

            # Pure white noise must not false-trigger.
            rng = np.random.default_rng(999)
            noise = (rng.standard_normal(32000) * 0.15).astype(np.float32)
            self.assertFalse(detector.detect(noise, 16000).is_spoof)

    def test_composite_spoof_detector_short_circuits_on_hit(self):
        comp = CompositeSpoofDetector([SpreadSpectrumWatermarkDetector(), _AlwaysHitDetector()])
        sig = np.sin(2.0 * np.pi * 440.0 * np.arange(16000, dtype=np.float32) / 16000.0) * 0.2
        res = comp.detect(sig, 16000)
        self.assertTrue(res.is_spoof)
        self.assertEqual(res.detector, 'stub')
        self.assertEqual(res.reason, 'custom_hit')

    def test_audio_synth_spoof_generators_deterministic(self):
        t = np.arange(32000, dtype=np.float32) / 16000.0
        clean = (0.25 * np.sin(2.0 * np.pi * 220.0 * t) + 0.15 * np.sin(2.0 * np.pi * 660.0 * t)).astype(np.float32)

        wm1 = apply_watermark_spoof(clean, 16000)
        wm2 = apply_watermark_spoof(clean, 16000)
        self.assertEqual(wm1.shape, clean.shape)
        self.assertEqual(wm1.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(wm1)))
        self.assertTrue(np.array_equal(wm1, wm2))
        self.assertLessEqual(float(np.max(np.abs(wm1))), 1.0)

        rep1 = apply_vocoder_replay_spoof(clean, 16000, seed=7)
        rep2 = apply_vocoder_replay_spoof(clean, 16000, seed=7)
        self.assertEqual(rep1.shape, clean.shape)
        self.assertEqual(rep1.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(rep1)))
        self.assertTrue(np.array_equal(rep1, rep2))
        self.assertLessEqual(float(np.max(np.abs(rep1))), 1.0)
        self.assertGreater(float(np.mean(np.abs(rep1 - clean))), 0.01)

    def test_voice_agent_spoof_guard_blocks_watermarked_clone_without_leaking_or_locking(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpspoof_ag_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            allowed.mkdir()
            takes = sandbox_enroll_takes(allowed, 'jia', speaker_index=0)
            clean_probe = generate_speaker_wav(allowed / 'probe_clean.wav', speaker_index=0, duration=3.0, phase=1.9, seed=7)
            wav, sr = sf.read(str(clean_probe), dtype='float32')
            wm_probe = allowed / 'probe_wm.wav'
            sf.write(str(wm_probe), apply_watermark_spoof(wav, sr), sr)

            agent = VoiceAgent(
                root / 'db',
                embedder,
                detail=False,
                audio_roots=[allowed],
                spoof_detector=SpreadSpectrumWatermarkDetector(),
            )
            try:
                # Watermarked enroll must be rejected
                bad_enr = agent.handle(f'注册 假人 本人同意 {quote_paths([wm_probe, takes[1], takes[2]])}')
                self.assertFalse(bad_enr['ok'])
                self.assertEqual(bad_enr['result'].get('reason'), 'spoof_detected')

                # Clean enroll succeeds
                ok_enr = agent.handle(f'注册 甲 本人同意 {quote_paths(takes)}')
                self.assertTrue(ok_enr['ok'])

                # 6 consecutive watermarked verify attempts: all rejected, no score leak, no events written, no lockout!
                for _ in range(6):
                    v_wm = agent.handle(f'验证 甲 "{wm_probe.as_posix()}"')
                    self.assertFalse(v_wm['ok'])
                    self.assertEqual(v_wm['result'].get('decision'), 'reject')
                    self.assertEqual(v_wm['result'].get('reason'), 'spoof_detected')
                    self.assertNotIn('spoof_score', v_wm['result'])
                    self.assertNotIn('score', v_wm['result'])
                    ok_reply, msg_reply = check_reply_payload(v_wm, 'reject', detail=False, speaker_name='甲')
                    self.assertTrue(ok_reply, msg_reply)

                i_wm = agent.handle(f'辨认 "{wm_probe.as_posix()}"')
                self.assertFalse(i_wm['ok'])
                self.assertEqual(i_wm['result'].get('decision'), 'reject')
                self.assertEqual(i_wm['result'].get('reason'), 'spoof_detected')
                self.assertNotIn('spoof_score', i_wm['result'])

                ev_count = agent.store.db.execute('select count(*) from events').fetchone()[0]
                self.assertEqual(ev_count, 0)

                # Clean genuine speaker STILL accepted (not locked out by spoof DoS)
                v_clean = agent.handle(f'验证 甲 "{clean_probe.as_posix()}"')
                self.assertTrue(v_clean['ok'])
                self.assertEqual(v_clean['result'].get('decision'), 'accept')

                i_clean = agent.handle(f'辨认 "{clean_probe.as_posix()}"')
                self.assertTrue(i_clean['ok'])
                self.assertEqual(i_clean['result'].get('decision'), 'accept')
                self.assertEqual(i_clean['result'].get('name'), '甲')
            finally:
                agent.close()


if __name__ == '__main__':
    unittest.main()
