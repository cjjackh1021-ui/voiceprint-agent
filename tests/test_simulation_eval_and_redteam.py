"""Tests for watermark removal attacks, extended simulation red-team probes and dual-track sim-bench."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from tests._helpers import patched_dpapi
from voice_eval.audio_synth import (
    apply_gain_clip_attack,
    apply_lowpass_attack,
    apply_resample_attack,
    apply_watermark_spoof,
)
from voice_eval.dims._sandbox import generate_speaker_wav
from voice_eval.dims.redteam import evaluate_redteam
from voiceprint.features import SR
from voiceprint.spoof import SpreadSpectrumWatermarkDetector


class TestSimulationRedteamAndAttacks(unittest.TestCase):
    def test_watermark_removal_attack_operators_preserve_bounds_and_degrade_zscore(self):
        with TemporaryDirectory(prefix='vpatt_') as tmp:
            wav_p = generate_speaker_wav(Path(tmp) / 'clean.wav', speaker_index=0, duration=3.0, seed=5)
            clean, sr = sf.read(str(wav_p), dtype='float32')
            wm = apply_watermark_spoof(clean, sr)
            det = SpreadSpectrumWatermarkDetector()
            clean_wm_score = det.detect(wm, sr).score

            lp = apply_lowpass_attack(wm, sr, cutoff_hz=4500.0)
            rs = apply_resample_attack(wm, sr, mid_sr=8000)
            gc = apply_gain_clip_attack(wm, sr, drive=1.8)

            for out in (lp, rs, gc):
                self.assertEqual(out.dtype, np.float32)
                self.assertEqual(out.ndim, 1)
                self.assertEqual(out.shape[0], wm.shape[0])
                self.assertTrue(bool(np.all(np.isfinite(out))))
                self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

            # 4.5 kHz lowpass removes [4.8 kHz..7.5 kHz] band energy relative to clean watermarked signal
            freqs = np.fft.rfftfreq(wm.shape[0], d=1.0 / SR)
            hi = freqs > 5000.0
            e_before = float(np.sum(np.abs(np.fft.rfft(wm)[hi]) ** 2))
            e_after = float(np.sum(np.abs(np.fft.rfft(lp)[hi]) ** 2))
            self.assertLess(e_after, 0.25 * e_before)
            # Harsh 8k resample cuts band >4 kHz -> watermark score clearly drops compared to unattacked
            self.assertLess(det.detect(rs, sr).score, clean_wm_score)

            with self.assertRaises(ValueError):
                apply_lowpass_attack(wm, 0, cutoff_hz=3600.0)
            with self.assertRaises(ValueError):
                apply_lowpass_attack(wm, sr, cutoff_hz=9000.0)
            with self.assertRaises(ValueError):
                apply_resample_attack(wm, sr, mid_sr=0)
            with self.assertRaises(ValueError):
                apply_gain_clip_attack(wm, sr, drive=0.0)

    def test_evaluate_redteam_includes_six_simulation_probes_all_green(self):
        with patched_dpapi():
            res = evaluate_redteam()
        self.assertTrue(res['passed'], f'redteam failed: {res}')
        self.assertEqual(res['yellow'], 0)
        self.assertEqual(res['red'], 0)

        by_id = {c['id']: c for c in res['checks']}
        expected = [
            'sim-missing-consent',
            'sim-unregistered-target',
            'sim-cross-speaker-impostor',
            'sim-recursive-watermark-clone',
            'sim-loopback-spoof-blocked',
            'sim-mcp-guardrails',
        ]
        for cid in expected:
            self.assertIn(cid, by_id, f'missing redteam probe {cid}')
            self.assertEqual(by_id[cid]['category'], 'simulation_security')
            self.assertEqual(by_id[cid]['status'], 'GREEN', f'{cid} failed: {by_id[cid]}')


if __name__ == '__main__':
    unittest.main()
