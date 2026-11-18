"""Tests for Task 2: deterministic distortion synthesis and LibriSpeech v3 case builder."""
import inspect
import json
import shlex
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from tests._helpers import synthetic_librispeech_tree
from voice_eval.audio_synth import (
    SR,
    add_babble_noise,
    add_pink_noise,
    add_white_noise,
    apply_reverb,
    apply_telephone_channel,
    clip_duration,
    mix_two_speakers,
)
from voice_eval.cases import (
    CASE_VERSION,
    DURATIONS,
    NOISE_TYPES,
    RT60_LEVELS,
    SNR_LEVELS,
    _sample_babble_interferers,
    build,
    build_split,
    case_version,
    load,
)


def _measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy.astype(np.float64) - clean.astype(np.float64)
    p_sig = float(np.mean(np.square(clean.astype(np.float64))))
    p_noise = float(np.mean(np.square(noise)))
    return 10.0 * float(np.log10(p_sig / max(p_noise, 1e-16)))


class SynthAndCasesV3Tests(unittest.TestCase):
    def test_noise_snr_and_determinism(self):
        t = np.arange(int(3.0 * SR), dtype=np.float64) / SR
        clean = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

        for snr_db in SNR_LEVELS:
            w1 = add_white_noise(clean, snr_db=snr_db, seed=42)
            w2 = add_white_noise(clean, snr_db=snr_db, seed=42)
            w3 = add_white_noise(clean, snr_db=snr_db, seed=43)
            self.assertEqual(w1.shape, clean.shape)
            self.assertEqual(w1.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(w1)))
            self.assertLessEqual(float(np.max(np.abs(w1))), 1.0)
            self.assertTrue(np.array_equal(w1, w2))
            self.assertFalse(np.array_equal(w1, w3))
            self.assertAlmostEqual(_measured_snr_db(clean, w1), float(snr_db), delta=0.5)

            p1 = add_pink_noise(clean, snr_db=snr_db, seed=42)
            p2 = add_pink_noise(clean, snr_db=snr_db, seed=42)
            p3 = add_pink_noise(clean, snr_db=snr_db, seed=99)
            self.assertEqual(p1.shape, clean.shape)
            self.assertEqual(p1.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(p1)))
            self.assertLessEqual(float(np.max(np.abs(p1))), 1.0)
            self.assertTrue(np.array_equal(p1, p2))
            self.assertFalse(np.array_equal(p1, p3))
            self.assertAlmostEqual(_measured_snr_db(clean, p1), float(snr_db), delta=0.5)

        # Verify 1/f spectral slope on pink noise alone (zero input doesn't add noise; subtract clean on low-amp signal)
        pink_noisy = add_pink_noise(clean, snr_db=0.0, seed=7)
        pink_only = pink_noisy.astype(np.float64) - clean.astype(np.float64)
        spec = np.abs(np.fft.rfft(pink_only)) ** 2
        freqs = np.fft.rfftfreq(len(pink_only), d=1.0 / SR)
        low_power = float(np.mean(spec[(freqs >= 100.0) & (freqs <= 800.0)]))
        high_power = float(np.mean(spec[(freqs >= 4000.0) & (freqs <= 7500.0)]))
        self.assertGreater(low_power, high_power * 3.0)

        interferers = [
            (0.4 * np.sin(2.0 * np.pi * f * t + idx)).astype(np.float32)
            for idx, f in enumerate((210.0, 330.0, 580.0))
        ]
        for snr_db in SNR_LEVELS:
            b1 = add_babble_noise(clean, interferers, snr_db=snr_db, seed=11)
            b2 = add_babble_noise(clean, interferers, snr_db=snr_db, seed=11)
            b3 = add_babble_noise(clean, interferers, snr_db=snr_db, seed=12)
            self.assertEqual(b1.shape, clean.shape)
            self.assertEqual(b1.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(b1)))
            self.assertLessEqual(float(np.max(np.abs(b1))), 1.0)
            self.assertTrue(np.array_equal(b1, b2))
            self.assertFalse(np.array_equal(b1, b3))
            self.assertAlmostEqual(_measured_snr_db(clean, b1), float(snr_db), delta=0.5)

        # Edge cases: very short interferers (even 1-sample) and silent interferers never yield NaN
        short_interferers = [
            np.array([0.2, -0.3, 0.4], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        ]
        b_short = add_babble_noise(clean, short_interferers, snr_db=5.0, seed=5)
        self.assertEqual(b_short.shape, clean.shape)
        self.assertTrue(np.all(np.isfinite(b_short)))
        self.assertAlmostEqual(_measured_snr_db(clean, b_short), 5.0, delta=0.5)

        b_silent = add_babble_noise(clean, [np.zeros(500, dtype=np.float32)], snr_db=5.0, seed=5)
        self.assertEqual(b_silent.shape, clean.shape)
        self.assertTrue(np.all(np.isfinite(b_silent)))

    def test_duration_channel_reverb_and_mixture(self):
        t4 = np.arange(int(4.0 * SR), dtype=np.float64) / SR
        sig = (
            0.2 * np.sin(2.0 * np.pi * 220.0 * t4)
            + 0.1 * np.sin(2.0 * np.pi * 660.0 * t4)
            + 0.08 * np.sin(2.0 * np.pi * 5500.0 * t4)
        ).astype(np.float32)

        for sec in DURATIONS:
            clipped = clip_duration(sig, SR, sec)
            self.assertEqual(clipped.shape, (int(round(sec * SR)),))
            self.assertEqual(clipped.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(clipped)))
            if sec <= 4.0:
                self.assertTrue(np.array_equal(clipped, sig[:len(clipped)]))
            else:
                self.assertTrue(np.array_equal(clipped[:len(sig)], sig))
                self.assertTrue(np.all(clipped[len(sig):] == 0.0))

        tel_wav, tel_sr = apply_telephone_channel(sig, SR)
        self.assertEqual(tel_sr, 8000)
        self.assertEqual(tel_wav.dtype, np.float32)
        self.assertEqual(tel_wav.ndim, 1)
        self.assertEqual(len(tel_wav), int(round(len(sig) * 8000 / SR)))
        self.assertTrue(np.all(np.isfinite(tel_wav)))
        self.assertLessEqual(float(np.max(np.abs(tel_wav))), 1.0)
        # Quantized mu-law output should differ from linear signal and preserve fundamental energy
        tel_spec = np.abs(np.fft.rfft(tel_wav.astype(np.float64)))
        tel_freqs = np.fft.rfftfreq(len(tel_wav), d=1.0 / tel_sr)
        f220_idx = int(np.argmin(np.abs(tel_freqs - 220.0)))
        self.assertGreater(float(tel_spec[f220_idx]), 10.0)
        # Compare against smooth unquantized 8 kHz downsampled signal to verify mu-law quantization step distortion
        from scipy.signal import resample_poly
        clean_8k = resample_poly(sig.astype(np.float64), 1, 2).astype(np.float32)
        self.assertGreater(float(np.max(np.abs(tel_wav - clean_8k))), 1e-3)

        in_rms = float(np.sqrt(np.mean(np.square(sig.astype(np.float64)))))
        rev_03 = apply_reverb(sig, SR, rt60=0.3, seed=10)
        rev_06 = apply_reverb(sig, SR, rt60=0.6, seed=10)
        rev_06_again = apply_reverb(sig, SR, rt60=0.6, seed=10)
        self.assertEqual(rev_06.shape, sig.shape)
        self.assertEqual(rev_06.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(rev_06)))
        self.assertLessEqual(float(np.max(np.abs(rev_06))), 1.0)
        self.assertTrue(np.array_equal(rev_06, rev_06_again))
        self.assertFalse(np.array_equal(rev_03, rev_06))
        self.assertGreater(float(np.max(np.abs(rev_06 - sig))), 0.05)
        out_rms = float(np.sqrt(np.mean(np.square(rev_06.astype(np.float64)))))
        self.assertAlmostEqual(out_rms, in_rms, delta=in_rms * 0.05)

        t3 = np.arange(int(3.0 * SR), dtype=np.float64) / SR
        spk_a = (0.1 * np.sin(2.0 * np.pi * 300.0 * t3)).astype(np.float32)
        spk_b = (0.6 * np.sin(2.0 * np.pi * 900.0 * t4)).astype(np.float32)
        mix, mix_sr = mix_two_speakers(spk_a, SR, spk_b, SR)
        self.assertEqual(mix_sr, 8000)
        self.assertEqual(mix.dtype, np.float32)
        self.assertEqual(mix.ndim, 1)
        self.assertEqual(len(mix), int(round(3.0 * 8000)))
        self.assertTrue(np.all(np.isfinite(mix)))
        self.assertLessEqual(float(np.max(np.abs(mix))), 0.9 + 1e-6)
        mix_spec = np.abs(np.fft.rfft(mix.astype(np.float64))) ** 2
        mix_freqs = np.fft.rfftfreq(len(mix), d=1.0 / mix_sr)
        p_300 = float(np.sum(mix_spec[(mix_freqs >= 270.0) & (mix_freqs <= 330.0)]))
        p_900 = float(np.sum(mix_spec[(mix_freqs >= 870.0) & (mix_freqs <= 930.0)]))
        self.assertLess(abs(10.0 * np.log10(p_300 / p_900)), 0.5)

    def test_build_v3_split_isolation_and_counts(self):
        self.assertEqual(CASE_VERSION, 3)
        self.assertEqual(NOISE_TYPES, ('white', 'pink', 'babble'))
        self.assertEqual(SNR_LEVELS, (20, 10, 5, 0))
        self.assertEqual(DURATIONS, (1.0, 1.5, 3.0, 5.0))
        self.assertEqual(RT60_LEVELS, (0.3, 0.6))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(case_version(root / 'nonexistent'), 1)

            train_root, test_root = synthetic_librispeech_tree(root / 'LibriSpeech', n_train=4, n_test=40)
            case_dir = root / 'cases_a'
            summary = build(test_root=test_root, case_dir=case_dir)
            self.assertEqual(case_version(case_dir), 3)
            self.assertEqual(summary['validation_same_session_probes'], 0)
            self.assertEqual(summary['locked_same_session_probes'], 0)

            split_speakers = {}
            for split in ('validation', 'locked'):
                pub_path = case_dir / f'{split}.public.json'
                hid_path = case_dir / f'{split}.hidden.json'
                pub_text = pub_path.read_text(encoding='utf-8')
                pub_rows = json.loads(pub_text)
                hid_rows = json.loads(hid_path.read_text(encoding='utf-8'))
                self.assertEqual(len(pub_rows), len(hid_rows))

                for forbidden_key in ('hidden', 'expect', 'true_speaker'):
                    self.assertNotIn(forbidden_key, pub_text)
                for row in pub_rows:
                    self.assertEqual(set(row.keys()), {'id', 'suite', 'split', 'steps'})
                    for step in row['steps']:
                        self.assertNotIn(str(train_root), step)

                loaded = load(split, case_dir=case_dir)
                counts = Counter(c['suite'] for c in loaded)
                self.assertEqual(counts, {
                    'verify': 40,
                    'verify_impostor': 360,
                    'identify': 20,
                    'enroll_quality': 5,
                    'robustness_noise': 480,
                    'robustness_duration': 160,
                    'robustness_channel': 40,
                    'robustness_reverb': 80,
                    'robustness_mixture': 5,
                })
                self.assertEqual(summary[split], sum(counts.values()))

                speakers_seen = {
                    c['hidden']['true_speaker'] for c in loaded if c['hidden'].get('true_speaker')
                } | {
                    c['hidden']['claimed'] for c in loaded if c['hidden'].get('claimed')
                }
                self.assertEqual(len(speakers_seen), 20)
                split_speakers[split] = speakers_seen

                noise_cases = [c for c in loaded if c['suite'] == 'robustness_noise']
                self.assertEqual(len(Counter(c['hidden']['variant'] for c in noise_cases)), 12)
                for c in noise_cases:
                    self.assertEqual(c['hidden']['axis'], 'noise')
                self.assertEqual(set(Counter(c['hidden']['variant'] for c in noise_cases).values()), {40})

                dur_cases = [c for c in loaded if c['suite'] == 'robustness_duration']
                self.assertEqual(
                    Counter(c['hidden']['variant'] for c in dur_cases),
                    {f'dur_{sec:.1f}s': 40 for sec in DURATIONS},
                )
                for sec in DURATIONS:
                    sample = next(c for c in dur_cases if c['hidden']['variant'] == f'dur_{sec:.1f}s')
                    self.assertEqual(sample['hidden']['axis'], 'duration')
                    probe_path = shlex.split(sample['steps'][1], posix=True)[2]
                    self.assertAlmostEqual(sf.info(probe_path).duration, sec, places=3)

                chan_cases = [c for c in loaded if c['suite'] == 'robustness_channel']
                self.assertEqual(
                    Counter(c['hidden']['variant'] for c in chan_cases),
                    {'tel8k_ulaw': 40},
                )
                self.assertEqual(chan_cases[0]['hidden']['axis'], 'channel')
                chan_probe = shlex.split(chan_cases[0]['steps'][1], posix=True)[2]
                self.assertEqual(sf.info(chan_probe).samplerate, 8000)

                rev_cases = [c for c in loaded if c['suite'] == 'robustness_reverb']
                self.assertEqual(
                    Counter(c['hidden']['variant'] for c in rev_cases),
                    {f'rt60_{rt:.1f}s': 40 for rt in RT60_LEVELS},
                )
                self.assertEqual(rev_cases[0]['hidden']['axis'], 'reverb')

                mix_cases = [c for c in loaded if c['suite'] == 'robustness_mixture']
                self.assertEqual(len(mix_cases), 5)
                for mc in mix_cases:
                    self.assertEqual(mc['hidden']['axis'], 'mixture')
                    self.assertEqual(mc['hidden']['variant'], 'mix2_0db_8k')
                    self.assertEqual(len(mc['hidden']['expect_speakers']), 2)
                    self.assertEqual(len(mc['hidden']['gallery_speakers']), 6)
                    for spk in mc['hidden']['expect_speakers']:
                        self.assertIn(spk, mc['hidden']['gallery_speakers'])
                    self.assertTrue(mc['steps'][-1].startswith('分离再辨认 '))
                    mix_probe = shlex.split(mc['steps'][-1], posix=True)[1]
                    self.assertEqual(sf.info(mix_probe).samplerate, 8000)

            self.assertTrue(split_speakers['validation'].isdisjoint(split_speakers['locked']))

            # Determinism across independent rebuilds into another directory
            case_dir_b = root / 'cases_b'
            build(test_root=test_root, case_dir=case_dir_b)
            loaded_a = load('validation', case_dir=case_dir)
            loaded_b = load('validation', case_dir=case_dir_b)
            self.assertEqual([c['id'] for c in loaded_a], [c['id'] for c in loaded_b])
            self.assertEqual([c['hidden'] for c in loaded_a], [c['hidden'] for c in loaded_b])
            for suite_pick in (
                'robustness_noise',
                'robustness_duration',
                'robustness_channel',
                'robustness_reverb',
                'robustness_mixture',
            ):
                ca = next(c for c in loaded_a if c['suite'] == suite_pick)
                cb = next(c for c in loaded_b if c['suite'] == suite_pick)
                pa = shlex.split(ca['steps'][-1], posix=True)[-1]
                pb = shlex.split(cb['steps'][-1], posix=True)[-1]
                wa, sra = sf.read(pa, dtype='float32')
                wb, srb = sf.read(pb, dtype='float32')
                self.assertEqual(sra, srb)
                self.assertTrue(np.array_equal(wa, wb), f'non-deterministic audio for {suite_pick}')

        repo_root = Path(__file__).resolve().parents[1]
        proto = json.loads((repo_root / 'voice_eval' / 'protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(proto['case_version'], 3)
        for req_suite in (
            'verify',
            'verify_impostor',
            'identify',
            'enroll_quality',
            'robustness_noise',
            'robustness_duration',
            'robustness_channel',
            'robustness_reverb',
            'robustness_mixture',
            'security',
        ):
            self.assertIn(req_suite, proto['suites'])
        self.assertNotIn('verify_short', proto['suites'])
        self.assertNotIn('verify_noise', proto['suites'])
        for req_key in ('dimensions', 'mindcf_params', 'gate_tiers'):
            self.assertIn(req_key, proto)
        gitignore_lines = (repo_root / '.gitignore').read_text(encoding='utf-8').splitlines()
        self.assertIn('/voice_eval/cache/', gitignore_lines)

    def test_babble_excludes_trial_target_and_impostor_and_defaults_backward_compat(self):
        self.assertIsNone(inspect.signature(build).parameters['test_root'].default)
        self.assertIsNone(inspect.signature(build).parameters['case_dir'].default)
        self.assertIsNone(inspect.signature(build_split).parameters['case_dir'].default)
        self.assertIsNone(inspect.signature(load).parameters['case_dir'].default)
        self.assertIsNone(inspect.signature(case_version).parameters['case_dir'].default)

        names = [f'spk_{i:02d}' for i in range(20)]
        clean_probes = {
            spk: (np.full(SR, float(i + 1) * 0.01, dtype=np.float32), SR)
            for i, spk in enumerate(names)
        }
        for idx, name in enumerate(names):
            claimant = names[(idx - 1) % len(names)]
            for snr in SNR_LEVELS:
                picked = _sample_babble_interferers(names, name, clean_probes, 'validation', snr)
                self.assertEqual(len(picked), 3)
                for arr in picked:
                    self.assertFalse(np.array_equal(arr, clean_probes[name][0]))
                    self.assertFalse(np.array_equal(arr, clean_probes[claimant][0]))


if __name__ == '__main__':
    unittest.main()
