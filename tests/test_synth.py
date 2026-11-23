"""Tests for controllable voice synthesizer interfaces, acoustic morphing and isolated neural subprocess runner."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav
from voiceprint.audio import MIN_SECONDS
from voiceprint.features import SR
from voiceprint.scoring import cosine
from voiceprint.synth import (
    AcousticFormantSynthesizer,
    MAX_SIM_TEXT_CHARS,
    SimulationRequest,
    SimulationResult,
    SubprocessNeuralSynthesizer,
    SynthesisError,
    VoiceSynthesizer,
    load_synthesizer,
    split_command_args,
    validate_simulation_result,
)


class TestVoiceSynth(unittest.TestCase):
    def test_acoustic_synthesizer_audio_and_text_modes_shift_toward_target_tone(self):
        synth = AcousticFormantSynthesizer()
        self.assertEqual(synth.name, 'acoustic_formant')
        self.assertEqual(synth.supported_modes, ('audio', 'text'))
        embedder = ToneEmbedder()

        with TemporaryDirectory(prefix='vpsynth_ac_') as tmp:
            root = Path(tmp)
            ref_wav = generate_speaker_wav(root / 'ref_0.wav', speaker_index=0, duration=3.0, seed=1)
            ref_wav_2 = generate_speaker_wav(root / 'ref_2.wav', speaker_index=2, duration=3.0, seed=2)
            src_wav = generate_speaker_wav(root / 'src_1.wav', speaker_index=1, duration=2.5, seed=3)

            ref_emb = embedder.embed_file(str(ref_wav))
            ref2_emb = embedder.embed_file(str(ref_wav_2))
            src_emb = embedder.embed_file(str(src_wav))

            low_req = SimulationRequest(
                target_name='jia',
                mode='audio',
                reference_audio_path=str(ref_wav),
                source_audio_path=str(src_wav),
                formant_blend=0.15,
                pitch_pull=0.15,
                gain_scale=0.85,
                seed=7,
            )
            mid_req = SimulationRequest(
                target_name='jia',
                mode='audio',
                reference_audio_path=str(ref_wav),
                source_audio_path=str(src_wav),
                formant_blend=0.55,
                pitch_pull=0.55,
                gain_scale=0.85,
                seed=7,
            )
            high_req = SimulationRequest(
                target_name='jia',
                mode='audio',
                reference_audio_path=str(ref_wav),
                source_audio_path=str(src_wav),
                formant_blend=0.92,
                pitch_pull=0.92,
                gain_scale=0.85,
                seed=7,
            )

            sims = []
            for label, req in (('low', low_req), ('mid', mid_req), ('high', high_req)):
                res = synth.synthesize(req)
                wav = validate_simulation_result(res)
                self.assertEqual(res.sr, SR)
                self.assertEqual(wav.dtype, np.float32)
                self.assertGreaterEqual(wav.shape[0], int(MIN_SECONDS * SR))
                self.assertLessEqual(float(np.max(np.abs(wav))), 1.0)
                out_p = root / f'out_{label}.wav'
                sf.write(str(out_p), wav, SR)
                sims.append(cosine(ref_emb, embedder.embed_file(str(out_p))))

            self.assertGreater(sims[1], sims[0])
            self.assertGreater(sims[2], sims[1])
            self.assertGreater(sims[2], cosine(ref_emb, src_emb))

            # Text mode with UTF-8 Chinese text: deterministic and closer to matching speaker
            txt_a = synth.synthesize(SimulationRequest(
                target_name='jia',
                mode='text',
                reference_audio_path=str(ref_wav),
                text='你好，请核对今天的声纹模拟结果。',
                formant_blend=0.85,
                pitch_pull=0.85,
                seed=42,
            ))
            txt_a_repeat = synth.synthesize(SimulationRequest(
                target_name='jia',
                mode='text',
                reference_audio_path=str(ref_wav),
                text='你好，请核对今天的声纹模拟结果。',
                formant_blend=0.85,
                pitch_pull=0.85,
                seed=42,
            ))
            self.assertTrue(np.allclose(txt_a.wav, txt_a_repeat.wav))

            txt_b = synth.synthesize(SimulationRequest(
                target_name='bing',
                mode='text',
                reference_audio_path=str(ref_wav_2),
                text='你好，请核对今天的声纹模拟结果。',
                formant_blend=0.85,
                pitch_pull=0.85,
                seed=42,
            ))
            p_a = root / 'txt_a.wav'
            p_b = root / 'txt_b.wav'
            sf.write(str(p_a), validate_simulation_result(txt_a), SR)
            sf.write(str(p_b), validate_simulation_result(txt_b), SR)
            emb_a = embedder.embed_file(str(p_a))
            emb_b = embedder.embed_file(str(p_b))
            self.assertGreater(cosine(ref_emb, emb_a), cosine(ref2_emb, emb_a))
            self.assertGreater(cosine(ref2_emb, emb_b), cosine(ref_emb, emb_b))

    def test_validate_simulation_result_rejects_nan_too_short_and_resamples(self):
        # Reject NaN
        with self.assertRaises(SynthesisError):
            validate_simulation_result(SimulationResult(
                wav=np.array([0.0, np.nan] * 5000, dtype=np.float32),
                sr=SR,
                backend='test',
                metadata={},
            ))
        # Reject too short (< 0.5s)
        with self.assertRaises(SynthesisError):
            validate_simulation_result(SimulationResult(
                wav=np.ones(int(0.2 * SR), dtype=np.float32) * 0.1,
                sr=SR,
                backend='test',
                metadata={},
            ))
        # Reject too long (> 60.0s)
        with self.assertRaises(SynthesisError):
            validate_simulation_result(SimulationResult(
                wav=np.zeros(int(61.0 * SR), dtype=np.float32),
                sr=SR,
                backend='test',
                metadata={},
            ))
        # 48 kHz stereo -> resampled mono 16 kHz clipped to [-1, 1]
        t = np.arange(int(1.0 * 48000), dtype=np.float32) / 48000.0
        stereo = np.stack([1.2 * np.sin(2.0 * np.pi * 220.0 * t), 0.6 * np.sin(2.0 * np.pi * 220.0 * t)], axis=1)
        out = validate_simulation_result(SimulationResult(wav=stereo, sr=48000, backend='test', metadata={}))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 1)
        self.assertAlmostEqual(out.shape[0], SR, delta=8)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_split_command_args_preserves_windows_quoted_spaces(self):
        fwd = split_command_args('"D:/voice confirm/.venv_tts/Scripts/python.exe" plugins/neural_tts_worker.py --engine f5_tts')
        self.assertEqual(fwd, ['D:/voice confirm/.venv_tts/Scripts/python.exe', 'plugins/neural_tts_worker.py', '--engine', 'f5_tts'])

        win_bs = split_command_args(r'"D:\voice confirm\.venv_tts\Scripts\python.exe" plugins\worker.py --tag "hello world"')
        self.assertEqual(win_bs, [r'D:\voice confirm\.venv_tts\Scripts\python.exe', r'plugins\worker.py', '--tag', 'hello world'])

        with self.assertRaises(ValueError):
            split_command_args('   ')
        with self.assertRaises(ValueError):
            split_command_args('"D:/voice confirm/unclosed')

    def test_subprocess_synthesizer_utf8_env_and_temp_cleanup_on_timeout(self):
        worker_ok = (
            "import argparse, json, os, numpy as np, soundfile as sf; "
            "p = argparse.ArgumentParser(); p.add_argument('--request'); a = p.parse_args(); "
            "assert os.environ.get('PYTHONUTF8') == '1'; "
            "assert os.environ.get('PYTHONIOENCODING', '').lower() == 'utf-8'; "
            "req = json.loads(open(a.request, encoding='utf-8').read()); "
            "assert '测试中文' in req['text'] and '参考句' in req['reference_text']; "
            "t = np.arange(16000, dtype=np.float32) / 16000.0; "
            "sf.write(req['output_wav_path'], 0.3 * np.sin(2*np.pi*200*t), 16000); "
            "print(json.dumps({'ok': True, 'model': 'mock_f5', 'device': 'cpu'}))"
        )
        worker_sleep = "import time; time.sleep(5)"
        worker_fail = "import sys; sys.stderr.write('boom'); sys.exit(2)"
        worker_bad_wav = (
            "import argparse, json; p = argparse.ArgumentParser(); p.add_argument('--request'); a = p.parse_args(); "
            "req = json.loads(open(a.request, encoding='utf-8').read()); "
            "open(req['output_wav_path'], 'wb').write(b'not-a-wav')"
        )

        tmp_root = Path(tempfile.gettempdir())
        before_dirs = {p.name for p in tmp_root.glob('vpsyn_*')}
        with TemporaryDirectory(prefix='vpref_') as rdir:
            ref_p = generate_speaker_wav(Path(rdir) / 'ref.wav', speaker_index=0, duration=1.5)
            req = SimulationRequest(
                target_name='jia',
                mode='text',
                reference_audio_path=str(ref_p),
                reference_text='这是一句参考句。',
                text='这是一段测试中文语音。',
            )

            syn_ok = SubprocessNeuralSynthesizer([sys.executable, '-c', worker_ok], timeout_s=10.0)
            self.assertEqual(syn_ok.supported_modes, ('text',))
            res = syn_ok.synthesize(req)
            self.assertEqual(res.backend, 'neural_subprocess')
            self.assertEqual(res.metadata.get('model'), 'mock_f5')
            self.assertGreaterEqual(res.wav.shape[0], 16000)

            # Timeout must raise SynthesisError and clean temp directory
            syn_to = SubprocessNeuralSynthesizer([sys.executable, '-c', worker_sleep], timeout_s=0.2)
            with self.assertRaises(SynthesisError):
                syn_to.synthesize(req)

            # Non-zero exit must raise SynthesisError and clean temp directory
            syn_fail = SubprocessNeuralSynthesizer([sys.executable, '-c', worker_fail], timeout_s=5.0)
            with self.assertRaises(SynthesisError):
                syn_fail.synthesize(req)

            # Corrupt output file must raise SynthesisError and clean temp directory
            syn_bad = SubprocessNeuralSynthesizer([sys.executable, '-c', worker_bad_wav], timeout_s=5.0)
            with self.assertRaises(SynthesisError):
                syn_bad.synthesize(req)

        after_dirs = {p.name for p in tmp_root.glob('vpsyn_*')}
        self.assertEqual(after_dirs, before_dirs)

    def test_load_synthesizer_variants(self):
        self.assertIsNone(load_synthesizer('none'))
        ac = load_synthesizer('acoustic')
        self.assertIsInstance(ac, AcousticFormantSynthesizer)
        self.assertIsInstance(ac, VoiceSynthesizer)
        ns = load_synthesizer('neural_subprocess', neural_cmd=f'"{sys.executable}" plugins/neural_tts_worker.py')
        self.assertIsInstance(ns, SubprocessNeuralSynthesizer)
        self.assertEqual(MAX_SIM_TEXT_CHARS, 500)
        with self.assertRaises(ValueError):
            load_synthesizer('neural_subprocess', neural_cmd=None)
        with self.assertRaises(ValueError):
            load_synthesizer('unknown_backend')

    def test_neural_tts_worker_cli_help_and_zero_core_dependency(self):
        import os
        from voiceprint.contained_subprocess import run_contained

        repo_root = Path(__file__).resolve().parents[1]
        worker_script = repo_root / 'plugins' / 'neural_tts_worker.py'
        help_res = run_contained([sys.executable, str(worker_script), '--help'], timeout=10.0)
        self.assertEqual(help_res.returncode, 0, help_res.stderr.decode('utf-8', errors='replace'))

        with TemporaryDirectory(prefix='vpworker_') as tmp:
            root = Path(tmp)
            ref_p = generate_speaker_wav(root / 'ref.wav', speaker_index=0, duration=1.5)

            # Passing unsupported mode='audio' to --engine f5_tts returns structured error
            bad_req = root / 'bad_req.json'
            bad_req.write_text(json.dumps({
                'mode': 'audio',
                'reference_audio_path': str(ref_p),
                'source_audio_path': str(ref_p),
                'output_wav_path': str(root / 'bad_out.wav'),
            }), encoding='utf-8')
            bad_res = run_contained([sys.executable, str(worker_script), '--engine', 'f5_tts', '--request', str(bad_req)], timeout=10.0)
            self.assertNotEqual(bad_res.returncode, 0)
            self.assertIn('unsupported', bad_res.stderr.decode('utf-8', errors='replace').lower() + bad_res.stdout.decode('utf-8', errors='replace').lower())

            # Inject stub f5_tts.api module via PYTHONPATH to test end-to-end SubprocessNeuralSynthesizer protocol
            stub_pkg = root / 'stub_site' / 'f5_tts'
            stub_pkg.mkdir(parents=True)
            (stub_pkg / '__init__.py').write_text('', encoding='utf-8')
            (stub_pkg / 'api.py').write_text(
                "import numpy as np, soundfile as sf\n"
                "class F5TTS:\n"
                "    def __init__(self, **kw):\n"
                "        print('noisy model init banner on stdout!')\n"
                "    def infer(self, ref_file, ref_text, gen_text, file_wave=None, **kw):\n"
                "        assert ref_text == '参考朗读句子' and '测试中文' in gen_text\n"
                "        t = np.arange(24000, dtype=np.float32) / 24000.0\n"
                "        wav = (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)\n"
                "        if file_wave:\n"
                "            sf.write(file_wave, wav, 24000)\n"
                "        return wav, 24000, None\n",
                encoding='utf-8',
            )
            syn = SubprocessNeuralSynthesizer(
                [sys.executable, str(worker_script), '--engine', 'f5_tts'],
                timeout_s=15.0,
                env_overrides={'PYTHONPATH': f"{root / 'stub_site'}{os.pathsep}{repo_root}"},
            )
            res = syn.synthesize(SimulationRequest(
                target_name='jia',
                mode='text',
                reference_audio_path=str(ref_p),
                reference_text='参考朗读句子',
                text='测试中文神经克隆协议。',
                gain_scale=0.85,
                seed=7,
            ))
            self.assertEqual(res.sr, SR)
            self.assertEqual(res.backend, 'neural_subprocess')
            self.assertEqual(res.metadata.get('engine'), 'f5_tts')
            self.assertGreaterEqual(res.wav.shape[0], SR)

            # Missing executable must raise SynthesisError (not OSError) and clean up temp dir
            import tempfile
            vpsyn_parent = Path(tempfile.gettempdir())
            before_missing = {p.name for p in vpsyn_parent.glob('vpsyn_*')}
            missing_syn = SubprocessNeuralSynthesizer(['/nonexistent/voiceprint_missing_bin/python'], timeout_s=5.0)
            with self.assertRaises(SynthesisError):
                missing_syn.synthesize(SimulationRequest(
                    target_name='jia',
                    mode='text',
                    reference_audio_path=str(ref_p),
                    text='测试不存在的可执行文件。',
                ))
            after_missing = {p.name for p in vpsyn_parent.glob('vpsyn_*')}
            self.assertEqual(after_missing, before_missing)

            # Missing reference/source audio in acoustic backend must also raise SynthesisError
            with self.assertRaises(SynthesisError):
                AcousticFormantSynthesizer().synthesize(SimulationRequest(
                    target_name='jia',
                    mode='audio',
                    reference_audio_path=str(root / 'missing_ref.wav'),
                    source_audio_path=str(ref_p),
                ))

            # Channel-first (2, T) tuple fallback without writing file_wave must average channels, not collapse time axis
            (stub_pkg / 'api.py').write_text(
                "import numpy as np\n"
                "class F5TTS:\n"
                "    def __init__(self, **kw):\n"
                "        pass\n"
                "    def infer(self, ref_file, ref_text, gen_text, file_wave=None, **kw):\n"
                "        t = np.arange(24000, dtype=np.float32) / 24000.0\n"
                "        mono = (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)\n"
                "        ch_first = np.stack([mono, mono * 0.8], axis=0)\n"
                "        return ch_first, 24000, None\n",
                encoding='utf-8',
            )
            res_ch = syn.synthesize(SimulationRequest(
                target_name='jia',
                mode='text',
                reference_audio_path=str(ref_p),
                reference_text='参考',
                text='通道前置张量测试。',
                gain_scale=0.85,
                seed=8,
            ))
            self.assertEqual(res_ch.sr, SR)
            self.assertGreaterEqual(res_ch.wav.shape[0], SR)


if __name__ == '__main__':
    unittest.main()
