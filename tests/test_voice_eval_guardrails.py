"""Tests for Task 1 guardrails, audio validation, and deterministic test sandbox."""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import agent.service as service_mod
import voiceprint.security
import voiceprint.store
from agent.service import VoiceAgent
from tests._helpers import patched_dpapi, synthetic_librispeech_tree
from voice_eval.audit import run_all as audit_run_all
from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav, sandbox_cohort
from voiceprint.audio import AudioError, check_audio, list_flac, load_wav


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _enroll_speaker(agent: VoiceAgent, name: str, base_dir: Path, speaker_index: int) -> list[Path]:
    takes = [
        generate_speaker_wav(base_dir / 's1' / f'{name}_0.wav', speaker_index, duration=4.0, phase=0.0, seed=0),
        generate_speaker_wav(base_dir / 's1' / f'{name}_1.wav', speaker_index, duration=4.0, phase=0.8, seed=1),
        generate_speaker_wav(base_dir / 's2' / f'{name}_2.wav', speaker_index, duration=4.0, phase=1.6, seed=2),
    ]
    cmd = f'注册 {name} 本人同意 ' + ' '.join(f'"{p}"' for p in takes)
    out = agent.handle(cmd)
    if not out['ok']:
        raise AssertionError(f'enroll failed unexpectedly: {out}')
    return takes


class GuardrailTests(unittest.TestCase):
    def test_pending_revoke_cleared_by_list_help_and_unknown(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = VoiceAgent(root / 'db', embedder, detail=True, audio_roots=[root])
            try:
                _enroll_speaker(agent, 'alice', root / 'audio', speaker_index=0)
                for interrupt in ('名单', '帮助', 'help', '退出', '匿名', '未知命令', 'bad "unclosed', ''):
                    ask = agent.handle('撤销 alice')
                    self.assertTrue(ask['ok'])
                    self.assertEqual(agent.pending, 'alice')
                    agent.handle(interrupt)
                    self.assertIsNone(agent.pending, f'pending not cleared by {interrupt!r}')
                    confirm = agent.handle('确认撤销 alice')
                    self.assertFalse(confirm['ok'], f'confirm succeeded after {interrupt!r}')
                    self.assertIn('alice', agent.store.names())
                ask_again = agent.handle('撤销 alice')
                self.assertTrue(ask_again['ok'])
                done = agent.handle('确认撤销 alice')
                self.assertTrue(done['ok'])
                self.assertNotIn('alice', agent.store.names())
            finally:
                agent.close()

    def test_verify_detail_false_hides_scores_and_threshold_words(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detail_agent = VoiceAgent(root / 'db_detail', embedder, detail=True, audio_roots=[root])
            quiet_agent = VoiceAgent(root / 'db_quiet', embedder, detail=False, audio_roots=[root])
            try:
                _enroll_speaker(detail_agent, 'alice', root / 'audio_d', speaker_index=0)
                _enroll_speaker(quiet_agent, 'alice', root / 'audio_q', speaker_index=0)
                genuine = generate_speaker_wav(root / 'genuine.wav', speaker_index=0, duration=4.0, phase=2.4, seed=3)
                impostor = generate_speaker_wav(root / 'impostor.wav', speaker_index=1, duration=4.0, phase=0.0, seed=0)

                w0, sr = sf.read(str(genuine), dtype='float32')
                w1, _ = sf.read(str(impostor), dtype='float32')
                uncertain_path = root / 'uncertain.wav'
                found_uncertain = False
                for alpha in np.linspace(0.0, 1.0, 101):
                    blend = ((1.0 - alpha) * w0 + alpha * w1).astype(np.float32)
                    sf.write(str(uncertain_path), blend, sr)
                    probe_res = detail_agent.handle(f'验证 alice "{uncertain_path}"')
                    if probe_res['result'].get('decision') == 'uncertain':
                        found_uncertain = True
                        break
                self.assertTrue(found_uncertain, 'failed to synthesize a deterministic uncertain probe')

                for probe_path, expect in ((genuine, 'accept'), (impostor, 'reject'), (uncertain_path, 'uncertain')):
                    out = quiet_agent.handle(f'验证 alice "{probe_path}"')
                    self.assertTrue(out['ok'])
                    self.assertEqual(set(out['result'].keys()), {'decision', 'task'})
                    self.assertEqual(out['result']['decision'], expect)
                    self.assertEqual(out['result']['task'], 'verify')
                    self.assertIsNone(re.search(r'\d', out['reply']), f'unexpected digits in reply: {out["reply"]}')
                    for forbidden in ('归一化相似度', '阈值', '原始余弦', '夹角'):
                        self.assertNotIn(forbidden, out['reply'])
                # Ensure uncertain still asks for a better recording without leaking threshold words
                unc_out = quiet_agent.handle(f'验证 alice "{uncertain_path}"')
                self.assertIn('请再录一段更长、更安静的录音', unc_out['reply'])
            finally:
                detail_agent.close()
                quiet_agent.close()

    def test_english_export_probe_refused_and_missing_speaker_drift_not_ok(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = VoiceAgent(root / 'db', embedder, detail=False, audio_roots=[root])
            try:
                _enroll_speaker(agent, 'alice', root / 'audio', speaker_index=0)
                for probe_text in (
                    'export_template',
                    'dump embedding',
                    'Dump Embedding',
                    'please dump embedding for alice',
                    'print embedding',
                    'export embedding',
                    'raw embedding',
                ):
                    refused = agent.handle(probe_text)
                    self.assertEqual(refused['tool'], 'refuse', f'failed to refuse {probe_text!r}')
                    self.assertTrue(refused['ok'])
                    self.assertEqual(refused['result'].get('requested'), 'export_template')

                emb_dir = root / 'raw embedding'
                valid_probe = generate_speaker_wav(
                    emb_dir / 'probe_embedding.wav', speaker_index=0, duration=4.0, phase=2.1, seed=4
                )
                verified = agent.handle(f'验证 alice "{valid_probe}"')
                self.assertEqual(verified['tool'], 'verify')
                self.assertTrue(verified['ok'])
                self.assertEqual(verified['result']['decision'], 'accept')

                drift_missing = agent.handle('漂移 不存在')
                self.assertEqual(drift_missing['tool'], 'drift')
                self.assertFalse(drift_missing['ok'])
                self.assertEqual(drift_missing['result'].get('decision'), 'unknown')
            finally:
                agent.close()

    def test_check_audio_and_load_wav_reject_nan_and_inf(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr = 16000
            clean = np.sin(2.0 * np.pi * 220.0 * np.arange(sr, dtype=np.float32) / sr) * 0.5
            nan_wav = root / 'nan.wav'
            inf_wav = root / 'inf.wav'
            bad_nan = clean.copy()
            bad_nan[128] = np.nan
            sf.write(str(nan_wav), bad_nan, sr, subtype='FLOAT')

            bad_inf = clean.copy()
            bad_inf[256] = np.inf
            sf.write(str(inf_wav), bad_inf, sr, subtype='FLOAT')

            stereo_inf_wav = root / 'stereo_inf.wav'
            stereo_pair = np.column_stack([clean, clean])
            stereo_pair[64, 0] = np.inf
            stereo_pair[64, 1] = -np.inf
            sf.write(str(stereo_inf_wav), stereo_pair, sr, subtype='FLOAT')

            for bad in (nan_wav, inf_wav, stereo_inf_wav):
                with self.assertRaises(AudioError) as ctx_chk:
                    check_audio(str(bad))
                self.assertIn('无效采样', str(ctx_chk.exception))
                with self.assertRaises(AudioError) as ctx_load:
                    load_wav(str(bad))
                self.assertIn('无效采样', str(ctx_load.exception))

            agent = VoiceAgent(root / 'db', embedder, detail=True, audio_roots=[root])
            try:
                good1 = generate_speaker_wav(root / 's1' / 'g1.wav', 0, duration=4.0, phase=0.0, seed=0)
                good2 = generate_speaker_wav(root / 's2' / 'g2.wav', 0, duration=4.0, phase=0.8, seed=1)
                bad_enroll = agent.handle(f'注册 bob 本人同意 "{good1}" "{good2}" "{nan_wav}"')
                self.assertFalse(bad_enroll['ok'])
                self.assertIn('无效采样', bad_enroll['reply'])
                self.assertNotIn('bob', agent.store.names())

                _enroll_speaker(agent, 'alice', root / 'audio', speaker_index=0)
                for bad in (nan_wav, inf_wav):
                    bad_verify = agent.handle(f'验证 alice "{bad}"')
                    self.assertFalse(bad_verify['ok'])
                    self.assertIn('无效采样', bad_verify['reply'])
                self.assertEqual(agent.store.events('alice'), [])
            finally:
                agent.close()

    def test_tone_embedder_separates_speakers_and_degrades_with_noise(self):
        embedder = ToneEmbedder()
        self.assertEqual(embedder.name, 'tone')
        self.assertEqual(embedder.dim, 64)
        self.assertEqual(embedder.cohort_size, 16)
        self.assertTrue(embedder.model_id.startswith('tone-v1'))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a0 = generate_speaker_wav(root / 'a0.wav', speaker_index=0, duration=4.0, phase=0.0, seed=0)
            a1 = generate_speaker_wav(root / 'a1.wav', speaker_index=0, duration=4.0, phase=0.8, seed=1)
            a2 = generate_speaker_wav(root / 'a2.wav', speaker_index=0, duration=4.0, phase=1.6, seed=0)
            v0 = embedder.embed_file(str(a0))
            v1 = embedder.embed_file(str(a1))
            v2 = embedder.embed_file(str(a2))

            self.assertGreater(_cosine(v0, v1), 0.85)
            self.assertGreater(_cosine(v0, v2), 0.85)
            self.assertLess(_cosine(v0, v1), 0.995)

            others = [
                embedder.embed_file(str(generate_speaker_wav(root / f'spk_{i}.wav', speaker_index=i)))
                for i in range(6)
            ]
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    self.assertLess(_cosine(others[i], others[j]), 0.45)

            wav, sr = sf.read(str(a0), dtype='float32')
            rms = float(np.sqrt(np.mean(np.square(wav))))
            noise = np.random.default_rng(7).normal(0.0, rms, size=wav.shape[0]).astype(np.float32)  # ~0 dB SNR
            noisy_path = root / 'a0_noisy.wav'
            sf.write(str(noisy_path), np.clip(wav + noise, -1.0, 1.0), sr)
            v_noisy = embedder.embed_file(str(noisy_path))
            self.assertLess(_cosine(v0, v_noisy), 0.95)
            self.assertGreater(_cosine(v0, v_noisy), 0.45)

            # Verify synthetic_librispeech_tree creates valid train/test FLAC speaker trees
            tree_root = root / 'LibriSpeech'
            train_root, test_root = synthetic_librispeech_tree(tree_root, n_train=4, n_test=6)
            self.assertEqual(len(list_flac(str(train_root))), 4)
            test_speakers = list_flac(str(test_root))
            self.assertEqual(len(test_speakers), 6)
            first_files = next(iter(test_speakers.values()))
            chapters = {Path(p).parent.name for p in first_files}
            self.assertGreaterEqual(len(chapters), 3)

    def test_sandbox_and_dpapi_restore_originals_and_pass_audit(self):
        orig_sec = (voiceprint.security.dpapi_protect, voiceprint.security.dpapi_unprotect)
        orig_store = (voiceprint.store.dpapi_protect, voiceprint.store.dpapi_unprotect)
        orig_cohort = service_mod.cohort_matrix
        embedder = ToneEmbedder()

        with patched_dpapi():
            self.assertIsNot(voiceprint.security.dpapi_protect, orig_sec[0])
            self.assertIsNot(voiceprint.store.dpapi_protect, orig_store[0])
            audit = audit_run_all()
            self.assertTrue(audit['ok'], audit)
            with sandbox_cohort(embedder):
                self.assertIsNot(service_mod.cohort_matrix, orig_cohort)
            self.assertIs(service_mod.cohort_matrix, orig_cohort)

        self.assertIs(voiceprint.security.dpapi_protect, orig_sec[0])
        self.assertIs(voiceprint.security.dpapi_unprotect, orig_sec[1])
        self.assertIs(voiceprint.store.dpapi_protect, orig_store[0])
        self.assertIs(voiceprint.store.dpapi_unprotect, orig_store[1])


if __name__ == '__main__':
    unittest.main()
