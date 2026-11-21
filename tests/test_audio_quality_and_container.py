"""Tests for audio container integrity validation and 100ms DSP signal quality gates."""
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from agent.service import VoiceAgent
from tests._helpers import patched_dpapi
from voice_eval.dims._sandbox import (
    ToneEmbedder,
    generate_speaker_wav,
    quote_paths,
    sandbox_cohort,
    sandbox_enroll_takes,
)
from voiceprint.audio import AudioError, check_audio, sniff_audio_format
from voiceprint.audio_quality import MAX_ENROLL_CLIPPING_RATIO, analyze_audio_quality


def _write_wav_with_odd_chunk(path: Path, samples_i16: np.ndarray, sr: int = 16000) -> Path:
    pcm = samples_i16.astype('<i2').tobytes()
    fmt_chunk = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, sr, sr * 2, 2, 16)
    # Odd-sized 11-byte JUNK chunk + 1 pad byte per RIFF spec.
    odd_payload = b'0123456789A'
    junk_chunk = struct.pack('<4sI', b'JUNK', len(odd_payload)) + odd_payload + b'\x00'
    data_chunk = struct.pack('<4sI', b'data', len(pcm)) + pcm
    body = b'WAVE' + fmt_chunk + junk_chunk + data_chunk
    riff = struct.pack('<4sI', b'RIFF', len(body)) + body
    path.write_bytes(riff)
    return path


class TestAudioQualityAndContainer(unittest.TestCase):
    def test_sniff_audio_format_magic_bytes(self):
        self.assertEqual(sniff_audio_format(b'RIFF\x24\x00\x00\x00WAVEfmt '), '.wav')
        self.assertEqual(sniff_audio_format(b'fLaC\x00\x00\x00\x22rest'), '.flac')
        self.assertEqual(sniff_audio_format(b'OggS\x00\x02\x00\x00rest'), '.ogg')
        self.assertEqual(sniff_audio_format(b'ID3\x03\x00\x00\x00\x00\x00\x00'), '.mp3')
        self.assertEqual(sniff_audio_format(b'\xff\xfb\x90\x00\x00\x00\x00\x00'), '.mp3')
        self.assertIsNone(sniff_audio_format(b'not audio bytes at all'))
        self.assertIsNone(sniff_audio_format(b'\x89PNG\r\n\x1a\n'))
        self.assertIsNone(sniff_audio_format(b''))

    def test_check_audio_rejects_magic_mismatch_truncated_wav_and_symlinks(self):
        with TemporaryDirectory(prefix='vpaq_cnt_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            outside = root / 'outside'
            allowed.mkdir()
            outside.mkdir()

            # 1. Magic mismatch: MP3 header disguised as .wav
            fake_wav = allowed / 'disguised.wav'
            fake_wav.write_bytes(b'ID3\x03\x00' + b'\x00' * 1024)
            with self.assertRaises(AudioError) as ctx:
                check_audio(str(fake_wav), [allowed])
            self.assertTrue('不匹配' in str(ctx.exception) or '无效' in str(ctx.exception))

            # 2. Valid WAV with odd-sized metadata chunk + pad byte -> passes
            t = np.arange(32000, dtype=np.float64) / 16000.0
            sig_i16 = (np.sin(2.0 * np.pi * 440.0 * t) * 8000).astype(np.int16)
            odd_wav = _write_wav_with_odd_chunk(allowed / 'odd_chunk.wav', sig_i16)
            resolved = check_audio(str(odd_wav), [allowed])
            self.assertEqual(resolved, odd_wav.resolve())

            # 3. Truncated WAV (valid RIFF header claiming full data, but payload cut in half)
            full_bytes = odd_wav.read_bytes()
            trunc_wav = allowed / 'truncated.wav'
            trunc_wav.write_bytes(full_bytes[:len(full_bytes) // 2])
            with self.assertRaises(AudioError):
                check_audio(str(trunc_wav), [allowed])

            # 4. Symlinks: outside-root -> '允许的目录'; inside-root -> '符号链接'
            out_wav = generate_speaker_wav(outside / 'out.wav', speaker_index=0, duration=2.0, seed=1)
            in_link = allowed / 'in_link.wav'
            out_link = allowed / 'out_link.wav'
            try:
                in_link.symlink_to(odd_wav)
                out_link.symlink_to(out_wav)
            except OSError:
                return
            with self.assertRaises(AudioError) as out_ctx:
                check_audio(str(out_link), [allowed])
            self.assertIn('允许的目录', str(out_ctx.exception))

            with self.assertRaises(AudioError) as in_ctx:
                check_audio(str(in_link), [allowed])
            self.assertIn('符号链接', str(in_ctx.exception))

    def test_analyze_audio_quality_detects_clean_silence_and_clipping(self):
        with TemporaryDirectory(prefix='vpaq_dsp_') as tmp:
            root = Path(tmp)
            clean = generate_speaker_wav(root / 'clean.wav', speaker_index=0, duration=2.0, seed=0)
            rep_clean = analyze_audio_quality(clean)
            self.assertTrue(rep_clean.passed_enroll_quality)
            self.assertEqual(rep_clean.clipping_ratio, 0.0)
            self.assertGreaterEqual(rep_clean.active_seconds, 1.5)
            self.assertEqual(rep_clean.reject_reasons, [])

            silence = root / 'silence.wav'
            sf.write(str(silence), np.zeros(32000, dtype=np.float32), 16000)
            rep_sil = analyze_audio_quality(silence)
            self.assertFalse(rep_sil.passed_enroll_quality)
            self.assertIn('silence', rep_sil.reject_reasons)
            self.assertEqual(rep_sil.active_seconds, 0.0)

            t = np.arange(32000, dtype=np.float32) / 16000.0
            clipped_sig = np.clip(np.sin(2.0 * np.pi * 440.0 * t) * 5.0, -1.0, 1.0).astype(np.float32)
            clipped = root / 'clipped.wav'
            sf.write(str(clipped), clipped_sig, 16000)
            rep_clip = analyze_audio_quality(clipped)
            self.assertFalse(rep_clip.passed_enroll_quality)
            self.assertIn('clipping', rep_clip.reject_reasons)
            self.assertGreater(rep_clip.clipping_ratio, MAX_ENROLL_CLIPPING_RATIO)

    def test_voice_agent_quality_command_and_gates(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpaq_ag_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            allowed.mkdir()
            takes = sandbox_enroll_takes(allowed, 'jia', speaker_index=0)
            silence = allowed / 'silence.wav'
            sf.write(str(silence), np.zeros(32000, dtype=np.float32), 16000)
            t = np.arange(32000, dtype=np.float32) / 16000.0
            clipped_sig = np.clip(np.sin(2.0 * np.pi * 300.0 * t) * 5.0, -1.0, 1.0).astype(np.float32)
            clipped = allowed / 'clipped.wav'
            sf.write(str(clipped), clipped_sig, 16000)

            agent = VoiceAgent(root / 'db', embedder, detail=False, audio_roots=[allowed])
            try:
                q_ok = agent.handle(f'质检 "{takes[0].as_posix()}"')
                self.assertEqual(q_ok['tool'], 'quality')
                self.assertTrue(q_ok['ok'])
                self.assertTrue(q_ok['result']['passed_enroll_quality'])

                # Pure silence enroll -> rejected
                e_sil = agent.handle(f'注册 静音人 本人同意 {quote_paths([silence, takes[1], takes[2]])}')
                self.assertFalse(e_sil['ok'])
                self.assertEqual(e_sil['result'].get('reason'), 'poor_audio_quality')

                # Severely clipped enroll -> rejected
                e_clip = agent.handle(f'注册 削波人 本人同意 {quote_paths([clipped, takes[1], takes[2]])}')
                self.assertFalse(e_clip['ok'])
                self.assertEqual(e_clip['result'].get('reason'), 'poor_audio_quality')

                # Enroll genuine speaker cleanly
                e_ok = agent.handle(f'注册 甲 本人同意 {quote_paths(takes)}')
                self.assertTrue(e_ok['ok'])

                # Verify & identify pure silence -> hard reject, 0 rows in events
                v_sil = agent.handle(f'验证 甲 "{silence.as_posix()}"')
                self.assertFalse(v_sil['ok'])
                self.assertEqual(v_sil['result'].get('decision'), 'reject')
                self.assertEqual(v_sil['result'].get('reason'), 'poor_audio_quality')
                self.assertIn('拒绝', v_sil['reply'])

                i_sil = agent.handle(f'辨认 "{silence.as_posix()}"')
                self.assertFalse(i_sil['ok'])
                self.assertEqual(i_sil['result'].get('decision'), 'reject')
                self.assertEqual(i_sil['result'].get('reason'), 'poor_audio_quality')
                self.assertIn('拒绝', i_sil['reply'])

                ev_count = agent.store.db.execute('select count(*) from events').fetchone()[0]
                self.assertEqual(ev_count, 0)

                # Verify clipped audio is NOT blocked by quality gate (reaches scoring)
                v_clip = agent.handle(f'验证 甲 "{clipped.as_posix()}"')
                self.assertTrue(v_clip['ok'])
                self.assertIn(v_clip['result'].get('decision'), ('accept', 'reject', 'uncertain'))
            finally:
                agent.close()


if __name__ == '__main__':
    unittest.main()
