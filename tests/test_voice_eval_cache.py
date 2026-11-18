"""Tests for content-hashed disk and memory embedding cache."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from voice_eval.cache import CachedEmbedder, DEFAULT_CACHE_DIR
from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav
from voiceprint.audio import file_digest


class CacheTests(unittest.TestCase):
    def test_disk_hit_persistence_and_model_id_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / 'cache'
            wav_path = generate_speaker_wav(root / 'spk0.wav', speaker_index=0, duration=2.0)
            wav_copy = root / 'spk0_copy.wav'
            shutil.copyfile(wav_path, wav_copy)
            digest = file_digest(str(wav_path))

            tone1 = ToneEmbedder()
            cached1 = CachedEmbedder(tone1, cache_dir=cache_dir)
            self.assertEqual(DEFAULT_CACHE_DIR.name, 'cache')
            self.assertEqual(cached1.name, tone1.name)
            self.assertEqual(cached1.dim, tone1.dim)
            self.assertEqual(cached1.checkpoint, tone1.checkpoint)
            self.assertEqual(cached1.model_id, tone1.model_id)
            self.assertEqual(cached1.cohort_size, tone1.cohort_size)
            self.assertEqual(cached1.describe(), tone1.describe())

            with patch.object(tone1, '_embed', wraps=tone1._embed) as spy1:
                vec1 = cached1.embed_file(str(wav_path))
                self.assertEqual(spy1.call_count, 1)
                self.assertEqual(vec1.shape, (tone1.dim,))
                self.assertEqual(vec1.dtype, np.float32)

                stats_after_miss = cached1.stats()
                self.assertEqual(stats_after_miss['misses'], 1)
                self.assertEqual(stats_after_miss['hits'], 0)
                self.assertEqual(stats_after_miss['total'], 1)
                self.assertEqual(stats_after_miss['hit_rate'], 0.0)
                self.assertEqual(len(stats_after_miss['compute_ms_list']), 1)
                self.assertEqual(stats_after_miss['lookup_ms_list'], [])

                # Second call on identical content at a copied path must hit memory cache.
                vec_mem = cached1.embed_file(str(wav_copy))
                self.assertEqual(spy1.call_count, 1)
                np.testing.assert_array_equal(vec_mem, vec1)
                # Mutating returned vector must not corrupt cached memory copy.
                vec_mem[0] = 999.0
                vec_mem_again = cached1.embed_file(str(wav_path))
                np.testing.assert_array_equal(vec_mem_again, vec1)

                stats_after_mem_hits = cached1.stats()
                self.assertEqual(stats_after_mem_hits['hits'], 2)
                self.assertEqual(stats_after_mem_hits['misses'], 1)
                self.assertEqual(stats_after_mem_hits['total'], 3)
                self.assertAlmostEqual(stats_after_mem_hits['hit_rate'], 2.0 / 3.0)
                self.assertEqual(len(stats_after_mem_hits['lookup_ms_list']), 2)

                cached1.reset_stats()
                self.assertEqual(
                    cached1.stats(),
                    {
                        'enabled': True,
                        'hits': 0,
                        'misses': 0,
                        'total': 0,
                        'hit_rate': 0.0,
                        'compute_ms_list': [],
                        'lookup_ms_list': [],
                    },
                )
                # reset_stats clears counters only, not cached entries.
                np.testing.assert_array_equal(cached1.embed_file(str(wav_path)), vec1)
                self.assertEqual(spy1.call_count, 1)
                self.assertEqual(cached1.stats()['hits'], 1)

            expected_path = cache_dir / 'tone' / tone1.model_id[:16] / f'{digest}.npy'
            self.assertTrue(expected_path.exists())
            self.assertEqual(list(cache_dir.rglob('*.tmp*')), [])

            # Fresh wrapper + fresh inner embedder must hit disk without calling inner._embed.
            tone2 = ToneEmbedder()
            cached2 = CachedEmbedder(tone2, cache_dir=cache_dir)
            with patch.object(tone2, '_embed', wraps=tone2._embed) as spy2:
                vec2 = cached2.embed_file(str(wav_path))
                self.assertEqual(spy2.call_count, 0)
                np.testing.assert_array_equal(vec2, vec1)
                self.assertEqual(cached2.stats()['hits'], 1)
                self.assertEqual(cached2.stats()['misses'], 0)

                # Changing inner.model_id must miss and isolate the cached .npy path.
                tone2.model_id = 'tone-v2-' + 'b' * 56
                self.assertEqual(cached2.model_id, tone2.model_id)
                self.assertEqual(cached2.describe()['model_id'], tone2.model_id)
                vec3 = cached2.embed_file(str(wav_path))
                self.assertEqual(spy2.call_count, 1)
                np.testing.assert_array_equal(vec3, vec1)
                self.assertEqual(cached2.stats()['hits'], 1)
                self.assertEqual(cached2.stats()['misses'], 1)

            alt_path = cache_dir / 'tone' / tone2.model_id[:16] / f'{digest}.npy'
            self.assertTrue(alt_path.exists())
            self.assertNotEqual(alt_path, expected_path)
            self.assertTrue(expected_path.exists())
            self.assertEqual(list(cache_dir.rglob('*.tmp*')), [])

    def test_corrupted_or_wrong_dim_cache_file_recomputes_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / 'cache'
            wav_path = generate_speaker_wav(root / 'spk1.wav', speaker_index=1, duration=2.0)
            digest = file_digest(str(wav_path))
            baseline = ToneEmbedder().embed_file(str(wav_path))

            def _write_garbage(p: Path):
                p.write_bytes(b'corrupted-not-a-numpy-file')

            def _write_empty(p: Path):
                p.write_bytes(b'')

            def _write_truncated_npy(p: Path):
                tmp_valid = p.with_suffix('.valid.npy')
                np.save(tmp_valid, baseline, allow_pickle=False)
                raw = tmp_valid.read_bytes()
                tmp_valid.unlink()
                p.write_bytes(raw[:len(raw) // 2])

            def _write_wrong_dim(p: Path):
                np.save(p, np.zeros((ToneEmbedder.dim + 3,), dtype=np.float32), allow_pickle=False)

            def _write_wrong_dtype(p: Path):
                np.save(p, np.zeros((ToneEmbedder.dim,), dtype=np.float64), allow_pickle=False)

            def _write_nan(p: Path):
                np.save(p, np.full((ToneEmbedder.dim,), np.nan, dtype=np.float32), allow_pickle=False)

            def _write_npz_as_npy(p: Path):
                tmp_npz = p.with_suffix('.npz')
                np.savez(tmp_npz, emb=baseline)
                tmp_npz.replace(p)

            def _write_truncated_npz(p: Path):
                p.write_bytes(b'PK\x03\x04\x14\x00trunc')

            def _write_pickle_obj(p: Path):
                np.save(p, np.array([{'bad': True}], dtype=object), allow_pickle=True)

            corruptors = [
                ('garbage', _write_garbage),
                ('empty', _write_empty),
                ('truncated_npy', _write_truncated_npy),
                ('wrong_dim', _write_wrong_dim),
                ('wrong_dtype', _write_wrong_dtype),
                ('nan', _write_nan),
                ('npz_as_npy', _write_npz_as_npy),
                ('truncated_npz', _write_truncated_npz),
                ('pickle_obj', _write_pickle_obj),
            ]

            for label, corrupt_fn in corruptors:
                with self.subTest(case=label):
                    case_cache = cache_dir / label
                    tone = ToneEmbedder()
                    cached = CachedEmbedder(tone, cache_dir=case_cache)
                    expected_path = case_cache / tone.name / tone.model_id[:16] / f'{digest}.npy'
                    expected_path.parent.mkdir(parents=True, exist_ok=True)
                    corrupt_fn(expected_path)

                    with patch.object(tone, '_embed', wraps=tone._embed) as spy:
                        repaired = cached.embed_file(str(wav_path))
                        self.assertEqual(spy.call_count, 1)

                    self.assertEqual(repaired.shape, (tone.dim,))
                    self.assertEqual(repaired.dtype, np.float32)
                    self.assertTrue(np.all(np.isfinite(repaired)))
                    np.testing.assert_array_equal(repaired, baseline)
                    self.assertEqual(cached.stats()['misses'], 1)
                    self.assertEqual(cached.stats()['hits'], 0)
                    self.assertEqual(list(case_cache.rglob('*.tmp*')), [])

                    # Next fresh wrapper must cleanly load the repaired cache file from disk.
                    fresh = CachedEmbedder(ToneEmbedder(), cache_dir=case_cache)
                    loaded = fresh.embed_file(str(wav_path))
                    np.testing.assert_array_equal(loaded, baseline)
                    self.assertEqual(fresh.stats()['hits'], 1)
                    self.assertEqual(fresh.stats()['misses'], 0)

    def test_disabled_cache_bypasses_read_and_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / 'cache'
            wav_path = generate_speaker_wav(root / 'spk2.wav', speaker_index=2, duration=2.0)

            tone = ToneEmbedder()
            disabled = CachedEmbedder(tone, cache_dir=cache_dir, enabled=False)
            with patch.object(tone, 'embed_file', wraps=tone.embed_file) as spy_ef:
                v1 = disabled.embed_file(str(wav_path))
                v2 = disabled.embed_file(str(wav_path))
                self.assertEqual(spy_ef.call_count, 2)

            np.testing.assert_array_equal(v1, v2)
            stats = disabled.stats()
            self.assertFalse(stats['enabled'])
            self.assertEqual(stats['misses'], 2)
            self.assertEqual(stats['hits'], 0)
            self.assertEqual(stats['total'], 2)
            self.assertEqual(stats['hit_rate'], 0.0)
            self.assertEqual(len(stats['compute_ms_list']), 2)
            self.assertEqual(stats['lookup_ms_list'], [])
            self.assertFalse(cache_dir.exists())

            # Even if disk cache is populated later, enabled=False still bypasses read & mem.
            warm = CachedEmbedder(ToneEmbedder(), cache_dir=cache_dir, enabled=True)
            warm.embed_file(str(wav_path))
            self.assertEqual(len(list(cache_dir.rglob('*.npy'))), 1)

            with patch.object(tone, 'embed_file', wraps=tone.embed_file) as spy_ef2:
                disabled.embed_file(str(wav_path))
                self.assertEqual(spy_ef2.call_count, 1)
            self.assertEqual(disabled.stats()['misses'], 3)
            self.assertEqual(disabled.stats()['hits'], 0)

    def test_memory_cache_bounded_and_disk_fallback_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / 'cache'
            wavs = [
                generate_speaker_wav(root / f's{i}.wav', speaker_index=i, duration=1.5)
                for i in range(3)
            ]
            tone = ToneEmbedder()
            cached = CachedEmbedder(tone, cache_dir=cache_dir)

            with patch('voice_eval.cache.CACHE_LIMIT', 2), patch.object(tone, '_embed', wraps=tone._embed) as spy:
                v0 = cached.embed_file(str(wavs[0]))
                cached.embed_file(str(wavs[1]))
                cached.embed_file(str(wavs[2]))
                self.assertLessEqual(len(cached._mem), 2)
                self.assertEqual(spy.call_count, 3)

                # Evicted entry from _mem must still hit disk without invoking inner._embed.
                v0_disk = cached.embed_file(str(wavs[0]))
                self.assertEqual(spy.call_count, 3)
                np.testing.assert_array_equal(v0_disk, v0)
                self.assertEqual(cached.stats()['hits'], 1)
                self.assertEqual(cached.stats()['misses'], 3)

    def test_none_model_id_and_non_finite_embedding_error_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / 'cache'
            wav_path = generate_speaker_wav(root / 'spk_nf.wav', speaker_index=0, duration=1.5)
            tone = ToneEmbedder()
            tone.model_id = None
            cached = CachedEmbedder(tone, cache_dir=cache_dir)
            vec = cached.embed_file(str(wav_path))
            self.assertEqual(vec.shape, (tone.dim,))
            digest = file_digest(str(wav_path))
            self.assertTrue((cache_dir / tone.name / 'unknown' / f'{digest}.npy').is_file())

            bad_vec = np.zeros(tone.dim, dtype=np.float32)
            bad_vec[0] = np.nan
            disabled = CachedEmbedder(tone, cache_dir=cache_dir, enabled=False)
            with patch.object(tone, 'embed_file', return_value=bad_vec):
                with self.assertRaises(ValueError) as ctx:
                    disabled.embed_file(str(wav_path))
            self.assertIn('non-finite', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
