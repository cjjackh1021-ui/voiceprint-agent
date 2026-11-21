"""Tests for cross-platform contained subprocess tree, offline audit, and isolated eval worker."""
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav
from voice_eval.isolated_worker import (
    IsolatedEmbedder,
    IsolatedWorkerError,
    WorkerCircuitOpenError,
)
from voice_eval.offline import collect_model_fingerprint, strict_offline_env
from voiceprint.contained_subprocess import ContainedProcess, run_contained


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class TestContainedSubprocessAndWorker(unittest.TestCase):
    def test_run_contained_timeout_reaps_grandchild_process_tree(self):
        if os.name != 'posix':
            self.skipTest('POSIX grandchild PID check')
        child_code = (
            "import subprocess, sys, time\n"
            "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print(gc.pid, flush=True)\n"
            "time.sleep(30)\n"
        )
        res = run_contained([sys.executable, '-c', child_code], timeout=0.6)
        self.assertTrue(res.timed_out)
        lines = [ln.strip() for ln in res.stdout.decode('utf-8', 'replace').splitlines() if ln.strip()]
        self.assertTrue(lines, f'expected grandchild pid on stdout, got stderr={res.stderr!r}')
        grandchild_pid = int(lines[0])
        # Give OS a short moment to finalize reaping after SIGKILL.
        deadline = time.monotonic() + 1.0
        while _pid_alive(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(_pid_alive(grandchild_pid), f'grandchild PID {grandchild_pid} survived terminate_tree!')

    def test_strict_offline_env_sets_and_restores_vars_and_fingerprints(self):
        keys = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE')
        before = {k: os.environ.get(k) for k in keys}
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        try:
            with strict_offline_env():
                for k in keys:
                    self.assertEqual(os.environ.get(k), '1')
            self.assertEqual(os.environ.get('HF_HUB_OFFLINE'), '0')
            self.assertIsNone(os.environ.get('TRANSFORMERS_OFFLINE'))

            # Restores cleanly even when body raises.
            with self.assertRaises(RuntimeError):
                with strict_offline_env():
                    self.assertEqual(os.environ.get('HF_HUB_OFFLINE'), '1')
                    raise RuntimeError('boom')
            self.assertEqual(os.environ.get('HF_HUB_OFFLINE'), '0')
        finally:
            for k, val in before.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val

        fp = collect_model_fingerprint(ToneEmbedder())
        self.assertTrue(fp['offline_enforced'])
        self.assertTrue(bool(fp['model_id']))
        self.assertEqual(fp['embedder'], 'tone')

        class _BadEmb:
            name = 'bad'
            model_id = ''
            checkpoint = None

            def describe(self):
                return {'embedder': 'bad', 'model_id': '', 'checkpoint': None}

        with self.assertRaises(ValueError):
            collect_model_fingerprint(_BadEmb())

    def test_isolated_embedder_tone_roundtrip_and_worker_rss(self):
        local = ToneEmbedder()
        with TemporaryDirectory(prefix='vpiso_ok_') as tmp:
            root = Path(tmp)
            w1 = generate_speaker_wav(root / 'a.wav', speaker_index=0, duration=2.0, seed=1)
            w2 = generate_speaker_wav(root / 'b.wav', speaker_index=1, duration=2.0, seed=2)
            with IsolatedEmbedder('tone', timeout_s=15.0) as iso:
                self.assertTrue(iso.is_isolated)
                self.assertEqual(iso.name, local.name)
                self.assertEqual(iso.dim, local.dim)
                self.assertEqual(iso.model_id, local.model_id)
                v1 = iso.embed_file(str(w1))
                v2 = iso.embed_file(str(w2))
                self.assertTrue(np.allclose(v1, local.embed_file(str(w1)), atol=1e-6))
                self.assertTrue(np.allclose(v2, local.embed_file(str(w2)), atol=1e-6))
                self.assertIsNotNone(iso.peak_worker_rss_mb)
                self.assertGreater(iso.peak_worker_rss_mb, 0.0)
                self.assertEqual(iso.restart_count, 0)
                self.assertFalse(iso.circuit_open)

    def test_isolated_embedder_recovers_single_crash_and_trips_circuit_breaker(self):
        flaky_script = (
            "import base64, json, os, sys, time, numpy as np\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    op = req.get('op')\n"
            "    if op == 'describe':\n"
            "        print(json.dumps({'ok': True, 'describe': {'embedder': 'flaky', 'dim': 4, 'checkpoint': None, 'model_id': 'flaky-v1', 'cohort_size': 0}, 'rss_mb': 12.5}), flush=True)\n"
            "    elif op == 'embed':\n"
            "        p = req['path']\n"
            "        if 'crash' in p:\n"
            "            sys.exit(2)\n"
            "        if 'hang' in p:\n"
            "            time.sleep(10)\n"
            "        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)\n"
            "        print(json.dumps({'ok': True, 'vec_b64': base64.b64encode(vec.tobytes()).decode('ascii'), 'dim': 4, 'rss_mb': 15.0}), flush=True)\n"
            "    elif op == 'shutdown':\n"
            "        print(json.dumps({'ok': True}), flush=True)\n"
            "        break\n"
        )
        with TemporaryDirectory(prefix='vpiso_cb_') as tmp:
            root = Path(tmp)
            script_path = root / 'flaky_worker.py'
            script_path.write_text(flaky_script, encoding='utf-8')
            ok_wav = generate_speaker_wav(root / 'good.wav', speaker_index=0, duration=1.5, seed=0)
            crash_wav = generate_speaker_wav(root / 'crash.wav', speaker_index=0, duration=1.5, seed=1)
            hang_wav = generate_speaker_wav(root / 'hang.wav', speaker_index=0, duration=1.5, seed=2)

            with IsolatedEmbedder(
                'flaky',
                timeout_s=0.4,
                max_consecutive_failures=3,
                worker_cmd=[sys.executable, str(script_path)],
            ) as iso:
                # 1. Single crash recovers on next call
                with self.assertRaises(IsolatedWorkerError):
                    iso.embed_file(str(crash_wav))
                self.assertEqual(iso.restart_count, 1)
                self.assertFalse(iso.circuit_open)

                vec = iso.embed_file(str(ok_wav))
                self.assertEqual(vec.shape, (4,))
                self.assertEqual(iso.consecutive_failures, 0)

                # 2. Three consecutive failures (crash + hang + crash) trip circuit breaker
                with self.assertRaises(IsolatedWorkerError):
                    iso.embed_file(str(crash_wav))
                with self.assertRaises(IsolatedWorkerError):
                    iso.embed_file(str(hang_wav))
                self.assertGreaterEqual(iso.timeout_count, 1)
                with self.assertRaises(WorkerCircuitOpenError):
                    iso.embed_file(str(crash_wav))
                self.assertTrue(iso.circuit_open)

                # Subsequent call fails immediately with WorkerCircuitOpenError without restarting
                restarts_before = iso.restart_count
                with self.assertRaises(WorkerCircuitOpenError):
                    iso.embed_file(str(ok_wav))
                self.assertEqual(iso.restart_count, restarts_before)

    def test_isolated_embedder_rejects_after_close_and_counts_malformed_vec_failures(self):
        bad_vec_script = (
            "import base64, json, sys, numpy as np\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    op = req.get('op')\n"
            "    if op == 'describe':\n"
            "        print(json.dumps({'ok': True, 'describe': {'embedder': 'badvec', 'dim': 8, 'checkpoint': None, 'model_id': 'badvec-v1', 'cohort_size': 0}, 'rss_mb': 10.0}), flush=True)\n"
            "    elif op == 'embed':\n"
            "        p = req['path']\n"
            "        if 'wrongdim' in p:\n"
            "            vec = np.zeros(3, dtype=np.float32)\n"
            "            print(json.dumps({'ok': True, 'vec_b64': base64.b64encode(vec.tobytes()).decode('ascii'), 'dim': 3, 'rss_mb': 10.0}), flush=True)\n"
            "        else:\n"
            "            print(json.dumps({'ok': True, 'vec_b64': '!!!not-valid-b64!!!', 'dim': 8, 'rss_mb': 10.0}), flush=True)\n"
            "    elif op == 'shutdown':\n"
            "        print(json.dumps({'ok': True}), flush=True)\n"
            "        break\n"
        )
        with TemporaryDirectory(prefix='vpiso_close_') as tmp:
            root = Path(tmp)
            script = root / 'badvec_worker.py'
            script.write_text(bad_vec_script, encoding='utf-8')
            w_wrong = generate_speaker_wav(root / 'wrongdim.wav', speaker_index=0, duration=1.5, seed=0)
            w_corrupt = generate_speaker_wav(root / 'corrupt.wav', speaker_index=0, duration=1.5, seed=1)

            # 1. Closed embedder must refuse calls and NEVER spawn a new worker subprocess
            iso_closed = IsolatedEmbedder('tone', timeout_s=10.0)
            iso_closed.close()
            with self.assertRaises(IsolatedWorkerError):
                iso_closed.embed_file(str(w_wrong))
            self.assertIsNone(iso_closed._proc)

            # 2. Wrong dim & malformed base64 must route through _handle_failure and trip circuit breaker
            with IsolatedEmbedder(
                'badvec',
                timeout_s=5.0,
                max_consecutive_failures=2,
                worker_cmd=[sys.executable, str(script)],
            ) as iso_bad:
                with self.assertRaises(IsolatedWorkerError):
                    iso_bad.embed_file(str(w_wrong))
                self.assertEqual(iso_bad.restart_count, 1)
                with self.assertRaises(WorkerCircuitOpenError):
                    iso_bad.embed_file(str(w_corrupt))
                self.assertTrue(iso_bad.circuit_open)


if __name__ == '__main__':
    unittest.main()
