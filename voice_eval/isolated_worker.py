"""Subprocess-isolated speaker embedder worker with auto-restart, circuit breaker and RSS reporting."""
import argparse
import base64
import json
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np

from voice_eval.dims.perf import current_rss_mb
from voice_eval.offline import strict_offline_env
from voiceprint.contained_subprocess import ContainedProcess
from voiceprint.embed import Embedder


class IsolatedWorkerError(RuntimeError):
    """Raised when an isolated embedder worker crashes, times out, or returns an invalid reply."""


class WorkerCircuitOpenError(IsolatedWorkerError):
    """Raised once consecutive worker failures reach max_consecutive_failures."""


def _reader_loop(stream, out_queue: queue.Queue) -> None:
    try:
        while True:
            line = stream.readline()
            if not line:
                out_queue.put(None)
                break
            out_queue.put(line)
    except (OSError, ValueError):
        out_queue.put(None)


def _drain_stderr(stream, sink: list[bytes]) -> None:
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            if len(sink) < 16:
                sink.append(chunk)
    except (OSError, ValueError):
        pass


class IsolatedEmbedder(Embedder):
    """Embedder proxy executing inference inside a contained worker subprocess."""

    is_isolated = True

    def __init__(
        self,
        embedder_name: str,
        *,
        checkpoint: str | Path | None = None,
        timeout_s: float = 30.0,
        max_consecutive_failures: int = 3,
        python_executable: str | None = None,
        worker_cmd: list[str] | None = None,
    ):
        super().__init__()
        self._embedder_name = str(embedder_name)
        self._checkpoint_arg = str(checkpoint) if checkpoint is not None else None
        self.timeout_s = float(timeout_s)
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self._python_exe = python_executable or sys.executable
        self._worker_cmd = list(worker_cmd) if worker_cmd is not None else None

        self._proc: ContainedProcess | None = None
        self._reply_q: queue.Queue | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_buf: list[bytes] = []
        self._lock = threading.RLock()

        self.peak_worker_rss_mb: float | None = None
        self.restart_count = 0
        self.timeout_count = 0
        self.consecutive_failures = 0
        self.circuit_open = False

        # Start worker once to fetch authoritative describe() metadata.
        self._start_worker()
        meta = self._rpc({'op': 'describe'})
        desc = meta.get('describe') or {}
        self.name = str(desc.get('embedder') or self._embedder_name)
        self.dim = int(desc.get('dim') or 0)
        self.checkpoint = desc.get('checkpoint')
        self.model_id = str(desc.get('model_id') or '')
        self.cohort_size = int(desc.get('cohort_size') or 0)

    def _build_cmd(self) -> list[str]:
        if self._worker_cmd is not None:
            return list(self._worker_cmd)
        cmd = [self._python_exe, '-m', 'voice_eval.isolated_worker', '--embedder', self._embedder_name]
        if self._checkpoint_arg:
            cmd.extend(['--checkpoint', self._checkpoint_arg])
        return cmd

    def _start_worker(self) -> None:
        env = dict(os.environ)
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'
        env['HF_DATASETS_OFFLINE'] = '1'
        root_dir = str(Path(__file__).resolve().parents[1])
        existing_pp = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = root_dir if not existing_pp else (root_dir + os.pathsep + existing_pp)

        self._stderr_buf = []
        self._reply_q = queue.Queue()
        self._proc = ContainedProcess(self._build_cmd(), cwd=root_dir, env=env)
        self._reader_thread = threading.Thread(
            target=_reader_loop,
            args=(self._proc.stdout, self._reply_q),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(self._proc.stderr, self._stderr_buf),
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()

    def _teardown_worker(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate_tree(grace_s=0.15)
        for t in (self._reader_thread, self._stderr_thread):
            if t is not None:
                t.join(timeout=0.5)
        self._reader_thread = None
        self._stderr_thread = None
        self._reply_q = None

    def _record_rss(self, rss_val) -> None:
        if rss_val is None:
            return
        val = float(rss_val)
        if np.isfinite(val) and val > 0.0:
            if self.peak_worker_rss_mb is None or val > self.peak_worker_rss_mb:
                self.peak_worker_rss_mb = val

    def _handle_failure(self, message: str, *, timed_out: bool = False) -> None:
        self._teardown_worker()
        self.restart_count += 1
        if timed_out:
            self.timeout_count += 1
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.circuit_open = True
            raise WorkerCircuitOpenError(
                f'isolated worker circuit open after {self.consecutive_failures} consecutive failures: {message}'
            )
        raise IsolatedWorkerError(message)

    def _rpc(self, request: dict) -> dict:
        with self._lock:
            if self.circuit_open:
                raise WorkerCircuitOpenError('isolated worker circuit breaker is already open')
            if self._proc is None or self._proc.poll() is not None:
                self._teardown_worker()
                self._start_worker()

            assert self._proc is not None and self._proc.stdin is not None and self._reply_q is not None
            payload = (json.dumps(request, ensure_ascii=False) + '\n').encode('utf-8')
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
            except OSError as exc:
                self._handle_failure(f'failed writing to worker stdin: {exc}')

            try:
                raw_line = self._reply_q.get(timeout=self.timeout_s)
            except queue.Empty:
                self._handle_failure(f'worker timed out after {self.timeout_s:.2f}s', timed_out=True)

            if raw_line is None:
                err_tail = b''.join(self._stderr_buf).decode('utf-8', 'replace').strip()[-300:]
                self._handle_failure(f'worker exited unexpectedly ({err_tail or "EOF"})')

            try:
                resp = json.loads(raw_line.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._handle_failure(f'invalid JSON reply from worker: {exc}')

            if not isinstance(resp, dict) or not resp.get('ok'):
                err_msg = resp.get('error', 'unknown worker error') if isinstance(resp, dict) else str(resp)
                self._handle_failure(f'worker reported error: {err_msg}')

            self._record_rss(resp.get('rss_mb'))
            self.consecutive_failures = 0
            return resp

    def embed_file(self, path: str) -> np.ndarray:
        if self.circuit_open:
            raise WorkerCircuitOpenError('isolated worker circuit breaker is already open')
        return super().embed_file(path)

    def _embed(self, path: str) -> np.ndarray:
        resp = self._rpc({'op': 'embed', 'path': str(path)})
        raw = base64.b64decode(resp['vec_b64'])
        dim = int(resp.get('dim') or self.dim)
        vec = np.frombuffer(raw, dtype=np.float32).copy()
        if vec.shape != (dim,) or not bool(np.all(np.isfinite(vec))):
            self._handle_failure(f'worker returned invalid embedding shape={vec.shape} expected=({dim},)')
        return vec

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.write(b'{"op": "shutdown"}\n')
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=0.5)
                except Exception:
                    pass
            self._teardown_worker()

    def __enter__(self) -> 'IsolatedEmbedder':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _load_worker_embedder(name: str, checkpoint: str | None):
    if name == 'tone':
        from voice_eval.dims._sandbox import ToneEmbedder

        return ToneEmbedder()
    if name == 'vpnet' and checkpoint:
        from voiceprint.embed import VPNetEmbedder

        return VPNetEmbedder(checkpoint)
    from voiceprint.embed import load_embedder

    return load_embedder(name)


def _worker_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Isolated voiceprint embedder subprocess worker')
    parser.add_argument('--embedder', required=True, choices=['stats', 'vpnet', 'ecapa', 'tone'])
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args(argv)

    with strict_offline_env():
        embedder = _load_worker_embedder(args.embedder, args.checkpoint)
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            try:
                req = json.loads(text)
                op = req.get('op')
                if op == 'describe':
                    desc = dict(embedder.describe())
                    desc['cohort_size'] = int(getattr(embedder, 'cohort_size', 0))
                    reply = {'ok': True, 'describe': desc, 'rss_mb': current_rss_mb()}
                elif op == 'embed':
                    vec = np.ascontiguousarray(embedder._embed(str(req['path'])), dtype=np.float32)
                    reply = {
                        'ok': True,
                        'vec_b64': base64.b64encode(vec.tobytes()).decode('ascii'),
                        'dim': int(vec.shape[0]),
                        'rss_mb': current_rss_mb(),
                    }
                elif op == 'shutdown':
                    sys.stdout.write(json.dumps({'ok': True, 'rss_mb': current_rss_mb()}) + '\n')
                    sys.stdout.flush()
                    break
                else:
                    reply = {'ok': False, 'error': f'unknown op: {op!r}'}
            except Exception as exc:
                reply = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + '\n')
            sys.stdout.flush()


if __name__ == '__main__':
    _worker_main()
