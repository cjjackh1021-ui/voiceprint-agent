"""Cross-platform contained subprocess tree ownership (POSIX PGID + Windows Job Object)."""
import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ContainedRunResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool


class _WindowsJobHandle:
    """Minimal zero-dependency Win32 Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE."""

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    def __init__(self):
        self._handle = None
        if sys.platform != 'win32':
            return
        from ctypes import wintypes

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_uint64),
                ('WriteOperationCount', ctypes.c_uint64),
                ('OtherOperationCount', ctypes.c_uint64),
                ('ReadTransferCount', ctypes.c_uint64),
                ('WriteTransferCount', ctypes.c_uint64),
                ('OtherTransferCount', ctypes.c_uint64),
            ]

        class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_int64),
                ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', _IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        self._kernel32 = kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(job)
            return
        self._handle = job

    def assign(self, proc: subprocess.Popen) -> None:
        if self._handle and sys.platform == 'win32':
            proc_handle = getattr(proc, '_handle', None)
            if proc_handle:
                self._kernel32.AssignProcessToJobObject(self._handle, int(proc_handle))

    def close(self) -> None:
        if self._handle and sys.platform == 'win32':
            handle, self._handle = self._handle, None
            self._kernel32.CloseHandle(handle)


class ContainedProcess:
    """Launch a subprocess inside an owned process group (POSIX) or Job Object (Windows)."""

    def __init__(
        self,
        cmd: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ):
        self._closed = False
        self._pgid: int | None = None
        self._win_job: _WindowsJobHandle | None = None

        popen_kwargs: dict = {
            'cwd': str(cwd) if cwd is not None else None,
            'env': env,
            'stdin': stdin,
            'stdout': stdout,
            'stderr': stderr,
        }
        if os.name == 'posix':
            popen_kwargs['start_new_session'] = True
        elif sys.platform == 'win32':
            popen_kwargs['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)

        self.proc = subprocess.Popen(list(cmd), **popen_kwargs)
        if os.name == 'posix':
            try:
                self._pgid = os.getpgid(self.proc.pid)
            except ProcessLookupError:
                self._pgid = self.proc.pid
        elif sys.platform == 'win32':
            self._win_job = _WindowsJobHandle()
            self._win_job.assign(self.proc)

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def stdin(self):
        return self.proc.stdin

    @property
    def stdout(self):
        return self.proc.stdout

    @property
    def stderr(self):
        return self.proc.stderr

    @property
    def returncode(self) -> int | None:
        return self.proc.returncode

    def poll(self) -> int | None:
        return self.proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.proc.wait(timeout=timeout)

    def _close_pipes(self) -> None:
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass

    def terminate_tree(self, grace_s: float = 0.2) -> int:
        """Terminate the entire process subtree (including grandchildren) and reap cleanly."""
        if self._closed:
            return self.proc.returncode if self.proc.returncode is not None else -1

        if os.name == 'posix' and self._pgid is not None:
            try:
                os.killpg(self._pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                if self.proc.poll() is None:
                    try:
                        self.proc.terminate()
                    except OSError:
                        pass
            deadline = time.monotonic() + max(0.0, float(grace_s))
            while self.proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            # Always SIGKILL the process group to clean up surviving grandchildren even if leader exited.
            try:
                os.killpg(self._pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                if self.proc.poll() is None:
                    try:
                        self.proc.kill()
                    except OSError:
                        pass
        elif sys.platform == 'win32':
            if self._win_job is not None:
                self._win_job.close()
            if self.proc.poll() is None:
                try:
                    self.proc.kill()
                except OSError:
                    pass

        try:
            rc = self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                self.proc.kill()
            except OSError:
                pass
            rc = self.proc.wait(timeout=2.0)

        self._close_pipes()
        self._closed = True
        return int(rc)

    def __enter__(self) -> 'ContainedProcess':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.terminate_tree()


def _drain_stream(stream, bucket: list[bytes]) -> None:
    if stream is None:
        return
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            bucket.append(chunk)
    except (OSError, ValueError):
        pass


def run_contained(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> ContainedRunResult:
    """Execute a command in a contained process tree, reaping all descendants on timeout."""
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    timed_out = False

    with ContainedProcess(cmd, cwd=cwd, env=env) as cp:
        t_out = threading.Thread(target=_drain_stream, args=(cp.stdout, out_chunks), daemon=True)
        t_err = threading.Thread(target=_drain_stream, args=(cp.stderr, err_chunks), daemon=True)
        t_out.start()
        t_err.start()

        if cp.stdin is not None:
            try:
                if input_bytes:
                    cp.stdin.write(input_bytes)
                cp.stdin.close()
            except OSError:
                pass

        try:
            rc = cp.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = cp.terminate_tree()
        else:
            # Ensure any lingering background grandchildren in the group are cleaned up too.
            cp.terminate_tree(grace_s=0.05)

        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)

    return ContainedRunResult(
        returncode=int(rc),
        stdout=b''.join(out_chunks),
        stderr=b''.join(err_chunks),
        timed_out=timed_out,
    )
