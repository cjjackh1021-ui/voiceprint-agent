# VoiceStudio-Inspired Hardening, MCP Server & Eval Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five zero-extra-dependency upgrades to `voiceprint-agent` — audio container & DSP quality guardrails, a standard JSON-RPC 2.0 stdio MCP Server, 16-bit spread-spectrum anti-spoofing guardrails, uncached Warm RTF + strict offline fingerprinting, and a cross-platform contained subprocess tree + isolated eval worker — wired into the 7-dimension eval harness and 3-tier gate.

**Architecture:** All five pillars build strictly on Python stdlib plus the existing 5 dependencies (`numpy`, `scipy`, `soundfile`, `cryptography`, `torch`). Core runtime guardrails live in `voiceprint/` (`audio.py`, `audio_quality.py`, `spoof.py`, `contained_subprocess.py`), external agent serving lives in `agent/mcp_server.py`, and evaluation/gating extensions integrate cleanly into `voice_eval/` (`offline.py`, `isolated_worker.py`, `cache.py`, `dims/perf.py`, `dims/redteam.py`, `runner.py`, `gate.py`, `protocol.json`, `__main__.py`) without altering any of the 6 existing test files.

**Tech Stack:** Python 3.11+ (`.venv/bin/python`), stdlib (`json`, `base64`, `struct`, `threading`, `subprocess`, `signal`, `ctypes`, `unittest`), `numpy`, `scipy`, `soundfile`, `cryptography`, `torch`.

**Spec:** `docs/superpowers/specs/2026-11-21-voicestudio-hardening-mcp-eval-design.md`

## Global Constraints

- Zero new packages in `requirements.txt` (strictly stdlib + existing `numpy`, `scipy`, `soundfile`, `cryptography`, `torch`).
- All 44 tests in the 6 existing test files (`tests/test_voice_eval_*.py`) must remain 100% unmodified and passing under `.venv/bin/python -W error -m unittest discover -s tests`.
- `CachedEmbedder.stats()` must keep its exact 7 existing keys (`enabled`, `hits`, `misses`, `total`, `hit_rate`, `compute_ms_list`, `lookup_ms_list`); audio durations are exposed via a separate method `compute_audio_seconds()`.
- `check_audio()` must check `roots` escape BEFORE rejecting in-root symlinks (`Path(text).is_symlink()`), and must NOT reject symlinks on macOS system parent dirs (e.g. `/var` -> `/private/var`).
- `VoiceAgent` defaults to `spoof_detector=None`; MCP Server defaults to `detail=False` and `SpreadSpectrumWatermarkDetector()` enabled (opt-out via `--no-spoof-guard`).
- Rejected audio (`AudioError`, pure silence, `spoof_detected`) is logged only in `store.audit` via `VoiceAgent.handle()`, NEVER inserted into `store.events` (avoiding NULL scores, `_drift()` pollution, and 5-failure lockout DoS).
- `agent/mcp_server.py` must write only newline-delimited JSON-RPC messages to `stdout`; all diagnostics go to `stderr`.
- All new features must be runnable and testable completely offline on macOS using `ToneEmbedder` + synthetic audio, with Windows Job Object code syntactically and structurally complete.

## Review Focus

- **Odd-sized WAV RIFF chunks with 1-byte pad**: valid WAV files with an odd-length metadata chunk (e.g. `LIST`/`INFO` of 11 bytes + 1 pad byte) must pass `_validate_wav_container()`, while a truncated `data` chunk must fail with `AudioError`.
- **Oversized or malformed Base64 `data:` URI in MCP**: invalid Base64, >32 MiB payload, or valid Base64 containing non-audio text/PNG bytes must be refused cleanly without leaving orphaned temp files on disk.
- **MCP `voiceprint_enroll` with `audio_base64_list` and invalid `session_ids`**: path-traversal characters (`../a`) or mismatched `len(session_ids)` must be refused; omitted `session_ids` must keep all clips in one temp folder so `< 30s` single-session enrollment is still rejected by `VoiceAgent`.
- **Grandchild process survival on POSIX timeout**: a child worker that spawns a sleeping grandchild process must have BOTH child and grandchild PIDs dead after `ContainedProcess.terminate_tree()` / `run_contained(timeout=...)`.
- **Isolated worker circuit breaker propagation through `VoiceAgent.handle()`**: `VoiceAgent.handle()` catches `Exception`, so `run_split()` must inspect `raw_embedder.circuit_open` after every case and raise `WorkerCircuitOpenError` immediately instead of continuing through remaining cases.

---

## File Structure

- **Create**:
  - `voiceprint/audio_quality.py` — 100ms sliding-window model-free DSP audio quality analyzer (`AudioWarning`, `AudioQualityReport`, `analyze_audio_quality`)
  - `voiceprint/spoof.py` — `SpoofResult`, `SpoofDetector`, `embed_spread_watermark`, `SpreadSpectrumWatermarkDetector`, `CompositeSpoofDetector`
  - `voiceprint/contained_subprocess.py` — cross-platform process-tree containment (`ContainedProcess`, `ContainedRunResult`, `run_contained`)
  - `agent/mcp_server.py` — zero-dependency JSON-RPC 2.0 `stdio` MCP Server (`VoiceprintMCPServer`, `main`)
  - `voice_eval/offline.py` — `strict_offline_env`, `collect_model_fingerprint`
  - `voice_eval/isolated_worker.py` — subprocess embedder worker entrypoint + `IsolatedEmbedder`, `IsolatedWorkerError`, `WorkerCircuitOpenError`
  - `tests/test_audio_quality_and_container.py`
  - `tests/test_spoof_guard.py`
  - `tests/test_mcp_server.py`
  - `tests/test_contained_subprocess_and_worker.py`
  - `tests/test_perf_rtf_and_offline.py`
  - `tests/test_redteam_extended.py`
- **Modify**:
  - `voiceprint/audio.py` — `sniff_audio_format`, `_validate_wav_container`, strengthened `check_audio`
  - `agent/service.py` — `spoof_detector` param, pure-silence verify/identify gate, enroll quality gate, `质检` command
  - `voice_eval/audio_synth.py` — `apply_watermark_spoof`, `apply_vocoder_replay_spoof`
  - `voice_eval/cache.py` — `compute_audio_seconds()` (keeping `stats()` 7 keys unchanged)
  - `voice_eval/dims/perf.py` — `measure_uncached_rtf`, extended `summarize_perf` (`cold_embed_ms`, `warm_embed`, `rtf`, `worker_rss_mb`)
  - `voice_eval/dims/__init__.py` — pass `compute_audio_s_list`, `rtf_bench`, `worker_rss_mb` into `summarize_perf`
  - `voice_eval/dims/redteam.py` — extend `_probe_symlink_escape` (in-root + out-of-root in 1 slot) + add `container_integrity`, `mcp_security`, `anti_spoofing` probes + `spoof_summary`
  - `voice_eval/runner.py` — `strict_offline_env`, uncached RTF micro-bench, `isolate_worker` + circuit-breaker check, `runtime_audit` in `summary.json` and Markdown report
  - `voice_eval/gate.py` & `voice_eval/protocol.json` — Tier 3 Warm RTF regression warning (`PERF_RTF_RATIO_WARN = 1.5`, `PERF_RTF_ABS_WARN = 0.05`) + Markdown report row
  - `voice_eval/__main__.py` — `--isolate-worker`, `--worker-timeout`, `--max-worker-failures`

---

### Task 1: Audio Container Integrity & 100ms DSP Signal Quality Gate

**Files:**
- Create: `voiceprint/audio_quality.py`
- Modify: `voiceprint/audio.py:1-76`
- Modify: `agent/service.py:65-248`
- Test: `tests/test_audio_quality_and_container.py`

**Interfaces:**
- Consumes: existing `voiceprint/audio.py` (`SR`, `MIN_SECONDS`, `MAX_SECONDS`, `AudioError`, `check_audio`)
- Produces:
  - `voiceprint.audio.sniff_audio_format(raw: bytes) -> str | None`
  - `voiceprint.audio_quality.MAX_ENROLL_CLIPPING_RATIO = 0.25`
  - `voiceprint.audio_quality.AudioWarning` dataclass (`kind: str`, `start: float`, `end: float`)
  - `voiceprint.audio_quality.AudioQualityReport` dataclass (`duration: float`, `analyzed_seconds: float`, `truncated: bool`, `active_seconds: float`, `silence_ratio: float`, `clipping_ratio: float`, `warnings: list[AudioWarning]`, `passed_enroll_quality: bool`, `reject_reasons: list[str]`) with `.to_dict() -> dict`
  - `voiceprint.audio_quality.analyze_audio_quality(path: str | Path, *, max_seconds: float = 1800.0, max_warnings: int = 100) -> AudioQualityReport`
  - `VoiceAgent` command `质检 <path>` (`tool='quality'`)

- [ ] **Step 1: Write failing tests in `tests/test_audio_quality_and_container.py`**

Tests to write (using `unittest.TestCase` + `TemporaryDirectory` + `ToneEmbedder` + `sandbox_cohort` + `patch_dpapi` from `tests/_helpers.py`):
- `test_sniff_audio_format_magic_bytes`: verifies RIFF/WAVE -> `'.wav'`, `b'fLaC...'` -> `'.flac'`, `b'OggS...'` -> `'.ogg'`, `b'ID3...'` and `b'\xff\xfb\x90\x00'` -> `'.mp3'`, plain text/PNG/empty -> `None`.
- `test_check_audio_rejects_magic_mismatch_truncated_wav_and_inroot_symlink`:
  - valid MP3/text bytes saved as `.wav` -> raises `AudioError` containing `'不匹配'` or `'无效'`;
  - valid 2s WAV with `data` payload truncated by half (header untouched) -> raises `AudioError`;
  - valid WAV with an odd-length 11-byte `INFO` chunk + 1-byte pad before `data` -> passes `check_audio()`;
  - in-root symlink inside `roots` -> raises `AudioError` containing `'符号链接'` (skip symlink subcheck only on `OSError`);
  - out-of-root symlink -> raises `AudioError` containing `'允许的目录'`.
- `test_analyze_audio_quality_detects_silence_clipping_and_clean_audio`:
  - clean `generate_speaker_wav` -> `passed_enroll_quality is True`, `clipping_ratio == 0.0`, `active_seconds >= 1.5`;
  - 2s all-zeros WAV -> `passed_enroll_quality is False`, `'silence'` in `reject_reasons`, `active_seconds == 0.0`;
  - 2s heavily clipped WAV (`np.clip(sine * 5.0, -1.0, 1.0)`) -> `passed_enroll_quality is False`, `'clipping'` in `reject_reasons`, `clipping_ratio > 0.25`.
- `test_voice_agent_quality_command_and_gates`:
  - `agent.handle('质检 "<clean>"')` returns `tool='quality', ok=True`, `result['passed_enroll_quality'] is True`;
  - enrolling pure silence or heavily clipped WAV fails with `ok=False`, `result['reason'] == 'poor_audio_quality'`;
  - verifying/identifying pure silence returns `ok=False`, `result['decision'] == 'reject'`, `result['reason'] == 'poor_audio_quality'`, reply contains `'拒绝'`, and `store.db` has 0 rows in `events`;
  - verifying heavily clipped genuine audio is NOT rejected by the quality gate (reaches scoring normally).

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_audio_quality_and_container -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Implement `sniff_audio_format` and `_validate_wav_container` in `voiceprint/audio.py`**

- Add `sniff_audio_format(raw: bytes) -> str | None` and `_validate_wav_container(resolved: Path) -> None` (parse RIFF header size <= file size, walk sub-chunks with `size + (size & 1)` padding, require `fmt ` and non-empty `data`, verify `data_size >= sf_info_frames * channels * sample_width` for PCM/IEEE-float format tags 1 and 3).
- Update `check_audio(path, roots)` order: empty/UNC text -> suffix -> `os.path.isfile` -> `resolve()` & network & `roots` check -> `Path(text).is_symlink()` check -> `MAX_BYTES` -> read 16-byte header & verify `sniff_audio_format(header) == resolved.suffix.lower()` -> `_validate_wav_container(resolved)` if `.wav` -> `sf.info` duration -> `_assert_finite_audio`.

- [ ] **Step 4: Implement `voiceprint/audio_quality.py`**

Implement `AudioWarning`, `AudioQualityReport`, `MAX_ENROLL_CLIPPING_RATIO = 0.25`, and `analyze_audio_quality(path, *, max_seconds=1800.0, max_warnings=100)` using 100ms `sf.SoundFile` windows, max-channel RMS (`20*log10(max(rms, 1e-12))`), active baseline (`> -50 dBFS` median), contiguous edge aggregation (`invalid`, `clipping`, `silence`, `quiet`, `loud`), chronological sort + 100-warning cap, and enroll-quality verdict (`active_seconds >= MIN_SECONDS`, `clipping_ratio <= MAX_ENROLL_CLIPPING_RATIO`, no `empty`/`invalid`).

- [ ] **Step 5: Integrate quality gate & `质检` command in `agent/service.py`**

- Add `'质检': self._quality` to `commands` in `_dispatch` and document in `HELP`.
- In `_enroll`: after `check_audio()`, call `analyze_audio_quality(p)`; reject with `tool='enroll', ok=False, result={'reason': 'poor_audio_quality', 'quality_reasons': report.reject_reasons}` when `not report.passed_enroll_quality`.
- In `_verify` and `_identify`: after `check_audio()`, call `analyze_audio_quality(p)`; if `report.active_seconds < MIN_SECONDS` or any warning kind in `{'empty', 'invalid'}`, immediately return `self._reply(tool, False, {'decision': 'reject', 'reason': 'poor_audio_quality'}, '拒绝：录音中有效语音不足或无效，请重新录制。')` (without writing `store.events`).

- [ ] **Step 6: Run new and existing tests**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_audio_quality_and_container tests.test_voice_eval_guardrails tests.test_voice_eval_agent_checks -v`
Expected: ALL PASS with 0 warnings.

- [ ] **Step 7: Commit Task 1**

```bash
git add voiceprint/audio.py voiceprint/audio_quality.py agent/service.py tests/test_audio_quality_and_container.py
git commit -m "feat(audio): add container integrity validation and 100ms DSP signal quality gate"
```

---

### Task 2: Anti-Spoofing Watermark Detector, Synthetic Spoof Generators & `VoiceAgent` Guardrail

**Files:**
- Create: `voiceprint/spoof.py`
- Modify: `voice_eval/audio_synth.py:160-184`
- Modify: `agent/service.py:35-51,120-248`
- Test: `tests/test_spoof_guard.py`

**Interfaces:**
- Consumes: `voiceprint.audio.load_wav`, `voiceprint.features.SR`
- Produces:
  - `voiceprint.spoof.DEFAULT_WATERMARK_PAYLOAD = 0x5653`
  - `voiceprint.spoof.DEFAULT_WATERMARK_SEED = 20261121`
  - `voiceprint.spoof.SpoofResult` dataclass (`is_spoof: bool`, `score: float`, `reason: str`, `detector: str`, `metadata: dict`)
  - `voiceprint.spoof.SpoofDetector` base class (`detect(wav: np.ndarray, sr: int) -> SpoofResult`)
  - `voiceprint.spoof.embed_spread_watermark(wav: np.ndarray, sr: int = 16000, *, payload: int = 0x5653, strength: float = 0.02, secret_seed: int = 20261121) -> np.ndarray`
  - `voiceprint.spoof.SpreadSpectrumWatermarkDetector(expected_payload: int | None = None, *, secret_seed: int = 20261121, z_threshold: float = 4.5)`
  - `voiceprint.spoof.CompositeSpoofDetector(detectors: list[SpoofDetector])`
  - `voice_eval.audio_synth.apply_watermark_spoof(wav: np.ndarray, sr: int, *, payload: int = 0x5653, strength: float = 0.02) -> np.ndarray`
  - `voice_eval.audio_synth.apply_vocoder_replay_spoof(wav: np.ndarray, sr: int, *, seed: int = 0) -> np.ndarray`
  - `VoiceAgent(..., spoof_detector: SpoofDetector | None = None)`

- [ ] **Step 1: Write failing tests in `tests/test_spoof_guard.py`**

Tests to write:
- `test_spread_watermark_roundtrip_and_clean_zero_false_positive`:
  - embed `0x5653` onto `generate_speaker_wav` (and on 20 dB noisy speech) -> `SpreadSpectrumWatermarkDetector().detect(wm, 16000)` returns `is_spoof=True`, `reason='watermark_detected'`, `metadata['payload'] == 0x5653`, `score >= 4.5`;
  - clean `generate_speaker_wav` across 5 speaker indices + white noise -> `is_spoof=False` for all 5 (`0%` false positive).
- `test_composite_spoof_detector_short_circuits_on_hit`:
  - chains a custom stub `SpoofDetector` with `SpreadSpectrumWatermarkDetector`, verifies hit returns `is_spoof=True`.
- `test_audio_synth_spoof_generators_deterministic`:
  - `apply_watermark_spoof` and `apply_vocoder_replay_spoof` are deterministic for identical inputs/seeds, return finite `float32` arrays of matching length within `[-1.0, 1.0]`, and `apply_vocoder_replay_spoof` differs meaningfully from clean input.
- `test_voice_agent_spoof_guard_blocks_watermarked_clone_without_leaking_or_writing_events`:
  - With `VoiceAgent(..., detail=False, spoof_detector=SpreadSpectrumWatermarkDetector())`:
    - clean genuine probe passes `verify` and `identify` (`decision='accept'`);
    - watermarked genuine probe fails `verify`, `identify`, and `enroll` with `ok=False`, `decision='reject'`, `reason='spoof_detected'`;
    - `detail=False` reply/result contains NO numeric score/z-score;
    - 6 consecutive watermarked verify probes do NOT write to `store.events` and do NOT lock out the genuine speaker (7th clean genuine probe still returns `accept`).

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_spoof_guard -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceprint.spoof'`.

- [ ] **Step 3: Implement `voiceprint/spoof.py`**

Algorithm for `embed_spread_watermark` & `SpreadSpectrumWatermarkDetector`:
- Resample to 16 kHz if `sr != 16000` (using `scipy.signal.resample_poly`).
- Build deterministic band-limited (`1500–4500 Hz`, 4th-order Butterworth `sosfiltfilt`) unit-RMS orthogonal PN chip bank of 16 chips of length `chip_len = 1000` samples (16,000 samples = 1.0s frame) seeded from `secret_seed`.
- `embed_spread_watermark`: tile the 16-bit bipolar (`±1`) modulated chip frame across `len(wav)`, add `strength * wm_signal` (scaled softly relative to floor `max(rms, 0.05)` so quiet signals stay imperceptible yet detectable), clip to `[-1.0, 1.0]` `float32`.
- `SpreadSpectrumWatermarkDetector.detect`: bandpass-filter input to `1500–4500 Hz`, fold/average into 1.0s frame aligned modulo `16000` (if shorter than 16,000 samples, analyze available full chips), compute normalized correlation of each bit chip against its segment, estimate noise floor std across uncorrelated control chips (from `secret_seed + 1`), compute aggregate signed/magnitude z-score and decoded 16-bit payload. Return `is_spoof = (z_score >= z_threshold) and (expected_payload is None or decoded == expected_payload)`.

- [ ] **Step 4: Add `apply_watermark_spoof` and `apply_vocoder_replay_spoof` to `voice_eval/audio_synth.py`**

- `apply_watermark_spoof`: thin deterministic wrapper calling `embed_spread_watermark`.
- `apply_vocoder_replay_spoof`: deterministic bandpass (`300–3400 Hz`) + short comb-filter resonance (`delay = 19..37` samples seeded by `seed`) + subtle phase-randomized STFT distortion, normalized and clipped to `[-1.0, 1.0]` `float32`.

- [ ] **Step 5: Integrate optional `spoof_detector` into `agent/service.py`**

- Add `spoof_detector=None` to `VoiceAgent.__init__`.
- Add helper `_check_spoof(self, path: str, tool: str) -> dict | None`: when `self.spoof_detector is not None`, decode `load_wav(path)` and call `detect(wav, SR)`. On `is_spoof`, return `self._reply(tool, False, result_dict, '拒绝：检测到疑似合成或水印伪造语音。')` where `result_dict` has `{'decision': 'reject', 'reason': 'spoof_detected'}` (and `detector`/`spoof_score` ONLY when `self.detail` is `True`). Call in `_enroll`, `_verify`, `_identify` after quality check and before embedding.

- [ ] **Step 6: Run tests**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_spoof_guard tests.test_audio_quality_and_container tests.test_voice_eval_guardrails -v`
Expected: ALL PASS with 0 warnings.

- [ ] **Step 7: Commit Task 2**

```bash
git add voiceprint/spoof.py voice_eval/audio_synth.py agent/service.py tests/test_spoof_guard.py
git commit -m "feat(spoof): add pluggable anti-spoofing watermark guardrail and synthetic spoof generators"
```

---

### Task 3: Zero-Dependency JSON-RPC 2.0 `stdio` MCP Server

**Files:**
- Create: `agent/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `agent.service.VoiceAgent`, `voiceprint.audio.sniff_audio_format`, `voiceprint.audio_quality.analyze_audio_quality`, `voiceprint.spoof.SpreadSpectrumWatermarkDetector`
- Produces:
  - `agent.mcp_server.MAX_BASE64_BYTES = 32 * 1024 * 1024`
  - `agent.mcp_server.VoiceprintMCPServer(agent: VoiceAgent, *, audio_roots: list[Path] | None = None, read_only: bool = False)`
    - `.handle_request(message: dict) -> dict | None`
    - `.serve_streams(in_stream, out_stream) -> None`
  - `agent.mcp_server.main(argv: list[str] | None = None) -> None` (CLI `python -m agent.mcp_server --data-dir ... [--audio-roots ...] [--embedder stats|vpnet|ecapa] [--detail] [--no-spoof-guard] [--read-only]`)

- [ ] **Step 1: Write failing tests in `tests/test_mcp_server.py`**

Tests to write:
- `test_mcp_initialize_tools_list_and_resources`:
  - `initialize` returns `protocolVersion == '2024-11-05'`, `capabilities` with `tools` and `resources`;
  - `tools/list` returns all 8 expected tool names with valid JSON schemas;
  - `resources/list` returns `voiceprint://speakers` and `voiceprint://health`; `resources/read` returns valid JSON content without embeddings/secrets.
- `test_mcp_enforces_audio_roots_boundary_and_blocks_escape`:
  - When `audio_roots=None` (or empty), calling `voiceprint_verify(name='甲', audio_path='/tmp/x.wav')` returns structured error `path_not_allowed`;
  - When `audio_roots=[allowed]`, `..` escape outside `allowed` is refused.
- `test_mcp_base64_data_uri_magic_sniffing_and_session_ids`:
  - `data:audio/wav;base64,<valid>` decodes cleanly and cleans up temp files after call;
  - invalid base64, oversized payload (`> MAX_BASE64_BYTES`), or non-audio text/PNG base64 is refused cleanly with 0 leaked temp files;
  - `voiceprint_enroll` with 3 short (3s) base64 clips WITHOUT `session_ids` lands in one dir and is REJECTED by `VoiceAgent`'s 30s single-session rule;
  - `voiceprint_enroll` with 3 short base64 clips WITH `session_ids=['s1', 's2', 's2']` succeeds; invalid `session_ids=['../bad', 's2', 's2']` or mismatched length is refused.
- `test_mcp_two_step_revoke_read_only_and_default_spoof_guard`:
  - Direct `voiceprint_revoke(name='甲', confirm=True)` without prior `confirm=False` is REJECTED and speaker remains enrolled;
  - `confirm=False` followed by `confirm=True` deletes speaker cleanly;
  - `read_only=True` rejects `voiceprint_enroll`, `voiceprint_revoke`, `voiceprint_rotate_key` with `read_only_mode`, while allowing `voiceprint_list_speakers`, `voiceprint_verify`, `voiceprint_check_audio_quality`, `voiceprint_health`;
  - Default server built via helper enables `SpreadSpectrumWatermarkDetector` and `detail=False` (watermarked clone rejected, no score in JSON output).
- `test_mcp_stdio_stream_roundtrip_no_stdout_pollution`:
  - Feeds newline-delimited JSON-RPC (`initialize`, `notifications/initialized`, `tools/call` `voiceprint_health`) into `serve_streams(io.StringIO(...), io.StringIO())` and verifies every output line parses as valid JSON-RPC 2.0 response.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_mcp_server -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.mcp_server'`.

- [ ] **Step 3: Implement `agent/mcp_server.py`**

Key implementation details:
- Use `threading.RLock()` around every tool call accessing `self.agent`.
- `_decode_base64_audio(b64_text: str, target_dir: Path, stem: str) -> Path`: strip `data:[^;]*;base64,` prefix, check encoded size ceiling before decoding, `base64.b64decode(..., validate=True)`, check `len(raw) <= MAX_BASE64_BYTES`, `ext = sniff_audio_format(raw)` (raise `ValueError('无法识别的音频格式')` if `None`), write `target_dir / f'{stem}{ext}'`.
- Validate `session_ids` with regex `^[A-Za-z0-9_-]{1,32}$` and `len(session_ids) == len(audio_base64_list)`.
- For `voiceprint_enroll` / `voiceprint_verify` / `voiceprint_identify` / `voiceprint_check_audio_quality` using Base64 temp dirs: pass temp dir into an ephemeral per-call allowed-root extension ONLY for those verified temp files (never weakening `self.audio_roots` for user-supplied paths!), and clean up `TemporaryDirectory` in `finally`.
- For `voiceprint_revoke(name, confirm=False)`: when `confirm=False`, dispatch `撤销 <name>`; when `confirm=True`, dispatch `确认撤销 <name>` (which `VoiceAgent` automatically rejects unless `self.agent.pending == ('revoke', name)`!).
- `build_default_server(data_dir, embedder, *, audio_roots=None, detail=False, spoof_guard=True, read_only=False) -> VoiceprintMCPServer` helper used by `main()` and tests.
- Configure logging to `sys.stderr` only in `main()`.

- [ ] **Step 4: Run tests**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_mcp_server -v`
Expected: ALL PASS with 0 warnings.

- [ ] **Step 5: Commit Task 3**

```bash
git add agent/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add zero-dependency JSON-RPC 2.0 stdio MCP server with strict security guardrails"
```

---

### Task 4: Cross-Platform Contained Subprocess Tree, Offline Audit & Isolated Eval Worker

**Files:**
- Create: `voiceprint/contained_subprocess.py`
- Create: `voice_eval/offline.py`
- Create: `voice_eval/isolated_worker.py`
- Test: `tests/test_contained_subprocess_and_worker.py`

**Interfaces:**
- Consumes: `voiceprint.audio.file_digest`, `voiceprint.embed.Embedder`, `voice_eval.dims.perf.current_rss_mb`
- Produces:
  - `voiceprint.contained_subprocess.ContainedRunResult` dataclass (`returncode: int`, `stdout: bytes`, `stderr: bytes`, `timed_out: bool`)
  - `voiceprint.contained_subprocess.ContainedProcess(cmd, *, cwd=None, env=None, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)` with `.terminate_tree(grace_s: float = 0.2) -> int`, `.wait(timeout=None) -> int`, context manager
  - `voiceprint.contained_subprocess.run_contained(cmd, *, timeout: float | None = None, cwd=None, env=None, input_bytes: bytes | None = None) -> ContainedRunResult`
  - `voice_eval.offline.strict_offline_env()` context manager
  - `voice_eval.offline.collect_model_fingerprint(embedder) -> dict`
  - `voice_eval.isolated_worker.IsolatedWorkerError(RuntimeError)`
  - `voice_eval.isolated_worker.WorkerCircuitOpenError(IsolatedWorkerError)`
  - `voice_eval.isolated_worker.IsolatedEmbedder(embedder_name: str, *, checkpoint: str | Path | None = None, timeout_s: float = 30.0, max_consecutive_failures: int = 3, python_executable: str | None = None)` with `is_isolated = True`, `.peak_worker_rss_mb`, `.restart_count`, `.timeout_count`, `.circuit_open`, `.close()`

- [ ] **Step 1: Write failing tests in `tests/test_contained_subprocess_and_worker.py`**

Tests to write:
- `test_run_contained_timeout_reaps_grandchild_process_tree`:
  - Spawns a python child that spawns a sleeping grandchild, writes the grandchild PID to stdout, then sleeps 30s;
  - `run_contained(..., timeout=0.5)` returns `timed_out=True`;
  - Verifies via `os.kill(grandchild_pid, 0)` (expecting `ProcessLookupError`) that the grandchild PID is dead (zero orphan processes).
- `test_strict_offline_env_sets_and_restores_vars_and_fingerprints`:
  - Inside `strict_offline_env()`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE` are `'1'`; restored accurately on exit (even on exception);
  - `collect_model_fingerprint(ToneEmbedder())` returns non-empty `model_id`, `offline_enforced=True`; raises `ValueError` if `model_id` is empty.
- `test_isolated_embedder_tone_roundtrip_and_worker_rss`:
  - `IsolatedEmbedder('tone', timeout_s=10.0)` embeds two synthetic WAVs matching in-process `ToneEmbedder` (`np.allclose(..., atol=1e-6)`), populates `peak_worker_rss_mb > 0`, and shuts down cleanly on `close()`.
- `test_isolated_embedder_recovers_single_crash_and_trips_circuit_breaker`:
  - Using a small test worker script (via `worker_cmd` hook or test helper) that crashes/hangs on demand:
    - 1 failure increments `restart_count == 1`, raises `IsolatedWorkerError`, next call succeeds and resets `consecutive_failures == 0`;
    - 3 consecutive failures (`max_consecutive_failures=3`) raises `WorkerCircuitOpenError`, sets `circuit_open=True`, and subsequent calls immediately raise `WorkerCircuitOpenError` without spawning new processes.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_contained_subprocess_and_worker -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `voiceprint/contained_subprocess.py`**

- On POSIX: `subprocess.Popen(..., start_new_session=True)`, `os.getpgid(proc.pid)`, `terminate_tree(grace_s=0.2)` sends `os.killpg(pgid, signal.SIGTERM)`, waits up to `grace_s`, escalates to `os.killpg(pgid, signal.SIGKILL)`, closes stdio pipes cleanly (zero `ResourceWarning` under `-W error`!), and reaps `proc.wait()`.
- On Windows (`sys.platform == 'win32'`): `ctypes` wrapper creating a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (`0x2000`), assigning `proc._handle`, closing handle in `terminate_tree()`.
- `run_contained(...)`: uses `ContainedProcess`, drains stdout/stderr safely via `communicate(input=input_bytes, timeout=timeout)`, and calls `terminate_tree()` on `TimeoutExpired`.

- [ ] **Step 4: Implement `voice_eval/offline.py`**

Implement `strict_offline_env()` (save/restore `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`) and `collect_model_fingerprint(embedder)` (validates `model_id` non-empty, computes `file_digest(checkpoint)` when `checkpoint` is an existing file path).

- [ ] **Step 5: Implement `voice_eval/isolated_worker.py`**

- Worker loop (`python -m voice_eval.isolated_worker --embedder <name> [--checkpoint <path>]`): runs inside `strict_offline_env()`, supports `tone` (imports `ToneEmbedder`), `stats`, `vpnet` (passing optional `checkpoint`), `ecapa`. Handles `{"op": "describe"}`, `{"op": "embed", "path": "..."}`, `{"op": "shutdown"}`. Every reply includes `rss_mb: current_rss_mb()`.
- `IsolatedEmbedder`: starts worker lazily (and in `__init__` fetches `describe()` to populate `name`, `dim`, `checkpoint`, `model_id`, `cohort_size`); implements per-request deadline via reader thread/queue so `timeout_s` works identically on macOS/Linux/Windows; on timeout or EOF/invalid JSON calls `terminate_tree()`, increments `restart_count`/`timeout_count`/`consecutive_failures`, trips `WorkerCircuitOpenError` at `max_consecutive_failures`. Accepts optional `worker_cmd: list[str] | None = None` for clean deterministic crash/timeout testing.

- [ ] **Step 6: Run tests**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_contained_subprocess_and_worker -v`
Expected: ALL PASS with 0 warnings (zero unclosed pipe `ResourceWarning`s!).

- [ ] **Step 7: Commit Task 4**

```bash
git add voiceprint/contained_subprocess.py voice_eval/offline.py voice_eval/isolated_worker.py tests/test_contained_subprocess_and_worker.py
git commit -m "feat(eval): add contained subprocess tree, strict offline audit and isolated worker embedder"
```

---

### Task 5: Warm RTF Micro-Benchmark, Cross-Process RSS Aggregation, Offline Audit & Gate Wiring

**Files:**
- Modify: `voice_eval/cache.py:20-31,116-176`
- Modify: `voice_eval/dims/perf.py:82-125`
- Modify: `voice_eval/dims/__init__.py:14-51`
- Modify: `voice_eval/runner.py:278-528`
- Modify: `voice_eval/gate.py:102-276`
- Modify: `voice_eval/protocol.json`
- Modify: `voice_eval/__main__.py:20-85`
- Test: `tests/test_perf_rtf_and_offline.py`

**Interfaces:**
- Consumes: Task 4 (`strict_offline_env`, `collect_model_fingerprint`, `IsolatedEmbedder`, `WorkerCircuitOpenError`)
- Produces:
  - `CachedEmbedder.compute_audio_seconds() -> list[float]` (keeping `.stats()` exact 7 keys intact!)
  - `voice_eval.dims.perf.measure_uncached_rtf(embedder, audio_paths: list[str], *, max_samples: int = 5) -> dict`
  - `voice_eval.dims.perf.summarize_perf(..., compute_audio_s_list: list[float] | None = None, rtf_bench: dict | None = None, worker_rss_mb: float | None = None) -> dict` (adds `cold_embed_ms`, `warm_embed`, `rtf`, `parent_rss_mb`, `worker_rss_mb`)
  - `voice_eval.gate.PERF_RTF_RATIO_WARN = 1.5`, `PERF_RTF_ABS_WARN = 0.05`
  - `voice_eval.runner.run_split(..., offline_strict: bool = True, isolate_worker: bool = False)` emitting `runtime_audit` in `summary.json`

- [ ] **Step 1: Write failing tests in `tests/test_perf_rtf_and_offline.py`**

Tests to write:
- `test_cached_embedder_keeps_stats_keys_and_tracks_audio_seconds`:
  - verifies `set(cached.stats().keys()) == {'enabled', 'hits', 'misses', 'total', 'hit_rate', 'compute_ms_list', 'lookup_ms_list'}`;
  - verifies `cached.compute_audio_seconds()` returns exact durations for misses only, and resets on `reset_stats()`.
- `test_measure_uncached_rtf_and_summarize_perf_cold_warm_split`:
  - runs `measure_uncached_rtf(ToneEmbedder(), wav_paths, max_samples=5)` -> returns `source='micro_bench'`, `n_samples=5`, finite `cold_rtf > 0`, `warm_rtf > 0`, `p50_warm_rtf`, `p95_warm_rtf`;
  - verifies `summarize_perf` with `worker_rss_mb=256.0` sets `worker_rss_mb=256.0` and `rss_mb >= 256.0`.
- `test_gate_tier3_warns_on_warm_rtf_regression`:
  - compares two synthetic summaries where `cand.dimensions.perf.rtf.warm_rtf` is `0.30` vs `base` `0.10` (`3.0x`, `+0.20`) -> Tier 3 `warnings` contains `'Warm RTF'` alert; small noise (`0.01` vs `0.02`) does NOT trigger alert.
- `test_run_split_emits_runtime_audit_and_uncached_rtf_even_on_100pct_cache_hit`:
  - runs mini `run_split()` twice against the same `cache_dir`: on 2nd run (`cache.hit_rate == 1.0`), `summary['dimensions']['perf']['rtf']['source'] == 'micro_bench'` and `warm_rtf is not None`; `summary['runtime_audit']['offline_enforced'] is True` and `model_fingerprint['model_id']` is non-empty.
- `test_run_split_aborts_immediately_when_isolated_worker_circuit_opens`:
  - runs `run_split()` with a fake `IsolatedEmbedder` that trips `circuit_open=True` on failure -> raises `WorkerCircuitOpenError` without executing all remaining cases.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_perf_rtf_and_offline -v`
Expected: FAIL with `AttributeError` / `ImportError`.

- [ ] **Step 3: Update `voice_eval/cache.py`, `voice_eval/dims/perf.py`, `voice_eval/dims/__init__.py`**

- `cache.py`: track `self._compute_audio_s: list[float] = []` in `_timed_compute` (via `duration_seconds(path)` inside try/except), expose `compute_audio_seconds() -> list[float]`, clear in `reset_stats()`. Do NOT modify `stats()` keys!
- `dims/perf.py`: implement `measure_uncached_rtf(embedder, audio_paths, *, max_samples=5)` (deduplicates readable paths, calls `embedder._embed(p)` directly to bypass cache, records 1 cold + up to 4 warm timings & durations) and extend `summarize_perf` with `cold_embed_ms`, `warm_embed`, `rtf`, `parent_rss_mb`, `worker_rss_mb`.
- `dims/__init__.py`: forward `compute_audio_s_list`, `rtf_bench`, `worker_rss_mb` from `perf_data` to `summarize_perf`.

- [ ] **Step 4: Update `voice_eval/runner.py`, `voice_eval/gate.py`, `voice_eval/protocol.json`, `voice_eval/__main__.py`**

- `runner.py`:
  - Wrap `run_split` in `strict_offline_env()` when `offline_strict=True`.
  - If `isolate_worker=True` and not already `getattr(embedder, 'is_isolated', False)`, wrap in `IsolatedEmbedder(embedder.name, checkpoint=embedder.checkpoint)` with `try/finally: close()`.
  - After each `run_case()`, if `getattr(raw_embedder, 'circuit_open', False)`, raise `WorkerCircuitOpenError('isolated worker circuit breaker opened')`.
  - Collect up to 5 distinct readable audio paths from executed cases and run `measure_uncached_rtf(raw_embedder, sample_paths, max_samples=5)`.
  - Pass `compute_audio_s_list`, `rtf_bench`, `worker_rss_mb=getattr(raw_embedder, 'peak_worker_rss_mb', None)` into `perf_data`.
  - Write `runtime_audit` into `summary.json` and render `Warm RTF` + `离线审计` lines in `_format_summary_markdown`.
- `gate.py` & `protocol.json`:
  - Add `perf_rtf_ratio_warn: 1.5`, `perf_rtf_abs_warn: 0.05` to `protocol.json` `gate_tiers.tier3_observational`.
  - Add Warm RTF warning check in `_evaluate_tier3_warnings` and display Warm RTF in `_format_compare_markdown`.
- `__main__.py`:
  - Add `--isolate-worker`, `--worker-timeout` (float, default `30.0`), `--max-worker-failures` (int, default `3`) to `run` subcommand; when `--isolate-worker` is set, wrap in `strict_offline_env()`, instantiate `IsolatedEmbedder(args.embedder, timeout_s=args.worker_timeout, max_consecutive_failures=args.max_worker_failures)` directly (skipping `load_embedder`), and `close()` in `finally`.

- [ ] **Step 5: Run tests**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_perf_rtf_and_offline tests.test_voice_eval_cache tests.test_voice_eval_dims_unit tests.test_voice_eval_e2e -v`
Expected: ALL PASS with 0 warnings.

- [ ] **Step 6: Commit Task 5**

```bash
git add voice_eval/cache.py voice_eval/dims/perf.py voice_eval/dims/__init__.py voice_eval/runner.py voice_eval/gate.py voice_eval/protocol.json voice_eval/__main__.py tests/test_perf_rtf_and_offline.py
git commit -m "feat(eval): add uncached Warm RTF micro-benchmark, worker RSS, offline audit and CLI isolation"
```

---

### Task 6: Expanded Active Red-Team Probes & Full Whole-Repo Regression Verification

**Files:**
- Modify: `voice_eval/dims/redteam.py:149-352`
- Test: `tests/test_redteam_extended.py`

**Interfaces:**
- Consumes: Task 1 (`audio.py`, `audio_quality.py`), Task 2 (`spoof.py`, `audio_synth.py`), Task 3 (`agent/mcp_server.py`)
- Produces:
  - Extended `evaluate_redteam()` output with `container_integrity`, `mcp_security`, `anti_spoofing` check categories + `spoof_summary: {'watermarked_clone_far': float, 'clean_genuine_frr': float}` (all passing GREEN, 0 yellow, 0 red; preserving single `SKIPPED_NO_PRIVILEGE` slot under mocked `os.symlink` `PermissionError`)

- [ ] **Step 1: Write failing tests in `tests/test_redteam_extended.py`**

Tests to write:
- `test_evaluate_redteam_includes_container_mcp_and_spoof_probes_all_green`:
  - calls `evaluate_redteam()`, asserts `res['passed'] is True`, `res['yellow'] == 0`, `res['red'] == 0`;
  - asserts check IDs include `redteam_container_magic_mismatch`, `redteam_container_truncated_wav`, `redteam_quality_pure_silence`, `redteam_quality_severe_clipping`, `redteam_mcp_no_roots_blocks_path`, `redteam_mcp_path_escape_blocked`, `redteam_mcp_base64_data_uri_and_magic`, `redteam_mcp_revoke_requires_two_step_and_readonly`, `redteam_spoof_watermarked_clone_blocked`, `redteam_spoof_clean_genuine_preserved`;
  - asserts `res['spoof_summary'] == {'watermarked_clone_far': 0.0, 'clean_genuine_frr': 0.0}`.
- `test_symlink_probe_covers_inroot_and_keeps_single_skip_on_permission_error`:
  - verifies normal run checks both out-of-root and in-root symlink rejection;
  - mocks `Path.symlink_to` raising `PermissionError` -> asserts `res['skipped'] == 1` and `res['passed'] is True`.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent/.venv/bin/python -W error -m unittest tests.test_redteam_extended -v`
Expected: FAIL because new check IDs and `spoof_summary` do not exist yet.

- [ ] **Step 3: Extend `voice_eval/dims/redteam.py`**

- In `_probe_symlink_escape`: after verifying out-of-root symlink returns `'允许的目录'`, also create an in-root symlink inside `sym_dir` pointing to an allowed WAV and verify rejection with `'符号链接'` (same `try/except OSError` block, so `PermissionError` still counts as exactly 1 `SKIPPED_NO_PRIVILEGE`!).
- Add the 4 `container_integrity` probes, 4 `mcp_security` probes, and 2 `anti_spoofing` probes + `spoof_summary` dict in `evaluate_redteam()`.

- [ ] **Step 4: Run full repository test suite and `voice_eval audit` under `-W error`**

Run:
```bash
cd /Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent && \
.venv/bin/python -W error -m unittest discover -s tests -v && \
.venv/bin/python -W error -m voice_eval audit
```
Expected: ALL tests across all 12 test files (44 original + all new tests) PASS with 0 warnings, and `评测健康检查通过` (exit 0) on Windows or expected DPAPI note if DPAPI is mocked only in tests (check `audit.py` on macOS: note `security_checks()` calls `dpapi_protect`, which is Windows-only — verify `grader_checks`, `recompute_check`, `dim_checks` directly under `-W error` on macOS as well!).

- [ ] **Step 5: Commit Task 6**

```bash
git add voice_eval/dims/redteam.py tests/test_redteam_extended.py
git commit -m "feat(redteam): add container integrity, MCP security and anti-spoofing watermark probes"
```
