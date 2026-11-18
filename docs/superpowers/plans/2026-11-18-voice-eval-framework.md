# Voiceprint Agent 全维度离线评测体系实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `voiceprint-agent` 的现有离线 `voice_eval` 评测框架扩展为适配声纹 Agent 的七维评测体系（任务/工具/回答/状态流/性能/红队安全/多轴鲁棒性），引入磁盘嵌入缓存、v3 用例集与通用三级采用门禁，并提供不依赖 Windows DPAPI 与 LibriSpeech 的 Mac 单元测试套件。

**Architecture:** 在保留 `python -m voice_eval {audit,build,run,compare}` 唯一入口、严格隔离 `validation`/`locked` 与隐藏标签的前提下，新增 `voice_eval/dims/`（7 维评估器）、`voice_eval/audio_synth.py`（确定性失真合成）、`voice_eval/cache.py`（哈希嵌入缓存）与 `voice_eval/gate.py`（通用三级门禁及一页中文报告），同时对 `agent/service.py` 与 `voiceprint/audio.py` 做 spec 批准的 3 处边界最小修复。

**Tech Stack:** Python 3.12+ / 3.13, NumPy, SciPy (`signal.resample_poly`, `signal.fftconvolve`), SoundFile, Cryptography (AES-GCM), PyTorch, standard `unittest` + `tracemalloc` + `sqlite3`.

**Spec:** `docs/superpowers/specs/2026-11-18-voice-eval-framework-design.md`

## Global Constraints

- **不改变核心模型与加密原语**：绝不修改 `voiceprint/security.py`（DPAPI / AES-GCM / 正交投影）、`voiceprint/scoring.py` 核心公式、`voiceprint/vpnet.py` 及 `voiceprint/checkpoints/`；`requirements.txt` 不新增任何第三方依赖。
- **生产修复严格限定 3 项边界**：仅在 `agent/service.py`（命令头解析后非 `确认撤销` 时清除 `self.pending`；`detail=False` 下 `_verify` 不拼接含 `"归一化相似度"`/`"阈值"` 的 `reason`；未知命令分支补充英文导出探测触发 `refuse`；`_drift` 对不存在说话人返回 `ok=False, decision='unknown'`）与 `voiceprint/audio.py`（`check_audio` 与 `load_wav` 校验 `np.isfinite` 拦截 `NaN`/`Inf` 音频抛出 `AudioError`）做最小精确修改。
- **`uncertain` 永不计为自动判对**：沿用 `voice_eval/grader.py` 规则，`decision == 'uncertain'` 在所有准确性与自动化统计中不计为自动正确。
- **数据严格防泄漏与版次隔离**：`validation`（前 20 名）与 `locked`（后 20 名）严格隔离，绝不读 `train-clean-100`；`*.public.json` 仅含 `id, suite, split, steps`，真值与条件元数据保存在 `*.hidden.json`；`CASE_VERSION = 3`，严禁 `v2` 与 `v3` 报告跨版本对比。
- **负载分离原则**：`task`、`robustness`、`perf`、`security`（原 8 项）及真实试次 `detail=False` 防泄分抽检跑当前候选 `embedder` + 真实 `split` 音频；`tooling`、`response`（含必现 `uncertain`/`accept`/`reject`/`unknown`/`locked`）、`flow` 与 `redteam` 攻击树跑确定性 `ToneEmbedder` + 合成 WAV 沙箱。
- **缺失分数惩罚口径**：`dims/task.py` 对因注册失败而缺少 `score` 的验证试次统一赋下界惩罚分 `score = -10.0`、`raw_score = -1.0` 计算主 `eer` / `minDCF` / `score_stats`，同时记录 `valid_trials`、`imputed_missing_trials` 与 `raw_eer`。
- **三级采用门禁**：Tier 1 红线（`security`、`redteam`、`tooling`、`response`、`flow` 必须 100% 通过）；Tier 2 主指标（仅 `validation` 允许采用：`paired.low > 0`、`cand_fa - base_fa <= 0.02`、`cand_eer <= base_eer + 1e-6`，`locked` 永不采用）；Tier 3 观察指标（`robustness`、`perf`、`fault_recovery`、`consistency` 退化时告警但不一票否决）。
- **测试零警告**：所有单元测试使用仓库 `.venv/bin/python -W error -m unittest` 运行通过；Mac 上通过 `unittest.mock.patch` 替换 DPAPI，不破坏 Windows 真实 DPAPI 行为。

## Review Focus

- **NaN/Inf 与畸形音频投毒防护**：含 `NaN` 或 `Inf` 采样的 `.wav` 文件、0 字节文件、文本伪装 `.wav` 在 `enroll` 与 `verify` 中均须被 `check_audio` / `load_wav` 抛 `AudioError` 拦截，绝不能流入特征提取或往 SQLite 写入 `NULL` 分数或通过 `NaN < threshold == False` 绕过 `enroll_min`（Task 1、Task 5 覆盖单测）。
- **悬垂 `pending` 与非 `detail` 泄分校验**：`撤销 A → 名单/帮助/未知命令 → 确认撤销 A` 必须拒绝删除并保持 `A` 仍在库中；`VoiceAgent(..., detail=False)` 下 `verify`/`identify` 的 `result` 键与 `reply` 文本绝不可含浮点数分值或 `"归一化相似度"`/`"阈值"`/`"夹角"` 等敏感字样（Task 1、Task 4 覆盖单测与负向测试）。
- **磁盘缓存损坏与跨模型隔离**：`CachedEmbedder` 在读到截断/非 `.npy`/维度不匹配的缓存文件时须自动重算并原子覆写，且相同音频在相同 `name` 不同 `model_id` 下绝对不能串读向量（Task 3 覆盖单测）。
- **EER/minDCF 退化输入与缺失分数惩罚**：正例或负例为空时返回 `None` 而非伪造 `0.0`；相同分布得分返回约 `0.5`；弱模型拒绝难例注册时按 `-10.0` 惩罚分计入 EER，保证不会因跳过难例而反杀强模型（Task 4 覆盖单测）。
- **Gate 跨版本/Locked/单一退化边界**：`gate.compare_reports` 对不匹配的 `split` 或非 `v3` 版本须明确拒绝对比；`locked` split 即使全绿也强制 `adopt=False`；任一 Tier 1 红线失败或 EER 退化 > `1e-6` 须强制 `adopt=False`；仅 Tier 3 告警时若 Tier 1/2 全过则仍 `adopt=True` 且保留中文 `warnings`（Task 6 覆盖单测与端到端测试）。

---

### Task 1: 修复 Agent/Audio 边界缺陷并建立确定性沙箱与 Mac 测试基础设施

**Files:**
- Modify: `agent/service.py`
- Modify: `voiceprint/audio.py`
- Create: `voice_eval/dims/__init__.py`
- Create: `voice_eval/dims/_sandbox.py`
- Create: `tests/__init__.py`
- Create: `tests/_helpers.py`
- Test: `tests/test_voice_eval_guardrails.py`

**Interfaces:**
- Produces in `agent/service.py`: fixed `VoiceAgent._dispatch`, `VoiceAgent._verify`, `VoiceAgent._drift` with updated `EXPORT_WORDS`/English export probe refusal when unknown command, pending clearance before early returns, non-numeric non-threshold `detail=False` `_verify` reply, and `ok=False` for missing-speaker `漂移`.
- Produces in `voiceprint/audio.py`: `check_audio` & `load_wav` validating finite waveform samples (`np.isfinite`).
- Produces in `voice_eval/dims/_sandbox.py`:
  - `ToneEmbedder(Embedder)`: `name='tone'`, `dim=64`, `cohort_size=16`, `model_id='tone-v1...'`, extracts log-spectrum bucket features + fixed orthogonal projection; degrades smoothly under noise/reverb/short duration.
  - `generate_speaker_wav(path: Path, speaker_index: int, duration: float = 4.0, sr: int = 16000, phase: float = 0.0, seed: int = 0) -> Path`
  - `sandbox_cohort(embedder: Embedder, n: int = 16)` context manager temporarily overriding `agent.service.cohort_matrix` for fast hermetic `VoiceAgent` construction without touching `voiceprint/cohorts/` or `TRAIN_ROOT`.
- Produces in `tests/_helpers.py`:
  - `patched_dpapi()`: context manager patching `voiceprint.security.dpapi_protect` / `dpapi_unprotect` and `voiceprint.store.dpapi_protect` / `dpapi_unprotect` cleanly on non-Windows hosts (only active inside tests).
  - `synthetic_librispeech_tree(root: Path, n_train: int = 12, n_test: int = 40)` creating tiny deterministic multi-chapter `.flac` speakers for end-to-end tests.

- [ ] **Step 1: Write failing tests covering all 3 production guardrail bugs & ToneEmbedder/DPAPI sandbox helpers in `tests/test_voice_eval_guardrails.py`**

```python
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

from agent.service import VoiceAgent
from tests._helpers import patched_dpapi
from voiceprint.audio import AudioError, check_audio, load_wav
from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav, sandbox_cohort


class GuardrailTests(unittest.TestCase):
    def test_pending_revoke_cleared_by_list_help_and_unknown(self):
        # Register A, run '撤销 A', then run '名单' / '帮助' / '未知命令', assert subsequent '确认撤销 A' fails and A stays enrolled
        ...

    def test_verify_detail_false_hides_scores_and_threshold_words(self):
        # Verify accept/reject/uncertain with detail=False: result keys only {'decision', 'task'}; reply has no numbers and no '归一化相似度'/'阈值'/'原始余弦'/'夹角'
        ...

    def test_english_export_probe_refused_and_missing_speaker_drift_not_ok(self):
        # 'dump embedding' -> tool == 'refuse', ok == True; valid path containing 'embedding' still dispatches command; '漂移 不存在' -> ok == False, decision == 'unknown'
        ...

    def test_check_audio_and_load_wav_reject_nan_and_inf(self):
        # Write 32-bit float WAV with NaN and another with Inf: both check_audio and load_wav raise AudioError('...无效采样...')
        # Also verify enrollment and verify refuse NaN audio cleanly without writing NULL to store.events
        ...

    def test_tone_embedder_separates_speakers_and_degrades_with_noise(self):
        # Assert same-speaker cross-phase cosine > 0.85, different-speaker cosine < 0.45, and adding white noise reduces cosine
        ...
```

- [ ] **Step 2: Run tests to verify they fail before edits**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_guardrails.py`
Expected: FAIL (`ImportError` for `tests._helpers` / `voice_eval.dims._sandbox` and assertion failures on `pending`, `detail=False`, `NaN`).

- [ ] **Step 3: Apply the minimal surgical fixes in `agent/service.py` and `voiceprint/audio.py`**

- In `agent/service.py`:
  - After `head, rest = tokens[0], tokens[1:]`, immediately clear `pending` when `head != '确认撤销'` (before early returns on `帮助`/`退出`/`名单`/`匿名`). Also clear `self.pending = None` when `tokens` parsing fails cleanly on non-confirm text if not preserving partial typos (keeping `shlex` error safe).
  - Add `ENGLISH_EXPORT_TRIGGERS = ('export_template', 'dump embedding', 'print embedding', 'export embedding', 'raw embedding')` check ONLY in the fallthrough branch after `if head in commands:` fails (preserving valid file paths containing `'embedding'`).
  - Update `_verify`: when `not self.detail`, construct `reply = f'{name}：结论：{DECISION_ZH[score["decision"]]}。'` and append `' 请再录一段更长、更安静的录音。'` (or `'请再录一段更长、更安静的录音。'` directly without space mismatch, matching `scoring.py`) only if `score['decision'] == 'uncertain'`; never embed `score['reason']` when `detail=False`.
  - Update `_drift`: if `not self.store.exists(name)`, return `self._reply('drift', False, {'decision': 'unknown'}, f'没有 {name} 的模板。')`.
- In `voiceprint/audio.py`:
  - Helper `_assert_finite_audio(resolved: Path)` reading bounded chunks (up to `int(MAX_SECONDS * info.samplerate)` frames) via `sf.blocks(str(resolved), blocksize=65536, dtype='float32')` and raising `AudioError(f'录音包含无效采样（NaN/Inf）：{resolved}')` if `not np.all(np.isfinite(block))`. Call inside `check_audio` and verify `np.all(np.isfinite(wav))` in `load_wav`.

- [ ] **Step 4: Implement `voice_eval/dims/__init__.py` (placeholder package init), `voice_eval/dims/_sandbox.py`, and `tests/_helpers.py`**

- `ToneEmbedder`: computes log-energy spectrum across 64 narrowband/harmonic log-spaced filters via `np.fft.rfft` (16 kHz mono via `load_wav`), divides by DC/total energy & subtracts spectral mean, projects via deterministic RNG orthogonal matrix `np.random.default_rng(2026).standard_normal(...)` + L2 normalize (`dim=64`, `name='tone'`, `model_id` SHA-256 of `'tone-v1'`, `cohort_size=16`).
- Add `'tone': {'alpha': 0.35, 'identify_extra': 0.25, 'gap': 0.15, 'band': 0.12, 'enroll_min': 0.25}` OR leave `scoring.py` untouched and configure `ToneEmbedder.name = 'tone'`! Wait: read `Global Constraints`: **"绝不修改 `voiceprint/scoring.py` 核心公式"** — and `profile_for('tone')` falls back to `PROFILES['vpnet']` (`alpha=0.45, identify_extra=0.0, gap=0.30, band=0.20, enroll_min=0.15`) cleanly! Design `ToneEmbedder` amplitudes/harmonics so genuine multi-take waveforms have cosine `~0.92 > 0.15`, different speakers `< 0.10`, and partial interpolation produces `score` cleanly inside `|score - threshold| < 0.20` for deterministic `uncertain` cases without touching `scoring.py`!
- In `tests/_helpers.py`, implement `patched_dpapi()` patching both `voiceprint.security` and `voiceprint.store` references ONLY during `with patched_dpapi():`. Note: on Windows, `sys.platform == 'win32'` still works cleanly under `patched_dpapi()`, and in production DPAPI is untouched.

- [ ] **Step 5: Run tests to verify PASS and commit**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_guardrails.py`
Expected: PASS (5+ tests, exit 0, zero warnings).
Commit: `feat(eval): fix agent/audio boundary guards and add deterministic test sandbox`

---

### Task 2: 确定性音频失真合成器与 `v3` 用例集构建（`audio_synth.py`, `cases.py`, `protocol.json`）

**Files:**
- Create: `voice_eval/audio_synth.py`
- Modify: `voice_eval/cases.py`
- Modify: `voice_eval/protocol.json`
- Modify: `.gitignore`
- Test: `tests/test_voice_eval_cases_v3.py`

**Interfaces:**
- Consumes: `voiceprint.audio.list_flac`
- Produces in `voice_eval/audio_synth.py`:
  - `SR = 16000`
  - `add_white_noise(wav: np.ndarray, snr_db: float, seed: int) -> np.ndarray`
  - `add_pink_noise(wav: np.ndarray, snr_db: float, seed: int) -> np.ndarray`
  - `add_babble_noise(wav: np.ndarray, interferers: list[np.ndarray], snr_db: float, seed: int) -> np.ndarray`
  - `clip_duration(wav: np.ndarray, sr: int, seconds: float) -> np.ndarray`
  - `apply_telephone_channel(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]` (returns 8 kHz mono `float32` after G.711 $\mu$-law companding/expanding + sample rate `8000`)
  - `apply_reverb(wav: np.ndarray, sr: int, rt60: float, seed: int) -> np.ndarray`
  - `mix_two_speakers(wav_a: np.ndarray, sr_a: int, wav_b: np.ndarray, sr_b: int, target_sr: int = 8000) -> tuple[np.ndarray, int]`
- Produces in `voice_eval/cases.py`:
  - `CASE_VERSION = 3`
  - `NOISE_TYPES = ('white', 'pink', 'babble')`, `SNR_LEVELS = (20, 10, 5, 0)`, `DURATIONS = (1.0, 1.5, 3.0, 5.0)`, `RT60_LEVELS = (0.3, 0.6)`
  - `build(test_root: Path | None = None, case_dir: Path | None = None) -> dict` (optional path overrides simplify isolated testing without mutating repo `voice_eval/cases/`)
  - `build_split(names: list[str], speakers: dict[str, list[str]], split: str, case_dir: Path | None = None)` generating:
    - `verify`, `verify_impostor`, `enroll_quality`, `identify`
    - `robustness_noise` (12 conditions × 40 trials, hidden has `axis='noise'`, `variant=f'{noise_type}_snr{snr}'`)
    - `robustness_duration` (4 conditions × 40 trials, hidden has `axis='duration'`, `variant=f'dur_{sec:.1f}s'`)
    - `robustness_channel` (1 condition × 40 trials, hidden has `axis='channel'`, `variant='tel8k_ulaw'`)
    - `robustness_reverb` (2 conditions × 40 trials, hidden has `axis='reverb'`, `variant=f'rt60_{rt:.1f}s'`)
    - `robustness_mixture` (5 two-speaker mix trials per split with `分离再辨认`, hidden has `axis='mixture'`, `variant='mix2_0db_8k'`, `expect_speakers=[spk_a, spk_b]`, `gallery_speakers=gallery`)
  - `load(split: str, case_dir: Path | None = None) -> list[dict]`
  - `case_version(case_dir: Path | None = None) -> int`

- [ ] **Step 1: Write failing tests in `tests/test_voice_eval_cases_v3.py`**

```python
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

from tests._helpers import synthetic_librispeech_tree
from voice_eval.audio_synth import (
    add_babble_noise, add_pink_noise, add_white_noise,
    apply_reverb, apply_telephone_channel, clip_duration, mix_two_speakers,
)
from voice_eval.cases import CASE_VERSION, build, case_version, load


class SynthAndCasesV3Tests(unittest.TestCase):
    def test_noise_snr_and_determinism(self):
        # Verify white/pink/babble SNR matches target +/- 0.5 dB on 3s sine signal, output clipped to [-1, 1], deterministic per seed, no babble NaN if short interferers
        ...

    def test_duration_channel_reverb_and_mixture(self):
        # Verify clip_duration lengths (1.0/1.5/3.0/5.0s), telephone returns 8000 Hz quantized distortion, reverb RT60=0.6s differs from clean while preserving RMS, mixture returns 8000 Hz float32
        ...

    def test_build_v3_split_isolation_and_counts(self):
        # Build 40-speaker synthetic LibriSpeech tree into temp case_dir
        # Assert case_version == 3, validation & locked speakers strictly disjoint (20 each), public JSON never contains 'hidden'/'expect'/'true_speaker', and robustness suites cover noise(480), duration(160), channel(40), reverb(80), mixture(5) plus verify(40), verify_impostor(360), identify(20), enroll_quality(5)
        ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_cases_v3.py`
Expected: FAIL (`ModuleNotFoundError: No module named 'voice_eval.audio_synth'`).

- [ ] **Step 3: Implement `voice_eval/audio_synth.py`, update `voice_eval/cases.py`, `voice_eval/protocol.json`, and `.gitignore`**

- Implement `add_white_noise`, `add_pink_noise` (real FFT $1/\sqrt{\max(k, 1)}$ spectral shaping with $k=0$ zeroed, `np.fft.irfft`, RMS normalized), `add_babble_noise` (tiles/crops interferers to match signal length cleanly, normalizes interferer sum RMS to target SNR), `clip_duration`, `apply_telephone_channel` ($\mu=255$ $\mu$-law companding/quantizing onto $[-127, 127]$ steps + expansion at 8 kHz), `apply_reverb` (exponential envelope with seedable white carrier, `fftconvolve(..., mode='full')[:len(wav)]`), `mix_two_speakers` (resamples both to 8 kHz, normalizes each to unit/common RMS, sums with 0 dB equal power, scales to peak <= 0.9).
- Update `voice_eval/cases.py` to `CASE_VERSION = 3`, supporting clean optional `test_root` and `case_dir` parameters while keeping zero-arg `build()`, `load(split)`, `case_version()` backward-compatible defaults (`TEST_ROOT`, `CASE_DIR`). Add robustness grid generation deterministically, excluding trial target & impostor speakers when sampling babble interferers from `names`.
- Update `voice_eval/protocol.json` with `case_version: 3`, updated `suites`, `dimensions`, `mindcf_params`, `gate_tiers`.
- Append `/voice_eval/cache/` to `.gitignore`.

- [ ] **Step 4: Run tests to verify PASS and commit**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_cases_v3.py`
Expected: PASS, zero warnings.
Commit: `feat(eval): add deterministic distortion synth and v3 multi-axis case builder`

---

### Task 3: 磁盘按内容哈希隔离的嵌入缓存（`voice_eval/cache.py`）

**Files:**
- Create: `voice_eval/cache.py`
- Test: `tests/test_voice_eval_cache.py`

**Interfaces:**
- Produces in `voice_eval/cache.py`:
  - `DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / 'cache'`
  - `CachedEmbedder`: wraps any `Embedder` instance (`inner`), forwarding `.name`, `.dim`, `.checkpoint`, `.model_id`, `.cohort_size`, `.describe()`. Implements `.embed_file(path: str) -> np.ndarray`, `.stats() -> dict` (`hits`, `misses`, `total`, `hit_rate`, `compute_ms_list`, `lookup_ms_list`), `.reset_stats() -> None`.

- [ ] **Step 1: Write failing tests in `tests/test_voice_eval_cache.py`**

```python
import tempfile
import unittest
from pathlib import Path
import numpy as np

from voice_eval.cache import CachedEmbedder
from voice_eval.dims._sandbox import ToneEmbedder, generate_speaker_wav


class CacheTests(unittest.TestCase):
    def test_disk_hit_persistence_and_model_id_isolation(self):
        # First wrapper computes & populates disk cache (misses=1, hits=0); second fresh wrapper hits disk (hits=1, misses=0, inner call_count stays 0); changing inner.model_id forces miss and isolates cached .npy path
        ...

    def test_corrupted_or_wrong_dim_cache_file_recomputes_atomically(self):
        # Write garbage bytes into expected cache .npy path and test another wrong-dim .npy; assert CachedEmbedder gracefully recomputes valid vector of shape (dim,) and repairs the cache file without leaving .tmp files
        ...

    def test_disabled_cache_bypasses_read_and_write(self):
        # enabled=False records misses on every call and writes nothing to cache_dir
        ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_cache.py`
Expected: FAIL (`ModuleNotFoundError: No module named 'voice_eval.cache'`).

- [ ] **Step 3: Implement `voice_eval/cache.py`**

- Compute SHA-256 via `file_digest(path)`, resolve cache file `{cache_dir}/{self.name}/{self.model_id[:16]}/{digest}.npy`.
- Guard loads with `np.load(cache_path, allow_pickle=False)` verifying `isinstance(arr, np.ndarray) and arr.shape == (self.dim,) and arr.dtype == np.float32 and np.all(np.isfinite(arr))`. On corruption or mismatch, unlink/recompute from `self.inner.embed_file(path)`.
- Write atomically via unique temp filename (`f'.{digest}.{os.getpid()}.tmp.npy'`) + `os.replace`. Note: `np.save` appends `.npy` automatically if filename doesn't end with `.npy`, so ALWAYS pass a filename ending with `.tmp.npy`!

- [ ] **Step 4: Run tests to verify PASS and commit**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_cache.py`
Expected: PASS, zero warnings.
Commit: `feat(eval): add atomic sha256+model_id disk embedding cache`

---

### Task 4: 维度一 `task`（EER/minDCF/分数分布）与维度二～四 `tooling` / `response` / `flow`

**Files:**
- Create: `voice_eval/dims/task.py`
- Create: `voice_eval/dims/tooling.py`
- Create: `voice_eval/dims/response.py`
- Create: `voice_eval/dims/flow.py`
- Test: `tests/test_voice_eval_dims_unit.py` (task section)
- Test: `tests/test_voice_eval_agent_checks.py` (tooling, response, flow positive & negative defect-catching tests)

**Interfaces:**
- Produces in `voice_eval/dims/task.py`:
  - `MISSING_NORM_SCORE = -10.0`, `MISSING_RAW_SCORE = -1.0`
  - `MINDCF_PARAMS = {'p_target': 0.01, 'c_miss': 1.0, 'c_fa': 1.0}`
  - `compute_eer(scores: list[float], labels: list[int]) -> tuple[float | None, float | None]`
  - `compute_mindcf(scores: list[float], labels: list[int], p_target: float = 0.01, c_miss: float = 1.0, c_fa: float = 1.0) -> dict` (`min_dcf_raw`, `min_dcf_norm`, `threshold`, `params`)
  - `score_distribution(scores: list[float], labels: list[int]) -> dict` (`genuine`, `impostor`, `d_prime`)
  - `summarize_verify_group(rows: list[dict]) -> dict` (extracts `score` / `raw_score`, imputes `-10.0`/`-1.0` when `score is None`, outputs `n`, `valid_trials`, `imputed_missing_trials`, `eer`, `eer_threshold`, `raw_eer`, `min_dcf_norm`, `min_dcf_raw`, `mindcf_params`, `score_stats`, `raw_score_stats`)
  - `evaluate_task(rows: list[dict]) -> dict` (outputs `by_suite`, `verify_metrics`, `verify_all_metrics`, `eer_primary`)
- Produces in `voice_eval/dims/tooling.py`:
  - `evaluate_tooling() -> dict` (`{'passed': bool, 'total': int, 'passed_count': int, 'accuracy': float, 'checks': list[dict]}`)
- Produces in `voice_eval/dims/response.py`:
  - `check_reply_payload(outcome: dict, expect_decision: str, detail: bool, speaker_name: str | None = None) -> tuple[bool, str]`
  - `evaluate_response(real_non_detail_checks: list[dict] | None = None) -> dict` (`real_non_detail_checks` 每项契约为 `{'label': str, 'outcome': dict, 'expect_decision': str, 'speaker_name': str | None}`；返回 `{'passed': bool, 'total': int, 'passed_count': int, 'leak_count': int, 'checks': list[dict]}`)
- Produces in `voice_eval/dims/flow.py`:
  - `evaluate_flow(separator=None) -> dict` (`{'passed': bool, 'total': int, 'passed_count': int, 'checks': list[dict]}`)

- [ ] **Step 1: Write failing tests for `task` math + `tooling` / `response` / `flow` (including injected agent bug tests verifying the checkers actually catch broken behavior)**

- Create `tests/test_voice_eval_dims_unit.py` covering `compute_eer` (separable=0.0, identical/inverted=0.5/1.0, empty class -> `None`), `compute_mindcf` bounds (`<= 1.0` when normalized for random guessing? Actually random/trivial accept-all gives normalized minDCF = `min(p_target, 1-p_target)/min(...) = 1.0`), `score_distribution` `$d'$`, and `summarize_verify_group` missing-score imputation (`valid_trials`, `imputed_missing_trials`, `raw_eer` vs `eer`).
- Create `tests/test_voice_eval_agent_checks.py` running `evaluate_tooling()`, `evaluate_response()`, `evaluate_flow(separator=...)` under `patched_dpapi()`, plus unit tests calling `check_reply_payload` with deliberate score/threshold leak (`'阈值 0.42'`) and large vector leak to prove the checker flags them!
- Note on `evaluate_flow`: when `separator` kwarg is `None`, pass a tiny 2-second 8 kHz synthetic mixture to `VoiceAgent.handle('分离再辨认 ...')` ONLY if `voiceprint/separate.py` checkpoint exists — wait: check runtime! Let's check how fast real `ConvTasNet` is on a 2s 8 kHz clip on CPU or keep `evaluate_flow` verifying the real `ConvTasNet` path on a tiny 1.0s clip (should take < 0.3s on CPU!) while also testing the injectable fast `separator` fallback in unit tests if needed! We can allow optional `separator=None` parameter on `evaluate_flow` so unit tests can mock it if desired while default exercises real `separate_file` on a 1.0s clip when `CKPT.exists()`.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_dims_unit.py tests/test_voice_eval_agent_checks.py`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `voice_eval/dims/task.py`, `tooling.py`, `response.py`, `flow.py`**

- Keep code clean and DRY:
  - In `task.py`: build ROC sweep sorting unique/step thresholds descending cleanly so tied scores jump atomically without mid-tie inflation; interpolate crossing linearly between adjacent `(FAR, FRR)` pairs. Impute missing scores deterministically when `result` in `trace[-1]` lacks `'score'`.
  - In `tooling.py`, `response.py`, `flow.py`: always run inside `TemporaryDirectory()` with `ToneEmbedder` + synthetic audio under `sandbox_cohort()`, wrapped carefully so `agent.close()` always executes cleanly in `finally` blocks (`-W error` catches unclosed sqlite connections!).
  - In `response.py`: `uncertain` case in `ToneEmbedder` generated smoothly by blending speaker A & B waveforms (e.g. weighted waveform or interpolated harmonic mix `alpha * spk_a + (1 - alpha) * spk_b`) calibrated dynamically on enrollment so `|score - threshold| < band` is guaranteed hit in both `verify` and `identify`! Verify both `detail=False` and `detail=True`. Also validate any `real_non_detail_checks` entries supplied by `runner.py` via `check_reply_payload`.

- [ ] **Step 4: Run tests to verify PASS and commit**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_dims_unit.py tests/test_voice_eval_agent_checks.py`
Expected: PASS, zero warnings (check no ResourceWarning from unclosed SQLite!).
Commit: `feat(eval): implement task(EER/minDCF), tooling, response and flow dims`

---

### Task 5: 维度五～七 `perf` / `redteam` / `robustness` 与升级 `audit.py`

**Files:**
- Create: `voice_eval/dims/perf.py`
- Create: `voice_eval/dims/redteam.py`
- Create: `voice_eval/dims/robustness.py`
- Modify: `voice_eval/audit.py`
- Modify: `tests/test_voice_eval_dims_unit.py` (append audit/robustness/perf unit tests)
- Modify: `tests/test_voice_eval_agent_checks.py` (append redteam + fault recovery + consistency + negative tests)

**Interfaces:**
- Produces in `voice_eval/dims/perf.py`:
  - `percentile_stats(values_ms: list[float]) -> dict` (`n`, `mean_ms`, `p50_ms`, `p95_ms`, `total_ms`)
  - `summarize_perf(case_timings_ms: list[float], step_timings_ms: dict[str, list[float]], cache_stats: dict | None, peak_tracemalloc_mb: float, rss_mb: float | None) -> dict`
  - `current_rss_mb() -> float | None` (cross-platform: Linux/macOS `resource.getrusage`, Windows fallback `None` or `ctypes` without failing)
- Produces in `voice_eval/dims/redteam.py`:
  - `evaluate_redteam(security_rows: list[dict] | None = None) -> dict`
  - Classifies each probe as `'GREEN'`, `'YELLOW'`, `'RED'`, or `'SKIPPED_NO_PRIVILEGE'` (ONLY for Windows non-privileged symlink `OSError`).
  - Returns `{'passed': bool, 'green': int, 'yellow': int, 'red': int, 'skipped': int, 'total_executed': int, 'checks': list[dict]}` where `passed == (yellow == 0 and red == 0)`.
- Produces in `voice_eval/dims/robustness.py`:
  - `evaluate_fault_recovery() -> dict` (`{'passed': bool, 'fault_total': int, 'recovered': int, 'recovery_rate': float, 'checks': list[dict]}`)
  - `evaluate_consistency(repeats: int = 3) -> dict` (`{'passed': bool, 'trials': int, 'repeats': int, 'agreement_rate': float, 'max_score_spread': float, 'checks': list[dict]}`)
  - `evaluate_robustness(rows: list[dict], baseline_verify_accuracy: float | None) -> dict` (`{'by_axis': dict, 'by_variant': dict, 'worst_drop': float | None, 'fault_recovery': dict, 'consistency': dict}`)
- Modifies `voice_eval/audit.py`:
  - Adds `dim_checks() -> list[str]` testing `compute_eer` / `compute_mindcf` identities, `ToneEmbedder` + `CachedEmbedder` temp round-trip & corruption recovery, and `protocol.json` v3 key validation (`case_version == 3`, `dimensions`, `gate_tiers`).
  - Includes `dim_checks()` inside `run_all()`.

- [ ] **Step 1: Write failing tests for `perf`, `redteam`, `robustness`, and upgraded `audit.run_all()`**

- Append tests to `tests/test_voice_eval_dims_unit.py` checking `summarize_perf`, `evaluate_robustness` grouping by `noise`/`duration`/`channel`/`reverb`/`mixture`, `worst_drop`, and `audit.run_all()` (inside `patched_dpapi()` on Mac) returning `ok=True` plus catching corrupt EER/protocol in negative subtest.
- Append tests to `tests/test_voice_eval_agent_checks.py` checking `evaluate_redteam()` (export variants, `../` & outside-root & UNC & symlink-or-skip, control-char/long names, empty/non-audio/silence/NaN/tiny wav poisoning, missing consent, no-overwrite), `evaluate_fault_recovery()`, and `evaluate_consistency(repeats=3)`. Also assert that if a mock symlink creation raises `PermissionError`/`OSError`, it classifies cleanly as `SKIPPED_NO_PRIVILEGE` without marking `RED`/`YELLOW`, while still enforcing real `outside-root` file refusal!

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_dims_unit.py tests/test_voice_eval_agent_checks.py`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `voice_eval/dims/perf.py`, `redteam.py`, `robustness.py` and update `voice_eval/audit.py`**

- Check platform-specific RSS detail in `perf.py`: on macOS `ru_maxrss` is bytes, on Linux `ru_maxrss` is KiB! Handle `sys.platform == 'darwin'` (`/ (1024 * 1024)`) vs Linux (`/ 1024.0`), and guard `import resource` with `try/except ImportError` for Windows!
- Cleanly structure `redteam.py` so every attack case runs in isolated temp storage with `audio_roots=[sandbox_dir]` (`ToneEmbedder` under `sandbox_cohort()`), closing every `VoiceAgent` in `try/finally`. Also integrate optional pre-computed `security_rows` from candidate `embedder` into the unified security/redteam summary count cleanly!

- [ ] **Step 4: Run tests to verify PASS and commit**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_dims_unit.py tests/test_voice_eval_agent_checks.py`
Expected: PASS, zero warnings.
Commit: `feat(eval): add perf, redteam, robustness dims and upgrade self-audit`

---

### Task 6: Runner 流水线集成、通用三级门禁（`gate.py`）、CLI 与端到端全链路测试

**Files:**
- Create: `voice_eval/gate.py`
- Modify: `voice_eval/dims/__init__.py`
- Modify: `voice_eval/runner.py`
- Modify: `voice_eval/__main__.py`
- Modify: `voice_eval/tune.py`
- Test: `tests/test_voice_eval_e2e.py`

**Interfaces:**
- Produces in `voice_eval/dims/__init__.py`:
  - `evaluate_all_dimensions(rows, *, security_rows, real_non_detail_checks, perf_data, skip_flow_separate=False) -> dict`
- Modifies `voice_eval/runner.py`:
  - `run_case(case: dict, embedder, detail: bool = True) -> dict` (attaches `latency_ms`, `step_latencies`, `axis`, `variant`, `expect_speakers` cleanly)
  - `run_split(embedder, split: str, with_security: bool, *, case_dir: Path | None = None, reports_dir: Path | None = None, cache_dir: Path | None = None, use_cache: bool = True, skip_mixture: bool = False) -> Path`
    - Wraps `embedder` in `CachedEmbedder` (when `use_cache=True`), tracks `tracemalloc` delta and step/case timings, runs 5 real `detail=False` verification/identify spot checks against the running `embedder` on sample trials from `split` (when `verify-accept-*` speakers exist), computes all 7 dimensions, writes `cases.jsonl`, `summary.json`, and `summary.md`.
    - For `robustness_mixture`: when `skip_mixture=True`, omits executing heavy Conv-TasNet mixture cases cleanly (marking mixture variant skipped cleanly); default `False` executes them when present.
- Produces in `voice_eval/gate.py`:
  - `resolve_report_dir(ref: str | Path, split: str | None = None, reports_root: Path = REPORTS) -> Path`
  - `compare_reports(base_ref: str | Path, cand_ref: str | Path, *, split: str | None = None, out_dir: Path | None = None, write_legacy_name: bool = False) -> dict`
  - Writes `compare_{base_tag}_vs_{cand_tag}_{split}.json` + `.md` (and `compare_{split}.json` + `.md` when `write_legacy_name=True`), returning the full report dict (`adopt`, `adoption_allowed`, `redline_passed`, `primary_passed`, `eer_delta`, `mindcf_delta`, `paired`, `warnings`, `gate`, `reason`, `markdown_path`, `json_path`).
- Modifies `voice_eval/__main__.py`:
  - `run`: adds `--no-cache` and `--skip-mixture`.
  - `compare`: adds `--base` (optional, default `stats`), `--cand` (optional, default `vpnet`), `--out` (optional), keeps `--split validation|locked`. When `--base` & `--cand` are omitted (default stats vs vpnet), sets `write_legacy_name=True`.
- Modifies `voice_eval/tune.py`:
  - Keeps validation-only search; reads optional `case_dir` safely and works cleanly on v3 cases (`verify` + `verify_impostor` suites).

- [ ] **Step 1: Write failing end-to-end tests in `tests/test_voice_eval_e2e.py` covering `runner`, `gate`, `CLI`, and `tune`**

```python
import tempfile
import unittest
from pathlib import Path

from tests._helpers import patched_dpapi, synthetic_librispeech_tree
from voice_eval.cases import build, load
from voice_eval.dims._sandbox import ToneEmbedder
from voice_eval.gate import compare_reports
from voice_eval.runner import run_split


class EndToEndEvalTests(unittest.TestCase):
    def test_full_v3_pipeline_and_three_tier_gate_outcomes(self):
        # 1. Build 40-speaker synthetic LibriSpeech tree + v3 cases in tempdir
        # 2. Run WeakEmbedder (low accuracy) vs ToneEmbedder (high accuracy) on validation & locked splits with patched_dpapi() and skip_mixture=True
        # 3. Verify cases.jsonl recomputation, all 7 dims present in summary.json, summary.md readable, cache reused across trials
        # 4. Assert compare_reports(weak_val, strong_val) -> adopt=True, redline_passed=True, primary_passed=True, markdown file written
        # 5. Assert compare_reports(weak_locked, strong_locked) -> adoption_allowed=False, adopt=False despite high accuracy
        # 6. Inject redline failure (e.g. redteam.red=1), assert adopt=False; inject EER regression (> +1e-6), assert adopt=False; inject Tier 3 perf regression only -> adopt=True with non-empty warnings
        # 7. Assert cross-version (case_version=2 vs 3) and mismatched-split raises ValueError cleanly
        ...

    def test_real_detail_false_candidate_no_leak_and_cli_compare_defaults(self):
        # Verify runner's real candidate detail=False spot check passes on valid ToneEmbedder and inspect summary.json structure
        ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_voice_eval_e2e.py`
Expected: FAIL (`ModuleNotFoundError: No module named 'voice_eval.gate'`).

- [ ] **Step 3: Implement `voice_eval/gate.py`, update `voice_eval/dims/__init__.py`, `voice_eval/runner.py`, `voice_eval/__main__.py`, and `voice_eval/tune.py`**

- Ensure `runner.py` handles `robustness_mixture` target matching cleanly (`decision == 'accept'` / detected `names` containing at least one expected speaker, or exact gallery hit accounting), and executes real candidate `detail=False` spot checks using 1 genuine + 1 impostor verify + 1 identify trial if enrolled speakers exist.
- Format clean, scannable one-page Chinese Markdown in `summary.md` & `compare_*_{split}.md` (table for 7 dims, Tier 1/2/3 gate breakdown, robustness by axis, perf P50/P95 & memory, warnings, verdict).
- Check `tracemalloc` start/stop hygiene: only call `tracemalloc.start()` if `not tracemalloc.is_tracing()` (if already tracing, measure snapshot or peak delta cleanly without breaking outer callers!), stopping only if we started it!

- [ ] **Step 4: Run ALL tests and CLI audit under `-W error`, verify zero regressions, and commit**

Run:
```bash
.venv/bin/python -W error -m unittest discover -s tests
```
Expected: All tests in `tests/test_voice_eval_guardrails.py`, `tests/test_voice_eval_cases_v3.py`, `tests/test_voice_eval_cache.py`, `tests/test_voice_eval_dims_unit.py`, `tests/test_voice_eval_agent_checks.py`, `tests/test_voice_eval_e2e.py` PASS with zero warnings!
Commit: `feat(eval): integrate 7-dimension runner, universal 3-tier gate and e2e tests`
