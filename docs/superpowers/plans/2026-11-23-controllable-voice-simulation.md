# 可控声音模拟、强制水印溯源与攻防闭环评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `voiceprint-agent` 构建“授权与身份归属校验 → 声音模拟合成（零依赖声学迁移 / Windows GPU F5-TTS 插件）→ 内核强制 16-bit 溯源水印 → 100 ms 质检与目标声纹 Self-Refine 自检重试 → MCP 暴露 → 去水印红队与 `sim-bench` 闭环评测”的完整双向安全 Agent 闭环。

**Architecture:** 在 `voiceprint/synth.py` 定义统一合成抽象 `VoiceSynthesizer` 与双后端（零重依赖 `AcousticFormantSynthesizer` + 进程树隔离 `SubprocessNeuralSynthesizer`），将水印嵌入与验签严格锁死在 `agent/service.py` 受信任内核的 Self-Refine 状态机中，通过动态能力暴露接入 `agent/mcp_server.py`，并在 `voice_eval/` 下新增去水印攻击算子、6 项仿真红队探针与双轨闭环评测命令 `sim-bench`。

**Tech Stack:** Python 3.12、`numpy`、`scipy`、`soundfile`、`cryptography`、`torch`（现有主仓零新增重依赖）；可选独立 Windows GPU 插件环境接入 `F5-TTS`（预留 `CosyVoice2` 插槽）。

**Spec:** `docs/superpowers/specs/2026-11-23-controllable-voice-simulation-design.md`

---

## Global Constraints

- `requirements.txt` 保持零新增重依赖，主仓包（`voiceprint/`、`agent/`、`voice_eval/`）禁止顶层或无条件 `import f5_tts`、`cosyvoice`、`whisper` 等外部生成库。
- 现有全部 73 项回归测试（含 `tests/test_mcp_server.py:55-66` 8 工具断言与 `anonymize_boundary`）保持零修改并在 `-W error` 下 100% 通过。
- 保留 `voice_eval/protocol.json` 既有 7 维发布门禁结构与 `verify` / `identify` 默认 `spoof_detector` 行为不变；保留 `匿名` / `变声` 拒答存根不变。
- 16-bit 扩频水印嵌入（`embed_spread_watermark`）与回读验签（`SpreadSpectrumWatermarkDetector`）仅能在主仓受信任内核执行，严禁下放到合成后端；未收敛（质量、水印或目标声纹任一不达标）样本绝不释放至 `simulated_dir`。
- `VoiceAgent` 在 `detail=False` 模式下，任何 `模拟` / `voiceprint_simulate` 返回载荷与文本均不得泄露 `score`、`raw_score`、`threshold`、`theta_deg`、`low`、`high`、`gap` 等敏感数值或术语。
- Windows GPU 外部子进程必须通过 `voiceprint/contained_subprocess.py`（Job Object）运行，强制注入 `PYTHONUTF8=1` 与 `PYTHONIOENCODING=utf-8`，并严格保留带空格路径（如 `"D:/voice confirm/.venv_tts/Scripts/python.exe"`）的引号切分与清理。

## Review Focus

- **含空格 Windows 命令切分与反斜杠保真**：当 `--neural-cmd` 包含 `"D:/voice confirm/.venv_tts/Scripts/python.exe" ...` 或带反斜杠的引号路径时，`split_command_args` 必须准确保留完整可执行路径而不是按空格截断 → 覆盖测试落于 **Task 1 (`test_split_command_args_preserves_windows_quoted_spaces`)**。
- **Windows UTF-8 中文传递与超时/异常临时目录回收**：含中文 `text`/`reference_text` 在子进程执行、或子进程超时/崩溃返回非零退出码时，环境变量必须包含 `PYTHONUTF8=1` 且宿主机临时 `request.json` 目录必须无残留 → 覆盖测试落于 **Task 1 (`test_subprocess_synthesizer_utf8_env_and_temp_cleanup_on_timeout`)**。
- **模拟产出文件名防路径穿越**：即使已注册说话人姓名包含 `../`、`\` 或特殊符号，落盘进 `data_dir/simulated/` 的文件名必须经过安全清洗且严格解析落在 `simulated_dir` 内 → 覆盖测试落于 **Task 2 (`test_simulate_sanitizes_speaker_name_in_output_filename`)**。
- **归属权校验拒答与多轮自评绝不污染验证失败锁定期**：跨人冒充参考音拒答以及 Self-Refine 连续多轮内部打分不通过时，绝不向 `store.events` 写入 `'verify'` 失败记录，也不触发 5 次锁定 → 覆盖测试落于 **Task 2 (`test_simulate_mismatch_and_failed_retries_do_not_pollute_lockout_events`)**。
- **MCP 禁用态强调用与布尔伪装整型防护**：当 `synthesizer=None`（如 `build_default_server` 默认）时，`tools/list` 不列出 `voiceprint_simulate` 且直接 `tools/call` 必须干净返回 `synthesizer_disabled`；传入 `max_attempts=True`（Python 中 `isinstance(True, int)` 为真）必须被严格拒绝 → 覆盖测试落于 **Task 3 (`test_mcp_simulate_disabled_when_no_synthesizer_and_rejects_bool_max_attempts`)**。

---

## File Structure

- Create: `voiceprint/synth.py` — 合成请求/结果数据契约、`VoiceSynthesizer` 基类与输出校验、零依赖 `AcousticFormantSynthesizer`、隔离进程 `SubprocessNeuralSynthesizer`、跨平台命令切分与工厂 `load_synthesizer`。
- Create: `tests/test_synth.py` — Task 1 单元测试（涵盖声学双模式迁移方向、输出校验、含空格路径切分、UTF-8 与超时进程树及临时目录回收）。
- Modify: `agent/service.py` — 扩展 `VoiceAgent.__init__`（`synthesizer=_UNSET`、`max_sim_attempts`、`simulated_dir`、`audio_roots` 联动、内部 `_watermark_guard`）、命令分发 `'模拟'`、5 道前置门禁、Self-Refine 核心 `_run_simulate_core`、`HELP` 更新。
- Modify: `agent/cli.py:15-39` — 增加 `--synthesizer`、`--neural-cmd`、`--neural-timeout`、`--max-sim-attempts`、`--spoof-guard` 参数透传。
- Create: `tests/test_agent_simulate.py` — Task 2 单元测试（5 道门禁、反递归水印、Self-Refine 自适应调参收敛、未达标拒放、安全文件名、`detail=False` 防泄分、不污染锁定期、CLI 参数）。
- Modify: `agent/mcp_server.py` — 新增 `voiceprint_simulate` schema 并纳入 `_WRITE_TOOLS`、`tools/list` 按能力动态过滤、严格参数校验、`include_audio_base64`、`_ephemeral_roots` 纳入 `simulated_dir`、`_health_dict` 与 `main()` 启动参数。
- Create: `tests/test_mcp_simulate.py` — Task 3 单元测试（动态 8/9 工具切换、`--read-only`、非布尔/禁用拒答、`--audio-roots` 路径闭环、无 `audio_roots` Base64 闭环、并发安全）。
- Modify: `voice_eval/audio_synth.py` — 新增去水印/降质攻击算子 `apply_lowpass_attack`、`apply_resample_attack`、`apply_gain_clip_attack`。
- Modify: `voice_eval/dims/redteam.py` — 新增 `_run_simulation_probes()` 共 6 项仿真安全红队探针并接入 `evaluate_redteam()`。
- Create: `voice_eval/dims/simulation.py` — 双轨闭环评测 `evaluate_simulation()` 与 JSON/MD 报告生成 `write_simulation_report()`。
- Modify: `voice_eval/__main__.py` — 新增 `sim-bench` 子命令入口。
- Create: `tests/test_simulation_eval_and_redteam.py` — Task 4 与 Task 5 单元测试（攻击算子、扩展红队全部 `GREEN`、`sim-bench` 双轨指标与报告输出）。
- Create: `plugins/neural_tts_worker.py` — 独立 Windows GPU 隔离脚本（主打 `--engine f5_tts`，预留 `--engine cosyvoice2` 插槽，主仓零导入依赖）。
- Modify: `README.md` — 更新双项目展示说明：Agent 闭环架构、MCP、七维评测、`模拟` 指令、`sim-bench` 及 Windows RTX 4070 SUPER 独立 `.venv_tts` F5-TTS 配置指南。
- Unchanged (Verified Intact): `requirements.txt`、`voice_eval/protocol.json`、`tests/test_mcp_server.py`、`tests/test_redteam_extended.py`、`tests/test_spoof_guard.py` 及全部既有 73 项测试文件。

---

### Task 1: 可插拔合成器接口、零依赖声学迁移后端与 Windows 隔离神经插件契约 (`voiceprint/synth.py`)

**Files:**
- Create: `voiceprint/synth.py`
- Create: `tests/test_synth.py`

**Interfaces:**
- Consumes: `voiceprint/features.py` (`SR = 16000`), `voiceprint/audio.py` (`MIN_SECONDS = 0.5`, `sniff_audio_format`), `voiceprint/contained_subprocess.py` (`run_contained`, `ContainedRunResult`).
- Produces:
  - `SynthesisError(RuntimeError)`
  - `MAX_SIM_TEXT_CHARS = 500`, `MAX_SIM_SECONDS = 60.0`
  - `SimulationRequest`, `SimulationResult`, `VoiceSynthesizer` (`supported_modes: tuple[str, ...]`)
  - `AcousticFormantSynthesizer(VoiceSynthesizer)` (`name = 'acoustic_formant'`, `supported_modes = ('audio', 'text')`)
  - `SubprocessNeuralSynthesizer(VoiceSynthesizer)` (`name = 'neural_subprocess'`, `supported_modes: tuple[str, ...] = ('text',)`)
  - `split_command_args(cmd: str | Sequence[str]) -> list[str]`
  - `validate_simulation_result(res: SimulationResult, *, min_seconds: float = MIN_SECONDS, max_seconds: float = MAX_SIM_SECONDS) -> np.ndarray`
  - `load_synthesizer(kind: str, *, neural_cmd: str | Sequence[str] | None = None, timeout_s: float = 120.0, supported_modes: tuple[str, ...] = ('text',)) -> VoiceSynthesizer | None`

- [ ] **Step 1: Write failing unit tests in `tests/test_synth.py`**

覆盖以下明确断言：
1. `test_acoustic_synthesizer_audio_and_text_modes_shift_toward_target_tone`：利用 `voice_eval.dims._sandbox` 的 `generate_speaker_wav` 与 `ToneEmbedder`，验证：
   - `mode='audio'` 下，当 `formant_blend` 与 `pitch_pull` 从 `0.15` 提高到 `0.90` 时，输出与参考说话人的 `cosine` 相似度单调提升（高参数 > 低参数），且输出经 `validate_simulation_result` 校验为 16 kHz 单声道有限 `float32`、`[-1.0, 1.0]`、时长 `>= 0.5 s`；
   - `mode='text'` 下支持中文 UTF-8 文本，不同说话人作为参考音输出不同谐波基频与频谱包络且可重复（同 `seed` 确定性一致）。
2. `test_validate_simulation_result_rejects_nan_too_short_and_resamples`：验证 NaN/Inf、过短（`< 0.5s`）、过长（`> 60.0s`）均抛出 `SynthesisError`；48 kHz 立体声输入会被自动单声道化并重采样至 16 kHz `float32`。
3. `test_split_command_args_preserves_windows_quoted_spaces`：验证 `split_command_args('"D:/voice confirm/.venv_tts/Scripts/python.exe" plugins/neural_tts_worker.py --engine f5_tts')` 返回 `['D:/voice confirm/.venv_tts/Scripts/python.exe', 'plugins/neural_tts_worker.py', '--engine', 'f5_tts']`，且反斜杠双引号路径 `r'"D:\voice confirm\.venv_tts\Scripts\python.exe" plugins/worker.py'` 同样完整保留 `D:\voice confirm\.venv_tts\Scripts\python.exe`；空命令或未闭合引号抛出 `ValueError`。
4. `test_subprocess_synthesizer_utf8_env_and_temp_cleanup_on_timeout`：
   - 正常 mock Worker（用 `sys.executable -c` 读 `--request` JSON、校验进程内 `PYTHONUTF8=='1'`、读取中文 `text` 与 `reference_text`、向指定 `output_wav_path` 写入 16 kHz 有效 WAV、打印 JSON 状态）返回正确 `SimulationResult`，且 `TemporaryDirectory(prefix='vpsyn_')` 在调用返回后已被彻底删除；
   - 超时 Worker（`sleep(5)`，设 `timeout_s=0.2`）触发 `SynthesisError` 并完整清理临时目录、不留孤儿进程；非 `0` 退出码或写超大/非法 WAV 均抛 `SynthesisError` 并清理临时目录。
5. `test_load_synthesizer_variants`：`'none'` 返回 `None`，`'acoustic'` 返回 `AcousticFormantSynthesizer`，`'neural_subprocess'` 无 `neural_cmd` 抛 `ValueError`，非法 `kind` 抛 `ValueError`。

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_synth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceprint.synth'`.

- [ ] **Step 3: Implement `voiceprint/synth.py`**

按以下边界最小高质实现：
- 定义 `SynthesisError`、`SimulationRequest`、`SimulationResult`、`VoiceSynthesizer`（含 `supported_modes` 默认 `('audio', 'text')` 及 `describe()`）；
- 实现 `validate_simulation_result`：检查有限性、单声道混音、必要时基于 `math.gcd` + `scipy.signal.resample_poly` 重采样至 `SR=16000`、校验时长边界、绝对峰值若超 `1.0` 做限幅并转连续 `np.float32`；
- 实现 `AcousticFormantSynthesizer`：
  - 基于短时自相关（70–350 Hz）提取参考音频与源音频中位基频 $F_0$ 与动态短时 STFT（25 ms Hanning 窗、10 ms 帧移、512 点 FFT）活跃帧功率谱包络；
  - `mode == 'audio'`：逐活跃帧计算瞬时基频靠拢与频谱重采样/包络迁移（按 `formant_blend` 混合参考包络与源细节包络、按 `pitch_pull` 向参考基频与谐波栅格拉齐、再乘 `gain_scale` 并经 `tanh` 软限幅）；
  - `mode == 'text'`：基于 `text` UTF-8 字节序列确定性生成音节时长序列（含浊音谐波段、清音摩擦噪声与句逗停顿静音，避免静态平稳音被 utterance CMVN 完全抵消），使用参考说话人 $F_0^{\text{ref}}$ + 微扰激励与动态包络滤波生成 `>= 1.2 s` 语音；
- 实现 `split_command_args(cmd)`：接受 `list/tuple[str]` 直接校验非空元素拷贝；接受 `str` 时按字符状态机切分空白并支持双引号/单引号，在保留引号外普通分离空格的同时不把 Windows 反斜杠路径分隔符 `\` 当作转义吞掉；
- 实现 `SubprocessNeuralSynthesizer`：创建 `TemporaryDirectory(prefix='vpsyn_')`，构造带 `PYTHONUTF8='1'` 与 `PYTHONIOENCODING='utf-8'` 的子进程环境，将请求写入 `request.json`，通过 `run_contained(cmd + ['--request', str(req_json)], timeout=self.timeout_s, cwd=self.cwd, env=child_env)` 执行，校验退出码、`timed_out`、`output_wav_path` 存在性与合法 WAV 内容（调用 `sf.read` + `validate_simulation_result`），全程用 `try/finally` 自动清理临时目录；
- 实现 `load_synthesizer`。

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -W error -m unittest tests/test_synth.py -v`
Expected: PASS (all tests in `tests/test_synth.py` OK with 0 warnings).

- [ ] **Step 5: Commit**

```bash
git add voiceprint/synth.py tests/test_synth.py
git commit -m "feat(synth): add controllable synthesizer interface, acoustic formant backend and isolated neural subprocess runner"
```

---

### Task 2: `VoiceAgent` 受控 `模拟` 指令、五道安全门禁与 Self-Refine 自检闭环 (`agent/service.py` & `agent/cli.py`)

**Files:**
- Modify: `agent/service.py:1-421`
- Modify: `agent/cli.py:1-42`
- Create: `tests/test_agent_simulate.py`

**Interfaces:**
- Consumes: `voiceprint/synth.py` (`VoiceSynthesizer`, `AcousticFormantSynthesizer`, `SimulationRequest`, `SynthesisError`, `MAX_SIM_TEXT_CHARS`, `validate_simulation_result`, `load_synthesizer`), `voiceprint/spoof.py` (`SpreadSpectrumWatermarkDetector`, `embed_spread_watermark`, `DEFAULT_WATERMARK_PAYLOAD`).
- Produces:
  - `VoiceAgent.__init__(..., synthesizer: VoiceSynthesizer | None | object = _UNSET, max_sim_attempts: int = 3)` (`self.simulated_dir = self.data_dir / 'simulated'`)
  - `VoiceAgent._run_simulate_core(self, name: str, *, consent: bool, ref_path: str, mode: Literal['audio', 'text'], src_path: str | None = None, text: str | None = None, reference_text: str | None = None, max_attempts: int | None = None, collect_best_candidate: bool = False) -> tuple[dict, np.ndarray | None]`
  - CLI 指令 `'模拟': self._simulate`；`agent/cli.py` 支持 `--synthesizer`（默认 `'acoustic'`）、`--neural-cmd`、`--neural-timeout`、`--max-sim-attempts`、`--spoof-guard`。

- [ ] **Step 1: Write failing unit tests in `tests/test_agent_simulate.py`**

覆盖以下明确断言（全部在 `patched_dpapi()` + `ToneEmbedder` / mock 合成器下确定性运行）：
1. `test_simulate_precondition_gates_enforce_consent_ownership_and_anti_recursive_watermark`：
   - 缺少 `本人授权`（如只写 `同意`）→ `ok=False`, `result['reason']=='consent_required'`, `simulated_dir` 无文件；
   - 未注册目标 → `ok=False`, `result['reason']=='unknown'`；
   - **跨人冒充**：目标写 `甲`，参考音传 `乙` 的真实录音 → `ok=False`, `result['reason']=='reference_speaker_mismatch'`；
   - **反递归克隆**：参考音嵌入了 16-bit 水印 → 即便 `VoiceAgent(..., spoof_detector=None)` 也能以 `result['reason']=='watermark_detected'` 强制拦截；
   - 不支持模式（`supported_modes=('text',)` 时传 `源音频`）→ `ok=False`, `result['reason']=='unsupported_mode'`。
2. `test_simulate_mismatch_and_failed_retries_do_not_pollute_lockout_events`（**Review Focus #4**）：
   - 连续执行 6 次跨人冒充拒绝 + 未收敛重试失败后，查询 `store.events('甲')` 与 `store.recent_failures('甲', LOCK_MINUTES)` 均仍为 `0`，紧接着正常 `验证 甲 <probe>` 仍返回 `ok=True`, `decision='accept'`。
3. `test_simulate_sanitizes_speaker_name_in_output_filename`（**Review Focus #3**）：
   - 注册含 `../evil\\spk` 字符的说话人名并模拟收敛后，检查产出文件路径 `Path(res['output_path']).resolve()` 严格位于 `agent.simulated_dir.resolve()` 内部，文件名不含 `..`、`/`、`\`。
4. `test_simulate_self_refine_retries_improve_and_converge_with_forced_watermark`：
   - 构造可控 mock `VoiceSynthesizer`：第 1 轮返回削波过大（峰值 `1.0` 且相似度略低）的波形，验证 Self-Refine 按失败原因降低 `gain_scale` 并提高 `formant_blend`/`pitch_pull`，在第 2 或第 3 轮满足 `质检通过 && 水印回读通过 && decision == 'accept'` 后提前退出并释放带 `0x5653` 水印 WAV 到 `simulated_dir`；
   - 若所有轮次均未达 `'accept'`（mock 始终输出他声波形），返回 `ok=False`, `result['reason']=='similarity_not_converged'`, `result['converged'] is False`，**不释放 WAV 文件到 `simulated_dir`**；但 `collect_best_candidate=True` 时 `_run_simulate_core` 第二个返回值返回质量与水印通过的最优候选波形（供 Task 5 Track B）。
5. `test_simulate_detail_false_redacts_scores_and_loopback_blocked_under_spoof_guard`：
   - `detail=False` 下运行 `模拟` 后，用 `FORBIDDEN_NON_DETAIL_KEYS`、`FORBIDDEN_NON_DETAIL_TERMS` 与浮点阈值正则检查返回 `result` 与 `reply` 零分数泄露；
   - 开启 `spoof_detector=SpreadSpectrumWatermarkDetector()`（且带 `audio_roots=[allowed_dir]`）时，直接对刚返回的 `output_path` 执行 `验证 甲 "<output_path>"` 与 `辨认 "<output_path>"`，不会被 `audio_roots` 拦截，且被防伪检测精准拒绝（`reason=='spoof_detected'`）。
6. `test_cli_supports_synthesizer_flags`：验证 `agent.cli` `argparse` 默认值与 `--synthesizer none`、`-h` 正常解析，且已有 `匿名`/`变声` 指令行为完全不变。

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_agent_simulate.py -v`
Expected: FAIL (constructor unexpected keyword `synthesizer` or missing `模拟` handler).

- [ ] **Step 3: Implement `simulate` gating, Self-Refine state machine & CLI args in `agent/service.py` and `agent/cli.py`**

按以下逻辑实现：
1. `agent/service.py`：
   - 定义 `_UNSET = object()` 与 `_safe_filename_token(name: str) -> str`（仅保留字母数字/下划线/短横线，兜底 `'spk'`）；
   - `VoiceAgent.__init__`: 接收 `synthesizer=_UNSET`、`max_sim_attempts: int = 3`（拒绝 `bool`，范围限 `1..5`），`_UNSET` 时默认实例化 `AcousticFormantSynthesizer()`；初始化 `self.simulated_dir = self.data_dir / 'simulated'`、纳入 `self.audio_roots`（当外部传 `audio_roots` 时）并初始化 `self._watermark_guard = SpreadSpectrumWatermarkDetector()`；
   - `_dispatch` 注册 `'模拟': self._simulate`（不改变 `匿名`/`变声` 逻辑）；更新 `HELP` 文本增补 `模拟` 用法；
   - 实现 `_parse_simulate_tokens(rest)` 支持 `模拟 <姓名> 本人授权 "<参考音>" (源音频 "<源>" | 文本 "<文本>" [参考文本 "<参考文本>"])`；
   - 实现 `_run_simulate_core(...) -> tuple[dict, np.ndarray | None]`：
     - 严格按 5 道门禁校验（不调用 `self.store.event`）；
     - 循环 `1..eff_max_attempts`：调用 `synthesizer.synthesize` → `validate_simulation_result` → `embed_spread_watermark` → `self._watermark_guard.detect` 强制回读验签 → 临时 WAV 落盘跑 `analyze_audio_quality` + `self._score(..., task='verify')` → 判定 `converged = wm_ok and quality_ok and score['decision'] == 'accept'`；
     - 根据失败类型单调调整 `gain_scale`、`formant_blend`、`pitch_pull`、`wm_strength`；记录通过水印与非致命质量检查的最佳候选 `(score - threshold, wav, wm_res, sim_res)`；
     - 收敛则原子写 `self.simulated_dir / f'sim_{safe_tok}_{uuid4().hex[:8]}.wav'`，构造遵循 `detail` 脱敏开关的结果与审计日志；未收敛则不写入 `simulated_dir`，返回 `(outcome, best_wav if collect_best_candidate else None)`。
2. `agent/cli.py`：新增 `--synthesizer`、`--neural-cmd`、`--neural-timeout`、`--max-sim-attempts`、`--spoof-guard` 参数构造合成器并传给 `VoiceAgent`。

- [ ] **Step 4: Run tests (new + existing agent tests) to verify pass**

Run: `.venv/bin/python -W error -m unittest tests/test_agent_simulate.py tests/test_voice_eval_agent_checks.py tests/test_spoof_guard.py -v`
Expected: PASS (0 failures, 0 warnings).

- [ ] **Step 5: Commit**

```bash
git add agent/service.py agent/cli.py tests/test_agent_simulate.py
git commit -m "feat(agent): add controlled voice simulation command, security gates and Self-Refine retry loop"
```

---

### Task 3: MCP Server 动态工具暴露与 `voiceprint_simulate` 安全协议 (`agent/mcp_server.py`)

**Files:**
- Modify: `agent/mcp_server.py:21-511`
- Create: `tests/test_mcp_simulate.py`

**Interfaces:**
- Consumes: `VoiceAgent._run_simulate_core`, `VoiceAgent.simulated_dir`, `voiceprint/synth.py` (`load_synthesizer`, `VoiceSynthesizer`).
- Produces:
  - `_WRITE_TOOLS` 包含 `'voiceprint_simulate'`
  - `build_default_server(..., synthesizer: VoiceSynthesizer | None = None, max_sim_attempts: int = 3)`
  - MCP Tool `'voiceprint_simulate'`（仅在 `agent.synthesizer is not None` 时出现在 `tools/list`），支持 `name`、`consent`、`reference_audio_path`/`reference_audio_base64`、`reference_text`、`source_audio_path`/`source_audio_base64`、`text`、`max_attempts`、`include_audio_base64`
  - `_health_dict()` 包含 `'synthesizer'` 与 `'simulation_enabled'`
  - `agent/mcp_server.py:main()` CLI 默认 `--synthesizer acoustic`、`--neural-cmd`、`--neural-timeout`、`--max-sim-attempts`。

- [ ] **Step 1: Write failing unit tests in `tests/test_mcp_simulate.py`**

覆盖以下明确断言：
1. `test_mcp_simulate_hidden_by_default_in_build_default_server_and_exposed_when_enabled`：
   - 无 `synthesizer` 参数调用 `build_default_server(db, embedder)` 时，`tools/list` 仍恰为原有 8 个工具且不包含 `voiceprint_simulate`（且 `_health_dict()['simulation_enabled'] is False`）；
   - 传入 `synthesizer=AcousticFormantSynthesizer()` 后，`tools/list` 暴露全部 9 个工具含 `voiceprint_simulate`（`_health_dict()['simulation_enabled'] is True`）。
2. `test_mcp_simulate_disabled_when_no_synthesizer_and_rejects_bool_max_attempts`（**Review Focus #5**）：
   - `synthesizer=None` 时直接 `tools/call` `voiceprint_simulate` 返回 `isError=True`, `error='synthesizer_disabled'`；
   - 启用合成器后，传 `consent="true"` 返回 `invalid_arguments`，传 `consent=False` 返回 `consent_required`，传 `max_attempts=True`（或 `0` / `6`）返回 `invalid_arguments`。
3. `test_mcp_simulate_readonly_blocked_and_forbids_arbitrary_output_path`：
   - `--read-only` 下拦截 `voiceprint_simulate` 返回 `error='read_only_mode'`；
   - 若请求附带额外字段或尝试覆盖目标（如 `source_audio_*` 与 `text` 混传或缺失）均干净报错，输出文件仅能落入 `server.agent.simulated_dir`。
4. `test_mcp_simulate_no_roots_base64_roundtrip_and_with_roots_path_loopback`：
   - **无 `audio_roots` 纯 Base64 隔离模式**：仍拒绝 `reference_audio_path`（`error='path_not_allowed'`）；使用 `reference_audio_base64` + `text` + `include_audio_base64=True` 成功生成后，载荷包含合法 base64 WAV 且无 `detail=False` 敏感分数字段；随后将返回的 `audio_base64` 直接传给 `voiceprint_verify` 验证被防伪系统拦截（`reason='spoof_detected'`）；
   - **有 `audio_roots` 本地路径模式**：使用白名单内的 `reference_audio_path` 与 `source_audio_path` 生成后，直接将返回的 `output_path` 传给 `voiceprint_verify` 时能顺利穿过 `_ephemeral_roots` 目录检查并被防伪系统拦截（`reason='spoof_detected'`）。
5. `test_mcp_cli_defaults_to_acoustic_synthesizer`：验证 `agent/mcp_server.py` CLI 参数解析 `--synthesizer` 默认为 `'acoustic'` 且支持 `'none'` 与 `'neural_subprocess'`。

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_mcp_simulate.py -v`
Expected: FAIL (`unknown_tool` / unexpected keyword `synthesizer`).

- [ ] **Step 3: Implement `voiceprint_simulate` and dynamic exposure in `agent/mcp_server.py`**

按以下边界修改 `agent/mcp_server.py`：
1. 在 `_TOOL_SCHEMAS` 末尾追加 `voiceprint_simulate` schema，将 `'voiceprint_simulate'` 加入 `_WRITE_TOOLS`；
2. `_health_dict()` 增加 `'simulation_enabled'` 与 `'synthesizer'`；
3. `_ephemeral_roots()` 更新为 `base = list(self.audio_roots) + [self.agent.separated_dir, self.agent.simulated_dir] + list(extra_dirs)`；
4. `tools/list` 分支在 `self.agent.synthesizer is None` 时过滤掉 `'voiceprint_simulate'`；
5. `_call_tool_locked` 实现 `voiceprint_simulate` 分支：
   - 若 `self.agent.synthesizer is None` 返回 `_tool_response({'ok': False, 'error': 'synthesizer_disabled', ...}, is_error=True)`；
   - 严格校验 `name`（非空字符串）、`consent`（`isinstance(consent, bool)` 且 `consent is True`）、`reference_text`（可选 str）、`max_attempts`（可选严格 `isinstance(v, int) and not isinstance(v, bool)` 且 `1 <= v <= 5`）、`include_audio_base64`（严格 bool，默认 False）；
   - 参考音频仅允许 `reference_audio_path` 或 `reference_audio_base64` 之一（无 `audio_roots` 时路径报错 `path_not_allowed`）；源输入仅允许 `source_audio_path`、`source_audio_base64`、`text` 三选一；
   - 临时目录中解码 base64 后通过 `_ephemeral_roots` 调用 `self.agent._run_simulate_core(...)`（同时记一次审计或直接在 `_run_simulate_core` 统一记审计），按需编码 output WAV 为 base64（检查文件大小 `<= MAX_BASE64_BYTES`）；
6. `build_default_server(...)` 新增默认 `synthesizer: VoiceSynthesizer | None = None, max_sim_attempts: int = 3`；`main()` 新增 CLI 选项（`--synthesizer` 默认 `'acoustic'`）。

- [ ] **Step 4: Run ALL existing MCP tests + new MCP simulate tests to verify pass**

Run: `.venv/bin/python -W error -m unittest tests/test_mcp_server.py tests/test_mcp_simulate.py -v`
Expected: PASS (`tests/test_mcp_server.py` passes completely unmodified + all new tests in `tests/test_mcp_simulate.py` pass).

- [ ] **Step 5: Commit**

```bash
git add agent/mcp_server.py tests/test_mcp_simulate.py
git commit -m "feat(mcp): expose voiceprint_simulate dynamically with strict consent, type and path guards"
```

---

### Task 4: 去水印攻击算子与 6 项仿真安全红队探针 (`voice_eval/audio_synth.py` & `voice_eval/dims/redteam.py`)

**Files:**
- Modify: `voice_eval/audio_synth.py:200-241`
- Modify: `voice_eval/dims/redteam.py:550-573` (plus helper `_run_simulation_probes`)
- Create: `tests/test_simulation_eval_and_redteam.py` (Part 1: 攻击算子 & 红队验证)

**Interfaces:**
- Consumes: `VoiceAgent` `模拟` 指令、`build_default_server`、`AcousticFormantSynthesizer`、`SpreadSpectrumWatermarkDetector`、`LOCK_MINUTES`.
- Produces:
  - `apply_lowpass_attack(wav: np.ndarray, sr: int, cutoff_hz: float = 3600.0) -> np.ndarray`
  - `apply_resample_attack(wav: np.ndarray, sr: int, mid_sr: int = 8000) -> np.ndarray`
  - `apply_gain_clip_attack(wav: np.ndarray, sr: int, drive: float = 1.8) -> np.ndarray`
  - `evaluate_redteam()` 额外包含 6 项 `'simulation_security'` 探针（`sim-missing-consent`、`sim-unregistered-target`、`sim-cross-speaker-impostor`、`sim-recursive-watermark-clone`、`sim-loopback-spoof-blocked`、`sim-mcp-guardrails`）。

- [ ] **Step 1: Write failing unit tests in `tests/test_simulation_eval_and_redteam.py`**

添加第一批单元测试：
1. `test_watermark_removal_attack_operators_preserve_bounds_and_degrade_zscore`：
   - 生成测试人声并嵌入 `0x5653` 水印，分别应用 `apply_lowpass_attack(..., cutoff_hz=4500.0)`、`apply_resample_attack(..., mid_sr=8000)`、`apply_gain_clip_attack(..., drive=1.8)`，校验返回类型为 1D `np.float32`、有限且位于 `[-1.0, 1.0]`、长度对齐且 4.5 kHz 低通下频域直方对比原干净能量在高频衰减；非法 `sr`/`cutoff_hz`/`drive` 抛 `ValueError`。
2. `test_evaluate_redteam_includes_six_simulation_probes_all_green`：
   - 在 `with patched_dpapi():` 下调用 `evaluate_redteam()`，断言总体 `passed is True`, `yellow == 0`, `red == 0`；
   - 断言返回的 `checks` 中包含全部 6 个新 ID：`sim-missing-consent`、`sim-unregistered-target`、`sim-cross-speaker-impostor`、`sim-recursive-watermark-clone`、`sim-loopback-spoof-blocked`、`sim-mcp-guardrails`，且每项 `status == 'GREEN'`。

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_simulation_eval_and_redteam.py -v`
Expected: FAIL (`ImportError` on attack helpers / missing `sim-*` redteam check IDs).

- [ ] **Step 3: Implement attack helpers and 6 simulation red-team probes**

1. 在 `voice_eval/audio_synth.py` 实现 `apply_lowpass_attack`（4 阶零相位 `butter(..., btype='lowpass', output='sos')` + `sosfiltfilt`）、`apply_resample_attack`（`sr -> mid_sr -> sr` `resample_poly` 往返对齐原长度）、`apply_gain_clip_attack`（`np.clip(sig * drive, -1.0, 1.0)`）。
2. 在 `voice_eval/dims/redteam.py` 实现 `_run_simulation_probes(embedder, allowed_dir, takes_a, takes_b)` 并接入 `evaluate_redteam()`（在 `for cid, st, det in ...: _record_check(checks, cid, 'simulation_security', st, det)` 中记录）：
   - 对每项使用独立的 `_run_isolated` / `build_default_server(..., synthesizer=AcousticFormantSynthesizer())` 并检查 `_has_null_event_scores` 与 zero plaintext vector leak；
   - `sim-cross-speaker-impostor` 额外核验 `agent.store.recent_failures('甲', LOCK_MINUTES) == 0`；
   - `sim-loopback-spoof-blocked` 必须在开启 `SpreadSpectrumWatermarkDetector()` 下执行合法 `模拟` 并在同一实例上分别运行 `验证 甲 "<sim_wav>"` 与 `辨认 "<sim_wav>"`，断言均因 `spoof_detected` 拒答。

- [ ] **Step 4: Run redteam existing + new tests to verify pass**

Run: `.venv/bin/python -W error -m unittest tests/test_redteam_extended.py tests/test_simulation_eval_and_redteam.py -v`
Expected: PASS (`tests/test_redteam_extended.py` passes completely unmodified + new tests pass).

- [ ] **Step 5: Commit**

```bash
git add voice_eval/audio_synth.py voice_eval/dims/redteam.py tests/test_simulation_eval_and_redteam.py
git commit -m "feat(redteam): add watermark removal attacks and 6 voice simulation security probes"
```

---

### Task 5: 双轨仿真闭环基准评测与 `sim-bench` 子命令 (`voice_eval/dims/simulation.py` & `voice_eval/__main__.py`)

**Files:**
- Create: `voice_eval/dims/simulation.py`
- Modify: `voice_eval/__main__.py:1-103`
- Modify: `tests/test_simulation_eval_and_redteam.py` (Part 2: 双轨 `sim-bench` 验证)

**Interfaces:**
- Consumes: `VoiceAgent._run_simulate_core`, `VoiceSynthesizer`, `AcousticFormantSynthesizer`, `load_synthesizer`, `SpreadSpectrumWatermarkDetector`, `analyze_audio_quality`, `voice_eval.audio_synth` 攻击算子, `voice_eval.dims._sandbox` (`ToneEmbedder`, `sandbox_cohort`, `sandbox_enroll_takes`, `generate_speaker_wav`), `voice_eval.dims.perf.current_rss_mb`.
- Produces:
  - `evaluate_simulation(embedder, synthesizer: VoiceSynthesizer, *, max_attempts: int = 3, speakers_limit: int = 4) -> dict`
  - `write_simulation_report(report: dict, out_dir: Path) -> tuple[Path, Path]`
  - `python -m voice_eval sim-bench [--embedder tone|stats|vpnet|ecapa] [--synthesizer acoustic|neural_subprocess] [--neural-cmd ...] [--neural-timeout 120.0] [--max-attempts 3] [--out <dir>]`

- [ ] **Step 1: Add failing unit tests for `simulation.py` and `sim-bench` CLI in `tests/test_simulation_eval_and_redteam.py`**

追加第二批测试：
1. `test_evaluate_simulation_tracks_a_and_b_under_tone_embedder`：
   - 在 `patched_dpapi()` + `sandbox_cohort(ToneEmbedder())` 下运行 `evaluate_simulation(ToneEmbedder(), AcousticFormantSynthesizer(), max_attempts=3, speakers_limit=2)`：
   - 验证输出包含 `track_a`（`total_cases > 0`, `attempt1_mean_raw_sim`, `final_mean_raw_sim`, `delta_raw_sim >= 0.0`, `attempt1_mean_norm_score`, `final_mean_norm_score`, `convergence_rate`, `avg_attempts`）、`track_b`（`quality_pass_rate == 1.0`, `watermark_readback_rate == 1.0`, `loopback_spoof_block_rate == 1.0`, `attack_survival` 包含全部 7 种攻击条件键并各自统计存活率与平均 z-score）、`latency_ms`（`p50`, `p95`）和 `cases`；
   - 验证 `eval` 沙箱中的授权与跨人门禁拒绝率为 `1.0`。
2. `test_evaluate_simulation_skips_unsupported_audio_mode_gracefully`：
   - 传入 `supported_modes=('text',)` 的自定义/subprocess 合成器时，`mode='audio'` 用例被显式计数为 `skipped_unsupported_cases`，仅基于 `mode='text'` 计算有效指标不抛异常。
3. `test_evaluate_simulation_track_b_collects_when_track_a_not_converged`：
   - 传入始终输出低 SIM 但合法波形的 mock 合成器（模拟 `acoustic` + `vpnet` 未过门禁情形），验证 `track_a['convergence_rate'] == 0.0` 且**未释放进 `simulated_dir`**，但 `track_b['watermark_readback_rate'] == 1.0`、`loopback_spoof_block_rate == 1.0` 及 `attack_survival` 仍完整算出非空统计！
4. `test_cli_sim_bench_writes_json_and_markdown_reports`：
   - 在临时目录执行 `voice_eval.__main__.main(['sim-bench', '--embedder', 'tone', '--synthesizer', 'acoustic', '--out', tmp])`，断言成功返回并在 `tmp` 生成 `sim_bench_tone_acoustic_formant.json` 与 `.md`。

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_simulation_eval_and_redteam.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'voice_eval.dims.simulation'`).

- [ ] **Step 3: Implement `voice_eval/dims/simulation.py` and `sim-bench` in `voice_eval/__main__.py`**

按以下边界实现：
1. `voice_eval/dims/simulation.py`：
   - 对接两类数据来源：
     - `embedder.name == 'tone'`：直接使用 `sandbox_cohort` + `sandbox_enroll_takes` 构建 2–4 个沙箱说话人，每名说话人配 `mode='audio'`（源音频为另一位说话人波形）与 `mode='text'`（中文验证短句）用例；
     - 其他真实后端（`stats|vpnet|ecapa`）：从 `data/LibriSpeech/test-clean` 取前 `speakers_limit` 位说话人（前 3 段作注册与参考音、若存在真实文本 `.trans.txt` 一并读取填入 `reference_text`！、另一说话人一段作 `源音频`），数据缺失时给出明确 `FileNotFoundError` 提示；
   - 对每个用例调用 `agent._run_simulate_core(..., collect_best_candidate=True)` 采集 Track A 统计，再对有效水印波形跑 Track B（回灌带防伪的 `验证` / `辨认` + 7 项去水印攻击：`white_noise_15db`、`white_noise_10db`、`duration_crop_0.6s`、`lowpass_4500hz`、`lowpass_3600hz`、`resample_8k_roundtrip`、`gain_clip_1.8x`）；
   - 实现 `write_simulation_report(report, out_dir)` 输出可追溯 JSON（不含任何原始 waveform/embedding）与格式化 Markdown 指标表。
2. `voice_eval/__main__.py`：保留现有 4 个子命令完全不变，新增 `sim-bench` 解析分支并调用 `load_synthesizer`、`evaluate_simulation` 与 `write_simulation_report`（注意支持 `tone`、`stats`、`vpnet`、`ecapa` 并在 `tone` 时自动挂载 `sandbox_cohort` + `ToneEmbedder`）。

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -W error -m unittest tests/test_simulation_eval_and_redteam.py -v`
Expected: PASS (0 failures, 0 warnings).

- [ ] **Step 5: Commit**

```bash
git add voice_eval/dims/simulation.py voice_eval/__main__.py tests/test_simulation_eval_and_redteam.py
git commit -m "feat(eval): implement dual-track simulation benchmark and sim-bench CLI command"
```

---

### Task 6: Windows GPU 神经克隆独立 Worker 脚本、README 同步与全仓 `-W error` 终验收

**Files:**
- Create: `plugins/neural_tts_worker.py`
- Modify: `tests/test_synth.py` (补充 `plugins/neural_tts_worker.py` `--help` 与 mock 协议端到端冒烟验证)
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 `SubprocessNeuralSynthesizer` JSON 请求格式（`mode`, `reference_audio_path`, `reference_text`, `source_audio_path`, `text`, `output_wav_path`, `gain_scale`, `seed`）。
- Produces: 可在 Windows RTX 4070 SUPER 独立 `.venv_tts` 中直接运行的 `plugins/neural_tts_worker.py --engine {f5_tts,cosyvoice2} --request <path>` 及更新后的 `README.md`。

- [ ] **Step 1: Write failing CLI contract test for `plugins/neural_tts_worker.py` in `tests/test_synth.py`**

添加测试 `test_neural_tts_worker_cli_help_and_zero_core_dependency`：
- 验证 `SubprocessNeuralSynthesizer` 实际通过子进程执行 `[sys.executable, 'plugins/neural_tts_worker.py', '--help']` 成功返回 0（且在未安装 `f5_tts` 的主仓 `.venv` 下导入或查看帮助绝不报 `ModuleNotFoundError`）；
- 验证向 `plugins/neural_tts_worker.py` 传递非法 `mode='audio'`（当 `--engine f5_tts` 时）返回结构化错误 JSON；以及通过 Python `-c` 注入轻量假 `f5_tts.api` 模块测试 Worker 完整解析 `request.json` → 输出规范 16 kHz WAV → 打印合规单行 JSON 状态的协议端到端链路！

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -W error -m unittest tests/test_synth.py -k worker -v`
Expected: FAIL (`plugins/neural_tts_worker.py` does not exist).

- [ ] **Step 3: Implement `plugins/neural_tts_worker.py` and update `README.md`**

1. 创建 `plugins/neural_tts_worker.py`：
   - 仅使用标准库 + `numpy` + `soundfile`（可选懒加载 `torch`、`f5_tts.api` 或 `cosyvoice`），按 `--engine f5_tts|cosyvoice2 --request <json>` 解析 UTF-8 JSON；
   - `--engine f5_tts`：支持 `mode == 'text'`，懒加载 `F5TTS` 并以 `ref_file`、`ref_text`（如有则透传防离线 Whisper 报错）、`gen_text`、`file_wave` 合成，结束后重采样/限幅写 `output_wav_path` 并调用 `torch.cuda.empty_cache()`，stdout 仅输出单行 JSON；
   - `--engine cosyvoice2`：预留独立分支支持 `text`（`inference_zero_shot`）与 `audio`（`inference_vc`），若未安装对应库则给出清晰中文安装报错并非零退出。
2. 更新 `README.md`：
   - 完整呈现项目现有能力（声纹 Agent + MCP + 100 ms 质检 + 0.5 s 16-bit 水印 + 七维评测）与新增的**可控声音模拟、Self-Refine 自检闭环与攻防评测（`sim-bench`）**；
   - 附上 Windows 显卡机（如 `"D:/voice confirm/.venv_tts/Scripts/python.exe"`）独立环境安装 `F5-TTS`、启动 `agent.cli` / `agent.mcp_server` 与跑真实 `sim-bench --embedder vpnet|ecapa` 的精确命令说明；
   - 保持文档严谨简洁、所有命令可直接复制运行。

- [ ] **Step 4: Run full verification suite cleanly with zero warnings (`-W error`)**

1. 确认未改动任何原有 73 项旧测试与冻结配置：
   `git diff --name-only main...HEAD`（若在隔离分支）或核对 `git log` 提交记录确认未改动 `requirements.txt`、`voice_eval/protocol.json` 及任何旧测试文件。
2. 全量回归测试（旧 73 项 + 全部新增测试）：
   `.venv/bin/python -W error -m unittest discover -s tests -v`
   Expected: ALL PASS with exit code 0 and 0 warnings.
3. 评测健康检查与沙箱闭环基准端到端冒烟：
   `.venv/bin/python -W error -m voice_eval audit`
   `.venv/bin/python -W error -m voice_eval sim-bench --embedder tone --synthesizer acoustic`
   Expected: both exit 0 and print `评测健康检查通过` + 生成可用 report 文件。

- [ ] **Step 5: Commit**

```bash
git add plugins/neural_tts_worker.py README.md tests/test_synth.py
git commit -m "feat(plugin): add isolated Windows F5-TTS worker adapter and update project README"
```
