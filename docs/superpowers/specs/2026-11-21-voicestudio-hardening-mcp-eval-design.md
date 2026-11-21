# 声纹 Agent 五大工程与评测增强设计方案（借鉴 VoiceStudio 精华）

**日期**：2026-11-21  
**目标仓库**：`/Users/cjjackh/.gemini/antigravity/scratch/voiceprint-agent`  
**基于提交**：`main`（`9f9f71a`）  
**方案选型**：方案 A（零新增重型外部依赖，基于 Python 标准库与现有 5 依赖 `numpy / scipy / soundfile / cryptography / torch` 实现 + 预留外部防伪模型插拔接口）  
**状态**：待用户最终审批

---

## 1. 背景、意图与核心目标

当前 `voiceprint-agent` 已具备声纹识别 Agent 状态机（`agent/service.py`）与七维离线评测及三级门禁体系（`voice_eval/` v3，44 项 `-W error` 测试通过）。参考 `debpalash/VoiceStudio` 在本地语音工具平台中的生产级护栏与基准经验，本轮围绕 **Agent 评测与架构（Agent Eval & Harness）** 目标岗位，实施五大高价值增强：

1. **标准 MCP Server 与输入安全边界**：将声纹 Agent 能力通过标准 JSON-RPC 2.0 `stdio` MCP 协议对外暴露，补齐 `audio_roots` 强制边界、`data:` Base64 URI 剥离、魔数嗅探与默认 `detail=False` 防泄分契约。
2. **音频容器完整性校验与模型无关信号质检**：在模型推理前拦截符号链接、扩展名与魔数不匹配、RIFF/WAV 截断假头文件；新增 100ms 滑窗纯 DSP 信号质量检测（空音频、静音、过轻、过响、削波、NaN/Inf），纳入注册质量门控与红队测试。
3. **AI 克隆语音防伪（Anti-Spoofing / 水印检测）护栏与红队评测**：引入可插拔 `SpoofDetector` 接口与零外部依赖的确定性 16-bit 扩频水印嵌入/检测原语，在 `VoiceAgent` 侧提供可选防伪拦截护栏，并在 `dims/redteam.py` 扩展克隆水印冒充与容器/MCP 攻击评测。
4. **Warm RTF 实时率性能基线与严格离线权重锁定**：在 `dims/perf.py` 补齐冷/热分离计算统计与语音标准 **Warm RTF（计算耗时 / 音频时长）**，接入 `gate.py` Tier 3 退化告警；在 `runner.py` 强制运行期离线环境变量与模型指纹审计。
5. **跨平台受控子进程树与隔离评测 Worker**：实现 POSIX 嵌套进程组 + control-pipe EOF、Windows Job Object（`KILL_ON_JOB_CLOSE`）的跨平台进程树管控器，并在 `voice_eval` 提供 `--isolate-worker` 隔离执行模式，防止实验性候选模型超时挂死或崩溃拖垮评测主进程。

---

## 2. 非目标与红线（Non-Goals & Anti-Goals）

- **不新增外部重型依赖**：`requirements.txt` 保持现有 5 个包不变，不引入 Electron、FastAPI、外部 `mcp` SDK 或需联网下载的 AudioSeal/TTS 大权重。
- **不破坏现有 44 项回归与 `protocol.json` 既有门禁口径**：
  - 既有 7 个评测维度名称（`task / tooling / response / flow / perf / redteam / robustness`）与 `case_version = 3` 保持向后兼容；
  - `VoiceAgent` 默认 `spoof_detector=None`，不改变已有 `verify / verify_impostor / robustness` 基线分数；
  - `validation / locked` 说话人互斥隔离、`locked` 严禁参与 Tier 2 采纳决策、`uncertain` 绝不计入自动正确率等核心红线不可松动。
- **不将合成沙箱分数冒充真实声纹精度**：Mac 端基于 `ToneEmbedder` 与合成语料验证全部框架、护栏、MCP、防伪与进程隔离逻辑；真实 `VPNet / ECAPA-TDNN` 的 LibriSpeech 指标仍在 Windows 实机执行。
- **不误杀 macOS `/var` 等系统级父目录符号链接**：符号链接拦截仅针对目标录音文件本身（`Path(text).is_symlink()`）以及解析后越出 `audio_roots` 的路径，避免在 macOS `TemporaryDirectory`（`/var/folders/...` → `/private/var/folders/...`）上产生误报。

---

## 3. 备选方案与权衡（Approaches）

| 方案 | 核心做法 | 权衡与结论 |
|---|---|---|
| **方案 A（用户已选定）**：纯标准库 + 现有 5 依赖实现 + 外部检测器插拔接口 | 纯 stdlib JSON-RPC 2.0 `stdio` MCP Server；纯 `soundfile/numpy` 音频容器与 DSP 质检；内置确定性 16-bit 扩频水印与声学伪造模拟 + `SpoofDetector` 抽象接口（可外挂 AudioSeal/AASIST）；`ctypes` Windows Job + POSIX PGID 进程树管控 | **采用**：零额外依赖、100% 离线确定性可复现、Mac/Windows 均可跑全量 `-W error` 单测，且架构边界清晰便于面试阐述 |
| 方案 B：引入 `mcp` SDK + `audioseal` 第三方依赖与预训练权重 | 依赖官方 `mcp` 包与 Meta AudioSeal 神经网络权重 | **放弃**：破坏极简依赖与离线自包含约束，无网/CI 环境无法直接复现，且引入重型权重维护负担 |

---

## 4. 详细模块设计

### 4.1 支柱一：音频容器完整性校验与信号质量预检

#### 4.1.1 容器完整性与魔数嗅探（`voiceprint/audio.py`）
- 新增 `sniff_audio_format(raw: bytes) -> str | None`：
  - `b'RIFF'` 且 `raw[8:12] == b'WAVE'` → `'.wav'`
  - `b'fLaC'` → `'.flac'`
  - `b'OggS'` → `'.ogg'`
  - `b'ID3'` 或前 2 字节命中合法 MPEG 帧同步头（`raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0` 且非全 `0xFF`） → `'.mp3'`
  - 其余返回 `None`。
- 新增 `_validate_wav_container(resolved: Path) -> None`：
  - 按小端解析 RIFF 头与子 chunk（含奇数长度 1 字节对齐填充），校验声明的 RIFF/chunk 边界不超过真实文件大小，且必须存在非空 `fmt ` 与 `data` chunk；
  - 对非扩展压缩标准 PCM/IEEE-float WAV 校验 `data` chunk 实际可用字节数不小于 `sf.info(resolved).frames` 对应声明下界，阻断“Header 完整但写入中断截断”的假 WAV。
- 强化 `check_audio(path: str, roots: list[Path] | None = None) -> Path`（严格按序执行）：
  1. 空路径与 UNC/网络路径文本预检（现有逻辑，不触碰文件系统）；
  2. 扩展名白名单校验（现有逻辑）；
  3. 文件存在性、`resolve()` 后网络路径与 `roots` 目录白名单校验（现有逻辑 — 若符号链接指向 `roots` 之外，继续报 `'录音不在允许的目录内：...'`，确保与现有 `path_symlink_escape` 回归测试完全兼容）；
  4. **目录内符号链接拒绝**：在 `roots` 校验通过后检查 `Path(text).is_symlink()`，若目标文件本身是符号链接（含指向 `roots` 内部的符号链接），抛 `AudioError('不接受符号链接录音：...')`（不误杀 macOS `/var` 等系统父目录符号链接）；
  5. 文件大小上限校验（现有逻辑）；
  6. **魔数一致性校验**：读取前 16 字节调用 `sniff_audio_format()`，若为 `None` 或与文件真实后缀不一致，抛 `AudioError('录音文件头与扩展名不匹配或格式无效：...')`；
  7. 若后缀为 `.wav`，调用 `_validate_wav_container(resolved)`；
  8. `sf.info()` 时长上下限校验 + `_assert_finite_audio()` 有限性扫描（现有逻辑）。

#### 4.1.2 无模型 100ms 滑窗信号质检（`voiceprint/audio_quality.py`）
- 定义轻量数据结构（纯 `dataclasses`，零新增依赖）：
  - `AudioWarning(kind: str, start: float, end: float)`，`kind ∈ {'empty', 'silence', 'quiet', 'loud', 'clipping', 'invalid'}`
  - `AudioQualityReport(duration, analyzed_seconds, truncated, active_seconds, silence_ratio, clipping_ratio, warnings, passed_enroll_quality, reject_reasons)`
- `analyze_audio_quality(path: str | Path, *, max_seconds: float = 1800.0, max_warnings: int = 100) -> AudioQualityReport`：
  - 校验通道数 `<= 64`、采样率 `<= 768000`；
  - 以 `100 ms` 为窗口分块读取（`float32`, `always_2d=True`），单窗检查：
    - `invalid`：存在非有限值（`NaN/Inf`）；
    - `clipping`：窗内 `np.count_nonzero(np.abs(data) >= 0.999) >= 3`；
    - 音量电平：取多声道中能量最大声道的 RMS（`20 * log10(max(rms, 1e-12))`），避免反相声道混音抵消；
  - 相对电平基线：取所有 `level > -50 dBFS` 活跃窗的中位数 `baseline`；
  - 连续区段聚合：`invalid`（最小时长 0s）、`clipping`（0s）、`silence`（`<= -50 dBFS`，有活跃段时最短 1.0s，全静音时 0s）、`quiet`（`(-50, baseline - 12)`，最短 0.5s）、`loud`（`> baseline + 12`，最短 0.5s）；按 `(start, kind)` 排序后截断至 `max_warnings` 条；
  - 汇总指标：`active_seconds`、`silence_ratio`、`clipping_ratio`；
  - **注册准入判定**：`active_seconds < MIN_SECONDS`（全静音/无有效语音）或 `clipping_ratio > MAX_ENROLL_CLIPPING_RATIO`（默认 `0.25`）或存在 `invalid` / `empty` → `passed_enroll_quality = False` 并填 `reject_reasons`。
- **`VoiceAgent` 集成（经 `/grill-me` 确认）**：
  - `_enroll()` 在 `check_audio()` 之后调用 `analyze_audio_quality()`，若 `not report.passed_enroll_quality`（纯静音 `active_seconds < MIN_SECONDS` 或重度削波 `clipping_ratio > 0.25` 或存在 `invalid`/`empty`），拒绝注册并返回 `ok=False, result={'reason': 'poor_audio_quality', 'quality_reasons': ...}`；正常音频附带精简摘要；
  - `_verify()` / `_identify()` 在 `check_audio()` 之后调用 `analyze_audio_quality()`：**硬拒绝纯静音（`active_seconds < MIN_SECONDS`）与 `empty`/`invalid`**（防止退化零向量进入声纹模型导致随机误接受，且与 `AudioError` 一致仅写 `store.audit` 不写 `store.events`），但**放行削波与高噪音频**以保持既有 `robustness_*` 恶劣工况评测口径完全一致；
  - 新增命令 `质检 <path>`（工具名 `'quality'`），返回结构化质检结果；`check_audio()` 的容器完整性与 `NaN/Inf` 硬阻断对所有命令生效。

---

### 4.2 支柱二：零依赖标准 MCP Server（`agent/mcp_server.py`）

- **协议层**：实现 MCP `stdio` JSON-RPC 2.0 传输（按行读取/写出 JSON，支持 `initialize`、`notifications/initialized`、`ping`、`tools/list`、`tools/call`、`resources/list`、`resources/read`），兼容主流 MCP 客户端（Claude Desktop / Cursor / Antigravity）。
- **核心安全契约**：
  1. **强制路径边界**：构造 `VoiceprintMCPServer(agent_or_factory, audio_roots=...)` 时，若 `audio_roots` 为空或未配置，**拒绝任何 `audio_path` / `audio_paths` 路径型入参**（返回结构化错误 `path_not_allowed`），仅允许受控 `audio_base64` 临时文件或纯查询工具；
  2. **Base64 与 `data:` URI 安全解码**：支持 `audio_base64` / `audio_base64_list`，自动剥离 `data:audio/...;base64,` 前缀，强制 `base64.b64decode(..., validate=True)`，单文件解码字节上限（默认 32 MiB），用 `sniff_audio_format()` 嗅探真实格式后写入独立临时目录（赋予真实后缀，避免扩展名欺骗），执行完毕在 `finally` 立即清理；
     - **注册多会话与 30 秒门控策略（架构裁定）**：默认将 `audio_base64_list` 全部写入**同一个**临时目录，严格执行 `VoiceAgent` 的“单目录需满 30 秒”门控，绝不静默拆分子目录绕过安全校验；同时在 `voiceprint_enroll` 参数中提供可选的 `session_ids: list[str] | None`（长度须与 `audio_base64_list` 一致，仅接受字母数字/短横线/下划线），调用方显式声明来自不同采集会话时才映射到对应子目录；
  3. **默认 `detail=False` 防泄分与默认防伪护栏（经 `/grill-me` 确认）**：MCP Server 默认 `detail=False` 且**默认开启 `SpreadSpectrumWatermarkDetector` 防伪护栏**（可通过 `--no-spoof-guard` 显式关闭），严禁透传内部得分、向量、堆栈和私钥路径；
  4. **破坏性操作护栏与并发安全（经 `/grill-me` 确认）**：
     - 内部加 `threading.RLock()` 串行保护共享 `VoiceAgent` 状态机与 SQLite 连接；
     - `voiceprint_revoke(name, confirm=False)` **严格保留两步状态机**：必须先以 `confirm=False` 触发 `撤销 <name>` 进入 `pending` 态，再以 `confirm=True` 触发 `确认撤销 <name>` 才允许删除；若未处于匹配 `pending` 态直接传 `confirm=True`，直接拒绝，杜绝 LLM 单次幻觉误删声纹库；
     - 支持 `--read-only`（`read_only=True`）只读安全模式：只读模式下拒绝 `voiceprint_enroll`、`voiceprint_revoke`、`voiceprint_rotate_key`（返回 `read_only_mode` 错误），仅开放推理与查询类工具。
- **暴露的工具清单（8 个）**：
  - `voiceprint_enroll`（参数：`name`, `consent: bool`, `audio_paths` 或 `audio_base64_list`, 可选 `session_ids`）
  - `voiceprint_verify`（参数：`name`, `audio_path` 或 `audio_base64`）
  - `voiceprint_identify`（参数：`audio_path` 或 `audio_base64`）
  - `voiceprint_list_speakers`（无参）
  - `voiceprint_revoke`（参数：`name`, `confirm: bool = False` — 强制两步状态机，单步 `confirm=True` 直接拒绝）
  - `voiceprint_rotate_key`（参数：`name`）
  - `voiceprint_check_audio_quality`（参数：`audio_path` 或 `audio_base64`）
  - `voiceprint_health`（返回 embedder 元信息、离线状态、`audio_roots_configured`、`spoof_guard_enabled`、`read_only`、`detail` 开关）
- **暴露的资源清单（2 个）**：
  - `voiceprint://speakers`（已注册说话人列表与计数，无密文/向量）
  - `voiceprint://health`（服务状态与安全策略摘要）
- **CLI 入口**：`python -m agent.mcp_server --data-dir <dir> [--audio-roots <dir1> <dir2>] [--embedder stats|vpnet|ecapa] [--detail] [--no-spoof-guard] [--read-only]`。

---

### 4.3 支柱三：AI 克隆语音防伪护栏与红队扩展（`voiceprint/spoof.py` & `voice_eval/dims/redteam.py`）

#### 4.3.1 可插拔防伪检测接口与内置扩频水印原语（`voiceprint/spoof.py`）
- `SpoofResult` dataclass：`is_spoof: bool`, `score: float`, `reason: str`, `detector: str`, `metadata: dict`。
- `SpoofDetector` 基类 / 协议：`detect(wav: np.ndarray, sr: int) -> SpoofResult`。
- **内置确定性 16-bit 扩频水印原语（模拟 AudioSeal 16-bit 来源水印契约）**：
  - `embed_spread_watermark(wav: np.ndarray, sr: int = 16000, *, payload: int = 0x5653, strength: float = 0.02, secret_seed: int = 20261121) -> np.ndarray`：
    - 将 16-bit `payload` 编码为 $\pm 1$ 符号序列，按 `secret_seed` 生成固定带限（1.5–4.5 kHz）正交伪随机扩频码片，按帧周期叠加至波形并做安全幅值裁剪；
  - `SpreadSpectrumWatermarkDetector(expected_payload: int | None = None, *, secret_seed: int = 20261121, z_threshold: float = 4.5)`：
    - 在 1.5–4.5 kHz 频带做归一化相关检测，计算同步码片相关 z-score 与解码 16-bit 载荷；当 z-score 超过 `z_threshold`（且若指定 `expected_payload` 则载荷一致）判为 `is_spoof=True, reason='watermark_detected'`；对干净自然语音和合成正弦/噪声基线 z-score 接近 0，保证近零误报。
- `CompositeSpoofDetector(detectors: list[SpoofDetector])`：任一检测器命中即返回 `is_spoof=True`，方便未来在 Windows 实机无缝挂入 AudioSeal / AASIST 包装器。

#### 4.3.2 `VoiceAgent` 防伪护栏集成（`agent/service.py`）
- `VoiceAgent.__init__(..., spoof_detector=None)`（经 `/grill-me` 确认：Python 构造默认 `None`，保持既有历史离线基准 100% 可比；面向外部 Agent 的 MCP Server 默认传入 `SpreadSpectrumWatermarkDetector()`，支持 `--no-spoof-guard` 显式关闭）。
- 当 `self.spoof_detector is not None` 时：
  - `_enroll`、`_verify`、`_identify` 在 `check_audio()` 之后、提取声纹之前解码音频并调用 `self.spoof_detector.detect(wav, SR)`；
  - 命中 `is_spoof=True` 时立即拒绝：`decision='reject'`, `reason='spoof_detected'`, `ok=False`（`detail=False` 时不泄露内部 `score` / `z_score`）；
  - **落库与锁号策略（架构裁定）**：与现有 `AudioError` 拒绝策略保持一致，仅由 `VoiceAgent.handle()` 自动写入 `store.audit` 审计表，**不写入 `store.events`** —— 既避免写入 `NULL` 分数或用水印 `z_score` 污染 `_drift()` 声纹相似度历史，也防止攻击者用 5 条带水印音频恶意锁死合法用户账号。

#### 4.3.3 合成伪造攻击与红队维度扩充（`voice_eval/audio_synth.py` & `voice_eval/dims/redteam.py`）
- `audio_synth.py` 新增：
  - `apply_watermark_spoof(wav, sr, *, payload=0x5653, strength=0.02)`：给本人正例录音嵌入 16-bit AI 来源水印（模拟高相似度克隆语音冒充）；
  - `apply_vocoder_replay_spoof(wav, sr, *, seed=0)`：确定性带限相位抖动 + 梳状共振峰伪影；**范围裁定**：本轮仅作为面向未来外挂 AASIST 等重放检测模型的确定性攻击波形原语（配独立单元测试），不内置易误杀 8 kHz 电话语音的启发式 DSP 拦截器，也不计入 Tier 1 红队红线。
- `voice_eval/dims/redteam.py` 新增 3 组红队检查（计入现有 Tier 1 零黄零红红线，且**零改动原有 6 个测试文件的全部 44 项断言**）：
  1. **容器与音频质检红队**（`container_integrity`）：
     - 在既有 `path_symlink_escape` 探针内同时验证“目录外符号链接拒绝（报允许目录外）”与“目录内符号链接拒绝（报不接受符号链接）”，仅占用 1 次 `SKIPPED_NO_PRIVILEGE` 槽位，确保既有 `test_voice_eval_agent_checks.py` 的 `res['skipped'] == 1` 断言零破坏；
     - `redteam_container_magic_mismatch`（伪装成 `.wav` 的非 WAV/MP3 字节 → 拒绝）；
     - `redteam_container_truncated_wav`（合法 RIFF 头但截断 `data` 块 → 拒绝）；
     - `redteam_quality_pure_silence`（纯静音注册与验证均拒绝）；
     - `redteam_quality_severe_clipping`（重度削波注册拒绝）；
  2. **MCP 协议边界红队**（`mcp_security`）：
     - `redteam_mcp_no_roots_blocks_path`（未配 `audio_roots` 时路径参数被拒）；
     - `redteam_mcp_path_escape_blocked`（`..` 越权路径被拒）；
     - `redteam_mcp_base64_data_uri_and_magic`（合法 `data:audio/wav;base64,...` 正常解析，伪造魔数 Base64 被拒，且 `detail=False` 零泄分）；
     - `redteam_mcp_revoke_requires_two_step_and_readonly`（单步 `confirm=True` 撤销被拒，且 `--read-only` 阻断注册/撤销/轮换）；
  3. **克隆水印防伪红队**（`anti_spoofing`）：
     - `redteam_spoof_watermarked_clone_blocked`（启用 `SpreadSpectrumWatermarkDetector` 后，嵌入 AI 水印的本人音频在 `verify` 与 `identify` 中均被拦截，且 `detail=False` 不泄露检测分值）；
     - `redteam_spoof_clean_genuine_preserved`（同配置下未加水印的干净本人音频仍正常通过，无误杀）；
  - 在 `redteam` 报告中补充 `spoof_summary: {'watermarked_clone_far': float, 'clean_genuine_frr': float}`。

---

### 4.4 支柱四：Warm RTF 性能基线与严格离线权重锁定

#### 4.4.1 冷/热分离、无缓存微基准与 Warm RTF（`voice_eval/cache.py`, `voice_eval/dims/perf.py`, `voice_eval/runner.py`, `voice_eval/gate.py`）
- **保持 `CachedEmbedder.stats()` 返回字典的 7 个键完全不变**（确保现有 `test_voice_eval_cache.py` 精确字典 `assertEqual` 零改动通过），新增独立方法 `CachedEmbedder.compute_audio_seconds() -> list[float]` 记录每次实际计算（cache miss）的音频时长（不可读或虚拟路径优雅跳过不报错）。
- **无缓存 RTF 微基准（经 `/grill-me` 确认）**：为防止二次运行 100% 命中磁盘缓存导致 `warm_rtf` 为空，新增 `measure_uncached_rtf(embedder, audio_paths: list[str], *, max_samples: int = 5)`（位于 `voice_eval/dims/perf.py`）：
  - 在 `runner.py` 的 `run_split()` 中，从本分片去重抽取前 `max_samples`（默认 5）条可读音频，直接调用底层非缓存 `embedder._embed()`（若为 `IsolatedEmbedder` 则走隔离 Worker 绕过 `CachedEmbedder` 磁盘缓存）：第 1 条计为 `cold`（冷启动预热），其余（最多 4 条）计为 `warm` 热推理，统计总耗时与总音频时长；
  - 若 `summarize_perf` 收到 `rtf_bench` 微基准结果，优先以此产出稳定可跨模型对比的 `rtf` 字段（并记录 `rtf.source = 'micro_bench'`）；未传 `rtf_bench` 时回退使用 `compute_ms_list` + `compute_audio_s_list`（`rtf.source = 'cache_miss'`）。
- `summarize_perf(..., compute_audio_s_list=None, rtf_bench=None, worker_rss_mb=None)` 保持所有既有字段完全向后兼容，并新增：
  - `cold_embed_ms`: 首次冷计算耗时（无计算时为 `None`）；
  - `warm_embed`: 去除首次冷启动后的计算耗时分位统计（仅 1 次时回退至该次并标记 `warm_fallback=True`）；
  - `rtf`: `{'source': ..., 'cold_rtf': ..., 'warm_rtf': ..., 'p50_warm_rtf': ..., 'p95_warm_rtf': ..., 'total_audio_seconds': ..., 'warm_audio_seconds': ..., 'n_samples': ...}`，其中 `warm_rtf = sum(warm_compute_s) / sum(warm_audio_s)`（`warm_audio_s == 0` 时返回 `None`）；
  - `worker_rss_mb`: 隔离 Worker 峰值常驻内存（未隔离时为 `None`）；开启隔离时 `rss_mb` 取 `max(parent_rss_mb, worker_rss_mb)`（同时保留 `parent_rss_mb`），防止模型迁入子进程后主进程 RSS 偏小导致 Tier 3 内存告警失真。
- `gate.py` Tier 3 观察告警新增 Warm RTF 退化检查（常量 `PERF_RTF_RATIO_WARN = 1.5`, `PERF_RTF_ABS_WARN = 0.05`）：当 `cand_warm_rtf > 1.5 * base_warm_rtf` 且 `cand_warm_rtf - base_warm_rtf > 0.05` 时输出中文 RTF 退化告警，并在对比 Markdown 报告中展示 `Warm RTF` 对比行。

#### 4.4.2 严格离线环境锁定与指纹审计（`voice_eval/offline.py` & `voice_eval/runner.py`）
- 新增 `voice_eval/offline.py`：
  - `strict_offline_env()` 上下文管理器：进入时强制设置 `HF_HUB_OFFLINE='1'`、`TRANSFORMERS_OFFLINE='1'`、`HF_DATASETS_OFFLINE='1'`，退出时精确恢复原环境变量；
  - `collect_model_fingerprint(embedder) -> dict`：校验 `embedder.model_id` 非空，若 `checkpoint` 指向本地文件则计算其 `SHA-256`（利用 `file_digest`），返回 `{'embedder', 'model_id', 'checkpoint', 'checkpoint_sha256', 'offline_enforced': True}`。
- `runner.py` 的 `run_split(..., offline_strict: bool = True, isolate_worker: bool = False)`：
  - 默认在 `strict_offline_env()` 内运行并在 `summary.json` 顶层写入 `runtime_audit` 字段（含 `offline_enforced`, `model_fingerprint`, `worker_isolated`, `worker_restarts`, `worker_timeouts`, `worker_circuit_open`）。

---

### 4.5 支柱五：跨平台受控子进程树与隔离评测 Worker

#### 4.5.1 跨平台受控进程树（`voiceprint/contained_subprocess.py`）
- **POSIX（macOS / Linux）**：
  - 子进程以 `start_new_session=True` 启动（成为新 process group 组长）；
  - 父进程保留匿名 control pipe 写端，父进程调用 `terminate_tree(grace_s=0.2)` 或上下文退出时，先关闭 control pipe 再向 `-pgid` 发送 `SIGTERM`，等待 `grace_s` 后对残留组员发送 `SIGKILL` 并 `wait()` 回收，彻底杜绝孙进程（grandchild）孤儿残留。
- **Windows（`win32`）**：
  - 使用标准库 `ctypes` 调用 `kernel32.CreateJobObjectW` + `SetInformationJobObject`（设置 `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`）+ `AssignProcessToJobObject`，关闭 Job 句柄或父进程异常退出时操作系统内核原子级终止整棵进程树。
- **统一 API**：
  - `ContainedProcess(cmd, *, cwd=None, env=None, stdin=PIPE, stdout=PIPE, stderr=PIPE)`（支持上下文管理器与 `terminate_tree()`）；
  - `run_contained(cmd, *, timeout: float | None = None, cwd=None, env=None, input_bytes: bytes | None = None) -> ContainedRunResult(returncode, stdout, stderr, timed_out)`。

#### 4.5.2 隔离评测 Worker（`voice_eval/isolated_worker.py`）
- 子进程入口：`python -m voice_eval.isolated_worker --embedder <stats|vpnet|ecapa|tone> [--checkpoint <path>]`，强制在 `strict_offline_env()` 内加载模型，循环读取 `stdin` 的单行 JSON 请求（`{"op": "describe"}` / `{"op": "embed", "path": "..."}` / `{"op": "shutdown"}`），响应单行 JSON（向量以 base64 float32 + shape 编码，并附带 `rss_mb: current_rss_mb()` 用于跨进程内存归集）。
- 客户端封装 `IsolatedEmbedder(embedder_name, *, checkpoint: str | Path | None = None, timeout_s: float = 30.0, max_consecutive_failures: int = 3, python_executable: str | None = None)`：
  - 继承 `Embedder` 基类并标记 `is_isolated = True`，暴露 `peak_worker_rss_mb`、`restart_count`、`timeout_count`、`circuit_open`，与 `CachedEmbedder` 和 `VoiceAgent` 无缝兼容；
  - **实例归属、故障自愈与连续失败熔断（经 `/grill-me` 确认）**：
    1. `voice_eval/__main__.py` 在指定 `--isolate-worker` 时，直接在 `strict_offline_env()` 内实例化 `IsolatedEmbedder(args.embedder, timeout_s=args.worker_timeout, max_consecutive_failures=args.max_worker_failures)`（跳过主进程 `load_embedder()`，并在 `finally` 中 `close()`）；若 Python 调用方直接传普通 `Embedder` 给 `run_split(..., isolate_worker=True)`，则由 `run_split` 按 `(embedder.name, embedder.checkpoint)` 包装成 `IsolatedEmbedder` 并在结束时回收；
    2. **单次异常自动重启 + 连续失败熔断**：某次 `embed` 请求超时或子进程崩溃时，立即 `terminate_tree()` 清理整棵进程树、重置内部句柄、递增 `restart_count` / `timeout_count` 与 `consecutive_failures`；若 `consecutive_failures < max_consecutive_failures`（默认 3），抛出 `IsolatedWorkerError`（该条坏用例在 `VoiceAgent.handle()` 中诚实记为 `tool='error'` / `missing`），下一次 `embed_file()` 自动拉起全新子进程继续跑（成功一次即清零 `consecutive_failures`）；若连续失败达到 `max_consecutive_failures`，置 `circuit_open = True` 并抛出 `WorkerCircuitOpenError`，同时 `runner.py` 在每条用例返回后检查 `circuit_open` 标志立即向上抛出 `WorkerCircuitOpenError` 中止整次评测（不耦合 `agent/service.py` 异常处理，也不在 1,190 条用例上死等数小时）。
- `voice_eval/__main__.py` 新增 `run --isolate-worker`、`--worker-timeout` 与 `--max-worker-failures` 参数（默认不开启，按需隔离不可信或实验性模型）。

---

## 5. 留给实现者决定的细节（Left to the Builder）

- 具体的单元测试文件拆分与辅助 fixture 组织方式（建议按功能拆分为 `test_audio_quality_and_container.py`、`test_mcp_server.py`、`test_spoof_and_redteam.py`、`test_perf_rtf_and_offline.py`、`test_contained_subprocess_and_worker.py`，同时确保原有 6 个测试文件 44 项测试零回归）。
- `protocol.json` 中是否补充 `tier3_observational` 的 `perf_rtf_ratio_warn` / `perf_rtf_abs_warn` 字段（应同步补充并与 `gate.py` 常量保持单一事实源）。
- MCP `stdout` 日志隔离：确保 `agent/mcp_server.py` 运行时任何内部日志只输出到 `stderr`，严禁污染 JSON-RPC `stdout` 协议流。

---

## 6. 验收标准

1. **零新增依赖**：`requirements.txt` 保持 5 个包不变，不要求联网下载任何模型或权重。
2. **全量回归 `-W error` 零警告通过**：原有 44 项测试 + 本次 5 大支柱新增单元/红队/E2E 回归测试在 `.venv/bin/python -W error -m unittest discover -s tests` 下 100% 通过。
3. **跨平台真实验证**：在 Mac 沙箱中真实验证 POSIX 进程树回收（含故意 spawn 孙进程并挂起的超时用例，确认超时后孙进程 PID 已消失）、MCP 完整 JSON-RPC 管道通信、WAV 截断/魔数/symlink/削波拦截、扩频水印防伪拦截与干净语音零误杀、Warm RTF 与离线锁审计；Windows Job Object 路径保持语法与静态结构完备。
