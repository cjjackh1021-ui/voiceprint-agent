# 可控声音模拟、强制水印溯源与攻防闭环评测 Agent 设计文档（路线 A）

- **日期**：2026-11-23
- **目标仓库**：`voiceprint-agent`
- **状态**：定稿待审批（已确认采用选项 1：`F5-TTS` 主打文本克隆 + 预留 `CosyVoice2` 插槽）
- **目标运行环境**：核心编排与默认确定性模拟后端全平台（Windows / macOS）零重依赖原生运行；可选真实神经克隆后端面向用户 Windows 显卡机（`trainbox`：NVIDIA RTX 4070 SUPER 12 GB，Python 3.12，`D:\voice confirm` 目录内已完备 `LibriSpeech`、`ECAPA`/`VPNet` 权重与 `cohorts`），并兼容跨机隧道调用。

---

## 1. 背景、目标、非目标与反目标

### 1.1 背景与动机
`voiceprint-agent` 已具备说话人注册、ZT-Norm + 自助法置信区间验证/辨认、双人混音分离、标准库 JSON-RPC 2.0 MCP Server、100 ms 音频质检、0.5 s 16-bit 频域扩频水印防伪及七维离线评测门禁体系，但核心交互链路仍停留在**单向识别与被动防御**，原有 `匿名` / `变声` 仅为空置拒答存根。

若要补齐项目的**生成能力与交互深度**并严格契合 **Agent 评测与架构**定位，不能只做“文本转克隆语音”的单步调包脚本，而应建设为一条**授权校验 → 可控声音模拟 → 强制水印溯源 → 声纹/质检 Self-Refine 重试 → 闭环回流验证与攻防评测**的双向安全 Agent 系统。

### 1.2 核心目标（Goals）
1. **受控生成 + 自我纠错（Self-Refine）Agent 闭环**：新增 `模拟` 指令与 `voiceprint_simulate` MCP 工具，实现“授权核验 → 合成 → 受信任内核强制嵌水印并验签 → 100 ms 质检 + 目标声纹自检 → 参数自适应调优重试”的多步状态机。
2. **双后端可插拔合成架构（轻量默认 + Windows GPU 神经插件）**：
   - **默认零依赖后端（`AcousticFormantSynthesizer`）**：仅用现有 `numpy` / `scipy` / `soundfile` 实现说话人基频与共振峰包络迁移（`源音频` 模式）以及可控韵律声学合成（`文本` 模式），确保离线 CI、无卡 Mac 与全部回归测试毫秒/秒级 100% 确定性复现。
   - **可选真实神经克隆后端（`SubprocessNeuralSynthesizer`）**：复用 `voiceprint/contained_subprocess.py` 的跨平台进程树隔离（Windows Job Object / POSIX PGID）对接运行于 Windows GPU 独立环境中的开源 Zero-Shot TTS/VC（首选 **F5-TTS**，兼容 **CosyVoice2**），主仓 `requirements.txt` **保持零新增重依赖**。
3. **信任边界与不可绕过溯源水印**：16-bit 扩频水印嵌入与回读验签**严格保留在主仓受信任内核层执行**，禁止任何外部合成插件自行决定是否加水印；模拟产物可安全回灌至现有 `验证` / `辨认` 流程验证防伪拦截链路。
4. **攻防闭环评测（`sim-bench`）与红队扩张**：在 `voice_eval/` 下新增去水印攻击算子、6 项仿真安全红队探针以及独立的生成–鉴伪闭环基准 `python -m voice_eval sim-bench`。

### 1.3 非目标（Non-Goals）
- 不将数 GB 的开源 TTS 模型、声码器或权重打包进主仓 `requirements.txt` 或 Git 历史。
- 不修改现有 73 项回归测试，不改变 `voice_eval/protocol.json` 既有 7 维发布门禁结构与采纳阈值公式。
- 不删除或改变现有 `匿名` / `变声` 拒答存根行为（保留向后兼容与原有 `anonymize_boundary` 测试）。
- 不承诺零依赖的纯声学迁移具备“欺骗人类听觉”的拟真自然度——自然拟真语音由 Windows GPU 上的可选神经克隆插件承担。

### 1.4 反目标（Anti-Goals / Failure Modes to Prevent）
- **绝对禁止“无授权声纹冒充”**：未注册目标、缺乏显式授权、或参考音频与目标声纹不匹配时必须拦截，禁止拿着张三的注册名+李四的公开录音偷梁换柱合成。
- **绝对禁止“递归洗白水印”**：已携带系统水印的合成语音不得再次作为 `参考音频` 进行二次模拟。
- **绝对禁止“自评闭环误锁用户 / 泄露精确分数”**：Agent 内部 Self-Refine 重试自评绝不能写入用户正常验证的失败计数（以免误触发 5 次暴力破解锁定）；且在 `detail=False` 模式下，返回载荷不得暴露被测模型精确相似度与阈值。
- **绝对禁止 Windows GBK / 子进程孤儿泄漏**：在 Windows 运行中文文本神经克隆子进程时，必须强制 UTF-8 编码与 Job Object 强杀清理，不得出现编码崩溃或 GPU 进程残留。

---

## 2. 备选方案对比与选型结论

| 方案 | 核心思路 | 优点 | 缺点与风险 | 结论 |
| :--- | :--- | :--- | :--- | :--- |
| **路线 A（已选）** | 可控模拟 + 强制水印溯源 + 声纹自检重试 Agent + 双后端隔离（默认声学迁移 + Windows GPU 神经插件）+ 闭环评测 | 兼顾 Agent 架构深度（Self-Refine、状态机、MCP、护栏）、攻防闭环评测、零依赖可复现性与 Windows 真实克隆 Demo 拓展能力 | 需跨越合成接口、Agent 编排、MCP 工具与 `voice_eval` 四个模块协同改造 | **采用** |
| **路线 B** | 重度耦合集成端到端开源 TTS/VC（如直接 `pip install f5-tts` 入主仓） | 可听 Demo 效果最直接 | 依赖极其沉重，Mac/CI 难以零门槛复现，且缺少自主 Agent 状态机与评测创新 | 放弃 |
| **路线 C** | 纯信号/对抗攻击红队引擎，不做可控文本/音色模拟 | 完全轻量、安全零风险 | 缺少面向用户文本与音色的“声音模拟”真实生成流程，演示直观性弱 | 作为路线 A 的评测子集吸收 |

---

## 3. 系统总体架构与数据流

```text
CLI ("模拟 姓名 本人授权 ...") / MCP (voiceprint_simulate)
   │
   ▼
[前置安全门禁 Gate 1..5]
   ├─ 1. 合成器就绪 & 显式授权校验 (本人授权 / consent is True)
   ├─ 2. 目标已注册且未锁定 (store.exists + model match + lock check)
   ├─ 3. 参考音频合规 + 100ms 质检 + 防伪校验 (拒绝已含水印音频防递归克隆)
   ├─ 4. 参考音频归属权核验 (verify decision == accept，不计入失败锁定计数)
   └─ 5. 输入负载校验 (源音频经 check_audio/质检 或 文本 <= 500 字符非空)
   │
   ▼
[Self-Refine 闭环执行引擎 (1 .. max_sim_attempts 轮)]
   │  ┌──────────────────────────────────────────────────────────┐
   │  ▼                                                          │ (未达标且未超上限:
   │  Synthesizer.synthesize(SimulationRequest)                  │  削波降增益 / SIM偏低
   │  (默认 AcousticFormantSynthesizer 或 Windows Subprocess 插件) │  调高 formant_blend /
   │  │                                                          │  pitch_pull 等参数)
   │  ▼                                                          │
   │  [受信任内核强制处理]                                         │
   │  embed_spread_watermark(wav, 16k) → 回读 SpreadSpectrumDetector 验签
   │  │                                                          │
   │  ▼                                                          │
   │  [自检评估 Evaluator]                                        │
   │  analyze_audio_quality(cand) + _score(name, probe, 'verify') │
   │  └─ 若 质量合格 && 水印有效 && SIM 达标(或取最优收束) ───────────┘
   ▼
落盘受控目录 data_dir/simulated/sim_<uuid>.wav + store.audit(...)
(此目录已纳入 audio_roots，支持立刻调用 "验证"/"辨认" 验证闭环拦截效果)
```

---

## 4. 模块详细设计

### 4.1 模块一：`voiceprint/synth.py`（合成接口契约与双后端实现）

#### 4.1.1 核心数据结构与抽象基类

```python
@dataclass(frozen=True)
class SimulationRequest:
    target_name: str
    mode: Literal['audio', 'text']
    reference_audio_path: str
    reference_text: str | None = None  # [新增] 可选参考音频文本；供 F5-TTS 等 Zero-Shot TTS 在离线模式跳过 Whisper ASR
    source_audio_path: str | None = None
    text: str | None = None
    target_sr: int = 16000
    formant_blend: float = 0.65   # 声道包络迁移强度 [0.0, 1.0]
    pitch_pull: float = 0.70      # 基频向参考说话人靠拢系数 [0.0, 1.0]
    gain_scale: float = 0.85      # 输出幅度安全缩放因子 (0.1, 1.0]
    seed: int = 0
    attempt_index: int = 1        # Self-Refine 轮次索引

@dataclass(frozen=True)
class SimulationResult:
    wav: np.ndarray               # 1D contiguous float32, in [-1.0, 1.0]
    sr: int                       # 固定校验为 16000 Hz
    backend: str                  # e.g., 'acoustic_formant' | 'neural_subprocess'
    metadata: dict[str, Any]      # 诊断耗时、基频估计、实际超参等（不含明文声纹）

class VoiceSynthesizer:
    name: str = 'base'
    supported_modes: tuple[str, ...] = ('audio', 'text')  # [新增] 能力声明，前置门禁据此提前拦截不支持的模式
    def synthesize(self, req: SimulationRequest) -> SimulationResult: ...
    def describe(self) -> dict[str, Any]: ...
```

**输出契约校验**：主仓核心对所有后端返回的 `SimulationResult` 强制执行规范化防御：校验 `wav` 为有限数值（无 `NaN`/`Inf`）、重采样至 16 kHz 单声道 `float32`、时长不少于 `MIN_SECONDS`（0.5 s）且不超过 60.0 s；不满足直接抛出 `SynthesisError` 并终止当前尝试。

#### 4.1.2 默认零依赖后端：`AcousticFormantSynthesizer`
仅依赖 `numpy`、`scipy`、`soundfile`，声明 `supported_modes = ('audio', 'text')`。注意特征层存在重要声学边界：**`VPNetEmbedder`（`voiceprint/features.py` `fbank_batch`）与 `ECAPA-TDNN` 均执行句级对数 Mel 均值减法（Utterance CMVN）**，纯静态长期滤波增益会被减法部分抵消；因此本后端采用**时变激励谐波迁移 + 动态音节包络调制**以产生可观测的音色/基频收敛行为，同时明确定位：
1. **参考声学画像提取**：从 `reference_audio_path` 抽取基频中位数 $F_0^{\text{ref}}$（自相关/倒谱基频估计算法，限定 70–350 Hz）及动态分帧共振峰包络分布。
2. **模式 A（`mode == 'audio'`，源音频迁移）**：
   - 将 `source_audio_path` 按短时帧切分（帧长 25 ms，帧移 10 ms）；
   - 按音节活跃帧逐帧迁移基频（`pitch_ratio = (F0_ref / F0_src) ** pitch_pull`）与谐波包络，保留源语音时间节奏的同时，使谐波间距与高频动态分布向参考说话人收敛。
3. **模式 B（`mode == 'text'`，文本驱动确定性韵律合成）**：
   - 依据 `text` 的 UTF-8 字节与标点结构确定性切分音节活跃帧、停顿静音帧与基频微扰（确保句内存在清晰浊音–清音–停顿对比，避免静态平稳音被 CMVN 抹平）；
   - 以参考说话人基频 $F_0^{\text{ref}}$ 为基频生成脉冲/噪声混合激励源并经参考频谱包络成形、`gain_scale` 软限幅；
   - **关键定位（设计者预设 / My Call）**：`AcousticFormantSynthesizer` 的首要目标是保证零权重环境下全链路闭环（水印、质检、Self-Refine `ΔSIM` 梯度上升与受控拒答防线）100% 确定性复现；在 `ToneEmbedder` 沙箱下可稳定收敛至 `'accept'`，而在真实深度声纹模型（`VPNet`/`ECAPA`）下因非真实人声发音机制仅展示 `Cosine SIM` 逐轮单调抬升、安全门槛拒放未达标伪音（真正要在 `VPNet`/`ECAPA` 下收敛拟真克隆，由 Windows GPU 神经后端完成）。

#### 4.1.3 Windows GPU 可选神经克隆插件：`SubprocessNeuralSynthesizer`
为在不污染主仓轻量依赖的前提下，让用户直接在 **Windows 显卡机**上接入真实中英 Zero-Shot 语音克隆，设计安全隔离的外部子进程协议：
1. **进程隔离与 Windows 兼容性**：
   - 调用 `voiceprint/contained_subprocess.py` 的 `run_contained(cmd, timeout=self.timeout_s, env=env, input_bytes=payload_bytes)`；
   - 在继承或构造的 `env` 中**强制注入**：`PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`（防止 Windows 中文文本输入/JSON 日志走 GBK 报错）；并依策略默认透传 `HF_HUB_OFFLINE` 与本地缓存目录环境变量；
   - 超时后 `ContainedProcess` 自动通过 **Windows Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`)** 彻底销毁 CUDA 子进程与衍生的 Worker，防止显存泄漏。
2. **标准文件 + JSON 协议 & 能力声明**：
   - `SubprocessNeuralSynthesizer` 支持传入 `supported_modes`（默认 `('text',)` 匹配主打引擎 **F5-TTS**；若后续切换或接入支持 VC 的引擎如 CosyVoice2 可显式配置为 `('audio', 'text')`）；
   - 宿主机在每次调用的 `TemporaryDirectory(prefix='vpsyn_')` 中写入 `request.json`（含 `mode`、`reference_audio_path`、可选 `reference_text`、`source_audio_path`、`text`、调优参数及宿主机分配的输出路径 `output_wav_path`）；
   - 调用参数以安全 `argv` 列表传入 `run_contained`，必须保留带空格的 Windows 路径（例如 `["D:/voice confirm/.venv_tts/Scripts/python.exe", "plugins/neural_tts_worker.py", "--engine", "f5_tts", "--request", "<tmp>/request.json"]`，禁止未加引号的字符串拼接以防 `D:\voice confirm` 目录含空格拆分错误）；
   - 规定 Worker **只能写指定临时路径下的 WAV**、stdout 仅打印单行 JSON 状态 `{"ok": true, "model": "f5_tts", "device": "cuda:0", "elapsed_ms": ...}`；宿主机校验返回文件合法性后读取波形并在主仓内核打入溯源水印，**立即销毁 `request` 临时目录**。
3. **独立插件脚本 `plugins/neural_tts_worker.py`（与主包运行完全隔离）**：
   - 主仓库测试与 `voice_eval` 绝不默认 `import f5_tts` 或 `cosyvoice`；
   - 默认主推 **`--engine f5_tts`**（`supported_modes=('text',)`），针对 12 GB RTX 4070 SUPER 优化单次轻量推理；同时预留 **`--engine cosyvoice2`** 扩展分支（`supported_modes=('audio', 'text')`）；当请求附带 `reference_text` 时优先透传给 F5-TTS 以在纯离线模式下跳过 Whisper ASR 转写；
   - 预留可选扩展说明：若后续需要从 Mac 通过 Tailscale / SSH 端口转发访问 Windows 上的常驻推理服务，只需实现同接口 `VoiceSynthesizer` 子类转发至受白名单保护的本地回环端口即可，上层 `VoiceAgent` 与闭环评测代码**零改动**。

---

### 4.2 模块二：`agent/service.py` 与 `agent/cli.py`（受控指令与 Self-Refine 闭环）

#### 4.2.1 `VoiceAgent` 构造扩展与目录隔离
```python
class VoiceAgent:
    def __init__(
        self,
        data_dir: str | Path,
        embedder,
        detail: bool = False,
        audio_roots: list[str | Path] | None = None,
        spoof_detector=None,
        synthesizer: VoiceSynthesizer | None = None,
        max_sim_attempts: int = 3,
    ):
```
- **默认值兼容决策（设计者预设 / My Call）**：
  - `VoiceAgent.__init__(..., synthesizer=_UNSET)` 使用哨兵值：直接实例化 `VoiceAgent`（如 CLI 与 Python API）时**默认自动挂载零依赖 `AcousticFormantSynthesizer()`**（除非显式传 `synthesizer=None` 或 CLI 传 `--synthesizer none`），既保证全交互环境开箱即用，又不破坏任何现有 73 项测试；
  - 而在 `agent/mcp_server.py` 的 `build_default_server(..., synthesizer: VoiceSynthesizer | None = None)` 工厂函数中**默认 `synthesizer=None`**，且 **`tools/list` 根据能力动态过滤**：当 `agent.synthesizer is None` 时自动隐藏 `voiceprint_simulate`（恰好维持原 8 个工具集合，**保证 `tests/test_mcp_server.py:55-66` 零修改 100% 通过**）；而 `mcp_server.main()` CLI 入口则将 `--synthesizer` 默认设为 `'acoustic'` 并主动传入合成器实例，使真实 MCP 服务启动后直接暴露全部 9 个工具。
- **受控输出目录**：
  - 新增 `self.simulated_dir = self.data_dir / 'simulated'`。
  - 当 `audio_roots` 非空时，`self.audio_roots` 初始化为 `[Path(root) for root in audio_roots] + [self.separated_dir, self.simulated_dir]`，保证生成的模拟录音可直接被同一 Agent 的 `验证` / `辨认` / `质检` 工具安全读取验证闭环防伪效果。
- **内部溯源检测基座（不可旁路）**：
  - 新增 `self._watermark_guard = SpreadSpectrumWatermarkDetector()`（固定使用系统默认秘密种子与 16-bit payload `0x5653`）。即使 `VoiceAgent` 的普通识别入口将 `spoof_detector=None`，`模拟` 流程仍强制执行：
    1. 检查输入 `reference_audio` 是否已含溯源水印（防止递归二次克隆已仿真音频）；
    2. 对合成产出嵌入 16-bit 水印并立即回读验签确保载荷完整性；
  - 对现有 `验证` / `辨认` / `注册` 流程保持原有 `self.spoof_detector` 行为不变（CLI 默认可通过 `--spoof-guard` 控制，MCP Server 保持默认开启 `spoof_guard=True`）。

#### 4.2.2 中文命令语法与解析规则
保持已有 `匿名`、`变声` 拒答逻辑原封不动，在 `_dispatch` 的 `commands` 字典中注册：
```text
'模拟': self._simulate
```
支持两种明确模式（用 `shlex.split` 解析，支持含空格与双引号路径），并支持可选尾缀参数 `参考文本 "<文本>"`：
1. **音频参考迁移模式**：
   `模拟 <姓名> 本人授权 "<参考录音路径>" 源音频 "<源音频路径>"`
2. **文本驱动克隆模式**：
   `模拟 <姓名> 本人授权 "<参考录音路径>" 文本 "<待合成文本>" [参考文本 "<参考录音对应文本>"]`

补充参数限制：
- `文本` 非空、去除首尾空白后长度 `1 <= len(text) <= 500`；可选 `参考文本` 长度同样上限 500；
- 在 `HELP` 文本末尾增补 `模拟` 用法提示（现有 73 项测试未对 `HELP` 全文做精确等值断言，安全兼容）。

#### 4.2.3 五道前置安全门禁（任一不通过即干净拒答并写入审计，不调用合成器）
1. **合成器启用、模式支持与参数结构校验**：未启用合成器、`req.mode not in synthesizer.supported_modes`（如 F5-TTS 拒绝 `源音频`）或参数格式不符时返回 `tool='simulate', ok=False`。
2. **显式授权校验**：第 2 个词元必须严格为 `本人授权`（或 MCP 传入严格布尔 `consent is True`），否则以 `reason='consent_required'` 拒答。
3. **目标模板存在性与锁定校验**：
   - `self.store.exists(name)` 必须为真，且能在当前 `self.model` 下加载模板（否则拒答提示未注册或模型不匹配）；
   - 若目标当前处于连续验证失败锁定期（`self.store.recent_failures(name, LOCK_MINUTES) >= MAX_FAILED_VERIFY`），同时禁止模拟（`reason='locked'`）。
4. **参考音频合规、质检与反递归防伪校验**：
   - 经 `self._audio(ref_path)` 校验路径白名单、扩展名、魔数、RIFF/WAVE 完整性、非符号链接、非网络路径、时长 `0.5s <= d <= 1800s`、无 NaN/Inf；
   - 经 `analyze_audio_quality(ref_path)` 检查有效语音与严重问题（拒绝 `empty` / `silence` / `clipping` / `invalid`）；
   - 经 `_check_spoof` 策略校验（强制跑 `self._watermark_guard` 以及若存在 `self.spoof_detector` 则同时执行）——**如果参考音频已含水印或被判定为伪造语音，强制以 `reason='watermark_detected'` 拒绝，防止多轮模拟“套娃洗白”**。
5. **参考音频归属权声纹核验（防跨人冒充）**：
   - 对 `ref_path` 提取 embedding，调用 `self._score(name, ref_emb, task='verify')`：
   - 只有 `decision == 'accept'` 才确认该参考音确实属于被授权的目标说话人 `name`；若未通过，以 `reason='reference_speaker_mismatch'` 拒答。
   - **不写入失败锁定配额**：上述归属校验与后续内部自评**不调用 `self.store.event(name, 'verify', ...)`**，仅记入 `self.store.audit('simulate', ...)`，避免恶意调试或内部调参误触发用户 5 次验证锁定。

#### 4.2.4 多步 Self-Refine 自检重试状态机
通过前置门禁后，进入最多 `K = clamp(max_sim_attempts, 1, 5)` 轮闭环（默认 `K = 3`）：

- **第 1 轮初始超参**：`formant_blend = 0.65`, `pitch_pull = 0.70`, `gain_scale = 0.85`, `wm_strength = 0.02`。
- **单轮执行步**：
  1. **合成**：调用 `synthesizer.synthesize(req)` 得到 16 kHz `wav`；若抛出 `SynthesisError` / 超时，记录本轮失败原因并按失败分支决定是否降载重试（超时直接终止不浪费算力）。
  2. **强制水印与回读验签（受信任内核）**：
     - 调用 `embed_spread_watermark(wav, 16000, payload=DEFAULT_WATERMARK_PAYLOAD, strength=wm_strength)`；
     - 在内存中或临时文件立刻调用 `self._watermark_guard.detect(watermarked_wav, 16000)` 核验是否成功读回 `is_spoof=True` 且 `payload == DEFAULT_WATERMARK_PAYLOAD`；若未读回，视为溯源失效，尝试温和提高 `wm_strength`（上限 `0.035`）后进入下一轮（绝不允许未通过水印验签的音频落盘发布！）。
  3. **质量与目标声纹自检**：
     - 写入当前候选临时 WAV，运行 `analyze_audio_quality(cand_path)` 检查是否存在 `empty` / `silence` / `invalid` / 严重 `clipping`（`clipping_ratio > 0.05` 或存在 clipping warning）；
     - 提取候选声纹调用 `self._score(name, cand_emb, task='verify')` 获得内部评测指标 `raw_score`、`score`、`threshold`、`decision`。
  4. **收敛判定与自适应参数调整**：
     - **达标条件（Converged）**：`水印验签通过` AND `无严重质检缺陷(empty/silence/clipping/invalid)` AND `decision == 'accept'`；一旦达标立刻提前退出循环！
     - **未达标重试调整策略**：
       - 若存在 `clipping` 警告：`gain_scale = max(0.35, gain_scale * 0.70)`；
       - 若 `silence`/`too_quiet` 警告：`gain_scale = min(0.95, gain_scale * 1.15)`；
       - 若质量合格但 `decision != 'accept'`（相似度不足或落入 `uncertain` 灰区）：提升 `formant_blend = min(0.95, formant_blend + 0.15)`、提升 `pitch_pull = min(1.0, pitch_pull + 0.15)`；
       - 若水印验签 `z_score` 低于阈值：`wm_strength = min(0.035, wm_strength + 0.008)`。
- **最终产物与返回边界**：
  - 记录所有同时满足 **`水印验签通过 && 无 empty/silence/invalid 致命缺陷`** 轮次中 `(score - threshold)` 最优的候选；
  - 若至少有一个有效水印且质量达标候选，并且最终达标（或配置允许最优退化兜底——**设计明确：CLI/MCP 严格要求至少满足质量与水印底线，若完全未达声纹门槛则明确标记 `converged: False` 并由策略决定；本文规定：在质量与水印底线全部通过的前提下，若所有轮次声纹 `decision` 仍未达到 `'accept'`，则返回 `ok=False`, `reason='similarity_not_converged'`, `converged=False`，不向 `simulated_dir` 释放未达标模拟文件，确保 Agent 自我门禁的真实有效性！** 同时在内部评测基准中完整记录每轮 `score` 变化以统计收敛率与 $\Delta\text{SIM}$ 提升）；
  - 当收敛成功时，原子写入 `self.simulated_dir / f'sim_{name_safe}_{uuid4().hex[:8]}.wav'`（文件名清理控制字符与路径分隔符，杜绝目录穿越）。
  - **防泄分约束（`detail=False`）**：
    - 当 `self.detail=True`：可在 `result` 中返回 `attempt_history`（每轮 `raw_score`、`score`、`threshold`、`watermark_z_score`、`quality_ok`）与 `reply` 摘要；
    - 当 `self.detail=False`：`result` 严禁包含 `score`、`raw_score`、`threshold`、`theta_deg`、`low`、`high`、`gap` 等敏感字段，仅保留 `{'output_path': ..., 'mode': ..., 'backend': ..., 'watermarked': True, 'watermark_payload': '0x5653', 'attempts': ..., 'converged': True}`，且 `reply` 文本严禁出现“归一化相似度”“原始余弦”“验证阈值”等敏感词，兼容 `voice_eval/dims/response.py` 防泄分审计规则。

---

### 4.3 模块三：`agent/mcp_server.py`（MCP 协议暴露与安全边界）

1. **新增工具 `voiceprint_simulate`（按 `agent.synthesizer is not None` 动态在 `tools/list` 暴露）**：
   - 纳入 `_TOOL_SCHEMAS` 并列入 `_WRITE_TOOLS`（受 `--read-only` 模式严格拦截；当 `agent.synthesizer is None` 时 `tools/list` 不列出该工具，且 `tools/call` 明确返回 `error='synthesizer_disabled'`）；
   - 输入 Schema 参数：
     - `name`: string（必选，已注册说话人）
     - `consent`: boolean（必选，必须严格为 JSON boolean `true`，拦截字符串 `"true"` 或整数 `1`）
     - 参考音频（二选一互斥）：`reference_audio_path` | `reference_audio_base64`
     - `reference_text`: string（可选，参考音频文本，供 Zero-Shot TTS 离线跳过 ASR）
     - 源输入（三类合法组合之一：仅 `source_audio_path`、仅 `source_audio_base64`、或仅 `text`）
     - `max_attempts`: integer（可选，严格校验为 `int` 且非 `bool`，范围 `1..5`）
     - `include_audio_base64`: boolean（可选，默认 `false`；为 `true` 且成功输出时附带编码后的带水印 WAV base64 字符串以便无共享磁盘或零路径安全模式的 MCP Client 试听，单文件受 `MAX_BASE64_BYTES` 上限控制）
   - **安全禁止**：不接收任何调用方指定的 `output_path` 参数，落盘文件名和目录全权由 `VoiceAgent.simulated_dir` 控制。
2. **目录边界与两种 MCP 运行模式（保持既有红线不变，设计者预设 / My Call）**：
   - **已配置 `--audio-roots` 的本地路径模式**：`VoiceprintMCPServer._ephemeral_roots` 更新为 `base = list(self.audio_roots) + [self.agent.separated_dir, self.agent.simulated_dir] + list(extra_dirs)`，使本地 MCP 客户端调用 `voiceprint_simulate` 后可将返回的 `output_path` 直接传入 `voiceprint_verify` 验证闭环拦截效果；
   - **未配置 `--audio-roots`（`audio_roots=None`）的纯 Base64 隔离模式**：严格维持 `agent/mcp_server.py` 与 `tests/test_mcp_server.py:95` 既有红线——**继续拒绝一切 `audio_path` 参数**（`path_not_allowed`），客户端须通过 `include_audio_base64=True` 获取返回音频字节流再传入 `voiceprint_verify(audio_base64=...)` 完成闭环验证，不做高风险路径特判。
3. **健康检查字段扩展**：
   - `_health_dict()` 增加 `'synthesizer': self.agent.synthesizer.describe() if self.agent.synthesizer else None`，以及 `'simulation_enabled': self.agent.synthesizer is not None`。
4. **CLI/MCP 启动参数向后兼容扩展**：
   - `agent/cli.py` 与 `agent/mcp_server.py` 新增可选参数：
     - `--synthesizer`：`choices=['acoustic', 'neural_subprocess', 'none']`，默认值 `'acoustic'`；
     - `--neural-cmd`：可选外部神经 Worker 命令字符串（使用保留引号边界且兼容 Windows 反斜杠的路径切分方式或传入独立可执行/脚本路径，仅 `--synthesizer neural_subprocess` 时必填，如 `'"D:/voice confirm/.venv_tts/Scripts/python.exe" plugins/neural_tts_worker.py --engine f5_tts'`）；
     - `--neural-timeout`：默认 `120.0` 秒；
     - `--max-sim-attempts`：默认 `3`。

---

### 4.4 模块四：`voice_eval/`（攻防闭环评测、去水印攻击与红队扩张）

#### 4.4.1 `voice_eval/audio_synth.py` 新增确定性去水印/降质攻击算子
在现有 `add_white_noise`、`clip_duration`、`apply_telephone_channel` 基础上新增以下轻量 DSP 攻击函数：
1. `apply_lowpass_attack(wav, sr, cutoff_hz=3600.0)`：零相位 Butterworth 低通滤波，抹除高频段以测试频域扩频水印（1.5–7.5 kHz）在窄带截断下的留存边界；
2. `apply_resample_attack(wav, sr, mid_sr=8000)`：16 kHz → 8 kHz → 16 kHz 重采样往返劣化；
3. `apply_gain_clip_attack(wav, sr, drive=1.8)`：过驱动放大 + 硬/软削波非线性失真攻击。

#### 4.4.2 `voice_eval/dims/redteam.py` 新增 6 项仿真安全红队探针
在 `evaluate_redteam()` 内部追加“Voice Simulation Security”探针分组（沿用 `GREEN / YELLOW / RED` 分级与 `_has_null_event_scores` 零空值检查，原有红队项全部保留且零回归）：
1. `sim-missing-consent`：不带 `本人授权` 调用 `模拟`，必须返回 `ok=False`、`tool='simulate'` 且零文件写入 `simulated_dir`；
2. `sim-unregistered-target`：试图模拟未注册说话人，必须拒绝；
3. `sim-cross-speaker-impostor`：目标写已注册 `甲`，但 `参考录音` 喂入 `乙` 的真实录音，必须因归属权核验不通过被拒绝（且不得消耗 `甲` 的验证失败锁定期次数）；
4. `sim-recursive-watermark-clone`：将系统刚产出（或带 16-bit 水印）的音频再作为 `参考录音` 企图二次克隆洗白，必须因 `watermark_detected` 拦截；
5. `sim-loopback-spoof-blocked`：将合法 `模拟` 产出的带水印音频直接喂回开启防伪的 `验证 甲 "<sim_wav>"` 与 `辨认 "<sim_wav>"`，必须被识别为携带溯源水印并拒绝通行（验证闭环防伪率 100%）；
6. `sim-mcp-guardrails`：验证 MCP Server 在 `--read-only` 下拦截 `voiceprint_simulate`、拒绝非布尔 `consent="true"`、拒绝无 `audio_roots` 时的 `reference_audio_path` 越权路径，以及验证 `detail=False` 时模拟响应无分数/阈值字段泄露。

#### 4.4.3 `voice_eval/dims/simulation.py` 与 `python -m voice_eval sim-bench`
在不修改现有 `protocol.json` 7 维门禁结构的前提下，新增独立评测模块与 CLI 子命令 `python -m voice_eval sim-bench [--embedder tone|stats|vpnet|ecapa] [--synthesizer acoustic|neural_subprocess] [--neural-cmd ...] [--max-attempts 3]`（`tone` 为零数据集默认沙箱选项，其余后端复用既有 `data/LibriSpeech` 与 `cohorts/` 数据规范）：
1. **双轨评测与数据集分工（消除未收敛门禁统计冲突，设计者预设 / My Call）**：
   - **`--embedder tone`（Mac/CI 无数据集默认跑通）**：基于 `sandbox_cohort` + `generate_speaker_wav` 构建多说话人 `源音频` 迁移与 `文本` 合成闭环用例；
   - **`--embedder stats|vpnet|ecapa`（真实声纹验证 / Windows GPU 实测）**：按既有 `voice_eval run` 规则加载 `data/LibriSpeech` 与负样本 Cohort（缺失时给出清晰数据准备提示并快速失败）；若 `synthesizer` 仅声明 `supported_modes=('text',)`（如 F5-TTS），`sim-bench` 自动将 `mode == 'audio'` 用例记为 `SKIPPED_UNSUPPORTED_MODE` 并仅对受支持模式计算收敛统计；
   - **双轨解耦度量**（生产门禁保持严格不放水，评测统计保持完整可观测）：
     - **Track A（Agent 闭环自检与发布门禁）**：真实执行 `VoiceAgent._simulate(..., detail=True)` 流程，统计第 1 轮 vs 最佳/最终轮次归一化得分与原始余弦增益（`ΔCosine SIM`、`ΔNormScore`）、**严格按照声纹 `'accept'` 门禁统计 `Convergence Rate`、平均尝试步数（`Avg Attempts`）与未达标安全拦停率**（未收敛样本绝不释放入 `simulated_dir`）。
     - **Track B（水印溯源与去水印抗攻击矩阵）**：优先采集 Track A 成功释放的带水印输出；对于因深层声纹阈值未达到 `'accept'` 而被生产门禁拦停的样本（例如 `acoustic` 搭配 `vpnet/ecapa`），`sim-bench` 通过专用评测采集通道取回**已通过 100 ms 质检与内核强制 16-bit 水印验签的最佳候选波形**单独喂入 `验证`/`辨认` 闭环及去水印攻击矩阵（**此内部基准采样通道绝不改变生产目录或绕过 `_simulate` 门禁**），从而确保在所有 `(embedder, synthesizer)` 组合下都能完整量化防伪拦截率与攻击存活曲线。
2. **核心量化指标汇总**：
   - **Self-Refine 闭环增益**：`attempt=1` vs 最终轮次 `Cosine SIM` / `NormScore` 变化与收敛率；
   - **质检、溯源与防伪拦截**：100 ms 质检通过率、16-bit 水印回读准确率（目标 100%）与回灌 `验证`/`辨认` 防伪拦截率（目标 100%）；
   - **去水印攻击存活率矩阵**：对带水印候选依次施加 `white_noise (SNR 15/10 dB)`、`duration_crop (0.6 s)`、`lowpass (4.5 kHz / 3.6 kHz)`、`resample (8k->16k)`、`gain_clip (1.8x)`，统计 `SpreadSpectrumWatermarkDetector` 的检出拦截率与 z-score 衰减情况；
   - **RTF 性能指标**：统计合成耗时与 Self-Refine 全闭环延迟 `P50 / P95`。
3. **报告输出**：在 `reports/voice_eval/` 下生成可追溯的 `sim_bench_<embedder>_<backend>.json` 与可读 Markdown 总结报告。

---

## 5. 关键工程边界与验收标准

1. **零破坏原有回归门禁**：运行全量现有测试（73 项）在 `-W error` 下 100% 通过；`python -m voice_eval audit` 通过；现有 7 维评测结果与 `protocol.json` 门禁完全保持一致。
2. **新增完整 TDD 测试覆盖**：
   - `voiceprint/synth.py` 单元测试（单测 `AcousticFormantSynthesizer` 音频/文本双模式、声学特征向参考迁移方向单调变化、`SubprocessNeuralSynthesizer` 模拟超时与 UTF-8 JSON 协议进程树回收）；
   - `agent/service.py` `模拟` 状态机测试（5 道前置门禁、Self-Refine 参数调整触发重试收敛、未收敛不释放危险产物、`detail=False` 防泄分、不污染 `verify` 锁定期事件）；
   - `agent/mcp_server.py` `voiceprint_simulate` 工具测试（参数校验、只读拦截、Base64 返回、与 `voiceprint_verify` 同会话闭环互操作）；
   - `voice_eval/dims/redteam.py` 6 项新增红队探针与 `voice_eval/dims/simulation.py` 端到端闭环基准测试。
3. **文档同步**：后续实施收尾阶段更新 `README.md`，明确写出 Windows 显卡机配置外部神经克隆插件（F5-TTS/CosyVoice2）以及运行 `sim-bench` 的方法，并把亮点同步到简历。

---

## 6. 留给实现阶段的设计决策点（What's Left to the Builder）

下列具体参数与算法细枝由实现阶段根据实测收敛情况自主选定并固化入代码与单测（无需再回头修改高层架构）：
1. `AcousticFormantSynthesizer` 短时 STFT 窗长（建议 25 ms Hanning 窗、10 ms 帧移、512 点 FFT）、基频估计平滑滤波器参数及高频抑制系数；
2. 文本合成器中汉字/ASCII 字符到音节时长与清浊音帧序列的确定性哈希映射细节；
3. `plugins/neural_tts_worker.py` 对 F5-TTS 与 CosyVoice2 具体调用 API 版本差异的分支封装与显存清理细节（如 `torch.cuda.empty_cache()`）；
4. Self-Refine 重试步长微小调节幅度（只要保证单调朝着缓解具体失败原因的方向调整，并在单测中验证能使临界失败样本经过重试转为收敛即可）。
