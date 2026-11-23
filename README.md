# Voiceprint Agent：受控声纹识别、可控声音模拟、MCP 服务与七维评测体系

集**声纹状态机 Agent、JSON-RPC 2.0 MCP Server、100 ms 音频质检、0.5 s 16-bit 扩频水印防伪、可控声音模拟与 Self-Refine 自检重试、以及七维离线评测与三级统计门禁**于一体的高可靠本地语音安全系统。

核心运行环境仅依赖 `numpy`、`scipy`、`soundfile`、`cryptography`、`torch`，支持 macOS 与 Windows（AES-256-GCM 模板加密 + 可撤销正交投影，Windows 原生 DPAPI 主密钥保护 + Job Object 隔离子进程）。

---

## 核心能力与架构

1. **声纹 Agent 状态机（`agent/service.py`, `agent/cli.py`）**
   - 支持中文指令：`注册`（`录入`）、`验证`（`核对`）、`辨认`（`识别`）、`分离再辨认`、`撤销`/`确认撤销`（两步防误删）、`轮换密钥`、`漂移`、`质检`、`模拟` 与 `名单`。
   - **对称 ZT-Norm + 自助法置信区间决策**，严禁导出声纹明文或低细节模式下泄露分数/阈值；连续 5 次验证失败自动触发 10 分钟安全锁定期。
   - **容器完整性与 100 ms 滑动窗质检（`voiceprint/audio.py`, `voiceprint/audio_quality.py`）**：魔数嗅探 + RIFF/WAVE 逐块长度校验 + 无模型静音/削波/非有限采样检测。
   - **0.5 s 16-bit 正交频域扩频水印防伪（`voiceprint/spoof.py`）**：覆盖 1.5–7.5 kHz 频段，在短语音与加噪条件下精准拦截合成或水印克隆语音。

2. **可控声音模拟与 Self-Refine 自检重试闭环（`voiceprint/synth.py`, `agent/service.py`）**
   - **双后端可插拔架构**：
     - **默认零依赖声学迁移后端（`AcousticFormantSynthesizer`）**：纯 `numpy`/`scipy` 实现跨说话人动态频谱包络 + 基频移调（`源音频` 模式）与确定性韵律合成（`文本` 模式），保障 macOS/Windows 无卡环境毫秒级 100% 确定性复现。
     - **Windows GPU 神经克隆隔离插件（`SubprocessNeuralSynthesizer` + `plugins/neural_tts_worker.py`）**：通过 POSIX PGID / Windows Job Object 与 `PYTHONUTF8=1` 隔离运行外部独立 `.venv_tts`（默认适配 **F5-TTS** `text` Zero-Shot 克隆并支持可选 `参考文本` 免离线 ASR，预留 **CosyVoice2** 插槽），主仓不引入任何重型生成依赖。
   - **五道前置安全门禁 + Self-Refine 状态机**：
     - 严格校验 `本人授权`（MCP 严格布尔 `consent=true`）、目标已注册未锁定、参考音反递归防伪（拦截已携带水印音频防止二次洗白）与参考音本人声纹归属核验（跨人冒充拒答且不污染验证失败锁定期）。
     - 循环 `1..max_sim_attempts` 轮：`合成 → 受信任内核强制嵌 16-bit 水印并回读验签 → 100 ms 质检 + 目标声纹自检 → 按削波降增益/按相似度低提高包络与基频迁移参数重试`；未通过水印、质量或目标声纹门禁时坚决不向 `simulated/` 释放产物。

3. **零依赖 JSON-RPC 2.0 `stdio` MCP Server（`agent/mcp_server.py`）**
   - 原生暴露 `voiceprint_enroll`、`voiceprint_verify`、`voiceprint_identify`、`voiceprint_list_speakers`、`voiceprint_revoke`、`voiceprint_rotate_key`、`voiceprint_check_audio_quality`、`voiceprint_health`，开启合成器时动态暴露 `voiceprint_simulate`。
   - 支持 `--audio-roots` 路径沙箱、Base64 Data URI 魔数解码（无 `audio_roots` 时严禁路径入参）、`--read-only` 写工具拦截与 `simulated/` 闭环回灌校验。

4. **七维离线评测、去水印红队与闭环模拟基准（`voice_eval/`）**
   - **七维评测与三级发布门禁（`voice_eval/runner.py`, `voice_eval/gate.py`）**：严格互斥划分 `validation`（前 20 位）与 `locked`（后 20 位）说话人，覆盖 `task`（精确插值 EER / NIST SRE minDCF / d'）、`tooling`、`response`、`flow`、`perf`（无缓存 Warm RTF + RSS 内存）、`redteam`（含 6 项仿真安全探针）、`robustness`（单分片 1,190 条多工况合成）。
   - **生成–鉴伪双轨评测 `sim-bench`（`voice_eval/dims/simulation.py`, `voice_eval/audio_synth.py`）**：量化 1-shot → Self-Refine 收敛率/SIM 增益（Track A），并评估 16-bit 水印完整性、回灌 `验证` 防伪拦截率及噪声/截断/4.5 kHz & 3.6 kHz 低通/8 kHz 重采样/削波 7 项去水印攻击存活率（Track B）。

---

## 快速开始与常用命令

### 1. 基础环境与全量回归测试
```bash
python -m venv .venv
# Windows: .venv\Scripts\python.exe  |  macOS/Linux: .venv/bin/python
.venv/bin/python -m pip install -r requirements.txt

# 严格零警告运行全部单元与集成回归测试
.venv/bin/python -W error -m unittest discover -s tests -v

# 离线仿真闭环评测冒烟（零数据集即可运行）
.venv/bin/python -W error -m voice_eval sim-bench --embedder tone --synthesizer acoustic
```

### 2. 启动交互 Agent 与 MCP 服务
```bash
# 本地交互 CLI（默认启用 zero-dep acoustic 模拟与可开启防伪拦截）
python -m agent.cli --embedder vpnet --spoof-guard

# CLI 交互指令示例
注册 张三 本人同意 "D:/recordings/day1/a.wav" "D:/recordings/day2/b.wav" "D:/recordings/day3/c.wav"
验证 张三 "D:/recordings/probe.wav"
模拟 张三 本人授权 "D:/recordings/day1/a.wav" 源音频 "D:/recordings/other.wav"
模拟 张三 本人授权 "D:/recordings/day1/a.wav" 文本 "你好，这是受控溯源测试语音" 参考文本 "参考音频原句文本"

# 启动 MCP stdio 服务
python -m agent.mcp_server --data-dir ./agent_data --embedder vpnet --audio-roots ./recordings
```

### 3. Windows GPU 显卡机接入真实 F5-TTS 神经克隆（独立环境隔离）
主仓库保持零重依赖；在 Windows 显卡机（如 RTX 4070 SUPER，`D:\voice confirm`）创建独立 `.venv_tts` 环境：

```powershell
py -3.12 -m venv "D:\voice confirm\.venv_tts"
& "D:\voice confirm\.venv_tts\Scripts\python.exe" -m pip install f5-tts soundfile numpy
```

随后即可在 CLI / MCP / `sim-bench` 中通过 `--synthesizer neural_subprocess` 隔离挂载真实 F5-TTS（需准备 `data/LibriSpeech/` 或本地语料）：

```powershell
# CLI / MCP 挂载真实 F5-TTS（目录含空格请保留双引号）
& "D:\voice confirm\.venv\Scripts\python.exe" -m agent.cli `
  --embedder vpnet --spoof-guard --synthesizer neural_subprocess `
  --neural-cmd '"D:/voice confirm/.venv_tts/Scripts/python.exe" plugins/neural_tts_worker.py --engine f5_tts'

# 在 LibriSpeech + VPNet/ECAPA 上运行真实生成-鉴伪闭环基准
& "D:\voice confirm\.venv\Scripts\python.exe" -m voice_eval sim-bench `
  --embedder vpnet --synthesizer neural_subprocess --supported-modes text `
  --neural-cmd '"D:/voice confirm/.venv_tts/Scripts/python.exe" plugins/neural_tts_worker.py --engine f5_tts'
```

### 4. 七维离线基准与三级发布门禁（需 Windows DPAPI 与 LibriSpeech 数据集）
- 首次使用 VPNet/ECAPA 前将 [LibriSpeech](https://www.openslr.org/12/) 的 `train-clean-100` 与 `test-clean` 放在 `data/LibriSpeech/`：
```bash
python -m voice_eval audit
python -m voice_eval build
python -m voice_eval run --embedder stats --split validation
python -m voice_eval run --embedder vpnet --split validation
python -m voice_eval compare --base stats --cand vpnet --split validation
```
