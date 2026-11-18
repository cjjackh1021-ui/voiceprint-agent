# 声纹 Agent

在 Windows 本机用中文命令注册说话人、验证指定身份、辨认库内说话人，也可以先分离双人混音再辨认。注册模板按模型标识隔离，使用 AES-GCM 加密，主密钥由当前 Windows 用户的 DPAPI 保护。

## 在 Windows 上运行

使用 Python 3.12。在仓库根目录的 PowerShell 中执行以下命令。

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m voice_eval audit
```

首次使用 VPNet 前，将 [LibriSpeech train-clean-100](https://www.openslr.org/12/) 解压到仓库中的 `data/LibriSpeech/train-clean-100/`。该目录下一级应为说话人编号。Agent 从每位训练说话人的一段 FLAC 建立评分用的参考向量，之后会复用与权重匹配的本地缓存。

```powershell
& .\.venv\Scripts\python.exe -m agent.cli --embedder vpnet
```

输入 `帮助` 查看全部命令。注册时准备至少三段本人单独发声的录音，建议来自多次录制，合计至少 30 秒。交互命令中的文件路径使用正斜杠，含空格时加双引号。

```text
注册 张三 本人同意 "D:/recordings/day1/a.wav" "D:/recordings/day2/b.wav" "D:/recordings/day3/c.wav"
验证 张三 "D:/recordings/probe.wav"
辨认 "D:/recordings/probe.wav"
分离再辨认 "D:/recordings/mixture.wav"
轮换密钥 张三
撤销 张三
确认撤销 张三
```

## 模型与评测

- VPNet 输出 192 维说话人向量。仓库内附当前使用的 `vpnet_posttrain.pt` 原始权重，便于按文件哈希识别模型版本。
- 双人分离采用 8 kHz Conv-TasNet，包含分离网络和当前推理入口使用的 `best.pt`。网络实现来自 [JusperLee/Conv-TasNet](https://github.com/JusperLee/Conv-TasNet)。
- 可选 ECAPA 后端使用 [SpeechBrain 的 VoxCeleb 权重](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)。额外安装与 torch 版本匹配的 torchaudio、speechbrain 和 huggingface_hub，首次使用 `--embedder ecapa` 时下载代码指定版本并校验哈希。统计基线 `--embedder stats` 需要 librosa。
- `python -m voice_eval audit` 检查加密流程和评测逻辑。语音效果评测还需将 LibriSpeech test-clean 放在 `data/LibriSpeech/test-clean/`，执行 `python -m voice_eval build`，再执行 `python -m voice_eval run --embedder vpnet --split validation`。

默认业务数据位于 `agent_data/`。录音、模板数据库、密钥和参考声纹缓存均在本地生成或提供。
