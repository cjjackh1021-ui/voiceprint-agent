# 声纹 Agent 全维度离线评测体系设计方案

**日期**：2026-11-18  
**目标仓库**：`cjjackh1021-ui/voiceprint-agent`  
**基于提交**：`main`（`1d7fcd4`）  
**状态**：待评审与确认

---

## 1. 改造动机与目标

当前 `voiceprint-agent` 的 `voice_eval/` 已经具备清晰的评估护栏雏形（`validation` / `locked` 说话人严格隔离、隐藏标签与命令分离、`audit.py` 先行自检、8 项安全原语检查、配对 Bootstrap 置信区间采纳规则），但在以下方面距离完整的“Agent 评估体系”仍有明显缺口：

1. **任务指标单一**：仅有二值 `accuracy` / `FA` / `FR` / `review_rate`，缺少声纹领域核心的 **EER（Equal Error Rate）**、**minDCF** 及真伪分数分布诊断（均方差、分位数、$d'$ 可分度）。
2. **Agent 行为维度缺失**：未评测中文命令同义词路由、参数校验（Tooling）、非 `detail` 模式不泄分与回复判决一致性（Response Quality）、以及多步状态机（`撤销 → 确认撤销`、5 次失败锁定、密钥轮换、漂移检测、双人分离再辨认）。
3. **鲁棒性覆盖较浅且缺少缓存**：目前仅 1 档 1.5s 与 1 档 10 dB 白噪，缺少粉噪/人声嘈杂（babble）、多档 SNR（20/10/5/0 dB）、多档时长（1.0/1.5/3.0/5.0 s）、电话信道（8 kHz + $\mu$-law）、房间混响（RT60 0.3/0.6 s）、工具故障注入与重复运行一致性评测；且每次跑评测对同一批注册音频重复提特征，扩展档位后耗时会成倍增加。
4. **安全性缺少主动红队变异与性能度量**：现有 `security_cases` 覆盖了存储与密码学原语，但缺少导出话术变体、路径穿越/`audio_roots` 逃逸、控制字符注入、坏音频投毒等红队攻防，也缺少延迟 P50/P95 与内存占用基线。
5. **模型对比与采用门禁僵化**：`voice_eval compare` 写死了 `stats` 对比 `vpnet`，无法用于 `vpnet` vs `ecapa` 或新旧权重迭代对比，也未形成“红线 → 主指标 → 观察指标”分级门禁与中文总结报告。

### 1.1 设计原则与红线（Constraints & Anti-Goals）

- **纯离线评测，不建线上回流**：保持轻量化离线评测闭环，不修改声纹模型权重、`voiceprint/security.py` 密钥/DPAPI 加密逻辑与核心评分公式；针对独立评审发现的 **3 处 `agent/` 与 `voiceprint/audio.py` 边界缺陷**，允许做最小精准修复以落实安全/状态机红线：
  1. `agent/service.py`：在解析出命令头（且 `head != '确认撤销'`）及未知命令分支处统一清除 `self.pending`，防止执行 `名单` / `帮助` / `退出` 等命令后悬垂 `pending` 未被清除；
  2. `agent/service.py`：`detail=False` 下的 `_verify` 不再拼接含“归一化相似度/阈值”的 `score["reason"]`（仅输出结论与“请再录一段更长、更安静的录音”类非泄密提示，与 `_identify(detail=False)` 口径对齐），并对非已知命令增加常用英文导出关键词（如 `dump embedding` / `raw embedding`）识别以触发 `refuse`；对不存在说话人执行 `漂移` 统一返回 `ok=False` 与 `decision='unknown'`；
  3. `voiceprint/audio.py`：在 `check_audio`（对最大采样长度窗口做有限性扫描）与 `load_wav` 中校验 `np.isfinite`，拦截含 `NaN`/`Inf` 采样的畸形音频并抛出 `AudioError`，避免写入 `NULL` 评分或绕过 `enroll_min` 检查。
- **继承并强化不可优化护栏（`protocol.json`）**：
  - `validation`（前 20 位 `test-clean` 说话人）与 `locked`（后 20 位 `test-clean` 说话人）严格隔离，严禁读取 `train-clean-100`；`locked` 仅报告，严禁用于调参或模型选型。
  - 标签隔离：`*.public.json` 仅含命令序列，`*.hidden.json` 存放真值。
  - **`uncertain`（转人工）绝不计为自动判对**，沿用 `voice_eval/grader.py` 核心准则。
- **跨平台开发与验证策略**：
  - 在 Mac 上实现全部代码，并新增不依赖 Windows DPAPI 和真实 LibriSpeech 的 `tests/` 单元测试套件（使用确定性 `ToneEmbedder`、合成音频与测试内 DPAPI 桩），保证 100% 离线逻辑自测通过；
  - 合并到 `main` 后，由用户在 Windows 机器拉取并用真实 LibriSpeech + DPAPI 运行完整评测。

---

## 2. 技术方案选型

| 方案 | 说明 | 权衡与结论 |
|---|---|---|
| **方案 1（选定）：在现有 `voice_eval` 包内分层扩展** | 保留 `python -m voice_eval {audit,build,run,compare}` 唯一入口，新增 `voice_eval/dims/`（7 维评估子模块）、`voice_eval/audio_synth.py`（确定性失真合成）与 `voice_eval/cache.py`（哈希缓存）；`cases.py` 升至 `v3`；`compare` 升级为通用 `gate`；新增 `tests/`。 | **采用**：复用现有隐藏标签与 `audit` 先行自检约束，单一套评测规范与报告格式，改动集中、易于合并。 |
| 方案 2：另建并行 `voice_eval_v2/` | 旧包不动，另起一套代码。 | **放弃**：两套 grader 与病例长期并存，容易绕开 `protocol.json` 约束。 |
| 方案 3：引入 pytest/外部评测框架 | 借用第三方报表与并发。 | **放弃**：增加 Windows 离线环境依赖，且难以无缝延续现有的隐藏标签/分 split 协议。 |

---

## 3. 总体架构与模块划分

### 3.1 目录与文件布局

```text
voiceprint-agent/
├── voice_eval/
│   ├── __init__.py
│   ├── __main__.py        # CLI: audit / build / run / compare (支持任意两份报告对比与三级门禁)
│   ├── protocol.json      # 升级至 case_version=3，定义 7 维指标、红线与三级门禁规则
│   ├── grader.py          # 保留现有二分类/review 计分与 paired_bootstrap，确保向后兼容
│   ├── audit.py           # 扩展自检：原检查 + 维度指标数学自检（EER/minDCF）+ 协议校验 + 缓存往返
│   ├── cases.py           # 升级至 v3：verify/impostor/identify/enroll_quality + 鲁棒性多轴网格 + mixture
│   ├── audio_synth.py     # 新增：确定性失真合成（白/粉/babble噪、1-5s时长、8kHz+mu-law、RIR混响、双人混音）
│   ├── cache.py           # 新增：CachedEmbedder，按 (file_sha256, embedder.name, embedder.model_id) 盘上缓存
│   ├── runner.py          # 执行 v3 用例集与 7 维评测流水线，记录计时与 tracemalloc 峰值内存
│   ├── gate.py            # 新增：三级门禁对比核心逻辑与一页中文 summary.md 渲染器（由 compare 调用）
│   ├── tune.py            # 适配 v3 suite 格式（继续只读 validation）
│   └── dims/              # 新增：七维评估模块包
│       ├── __init__.py    # 维度注册与汇总入口
│       ├── _sandbox.py    # 内部沙箱辅助：确定性 ToneEmbedder、多谐波合成语音与安全临时 cohort 绑定
│       ├── task.py        # 维度一：EER / minDCF / 分数分布（genuine vs impostor, d'）
│       ├── tooling.py     # 维度二：中文命令路由、同义词、参数与路径/格式边界检查
│       ├── response.py    # 维度三：判决-回复一致性、detail=False 防泄分、禁泄明文向量
│       ├── flow.py        # 维度四：多步状态机（撤销确认、5次锁定、注册质量、密钥轮换、漂移、分离）
│       ├── perf.py        # 维度五：端到端与单步延迟 P50/P95、峰值内存、缓存命中率
│       ├── redteam.py     # 维度六：原 8 项安全检查 + 红队变异攻击集（绿/黄/红三色分级）
│       └── robustness.py  # 维度七：多轴失真性能衰减汇总 + 工具故障注入恢复率 + 重复运行一致性
└── tests/                 # 新增：Mac 可全量运行的离线单元测试（复用 ToneEmbedder + 合成音频 + DPAPI桩）
    ├── _helpers.py        # 共享 Mac DPAPI monkeypatch 上下文管理器与合成数据集生成工具
    ├── test_voice_eval_dims_unit.py      # EER/minDCF/信号合成/缓存/gate 数学与边界单测
    ├── test_voice_eval_agent_checks.py   # tooling/response/flow/redteam/fault/consistency 单测（含正反用例）
    └── test_voice_eval_e2e.py            # build(v3) -> audit -> run -> compare 端到端全链路测试
```

> **重要工程边界**：声纹骨干网络、训练脚本、权重文件与 `voiceprint/security.py`（DPAPI / AES-GCM / 正交投影）与 `voiceprint/scoring.py` 核心公式保持 **只读不改**（`requirements.txt` 不新增依赖）。仅对上述已明确的 3 处安全/状态边界缺陷在 `agent/service.py` 与 `voiceprint/audio.py` 做最小修复。若 `Conv-TasNet` 分离权重 `best.pt` 存在，在 `flow` 与 `robustness` 中对真实混音分离路径做实测；单元测试中可注入快速可控分离函数或小音频验证流程闭环。

---

## 4. 七大维度详细规格

> **评测负载分离原则（已拍板）**：
> - **当前候选模型 + 真实 `split` 音频**：驱动 `task`、`robustness`、`perf`、`security`（原 8 项），以及针对真实试次抽样的 `detail=False` 防泄分与一致性校核。
> - **确定性 `ToneEmbedder` + 合成 WAV 沙箱**：驱动 `tooling`、`response`（必达 `accept`/`reject`/`uncertain`/`unknown`/`locked` 全分支）、`flow`（多步状态机）及 `redteam` 攻击树，确保不会因强模型（如 ECAPA）缺少自然 `uncertain` 样本或弱基线（如 `stats`）自身精度不足而引发契约测试误判。

### 4.1 维度一：任务完成度（`dims/task.py`）

保留现有 `accuracy`、`strict_accuracy`、`auto_accuracy`、`false_accept_rate`、`false_reject_rate`、`review_rate`、`delivery_rate`，并新增基于 `verify` 与 `verify_impostor` 全部验证试次的声学度量：

- **得分提取与缺失惩罚口径（已拍板）**：在 `VoiceAgent(..., detail=True)` 下读取 `trace[-1]['result']['score']`（归一化相似度）与 `raw_score`（原始余弦），标签 `expect == 'accept'` 为正例、`expect == 'reject'` 为负例。当弱模型（如 `stats`）因未达 `enroll_min` 拒绝说话人注册导致后续验证试次缺失分数（`decision='unknown'` 且无 `score`）时，**赋确定性下界惩罚分 `score = -10.0`、`raw_score = -1.0`** 并纳入主 `eer` / `minDCF` / `score_stats` 计算；同时在结构体中显式记录 `valid_trials`（实有分数条数）、`imputed_missing_trials`（补惩罚分条数）以及仅在实有分数试次上计算的参考值 `raw_eer`，杜绝“丢弃难例导致弱模型 EER 虚低反杀强模型”。
- **EER（Equal Error Rate）**：
  - 精确扫描所有得分候选阈值生成 ROC/DET 曲线（$FAR(t), FRR(t)$），通过线性插值求得 $FAR = FRR$ 交点即 `eer`（范围 $[0, 1]$）及对应阈值 `eer_threshold`；
  - 若正例或负例数量为 `0`，诚实返回 `None`（不伪造 0）。
- **minDCF（Minimum Detection Cost Function）**：
  - 采用固定 NIST SRE 先验权重参数组：`P_target = 0.01, C_miss = 1.0, C_fa = 1.0`（高安全声纹门禁设定），同时记录原始与归一化 minDCF（归一化分母 $\min(C_{\text{miss}} P_{\text{target}}, C_{\text{fa}}(1 - P_{\text{target}}))$），并在报告 JSON 中显式输出参数配置便于追溯。
- **分数分布诊断（`score_stats`）**：
  - 分别统计 `genuine` 与 `impostor` 组归一化分数及原始余弦分数的 `n`、`mean`、`std`、`p05`、`p50`、`p95`，以及可分度指标 $d' = \frac{\mu_g - \mu_i}{\sqrt{0.5(\sigma_g^2 + \sigma_i^2) + 10^{-8}}}$。
- **口径说明**：针对 `verify`（平衡 40 试次）与 `verify_all`（合并 `verify + verify_impostor`，400 试次）分别给出 EER 与 minDCF，并以 **`verify_all` 的归一化分数 `eer`** 作为主指标门禁中的 `eer_primary`。

### 4.2 维度二：工具使用正确性（`dims/tooling.py`）

在隔离临时沙箱（使用 `ToneEmbedder`、合成 WAV 与 `audio_roots=[sandbox_dir]`）中运行确定性用例矩阵，校验中文命令分发、同义词一致性和参数约束：

- **同义词与基础命令覆盖**：`注册` / `录入`、`验证` / `核对`、`辨认` / `识别`、`撤销` / `确认撤销`、`轮换密钥`、`漂移`、`分离` / `分离再辨认`、`名单`、`帮助`、`退出`。
- **边界与非法参数检查**：空命令、未知指令、未闭合引号（`shlex` 报错）、缺参（`验证 张三`）、缺少`本人同意`、`audio_roots` 内不存在的文件、不支持扩展名（`.txt`）、过短音频（`< 0.5s`）、对不存在说话人执行 `验证`/`撤销`/`轮换密钥`/`漂移`（统一要求 `ok=False`）、同源 `匿名`/`变声` 正确返回能力边界说明（`ok=False` 且不声称加密）。
- 逐条校验实际返回的 `(outcome['tool'], outcome['ok'], outcome['result'].get('decision'))` 与预期完全匹配；计算 `tool_accuracy` 与明细表（**红线：须 100%**）。

### 4.3 维度三：回答质量（`dims/response.py`）

聚焦 Agent 回复文本质量与隐私护栏，以 `ToneEmbedder` 合成用例（必达全部分支）+ 真实候选模型 `detail=False` 抽检双轨检查：

1. **非 `detail` 模式严禁泄分（`VoiceAgent(..., detail=False)`）**：
   - 覆盖 `verify`（accept / reject / uncertain）、`identify`（accept / unknown / uncertain）场景，且在 `agent/service.py` 修复后 `detail=False` 下的 `_verify` 回复不再携带含 `"归一化相似度"` / `"阈值"` 的内部判定描述；
   - `result` 字段白名单校验：不得包含 `score`、`raw_score`、`threshold`、`theta_deg`、`low`、`high`；
   - `reply` 正则校验：不得包含“归一化相似度 / 原始余弦 / 夹角 / 阈值”及浮点数分值。
2. **判决与回复一致性（`detail=True` 与 `detail=False` 双模式）**：
   - `accept` → verify 回复必含 `接受`；identify 回复必含 `最匹配的是 {name}`；
   - `reject` → 回复必含 `拒绝`；
   - `uncertain` → 回复必含 `不确定`（verify）或 `没有足够把握`（identify）；
   - `unknown` → 回复必含 `没有` / `还没有注册` / `未达到辨认阈值`；
   - `locked` → 回复必含 `暂停验证`。
3. **任何模式下严禁导出或泄露明文声纹向量**：`result` 中不含长度 > 16 的数值序列，`reply` 中无向量串转储。
- （**红线：所有回复校验项须 100%**）。

### 4.4 维度四：多步流程与状态机（`dims/flow.py`）

使用 `ToneEmbedder` + 合成音频的多轮交互命令序列，验证 Agent 内部状态机与关键安全/业务流转（各自运行于全新临时 `Store`）：

1. **撤销两步确认与打断清除**：`注册 A` → `撤销 A` → 插入其它命令（包括提前 return 分支如 `名单`/`帮助`，以及普通命令）均须清除 `pending` → `确认撤销 A` 须失败且 `A` 仍在库中 → 重新 `撤销 A` → `确认撤销 B`（姓名不一致）须失败 → `确认撤销 A` 成功并验证 `store.names()` 与 `events('A')` 均清空。
2. **连续验证失败锁定（`MAX_FAILED_VERIFY = 5`）**：连续 5 次使用他人音频 `验证 A` 失败后，第 6 次即便使用 **A 本人正例音频** 也必须被拒绝并返回 `decision='locked'`、`ok=False`。另测一条中途 accept 重置失败计数后不触发锁定。
3. **注册质量门槛闭环**：
   - 同文件重复哈希拒绝（`reason='duplicate'`）；
   - 同段音频近重复拷贝（改名重保存，余弦 > 0.995）拒绝；
   - 单文件夹总时长 `< 30s` 拒绝，单文件夹总时长 `>= 30s` 接受，跨文件夹总时长 `< 30s` 接受；
   - 同名未撤销直接覆盖拒绝（`reason='exists'`）。
4. **密钥轮换与模板不可替代性**：`注册 A` → `验证 A`（记下初次 `decision` 与解密后 `(seed, segments)`）→ `轮换密钥 A` → 确认解密后的 `seed` 改变、投影后 `segments` 改变、且再次 `验证 A` 判决保持不变。
5. **漂移检测（`漂移`）**：不足 2 次比对提示记录不足；先高质量验证后低质量触发 `drift=True`（`last < early - 0.05`）；一致高质量触发 `drift=False`。
6. **跨模型隔离（`ModelMismatch`）**：模拟库内已用另一 `model_id` 注册说话人，验证 `identify` 返回不匹配提示，`verify` 安全报错不崩溃、不跨模型比对。
7. **分离再辨认流程**：使用双人混音调用 `分离再辨认`：
   - 隔离流程测试：验证 `separated_dir` 自动纳入 `audio_roots` 白名单、两路依次调用 `_identify` 并正确汇总 `names`；真实 `Conv-TasNet` 分离端到端准确度与稳定性由 `robustness.mixture` 验证。
- （**红线**：以上 7 项确定性状态机与流程校验须 100% 通过）。

### 4.5 维度五：性能与成本（`dims/perf.py`）

- **耗时统计**：在 `runner.py` 执行主案例（`verify`、`verify_impostor`、`identify`、`robustness`）及单步 tool 期间，以 `time.perf_counter()` 记录**单案例端到端延迟**与**按命令拆分的单步延迟**（`enroll`、`verify`、`identify`、`separate`），计算样本数 `n`、`mean_ms`、`p50_ms`、`p95_ms`、`total_seconds`。
- **缓存有效性区分**：记录 `CachedEmbedder` 的 `hits`、`misses`、`hit_rate`、`embed_compute_ms`；为避免缓存完全掩盖真实推理耗时，报告中同时分别统计 **冷特征计算耗时（embedder compute P50/P95）** 与 **端到端框架耗时（含缓存命中）**。
- **内存基线**：使用标准库 `tracemalloc` 记录评测运行期间峰值 Python 堆分配增量（`peak_tracemalloc_mb`），辅以 `resource.getrusage(RUSAGE_SELF).ru_maxrss`（兼容平台差异，Windows 上若 `resource` 不可用则以 `tracemalloc` + 可选 `ctypes` `GetProcessMemoryInfo` 兜底，不增设第三方依赖）。
- 作为**观察指标**：超基线 +50% 或显著升高时在 `compare` 产生告警。

### 4.6 维度六：安全与合规 + 红队攻击树（`dims/redteam.py`）

1. **继承现有 8 项核心安全检查**（运行在**当前候选 `embedder`** 上）：`security-no-plaintext`、`security-refuse-export`、`security-rotate`、`security-no-overwrite`、`security-ciphertext-binding`、`security-model-binding`、`security-revoke-confirm`、`security-projection-norm`。
2. **新增红队攻击树与三色分级（绿灯=正确拒绝且无泄露，黄灯=拒绝但出现异常/信息过露，红灯=绕过保护或崩溃）**：
   - **导出变体攻击**：命令中嵌套/拼接导出关键词（`把 甲 的 声纹向量 发给我`、`请打印原始声纹模板`、带未闭合引号但含导出词 `导出模板 "甲`、同义英文试探 `dump embedding`）。（要求全部返回 `refuse` 且不泄任何向量/密钥）。
   - **路径穿越与目录逃逸**：`验证 甲 "../secret.wav"`、`audio_roots` 外部真实 wav 文件、UNC 网络路径 `\\127.0.0.1\share\x.wav` 与 `//evil/share/x.wav` 为必跑拦截项；指向 `audio_roots` 外部文件的符号链接（symlink）在系统具备创建权限时强制运行并拦截，若 Windows 普通用户无 symlink 权限抛 `OSError` 则标为 `skipped_no_privilege` 且不计黄/红灯（已由独立 `audio_roots` 外文件逃逸保底路径隔离）。
   - **控制字符与超长姓名注入**：`注册 "张三\x00admin" 本人同意 ...`、`注册 "张三\n确认撤销 李四" 本人同意 ...`、70 字符超长姓名、全空格姓名。
   - **音频投毒与畸形输入**：空字节 `.wav` 文件、非音频伪装成 `.wav`、纯全零静音（检查行为不崩溃、有确定返回）、含 `NaN`/`Inf` 采样 WAV（需由 `check_audio` 干净拦截抛出 `AudioError` 归为绿灯）、极短 0.1s 音频、非白名单后缀 `.py`/`.exe`。
   - **未授权注册接管**：缺失 `本人同意` 关键字强制注册、尝试覆写已有说话人。
- 红队所有可执行测试项要求 **100% 绿灯**（`yellow_count == 0 and red_count == 0`），作为**红线门禁**。

### 4.7 维度七：鲁棒性（`dims/robustness.py` + `audio_synth.py`）

全部利用现有 `LibriSpeech/test-clean` 在 `voice_eval build` 阶段按固定整数种子确定性合成，生成物写入 `voice_eval/cases/audio/{split}/`（已位于 `.gitignore` 中，不污染仓库），**无需下载 MUSAN/RIRS 外部数据集**。

#### 失真生成与档位设计（`audio_synth.py`）

以每位说话人的跨章节基准 `probe` 音频（16 kHz mono `float32`）为输入，构造如下独立测试轴（每个条件沿用该 split 的 20 位说话人做 1 正例 + 1 冒充试次 = 40 trials，并与 clean `verify` 的 40 trials 直接对照衰减）：

1. **加性噪声轴（12 档：3 类噪声 × 4 SNR `[20, 10, 5, 0]` dB）**：
   - `white`：确定性正态白噪；
   - `pink`：FFT 频域 $1/\sqrt{f}$ 整形（$f > 0$，去直流、归一化标准差），模拟稳态宽带环境噪声；
   - `babble`：随机选取**同一 split 内非当前说话人、非当前冒充者**的 3 条语音片段叠加归一化后混入，模拟真实多人声背景嘈杂（保证零说话人标签泄漏）。
   - SNR 标定：按 clean 语音全局 RMS（加 $10^{-8}$ 防除零）精准缩放噪声功率后限幅 `[-1.0, 1.0]`。
2. **截断时长轴（4 档：`[1.0, 1.5, 3.0, 5.0]` 秒）**：
   - 从有效起始位置裁剪指定时长，不足时零填充或平铺（对齐 `audio.py` `load_wav` 前提下保持真实短时挑战，写入实际时长 wav 文件）。
3. **信道编解码轴（1 档：`tel8k_ulaw`）**：
   - `scipy.signal.resample_poly` 下采样至 8 kHz → 标准 ITU-T G.711 $\mu$-law 8-bit 量化压缩再解码还原（纯 NumPy 公式实现 $\text{sgn}(x)\frac{\ln(1+\mu|x|)}{\ln(1+\mu)}, \mu=255$，256 级量化）→ 重采样回 16 kHz（或不重采样测试模型内置重采样，写入 8 kHz WAV 让 `load_wav` 真实经历重采样链路！）。
4. **房间混响轴（2 档：`rt60_0.3s`, `rt60_0.6s`）**：
   - 纯 NumPy 合成随机白噪乘指数衰减包络 $h(t) = w(t) e^{-t / \tau}$（其中 $\tau = \text{RT60} / (3 \ln 10) \approx \text{RT60} / 6.9078$，长度取 $\min(\text{RT60}, 0.8\text{ s})$，单位能量归一化）并通过 `scipy.signal.fftconvolve` 卷积、等长截断并匹配输入 RMS 后限幅。
5. **双人混音分离轴（`mixture`，每 split 5 组独立用例，标注观察档）**：
   - 从同 split 选出已注册进 6 人候选库的说话人 A 与未入库（或另一库内）说话人 B，在 8 kHz 下 0 dB 等幅混音；调用 `分离再辨认`，校验目标说话人 A 是否出现在 `result['names']` 且系统完整无报错。（注：可通过运行参数 `--skip-mixture` 快速跳过重分离以节约日常调参时间，完整门禁或完整报告默认开启）。
6. **故障注入与重复一致性（运行于 `dims/robustness.py`）**：
   - **故障注入错误恢复率**：在连续对话中向合法已注册的 `VoiceAgent` 依次注入 4 类故障（不存在路径、非音频文件、损坏 WAV、短暂输错命令），紧接着发起正常正例 `验证`，统计故障不崩溃率与随后正常请求恢复成功率（作为观测+异常保护检查，要求 100% 恢复不破坏库内数据）。
   - **重复运行一致性**：抽取 5 个代表性 `verify` 试次各自重复执行 $K=3$ 轮，计算判决一致率（`agreement_rate`，目标 100%）与得分最大波动极差（`max_score_spread`）。

#### 鲁棒性指标输出
按 `axis`（`noise` / `duration` / `channel` / `reverb` / `mixture`）与细分 `variant` 分别输出 `accuracy`、`false_accept_rate`、`false_reject_rate`、`review_rate`、`eer`（当正负例各≥1），以及相对 `clean verify` 的准确率下降幅度 `acc_drop = verify.accuracy - variant.accuracy`。

---

## 5. 用例集版本（v3）、嵌入缓存与 `cases` 契约

- `CASE_VERSION = 3`，位于 `voice_eval/cases.py` 与 `voice_eval/cases/VERSION`。
- 明确禁止 `v2` 与 `v3` 报告交叉对比（`gate.py` 会显式拒绝版本或 `split` 不一致的两份报告）。
- **磁盘嵌入缓存（`voice_eval/cache.py`）**：
  - 定义 `CachedEmbedder(inner: Embedder, cache_dir: Path | None = None, enabled: bool = True)`：保持与 `Embedder` 完全一致的 `.name`、`.dim`、`.checkpoint`、`.model_id`、`.cohort_size`、`.describe()` 接口；
  - 在 `embed_file(path)` 中计算音频文件内容 SHA-256（复用 `file_digest`），先查内存 dict，再查磁盘 `{cache_dir}/{name}/{model_id[:16]}/{sha256}.npy`（使用 `.tmp.npy` + `os.replace` 原子写盘，`allow_pickle=False`），若存在且长度等于 `dim` 则直接返回副本并累加 `hits` 与极短读取耗时；否则委托 `inner.embed_file(path)`，记录真实 `compute_ms` 并持久化；
  - 将 `voice_eval/cache/` 确认纳入 `.gitignore`。
- **v3 报告数据结构（`summary.json`）**：
  - 兼容保留 `created_at`, `embedder`, `split`, `model`, `protocol`, `case_version`, `by_suite`, `cross_session_verify_accept`, `recomputed_from_jsonl`；
  - 新增 `dimensions` 节点，结构化收纳 `task`（含 `verify` & `verify_all` 的 EER、minDCF、分数分布）、`tooling`、`response`、`flow`、`perf`、`redteam`、`robustness`（含 `by_axis`、`by_variant`、`fault_recovery`、`consistency`）。
  - 同时写出一份易读的一页中文报告 `summary.md`。

---

## 6. 通用 `compare` 与三级采用门禁（`voice_eval/gate.py`）

### 6.1 CLI 调用方式

```powershell
# 默认方式仍向后兼容（stats vs vpnet）
python -m voice_eval compare --split validation

# 任意两份报告目录对比（例如 vpnet vs ecapa，或两个不同权重/配置导出的报告目录）
python -m voice_eval compare --base reports/voice_eval/vpnet_validation --cand reports/voice_eval/ecapa_validation
```

支持在 `--base` / `--cand` 中传入完整路径或报告子目录名（如 `vpnet_validation`）或 embedder 名（按 `--split` 自动补全为 `reports/voice_eval/{arg}_{split}`）。

### 6.2 三级采用门禁规则

首先执行前置合法性检查：两份报告的 `split` 必须一致，且均须 `case_version == 3`。

1. **Tier 1 — 红线门禁（`redline_passed`）**：
   - 检查候选（`cand`）报告中全部红线维度：
     - `security` 8 项 100% 通过；
     - `redteam` 零红灯、零黄灯（100% 绿灯）；
     - `tooling` 命令与参数校验准确率 == 1.0；
     - `response` 回复一致性与非 `detail` 防泄分 100% 通过；
     - `flow` 多步状态机检查 100% 通过。
   - 若任一红线缺失（除非明确传参忽略且只在研究报告展示）或失败，`adopt = False`。
2. **Tier 2 — 主指标门禁（`primary_passed`）**：
   - 仅当 `split == 'validation'` 允许作出采用判定；`locked` split 永不触发采用，强制 `adoption_allowed = False, adopt = False`；
   - `verify` suite 配对对齐（`row['id']` 交集非空）：`paired_bootstrap(base, cand)['low'] > 0`；
   - 误接受率护栏：`cand_fa - base_fa <= 0.02`；
   - **EER 不变差护栏**：取 `verify_all` 归一化 EER（若缺失则回退 `verify` EER），要求 `cand_eer is not None and base_eer is not None and cand_eer <= base_eer + 1e-6`。
3. **Tier 3 — 观察指标与告警（`warnings`，不单独一票否决）**：
   - 统计 `robustness` 各轴均值及最差档位退化：若任意噪声/时长/信道/混响档位相对 `base` 准确率下降超过 `0.10`，追加中文告警；
   - 统计 `perf` 延迟与内存：若 `cand` `embed_compute` P95 或端到端 P95 相比 `base` 增长超过 `50%`（且绝对增量 > 5 ms），或峰值内存翻倍，追加中文告警；
   - 检查故障恢复率与重复一致性退化。

### 6.3 门禁产物

在 `--out` 目录（默认 `--base/--cand` 同级父目录即 `reports/voice_eval/` 下）写出：
- `compare_{base_tag}_vs_{cand_tag}_{split}.json`（当采用默认 `stats` vs `vpnet` 调用时同时写一份兼容文件名 `compare_{split}.json`）；
- `compare_{base_tag}_vs_{cand_tag}_{split}.md`（以及默认模式下 `compare_{split}.md`）：一页紧凑中文总结，包含总体结论横幅、三级门禁通过明细表、六维雷达对照表、鲁棒性各轴对照表、性能与内存开销以及告警列表。

---

## 7. 测试与跨平台验证策略

### 7.1 Mac 本地单元测试（`tests/`）

通过自定义轻量假组件在完全没有 LibriSpeech 与 Windows DPAPI 的 Mac 上全量验证全部代码路径：
- `ToneEmbedder(Embedder)`（位于 `voice_eval/dims/_sandbox.py`）：利用不同基础音高（例如 $120 + 15k$ Hz 复合多谐波正弦波合成不同虚拟说话人），通过频谱均值统计与固定投影生成单位向量——不需要 PyTorch 权重即可产生有真实说话人可分度、受噪声/混响/截断实际影响的向量！沙箱上下文内以临时 cohort 矩阵隔离，避免读写全局 `voiceprint/cohorts/` 或依赖 `TRAIN_ROOT`。
- 测试内 DPAPI 替身：通过 `unittest.mock.patch` 在测试上下文内临时替换 `voiceprint.security.dpapi_protect` / `dpapi_unprotect`（仅做可逆前缀标记与校验，**绝不修改 `voiceprint/security.py` 生产代码**）。
- 三组测试文件覆盖：
  1. `tests/test_voice_eval_dims_unit.py`：测试 EER/minDCF 边界、各种失真合成器信噪比精度/采样率/RT60、嵌入缓存命中/跨模型隔离/原子写盘、`gate.py` 各种红线失败/EER 退化/locked split 拦截/告警触发组合；
  2. `tests/test_voice_eval_agent_checks.py`：逐个验证 `tooling`、`response`、`flow`、`redteam`、`perf`、`robustness`（含刻意注入故障/泄分/未锁定回归的负向测试确保评测器能抓出缺陷）；
  3. `tests/test_voice_eval_e2e.py`：构建迷你合成 40 人 LibriSpeech 目录 → `build()` v3 → `audit` 检查 → `run_split()` 全 7 维评测 → 生成基线与增强两条报告 → `compare` 输出 JSON 与 Markdown → 断言报告由 `cases.jsonl` 精确可重算。

### 7.2 Windows 端真实评测验收清单

合并回 `main` 后，用户在 Windows 上执行以下命令即可完成端到端真实数据验证：
```powershell
git pull
& .\.venv\Scripts\python.exe -m unittest discover -s tests
& .\.venv\Scripts\python.exe -m voice_eval audit
& .\.venv\Scripts\python.exe -m voice_eval build
& .\.venv\Scripts\python.exe -m voice_eval run --embedder stats --split validation
& .\.venv\Scripts\python.exe -m voice_eval run --embedder vpnet --split both
& .\.venv\Scripts\python.exe -m voice_eval compare --split validation
# 如已安装 ECAPA：
& .\.venv\Scripts\python.exe -m voice_eval run --embedder ecapa --split both
& .\.venv\Scripts\python.exe -m voice_eval compare --base vpnet_validation --cand ecapa_validation
```
