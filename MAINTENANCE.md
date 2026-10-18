# 维护文档 — 实验中遇到的问题及调整

本文档记录从 Intan RHX Demo 模式接入流水线起，调试过程中所有暴露出的问题、根因、以及落地的修复。条目按时间顺序编号；每条都给出**现象 / 根因 / 解决 / 涉及文件**。后续如再有调整，请在文末以同样格式追加。

---

## 0. 当前验证状态

| 级别 | 脚本 | 状态 | 关键判据 |
|------|------|------|----------|
| Level 1 (8ch / 30s) | `experiments/stream_stress_test.py --duration 30 --n-channels 8` | **PASS** | 900,000 / 900,000 帧；0 gap，0 dup，0 out-of-order，0 cross-batch gap |
| Level 1 (64ch / 60s) | `experiments/stream_stress_test.py --duration 60 --n-channels 64` | **PASS** | 1,800,000 / 1,800,000 帧；0 gap / dup / OOO / cross-batch gap；avg batch read 113 ms |
| Level 1 (128ch / 60s) | 同上 `--n-channels 128` | **未跑** | 需要 4 个 headstage 都物理连上（见 §8）+ Demo 切到 4-headstage 配置 |
| Level 2 | `experiments/demo_pipeline_test.py` | **PASS** | 全 9 步通过；Steps 1-2 命令通道、3 协议生成、4 stim 配置 (80 cmd)、5 Run/Stop、6 µV 数据 (ts 连续)、7 三类特征、8 SNN forward、9 三种编码器 |
| SNN 预训练 (smoke) | `experiments/pretrain_snn_decoder.py --synthetic --n-epochs 10` | **PASS** | 240 trials × (40, 8) → 训练循环 + checkpoint I/O 通 |
| SNN 预训练 (500 epoch) | `--synthetic --n-epochs 500` | **PASS** | train/val acc = 100% / 100%（chance 33.3%）—— 训练循环可学合成可分数据 |
| SNN 预训练 (合成 classify 目录) | `--data-dir data/classify_synthetic_<ts>/` | **PASS** | 6 trials × 30s amplifier.dat → 解析 → 训练 → checkpoint；classify→pretrain 接口契约通 |
| Checkpoint 加载 (`run_with_dashboard` 路径) | `torch.load` + `SNNDecoder.load_state_dict` + forward | **PASS** | metadata 完整、模型 forward 输出正确 shape |
| 编码器电荷安全 | 100 张随机图 × 3 编码器 | **PASS** | 0/300 违例（trim 机制启用，见 §13） |
| **真 RHX 数据 → pretrain → loop 完整 dry-run** | 64ch / lif 编码 / 18 trials / 真 amplifier.dat | **PASS** | classify 18 trials × 3 classes 平衡，346 MB 真录音被 pretrain 正确解析、SNN 训练完成、checkpoint 加载、闭环 5 trials 跑通（latency ~400 ms/trial） |
| 真 RHX amplifier.dat 字节格式 | int16 × 64ch 交错、×0.195 µV | **PASS** | 真 RHX 输出 = 我们假设的格式，per-ch std 81.8 µV 与 stress test 完全一致 |
| TCP Set RunMode Record | 从 Stop 状态触发 | **PASS** | RHX 在 GUI 配置的 path + BaseFilename 下建子目录写 amplifier.dat |
| TCP Set RunMode Stop | 在 Record 中触发 | **不稳定** | 见 §4c / §4e — 需要 3s settle 才能下一次 Record，有时 Stop 完全不生效 |
| TCP Set Filename.Path | Windows 反斜杠路径 | **FAIL** | RHX 忽略，仍用 GUI 路径。已绕过：用 `find_new_rhx_subdir` 自动发现（见 §4d） |
| Level 3 | 真器官实验 | 未开始 | 见 `CHECKLIST_REAL_ORGANOID.md` |

`WaveformFrameReader` 已是 block-based 实现（`BLOCK_HEADER_SIZE = 4`、`SAMPLES_PER_BLOCK = 128`，每 block = 2564 字节 @ 8 通道）。

---

## 1. Windows PowerShell 终端打印 `µ` (U+00B5) 报错

**现象**
```
UnicodeEncodeError: 'gbk' codec can't encode character '\xb5' in position 73
```
出现在任何 `print(f"... µV ...")` 上，例如 `tests/test_waveform_format.py`、`experiments/stream_stress_test.py`、`experiments/demo_pipeline_test.py`、`experiments/realtime_loop.py`。

**根因**
Windows 中文系统 PowerShell 默认 stdout 编码是 `cp936/GBK`，无法编码 `µ`、`±` 等字符。Python 输出走 `sys.stdout.encoding = 'gbk'`。

**解决**
跑任何含 µ 输出的脚本前，先设环境变量：
```powershell
$env:PYTHONIOENCODING = "utf-8"
```
或单次命令前缀：
```powershell
$env:PYTHONIOENCODING="utf-8"; python experiments/stream_stress_test.py ...
```
（也可以在 PowerShell profile 里永久设置，或用 `chcp 65001` 把控制台切到 UTF-8。）

**涉及文件**：所有打印 µ 的脚本。**未改源码**（这是环境问题，不是逻辑问题）。

---

## 2. Intan RHX TCP 状态必须每次手动 re-arm

**现象**
脚本一旦断开（正常 disconnect、异常退出、Ctrl-C），下次启动立刻 `ConnectionRefusedError`。Intan RHX 端 TCP 状态显示从 `Connected` 回到 `Pending`。

**根因**
Intan RHX 的 TCP server 设计：每个 socket 只接受**一次**连接。客户端断开后，RHX 不会自动 listen — 必须在 GUI 里再点 **Connect** 把状态推到 `Pending`，下一个 client 才能 attach。

**解决（操作流程）**
每跑一次脚本之前必须做：
1. Intan RHX → **Network → Remote TCP Control**
2. **Command** 行点 **Connect**（状态：`Pending`，端口 5000）
3. **Waveform Output** 行点 **Connect**（状态：`Pending`，端口 5001）
4. 确认 "Channels To Stream" 里有 A-000 ~ A-007（或目标通道集）
5. RHX 自身的 `Run` 按钮可按可不按 — 脚本会发 `Set RunMode Run/Stop`

如果脚本异常 crash 中途、socket 半关闭，**也得 re-arm**。

**涉及文件**：操作惯例，无代码变更。

---

## 3. `demo_pipeline_test.py` Step 4：所有 80 条 stim Set 命令 timeout

**现象**
```
Step 4: Configure Stimulus Parameters via TCP
  Generated 80 commands
  [FAIL] Set A-000.stimenabled true: No response for command: ...
  [FAIL] Set A-000.source KeyPressF1: No response for command: ...
  ... (×80, 每条等 2s)
  Sent: 0 OK, 80 FAIL
```
总耗时约 160 秒。

**根因**
该 step 直接调 `client.send_command(cmd, timeout=2.0)`，`send_command` 默认 `expect_response=True`，会 `recv` 等回包。但 `tcp_client.py:312-316` 已经记录：

> VERIFIED: many Set commands do NOT echo a response.

RHS 的 `Set <channel>.<stim_param>` 系列**不会** echo。所以 80 条命令实际都发到了 RHX、RHX 也执行了，只是脚本死等不存在的回包。`tcp_client.py:548` 自家的 `send_protocol_config()` 用的是 `expect_response=False` 才对。

**解决**
`experiments/demo_pipeline_test.py:110-116` 改为：
```python
for cmd, desc in commands:
    try:
        client.send_command(cmd, expect_response=False)   # ← 关键
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] {cmd}: {e}")
        fail_count += 1
```
并配一行 comment 说明 RHX 协议事实，免得后人再踩。

**涉及文件**：`experiments/demo_pipeline_test.py` (Step 4 块)

---

## 4. `demo_pipeline_test.py` Step 5/6：`client.run()` 后 `Get RunMode` timeout

**现象**
Step 4 修好后 Step 5 暴露新问题：
```
Step 5: Start/Stop Data Acquisition
  [FAIL] Run/Stop: No response for command: Get RunMode
Step 6: ...
TimeoutError: No response for command: Get RunMode
```
注意 Step 2 同样的 `Get RunMode` 秒回。

**根因**
`client.run()` 默认 `verify=True`，先 `Set RunMode Run`（不等回包），紧接着 `Get RunMode` 读回包。但**刚发了 80 条 stim Set 命令**，RHX 命令通道处于 backlog 状态，此时立刻 Get 拿不到回应。`stream_stress_test.py:74` 早就发现这点，统一用 `client.run(verify=False)`。`demo_pipeline_test.py` 没沿用。

**解决**
Step 5 / 6 / Step 6 末尾三处 `client.run()` / `client.stop()` 全部加 `verify=False`：
```python
client.run(verify=False)
client.stop(verify=False)
```
基础的 `Get RunMode` 读路径在 Step 2 已经验证过，这里跳过 verify 完全合理。

**涉及文件**：`experiments/demo_pipeline_test.py` (Step 5、Step 6 顶部、Step 6 末尾)

### 4b. 同样的问题扩散到 `run_2d_experiment.py` 和 `realtime_loop.py`

后来跑 baseline phase 时 `client.record()` 又重演了同样的 timeout——`client.record()` / `run()` / `stop()` 三个方法的默认值是 `verify=True`，但项目里**所有实际工作的 caller 都用 `verify=False`**（stress test、demo pipeline 都是手动改的）。

**根因**：`client.set_filename()` 内部连发 2 个 Set（path + basefilename），紧接着 `client.record()` 立刻 Get RunMode——RHX 的命令通道还在 backlog 中。

**根本治疗**：直接把 `tcp_client.py` 里 `run` / `stop` / `record` 三个方法的默认改成 `verify=False`，让 API 默认值匹配实际使用惯例。Docstring 解释了原因。

涉及修改：`intan_interface/tcp_client.py`（3 个 def 默认值 + docstring）。

之后 `run_2d_experiment.py`、`realtime_loop.py` 等不需要改任何调用代码——它们用了默认值，现在默认值变成了正确的。

历史 caller（stress test、demo pipeline）显式传 `verify=False`，不受影响。

### 4c. TCP `Set RunMode Stop → Record` 竞态：必须 3 秒 settle

**现象**：跑 classify dry-run 时观察到 RHX 行为：
- pre-spontaneous Record → 写文件 `test_<ts1>/` ✓
- pre-spontaneous Stop ← 文件确实在 2 分钟整停止增长 ✓
- 紧接着主 classify Record → **无声丢弃，没有新目录**
- 主 loop 跑了 90s 但 RHX 没在录
- 主 Stop → no-op（RHX 早就在 Stop 状态）
- post-spontaneous Record → 写新文件 `test_<ts2>/` ✓（因为已经隔了 90s）
- post-spontaneous Stop → 文件 2 分钟整停止 ✓

**根因**：RHX 一次真录音 Stop 之后，**立刻**（< 1 秒）发的 Record 会被丢弃。Stop 似乎需要几秒去 finalize 文件、释放内部资源，期间 Record 是 no-op。

不丢弃的情况：
- 第一个 Record（RHX 从 idle Stop 起步）—— 一直工作
- Stop 之后等几秒再 Record —— 工作

**解决**：每次 `client.stop()` 后面如果还要 `client.record()`，**强制等 3 秒** settle：

```python
client.stop()
# ... 处理刚结束的录音 ...
time.sleep(3.0)  # 关键：等 RHX 完全释放上一次录音
client.record()
```

涉及修改：`experiments/run_2d_experiment.py` 的 `run_classify` 两处。

### 4d. RHX 在 Stop 之后**不**响应 `Set Filename.Path`，必须手动 Stop 才能开新会话

诊断结果（用 `Get Filename.Path` 回读验证）：
- RHX 内部以**正斜杠**形式存储 Path：`F:/YKX/Intan-main1/Intan-main/data`
- 我们 Python 用 `str(Path.absolute())` 发的是反斜杠路径，RHX **完全忽略**
- BaseFilename 同理——RHX 用 GUI 配置的值，TCP 改不动

**结果**：RHX 总是写到 GUI 配置的目录下、用 GUI 配置的 BaseFilename + 自己生成的 `_YYMMDD_HHMMSS` 时间戳建子目录。

**应对方案**（实现在 `run_2d_experiment.py`）：
1. 脚本不再调 `client.set_filename()`——RHX 反正不听
2. 加 `get_rhx_save_root(client)`：脚本启动时查 `Filename.Path`，知道 RHX 实际写到哪
3. 加 `find_new_rhx_subdir(rhx_root, before_snapshot)`：每次 record/stop 后做 `os.listdir` diff，找到 RHX 新建的子目录
4. 加 `attach_metadata_to_rhx_dir(...)`：把我们 Python 维护的 `config.json` / `trial_log.json` / `protocols.json` `shutil.copy` 到 RHX 子目录里，让下游 `pretrain_snn_decoder.py` 一个目录读完所有东西

`log_experiment(..., data=...)` 字段现在指向 RHX 子目录（而不是 Python 元数据目录），方便 `experiment_log.jsonl` 直接定位真数据。

### 4e. RHX TCP `Set RunMode Stop` 在 Record 中**有时**不生效（仅手动 GUI 可保险）

诊断脚本里观察到：

- `Set Record`（from Stop）→ Get 返回 "Record" ✓
- 接下来 `Set Stop` → 后续 Get 全部 timeout，**文件继续增长 11 分钟**直到我们手动 GUI 点 Stop

但 baseline 测试里同样的 `Set Stop` **又是**生效的（文件正好 20s）。

**当前理论**：Stop 在某些 RHX 内部状态下（例如刚 transition 不久、命令通道 backlog 等）会被丢弃；状态干净时正常。这个我们目前还没有可靠的 trigger 条件。

**当前 workaround**：
1. 每次跑脚本前用户**必须在 RHX GUI 手动点 Stop**（保险，确保起步状态干净）
2. 脚本内部依赖 §4c 的 3s settle，让连续 record→stop→record 大概率工作

未来要做：在 RHX 文档/源码里找 Stop 的真实状态机，或者用一个轮询 `Get RunMode` 验证 Stop 真生效的封装。

---

## 5. `demo_pipeline_test.py` Step 8：SNN 输入张量形状反了

**现象**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x40 and 8x3)
```
在 `decoders/snn_decoder.py:151` 的 `self.fc(spike_input[:, t, :])` 处。

**根因**
原代码：
```python
spike_input = torch.from_numpy(
    (np.abs(data_uv[:, :n_timesteps * 75]) > 50)
    .astype(np.float32)
    .reshape(n_input, n_timesteps, 75)
    .mean(axis=2)
).unsqueeze(0)  # 注释写的 (1, n_timesteps, n_input)
```
但 reshape `(n_input=8, n_timesteps=40, 75)` → mean(axis=2) → `(8, 40)` → unsqueeze → `(1, 8, 40)`。
**实际是 `(1, n_input, n_timesteps)`，与 `SNNDecoder.forward` 契约 `(batch, n_timesteps, n_input)` 正好相反**。注释和代码不一致，缺一个转置。

**解决**
显式分离两步、转置后再 unsqueeze，并改 comment 与契约对齐：
```python
binned = (
    (np.abs(data_uv[:, :n_timesteps * 75]) > 50)
    .astype(np.float32)
    .reshape(n_input, n_timesteps, 75)
    .mean(axis=2)
)  # (n_input, n_timesteps)
spike_input = torch.from_numpy(binned.T).unsqueeze(0)  # (1, n_timesteps, n_input)
```

**涉及文件**：`experiments/demo_pipeline_test.py` (Step 8)

---

## 6. 数据产物落盘位置

stress test 和波形诊断会写入 `data/` 子目录（不入 git，见 `.gitignore` 的 `*.dat / *.rhs`）：
- `data/stream_stress_<n>ch_<sec>s_<timestamp>.npz` — stress test 报告（含全部 timestamps、per-ch 均值方差、batch 耗时）
- `data/waveform_raw_diagnostic.bin` — `diagnose_waveform_format.py` 抓的原始字节，给 `locate_block_boundary.py` 用
- `data/realtime_<timestamp>/realtime_results.json` — 实时闭环每 trial 结果
- `data/<phase>_<timestamp>/` — `run_2d_experiment.py` 三阶段输出（baseline / probe / classify）

如果磁盘空间紧张，stress test 的 `.npz` 可以放心删；它纯属诊断快照。

---

## 7. 跑实验前的快速 precheck

按本顺序依次：

```powershell
# 0. 环境变量（每个新 shell 都要设一次，除非进了 profile）
$env:PYTHONIOENCODING = "utf-8"

# 1. 离线单元测试 — 不需要 Intan 在跑
python tests/test_waveform_format.py     # 期望: 9/9 PASS
python tests/test_stimulus_safety.py     # 期望: 全 PASS
python tests/test_integration.py         # 期望: 全 PASS

# 1b. SNN 训练循环 smoke (合成数据)
python experiments/pretrain_snn_decoder.py --synthetic --n-epochs 10 `
    --out-checkpoint checkpoints/snn_synthetic_smoke.pt
# 期望: 训练循环跑完，checkpoint 落盘

# 2. 启动 Intan RHX，Demo 模式 / RHS / 30 kS/s
#    Network → Remote TCP Control:
#      - Command port=5000, Connect → Pending
#      - Waveform Output port=5001, Connect → Pending
#      - "Channels To Stream" 至少 A-000..A-007

# 3. Level 1 — TCP 流稳定性
python experiments/stream_stress_test.py --duration 30 --n-channels 8
# 期望: VERDICT: STREAM STABLE — strict criteria met

# 4. Level 2 — 全流水线 (re-arm TCP 之后)
python experiments/demo_pipeline_test.py
# 期望: 9 步全 [OK]，结尾 "DEMO MODE TEST COMPLETE"

# 5. Level 3 — 真 record + pretrain + closed-loop dry-run
#    每次脚本前都必须：在 RHX GUI 点 Stop（黑方块）+ re-arm 两个 Connect

# 5a. baseline 真录 30 秒
python experiments/run_2d_experiment.py --phase baseline `
    --n-rec-channels 64 --spontaneous-duration 30
# 期望: 日志显示 "RHX wrote to: F:/.../data/test_<ts>" + amplifier.dat ~115 MB

# 5b. classify 主任务（lif 编码最稳，避开 rate / 边界 delayed_phase）
python experiments/run_2d_experiment.py --phase classify `
    --encoding lif --n-reps 2 --isi 5 --n-rec-channels 64
# 期望: 三段录音都拿到 RHX 子目录（pre / main / post）

# 5c. 离线 pretrain (不需要 RHX)
$LATEST_CLASSIFY = (Get-ChildItem data -Directory -Filter test_* | Sort-Object LastWriteTime | Select-Object -Last 3)[1].FullName
python experiments/pretrain_snn_decoder.py --data-dir $LATEST_CLASSIFY `
    --n-epochs 100 --out-checkpoint checkpoints/snn_dryrun.pt
# 期望: per-class counts 平衡，训练循环跑完，checkpoint 落盘

# 5d. closed-loop dry-run (RHX 再 re-arm 一次)
python run_with_dashboard.py --mode closedloop `
    --snn-checkpoint checkpoints/snn_dryrun.pt `
    --n-trials 5 --n-rec 64 --isi 3 --port 8050
# 期望: dashboard 起来在 8050、checkpoint 加载、5 trials 跑完、每 trial ~400ms latency
```

每个 Level 之间都要**回 RHX GUI 重新点 Connect**（见 §2）+ **手动 Stop**（见 §4e）。

---

## 8. 硬件拓扑：128 通道 = 4 × 32 ch headstage

Intan RHS Stim/Recording Controller 一台机最多 128 通道，物理上 = **4 个 32 通道 headstage**（端口 A / B / C / D），每个端口接一个 headstage。

| 端口 | 通道命名 | 用途 |
|------|---------|------|
| A | `A-000` ~ `A-031` | 默认 stim 走 A-000~A-007（与 `protocol_to_commands(channel_prefix="A")` 对应） |
| B | `B-000` ~ `B-031` | 仅记录 |
| C | `C-000` ~ `C-031` | 仅记录 |
| D | `D-000` ~ `D-031` | 仅记录 |

**对实验流程的影响**：
1. 跑 128 通道 stress test **必须 4 个 headstage 都物理插上**，否则 Intan 会缺通道；
2. RHX 的 "Channels To Stream" 要把 A-000..D-031 全勾上才能拿到 128 路数据；
3. 我们代码里 stim 只用 A 端口（仅 8 路），改更多 stim 通道需要 `protocol_to_commands` 支持多前缀；
4. Wang et al. 的设计也是少量 stim + 多 rec，128ch 主要是为了拿到更多 evoked response。

---

## 9. 后续可能踩到的坑（预警）

- **128 通道 stream**：`stream_stress_test.py` 在 strict 判据下要求 cross-batch gap = 0；64 通道已过（2026-05-11，PASS），128 通道未测。先把 `n_channels=128` 跑一遍 30–60 秒看 verdict（参考 §8 把 4 个 headstage 都插上）。
- **`run_2d_experiment.py` classify 阶段**：循环里直接 `client.send_protocol_config(proto)` → 立刻发 `Execute ManualStimTriggerPulse Fk`。`send_protocol_config` 内部已对每条 Set 用 `expect_response=False`，所以 Step 3/4 的问题不会重现。但**与 §4 同源**：刚发完一大批 Set 之后如果某处 Get 跟着，仍可能 timeout。如发现，沿用 `verify=False` 惯例。
- **真实刺激（live stim）**：`run_with_dashboard.py --live-stim` 需要在终端交互式输入 `ENABLE_STIM`，**所有非交互式 runner（CI、后台 nohup）都会跳过实刺激**。这是有意的安全门，别绕。
- **多通道触发 jitter**：`realtime_loop.stimulate` 串行发 8 次 `Execute ManualStimTriggerPulse Fk`，通道间会有 ms 级偏差。实操前用示波器测一下最大 jitter，超过协议容忍度（一般 200 ms 间隔下 5 ms 内没事）就需要换 TTL 单触发。
- **滤波器**：现在 `realtime_loop.detect_spikes` 直接对原始 µV 阈值。真器官信号建议在 **Intan 端**开 band-pass (300–3000 Hz) + notch (50/60 Hz)，省 CPU；如果不想改 RHX，就在 read 完之后用 `scipy.signal.sosfiltfilt` 加一道。
- **Intan 录文件格式**：`analyze_results.py` 和 `pretrain_snn_decoder.py` 都读 `amplifier.dat` (int16)。RHX 设置 → File Format 必须选 **"One File Per Signal Type"**，否则你只能拿到 `.rhs` 整包文件，离线分析跑不起来。
- **`utils/` 不再是空目录**：已有 `experiment_log.py`（§11）。

---

## 13. 编码器电荷预算 trim 机制

### 背景

`SAFETY_MAX_TOTAL_CHARGE_PER_TRIAL_NC = 200 nC` 是 schema 层硬安全限。但三个编码器原本谁都不知道这个上限：

| 编码器 | 默认参数下 max 电荷 | 100 张随机图违例率 |
|--------|------------------|--------------------|
| `delayed_phase` | 200 nC（贴线） | 0%（边界） |
| `lif` | ~150 nC | 0%（有余量） |
| `rate` | ~400+ nC | **100%（全爆）** |

`rate` 直接不可用，`delayed_phase` 不可预测会随机崩。

### 解决方案

在 `encoders/stimulus_schema.py` 加 **`trim_to_charge_budget(protocol, target_charge_nc)`**：

- 按 `(time_ms, channel)` 排序，从早到晚累加每个脉冲的双相电荷
- 累加超过 `target_charge_nc` 就停，丢掉之后的脉冲
- 删 pulse 不会破坏 min inter-pulse interval（只可能让间隔变大）
- 返回新建的、`validate()` 已通过的 StimulusProtocol

默认 `DEFAULT_TRIM_TARGET_NC = SAFETY_MAX_TOTAL_CHARGE_PER_TRIAL_NC × 0.9 = 180 nC`（10% headroom）。

### EncodingPipeline 自动应用

`EncodingPipeline.encode_to_protocol()` 内部在编码器输出后**始终**调用 `trim_to_charge_budget`，并在**第一次**真正裁剪时打印一行通知：

```
[EncodingPipeline] trimmed 19 pulse(s) (37->18) to fit 180 nC budget (rate). Further trims will be silent.
```

之后同实例的裁剪静默。这样：
- 用户能知道自己的编码器在被裁剪、裁了多少
- 不会被刷屏

可关：构造 `EncodingPipeline(trim_target_nc=None)` 跳过裁剪（仅研究用——结果可能 fail `validate_safety`）。

### 验证结果

100 张随机图 × 3 编码器 = **300 个协议，0 违例**：

| 编码器 | avg pulses | avg charge | max charge | 违例 |
|--------|-----------|-----------|----------|------|
| `delayed_phase` | 17.4 | 173.5 nC | 180.0 nC | 0/100 |
| `lif` | 10.9 | 109.0 nC | 180.0 nC | 0/100 |
| `rate` | 18.0 | 180.0 nC | 180.0 nC | 0/100 |

全 6 个测试套件回归通过（test_stimulus_safety / test_waveform_format / test_feature_extraction / test_integration / test_encoders / test_decoders）。

### 早期裁剪 = 保留最强特征

按时间早→晚的顺序保留脉冲，对每个编码器的语义：

- `delayed_phase`：arctan firing time 公式下，**早 = 高激活**。保留早期脉冲 = 保留最强特征 ✓
- `lif`：膜电位积分驱动，spike 时间近均匀分布。早期 vs 晚期信息量相当
- `rate`：Poisson 过程，时间均匀。早期一段就够代表信号

不是最优策略（最优需要按"每脉冲信息量"打分），但比硬限丢任意脉冲好，也比强制改编码器参数好。

---

## 10. SNN decoder 预训练 → 实时 loop 加载

实时 loop 默认 `SNNDecoder` 随机权重 → 预测无意义。完整闭环工作流：

```powershell
# 1) 收数据（Intan RHX 端先设 "One File Per Signal Type"）
python experiments/run_2d_experiment.py --phase classify --encoding delayed_phase
# -> data/classify_<ts>/{amplifier.dat, trial_log.json, config.json, ...}

# 2) 预训练 SNN
python experiments/pretrain_snn_decoder.py `
    --data-dir data/classify_<ts>/ `
    --out-checkpoint checkpoints/snn_<ts>.pt
# -> 输出 train/val acc，存 checkpoint

# 3) 用 checkpoint 闭环
python run_with_dashboard.py `
    --mode closedloop `
    --snn-checkpoint checkpoints/snn_<ts>.pt `
    --n-trials 50
```

`--snn-checkpoint` 没给时仍可跑（保留对照实验能力），但启动时会打印 `[WARN] No --snn-checkpoint given. SNN has random weights; predictions are meaningless until trained.`。

**Sanity check**（不需要任何 Intan 数据）：
```powershell
python experiments/pretrain_snn_decoder.py --synthetic --n-epochs 10 `
    --out-checkpoint checkpoints/snn_synthetic_smoke.pt
```
跑过即说明训练循环、checkpoint I/O、SNN forward/backward 全通。

`checkpoints/` 目录会自动建，已加入 `.gitignore`。

---

## 11. 实验日志机制（每次实验自动留痕）

所有实验脚本结束时**自动**追加一条记录到仓库根目录的 `experiment_log.jsonl`。一条 = 一行 JSON，append-only，永不重写。

**记录字段**：
```json
{
  "ts":       "ISO 8601 本地时间",
  "script":   "脚本文件名",
  "args":     { ...完整 argparse 参数... },
  "verdict":  "PASS / STREAM STABLE / COMPLETE / FAIL: <error>",
  "summary":  { ...关键指标，每个脚本不同... },
  "data":     "细节数据落盘路径"
}
```

**接入脚本**：

| 脚本 | verdict | 关键 summary 字段 |
|------|---------|---------------|
| `stream_stress_test.py` | STREAM STABLE / PASSABLE / FAILED | `total_frames` / `loss_rate_pct` / `n_gaps` / `avg_batch_read_ms` |
| `demo_pipeline_test.py` | PASS / FAIL | `error`（仅失败时） |
| `pretrain_snn_decoder.py` | PASS | `mode` / `n_classes` / `final_train_acc` / `final_val_acc` / `chance_level` |
| `run_2d_experiment.py` (baseline) | COMPLETE | `phase` / `duration_s` / `n_rec_channels` |
| `run_2d_experiment.py` (probe) | COMPLETE | `phase` / `n_channels_probed` / `n_pulses_total` |
| `run_2d_experiment.py` (classify) | COMPLETE | `phase` / `encoding` / `n_classes` / `n_trials_total` |
| `realtime_loop.py` | COMPLETE | `n_trials_done` / `accuracy` / `latency_mean_ms` / `total_charge_nc` |
| `analyze_results.py` | COMPLETE | `feature_method` / `acc_mean_<decoder>` × N |

**查看日志**：

```powershell
python utils/experiment_log.py --tail 10              # 最近 10 条，单行格式
python utils/experiment_log.py --tail 5 --full        # 最近 5 条完整 JSON
python utils/experiment_log.py --filter script=stream_stress_test.py
python utils/experiment_log.py --filter verdict=PASS
python utils/experiment_log.py --path                 # 打印 jsonl 文件路径
```

**`utils/experiment_log.py` 同时也是个 Python API**，新脚本里：

```python
from utils.experiment_log import log_experiment

log_experiment(
    script="my_new_experiment.py",
    args=vars(args),
    verdict="PASS",
    summary={"key_metric": 0.92, "n_samples": 100},
    data="data/my_run_<ts>/",
)
```

**保留位置 / 备份**：
- `experiment_log.jsonl` 在仓库根，**不** 被 `.gitignore` 排除——它本身是元数据，定期 commit 进 git 比丢盘里好（即使现在还不是 git repo）
- 每个实验的**细节数据**仍在 `data/<...>/` 下（npz、json、log、amplifier.dat 等），日志只是个**索引**

---

## 12. 修改 changelog

| 日期 | 文件 | 改动 |
|------|------|------|
| 2026-05-11 | `experiments/demo_pipeline_test.py` | Step 4：stim Set 用 `expect_response=False`（§3） |
| 2026-05-11 | `experiments/demo_pipeline_test.py` | Step 5/6：`run/stop` 用 `verify=False`（§4） |
| 2026-05-11 | `experiments/demo_pipeline_test.py` | Step 8：补 `.T` 修正张量形状（§5） |
| 2026-05-11 | `experiments/pretrain_snn_decoder.py` (新增) | SNN 预训练脚本，支持 real recording + `--synthetic`（§10） |
| 2026-05-11 | `run_with_dashboard.py` | 新增 `--snn-checkpoint`；不指定时仍可跑、附 WARN（§10） |
| 2026-05-11 | `.gitignore` | 加 `checkpoints/` |
| 2026-05-11 | `utils/experiment_log.py` (新增) | 实验日志机制：`log_experiment` API + CLI 查看器（§11） |
| 2026-05-11 | `experiments/{stream_stress_test, demo_pipeline_test, pretrain_snn_decoder, run_2d_experiment, realtime_loop, analyze_results}.py` | 末尾接入 `log_experiment` 调用（§11） |
| 2026-05-11 | `encoders/stimulus_schema.py` | 加 `trim_to_charge_budget()` + `DEFAULT_TRIM_TARGET_NC=180` 常量（§13） |
| 2026-05-11 | `encoders/pipeline.py` | `EncodingPipeline.__init__` 加 `trim_target_nc` / `trim_warn` 参数；`encode_to_protocol` 末尾统一裁剪并首次裁剪时打印通知（§13） |
| 2026-05-11 | `encoders/__init__.py` | 导出 `trim_to_charge_budget` |
| 2026-05-11 | `intan_interface/tcp_client.py` | `run` / `stop` / `record` 默认 `verify=False`（§4b）—— 与项目实际惯例一致 |
| 2026-05-11 | `experiments/run_2d_experiment.py` | 加 `--n-rec-channels` 和 `--spontaneous-duration` CLI 参数（允许覆盖 DEFAULT_CONFIG） |
| 2026-05-12 | `experiments/run_2d_experiment.py` | RHX 子目录自动发现：加 `get_rhx_save_root` / `find_new_rhx_subdir` / `attach_metadata_to_rhx_dir`（§4d）；删除 `set_filename` 调用；run_classify 加 3s settle（§4c）；log_experiment 的 data 字段指向 RHX 真目录 |
