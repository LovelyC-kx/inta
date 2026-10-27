# 工程修改记录 / Engineering changes

本文件区分软件回归与现场验证；离线测试通过不等于电极端刺激或生物任务已验收。
Offline regression is not an electrical or biological acceptance test.

## 1. RHX session and stimulation foundation

- 默认 `execution_mode="acquire_only"`；读取允许，刺激 Set/Upload/Trigger 被拒绝。
- `simulate` 不打开硬件 socket；`play_protocol_in_time(..., upload_to_hardware=False)`
  仅返回 `sent=False` 事件，连已有非零硬件参数也不会被触发。
- 实刺激显式使用 `execution_mode="live"`。先 Stop，预装完整会话的固定波形 profile，
  等待 UploadInProgress=False 并回读，再 Run/Record；运行时只执行事件时序。
- 所有已连接 RHS amplifier 通道先禁用，避免其他 headstage 的旧 F-key 路由同时触发。
- 拒绝负数/NaN/Inf、不支持的波形和极性；同通道无法在会话内静默改变波形参数。
- 感觉和反馈共用跨调用的200ms间隔和滚动2秒电荷预算；沿用现有上限，不扩大参数范围。
- Get 追加有序 RunMode 查询作为响应边界，支持没有换行的响应；线程共享锁避免争抢。
  超时后关闭 command 连接，防止迟到回复被当成下一请求的结果。
- 前端单次刺激使用公共执行器并支持取消；状态/控制请求在刺激操作中返回busy。

```python
from intan_interface.tcp_client import IntanTCPClient

client = IntanTCPClient(execution_mode="live")
client.connect_command()
client.stop()
client.wait_for_run_mode("Stop")
client.prepare_protocols([sensory_protocol, feedback_protocol])
client.record()
client.wait_for_run_mode("Record")
client.play_prepared_protocol(sensory_protocol)
```

上面例子需要预先定义、标定并验证协议。实际事件时间仍需同一采样时钟的STIM/TTL marker；
返回的 `actual_ms` 仅代表主机发送调度，不是电极电流起始时间。

The client now defaults to acquisition only. Live output requires explicit live mode and a
stopped-board upload with readback. Prepare the union of sensory/feedback channel profiles
once, then schedule all pulses during continuous acquisition. Host scheduling timestamps are
not measurements of electrode current onset.

依据：随库 `docs/IntanRHX_TCPDocumentation.pdf`（RHX 3.5.0），Stimulation Parameters、
UploadStimParameters、RunMode；官方 `GUI/Widgets/tcpdisplay.cpp::parseCommands` 和
`Engine/Processing/commandparser.cpp::getRunModeCommand`。

兼容性：旧脚本直接在Run状态重新上传、跳过上传却仍触发、或用默认client发送刺激，
会被拒绝。请使用上面的session顺序；不要仅为了恢复旧行为而绕过这些检查。

仍待现场验证：RHX版本、controller通道映射、电流步长和量化、负载波形、实际marker时延、
电极/组织响应、模拟分包测试之外的真实TCP表现。

## 2. RHX 5002 events and hardware-time windows

- 公共 `SpikePacketParser` 保留native name、uint32原始时间、展开后的sample index、spike ID、
  主机单调接收时间和acquisition epoch。A-000/B-000不会混合；0x80伪迹默认不参与窗口。
- `SpikeStreamReader` 只有一个socket读取线程，支持原始二进制留存。EOF、溢出、损坏字节、
  时间倒退和磁盘错误都使采集无效，不会补零或继续正常控制。
- `build_spike_window` 使用半开硬件采样窗口和逐电极/全局伪迹区间；保留有效观察时长。
  窗口结束需由同epoch的硬件进度marker确认；主机时间或最新稀疏spike不能替代marker。
- 阈值演示已迁移到公共接收器。默认只采集；跨headstage请显式指定完整名字：

```bash
python experiments/spike_triggered_stim_demo.py --channels A-000 B-000 --dry-run
```

演示保存 `spikes.raw`、`events.jsonl`、`session.json`。它只是接口/负载工程工具，
不是分类或Pong学习实验。live bench模式需要 `--live-stim`，先Stop配置再Run；未接实际
marker的演示不具备电极时延证明，也不具备逐刺激marker校准的生物反馈伪迹屏蔽。

The shared receiver preserves full event identity and fails explicitly on acquisition errors.
Hardware-time extraction requires a matching progress marker, retains raw event timing and
never clips out-of-window spikes into the last bin. The threshold demo now uses this receiver;
its live option is a bench-interface tool, not a validated biological feedback experiment.

下一批：把5001时钟/STIM marker接入连续分类主入口，统一真实事件训练与在线窗口。
旧 `realtime_openloop.py` / `realtime_loop.py` 的标签、分窗及机械臂任务整合尚未完成；
第一批的模式检查会阻止它们直接沿用旧的未确认触发路径。

## 3. Digital stimulation record analysis

- 按官方RHS reader解析uint16位域。amp settle/charge recovery/compliance位不算电流。
- 识别两个相反极性的phase为一个双相pulse，支持≤5ms的interphase gap，并报告缺相、
  电荷码不平衡、幅度变化和compliance limit。
- 没有实际电流step时不换算nC；提供step后按幅度码与采样时间积分计算。
- time.dat存在时校验样本数并保存原始RHX时间戳。文件截断或通道数量不符直接失败。
- 数量不匹配明确标成STIM_COUNT_MISMATCH；即使数量匹配也只标
  STIM_RECORDED_COUNT_MATCH，`physical_delivery_verified`始终为false。
- 前端使用RHX实际保存路径，只分析本次唯一新录音；禁止回退到最近的旧录音。
- 默认pytest只收集tests/，不会导入会连接实验硬件的experiments/test_5002_*.py。

Digital status bits no longer count as current, and two phases count as one pulse. Actual
current step size is required for charge conversion. Count matching is explicitly separated
from electrical delivery verification; old or ambiguous recording directories are rejected.

依据：官方 `Engine/Processing/DataFileReaders/filepersignaltypemanager.cpp` 的位域解码；
随库RHX 3.5.0 TCP文档规定InterphaseDelayMicroseconds为0..5000µs。

## 4. Review against the user-supplied RHX manuals

详见 [手册逐项复审与剩余缺口](RHX_MANUAL_REVIEW_2026-10-26.md)。

- 上传前查询 RHS 类型、版本、Demo/Playback、采样率、电流步长及扩展板状态。
  幅度必须为 1..255 个电流步，相宽/间隔必须为整数采样步；拒绝静默量化。
- Stop/Run/Record 按已确认状态转换；重复 Stop 不再向 RHX 发送会报错的命令。
- 所有 amplifier、模拟/数字刺激路由禁用状态在上传前核实，防止旧 F-key 配置一起触发。
- 通道 settle/recovery 字段明确设置，另上传全局 settle 配置并等待完成。
- Stop 配置事务保存中间态 warning；最终完整字段回读与所有禁用状态通过后才上传。
  Error 始终失败；加载配置、重扫、相关全局设置变更和 RHX 错误撤销触发缓存。
- 5001 验证 magic、块内/块间连续时间，支持 uint32 回绕；EOF/总超时不再返回短窗口。
  当前解析器仅适用于 amplifier voltage words，不能用于 STIM/TTL/DC 混合布局。
- 前端保存 rhx_session.json 和 trial_log.json，用真实采样率/电流步长分析本次录音；
  time.dat 跳变、参数冲突均失败。保存通道名映射仍待从 info.rhs 接入。

English: manual review adds RHS representability checks, idempotent state transitions, complete
output-route verification, explicit settle settings, validated configuration transactions, waveform
integrity checks, and recording metadata binding. Typed STIM/TTL alignment and real task integration
remain pending; see the linked review for source pages and acceptance requirements.

验证 / Validation: **150 offline tests passed**, including 35 new regressions; no hardware test.

## 5. Continuous open-loop owner

新增统一入口 [run_openloop.py](../experiments/run_openloop.py)，启动说明见 [OPENLOOP.md](OPENLOOP.md)。

- 任意 PIL 图片/二维 numpy 灰度图通过清单进入同一 CNN/编码器；实际刺激参数先覆盖、再按电荷裁剪。
- 5001 typed STIM/可选TTL、5002原生RHX spike独立接收；公共收集线程持续推进，刺激和机械臂不会占用接收线程。
- 数字脉冲各相与预装profile核对；按真实marker取窗口并应用逐脉冲遮罩。
- 采集、LR/MLP训练、冻结JSON模型、真实推理和原始字节回放使用同一特征契约。
- 明确组别留出，scaler/模型仅拟合训练集；保留字符串类别，不再混淆数字标签和输出下标。
- 机械臂任务单独线程运行，支持动作序列及置信度/忙碌/重复/过期拒绝；故障可取消执行中任务。
- 组织编号、CNN hash、刺激profile/端口、通道顺序、采样率、RHX检测设置和特征窗口均参与模型匹配。
- 模拟模型不能用于物理后端；实机机械臂还要求当前与训练数据具有明确的时序测量声明。
- 单个新输入只使用部分通道时仍预装训练profile bank，配置行不会被额外执行。
- 原生5002无逐块心跳，因此实机采用显式cross-socket holdback假设；迟到事件、积压/停滞、
  marker不符或断连会使会话失败。不能将该假设写成硬件已经提供完成证明。
- 旧实时/图片刺激CLI已停用并指向新入口；浏览器连续实验控制和行为结果反馈尚未加入。

完整模拟演练已走通图片→原始字节→训练/测试→冻结推理→机械臂模拟→原始回放。
真实后端的socketpair/fake RHX测试包含持续接收、marker缺失、spike迟到/断连、流停滞和任务适配。
所有验证仍为离线软件验证，未进行实机实验；最新测试数量以本轮提交说明为准。

本轮验证：**200 offline tests passed**, including 50 new tests; standalone CLI demo and raw replay passed.
