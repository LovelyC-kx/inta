# RHX 手册复审 / Manual-based review

本轮在用户新增 PDF 后复审前三批修改，基于 `568a5c9`。所有测试均为离线模拟或本地
socketpair；没有连接 Intan、电极、组织或机械臂。软件检查不代表实际电流、延迟或生物任务验收。

## 1. 文档身份和证据范围

- 主依据：[根目录 TCP Guide](../IntanRHX_TCPDocumentation.pdf)，封面标明 **3.5.0**，64 个 PDF 页面。
- 主依据：[根目录 User Guide](../Intan_RHX_user_guide.pdf)，**9 January 2026** 更新，35 页。
- 与 `docs/` 副本比较：两份 User Guide 的 SHA-256 相同；TCP Guide 二进制不同，
  但 `pdftotext -layout` 提取的全部正文逐字相同。它们不是两个不同版本的 TCP 协议。
- 下文页码为文档**印刷页码**。TCP Guide 有封面，PDF 页面索引比印刷页码大 1。
- 实现补证：[Intan 官方源码固定提交](https://github.com/Intan-Technologies/Intan-RHX/tree/3c5867718d279af15965e6867cb657f7bca49c71)，
  源码头标记 **3.5.2**。源码用于补充手册未写清的行为，不等于已经在 3.5.0 二进制实测。

本轮聚焦命令、运行/上传状态、刺激硬件表达能力、输出路由、spike/波形传输、录音验收。
显示、音频、谱图等界面参数没有逐一实现自动化。

## 2. 发现、依据和已完成修复

| 问题 | 证据 | 本轮修改 |
|---|---|---|
| 软件回读的 µA/µs 未必等于硬件可表达值 | TCP pp.6-8、54；User Guide pp.4、24；`ControllerInterface::setStimSequenceParameters` 使用 `round()` | 上传前读取 RHS 类型、采样率、电流步长等；检查幅度为 1..255 个电流步、相宽和间隔为整数采样步；不隐式取整 |
| 已 Stop 时再次 Stop 会产生错误，污染后续回复 | 官方 `CommandParser::setRunModeCommand`；TCP p.18 要求确认状态改变 | `stop()` 和同状态的 Run/Record 幂等；Run 与 Record 之间需先明确 Stop；开始采集前等待 UploadInProgress=False |
| 只禁用 amplifier，遗漏其他输出的旧 F-key 路由 | TCP pp.55-59、61；User Guide pp.25-26；官方 SignalSources/AbstractRHXController | 按 ExpanderConnected 枚举并禁用模拟/数字刺激输出，逐个回读；缺失通道在任何配置写入前拒绝 |
| amp settle、charge recovery 参数继承 GUI 残留值 | TCP pp.53-55、60-61；User Guide pp.24-25 | 明确设置并回读通道时序；全局 amp settle 单独上传、等待完成，再上传刺激参数；保存所用全局设置 |
| 每个 Set 的中间态 warning 可能使最终有效配置失败 | 官方 `CommandParser::setCommandSlot`、`StimParameters::validate` | 仅在 Stop 配置事务结束时收集 warning 并保留原文；Error 始终失败；完整配置和所有禁用状态回读通过后才上传 |
| acquire_only 可通过加载 XML 间接改变刺激，旧缓存可能失效 | TCP pp.62-63 的 LoadSettingsFile/LoadStimulationSettingsFile | 非 live 模式禁止这些路径及全局刺激设置；加载、重扫、全局设置/上传变更和 RHX 错误均撤销已验证触发缓存 |
| 5001 无 magic/连续性检查，EOF 可返回短窗口 | TCP pp.3-4；官方 `TCPDataOutputThread::run` | 每块校验 magic、逐采样/跨块连续性，允许 uint32 自然回绕；损坏后解析器失效；短读/超时关闭连接；使用总截止时间 |
| 前端录音分析仍使用默认 30 kHz，电荷没有绑定本次 step | TCP pp.6-7；User Guide p.24；实际前端调用链 | 使用本次查询的采样率与电流步长；保存 rhx_session.json、trial_log.json；检查 time.dat 连续性；配置冲突直接失败 |
| 旧注释仍声称非零 stim.dat 就证明真实刺激 | 官方位域定义；User Guide pp.27-28 | 修正文档措辞，保持 physical_delivery_verified=False；数码记录只证明控制器记录，不能代替电极负载测量 |

### 2.1 硬件量化检查的实际含义

新文件：[`hardware_profile.py`](../intan_interface/hardware_profile.py)。

硬件时间步为 `1e6 / sample_rate_hz` µs，电流范围为 `255 * stim_step_ua`。
例如：

| 参数 | 可否原样表达 | 处理 |
|---|---|---|
| 30 kHz，500 µs/phase，0.5 µA/step，10 µA | 15 samples、20 current steps | 允许 |
| 25 kHz，500 µs/phase | 12.5 samples | 在 Set/Upload 之前拒绝；需显式选择 480 或 520 µs 等可表达值，并重新检查电荷 |
| 0.5 µA/step，10.1 µA | 20.2 current steps | 拒绝 |
| 0.02 µA/step，10 µA | 500 current steps，超出 255 | 拒绝 |

这里只检查脉冲**相宽、相间隔和幅度**的硬件表达能力。`time_ms` 仍为主机调度目标，
没有被提升为采样精度的硬件触发保证。RHSProfile 和每通道的幅度码/样本数保存在
`client.prepared_session` 中。Synthetic 标记会被保留；Demo 仍可用于接口测试。
Playback 被拒绝，RHD recording-only controller 也不能使用此电流刺激路径。

### 2.2 明确的第一版 settle/recovery profile

当前固定通道 profile 使用：

| 字段 | 值 |
|---|---|
| EnableAmpSettle | True |
| MaintainAmpSettle | False |
| PreStimAmpSettleMicroseconds | 0 |
| PostStimAmpSettleMicroseconds | 1000 |
| EnableChargeRecovery | False |
| PostStimChargeRecovOn/OffMicroseconds | 0 / 0 |
| RefractoryPeriodMicroseconds | 1000 |
| PulseOrTrain / NumberOfStimPulses | SinglePulse / 1 |

这些明确恢复 RHX 默认通道设置，不声称 1 ms 已消除组织实验的刺激伪迹。
全局 UseFastSettle、HeadstageGlobalSettle 和 lower-settle bandwidth 保留操作者所选值，
上传后重新查询并保存。需要改变这些实验变量时应通过显式 session 配置扩展。
ChargeRecovery=False 是本版本固定 profile 的边界，不是对长期组织实验最优参数的结论。

禁用模拟/数字刺激路由的范围只保证它们不会因本执行器的 F1..F8 同时启动旧脉冲。
它不证明所有模拟/数字物理端口为零电平，也不包含外部设备状态验收。

### 2.3 中间态 warning 与最终校验

RHX 在每个相关 Set 后检查依赖关系，即使正在 Stop 状态下分多条命令组装 profile。
例如，旧 post-settle 很长时，先降低 refractory 会得到 warning；后续设置可能使最终配置有效。
本实现保存中间态 warning，而不将其当作已经通过验证；最终字段全部回读匹配、
unused outputs 全部确认禁用后才执行 Upload。

`Error:` 不被忽略，普通 Get/运行路径中的 warning 仍使操作失败。
测试模拟器也改为返回官方的裸字段名，如 `Return: StimEnabled True`，并拒绝错误 RunMode 转换。

## 3. 本轮验证

关键回归场景包括：错误 controller/playback、NaN 步长、25 kHz 半采样相宽、
幅度非整数码或超量程、无/有扩展板的所有输出禁用、禁用命令静默丢失、
重复 Stop/Run、Run→Record 非法转换、配置 warning 与 Error、间接加载配置、
缓存失效、magic 损坏、时间跳变、自然回绕、短 EOF、总超时、实际采样率下的电荷计算。

运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

本轮全套离线回归：**150 passed**，较前一批 115 项新增 35 项。没有实机结果。

## 4. 下一轮必须接通或补证的事项

| 顺序 | 具体事项 | 依据与验收要求 |
|---|---|---|
| 1 | 支持 5001 的 typed STIM/TTL layout 和公共采样时钟 | 当前 WaveformFrameReader 仅支持 WIDE/LOW/HIGH amplifier voltage words；DC/STIM/数字/模拟端口的布局和缩放不同。不能把同宽 uint16 都当成 µV。需要 typed layout、逐脉冲 marker，并与 5002 epoch 对齐 |
| 2 | 处理 5001/5002 的跨 socket 到达差与完整性 | 同一硬件时间不代表两个 TCP socket 同时送达。progress marker 不能单独证明该窗口的 spike 已全部到达；需要实测延迟界限和回放/对账验收 |
| 3 | 固定并保存 RHX spike 检测契约 | User Guide pp.7、10、14-15，TCP pp.25-28、40-41、49-52：SPK 来自 HIGH 阈值检测，受滤波、参考、阈值和 artifact suppression 影响。现有 prepared_session 不是完整解码模型契约；真实训练前必须保存并匹配这些设置 |
| 4 | 从 info.rhs/保存通道表取得录音列顺序 | 当前 stim.dat 分析仍要求明确 n_channels。time.dat 能发现数量不符，但不能单独证明每列 native channel identity；不能将录音列号直接等同于 A-xxx |
| 5 | 连续负载和时延实测 | TCP pp.3、14；User Guide pp.31-32 明示 backlog。10 blocks 在 30 kHz 的攒批量为 42.67 ms，并非端到端最大延迟。评估小 batch、队列增长、持久化与组织响应，不能直接宣称 10 ms 实时闭环 |
| 6 | 电极端的幅度、相宽、伪迹与 compliance | User Guide pp.24-29：全局 settle 可能影响同 headstage 邻道；charge recovery 本身会有伪迹；compliance 表示可能无法输出要求的电流；DC 是粗略电压指示。仍需负载测量和本电极条件标定 |
| 7 | 开环分类主入口、冻结解码器与机械臂任务 | 将上述有效性和 marker 接入实时分类，统一真实 RHX events 的训练/在线处理，再统计独立任务结果；旧实时脚本尚未完成该迁移 |

`CurrentTimestamp` 和 LiveNotes 都不能替代逐脉冲硬件 marker：TCP Guide p.18 明示
CurrentTimestamp 有 USB/TCP 延迟；p.6 的 LiveNotes 只给 0.1 s 精度的记录时间。
本轮没有把 command 查询作为精确刺激时间源，也没有将主机触发记录当成电极测量。

## 5. English record

The supplied TCP PDFs contain identical extracted RHX 3.5.0 text despite different binary hashes.
The User Guide copies are byte-identical and were updated on 9 January 2026. Intan source at the
pinned commit is labeled 3.5.2, so source evidence is distinguished from a 3.5.0 hardware test.

This revision adds pre-mutation RHS/current/timing representability checks; idempotent run-state
transitions with upload waits; disable/readback of inherited amplifier, analog and digital stimulation
routes; explicit settle/recovery fields and a separate global settle upload; retention of transient
configuration warnings with mandatory final readback; indirect configuration guards and cache
invalidation; waveform magic, continuity and deadline checks; and recording analysis bound to the
queried sample rate/current step, including timestamp continuity and session sidecars.

Off-grid values are rejected rather than silently rounded. Demo status remains explicit, playback
cannot execute live stimulation, and digital pulse records never certify physical delivery.
All **150 offline tests passed** (35 added since the previous 115-test baseline).
Tests use fake RHX sockets, synthetic bytes and local socketpairs only.

Still pending: typed STIM/TTL acquisition and 5001/5002 clock alignment; cross-stream completeness;
the filter/reference/threshold contract for real RHX spikes; saved native channel ordering from the
recording header; measured latency/backlog and electrode artifacts; and integration into the
classification-to-arm task. Neither host scheduling nor CurrentTimestamp/LiveNotes establishes
electrode onset timing or completion of the biological control loop.
