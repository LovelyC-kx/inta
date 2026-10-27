# 连续开环的统一入口

入口：`python experiments/run_openloop.py`。需要 Python 3.10+ 和项目依赖。

本版本已把图片编码、完整刺激调度、5001 STIM/可选 TTL、5002 RHX spike、统一分窗、
冻结解码器和机械臂任务接进同一程序。新增的实机后端经过本地 socketpair/fake RHX 测试；
**尚未连接真实 Intan、电极、类器官或机械臂验证。**

## 1. 先运行完整软件演练

在仓库根目录：

```bash
python -m pip install -r requirements-dev.txt
python experiments/run_openloop.py demo --out data/openloop_demo_01
```

这个命令自动生成示例图片，然后执行：

1. 图片 → CNN 特征 → 刺激协议；
2. 协议 → 模拟的 5001/5002 原始字节 → 同一套真实格式解析器；
3. 12 个训练输入、6 个明确留出的测试输入 → LR 冻结模型；
4. 冻结模型 → 6 次机械臂模拟任务；
5. 重新解析保存的原始字节 → 重建窗口/特征 → 核对回放结果。

预期 `collection=COMPLETE`、`inference=COMPLETE`、`replay=REPLAY_MATCH`。
模拟神经响应只由脉冲协议产生，不读取标签；它是接口夹具，不是生物模型。
演练中的准确率没有生物学或算法性能含义，示例使用 `min_confidence=0` 仅为检验动作路径。
输出目录必须是新目录，程序不会覆盖已有实验或模型。

## 2. 准备真实输入和参数

### 输入清单

参考 [`manifest.example.json`](../examples/openloop/manifest.example.json)。示例图片路径需要换成你自己的文件。
支持 PIL 可读图片及 `.npy`：普通图片转灰度并按 CNN 输入尺寸缩放；numpy 文件必须是
二维、有限的 `[0,1]` 数值。当前不是视频/音频等模态的统一编码器。

```json
{
  "trials": [
    {"trial_id": "input-001", "image": "images/zero_a.png", "label": "0", "group": "training"},
    {"trial_id": "input-002", "image": "images/one_a.png", "label": "1", "group": "heldout"}
  ]
}
```

相对图片路径以清单文件所在目录为基准。`trial_id` 必须唯一；类别使用字符串，
例如 `"0" / "1" / "7"`，不会暗中变成另一套 `[0,1,2]` 标签。
正式推理时可省略 `label`，决策从神经特征产生，标签不进入刺激后端或机械臂策略。
`group` 省略时使用预处理后的图片 hash；指定训练/测试组时，不允许完全相同图片跨组泄漏。
示例的少量图片用于说明格式，不能作为有统计说服力的数据量。

### 参数文件

- [`features.json`](../examples/openloop/features.json)：默认从第一个数字 STIM onset 开始，观察 2500 ms，
  50 bins；每个脉冲对所有记录电极应用前 1 ms、脉冲结束后 10 ms 的伪迹遮罩。
  这些遮罩时长是显式实验参数，尚未证明足以消除本电极/组织的伪迹。
- [`pulse_profile.json`](../examples/openloop/pulse_profile.json)：电流、相宽、相间隔、极性和波形。
  修改先作用于脉冲，再按实际电荷裁剪；之后执行安全与硬件量化检查。
  例如 25 kHz 的 500 µs 不可原样表示，需要你显式选择合适相宽并重新标定。
- [`timing.example.json`](../examples/openloop/timing.example.json)：跨 socket 延迟、流停滞/积压、
  实际数字刺激时序误差的上界。**示例数值没有实测依据，`validated` 保持 false。**
- [`actions.json`](../examples/openloop/actions.json)：每个模型类别映射到一个动作或有序动作序列。
  例如 `"7": ["grab", "release"]` 是预编程宏任务；不能据此声称类器官独立规划了所有子动作。

`--preparation-id` 是组织/培养物身份，需要在该模型的训练和运行中一致。
默认刺激映射仍是所选端口的通道 000..007，最多 8 路独立 F-key profile；
记录通道通过 `--channels` 单独指定完整 native names。跨端口记录不会合并通道。

## 3. 第一轮真实采集：先不做动作分类

先在 RHX 选择实际 RHS controller、采样率、电流步长和记录通道，完成电极/负载标定，
设置参考、HIGH 滤波、spike 阈值及 artifact suppression。打开三个 TCP server，让它们 Pending；
RHX 保持 **Stop**，保存路径在 RHX 主机上必须存在，且使用不含空白字符的 ASCII 路径。

以下是命令模板；把路径、组织编号和记录通道换成实际值：

```bash
python experiments/run_openloop.py collect --manifest examples/openloop/manifest.example.json --cnn-checkpoint checkpoints/cnn_mnist.pt --out data/culture01_collect01 --channels A-000 A-001 A-002 A-003 A-004 A-005 A-006 A-007 --preparation-id culture01 --rhx-path D:/IntanData --features examples/openloop/features.json --pulse-profile examples/openloop/pulse_profile.json --timing-calibration examples/openloop/timing.example.json --shuffle --live-stim
```

这条命令**会执行刺激**。`--live-stim` 是显式输出开关；本开发验证没有执行这条硬件命令。
程序会执行以下顺序：

1. 离线加载全部图片、CNN，生成协议并验证窗口能容纳计划刺激；
2. 确认 RHS/物理模式和 Stop，上传硬件滤波设置，预装整场固定 profile；
3. 禁用并回读旧刺激路由，清空旧 TCP 数据选择，只启用需要的 STIM 和 RHX SPK；
4. 读取并固定完整记录通道顺序、HIGH/硬件滤波、参考、阈值及伪迹设置；
5. 开启独立 5001、5002 接收线程、连续收集线程和故障监控；整场持续 Record；
6. 每个输入调度全部脉冲，按数字 STIM 时钟检查各相幅度码、时长、极性、数量和时序；
7. 使用硬件样本时间和同一伪迹策略构建神经特征，记录 VALID/SILENT/INVALID；
8. 结束后停止 RHX，检查残包/未完成脉冲，保存日志 hash。

默认 `TCPNumberDataBlocksPerWrite=1`，仍需测量真实负载和总延迟。
可以加 `--ttl-bit 0` 保存 DIGITAL-IN 的 bit 0 上升沿；位号是 0-based。
当前试次锚点使用 STIM，TTL 是额外对照记录，不会用一个未知 TTL 猜测试次起点。

## 4. 用本次 RHX spike 训练并冻结解码器

```bash
python experiments/run_openloop.py train --sessions data/culture01_collect01 --test-groups heldout --kind lr --out checkpoints/culture01_decoder.json
```

支持 `lr` 和多层 `mlp`。二者的标准化和拟合只使用训练组；测试组必须显式指定，
训练组需要覆盖测试类别。多个兼容会话可一起传给 `--sessions`。
SILENT/INVALID 试次不用于训练，数量会报告。失败会话、日志 hash 不符或特征契约不一致会拒绝。

冻结文件是可检查的 JSON：包含模型权重、缩放、类别名、输入/神经/刺激契约及数据来源；
在线不用随机初始化读出、不从 waveform 重新检测 spike，也不加载 pickle 模型。
真实推理核对 CNN hash、编码方式、刺激端口/profile、组织编号、采样率、通道顺序、
RHX 检测设置和特征定义。模型输出概率尚未做置信度校准。

测试结果只针对所指定的 held-out groups。跨会话、跨时间或新图片泛化需要相应组别和数据，
不能由一次随机图片划分替代。现有 `.pt` 合成/旧 waveform 解码器不能直接用于此入口。

## 5. 真实采集与冻结推理：默认机械臂模拟

建立新输入清单；可以只包含一张新图片，省略 label。运行：

```bash
python experiments/run_openloop.py run --manifest data/new_inputs.json --cnn-checkpoint checkpoints/cnn_mnist.pt --decoder checkpoints/culture01_decoder.json --actions examples/openloop/actions.json --out data/culture01_infer01 --preparation-id culture01 --rhx-path D:/IntanData --timing-calibration examples/openloop/timing.example.json --live-stim
```

这时刺激和采集是实机路径，机械臂仍为 dry-run。编码/特征参数和记录通道默认采用模型契约，
显式覆盖后不一致会在开始 Record/触发之前拒绝。一次新输入只使用部分刺激通道也可运行，
程序会保留训练时的完整 profile bank，但不会额外执行这些配置行。

只有有效、非静默窗口才送入模型。低于 `--min-confidence`（默认 0.8）、重复、过期、忙碌
或无效决策会被拒绝。机械臂独立线程执行任务，接收和刺激时钟持续运行。
每次成功写串口仍只记 `COMMANDS_SENT`、`position_verified=false`，实际任务成功率需要独立观察。

完成传输/负载时延测量后，在 timing 文件中填写实测上界、`validated: true` 和非空 `evidence`。
**这是操作者的测量声明，软件不会自动证明它。** 对实机机械臂，训练数据和当前会话都必须
声明时序已验证；之后重新采集/训练，并在运行命令增加：

```text
--live-arm --arm-port COM4 --home
```

端口需换成实际串口。`--home` 会实际移动机械臂，是明确初始化动作，不会由分类结果隐式触发。
`Ctrl+C` 会取消未发脉冲并进入清理；采集故障会通知机械臂停止。故障时软件尝试 Stop RHX，
命令连接损坏时不能保证远程 Stop 已送达，现场仍需能直接操作 RHX/机械臂停止。

## 6. 回放与文件

```bash
python experiments/run_openloop.py replay --session data/culture01_infer01 --decoder checkpoints/culture01_decoder.json --actions examples/openloop/actions.json
```

回放不打开任何硬件连接。先核对日志 hash，再重新解析原始 5001/5002、核对数字脉冲归属、
重建窗口/特征并比较保存值；可重新推理和模拟执行动作。`REPLAY_MATCH` 表示文件内重建一致，
不证明原始硬件窗口已经满足跨端口延迟假设。

| 文件 | 内容 |
|---|---|
| `session.json` | 状态、组织/硬件/特征契约、时序声明、统计、日志 hash |
| `markers.raw` | 原始 5001 STIM/可选 DIGITAL-IN 字节 |
| `spikes.raw` | 原始 5002 RHX spike 字节，含通道、sample、ID |
| `events.jsonl` | 输入、响应、预测、调度、动作、故障的因果记录 |
| `trials.jsonl` | 每试次实际 marker、窗口、伪迹遮罩结果、计数和拒绝理由 |
| `decoder.json` | 独立保存的冻结模型、训练来源和分组测试结果 |

整场硬件采集限一小时一个 epoch；随后明确结束并建立新会话，避免长静默跨 uint32 半周期
无法仅凭稀疏 spike 解包的歧义。原图和 CNN checkpoint 仍需与实验数据一起保留。

## 7. 5002 完整性的边界

RHX 原生 5002 没有逐块心跳。5001 已到样本 N，不保证另一个 socket 的对应 spike 同时到达。
实机后端采用明确声明的 `max_cross_socket_delay_ms` holdback；随后收到早于已关闭窗口的
spike 会使整个会话 FAILED，并停止后续动作。流停滞、队列积压、硬件时钟落后主机、
compliance、残包、missing pulse、spike 设置改变等也会失败。

已经发出的物理动作不能由之后发现的迟到事件撤销。因此，在真实机械臂运行前实测这些界限，
是此版本的数据完整性前提。严格的传输完成证明需要 RHX 输出端增加跨流序列/心跳等协议支持；
本版本没有把安静的 socket 或 CurrentTimestamp 当作这种证明。

## 8. 旧入口与下一阶段

`run_with_dashboard.py`、`experiments/realtime_loop.py`、`experiments/realtime_openloop.py` 和
`experiments/stim_one_image.py` 的旧 CLI 已停止运行，并指向本入口。
`frontend/app.py` 保留为单次刺激/录音仪表板，尚未接入新连续实验的浏览器控制。
`run_2d_experiment.py` 只保留原 baseline 入口，旧 probe/classify 路径在连接硬件前拒绝。
旧 waveform 训练/分析脚本保留为历史方法，不能当作新 RHX spike 模型契约。

本轮完成第一阶段的开环软件链路。尚未加入结果反馈刺激、环境成功观察或 DishBrain 式行为闭环；
后续应在这一主干上增加环境/结果接口、反馈策略、恢复阶段和相应对照。

## 9. 依据与离线验证

- 随库 RHX TCP Guide 3.5.0：pp.3-4（128 samples/block、两个数据端口）、pp.6-8（采样率/step）、
  p.14（攒批）、p.18（RunMode/CurrentTimestamp）、pp.25-28/40-41/49-55（滤波、伪迹、参考和刺激）、
  pp.60-62（上传与触发）。
- User Guide：pp.22-29（数字刺激记录、settle、compliance 和电极电压）、pp.30-32（TCP 积压）。
- [Intan 官方固定源码](https://github.com/Intan-Technologies/Intan-RHX/tree/3c5867718d279af15965e6867cb657f7bca49c71)：
  `Engine/Threads/tcpdataoutputthread.cpp`，`Engine/Processing/channel.cpp`、`signalsources.cpp`、
  `commandparser.cpp` 和 `controllerinterface.cpp`。该源码标记 3.5.2，与随库 3.5.0 文档区分。

测试包括跨 TCP/数据块的双相脉冲、TTL、损坏数据、时间回绕、完整图片演练、原始回放、
训练/测试泄漏与 scaler 拟合范围、模型契约拒绝、双 socket 并发、迟到事件、断连/流停滞中止、
机械臂忙碌/重复/过期/低置信拒绝以及执行中取消。运行 `python -m pytest -q tests`。

最新结果：**201 passed**（较 150 项基线新增 51 项），独立 CLI 演练完成 18 次模拟采集、
12/6 训练/测试划分、6 次机械臂模拟任务和原始字节回放匹配。没有实机结果。
最新复测修正了 socket 模拟源在高负载下的供数时钟，并新增源时钟积压故障检验；
生产发送、读取、分类代码及实机时延界限保持不变，详见工程修改记录第 6 节。

## English

`experiments/run_openloop.py` now provides simulate, collect, train, run, replay and demo commands.
The shared pipeline uses typed 5001 STIM/TTL records and native 5002 RHX spikes, hardware-time
windows, a frozen JSON LR/MLP decoder, and an asynchronous arm task adapter. Explicit held-out
groups prevent scaler/model fitting on test data, and identical images cannot cross the split.
Real inference checks the encoder, preparation identity, electrode order and acquisition contract.

Run `python experiments/run_openloop.py demo --out data/openloop_demo_01` for the complete synthetic
interface exercise. Real collection needs a trained CNN, input manifest, physical RHS in Stop,
all three TCP servers, recording path, preparation ID, declared timing bounds and `--live-stim`.
Inference defaults to a simulated arm; physical arm output also requires measured timing declarations
for training and runtime, plus `--live-arm --arm-port ... --home`.

All current verification is offline/fake-controller verification. The native 5002 protocol lacks a
heartbeat, so online completeness depends on a declared cross-socket delay bound; a late violation
fails the session and stops further actions. A replay match verifies saved bytes, not that hardware
assumption. Physical current, biological decoding performance, task success and behavior-contingent
feedback remain to be measured or implemented. The new continuous pipeline is currently CLI-based;
the existing dashboard remains a separate single-stimulation tool.
