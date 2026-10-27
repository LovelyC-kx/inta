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
