# Demo bench commands

Run from the repository root. The user's Windows baseline passed all 201 existing tests.

## 真正连接正在运行的 RHX Demo（不是 pytest）

```powershell
python experiments/rhx_tcp_bench.py --demo --seconds 10 --out data/rhx_tcp01
```

操作：选择 RHS Stimulation/Recording 的 Demo，采样率 30 kS/s、电流步长 0.5 µA；
设置并应用 spike 阈值（可沿用负向 4 RMS），随后保持 Stop。
在 Network → Remote TCP Control 打开 Commands 5000、Waveform Output 5001、Spike Output 5002，
三个服务均为 Pending，关闭其他占用这些服务的客户端，再执行上面的命令。
程序自行配置 A-000 的三个 10 µA、500 µs/phase 双相脉冲（0/500/1000 ms），
开启 Run 并持续读取 A-000 STIM 和 A-000/A-001 spike，结束后 Stop。
准备和逐项回读输出配置可能需要一些时间。

数字端口 native names 必须补零（DIGITAL-OUT-01、DIGITAL-IN-01），模拟端口仍为
ANALOG-OUT-1。若旧版报 DIGITAL-OUT-1 的回读错误，请更新 intan_interface/tcp_client.py；
可选 TTL 修正位于 experiments/openloop_backend.py。失败目录保留原始诊断记录，重跑使用
新的 --out（如 data/rhx_tcp02），不要覆盖此前的输出。

如果已收到块和 spike，却在 CONNECTED 之后、RUN CONFIRMED 之前出现命令确认超时：
先手动 Stop RHX，更新 intan_interface/tcp_client.py 和 experiments/rhx_tcp_bench.py。
新版本先等待实际 5001 采集数据，再查询启动状态，并在报告中显示 phase。
不要把尚未确认 Run 的计数当成触发成功；triggers_sent 仍必须完成后才计数。

这是通过 TCP 连接用户实际运行的 RHX 软件；没有模拟服务器和失败后的替代数据。
关闭 RHX 后，用新输出目录再次执行，必须报告 FAILED / connected=false 并返回非零退出码。
该入口仅允许 Synthetic=True 的 RHS Demo，在任何配置写入前拒绝真实硬件及其他 controller。

关键限制：固定官方源码 3.5.2 的
`Engine/API/Synthetic/syntheticrhxcontroller.h::setManualStimTrigger` 是空实现；
`synthdatablockgenerator.cpp` 生成的 stim on/off 和 polarity 均为零。
因此命令上传/回读/触发未报错，不等于 Demo 产生了脉冲，也不会产生对应的生物诱发响应。
不要为获得 MATCH 而向原始数据添加假 marker。

观察每秒 RECEIVING 的 blocks/spikes 计数，以及最终 `session.json`：

- `connected=true`、`triggers_sent=3`、`waveform_blocks>0`、`spike_events>0`、
  `stop_confirmed=true` 时为 `DEMO_TCP_OK`，只表示软件 TCP 配置、触发命令和读取完成。
- 官方 Demo 通常为 `stim_markers=0`、
  `stim_record=NOT_VERIFIED_DEMO_DOES_NOT_SIMULATE_STIM`，不能写成刺激验收通过。
- 无 spike 时为 `DEMO_TCP_NO_SPIKES`，返回非零；先检查阈值、输出通道及运行时长。
- 真实读取的 `markers.raw`、`spikes.raw`、spike 事件及命令调度报告均保存到指定的新目录。
- 本入口不测试分类，不输出机械臂命令。真实电流及诱发响应仍需在实机负载/组织条件下验证。

开发验证：79 项相关软件回归通过，含关闭真实本地 TCP 端口时连接失败、拒绝硬件配置写入、
有/无数字刺激记录时的报告区分。还单独执行了 CLI 的关闭端口检查，确认非零退出且没有替代数据。
尚未在开发环境启动 RHX 桌面软件进行 GUI/TCP 实测，需用户按上述步骤运行本地 RHX 完成。

## 1. Stimulation commands and response reading together (no lab hardware)

```powershell
python -m pytest -v tests/test_rhx_session.py::test_prepared_profile_executes_all_pulses_and_checks_cancellation tests/test_openloop_live_transport.py::test_physical_path_keeps_collecting_through_blocking_stimulation_schedule tests/test_openloop_live_transport.py::test_frozen_inference_reaches_arm_adapter_with_real_backend_socket_fixture tests/test_openloop_live_transport.py::test_missing_stim_marker_fails_session_instead_of_decoding_zeros --basetemp .pytest_joint_01
```

Expect 4 passed. These exercise command scheduling, simultaneous 5001/5002 parsing, trial windows,
frozen inference with a simulated arm, and refusal when a pulse marker is missing. The fixture uses
an in-memory command controller and real local socketpairs. It does not connect to the open RHX app
or prove physical current. RHX Demo spontaneous spikes are not evidence of a response to stimulation.
The pytest directory is disposable and may be cleared on reuse; keep experimental data elsewhere.

## 2. Number console

No hardware:

```powershell
python virtual_arm_test.py --interactive
```

Physical arm (replace COM4 with the actual STM32 port):

```powershell
python virtual_arm_test.py --interactive --live --port COM4 --home
```

Enter one number and press Enter; wait for completion before the next command.
1 left, 2 right, 3 close gripper, 4 open gripper, 5 up, 6 down, 7 forward, 8 back,
9/10 rotate wrist, 11 center wrist. 0 stops and locks; h explicitly homes again; q exits and stops.
Default translation step is 5 mm; use --step-mm 1 for a 1 mm command step. This is a commanded
step, not measured positioning accuracy. Every session homes once at startup in simulation;
physical startup homing requires --home. A serial fault can prevent software stop delivery.

This is a new console over the existing unchanged driver, not a byte-identical recovery of the
old local ZIP's script. Use the displayed mapping; it is separate from classification action maps.
Real position and grasp success still need observation. This console does not consume RHX data.
