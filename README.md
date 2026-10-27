# inta

类器官电生理工程组件：图像编码、Intan RHX 接口、事件解码与机械臂控制。
当前修改与接口迁移见 [ENGINEERING_CHANGES.md](docs/ENGINEERING_CHANGES.md)。

## 连续开环入口

已提供统一 CLI，包含图片输入、完整刺激协议、连续 RHX spike/STIM 采集、同一硬件时钟分窗、
LR/MLP 训练与冻结推理、机械臂任务适配和原始字节回放。
完整启动步骤、输入格式、真实采集条件和已知边界见 [OPENLOOP.md](docs/OPENLOOP.md)。

```bash
python experiments/run_openloop.py demo --out data/openloop_demo_01
```

这条命令只运行完整的**模拟接口演练**，不连接硬件。真实后端已做 fake RHX/socketpair 测试，
尚未完成现场硬件/组织验收。机械臂任务结果仍需独立观察，行为反馈闭环尚未加入。

## 当前组件

- `intan_interface/tcp_client.py`：默认只采集；实刺激使用Stop→上传/回读→持续Run/Record→逐事件触发。
- `intan_interface/spike_stream.py`：5002完整spike事件、原始字节留存、采集故障检测。
- `intan_interface/spike_windows.py`：硬件采样时间分窗、spike ID与逐刺激伪迹区间。
- `intan_interface/marker_stream.py`：typed 5001 STIM/TTL、数字脉冲逐相核对。
- `experiments/run_openloop.py`：统一 simulate/collect/train/run/replay/demo 入口。
- `frontend/app.py`：单次手动/编码刺激、录音及数字刺激记录分析。
- `experiments/spike_triggered_stim_demo.py`：默认只采集的接口演示；不是Pong学习实验。
- `01_basic_motion/01_basic_motion/`：独立机械臂动作接口。

旧 `run_with_dashboard.py`、`realtime_loop.py`、`realtime_openloop.py`、`stim_one_image.py`
的 CLI 已停用，避免继续执行未迁移路径。仪表板仍为单次刺激工具。
原生5002没有逐块心跳；在线窗口完整性依赖显式、实测的跨端口延迟上界，详见启动文档。

## 离线检查

在项目Python环境安装 `requirements-dev.txt`，运行：

```bash
python -m pytest -q
```

测试目录限制为 `tests/`，不连接实验硬件。GitHub Actions同样运行离线回归。

## 5002采集演示

在RHX启用Command/Spike TCP服务并开始Run/Record后：

```bash
python experiments/spike_triggered_stim_demo.py --channels A-000 B-000 --dry-run
```

保存于 `data/spike_response_*/`：原始字节、完整事件与会话状态。默认不发送刺激。
电极波形、时延和组织响应需现场验证，数字记录不能替代这些测量。

## English

Engineering components for organoid electrophysiology, image encoding, Intan RHX and arm
control. The client defaults to acquisition only. Live stimulation requires a stopped-board
upload/readback before continuous acquisition; every scheduled pulse uses a verified profile.

The shared 5002 receiver preserves native channel names, hardware timestamps and spike IDs.
Window extraction requires acquisition-clock progress evidence and supports artifact masks.
The threshold demo is an interface test, not a validated biological feedback or Pong experiment.

Run `python -m pytest -q` for offline regression. The unified open-loop CLI now integrates
typed hardware markers, native RHX event training, frozen inference and asynchronous arm tasks.
See [OPENLOOP.md](docs/OPENLOOP.md). Current validation is offline/fake-controller only; native
5002 completeness requires a declared delay bound. Physical validation and behavioral feedback
remain outstanding. `demo` runs the entire synthetic interface workflow without hardware.
