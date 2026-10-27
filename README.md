# inta

类器官电生理工程组件：图像编码、Intan RHX 接口、事件解码与机械臂控制。
当前修改与接口迁移见 [ENGINEERING_CHANGES.md](docs/ENGINEERING_CHANGES.md)。

## 当前组件

- `intan_interface/tcp_client.py`：默认只采集；实刺激使用Stop→上传/回读→持续Run/Record→逐事件触发。
- `intan_interface/spike_stream.py`：5002完整spike事件、原始字节留存、采集故障检测。
- `intan_interface/spike_windows.py`：硬件采样时间分窗、spike ID与逐刺激伪迹区间。
- `frontend/app.py`：单次手动/编码刺激、录音及数字刺激记录分析。
- `experiments/spike_triggered_stim_demo.py`：默认只采集的接口演示；不是Pong学习实验。
- `01_basic_motion/01_basic_motion/`：独立机械臂动作接口。

连续分类主入口的5001 marker接入、真实事件训练、标签映射和机械臂任务整合仍在后续修改中。
旧脚本不能通过跳过上传保证无电流；公共client现在会拒绝未确认的旧触发方式。

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

Run `python -m pytest -q` for offline regression. Continuous classification with hardware
markers, event-based model training and arm-task integration are subsequent engineering work.
