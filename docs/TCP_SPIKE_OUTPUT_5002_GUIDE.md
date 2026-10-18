# Intan RHX TCP Spike Output (Port 5002) — 完整调试指南

## 背景

Intan RHX 软件提供三个 TCP 端口：

| 端口 | 用途 | 说明 |
|------|------|------|
| 5000 | Command | 请求-响应，发送控制命令 |
| 5001 | Waveform Output | 持续推流原始波形（每包 128 帧） |
| 5002 | Spike Output | 检测到 spike 后推送事件包 |

本文档记录了让 5002 正常工作所经历的所有坑，以及最终有效的配置方式。

---

## 硬件/软件环境

- **控制器类型**：`ControllerStimRecord`（Intan RHS 刺激+记录控制器）
- **RHX 版本**：3.5.0
- **采样率**：30000 Hz
- **实验介质**：PBS（磷酸缓冲盐溶液，无组织），噪声阈值约 13-21 µV

---

## 核心发现（必读）

### 发现 1：连接顺序至关重要

**错误做法**（之前所有尝试均失败的原因）：
```python
# 先配置通道，再连接 5002 → 0 字节
client.connect_command()
send("Set A-000.TCPDataOutputEnabledSpike True")
client.connect_spike()   # 此时连接，RHX 不会推流
```

**正确做法**：
```python
# 必须先连接 5002，再发 Set 命令
client.connect_command()
client.connect_spike()   # 先连接！
send("Set A-000.TCPDataOutputEnabledSpike True")  # 连接后再启用
# RHX 立刻开始向已连接的 socket 推送 spike 数据
```

> **原理推测**：RHX 在处理 `TCPDataOutputEnabledSpike True` 命令时，会检查 5002 是否有活跃连接，并立刻开始向其推流。如果先配置后连接，RHX 不会重新扫描并启动推流。

---

### 发现 2：TCPDataOutputEnabledSpike 默认为 False

5002 端口即使在 GUI 显示 `Connected` 状态，**每个通道的 spike 输出默认是关闭的**。

必须为每个通道单独启用：
```
Set A-000.TCPDataOutputEnabledSpike True
Set A-001.TCPDataOutputEnabledSpike True
...
```

可用命令验证：
```
Get A-000.TCPDataOutputEnabledSpike
→ Return: TCPDataOutputEnabledSpike True
```

不能用 `Execute ClearAllDataOutputs`，否则会清掉已设置的输出配置。

---

### 发现 3：实际数据包格式（14 字节固定）

**正确格式**：
```
字节 0-3  : magic = 0x3AE2710F（小端序）
字节 4-8  : 通道名 "A-xxx"（5字节 ASCII，无 null 终止符）
字节 9-12 : 时间戳（uint32 小端序，单位：采样点，30kHz 下除以 30000 = 秒）
字节 13   : spike 类型（固定为 0x01）
```

**示例**：`0f 71 e2 3a  41 2d 30 30 30  ef 25 bd 00  01`
- `0f 71 e2 3a` → magic 0x3AE2710F ✓
- `41 2d 30 30 30` → "A-000"（A=0x41, -=0x2D, 0=0x30×3）
- `ef 25 bd 00` → 时间戳 0x00BD25EF = 12,394,991 采样点
- `01` → spike 类型

**之前错误的假设格式**（导致解析完全失败）：
```
# 错误：以为是 magic + timestamp + null终止通道名 + 波形数据
magic(4) + timestamp(4) + channel_name_nullterm + n_samples(2) + waveform(n*2)
```

---

### 发现 4：不能在 Run 模式下修改输出设置

部分配置命令（如 `Set RunMode`）需要在 Stop 状态下执行。但 `TCPDataOutputEnabledSpike` 可以在 Run 状态下动态修改，修改后立即生效。

---

## 失败历史（排查过程）

### 阶段 1：直接连接 5002，0 字节

**现象**：Python 连接 5002 成功，但 20 秒内收到 0 字节。

**错误原因**：没有启用任何通道的 TCPDataOutputEnabledSpike。

**误判**：以为 5002 是"持续推流"类型（像 5001 一样），连上就有数据。

---

### 阶段 2：发送"激活字节"（错误方向）

**尝试**：向 5002 发送 `\n`、`start\n`、`subscribe\n`，希望触发推流。

**结果**：依然 0 字节。

**原因**：5002 是服务端推送的单向流，不接受客户端命令。控制必须通过 5000 命令端口。

---

### 阶段 3：启用 TCPDataOutputEnabledSpike，顺序错误

**尝试**（`test_5002_minimal.py`）：
```
Stop → ClearAllDataOutputs → Set A-000.TCPDataOutputEnabledSpike True
→ Set RunMode Run → 连接 5002 → 读取
```

**结果**：0 字节。`Get RunMode` 还出现 `<no reply>`。

**原因**：`ClearAllDataOutputs` 清掉了配置；`Set RunMode Run` 后连接 5002，RHX 不会对已有 spike 配置重新启动推流。

---

### 阶段 4：先连 5002 再启用通道（正确！）

**尝试**（`test_5002_v2.py` + 内联脚本）：
```
Connect 5002 → Set A-000..A-007.TCPDataOutputEnabledSpike True → 读取
```

**结果**：20 秒收到 6104 字节，MAGIC 出现 436 次 ✓

**结论**：连接顺序是关键。

---

## 正确的初始化流程

```python
import socket, struct, threading, queue

MAGIC = struct.pack("<I", 0x3AE2710F)
PKT_SIZE = 14

# 1. 先连命令端口
sc = socket.create_connection(("127.0.0.1", 5000), timeout=5)

# 2. 先连 5002（顺序关键！）
ss = socket.create_connection(("127.0.0.1", 5002), timeout=5)
ss.settimeout(0.05)

# 3. 确认 RHX 在 Run 模式
sc.sendall(b"Get RunMode\n")
# ... 读取响应，确认包含 "Run"

# 4. 连接后再启用通道（动态生效）
for ch in range(8):
    sc.sendall(f"Set A-{ch:03d}.TCPDataOutputEnabledSpike True\n".encode())
    # 无需等待响应，read=False

# 5. 开始读取
def parse_spike(buf: bytearray):
    """返回 (channel_index, timestamp_samples) 列表"""
    results = []
    while True:
        pos = buf.find(MAGIC)
        if pos == -1:
            if len(buf) > 3:
                del buf[:len(buf) - 3]
            break
        if pos > 0:
            del buf[:pos]
        if len(buf) < PKT_SIZE:
            break
        name = bytes(buf[4:9]).decode("ascii")
        ts   = struct.unpack_from("<I", buf, 9)[0]
        try:
            ch = int(name.split("-")[1])
            results.append((ch, ts))
        except (IndexError, ValueError):
            pass
        del buf[:PKT_SIZE]
    return results
```

---

## Spike 速率参考（PBS 环境，阈值 3.5× 噪声）

| 通道 | 阈值 (µV) | 典型速率 |
|------|-----------|---------|
| A-000 | -16 | ~2-5 spikes/s |
| A-001 | -16 | ~2-5 spikes/s |
| A-002 | -21 | ~1-3 spikes/s |
| A-003 | -30 | ~1-2 spikes/s |

注意：刺激伪迹会导致瞬时 spike 速率暴增（可达 50-80+/s），这是正常的交叉污染现象，不是真实神经信号。

---

## 闭环演示脚本使用方法

```powershell
# Dry-run 模式（只检测，不发刺激）
python experiments/spike_triggered_stim_demo.py --dry-run --n-channels 8 --threshold 8 --window 2.0

# Live 模式（真实发出双相方波刺激）
python experiments/spike_triggered_stim_demo.py --n-channels 8 --threshold 8 --window 2.0 --amplitude 10 --phase-duration 200
```

**参数说明**：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--n-channels` | 8 | 监听通道数（A-000 到 A-N） |
| `--threshold` | 10 | 触发刺激所需的 spike 数（在 window 时间内） |
| `--window` | 1.0 | 滑动窗口时长（秒） |
| `--refractory` | 500 | 同一通道两次刺激最小间隔（毫秒） |
| `--amplitude` | 10.0 | 刺激幅度（µA） |
| `--phase-duration` | 200 | 每相时长（µs） |
| `--dry-run` | - | 不发刺激，只打印触发日志 |

**防止刺激-spike 正反馈建议**：
- `--refractory 1000` 延长不应期至 1 秒
- `--threshold 15` 提高触发门槛，避免伪迹触发

---

## 常见问题排查

### Q: 连上 5002 但 0 字节

1. 检查 RHX 是否在 **Run** 模式（`Get RunMode` → `Return: RunMode Run`）
2. 检查连接顺序：**必须先 connect_spike()，再发 Set TCPDataOutputEnabledSpike True**
3. 在 RHX GUI 中，TCP Spike Output 面板状态应为 **Connected**（不是 Listening 或 Pending）
4. 不要调用 `Execute ClearAllDataOutputs`（会清掉 spike 输出配置）

### Q: 有字节但 MAGIC COUNT = 0

说明数据包格式与预期不同。使用 `diagnose_spike_format.py` 捕获原始字节并分析。

### Q: Spike 速率异常高（>50/s）

通常是刺激伪迹被检测为 spike。设置更长的不应期（`--refractory 2000`）或提高阈值。

### Q: `Get RunMode` 返回 `<no reply>`

RHX 正在从 Stop → Run 状态切换，等待 2 秒后重试。也可能是命令端口被阻塞，检查是否有其他进程在使用 5000 端口。

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `experiments/spike_triggered_stim_demo.py` | 主闭环演示脚本 |
| `experiments/test_5002_minimal.py` | 最小 5002 连通性测试（Stop→配置→Run） |
| `experiments/test_5002_v2.py` | 不改 RunMode 的诊断（启用32通道） |
| `experiments/diagnose_spike_format.py` | 原始字节格式诊断，hex dump |
| `intan_interface/tcp_client.py` | IntanTCPClient，封装命令/波形/spike 连接 |
