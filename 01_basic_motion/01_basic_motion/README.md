# 01 基础动作

本目录可以单独复制或打包，Python 3.9+，不需要另外两个目录或 OpenMV。
链路：类器官采集与解码 → 神经解码输出动作 → 本目录 Python → STM32 舵机控制板。
不包含头环连接、SSVEP、陀螺仪、视觉或测时间功能。

## 文件与运行

- `organoid_arm/controller.py`：动作执行、串口、动作等待、可选事件过滤。
- `organoid_arm/kinematics.py`：沿用竞赛 AM1 参数的逆运动学。
- `example.py`：默认不连接硬件的调用示例。
- `requirements.txt`：真实连接时安装 pyserial。

在本目录打开终端：

```powershell
python example.py
python -m pip install -r requirements.txt
python -m serial.tools.list_ports
```

## 最简单的接法

```python
from organoid_arm import ArmController

with ArmController(dry_run=True) as arm:
    arm.home()  # 操作者明确初始化；会移动到初始点，不自动松爪
    # 每产生一次有效动作决策，调用一次：
    result = arm.execute("right")
    print(result)
```

实际连接改为 `ArmController(port="COM4", dry_run=False)`。
COM4 只是示例，选择 STM32 控制板串口，不是 OpenMV 的 USB 串口；115200、8N1。
运行时停用 OpenMV 自动动作和其它控制程序，确保唯一控制源。

| 动作 | 含义 |
| --- | --- |
| `up`, `down` | z ±5 mm |
| `left`, `right` | x 减少 / 增加 5 mm |
| `forward`, `back` | y ±5 mm |
| `grab`, `release` | 夹爪闭合、张开；没有下探或抬升 |
| `cw`, `ccw` | 手腕 PWM ±222，约 30°，需确认实机观察方向 |
| `center` | 手腕回中 |
| `idle` | 不动 |
| `stop` | `$DST!` 停止并锁定，需操作者再次 home |
| `home` | 到 (0,100,70) mm，手腕回中 |

可配置 `step_mm=5`、`move_time_ms=800`、`open_pwm=1000`、`close_pwm=1550`。
`move_time_ms` 是舵机动作时长；动作等待用于避免指令覆盖，不是时延测量功能。
夹爪值需根据装配和物体标定。工作区 x [-120,120]、y [45,190]、z [20,140] mm，
越界和不可达直接报错；此边界不提供路径避障。

## 可选事件接口

```python
import time
from organoid_arm import ArmController, SignalAdapter

with ArmController(dry_run=True) as arm:
    arm.home()
    adapter = SignalAdapter(arm)
    print(adapter.handle({"seq": 1, "timestamp": time.time(),
                          "action": "up", "confidence": 0.95}))
```

这是可选调用方式；只想输出动作名称时直接 execute 即可。
四字段均必填，seq 在会话中递增，timestamp 为产生决策时的 Unix 秒数，confidence 为 0..1。
默认拒绝重复/乱序、超过 1 秒、置信度低于 0.8 的事件；home 仅由操作者调用。
同一动作的新序号会再次移动，操作者需在解码侧决定触发时机，不能每个采样点都发送一次。
不必复用旧 SSVEP 编号，信号类别与动作映射由双方约定。

## 返回与限制

空运行返回 simulated；真实模式返回 sent，仅说明写入并等待了估计动作时间。
位姿为目标位置估计，不是舵机位置反馈；没有物体夹持成功检测。
同一控制器忙时并发动作丢弃，不积压执行；非法输入和硬件故障抛异常。
串口故障或停止后清除位置估计并锁定。
直接 Python 接法由调用方在异常/退出时显式 `arm.stop()`，`close()` 仅关闭串口。
停止可由另一个线程调用；串口断线时停止不能保证生效，保留电源开关。
import 和创建默认实例均不会发送实机动作。

本次整理未连接硬件，未运行测试；需要安装标定和与神经解码结果联调。
