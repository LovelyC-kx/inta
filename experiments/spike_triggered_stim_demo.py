#!/usr/bin/env python3
"""
spike_triggered_stim_demo.py — 闭环 Spike-triggered 刺激演示
使用 Intan TCP Spike Output (5002) + 后台立刻开始读取
"""

import sys, os, socket, struct, time, threading, argparse
from collections import defaultdict, deque
from datetime import datetime
import queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intan_interface.tcp_client import IntanTCPClient

_SPIKE_MAGIC = struct.pack("<I", 0x3AE2710F)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_CMD_PORT = 5000
DEFAULT_SPIKE_PORT = 5002
DEFAULT_N_CHANNELS = 8
DEFAULT_THRESHOLD = 10
DEFAULT_WINDOW_S = 1.0
DEFAULT_REFRACTORY_MS = 500
DEFAULT_AMPLITUDE_UA = 10.0
DEFAULT_PHASE_US = 200
DEFAULT_PREFIX = "A"


class _SpikeParser:
    """
    实际包格式（14 字节固定）：
      bytes 0-3  : magic 0x3AE2710F (LE)
      bytes 4-8  : channel name "A-xxx" (5 bytes ASCII, no null)
      bytes 9-12 : timestamp uint32 LE (sample index @ 30 kHz)
      byte  13   : spike type (0x01)
    """
    _PKT = 14

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data):
        self._buf.extend(data)

    def drain(self):
        results = []
        buf = self._buf
        while True:
            pos = buf.find(_SPIKE_MAGIC)
            if pos == -1:
                if len(buf) > 3:
                    del buf[:len(buf) - 3]
                break
            if pos > 0:
                del buf[:pos]
            if len(buf) < self._PKT:
                break
            name_bytes = bytes(buf[4:9])
            ts = struct.unpack_from("<I", buf, 9)[0]
            try:
                name = name_bytes.decode("ascii")
                ch = int(name.split("-")[1])
                results.append((ch, ts))
            except (IndexError, ValueError, UnicodeDecodeError):
                pass
            del buf[:self._PKT]
        return results


def _spike_reader(spike_sock, event_queue: queue.Queue,
                  raw_bytes_counter: list, stop_event: threading.Event):
    """后台线程：立刻开始读取 spike socket，把事件放入队列。"""
    parser = _SpikeParser()
    spike_sock.settimeout(0.05)
    while not stop_event.is_set():
        try:
            data = spike_sock.recv(65536)
            if data:
                raw_bytes_counter[0] += len(data)
                parser.feed(data)
                for ev in parser.drain():
                    event_queue.put(ev)
        except socket.timeout:
            pass
        except OSError:
            break


def _configure_stim(client, n_ch, prefix, amplitude_ua, phase_us):
    print(f"  配置 {min(n_ch,8)} 个刺激通道...", end="", flush=True)
    for ch in range(min(n_ch, 8)):
        name = f"{prefix}-{ch:03d}"
        for cmd in [
            f"Set {name}.stimenabled true",
            f"Set {name}.source KeyPressF{ch+1}",
            f"Set {name}.shape Biphasic",
            f"Set {name}.polarity CathodicFirst",
            f"Set {name}.firstphasedurationmicroseconds {phase_us}",
            f"Set {name}.secondphasedurationmicroseconds {phase_us}",
            f"Set {name}.interphasedelaymicroseconds 0",
            f"Set {name}.firstphaseamplitudemicroamps {amplitude_ua:.1f}",
            f"Set {name}.secondphaseamplitudemicroamps {amplitude_ua:.1f}",
            f"Set {name}.numberofstimpulses 1",
        ]:
            client.send_command(cmd, expect_response=False)
    # 必须上传参数到硬件，否则 ManualStimTriggerPulse 无效
    client.send_command("Execute UploadStimParameters", expect_response=False)
    print(" 完成")


def run_demo(host, cmd_port, spike_port, n_channels, threshold,
             window_s, refractory_ms, amplitude_ua, phase_us, prefix, dry_run):

    print()
    print("=" * 62)
    print("   CLOSED-LOOP SPIKE-TRIGGERED STIMULATION DEMO")
    print("=" * 62)
    print(f"  Spike 来源 : TCP Spike Output {host}:{spike_port}")
    print(f"  触发条件   : 任意通道 {window_s:.1f}s 内 spike > {threshold} 次")
    print(f"  刺激       : 双相方波, {phase_us}us/相, {amplitude_ua:.0f}uA, 阴极优先")
    print(f"  不应期     : {refractory_ms:.0f}ms")
    print(f"  模式       : {'DRY-RUN' if dry_run else 'LIVE — 刺激将发送至硬件'}")
    print("=" * 62)

    # 连接
    client = IntanTCPClient(host=host, command_port=cmd_port,
                            waveform_port=5001, spike_port=spike_port)
    print()
    print(f"连接 Command ({host}:{cmd_port})...", end="", flush=True)
    client.connect_command(timeout=10.0)
    print(" OK")

    # ── 关键：先连 5002，再发 Set 命令，否则 RHX 不推流 ──────────
    print(f"连接 Spike   ({host}:{spike_port})...", end="", flush=True)
    client.connect_spike(timeout=10.0)
    print(" OK")

    # 启用各通道的 TCP Spike Output（连接后立即发，RHX 立刻开始推流）
    print(f"[配置] 启用 {n_channels} 个通道的 TCPDataOutputEnabledSpike...", end="", flush=True)
    for ch in range(n_channels):
        cname = f"{prefix}-{ch:03d}"
        client.send_command(f"Set {cname}.TCPDataOutputEnabledSpike True",
                            expect_response=False)
    print(" 完成")

    # ── 立刻在后台开始读 5002 ────────────────────────────────────
    event_queue: queue.Queue = queue.Queue()
    raw_bytes = [0]
    stop_ev = threading.Event()
    t = threading.Thread(target=_spike_reader,
                         args=(client._spike_socket, event_queue, raw_bytes, stop_ev),
                         daemon=True)
    t.start()
    print("[后台] 已开始读取 Spike socket")

    # 等 3 秒验证数据到达
    print("  等待 3s 检查初始数据...", end="", flush=True)
    time.sleep(3.0)
    early = raw_bytes[0]
    print(f" 收到 {early} 字节, {event_queue.qsize()} 个事件")

    if early == 0:
        print()
        print("  [诊断] 3 秒内仍然 0 字节。")
        print("  请确认 RHX 已 ▶ Run，且 TCP Spike Output 面板显示 Connected。")
        print()

    # 刺激配置
    if not dry_run:
        _configure_stim(client, n_channels, prefix, amplitude_ua, phase_us)
    else:
        print("[DRY-RUN] 跳过刺激配置。")

    print()
    print("[操作] 确认 Intan RHX 已 Run，按 Enter 开始主循环...")
    try:
        input()
    except EOFError:
        time.sleep(1.0)

    print(f"[当前] 已收到原始字节: {raw_bytes[0]}  已解析事件: {event_queue.qsize()}")

    spike_times: dict = defaultdict(deque)
    last_stim_t: dict = {}
    total_stims = 0
    start_t = time.perf_counter()
    last_status = start_t
    warned = False

    print()
    print("[运行中] Ctrl+C 停止。")
    print()
    print(f"  {'时间':12s}  {'通道':>6s}  {'Spike/s':>8s}  动作")
    print("  " + "-" * 54)

    try:
        while True:
            now = time.perf_counter()

            # 消费队列中的事件
            drained = 0
            while not event_queue.empty():
                try:
                    ch_idx, _ts = event_queue.get_nowait()
                    if 0 <= ch_idx < n_channels:
                        spike_times[ch_idx].append(now)
                        drained += 1
                except queue.Empty:
                    break

            # 无数据 5 秒提示（每次只提示一次）
            if not warned and now - start_t > 5.0 and raw_bytes[0] == 0:
                warned = True
                print()
                print("  [结论] 5002 端口始终 0 字节。此 Intan 配置/版本不支持 TCP Spike Output 推流。")
                print("  建议：使用波形方案 (5001)，或升级/重装 Intan RHX。")
                print()

            # 滑动窗口 & 触发
            cutoff = now - window_s
            for ch in range(n_channels):
                q = spike_times[ch]
                while q and q[0] < cutoff:
                    q.popleft()
                count = len(q)
                if count > threshold:
                    last_t = last_stim_t.get(ch, 0.0)
                    if (now - last_t) * 1000 >= refractory_ms:
                        ts_str = datetime.now().strftime("%H:%M:%S.%f")[:12]
                        action = "DRY" if dry_run else "STIM *"
                        print(f"  {ts_str}  ch{ch:04d}  {count/window_s:8.1f}  [{action}]")
                        if not dry_run and ch < 8:
                            client.send_command(
                                f"Execute ManualStimTriggerPulse F{ch+1}",
                                expect_response=False)
                        last_stim_t[ch] = now
                        total_stims += 1
                        q.clear()

            # 3 秒状态
            if now - last_status >= 3.0:
                rates = {ch: len(q) for ch, q in spike_times.items() if len(q) > 0}
                t_str = f"{int(now-start_t):4d}s | {raw_bytes[0]}B"
                if rates:
                    rs = "  ".join(f"ch{ch}:{r}" for ch, r in sorted(rates.items()))
                    print(f"  [{t_str}] spike/s: {rs}")
                else:
                    print(f"  [{t_str}] spike/s 均为 0")
                last_status = now

            time.sleep(0.002)

    except KeyboardInterrupt:
        print()
        print(f"[停止] 收到原始字节: {raw_bytes[0]} | 触发刺激: {total_stims} 次")
    finally:
        stop_ev.set()
        t.join(timeout=2.0)
        client.disconnect()
        print("[已断开连接]")


def main():
    p = argparse.ArgumentParser(description="闭环 Spike-triggered 刺激演示 (5002)")
    p.add_argument("--host",          default=DEFAULT_HOST)
    p.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    p.add_argument("--spike-port",type=int, default=DEFAULT_SPIKE_PORT)
    p.add_argument("--n-channels",type=int, default=DEFAULT_N_CHANNELS)
    p.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    p.add_argument("--window",   type=float, default=DEFAULT_WINDOW_S)
    p.add_argument("--refractory",type=float,default=DEFAULT_REFRACTORY_MS)
    p.add_argument("--amplitude", type=float,default=DEFAULT_AMPLITUDE_UA)
    p.add_argument("--phase-duration",type=int,default=DEFAULT_PHASE_US)
    p.add_argument("--prefix",   default=DEFAULT_PREFIX)
    p.add_argument("--dry-run",  action="store_true")
    args = p.parse_args()
    run_demo(args.host, args.cmd_port, args.spike_port, args.n_channels,
             args.threshold, args.window, args.refractory, args.amplitude,
             args.phase_duration, args.prefix, args.dry_run)

if __name__ == "__main__":
    main()
