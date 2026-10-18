"""
diagnose_spike_format.py — 诊断 Intan TCP Spike Output 实际字节格式

连接 5002 端口，捕获 5 秒原始数据，打印十六进制 dump，
帮助确认 spike 事件的实际二进制结构。

运行前确认：
  - Intan RHX 已 ▶ Run
  - TCP Spike Output 端口 5002 已 Connect (Pending)
  - Spike Output 面板中已添加通道

Usage:
  python experiments/diagnose_spike_format.py
"""

import sys
import os
import socket
import time
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOST = "127.0.0.1"
SPIKE_PORT = 5002
CAPTURE_S = 5.0


def hex_dump(data: bytes, width: int = 16, max_bytes: int = 512) -> str:
    lines = []
    data = data[:max_bytes]
    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {offset:5d}  {hex_part:<{width * 3}}  {ascii_part}")
    return "\n".join(lines)


def search_patterns(data: bytes):
    """扫描可能的 4 字节 magic 候选（出现 ≥3 次且间距相对规律）。"""
    if len(data) < 64:
        return

    # 统计所有 4 字节值的出现位置
    from collections import defaultdict
    positions: dict = defaultdict(list)
    for i in range(len(data) - 3):
        val = struct.unpack_from("<I", data, i)[0]
        positions[val].append(i)

    # 只关注出现 ≥3 次的值
    candidates = {v: ps for v, ps in positions.items() if len(ps) >= 3}
    if not candidates:
        print("  未发现出现 ≥3 次的 4 字节重复值。")
        return

    print(f"  发现 {len(candidates)} 个重复 4 字节值 (≥3次)，间距最规律的前 10 个：")
    scored = []
    for val, ps in candidates.items():
        if len(ps) < 3:
            continue
        diffs = [ps[i+1] - ps[i] for i in range(len(ps)-1)]
        # 间距标准差越小越可能是 magic
        if len(diffs) >= 2:
            mean_d = sum(diffs) / len(diffs)
            variance = sum((d - mean_d)**2 for d in diffs) / len(diffs)
            std_d = variance ** 0.5
            scored.append((std_d, val, ps, mean_d))

    scored.sort()
    for std_d, val, ps, mean_d in scored[:10]:
        print(f"    0x{val:08x}  出现{len(ps):4d}次  平均间距={mean_d:8.1f}B  间距std={std_d:8.1f}B  首次位置={ps[0]}")


def main():
    print("=" * 66)
    print("  Intan TCP Spike Output 格式诊断")
    print("=" * 66)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    print(f"连接 {HOST}:{SPIKE_PORT}...", end="", flush=True)
    try:
        sock.connect((HOST, SPIKE_PORT))
    except (ConnectionRefusedError, OSError) as e:
        print(f" 失败: {e}")
        return
    print(" OK")
    sock.settimeout(0.1)

    print(f"捕获 {CAPTURE_S:.0f}s 原始数据...", end="", flush=True)
    chunks = []
    deadline = time.perf_counter() + CAPTURE_S
    while time.perf_counter() < deadline:
        try:
            chunk = sock.recv(65536)
            if chunk:
                chunks.append(chunk)
        except socket.timeout:
            pass
    sock.close()

    raw = b"".join(chunks)
    print(f" 完成，共 {len(raw):,} 字节")

    if len(raw) == 0:
        print()
        print("[结论] 未收到任何字节。可能原因：")
        print("  1. Intan 不在 ▶ Run 模式")
        print("  2. TCP Spike Output 面板未添加通道")
        print("  3. 当前信号未超过 spike 阈值，无事件产生")
        return

    # 保存原始数据
    os.makedirs("data", exist_ok=True)
    out_path = "data/spike_raw_diagnostic.bin"
    with open(out_path, "wb") as f:
        f.write(raw)
    print(f"原始数据已保存至 {out_path}")

    print()
    print("=" * 66)
    print(f"前 512 字节 HEX DUMP")
    print("=" * 66)
    print(hex_dump(raw, width=16, max_bytes=512))

    print()
    print("=" * 66)
    print("扫描可能的 Magic Number 候选")
    print("=" * 66)
    search_patterns(raw)

    # 检查是否有可打印的 ASCII 字符串（通道名如 "A-000" 等）
    print()
    print("=" * 66)
    print("扫描 ASCII 字符串 (通道名候选, 如 'A-000')")
    print("=" * 66)
    printable_runs = []
    run_start = None
    for i, b in enumerate(raw[:2000]):
        if 32 <= b < 127:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= 4:
                s = raw[run_start:i].decode("ascii")
                printable_runs.append((run_start, s))
            run_start = None
    if printable_runs:
        for pos, s in printable_runs[:20]:
            print(f"  offset {pos:5d}: {repr(s)}")
    else:
        print("  未发现可打印 ASCII 字符串。")

    print()
    print("=" * 66)
    print("基本统计")
    print("=" * 66)
    print(f"  总字节数  : {len(raw):,}")
    print(f"  捕获时长  : {CAPTURE_S:.0f}s")
    print(f"  平均速率  : {len(raw)/CAPTURE_S:.0f} 字节/s")
    if len(raw) >= 8:
        first8 = raw[:8]
        print(f"  前 8 字节 : {' '.join(f'{b:02x}' for b in first8)}")
        val_le = struct.unpack_from("<I", raw, 0)[0]
        print(f"  前 4 字节 (LE uint32) : 0x{val_le:08x} = {val_le}")


if __name__ == "__main__":
    main()
