"""
test_5002_minimal.py — 5002 端口最小可行测试

目标：只确认 5002 能不能收到原始字节。
不做任何解析、不做刺激、不做闭环。

按顺序：
  1. Stop RHX
  2. 清空所有 TCP 数据输出
  3. 开启 A-000 的 TCPDataOutputEnabledSpike
  4. 设置低阈值 (-30 µV) 保证能触发
  5. Run
  6. 读 10 秒原始字节并打印 hex
  7. 报告：total bytes / magic count

看结果：
  TOTAL BYTES > 0  → 5002 通，可继续修 parser
  TOTAL BYTES = 0  → RHX 没检测到 spike，看 GUI raster 是否有显示
"""

import socket
import time
import struct
import select

HOST      = "127.0.0.1"
CMD_PORT  = 5000
SPIKE_PORT = 5002
MAGIC     = struct.pack("<I", 0x3AE2710F)
READ_SECS = 10
THRESHOLD_UV = -30   # µV, 调低保证 demo 能触发


def cmd(sock, text, wait=0.1, read=True):
    print(f">> {text}")
    sock.sendall((text + "\n").encode("ascii"))
    time.sleep(wait)
    if read:
        sock.settimeout(0.5)
        try:
            resp = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                resp += chunk
                if b"\n" in resp or resp.startswith(b"Return:") or resp.startswith(b"Error:"):
                    break
            print(f"<< {resp.decode('ascii', errors='replace').strip()}")
            return resp
        except socket.timeout:
            print("<< <no reply>")
            return b""
    return b""


print("=" * 60)
print("5002 最小测试")
print("=" * 60)

# 连接
scommand = socket.create_connection((HOST, CMD_PORT), timeout=5)
sspike   = socket.create_connection((HOST, SPIKE_PORT), timeout=5)
sspike.setblocking(False)
print(f"Command  {HOST}:{CMD_PORT}  OK")
print(f"Spike    {HOST}:{SPIKE_PORT}  OK")
print()

# 1. 基本信息
cmd(scommand, "Get Version")
cmd(scommand, "Get Type")
cmd(scommand, "Get RunMode")
cmd(scommand, "Get SampleRateHertz")
print()

# 2. Stop 后配置（必须在 Stop 状态下修改输出设置）
cmd(scommand, "Set RunMode Stop", read=False)
time.sleep(0.5)
cmd(scommand, "Get RunMode")

# 3. 清空所有 TCP 数据输出
cmd(scommand, "Execute ClearAllDataOutputs", read=False)
time.sleep(0.3)

# 4. 开启 A-000 的 spike TCP 输出
cmd(scommand, "Set A-000.TCPDataOutputEnabledSpike True", read=False)
time.sleep(0.1)
resp = cmd(scommand, "Get A-000.TCPDataOutputEnabledSpike")

# 5. 设置阈值
cmd(scommand, f"Set A-000.SpikeThresholdMicroVolts {THRESHOLD_UV}", read=False)
time.sleep(0.1)
cmd(scommand, "Get A-000.SpikeThresholdMicroVolts")
print()

# 6. Run
cmd(scommand, "Set RunMode Run", read=False)
time.sleep(0.5)
cmd(scommand, "Get RunMode")

print()
print(f"读取 {READ_SECS} 秒原始字节，等待 spike 数据...")
print("-" * 60)

total = 0
buf   = bytearray()
t0    = time.time()

while time.time() - t0 < READ_SECS:
    r, _, _ = select.select([sspike], [], [], 0.1)
    if r:
        data = sspike.recv(65536)
        if data:
            total += len(data)
            buf.extend(data)
            hex32 = data[:32].hex(" ")
            print(f"  +{len(data):5d} 字节  total={total}  hex前32: {hex32}")

print("-" * 60)
print(f"TOTAL BYTES : {total}")
print(f"MAGIC COUNT : {buf.count(MAGIC)}  (0x3ae2710f 出现次数)")

if total > 0 and buf.count(MAGIC) == 0:
    # 有数据但没有已知 magic，打印前 128 字节帮助识别格式
    print()
    print("有数据但未找到 0x3ae2710f magic，前 128 字节：")
    raw = bytes(buf[:128])
    for i in range(0, len(raw), 16):
        chunk = raw[i:i+16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        asc_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {i:4d}  {hex_part:<48}  {asc_part}")

# Stop
cmd(scommand, "Set RunMode Stop", read=False)
scommand.close()
sspike.close()
print()
print("连接已关闭。")
