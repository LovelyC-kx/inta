"""
test_5002_v2.py — 5002 诊断 v2

区别：
  - 不 Stop/Run，保持 RHX 当前状态（必须已在 Run 中）
  - 对 A-000..A-031 全部启用 TCPDataOutputEnabledSpike
  - 不改动已有阈值，只读回实际阈值供参考
  - 先断开再重连 5002（有些版本需要订阅建立后再 Configure）
  - 读取 20 秒
"""

import socket, time, struct, select

HOST       = "127.0.0.1"
CMD_PORT   = 5000
SPIKE_PORT = 5002
MAGIC      = struct.pack("<I", 0x3AE2710F)
N_CH       = 32          # 启用 A-000 … A-031
READ_SECS  = 20


def cmd(sock, text, wait=0.15, read=True):
    print(f">> {text}")
    sock.sendall((text + "\n").encode("ascii"))
    time.sleep(wait)
    if not read:
        return b""
    sock.settimeout(1.0)
    try:
        resp = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            resp += chunk
            if b"\n" in resp or resp.startswith(b"Return:") or resp.startswith(b"Error:"):
                break
        s = resp.decode("ascii", errors="replace").strip()
        print(f"<< {s}")
        return resp
    except socket.timeout:
        print("<< <no reply>")
        return b""


print("=" * 62)
print("5002 诊断 v2 — 不改 RunMode，启用 32 通道")
print("=" * 62)

scommand = socket.create_connection((HOST, CMD_PORT), timeout=5)
print(f"Command  {HOST}:{CMD_PORT}  OK")

# 检查当前状态
r = cmd(scommand, "Get RunMode")
print(f"[状态] 当前 RunMode: {r.decode('ascii','replace').strip()}")
print()

if b"Run" not in r:
    print("[警告] RHX 不在 Run 模式！请先在 RHX GUI 点击 ▶ Run，然后重新运行此脚本。")
    scommand.close()
    raise SystemExit(1)

# 启用 N_CH 个通道的 Spike TCP 输出
print(f"[配置] 启用 A-000..A-{N_CH-1:03d} 的 TCPDataOutputEnabledSpike ...")
for ch in range(N_CH):
    name = f"A-{ch:03d}"
    cmd(scommand, f"Set {name}.TCPDataOutputEnabledSpike True", read=False)
print("[配置] 完成")

# 读回几个通道确认
for ch in [0, 1, 2, 3]:
    name = f"A-{ch:03d}"
    r = cmd(scommand, f"Get {name}.TCPDataOutputEnabledSpike")
    t = cmd(scommand, f"Get {name}.SpikeThresholdMicroVolts")
print()

# ── 先连接 5002，配置完成后立刻开读 ──────────────────────────
print(f"连接 {HOST}:{SPIKE_PORT} ...", end="", flush=True)
sspike = socket.create_connection((HOST, SPIKE_PORT), timeout=5)
sspike.setblocking(False)
print(" OK")
print()

print(f"读取 {READ_SECS} 秒原始字节 ...")
print("-" * 62)

total = 0
buf   = bytearray()
t0    = time.time()
last_print = t0

while time.time() - t0 < READ_SECS:
    r, _, _ = select.select([sspike], [], [], 0.2)
    if r:
        try:
            data = sspike.recv(65536)
        except BlockingIOError:
            data = b""
        if data:
            total += len(data)
            buf.extend(data)
            hex32 = data[:32].hex(" ")
            print(f"  +{len(data):6d}B  total={total:8d}  hex前32: {hex32}")
    else:
        now = time.time()
        if now - last_print >= 5.0:
            elapsed = int(now - t0)
            print(f"  [{elapsed:3d}s] 仍为 0 字节...")
            last_print = now

print("-" * 62)
print(f"TOTAL BYTES : {total}")
magic_count = buf.count(MAGIC)
print(f"MAGIC COUNT : {magic_count}  (0x3ae2710f 出现次数)")

if total > 0 and magic_count == 0:
    print()
    print("有数据但未找到 0x3ae2710f magic，前 128 字节：")
    raw = bytes(buf[:128])
    for i in range(0, len(raw), 16):
        chunk = raw[i:i+16]
        hp = " ".join(f"{b:02x}" for b in chunk)
        ap = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {i:4d}  {hp:<48}  {ap}")

scommand.close()
sspike.close()
print()
print("连接已关闭。")
