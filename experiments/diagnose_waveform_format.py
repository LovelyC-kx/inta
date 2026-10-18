"""
Raw TCP byte diagnostic for Intan RHX waveform stream.

Captures raw bytes from the waveform TCP socket and analyzes them to
determine the ACTUAL frame structure. Does NOT assume any particular
frame format.

Steps:
  1. Capture a few MB of raw bytes
  2. Print hex dump of first 200 bytes
  3. Try multiple candidate frame sizes (20, 36, 52, 68, 100, etc.)
  4. For each candidate, check if extracted timestamps are continuous
  5. Report which frame size produces continuous timestamps

This tells us what we need to know to fix WaveformFrameReader.

Usage:
  python experiments/diagnose_waveform_format.py
"""

import sys
import os
import socket
import time
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hex_dump(data, width=20, max_lines=20):
    """Pretty hex dump."""
    lines = []
    for offset in range(0, min(len(data), max_lines * width), width):
        chunk = data[offset:offset + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {offset:6d}  {hex_part:<{width*3}}  {ascii_part}")
    return "\n".join(lines)


def try_frame_size(raw, frame_size, header_offset=0, n_ch_candidate=8):
    """
    Try parsing raw bytes with a given frame size.
    Returns (n_good_diffs, n_total_diffs, first_10_timestamps) or None if too short.
    """
    usable = raw[header_offset:]
    n_frames = len(usable) // frame_size
    if n_frames < 20:
        return None

    buf = usable[:n_frames * frame_size]
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n_frames, frame_size)

    # Extract timestamps from first 4 bytes of each frame
    timestamps = arr[:, 0:4].copy().view("<u4").reshape(-1)
    diffs = np.diff(timestamps[:min(1000, n_frames)])

    n_good = int(np.sum(diffs == 1))
    n_total = len(diffs)
    first_10 = timestamps[:10].tolist()

    return n_good, n_total, first_10


def main():
    host = "127.0.0.1"
    command_port = 5000
    waveform_port = 5001
    n_channels_configured = 8

    print("=" * 70)
    print("Intan Waveform TCP Format Diagnostic")
    print("=" * 70)

    # Connect command socket to start acquisition
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cmd_sock.settimeout(5)
    try:
        cmd_sock.connect((host, command_port))
        print(f"[OK] Command connected to {host}:{command_port}")
    except ConnectionRefusedError:
        print("[FAIL] Cannot connect command socket. Is Intan RHX running with TCP enabled?")
        return

    # Start run mode
    cmd_sock.sendall(b"Set RunMode Run\n")
    time.sleep(0.5)

    # Connect waveform socket
    wave_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    wave_sock.settimeout(5)
    try:
        wave_sock.connect((host, waveform_port))
        print(f"[OK] Waveform connected to {host}:{waveform_port}")
    except ConnectionRefusedError:
        print("[FAIL] Cannot connect waveform socket.")
        cmd_sock.close()
        return

    # Capture raw bytes (2 seconds worth)
    print(f"\nCapturing raw bytes for ~2 seconds...")
    time.sleep(0.5)

    raw_chunks = []
    target_bytes = 2 * 30000 * (4 + n_channels_configured * 2) * 3  # ~3x estimate
    t0 = time.time()
    while time.time() - t0 < 3.0 and sum(len(c) for c in raw_chunks) < target_bytes:
        try:
            chunk = wave_sock.recv(65536)
            if chunk:
                raw_chunks.append(chunk)
        except socket.timeout:
            break

    raw = b"".join(raw_chunks)
    print(f"Captured {len(raw):,} bytes ({len(raw)/1e6:.2f} MB)")

    # Stop
    cmd_sock.sendall(b"Set RunMode Stop\n")
    time.sleep(0.2)
    wave_sock.close()
    cmd_sock.close()

    # Save raw for offline analysis
    raw_path = "data/waveform_raw_diagnostic.bin"
    os.makedirs("data", exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(raw)
    print(f"Raw bytes saved to {raw_path}")

    # ── Hex dump of first 400 bytes ─────────────────────────
    print(f"\n{'='*70}")
    print("HEX DUMP: first 400 bytes")
    print(hex_dump(raw[:400], width=20, max_lines=20))

    # ── Try different header offsets ─────────────────────────
    print(f"\n{'='*70}")
    print("PROBING: try different header offsets (0 to 8)")
    print(f"{'='*70}")

    # For RHS system, possible per-sample data types:
    # - amplifier: 2 bytes/ch
    # - DC amplifier: 2 bytes/ch (if enabled)
    # - stim data: 2 bytes/ch
    # - compliance limit: 2 bytes/ch
    # - amp settle: 2 bytes/ch
    # - charge recovery: 2 bytes/ch
    # So per channel could be 2, 4, 6, 8, 10, or 12 bytes
    # Plus 4 bytes timestamp per sample
    # frame_size = 4 + n_ch * bytes_per_channel

    n_ch = n_channels_configured
    candidate_frame_sizes = []
    for bytes_per_ch in [2, 4, 6, 8, 10, 12]:
        candidate_frame_sizes.append(4 + n_ch * bytes_per_ch)
    # Also try some raw sizes
    for extra in [0, 2, 4, 8, 16, 32]:
        candidate_frame_sizes.append(4 + n_ch * 2 + extra)
    candidate_frame_sizes = sorted(set(candidate_frame_sizes))

    print(f"\nCandidate frame sizes: {candidate_frame_sizes}")
    print()

    best_score = 0
    best_config = None

    for header_offset in [0, 2, 4, 8]:
        for fs in candidate_frame_sizes:
            result = try_frame_size(raw, fs, header_offset)
            if result is None:
                continue
            n_good, n_total, first_10_ts = result
            pct = n_good / n_total * 100 if n_total > 0 else 0

            if pct > 50:  # Only show promising results
                print(f"  header={header_offset:2d}, frame_size={fs:4d}: "
                      f"{n_good:5d}/{n_total:5d} diffs==1 ({pct:5.1f}%) "
                      f"ts[:5]={first_10_ts[:5]}")

            if n_good > best_score:
                best_score = n_good
                best_config = {
                    "header_offset": header_offset,
                    "frame_size": fs,
                    "bytes_per_ch": (fs - 4) // n_ch,
                    "n_good": n_good,
                    "n_total": n_total,
                    "pct": pct,
                    "first_timestamps": first_10_ts[:10],
                }

    print(f"\n{'='*70}")
    print("BEST MATCH:")
    if best_config:
        for k, v in best_config.items():
            print(f"  {k}: {v}")

        if best_config["pct"] > 95:
            print(f"\nConclusion: frame_size={best_config['frame_size']} bytes "
                  f"with header_offset={best_config['header_offset']} "
                  f"gives {best_config['pct']:.1f}% continuous timestamps.")
            bpc = best_config["bytes_per_ch"]
            if bpc == 2:
                print("  → Only amplifier data (2B/ch)")
            elif bpc == 4:
                print("  → Amplifier + 1 extra stream (4B/ch)")
            elif bpc == 6:
                print("  → Amplifier + 2 extra streams (6B/ch)")
            elif bpc == 8:
                print("  → Amplifier + 3 extra streams (8B/ch)")
            elif bpc == 10:
                print("  → Amplifier + 4 extra streams (10B/ch)")
            elif bpc == 12:
                print("  → Amplifier + 5 extra streams (12B/ch, all RHS data)")
            print(f"\n  UPDATE WaveformFrameReader to use frame_size={best_config['frame_size']}")
        else:
            print(f"\nNo frame size achieved >95% continuity. Best was {best_config['pct']:.1f}%.")
            print("May need more complex block-level parsing.")
    else:
        print("  No valid configuration found!")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
