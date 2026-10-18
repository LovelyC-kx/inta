"""
Locate exact block boundaries in the captured Intan waveform TCP data.

Reads the .bin file from diagnose_waveform_format.py and:
1. Finds where the first timestamp discontinuity occurs
2. Shows hex dump around the break point
3. Scans for the next valid timestamp after the break
4. Determines exact block header size
5. Validates the hypothesis across the entire file

Run on your local machine where the .bin file was saved:
  python experiments/locate_block_boundary.py
"""

import os
import sys
import struct
import numpy as np

BIN_PATH = "data/waveform_raw_diagnostic.bin"
FRAME_SIZE = 20  # 4B timestamp + 8×2B channels
INITIAL_HEADER = 4


def hex_dump_range(data, start, length=60, context=10):
    """Show hex dump of a specific range with context."""
    actual_start = max(0, start - context)
    actual_end = min(len(data), start + length + context)
    lines = []
    for off in range(actual_start, actual_end, 20):
        chunk = data[off:off + 20]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        marker = " <<<" if off == start else ""
        lines.append(f"  {off:8d}  {hex_str}{marker}")
    return "\n".join(lines)


def find_timestamp_pattern(data, base_ts_upper_3bytes):
    """Find all offsets where upper 3 bytes of a uint32 LE match the known timestamp prefix."""
    b1, b2, b3 = base_ts_upper_3bytes
    offsets = []
    for i in range(len(data) - 3):
        if data[i + 1] == b1 and data[i + 2] == b2 and data[i + 3] == b3:
            ts = struct.unpack_from("<I", data, i)[0]
            offsets.append((i, ts))
    return offsets


def main():
    if not os.path.exists(BIN_PATH):
        print(f"[ERROR] {BIN_PATH} not found. Run diagnose_waveform_format.py first.")
        return

    with open(BIN_PATH, "rb") as f:
        raw = f.read()

    print(f"File: {BIN_PATH}")
    print(f"Size: {len(raw):,} bytes\n")

    # Step 1: Parse frames with known format, find first break
    print("=" * 70)
    print("STEP 1: Find first timestamp discontinuity")
    print("=" * 70)

    offset = INITIAL_HEADER
    prev_ts = None
    first_break_frame = None
    first_break_offset = None
    consecutive_good = 0

    for i in range(min((len(raw) - offset) // FRAME_SIZE, 5000)):
        pos = offset + i * FRAME_SIZE
        ts = struct.unpack_from("<I", raw, pos)[0]

        if prev_ts is not None:
            if ts == prev_ts + 1:
                consecutive_good += 1
            else:
                if first_break_frame is None:
                    first_break_frame = i
                    first_break_offset = pos
                    expected_ts = prev_ts + 1
                    print(f"First break at frame {i} (byte offset {pos})")
                    print(f"  Expected timestamp: {expected_ts}")
                    print(f"  Got timestamp:      {ts}")
                    print(f"  Consecutive good frames before break: {consecutive_good + 1}")
                    print(f"  Good frame data: {consecutive_good + 1} × {FRAME_SIZE} = "
                          f"{(consecutive_good + 1) * FRAME_SIZE} bytes")
                    break
        prev_ts = ts

    if first_break_frame is None:
        print("No discontinuity found in first 5000 frames!")
        return

    # Step 2: Hex dump around the break point
    print(f"\n{'='*70}")
    print(f"STEP 2: Hex dump around break (byte {first_break_offset})")
    print(f"{'='*70}")
    # Show the last good frame, the break point, and bytes after
    last_good_start = first_break_offset - FRAME_SIZE
    print(f"\nLast good frame starts at byte {last_good_start}:")
    print(hex_dump_range(raw, last_good_start, 80))

    # Step 3: Find the next valid timestamp after the break
    print(f"\n{'='*70}")
    print("STEP 3: Scan for next valid timestamp after break")
    print(f"{'='*70}")

    # The expected next timestamp
    expected_next_ts = struct.unpack_from("<I", raw, first_break_offset - FRAME_SIZE)[0] + 2
    # Extract the upper 3 bytes of the expected timestamp range
    # Timestamps should be in the range prev_ts to prev_ts + 1000
    target_upper = raw[first_break_offset - FRAME_SIZE + 1 : first_break_offset - FRAME_SIZE + 4]

    print(f"Searching for timestamps with upper bytes: "
          f"{' '.join(f'{b:02x}' for b in target_upper)}")

    scan_start = first_break_offset - 4
    scan_end = min(first_break_offset + 200, len(raw))

    found_next = None
    for probe_offset in range(scan_start, scan_end):
        if probe_offset + 4 > len(raw):
            break
        probe_ts = struct.unpack_from("<I", raw, probe_offset)[0]
        # Check if this looks like a valid timestamp (within expected range)
        if abs(probe_ts - expected_next_ts) < 10:
            print(f"\n  Found valid timestamp at byte {probe_offset}:")
            print(f"    Timestamp value: {probe_ts} (expected ~{expected_next_ts})")
            print(f"    Offset from break: {probe_offset - first_break_offset} bytes")
            found_next = probe_offset
            break

    if found_next is None:
        print("  No valid timestamp found within 200 bytes of break!")
        # Try wider scan
        for probe_offset in range(first_break_offset, min(first_break_offset + 2000, len(raw))):
            probe_ts = struct.unpack_from("<I", raw, probe_offset)[0]
            if abs(probe_ts - expected_next_ts) < 500:
                print(f"\n  (Wide scan) Found at byte {probe_offset}: ts={probe_ts} "
                      f"(offset from break: {probe_offset - first_break_offset})")
                found_next = probe_offset
                break

    if found_next is not None:
        block_header_size = found_next - first_break_offset
        n_good_frames = consecutive_good + 1
        block_data_bytes = n_good_frames * FRAME_SIZE
        total_block_bytes = block_data_bytes + block_header_size

        print(f"\n{'='*70}")
        print(f"STEP 4: Block structure determination")
        print(f"{'='*70}")
        print(f"  Good frames per block:  {n_good_frames}")
        print(f"  Frame data per block:   {block_data_bytes} bytes")
        print(f"  Block header size:      {block_header_size} bytes")
        print(f"  Total block size:       {total_block_bytes} bytes")
        print(f"  Initial stream header:  {INITIAL_HEADER} bytes")

        # Show the block header bytes
        header_bytes = raw[first_break_offset:found_next]
        print(f"\n  Block header bytes ({block_header_size} bytes):")
        print(f"    Hex: {' '.join(f'{b:02x}' for b in header_bytes)}")

        # Step 5: Validate across the file
        print(f"\n{'='*70}")
        print(f"STEP 5: Validate block structure across entire file")
        print(f"{'='*70}")

        n_blocks_expected = (len(raw) - INITIAL_HEADER) // total_block_bytes
        print(f"  Expected blocks in file: ~{n_blocks_expected}")

        blocks_validated = 0
        blocks_broken = 0
        ts_continuous_count = 0
        prev_block_last_ts = None

        for block_idx in range(n_blocks_expected + 1):
            block_start = INITIAL_HEADER + block_idx * total_block_bytes

            if block_idx > 0:
                # Skip block header
                block_start += 0  # header is at the START of the block
                data_start = block_start + block_header_size
            else:
                data_start = block_start  # first block has no block header (initial header instead)

            if data_start + n_good_frames * FRAME_SIZE > len(raw):
                break

            # Check first and last timestamp in this block
            first_ts = struct.unpack_from("<I", raw, data_start)[0]
            last_ts = struct.unpack_from("<I", raw, data_start + (n_good_frames - 1) * FRAME_SIZE)[0]

            intra_ok = (last_ts - first_ts == n_good_frames - 1)

            if prev_block_last_ts is not None:
                inter_ok = (first_ts == prev_block_last_ts + 1)
                if inter_ok:
                    ts_continuous_count += 1
            else:
                inter_ok = True  # First block

            if intra_ok and inter_ok:
                blocks_validated += 1
            else:
                blocks_broken += 1
                if blocks_broken <= 3:
                    print(f"  Block {block_idx}: first_ts={first_ts}, last_ts={last_ts}, "
                          f"intra={'OK' if intra_ok else 'FAIL'}, "
                          f"inter={'OK' if inter_ok else 'FAIL'}")

            prev_block_last_ts = last_ts

        total_blocks = blocks_validated + blocks_broken
        print(f"\n  Blocks validated: {blocks_validated}/{total_blocks}")
        print(f"  Cross-block continuity: {ts_continuous_count}/{total_blocks - 1}")

        if blocks_validated == total_blocks and ts_continuous_count == total_blocks - 1:
            print(f"\n  === BLOCK STRUCTURE CONFIRMED ===")
            print(f"  Stream = [4B initial header]")
            print(f"         + blocks of [{block_header_size}B block header"
                  f" + {n_good_frames} × {FRAME_SIZE}B frames]")
            print(f"  Total block size: {total_block_bytes} bytes")
            print(f"\n  UPDATE WaveformFrameReader:")
            print(f"    BLOCK_HEADER_SIZE = {block_header_size}")
            print(f"    SAMPLES_PER_BLOCK = {n_good_frames}")
        else:
            print(f"\n  Block structure partially validated "
                  f"({blocks_validated}/{total_blocks} good)")
            print(f"  May need manual investigation of broken blocks.")
    else:
        print("\nCould not determine block structure automatically.")
        print("Manual hex inspection needed.")


if __name__ == "__main__":
    main()
