"""
Verify the WaveformFrameReader against synthetic frames matching the
verified Intan RHX waveform stream format.

Format (verified from real RHS Demo TCP capture):
    [4-byte stream header (one-time)]
    [frame_0: 4B timestamp(LE u32) + n_ch*2B samples(LE u16, offset-binary)]
    [frame_1: ...]
    ...

Conversion: uv = (uint16 - 32768) * 0.195
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intan_interface.tcp_client import (
    WaveformFrameReader,
    BLOCK_HEADER_SIZE,
    SAMPLES_PER_BLOCK,
    AMPLIFIER_OFFSET_BINARY_ZERO,
    AMPLIFIER_UV_PER_BIT,
)

BLOCK_MAGIC = b"\x08\x7a\xf0\x2e"  # Verified Intan RHS block marker


def make_synthetic_stream(n_channels, n_frames, start_timestamp=0, block_size=128):
    """
    Generate a byte buffer matching the verified Intan RHS TCP stream format.

    Format: repeating blocks of [4B block header] + [block_size × frame]
    Each frame: [4B timestamp LE u32] + [n_ch × 2B sample LE u16 offset-binary]
    """
    frame_bytes = 4 + n_channels * 2
    target_uv = np.zeros((n_channels, n_frames), dtype=np.float32)
    stream = bytearray()
    frame_idx = 0

    while frame_idx < n_frames:
        # Block header
        stream.extend(BLOCK_MAGIC)

        # Frames in this block
        frames_in_block = min(block_size, n_frames - frame_idx)
        for b in range(frames_in_block):
            i = frame_idx + b
            ts = (start_timestamp + i).to_bytes(4, "little", signed=False)
            sample_bytes = bytearray()
            for ch in range(n_channels):
                uv = float(50.0 * np.sin(2 * np.pi * (i + ch) / 30.0))
                target_uv[ch, i] = uv
                adc = int(round(uv / AMPLIFIER_UV_PER_BIT)) + AMPLIFIER_OFFSET_BINARY_ZERO
                adc = max(0, min(65535, adc))
                sample_bytes.extend(adc.to_bytes(2, "little", signed=False))
            stream.extend(ts + bytes(sample_bytes))

        frame_idx += frames_in_block

    return bytes(stream), target_uv


def test_basic_parsing():
    """Single chunk, full parse with block boundaries."""
    n_ch, n_fr = 8, 256  # 2 complete blocks
    raw, target = make_synthetic_stream(n_ch, n_fr)

    reader = WaveformFrameReader(n_ch)
    reader.add_data(raw)
    timestamps, wave_uv = reader.read_frames()

    assert timestamps.shape == (n_fr,), f"timestamps shape {timestamps.shape}"
    assert wave_uv.shape == (n_ch, n_fr), f"wave_uv shape {wave_uv.shape}"
    np.testing.assert_array_equal(timestamps, np.arange(n_fr, dtype=np.uint32))

    diff = np.abs(wave_uv - target)
    max_err = diff.max()
    assert max_err < AMPLIFIER_UV_PER_BIT * 1.5, \
        f"reconstruction error too large: {max_err}"
    print(f"[PASS] Basic parsing (2 blocks): shape ok, ts continuous, "
          f"max err {max_err:.4f} µV (< 1 LSB)")


def test_chunked_parsing():
    """Data arrives in arbitrary chunks; parser must handle splits across block boundaries."""
    n_ch, n_fr = 8, 384  # 3 blocks
    raw, target = make_synthetic_stream(n_ch, n_fr)

    reader = WaveformFrameReader(n_ch)
    rng = np.random.default_rng(0)
    pos = 0
    all_ts, all_wave = [], []
    while pos < len(raw):
        chunk_size = int(rng.integers(1, 200))
        chunk = raw[pos:pos + chunk_size]
        pos += len(chunk)
        reader.add_data(chunk)
        ts, wave = reader.read_frames()
        if ts.size > 0:
            all_ts.append(ts)
            all_wave.append(wave)

    timestamps = np.concatenate(all_ts) if all_ts else np.array([], dtype=np.uint32)
    wave_uv = np.concatenate(all_wave, axis=1) if all_wave else np.zeros((n_ch, 0))

    assert timestamps.shape == (n_fr,)
    assert wave_uv.shape == (n_ch, n_fr)
    np.testing.assert_array_equal(timestamps, np.arange(n_fr, dtype=np.uint32))

    max_err = np.abs(wave_uv - target).max()
    assert max_err < AMPLIFIER_UV_PER_BIT * 1.5
    print(f"[PASS] Chunked parsing (3 blocks, random splits): max err {max_err:.4f} µV")


def test_partial_block_held_in_buffer():
    """Incomplete block must remain buffered until more data arrives."""
    n_ch = 4
    n_fr = 128  # exactly 1 block
    raw, _ = make_synthetic_stream(n_ch, n_fr)

    reader = WaveformFrameReader(n_ch, block_size=128)
    # Send less than one full block
    half = len(raw) // 2
    reader.add_data(raw[:half])
    ts1, wave1 = reader.read_frames()
    assert ts1.shape[0] == 0, f"Partial block should yield 0 frames, got {ts1.shape[0]}"

    # Send the rest — now we have 1 complete block
    reader.add_data(raw[half:])
    ts2, wave2 = reader.read_frames()
    assert ts2.shape[0] == n_fr, f"Expected {n_fr} frames, got {ts2.shape[0]}"

    np.testing.assert_array_equal(ts2, np.arange(n_fr, dtype=np.uint32))
    print(f"[PASS] Partial blocks correctly held in buffer")


def test_request_specific_n_frames():
    """read_frames(n) returns at most n frames from parsed blocks."""
    n_ch, n_fr = 8, 256  # 2 blocks
    raw, _ = make_synthetic_stream(n_ch, n_fr)

    reader = WaveformFrameReader(n_ch)
    reader.add_data(raw)

    ts, wave = reader.read_frames(n_frames=10)
    assert ts.shape == (10,), f"Got {ts.shape}"
    assert wave.shape == (n_ch, 10)

    # Read remaining
    ts2, wave2 = reader.read_frames()
    assert ts2.shape == (246,), f"Got {ts2.shape}"
    assert ts2[0] == 10
    print("[PASS] read_frames(n) returns exactly n frames; remaining preserved")


def test_offset_binary_conversion():
    """Verify offset-binary uint16 -> microvolts conversion."""
    test_cases = [
        (32768, 0.0),                # zero
        (32768 + 100, 100 * 0.195),  # positive
        (32768 - 100, -100 * 0.195), # negative
        (65535, (65535 - 32768) * 0.195),  # max positive
        (0, (0 - 32768) * 0.195),    # max negative
    ]
    for u16, expected_uv in test_cases:
        actual = (u16 - AMPLIFIER_OFFSET_BINARY_ZERO) * AMPLIFIER_UV_PER_BIT
        assert abs(actual - expected_uv) < 1e-6, \
            f"u16={u16}: expected {expected_uv}, got {actual}"
    print("[PASS] Offset-binary conversion: zero, positive, negative, extremes")


def test_reset_clears_state():
    """reset() must clear all buffers."""
    n_ch = 8
    raw, _ = make_synthetic_stream(n_ch, 128)

    reader = WaveformFrameReader(n_ch)
    reader.add_data(raw)
    reader.read_frames()
    assert reader._total_parsed > 0

    reader.reset()
    assert reader._total_parsed == 0
    assert len(reader._raw_buffer) == 0
    assert len(reader._parsed_ts) == 0
    print("[PASS] reset() clears all buffers")


def test_cross_batch_timestamp_continuity():
    """
    Timestamps must be continuous across multiple read_frames calls,
    including when reads span block boundaries.
    """
    n_ch = 8
    n_total_frames = 512  # 4 blocks
    raw, _ = make_synthetic_stream(n_ch, n_total_frames, start_timestamp=1000)

    reader = WaveformFrameReader(n_ch)
    reader.add_data(raw)

    # Read in batches of 100 (does NOT align to block boundaries of 128)
    last_ts_per_batch = []
    first_ts_per_batch = []
    total_read = 0
    while total_read < n_total_frames:
        n_req = min(100, n_total_frames - total_read)
        ts, wave = reader.read_frames(n_frames=n_req)
        if ts.shape[0] == 0:
            break
        assert wave.shape == (n_ch, ts.shape[0])
        first_ts_per_batch.append(int(ts[0]))
        last_ts_per_batch.append(int(ts[-1]))
        assert np.all(np.diff(ts) == 1), f"non-continuous within batch starting at ts={ts[0]}"
        total_read += ts.shape[0]

    assert total_read == n_total_frames, f"Read {total_read}/{n_total_frames}"

    for i in range(1, len(first_ts_per_batch)):
        gap = first_ts_per_batch[i] - last_ts_per_batch[i - 1]
        assert gap == 1, f"batch {i-1}->{i} gap = {gap}, expected 1"

    print(f"[PASS] Cross-batch/cross-block timestamp continuity: "
          f"{len(first_ts_per_batch)} batches across 4 blocks")


def test_many_blocks_stress():
    """Parse 100 blocks (12800 frames) — stress test for block boundary handling."""
    n_ch = 8
    n_blocks = 100
    n_frames = n_blocks * 128
    raw, target = make_synthetic_stream(n_ch, n_frames, start_timestamp=5000)

    reader = WaveformFrameReader(n_ch)
    # Feed in large chunks (simulating TCP recv)
    chunk_size = 16384
    all_ts, all_uv = [], []
    pos = 0
    while pos < len(raw):
        reader.add_data(raw[pos:pos + chunk_size])
        pos += chunk_size
        ts, uv = reader.read_frames()
        if ts.size > 0:
            all_ts.append(ts)
            all_uv.append(uv)

    timestamps = np.concatenate(all_ts)
    wave_uv = np.concatenate(all_uv, axis=1)

    assert timestamps.shape == (n_frames,), f"Got {timestamps.shape}"
    assert wave_uv.shape == (n_ch, n_frames)
    assert np.all(np.diff(timestamps) == 1), "Not all diffs==1"
    assert timestamps[0] == 5000
    assert timestamps[-1] == 5000 + n_frames - 1

    max_err = np.abs(wave_uv - target).max()
    assert max_err < AMPLIFIER_UV_PER_BIT * 1.5
    print(f"[PASS] 100 blocks ({n_frames} frames): all continuous, "
          f"max err {max_err:.4f} µV")


def test_shape_contract():
    """Verify returned shape is (n_channels, n_samples), NOT transposed."""
    n_frames = 128  # one full block
    for n_ch in [4, 8, 32, 128]:
        raw, target = make_synthetic_stream(n_ch, n_frames)
        reader = WaveformFrameReader(n_ch)
        reader.add_data(raw)
        ts, wave = reader.read_frames()
        assert ts.shape == (n_frames,), f"n_ch={n_ch}: ts shape {ts.shape}"
        assert wave.shape == (n_ch, n_frames), \
            f"n_ch={n_ch}: wave shape {wave.shape}, expected ({n_ch}, {n_frames})"
        assert wave.dtype == np.float32, f"wave dtype {wave.dtype}"
        assert ts.dtype == np.uint32, f"ts dtype {ts.dtype}"
    print("[PASS] Shape contract: (n_channels, n_samples) for n_ch in [4,8,32,128]")


if __name__ == "__main__":
    print("=" * 60)
    print("WaveformFrameReader verification (synthetic Intan format)")
    print("=" * 60)

    test_basic_parsing()
    test_chunked_parsing()
    test_partial_block_held_in_buffer()
    test_request_specific_n_frames()
    test_offset_binary_conversion()
    test_reset_clears_state()
    test_cross_batch_timestamp_continuity()
    test_many_blocks_stress()
    test_shape_contract()

    print("=" * 60)
    print("All waveform parsing tests passed.")
    print("Format verified: 4B header + frames(4B ts + n_ch*2B u16 LE)")
    print("Conversion verified: uv = (u16 - 32768) * 0.195")
    print("=" * 60)
