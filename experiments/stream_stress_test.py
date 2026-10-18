"""
Real-time Intan TCP waveform stream stability test.

Connects to a running Intan RHX (Demo or hardware) and reads waveform data
continuously for a fixed duration. Reports:
  - Total frames received vs expected
  - Number of dropped frames (timestamp gaps)
  - Number of duplicate timestamps
  - Maximum cross-batch gap
  - Per-channel µV statistics (mean, std, min, max)
  - Buffer backlog over time

Use this BEFORE running real experiments to verify:
  - Stream is stable for the duration of your experiment
  - No frame drops at your channel count
  - Per-channel signal quality is reasonable
  - WaveformFrameReader correctly handles the live stream

Usage:
    python experiments/stream_stress_test.py --duration 30 --n-channels 8
    python experiments/stream_stress_test.py --duration 60 --n-channels 32
    python experiments/stream_stress_test.py --duration 30 --n-channels 128
"""

import sys
import os
import time
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intan_interface.tcp_client import IntanTCPClient
from utils.experiment_log import log_experiment


def stress_test(host, command_port, waveform_port, duration_s, n_channels,
                save_path=None):
    """Run a continuous-read stability test."""

    print("=" * 70)
    print(f"Intan Waveform Stream Stress Test")
    print(f"  Host:     {host}")
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {duration_s} s")
    print(f"  Expected: ~{int(30000 * duration_s)} frames at 30 kS/s")
    print("=" * 70)

    client = IntanTCPClient(
        host=host,
        command_port=command_port,
        waveform_port=waveform_port,
    )

    try:
        # Connect
        client.connect_command(timeout=5.0)
        print("[OK] Command connected")

        client.connect_waveform(timeout=5.0)
        print("[OK] Waveform connected")

        client.init_waveform_reader(n_channels)

        # Verify sample rate
        sr_response = client.get_sample_rate()
        print(f"[OK] Sample rate: {sr_response} Hz")
        if isinstance(sr_response, int):
            sample_rate = sr_response
        else:
            sample_rate = 30000  # fallback

        # Start acquisition
        client.run(verify=False)
        print("[OK] RunMode set to Run")
        time.sleep(0.3)

        # Read in batches
        batch_samples = sample_rate // 10  # 100 ms batches
        n_batches_target = int(duration_s * 10)

        all_timestamps = []
        all_first_per_batch = []
        all_last_per_batch = []
        per_channel_sum = np.zeros(n_channels, dtype=np.float64)
        per_channel_sumsq = np.zeros(n_channels, dtype=np.float64)
        per_channel_min = np.full(n_channels, np.inf)
        per_channel_max = np.full(n_channels, -np.inf)
        total_frames = 0
        timing_per_batch = []

        print(f"\nReading {n_batches_target} batches of {batch_samples} frames each...")
        t_start = time.perf_counter()

        for batch_idx in range(n_batches_target):
            t_b0 = time.perf_counter()
            try:
                ts, wave_uv = client.read_waveform_samples(
                    batch_samples, n_channels, timeout=5.0
                )
            except TimeoutError as e:
                print(f"\n[TIMEOUT] batch {batch_idx}: {e}")
                break

            t_b1 = time.perf_counter()
            timing_per_batch.append((t_b1 - t_b0) * 1000)  # ms

            all_timestamps.append(ts)
            all_first_per_batch.append(int(ts[0]))
            all_last_per_batch.append(int(ts[-1]))
            total_frames += ts.shape[0]

            per_channel_sum += wave_uv.sum(axis=1)
            per_channel_sumsq += (wave_uv ** 2).sum(axis=1)
            per_channel_min = np.minimum(per_channel_min, wave_uv.min(axis=1))
            per_channel_max = np.maximum(per_channel_max, wave_uv.max(axis=1))

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  Batch {batch_idx+1:4d}/{n_batches_target} | "
                      f"frames={total_frames:7d} | "
                      f"elapsed={elapsed:5.1f}s | "
                      f"batch_time={timing_per_batch[-1]:5.1f}ms")

        elapsed_total = time.perf_counter() - t_start
        client.stop(verify=False)

        # ── Analyze ────────────────────────────────────────────
        timestamps = np.concatenate(all_timestamps) if all_timestamps else np.array([])

        if len(timestamps) == 0:
            print("\n[FAIL] No frames received.")
            return

        diffs = np.diff(timestamps)
        n_continuous = int(np.sum(diffs == 1))
        n_gaps = int(np.sum(diffs > 1))
        n_dups = int(np.sum(diffs == 0))
        n_negative = int(np.sum(diffs < 0))
        max_gap = int(diffs.max()) if len(diffs) > 0 else 0

        # Cross-batch gaps
        cross_batch_gaps = []
        for i in range(1, len(all_first_per_batch)):
            gap = all_first_per_batch[i] - all_last_per_batch[i - 1]
            cross_batch_gaps.append(gap)
        n_cross_batch_gaps = sum(1 for g in cross_batch_gaps if g != 1)

        # Per-channel stats
        ch_mean = per_channel_sum / total_frames
        ch_var = per_channel_sumsq / total_frames - ch_mean ** 2
        ch_std = np.sqrt(np.maximum(ch_var, 0))

        expected_frames = int(sample_rate * duration_s)
        loss_rate = (expected_frames - total_frames) / expected_frames * 100

        # ── Report ─────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("STREAM STABILITY REPORT")
        print("=" * 70)
        print(f"Wall-clock duration:     {elapsed_total:.2f} s")
        print(f"Frames received:         {total_frames:,}")
        print(f"Frames expected:         ~{expected_frames:,} (sr={sample_rate})")
        print(f"Frame loss:              {loss_rate:+.2f}%")
        print(f"Average frame rate:      {total_frames / elapsed_total:.1f} Hz")
        print()
        print("Timestamp continuity:")
        print(f"  diff == 1 (continuous):    {n_continuous:,}")
        print(f"  diff > 1  (gaps):          {n_gaps}")
        print(f"  diff == 0 (duplicates):    {n_dups}")
        print(f"  diff < 0  (out-of-order):  {n_negative}")
        print(f"  max gap:                    {max_gap}")
        print(f"  cross-batch gaps:           {n_cross_batch_gaps}")
        print()
        print("Per-batch read time (ms):")
        if timing_per_batch:
            tbat = np.array(timing_per_batch)
            print(f"  mean = {tbat.mean():.2f}, std = {tbat.std():.2f}, "
                  f"max = {tbat.max():.2f}, min = {tbat.min():.2f}")
        print()
        print("Per-channel µV statistics:")
        print(f"  ch  |    mean  |     std  |     min  |     max")
        for ch in range(n_channels):
            print(f"  {ch:3d} | {ch_mean[ch]:8.2f} | {ch_std[ch]:8.2f} | "
                  f"{per_channel_min[ch]:8.1f} | {per_channel_max[ch]:8.1f}")
        print()

        # Verdict — strict for Demo / local TCP
        verdict_strict = (
            n_gaps == 0 and n_dups == 0 and n_negative == 0
            and n_cross_batch_gaps == 0 and abs(loss_rate) < 0.1
        )
        verdict_acceptable = (
            n_gaps == 0 and n_dups == 0 and n_negative == 0
            and abs(loss_rate) < 1.0
        )

        print("Pass criteria (strict, recommended for Demo / 8-32 ch):")
        print(f"  gaps == 0:              {'OK' if n_gaps == 0 else 'FAIL'}")
        print(f"  duplicates == 0:        {'OK' if n_dups == 0 else 'FAIL'}")
        print(f"  out-of-order == 0:      {'OK' if n_negative == 0 else 'FAIL'}")
        print(f"  cross-batch gaps == 0:  {'OK' if n_cross_batch_gaps == 0 else 'FAIL'}")
        print(f"  loss rate < 0.1%:       {'OK' if abs(loss_rate) < 0.1 else 'FAIL'}")
        print()

        if verdict_strict:
            verdict_str = "STREAM STABLE"
            print(f"VERDICT: {verdict_str} — strict criteria met ✓")
            print("         Safe to proceed to next level.")
        elif verdict_acceptable:
            verdict_str = "STREAM PASSABLE"
            print(f"VERDICT: {verdict_str} but not strict-clean.")
            print("         At 128 channels this may be expected. At 8-32 channels,")
            print("         investigate before proceeding.")
        else:
            verdict_str = "STREAM FAILED"
            print(f"VERDICT: {verdict_str} — DO NOT proceed to real-time loop.")
            print("         Likely causes: Python recv too slow, buffer too small,")
            print("         or Intan side dropping frames.")

        # Save
        if save_path:
            np.savez(
                save_path,
                timestamps=timestamps,
                ch_mean=ch_mean,
                ch_std=ch_std,
                ch_min=per_channel_min,
                ch_max=per_channel_max,
                timing_per_batch=np.array(timing_per_batch),
                cross_batch_gaps=np.array(cross_batch_gaps),
                metadata={
                    "n_channels": n_channels,
                    "sample_rate": sample_rate,
                    "duration_s": duration_s,
                    "total_frames": total_frames,
                    "loss_rate_pct": loss_rate,
                    "n_gaps": n_gaps,
                    "n_dups": n_dups,
                },
            )
            print(f"\nSaved to {save_path}")

        log_experiment(
            script="stream_stress_test.py",
            args={"host": host, "command_port": command_port,
                  "waveform_port": waveform_port,
                  "duration_s": duration_s, "n_channels": n_channels},
            verdict=verdict_str,
            summary={
                "total_frames": int(total_frames),
                "expected_frames": int(expected_frames),
                "loss_rate_pct": round(float(loss_rate), 4),
                "n_gaps": int(n_gaps),
                "n_dups": int(n_dups),
                "n_out_of_order": int(n_negative),
                "n_cross_batch_gaps": int(n_cross_batch_gaps),
                "wall_clock_s": round(float(elapsed_total), 2),
                "avg_frame_rate_hz": round(total_frames / elapsed_total, 1),
                "avg_batch_read_ms": round(float(np.mean(timing_per_batch)), 2) if timing_per_batch else None,
            },
            data=str(save_path) if save_path else None,
        )

    finally:
        client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cmd-port", type=int, default=5000)
    parser.add_argument("--wave-port", type=int, default=5001)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Test duration in seconds")
    parser.add_argument("--n-channels", type=int, default=8,
                        help="Must match Intan TCP Channels To Stream count")
    parser.add_argument("--save", default=None,
                        help="Optional .npz path to save report data")
    args = parser.parse_args()

    save = args.save
    if save is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save = f"data/stream_stress_{args.n_channels}ch_{int(args.duration)}s_{ts}.npz"
        os.makedirs("data", exist_ok=True)

    stress_test(
        host=args.host,
        command_port=args.cmd_port,
        waveform_port=args.wave_port,
        duration_s=args.duration,
        n_channels=args.n_channels,
        save_path=save,
    )
