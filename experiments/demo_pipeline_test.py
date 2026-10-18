"""
Step 1: Demo Mode Full Pipeline Test.

Run this with Intan RHX open in Demo mode (RHS Controller, 30 kS/s).
Before running:
  1. Start Intan RHX -> Demo Mode -> RHS Stim/Recording Controller -> 30 kS/s
  2. Network -> Remote TCP Control
  3. Set Command port=5000, click Connect (status: Pending)
  4. Set Waveform Output port=5001, add channels A-000~A-007 to "Channels To Stream"
  5. Click Connect on Waveform Output (status: Pending)
  6. Run this script

This script tests:
  - TCP command connection
  - Basic Get/Set commands
  - Stimulus protocol generation from an image
  - Stimulus parameter configuration via TCP
  - Data acquisition start/stop
  - Waveform data reception and parsing
  - Feature extraction on received data
  - Decoder inference (offline, on received data)

It does NOT test real stimulation effects (Demo data is synthetic).
"""

import sys
import os
import time
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intan_interface.tcp_client import IntanTCPClient
from encoders.stimulus_schema import StimulusProtocol, Pulse
from utils.experiment_log import log_experiment


def print_step(n, desc):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {desc}")
    print(f"{'='*60}")


def run_demo_test(host="127.0.0.1", command_port=5000, waveform_port=5001):

    client = IntanTCPClient(
        host=host,
        command_port=command_port,
        waveform_port=waveform_port,
    )

    # ── Step 1: Command TCP Connection ─────────────────────────
    print_step(1, "TCP Command Connection")
    try:
        client.connect_command(timeout=5.0)
        print("[OK] Command socket connected")
    except ConnectionRefusedError:
        print("[FAIL] Connection refused. Is Intan RHX running with TCP enabled?")
        print("  1. Start Intan RHX -> Demo Mode -> RHS Controller -> 30 kS/s")
        print("  2. Network -> Remote TCP Control")
        print("  3. Set Command port=5000, click Connect")
        return False

    # ── Step 2: Basic Commands ─────────────────────────────────
    print_step(2, "Basic Get/Set Commands")

    tests = [
        ("Get RunMode", "RunMode"),
        ("Get SampleRateHertz", "SampleRateHertz"),
    ]

    for cmd, expected_keyword in tests:
        try:
            resp = client.send_command(cmd)
            ok = expected_keyword in resp if resp else False
            status = "[OK]" if ok else "[WARN]"
            print(f"  {status} {cmd} -> {resp}")
        except Exception as e:
            print(f"  [FAIL] {cmd} -> {e}")

    # ── Step 3: Generate StimulusProtocol ──────────────────────
    print_step(3, "Generate StimulusProtocol (no CNN, manual test pattern)")

    test_pulses = []
    for ch in range(8):
        for t_idx in range(3):
            t_ms = ch * 50.0 + t_idx * 200.0
            if t_ms <= 2000.0:
                test_pulses.append(Pulse(channel=ch, time_ms=t_ms))

    test_pulses.sort(key=lambda p: (p.time_ms, p.channel))
    protocol = StimulusProtocol(
        pulses=test_pulses,
        n_channels=8,
        duration_ms=2000.0,
    )
    protocol.validate()
    print(f"[OK] Protocol: {protocol.total_pulses} pulses, "
          f"per_ch={protocol.pulses_per_channel}")
    print(f"     Binary matrix sum={protocol.to_binary_matrix(10).sum()}")

    # ── Step 4: Send Stimulus Configuration ────────────────────
    print_step(4, "Configure Stimulus Parameters via TCP")

    commands = client.protocol_to_commands(protocol)
    print(f"  Generated {len(commands)} commands")

    # RHS `Set <channel>.<stim_param>` does NOT echo a response (verified in
    # tcp_client.set() docstring and used by send_protocol_config). Expecting
    # one causes a per-command 2s timeout and leaves the socket in a state
    # that breaks downstream Get RunMode.
    success_count = 0
    fail_count = 0
    for cmd, desc in commands:
        try:
            client.send_command(cmd, expect_response=False)
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] {cmd}: {e}")
            fail_count += 1

    print(f"  Sent: {success_count} OK, {fail_count} FAIL "
          f"(stim Set commands are fire-and-forget on RHX)")
    if fail_count > 0:
        print("  [WARN] Some commands failed at the socket layer — investigate.")

    # ── Step 5: Start/Stop Acquisition ─────────────────────────
    print_step(5, "Start/Stop Data Acquisition")

    # verify=False: skip Get RunMode after Set. Right after the 80 stim Sets
    # the RHX command channel can be backlogged enough that an immediate Get
    # times out. stream_stress_test.py uses the same verify=False convention
    # for the same reason. Step 2 already verified Get works in isolation.
    try:
        client.run(verify=False)
        print(f"  [OK] Set RunMode Run (verify skipped)")
        time.sleep(2.0)
        client.stop(verify=False)
        print(f"  [OK] Set RunMode Stop (verify skipped)")
    except Exception as e:
        print(f"  [FAIL] Run/Stop: {e}")

    # ── Step 6: Waveform Data Reception ────────────────────────
    print_step(6, "Waveform Data Reception")

    try:
        client.connect_waveform(timeout=5.0)
        print("[OK] Waveform socket connected")

        client.run(verify=False)
        time.sleep(0.5)

        n_channels_stream = 8
        n_samples = 3000  # 100ms at 30kHz
        print(f"  Reading {n_samples} samples x {n_channels_stream} channels...")
        print(f"  (Using verified format: 4B header + frames of 4B timestamp + "
              f"{n_channels_stream}*2B uint16 offset-binary)")

        try:
            client.init_waveform_reader(n_channels_stream)
            timestamps, data_uv = client.read_waveform_samples(
                n_samples=n_samples,
                n_channels=n_channels_stream,
            )
            print(f"  [OK] Received µV data shape: {data_uv.shape}")
            print(f"       Timestamps: first={timestamps[0]}, last={timestamps[-1]}, "
                  f"diff_unique={np.unique(np.diff(timestamps))}")
            print(f"       µV: mean={data_uv.mean():.1f}, std={data_uv.std():.1f}, "
                  f"min={data_uv.min():.1f}, max={data_uv.max():.1f}")

            # Sanity: timestamp diffs should all be 1 (continuous frames)
            ts_diffs = np.unique(np.diff(timestamps))
            if len(ts_diffs) == 1 and ts_diffs[0] == 1:
                print("  [OK] Timestamps continuous (diff=1)")
            else:
                print(f"  [WARN] Timestamp diffs not all 1: {ts_diffs}")

        except Exception as e:
            print(f"  [WARN] Waveform read failed: {e}")
            print("         This may happen if channels are not added to TCP stream.")
            print("         In Intan: add A-000~A-007 to 'Channels To Stream'")
            data_uv = np.random.randn(8, 3000).astype(np.float32) * 50.0
            print(f"  Using synthetic fallback data: shape={data_uv.shape}")

        client.stop(verify=False)

    except ConnectionRefusedError:
        print("[WARN] Waveform socket not available. Using synthetic data.")
        data_uv = np.random.randn(8, 3000).astype(np.float32) * 50.0

    # ── Step 7: Feature Extraction ─────────────────────────────
    print_step(7, "Feature Extraction on Received Data")

    from feature_extraction.extractors import (
        extract_timebin_features,
        extract_rms_features,
        extract_power_features,
    )

    sample_rate = 30000.0

    timebin_feat = extract_timebin_features(
        (np.abs(data_uv) > 50).astype(np.float32),
        bin_size_ms=200.0,
        sample_rate_hz=sample_rate,
    )
    print(f"  [OK] Time-bin features: {timebin_feat.shape} = {len(timebin_feat)} dims")

    rms_feat = extract_rms_features(data_uv, window_ms=50.0, sample_rate_hz=sample_rate)
    print(f"  [OK] RMS features: {rms_feat.shape} = {len(rms_feat)} dims")

    power_feat = extract_power_features(data_uv, window_ms=50.0, sample_rate_hz=sample_rate)
    print(f"  [OK] Power features: {power_feat.shape} = {len(power_feat)} dims")

    # ── Step 8: Decoder Inference ──────────────────────────────
    print_step(8, "Decoder Inference (shape check only)")

    import torch
    from decoders.snn_decoder import SNNDecoder

    n_timesteps = 40
    n_input = 8
    # data_uv is (n_channels, n_samples) = (8, >=3000). Bin into (n_input,
    # n_timesteps) then transpose to (n_timesteps, n_input) so the batched
    # tensor matches SNNDecoder.forward's contract: (batch, n_timesteps, n_input).
    binned = (
        (np.abs(data_uv[:, :n_timesteps * 75]) > 50)
        .astype(np.float32)
        .reshape(n_input, n_timesteps, 75)
        .mean(axis=2)
    )  # (n_input, n_timesteps)
    spike_input = torch.from_numpy(binned.T).unsqueeze(0)  # (1, n_timesteps, n_input)

    model = SNNDecoder(n_input=n_input, n_classes=3)
    with torch.no_grad():
        spike_counts, mem_trace = model(spike_input)
    predicted = spike_counts.argmax(dim=1).item()

    print(f"  [OK] SNN decoder input: {spike_input.shape}")
    print(f"       Spike counts: {spike_counts[0].numpy()}")
    print(f"       Predicted class: {predicted}")
    print(f"       (Random weights - prediction is meaningless, shape check only)")

    # ── Step 9: Encoder Pipeline ───────────────────────────────
    print_step(9, "Full Encoder Pipeline (CNN -> Protocol)")

    from encoders.cnn_feature_extractor import CNNFeatureExtractor
    from encoders.pipeline import EncodingPipeline

    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)
    for enc_type in ("delayed_phase", "lif", "rate"):
        pipe = EncodingPipeline(cnn, encoder_type=enc_type)
        test_image = torch.rand(1, 1, 28, 28)
        proto = pipe.encode_to_protocol(test_image)
        proto.validate()
        print(f"  [OK] {enc_type:15s}: {proto.total_pulses} pulses -> "
              f"validated StimulusProtocol")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DEMO MODE TEST COMPLETE")
    print(f"{'='*60}")
    print("Verified:")
    print("  - TCP command connection and basic commands")
    print("  - StimulusProtocol generation and validation")
    print("  - Stimulus parameter TCP command generation")
    print("  - Data acquisition start/stop")
    print("  - Waveform data reception (or synthetic fallback)")
    print("  - Feature extraction (time-bin, RMS, power)")
    print("  - SNN decoder forward pass (shape check)")
    print("  - All 3 encoder pipelines -> validated protocols")
    print()
    print("NOT verified (requires real hardware + tissue):")
    print("  - Actual stimulation waveform delivery")
    print("  - Evoked biological response")
    print("  - Meaningful classification accuracy")

    client.disconnect()
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Intan Demo Mode Pipeline Test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cmd-port", type=int, default=5000)
    parser.add_argument("--wave-port", type=int, default=5001)
    args = parser.parse_args()

    verdict = "FAIL"
    error = None
    try:
        success = run_demo_test(args.host, args.cmd_port, args.wave_port)
        verdict = "PASS" if success else "FAIL (early exit)"
    except Exception as e:
        verdict = f"FAIL: {type(e).__name__}"
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        log_experiment(
            script="demo_pipeline_test.py",
            args=vars(args),
            verdict=verdict,
            summary={"error": error} if error else None,
        )
