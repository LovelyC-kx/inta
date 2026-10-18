"""
Step 2: Real 2D Culture Experiment.

Full experiment pipeline for 2D dissociated culture on Intan RHS 128ch.

Usage:
  python run_2d_experiment.py --phase baseline
  python run_2d_experiment.py --phase probe
  python run_2d_experiment.py --phase classify --encoding delayed_phase
  python run_2d_experiment.py --phase classify --encoding lif
  python run_2d_experiment.py --phase classify --encoding rate

Each phase saves all raw data and logs to a timestamped directory.
"""

import sys
import os
import time
import json
import shutil
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intan_interface.tcp_client import IntanTCPClient
from encoders.stimulus_schema import StimulusProtocol, Pulse
from encoders.cnn_feature_extractor import CNNFeatureExtractor
from encoders.pipeline import EncodingPipeline
from feature_extraction.extractors import (
    extract_timebin_features,
    extract_rms_features,
)
from utils.experiment_log import log_experiment


# ── Configuration ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "command_port": 5000,
    "waveform_port": 5001,
    "spike_port": 5002,
    "sample_rate_hz": 30000,
    "n_stim_channels": 8,
    "n_rec_channels": 128,
    "stim_duration_ms": 2000,
    "inter_trial_interval_s": 15,
    "n_repetitions": 50,
    "mnist_digits": [0, 1, 7],
    "spontaneous_record_duration_s": 300,
    "evoked_window_ms": 500,
    "artifact_blanking_ms": 10,
}


def create_output_dir(phase_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"data/{phase_name}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_config(out_dir, config):
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_array(out_dir, name, arr):
    np.save(out_dir / f"{name}.npy", arr)


def log(out_dir, msg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(out_dir / "experiment.log", "a") as f:
        f.write(line + "\n")


def get_rhx_save_root(client, fallback="data"):
    """
    Ask RHX where it actually writes files. Falls back to `fallback` on timeout.

    Background: setting `Filename.Path` via TCP is unreliable (RHX may ignore
    backslash paths on Windows or silently keep the GUI value), so the only
    safe way to know where amplifier.dat lands is to query RHX before any
    recording starts.
    """
    try:
        resp = client.get("Filename.Path")
        # "Return: Filename.path F:/YKX/Intan-main1/Intan-main/data"
        return resp.split(maxsplit=2)[-1]
    except Exception as e:
        print(f"[WARN] Could not query Filename.Path ({e}); using fallback '{fallback}'")
        return fallback


def find_new_rhx_subdir(rhx_root, before_snapshot):
    """
    Diff os.listdir(rhx_root) against `before_snapshot` to find the subdir
    RHX just created. Returns Path to the new dir, or None.

    RHX names subdirs `<BaseFilename>_YYMMDD_HHMMSS`. If multiple new dirs
    exist (shouldn't happen in normal flow), picks the lexicographically
    largest, which is the most recent.
    """
    rhx_root = Path(rhx_root)
    if not rhx_root.exists():
        return None
    after = set(os.listdir(rhx_root))
    new = after - before_snapshot
    if not new:
        return None
    return rhx_root / sorted(new)[-1]


def attach_metadata_to_rhx_dir(rhx_dir, our_dir, files=("config.json",
                                                          "trial_log.json")):
    """
    Copy our Python-side metadata into the RHX-generated dir so downstream
    scripts (pretrain_snn_decoder, analyze_results) can read one directory.

    Idempotent: missing source files are silently skipped.
    """
    if rhx_dir is None:
        return
    for fname in files:
        src = Path(our_dir) / fname
        if src.exists():
            shutil.copy(src, rhx_dir / fname)


# ── Phase: Baseline ───────────────────────────────────────────

def run_baseline(client, config, out_dir):
    """
    Record spontaneous activity. No stimulation. Pre-experiment baseline.

    RHX writes to its GUI-configured Filename.Path; we discover the
    new subdir RHX creates and copy our metadata in.
    """
    log(out_dir, "=== PHASE: BASELINE (spontaneous activity) ===")
    log(out_dir, f"Recording {config['spontaneous_record_duration_s']}s of spontaneous activity")

    duration_s = config["spontaneous_record_duration_s"]

    rhx_root = get_rhx_save_root(client)
    log(out_dir, f"RHX save root (queried): {rhx_root}")
    before = set(os.listdir(rhx_root)) if Path(rhx_root).exists() else set()

    log(out_dir, "Starting recording...")
    client.record()

    for elapsed in range(0, duration_s, 30):
        remaining = duration_s - elapsed
        log(out_dir, f"  Recording... {remaining}s remaining")
        time.sleep(min(30, remaining))

    client.stop()
    log(out_dir, "Recording stopped.")

    rhx_dir = find_new_rhx_subdir(rhx_root, before)
    log(out_dir, f"RHX wrote to: {rhx_dir}")
    attach_metadata_to_rhx_dir(rhx_dir, out_dir)

    log(out_dir, f"Python metadata: {out_dir}")
    log(out_dir, "=== BASELINE COMPLETE ===")

    log_experiment(
        script="run_2d_experiment.py",
        args={"phase": "baseline", **{k: v for k, v in config.items()
              if k in ("host", "command_port", "sample_rate_hz",
                       "n_rec_channels", "spontaneous_record_duration_s")}},
        verdict="COMPLETE" if rhx_dir else "COMPLETE (no RHX dir found)",
        summary={
            "phase": "baseline",
            "duration_s": int(config["spontaneous_record_duration_s"]),
            "n_rec_channels": int(config["n_rec_channels"]),
            "rhx_dir": str(rhx_dir) if rhx_dir else None,
        },
        data=str(rhx_dir) if rhx_dir else str(out_dir),
    )


# ── Phase: Probe ──────────────────────────────────────────────

def run_probe(client, config, out_dir):
    """
    Single-channel stimulation probing.
    Activate each stim channel one at a time, 20 repetitions.
    Map which recording channels respond to which stim channel.
    """
    log(out_dir, "=== PHASE: PROBE (single-channel stimulation) ===")

    n_stim = config["n_stim_channels"]
    n_reps = 20
    isi_s = 3  # inter-stimulus interval for probing

    client.set_filename(str(out_dir.absolute()), "probe")

    log(out_dir, f"Probing {n_stim} stim channels, {n_reps} reps each, ISI={isi_s}s")

    probe_log = []

    for ch in range(n_stim):
        ch_name = f"A-{ch:03d}"
        log(out_dir, f"  Probing channel {ch_name}...")

        # Configure single channel
        single_pulse = StimulusProtocol(
            pulses=[Pulse(channel=ch, time_ms=0.0)],
            n_channels=n_stim,
            duration_ms=config["stim_duration_ms"],
        )
        single_pulse.validate()
        client.send_protocol_config(single_pulse)

        # Start recording
        client.record()
        time.sleep(0.5)

        for rep in range(n_reps):
            # Trigger stimulation (send keypress command)
            client.send_command(
                f"Execute ManualStimTriggerPulse F{ch + 1}",
                expect_response=False,
            )
            probe_log.append({
                "channel": ch,
                "rep": rep,
                "timestamp": datetime.now().isoformat(),
            })
            time.sleep(isi_s)

        client.stop()
        log(out_dir, f"    Done. {n_reps} pulses delivered.")

    # Save probe log
    with open(out_dir / "probe_log.json", "w") as f:
        json.dump(probe_log, f, indent=2)

    log(out_dir, "=== PROBE COMPLETE ===")

    log_experiment(
        script="run_2d_experiment.py",
        args={"phase": "probe", "n_stim_channels": int(config["n_stim_channels"]),
              "n_reps_per_channel": n_reps, "isi_s": isi_s},
        verdict="COMPLETE",
        summary={
            "phase": "probe",
            "n_channels_probed": int(config["n_stim_channels"]),
            "n_pulses_total": len(probe_log),
        },
        data=str(out_dir),
    )


# ── Phase: Classify ───────────────────────────────────────────

def run_classify(client, config, out_dir, encoding_type="delayed_phase"):
    """
    Main classification experiment.
    Encode MNIST digits -> stimulate -> record -> save for offline analysis.
    """
    log(out_dir, f"=== PHASE: CLASSIFY (encoding={encoding_type}) ===")

    digits = config["mnist_digits"]
    n_reps = config["n_repetitions"]
    isi_s = config["inter_trial_interval_s"]
    n_classes = len(digits)

    # Generate synthetic MNIST-like test images
    # (replace with real MNIST when torchvision is available)
    log(out_dir, f"Generating test images for digits {digits}")
    import torch
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        mnist = datasets.MNIST(root="./data/mnist", train=True,
                               download=True, transform=transform)
        images_by_class = {}
        for digit in digits:
            idxs = [i for i, (_, label) in enumerate(mnist) if label == digit][:3]
            images_by_class[digit] = [mnist[i][0] for i in idxs]
        log(out_dir, "  Using real MNIST images")
    except Exception:
        log(out_dir, "  torchvision not available, using synthetic images")
        images_by_class = {}
        for digit in digits:
            images_by_class[digit] = [
                torch.rand(1, 28, 28) * (0.3 + 0.1 * digit) + 0.1 * digit
                for _ in range(3)
            ]

    # Build encoder pipeline
    cnn = CNNFeatureExtractor(n_channels=config["n_stim_channels"], n_classes=10)
    pipeline = EncodingPipeline(cnn, encoder_type=encoding_type)

    # Generate all protocols
    log(out_dir, "Encoding images to StimulusProtocols...")
    protocols = {}
    for digit in digits:
        protocols[digit] = []
        for img_idx, img in enumerate(images_by_class[digit]):
            proto = pipeline.encode_to_protocol(img)
            proto.validate()
            protocols[digit].append(proto)
            log(out_dir, f"  digit={digit}, img={img_idx}: "
                f"{proto.total_pulses} pulses")

    # Save protocols
    protocol_data = {}
    for digit in digits:
        protocol_data[str(digit)] = [
            {
                "binary_matrix": p.to_binary_matrix(10).tolist(),
                "total_pulses": p.total_pulses,
                "pulses_per_channel": p.pulses_per_channel,
            }
            for p in protocols[digit]
        ]
    with open(out_dir / "protocols.json", "w") as f:
        json.dump(protocol_data, f, indent=2)

    # Build trial sequence (randomized)
    trials = []
    for digit in digits:
        for img_idx in range(len(protocols[digit])):
            for rep in range(n_reps):
                trials.append((digit, img_idx, rep))

    rng = np.random.default_rng(42)
    rng.shuffle(trials)

    total_trials = len(trials)
    est_time_min = total_trials * isi_s / 60
    log(out_dir, f"Total trials: {total_trials}, "
        f"estimated time: {est_time_min:.0f} min")

    rhx_root = get_rhx_save_root(client)
    log(out_dir, f"RHX save root (queried): {rhx_root}")

    # Pre-training spontaneous activity (2 min)
    log(out_dir, "Recording 2 min pre-training spontaneous activity...")
    pre_before = set(os.listdir(rhx_root)) if Path(rhx_root).exists() else set()
    client.record()
    time.sleep(120)
    client.stop()
    pre_rhx_dir = find_new_rhx_subdir(rhx_root, pre_before)
    log(out_dir, f"  pre-spontaneous -> {pre_rhx_dir}")
    # RHX needs settle time before accepting next Record. Without this,
    # the immediately-following record() is silently dropped (see §4c).
    time.sleep(3.0)

    # Main experiment loop
    main_before = set(os.listdir(rhx_root)) if Path(rhx_root).exists() else set()
    client.record()

    trial_log = []
    for trial_idx, (digit, img_idx, rep) in enumerate(trials):
        proto = protocols[digit][img_idx]

        # Configure stimulation
        client.send_protocol_config(proto)

        # Log
        trial_entry = {
            "trial": trial_idx,
            "digit": int(digit),
            "image_idx": int(img_idx),
            "repetition": int(rep),
            "timestamp": datetime.now().isoformat(),
            "total_pulses": proto.total_pulses,
        }

        # Trigger all channels
        for ch in range(config["n_stim_channels"]):
            if any(p.channel == ch for p in proto.pulses):
                client.send_command(
                    f"Execute ManualStimTriggerPulse F{ch + 1}",
                    expect_response=False,
                )

        trial_log.append(trial_entry)

        # Progress
        if (trial_idx + 1) % 10 == 0 or trial_idx == 0:
            log(out_dir, f"  Trial {trial_idx+1}/{total_trials}: "
                f"digit={digit}, img={img_idx}, rep={rep}")

        # Inter-trial interval
        time.sleep(isi_s)

    client.stop()
    main_rhx_dir = find_new_rhx_subdir(rhx_root, main_before)
    log(out_dir, f"  classify-main -> {main_rhx_dir}")

    # Save trial log into both our metadata dir and the RHX dir
    with open(out_dir / "trial_log.json", "w") as f:
        json.dump(trial_log, f, indent=2)
    attach_metadata_to_rhx_dir(
        main_rhx_dir, out_dir,
        files=("config.json", "trial_log.json", "protocols.json"),
    )

    # Settle before next Record (see §4c).
    time.sleep(3.0)

    # Post-training spontaneous activity (2 min)
    log(out_dir, "Recording 2 min post-training spontaneous activity...")
    post_before = set(os.listdir(rhx_root)) if Path(rhx_root).exists() else set()
    client.record()
    time.sleep(120)
    client.stop()
    post_rhx_dir = find_new_rhx_subdir(rhx_root, post_before)
    log(out_dir, f"  post-spontaneous -> {post_rhx_dir}")

    log(out_dir, f"=== CLASSIFY COMPLETE ({encoding_type}) ===")
    log(out_dir, f"Python metadata: {out_dir}")
    log(out_dir, f"Main RHX dir (for pretrain): {main_rhx_dir}")
    log(out_dir, f"Next: python experiments/pretrain_snn_decoder.py "
        f"--data-dir {main_rhx_dir}")

    log_experiment(
        script="run_2d_experiment.py",
        args={"phase": "classify", "encoding_type": encoding_type,
              "digits": digits, "n_reps_per_image": n_reps,
              "isi_s": isi_s, "n_rec_channels": int(config["n_rec_channels"])},
        verdict="COMPLETE" if main_rhx_dir else "COMPLETE (no RHX dir found)",
        summary={
            "phase": "classify",
            "encoding": encoding_type,
            "n_classes": int(n_classes),
            "n_trials_total": len(trials),
            "n_images_per_class": int(len(images_by_class[digits[0]])),
            "n_reps_per_image": int(n_reps),
            "rhx_main_dir": str(main_rhx_dir) if main_rhx_dir else None,
            "rhx_pre_spontaneous_dir": str(pre_rhx_dir) if pre_rhx_dir else None,
            "rhx_post_spontaneous_dir": str(post_rhx_dir) if post_rhx_dir else None,
        },
        data=str(main_rhx_dir) if main_rhx_dir else str(out_dir),
    )


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2D Culture Experiment")
    parser.add_argument("--phase", required=True,
                        choices=["baseline", "probe", "classify"],
                        help="Experiment phase to run")
    parser.add_argument("--encoding", default="delayed_phase",
                        choices=["delayed_phase", "lif", "rate"],
                        help="Encoding type for classify phase")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cmd-port", type=int, default=5000)
    parser.add_argument("--wave-port", type=int, default=5001)
    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--isi", type=float, default=15.0,
                        help="Inter-trial interval in seconds")
    parser.add_argument("--n-rec-channels", type=int, default=None,
                        help="Number of recording channels Intan is streaming "
                             "(must match RHX 'Channels To Stream' count "
                             "AND the channels Intan is recording to disk). "
                             "Default 128.")
    parser.add_argument("--spontaneous-duration", type=int, default=None,
                        help="Override baseline phase duration in seconds "
                             "(default 300 = 5 min). Use shorter for dry-run.")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["host"] = args.host
    config["command_port"] = args.cmd_port
    config["waveform_port"] = args.wave_port
    config["n_repetitions"] = args.n_reps
    config["inter_trial_interval_s"] = args.isi
    if args.n_rec_channels is not None:
        config["n_rec_channels"] = args.n_rec_channels
    if args.spontaneous_duration is not None:
        config["spontaneous_record_duration_s"] = args.spontaneous_duration

    out_dir = create_output_dir(args.phase)
    save_config(out_dir, config)

    client = IntanTCPClient(
        host=config["host"],
        command_port=config["command_port"],
        waveform_port=config["waveform_port"],
    )

    try:
        client.connect_command(timeout=10.0)
        log(out_dir, f"Connected to Intan RHX at {config['host']}:{config['command_port']}")

        # IMPORTANT: RHX must be in Stop state before this script runs.
        # See MAINTENANCE.md §4c — TCP `Set RunMode Stop` doesn't reliably
        # take effect mid-session, so the user must manually click Stop in
        # the RHX GUI between scripts. If RHX is still recording when we
        # start, the record() calls below will be no-ops and no new RHX
        # subdir will be discovered.

        sample_rate = client.get_sample_rate()
        log(out_dir, f"Sample rate: {sample_rate} Hz")

        if args.phase == "baseline":
            run_baseline(client, config, out_dir)
        elif args.phase == "probe":
            run_probe(client, config, out_dir)
        elif args.phase == "classify":
            run_classify(client, config, out_dir, args.encoding)

    except ConnectionRefusedError:
        print("[FAIL] Cannot connect to Intan RHX.")
        print("Check that the software is running and TCP is enabled.")
    except KeyboardInterrupt:
        log(out_dir, "Interrupted by user. Stopping...")
        client.stop()
    finally:
        client.disconnect()
        log(out_dir, "Disconnected.")


if __name__ == "__main__":
    main()
