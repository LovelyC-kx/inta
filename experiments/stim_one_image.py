"""
Encode ONE MNIST image with the trained CNN + a chosen encoder, then send
the resulting StimulusProtocol to RHX via TCP and trigger it. Designed
purely to validate that:

  1. CNN-derived features + encoder produce a sensible StimulusProtocol
  2. TCP send + UploadStimParameters + ManualStimTriggerPulse actually
     causes RHX to commit and fire real current
  3. RHX's stim.dat captures the resulting pulses

No biology / no decoder / no closed loop. Just stim out, then verify.

Workflow:
  1. User: RHX must be in Record mode BEFORE running this script
  2. Run this script (sends one protocol, triggers all stim channels)
  3. User: click Stop in RHX after seeing the artifact / waiting a few seconds
  4. Run verify_stim_dat.py on the RHX-created subdir to confirm stim landed

Usage:
  python experiments/stim_one_image.py \
      --cnn-checkpoint checkpoints/cnn_mnist_<ts>.pt \
      --encoding lif --digit 0
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intan_interface.tcp_client import IntanTCPClient
from encoders.cnn_feature_extractor import CNNFeatureExtractor
from encoders.pipeline import EncodingPipeline
from utils.experiment_log import log_experiment


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cnn-checkpoint", required=True,
                   help="Path to trained CNN .pt from train_cnn_on_mnist.py")
    p.add_argument("--encoding", default="lif",
                   choices=["delayed_phase", "lif", "rate"])
    p.add_argument("--digit", type=int, default=0,
                   help="MNIST digit to encode (0-9)")
    p.add_argument("--mnist-root", default="./data/mnist")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--cmd-port", type=int, default=5000)
    p.add_argument("--no-trigger", action="store_true",
                   help="Send config only, do not actually trigger stim "
                        "(useful for dry verification of Set commands)")
    args = p.parse_args()

    # ── Load CNN ─────────────────────────────────────────────
    print(f"Loading CNN checkpoint: {args.cnn_checkpoint}")
    ckpt = torch.load(args.cnn_checkpoint, map_location="cpu", weights_only=False)
    cnn = CNNFeatureExtractor(
        n_channels=ckpt["n_channels"],
        n_classes=ckpt["n_classes"],
        input_size=ckpt.get("input_size", 28),
    )
    cnn.load_state_dict(ckpt["state_dict"])
    cnn.eval()
    print(f"  CNN val acc: {ckpt['final_val_acc']:.2%}")
    print(f"  n_channels: {ckpt['n_channels']}")

    # ── Get one MNIST digit ──────────────────────────────────
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root=args.mnist_root, train=True,
                           download=True, transform=transform)
    matching_idxs = [i for i, (_, lbl) in enumerate(mnist) if lbl == args.digit]
    if not matching_idxs:
        sys.exit(f"No MNIST digit {args.digit} found")
    idx = matching_idxs[0]
    img, _ = mnist[idx]
    print(f"Selected MNIST digit {args.digit} (dataset index {idx}, shape {tuple(img.shape)})")

    # ── Encode ───────────────────────────────────────────────
    pipe = EncodingPipeline(cnn, encoder_type=args.encoding, trim_warn=False)
    proto = pipe.encode_to_protocol(img)
    proto.validate()
    proto.validate_safety()

    report = proto.safety_report()
    print(f"\nEncoded → {proto.total_pulses} pulses")
    print(f"  per-channel pulse counts: {proto.pulses_per_channel}")
    print(f"  total charge: {report['total_charge_nc']:.1f} nC (budget 200 nC)")
    print(f"  max amplitude: {report['max_amplitude_ua']:.1f} µA")
    print(f"  max rate per channel: {report['max_rate_hz']:.1f} Hz")

    # ── Connect TCP ──────────────────────────────────────────
    print(f"\nConnecting to RHX at {args.host}:{args.cmd_port} ...")
    client = IntanTCPClient(host=args.host, command_port=args.cmd_port)
    client.connect_command(timeout=5.0)
    print(f"[OK] Command socket connected")

    # Snapshot data/ BEFORE user clicks Record — so that after the run,
    # whatever new dir appeared is the one we want.
    from pathlib import Path
    import os
    data_root = Path("data")
    before_dirs = set(os.listdir(data_root)) if data_root.exists() else set()

    print(f"\n[ACTION 1/2] In RHX, click ● Record (red circle button) NOW.")
    print(f"             RHX should show 'Recording...' and a file size that")
    print(f"             starts counting up.")
    try:
        input(f"             Press ENTER here once you've clicked Record: ")
    except EOFError:
        print("[WARN] no stdin, assuming user clicked Record; sleeping 2s")
        time.sleep(2.0)

    # ── Configure + play protocol in time ───────────────────
    print(f"\nConfiguring + playing {proto.total_pulses}-pulse protocol "
          f"({proto.duration_ms:.0f} ms duration)...")
    if args.no_trigger:
        print("[INFO] --no-trigger set, only configure (no triggers)")
        t0 = time.perf_counter()
        client.send_protocol_config(proto)
        t1 = time.perf_counter()
        print(f"[OK] Config sent in {(t1-t0)*1000:.0f} ms (skipped triggers)")
        timing_log = []
    else:
        t0 = time.perf_counter()
        timing_log = client.play_protocol_in_time(proto)
        t1 = time.perf_counter()
        print(f"[OK] Config + {len(timing_log)} timed triggers done in {(t1-t0)*1000:.0f} ms")
        print()
        print(f"  pulse | target_ms | actual_ms | drift_ms | channel")
        print(f"  ------+-----------+-----------+----------+--------")
        for i, e in enumerate(timing_log):
            print(f"  {i:>5d} | {e['target_ms']:>9.1f} | {e['actual_ms']:>9.3f} "
                  f"| {e['drift_ms']:>+8.3f} | A-{e['channel']:03d}")

    # Hold a moment so RHX captures the post-trigger window
    time.sleep(1.0)

    print(f"\n[ACTION 2/2] In RHX, click ■ Stop (black square button) NOW.")
    print(f"             This finalizes amplifier.dat + stim.dat to disk.")
    try:
        input(f"             Press ENTER here once you've clicked Stop: ")
    except EOFError:
        print("[WARN] no stdin, sleeping 3s for RHX to flush")
        time.sleep(3.0)

    client.disconnect()
    print(f"[OK] TCP disconnected.")

    # ── Find the RHX-created dir ─────────────────────────────
    after_dirs = set(os.listdir(data_root)) if data_root.exists() else set()
    new_dirs = sorted(after_dirs - before_dirs)
    rhx_dir = None
    for name in new_dirs:
        candidate = data_root / name
        if candidate.is_dir() and (candidate / "stim.dat").exists():
            rhx_dir = candidate

    if rhx_dir is None:
        print(f"\n[FAIL] No new RHX directory found in data/ after this run.")
        print(f"       This means RHX did NOT actually enter Record mode.")
        print(f"       Possible causes:")
        print(f"         - You clicked Record but RHX rejected (status didn't change)")
        print(f"         - You clicked the wrong button")
        print(f"         - RHX has a config issue preventing record")
        if new_dirs:
            print(f"       New entries (no stim.dat): {new_dirs}")
        sys.exit(1)

    print(f"\n[OK] RHX wrote to: {rhx_dir}")
    print(f"     stim.dat size: {(rhx_dir / 'stim.dat').stat().st_size} bytes")

    # ── Auto-verify ──────────────────────────────────────────
    print(f"\n[VERIFY] Running stim verification...")
    print()
    from experiments.verify_stim_dat import summarize
    summarize(rhx_dir, n_channels=64, plot=True)

    log_experiment(
        script="stim_one_image.py",
        args=vars(args),
        verdict="PROTOCOL_SENT",
        summary={
            "encoding": args.encoding,
            "digit": int(args.digit),
            "total_pulses": int(proto.total_pulses),
            "total_charge_nc": round(float(report["total_charge_nc"]), 2),
            "n_triggers_fired": len(timing_log),
            "max_drift_ms": round(max((abs(e["drift_ms"]) for e in timing_log), default=0.0), 3),
            "cnn_val_acc": round(float(ckpt["final_val_acc"]), 4),
        },
    )


if __name__ == "__main__":
    main()
