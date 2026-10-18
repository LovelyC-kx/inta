"""
Launch real-time experiment with web dashboard.

Starts the Flask dashboard server in a background thread,
then runs the experiment engine in the main thread.
Dashboard receives live updates via the DashboardBridge.

Usage:
  python run_with_dashboard.py --mode demo --port 8050
  python run_with_dashboard.py --mode closedloop --n-trials 50

Then open http://localhost:8050 in your browser.
"""

import sys
import os
import threading
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from frontend.app import run_server, DashboardBridge
from experiments.realtime_loop import RealtimeLoop
from encoders.cnn_feature_extractor import CNNFeatureExtractor
from encoders.pipeline import EncodingPipeline
from decoders.snn_decoder import SNNDecoder


def main():
    parser = argparse.ArgumentParser(description="Experiment + Dashboard")
    parser.add_argument("--mode", default="demo",
                        choices=["demo", "classify", "closedloop"])
    parser.add_argument("--host", default="127.0.0.1",
                        help="Intan RHX host")
    parser.add_argument("--cmd-port", type=int, default=5000)
    parser.add_argument("--wave-port", type=int, default=5001)
    parser.add_argument("--port", type=int, default=8050,
                        help="Dashboard web port")
    parser.add_argument("--n-rec", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--isi", type=float, default=5.0)
    parser.add_argument("--encoding", default="delayed_phase",
                        choices=["delayed_phase", "lif", "rate"])
    parser.add_argument("--live-stim", action="store_true",
                        help="Enable real stimulation. Default: dry-run only. "
                             "When set, requires interactive ENABLE_STIM confirmation.")
    parser.add_argument("--snn-checkpoint", default=None,
                        help="Path to .pt checkpoint from pretrain_snn_decoder.py. "
                             "If omitted, the SNN runs with random weights and "
                             "predictions are meaningless.")
    args = parser.parse_args()

    # ── Mandatory two-step confirmation for live stim ──────────
    dry_run = not args.live_stim
    if not dry_run:
        print("\n" + "!" * 70)
        print("!  WARNING: REAL STIMULATION ENABLED")
        print("!  This will send actual current commands to Intan RHS hardware.")
        print("!  Confirm:")
        print("!    1. Oscilloscope verification has been done with dummy load")
        print("!    2. Tissue/electrode/preparation is at expected state")
        print("!    3. Emergency stop is reachable")
        print("!" * 70)
        try:
            answer = input("Type 'ENABLE_STIM' (uppercase, exactly) to proceed: ")
        except EOFError:
            answer = ""
        if answer.strip() != "ENABLE_STIM":
            print("[ABORT] Confirmation phrase not provided. Exiting in dry-run safe mode.")
            print("        To enable live stim you must type ENABLE_STIM exactly.")
            return

    # Start dashboard server in background thread
    print(f"Starting dashboard on http://localhost:{args.port}")
    dash_thread = threading.Thread(
        target=run_server,
        kwargs={"port": args.port, "debug": False},
        daemon=True,
    )
    dash_thread.start()

    # Build experiment components
    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)
    pipeline = EncodingPipeline(cnn, encoder_type=args.encoding)

    if args.snn_checkpoint:
        ckpt = torch.load(args.snn_checkpoint, map_location="cpu", weights_only=False)
        snn_model = SNNDecoder(n_input=ckpt["n_input"], n_classes=ckpt["n_classes"])
        snn_model.load_state_dict(ckpt["state_dict"])
        snn_model.eval()
        print(f"[OK] Loaded SNN checkpoint: {args.snn_checkpoint}")
        print(f"     n_input={ckpt['n_input']}, n_classes={ckpt['n_classes']}, "
              f"trained_at={ckpt.get('trained_at', 'unknown')}")
    else:
        snn_model = SNNDecoder(n_input=args.n_rec, n_classes=3)
        print("[WARN] No --snn-checkpoint given. SNN has random weights; "
              "predictions are meaningless until trained.")

    # Test images
    images_by_class = {}
    for digit in [0, 1, 7]:
        try:
            from torchvision import datasets, transforms
            transform = transforms.Compose([transforms.ToTensor()])
            mnist = datasets.MNIST(root="./data/mnist", train=True,
                                   download=True, transform=transform)
            idxs = [i for i, (_, l) in enumerate(mnist) if l == digit][:2]
            images_by_class[digit] = [mnist[i][0] for i in idxs]
        except Exception:
            images_by_class[digit] = [
                torch.rand(1, 28, 28) * (0.3 + 0.05 * digit)
                for _ in range(2)
            ]

    # Create experiment engine with dashboard bridge
    bridge = DashboardBridge()
    bridge.update_system_status(dry_run=dry_run)
    engine = RealtimeLoop(
        host=args.host,
        command_port=args.cmd_port,
        waveform_port=args.wave_port,
        n_rec_channels=args.n_rec,
        dashboard=bridge,
        dry_run=dry_run,
    )

    print(f"\nOpen http://localhost:{args.port} in your browser")
    print("Then the experiment will start automatically.\n")

    import time
    time.sleep(2)  # Let dashboard start

    try:
        engine.connect()
        engine.run_classification_loop(
            images_by_class=images_by_class,
            encoder_pipeline=pipeline,
            snn_model=snn_model,
            n_trials=args.n_trials,
            isi_s=args.isi,
            enable_feedback=(args.mode == "closedloop"),
        )
    except ConnectionRefusedError:
        print("[FAIL] Cannot connect to Intan RHX.")
        print("Dashboard is still running — you can test the UI.")
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        engine.disconnect()


if __name__ == "__main__":
    main()
