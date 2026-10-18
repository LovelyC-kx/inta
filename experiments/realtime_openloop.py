"""
Real-time OPEN-LOOP dry-run demo.

Trial-by-trial pipeline exercise. No real current leaves the headstage —
send_protocol_config is called with upload_to_hardware=False (the new
default). RHX accepts triggers but fires the silent initial-state stim
params, so stim.dat stays empty. The pipeline still gets exercised end to
end because the 5002 spike channel keeps producing events from noise and
artefacts, which is enough to drive the SNN decoder forward.

What this verifies:
  * encoder → protocol generation works for real MNIST images
  * Set burst + trigger does not crash the RHX command channel
  * 5002 spike socket keeps streaming for the whole session
  * SNN decoder forward pass produces a prediction every trial
  * per-trial latency stays bounded (target < ISI)
  * no closed-loop feedback path is taken

Usage:
    python experiments/realtime_openloop.py \
        --cnn-checkpoint checkpoints/cnn_mnist_<ts>.pt \
        --snn-checkpoint checkpoints/snn_synthetic_500ep.pt \
        --n-trials 10 --isi 2.0

Pre-flight (each new run):
  1. RHX Demo or hardware mode, sample rate 30 kS/s
  2. Network → Remote TCP Control: Command (5000) + Spike (5002) both Pending
  3. RHX must be in Run mode (Run button) — script does not toggle RunMode
  4. (optional) click Record if you want amplifier.dat + stim.dat saved
"""

import argparse
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intan_interface.tcp_client import IntanTCPClient
from encoders.cnn_feature_extractor import CNNFeatureExtractor
from encoders.pipeline import EncodingPipeline
from decoders.snn_decoder import SNNDecoder
from utils.experiment_log import log_experiment


# Spike packet format (5002): see docs/TCP_SPIKE_OUTPUT_5002_GUIDE.md
SPIKE_MAGIC = struct.pack("<I", 0x3AE2710F)
SPIKE_PKT_SIZE = 14
SAMPLE_RATE_HZ = 30000


class SpikeStreamReader:
    """Background reader of the 5002 spike socket.

    Parses Intan's 14-byte spike packet (magic + 'A-xxx' channel name +
    uint32 sample timestamp + 1-byte type) and pushes (channel_idx,
    timestamp_samples, wallclock_t) tuples into a queue.
    """

    def __init__(self, sock, wallclock_origin):
        self.sock = sock
        self.t0 = wallclock_origin
        self.buf = bytearray()
        self.queue: queue.Queue = queue.Queue()
        self.raw_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.sock.settimeout(0.05)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            self.raw_bytes += len(data)
            self.buf.extend(data)
            now = time.perf_counter() - self.t0
            self._drain(now)

    def _drain(self, wallclock_now):
        buf = self.buf
        while True:
            pos = buf.find(SPIKE_MAGIC)
            if pos == -1:
                if len(buf) > 3:
                    del buf[: len(buf) - 3]
                return
            if pos > 0:
                del buf[:pos]
            if len(buf) < SPIKE_PKT_SIZE:
                return
            name = bytes(buf[4:9])
            ts = struct.unpack_from("<I", buf, 9)[0]
            del buf[:SPIKE_PKT_SIZE]
            try:
                ch = int(name.decode("ascii").split("-")[1])
            except (ValueError, UnicodeDecodeError, IndexError):
                continue
            self.queue.put((ch, ts, wallclock_now))


def load_mnist_subset(digits, root="./data/mnist", n_per_class=4):
    """Return {label: [tensor(1,28,28), ...]}."""
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    out = {d: [] for d in digits}
    for img, lbl in mnist:
        if lbl in out and len(out[lbl]) < n_per_class:
            out[lbl].append(img)
        if all(len(v) >= n_per_class for v in out.values()):
            break
    return out


def collect_trial_spikes(reader, t_start, t_end, n_channels):
    """Drain the spike queue, keep events that fired within [t_start, t_end].

    Returns ndarray (n_events, 2): cols = (channel_idx, relative_time_s).
    """
    keep = []
    while True:
        try:
            ch, _ts, wt = reader.queue.get_nowait()
        except queue.Empty:
            break
        if t_start <= wt < t_end and 0 <= ch < n_channels:
            keep.append((ch, wt - t_start))
    if not keep:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(keep, dtype=np.float32)


def bin_spikes_for_snn(events, evoked_window_s, n_channels, n_timesteps):
    """(n_events, 2) → (n_timesteps, n_channels) float32 spike counts.

    Matches the input shape SNNDecoder.forward expects after batch unsqueeze.
    """
    binned = np.zeros((n_timesteps, n_channels), dtype=np.float32)
    if events.shape[0] == 0:
        return binned
    bin_width = evoked_window_s / n_timesteps
    bin_idx = np.clip((events[:, 1] / bin_width).astype(np.int64),
                      0, n_timesteps - 1)
    ch_idx = events[:, 0].astype(np.int64)
    for t, c in zip(bin_idx, ch_idx):
        binned[t, c] += 1.0
    return binned


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cnn-checkpoint", required=True,
                   help="CNN checkpoint from train_cnn_on_mnist.py")
    p.add_argument("--snn-checkpoint", required=True,
                   help="SNN checkpoint from pretrain_snn_decoder.py")
    p.add_argument("--digits", nargs="+", type=int, default=[0, 1, 7])
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--isi", type=float, default=2.0,
                   help="Inter-trial interval in seconds")
    p.add_argument("--evoked-window-ms", type=float, default=500.0,
                   help="Spike collection window after each trigger")
    p.add_argument("--n-stim-channels", type=int, default=8)
    p.add_argument("--n-rec-channels", type=int, default=8,
                   help="Number of A-0xx channels to enable on 5002")
    p.add_argument("--encoding", default="lif",
                   choices=["delayed_phase", "lif", "rate"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--cmd-port", type=int, default=5000)
    p.add_argument("--spike-port", type=int, default=5002)
    p.add_argument("--mnist-root", default="./data/mnist")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--upload-stim", action="store_true",
                   help="Send Execute UploadStimParameters so RHX actually "
                        "fires stim on each trigger. In Demo mode this is "
                        "safe (no real hardware) and lets you see stim "
                        "events on the RHX screen + in stim.dat. With real "
                        "hardware DO NOT enable without prior oscilloscope "
                        "verification (CHECKLIST Gate A).")
    args = p.parse_args()

    print("=" * 64)
    print("  REAL-TIME OPEN-LOOP DRY-RUN")
    print("=" * 64)
    print(f"  digits          : {args.digits}")
    print(f"  trials          : {args.n_trials}   ISI: {args.isi}s")
    print(f"  evoked window   : {args.evoked_window_ms} ms")
    print(f"  stim channels   : 0..{args.n_stim_channels-1}  (A-000..A-{args.n_stim_channels-1:03d})")
    print(f"  rec  channels   : A-000..A-{args.n_rec_channels-1:03d} via 5002 spike output")
    print(f"  encoding        : {args.encoding}")
    print(f"  SNN ckpt        : {args.snn_checkpoint}")
    print(f"  upload to HW    : {'TRUE  (RHX will fire real stim events)' if args.upload_stim else 'FALSE (dry-run; no real current)'}")
    print("=" * 64)

    # ── Load models ──────────────────────────────────────────────
    print("\n[load] CNN ...", end="", flush=True)
    cnn_ckpt = torch.load(args.cnn_checkpoint, map_location="cpu", weights_only=False)
    cnn = CNNFeatureExtractor(
        n_channels=cnn_ckpt["n_channels"],
        n_classes=cnn_ckpt["n_classes"],
        input_size=cnn_ckpt.get("input_size", 28),
    )
    cnn.load_state_dict(cnn_ckpt["state_dict"])
    cnn.eval()
    print(f" OK (val_acc={cnn_ckpt['final_val_acc']:.2%})")

    print("[load] SNN ...", end="", flush=True)
    snn_ckpt = torch.load(args.snn_checkpoint, map_location="cpu", weights_only=False)
    snn = SNNDecoder(n_input=snn_ckpt["n_input"], n_classes=snn_ckpt["n_classes"])
    snn.load_state_dict(snn_ckpt["state_dict"])
    snn.eval()
    n_timesteps = snn_ckpt["n_timesteps"]
    n_classes = snn_ckpt["n_classes"]
    print(f" OK (n_in={snn_ckpt['n_input']}, n_cls={n_classes}, n_t={n_timesteps})")

    if snn_ckpt["n_input"] != args.n_rec_channels:
        print(f"  [WARN] SNN expects {snn_ckpt['n_input']} channels, but "
              f"--n-rec-channels={args.n_rec_channels}. Using SNN's value.")
        args.n_rec_channels = snn_ckpt["n_input"]

    pipeline = EncodingPipeline(cnn, encoder_type=args.encoding, trim_warn=True)

    print("[load] MNIST ...", end="", flush=True)
    images = load_mnist_subset(args.digits, root=args.mnist_root)
    print(f" OK ({sum(len(v) for v in images.values())} images across "
          f"{len(images)} classes)")

    if len(args.digits) != n_classes:
        print(f"  [WARN] {len(args.digits)} digits given but SNN trained on "
              f"{n_classes} classes. Decoder output will be modulo n_classes.")

    digit_to_class = {d: i for i, d in enumerate(sorted(args.digits))}

    # ── Pre-build trial sequence ─────────────────────────────────
    rng = np.random.default_rng(args.seed)
    trial_plan = []
    for _ in range(args.n_trials):
        d = int(rng.choice(args.digits))
        img_idx = int(rng.integers(0, len(images[d])))
        trial_plan.append((d, img_idx))

    # ── Connect to RHX ───────────────────────────────────────────
    print("\n[tcp ] connect command 5000 ...", end="", flush=True)
    client = IntanTCPClient(host=args.host, command_port=args.cmd_port,
                            spike_port=args.spike_port)
    client.connect_command(timeout=5.0)
    print(" OK")

    # First Get often hits a backlog timeout if RHX just transitioned
    # Pending -> Connected. Retry quietly a couple of times; if it still
    # fails just continue — the rest of the protocol does not depend on it.
    runmode = None
    for attempt in range(3):
        try:
            runmode = client.get("RunMode")
            break
        except Exception as e:
            print(f"       (Get RunMode attempt {attempt+1}/3: {type(e).__name__})")
            time.sleep(1.5)
    if runmode is None:
        print("       [WARN] Could not read RunMode after retries. "
              "Continuing anyway. Make sure RHX is in Run/Record mode.")
    else:
        print(f"       RunMode: {runmode}")
        if "Run" not in runmode and "Record" not in runmode:
            print("       [WARN] RHX is not running. 5002 will be silent. "
                  "Start Run in RHX before continuing.")

    # MEMORY rule: connect_spike FIRST, then Set ...TCPDataOutputEnabledSpike
    print("[tcp ] connect spike 5002 ...", end="", flush=True)
    client.connect_spike(timeout=5.0)
    print(" OK")

    print(f"[tcp ] enable {args.n_rec_channels} channels for 5002 spike "
          f"output ...", end="", flush=True)
    for ch in range(args.n_rec_channels):
        client.send_command(
            f"Set A-{ch:03d}.TCPDataOutputEnabledSpike True",
            expect_response=False,
        )
        time.sleep(client.INTER_COMMAND_DELAY_S)
    print(" OK")

    # ── Start background spike reader ───────────────────────────
    t_origin = time.perf_counter()
    reader = SpikeStreamReader(client._spike_socket, t_origin)
    reader.start()

    print("[tcp ] warming up 5002 (1 s) ...", end="", flush=True)
    time.sleep(1.0)
    print(f" got {reader.raw_bytes} bytes")
    if reader.raw_bytes == 0:
        print("       [WARN] zero bytes from 5002 — is RHX in Run mode? "
              "is something connected to the Spike port? continuing anyway.")

    # ── Output dir ───────────────────────────────────────────────
    out_dir = Path(f"data/openloop_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[out ] {out_dir}")

    # ── Trial loop ──────────────────────────────────────────────
    evoked_window_s = args.evoked_window_ms / 1000.0
    results = []
    latencies = []

    print()
    print(f"{'trial':>5}  {'true':>4}  {'pred':>4}  {'n_spikes':>9}  "
          f"{'latency_ms':>11}  {'evk_bytes':>10}")
    print("-" * 60)

    for trial_idx, (digit, img_idx) in enumerate(trial_plan):
        t_trial_start = time.perf_counter() - t_origin
        img = images[digit][img_idx]

        try:
            proto = pipeline.encode_to_protocol(img)
            proto.validate()
            # Safety still checked even in dry-run; it's free protection.
            proto.validate_safety()
        except Exception as e:
            print(f"  trial {trial_idx} encode failed: {e}")
            continue

        # Configure stim. If --upload-stim, commit params to hardware so the
        # subsequent triggers actually fire.
        client.send_protocol_config(proto, upload_to_hardware=args.upload_stim)

        # Trigger each configured stim channel. RHX accepts triggers but
        # the hardware uploaded params are still silent (amp=0), so nothing
        # actually fires. From the 5002 stream's POV, this trial window is
        # just "look at the next 500 ms of noise".
        t_stim = time.perf_counter() - t_origin
        for ch in range(args.n_stim_channels):
            client.send_command(
                f"Execute ManualStimTriggerPulse F{ch + 1}",
                expect_response=False,
            )
            time.sleep(client.INTER_COMMAND_DELAY_S)

        # Spike collection window
        raw_before = reader.raw_bytes
        time.sleep(evoked_window_s)
        raw_after = reader.raw_bytes
        evk_bytes = raw_after - raw_before

        t_window_end = time.perf_counter() - t_origin
        events = collect_trial_spikes(
            reader, t_stim, t_window_end, args.n_rec_channels,
        )

        # SNN decode
        binned = bin_spikes_for_snn(
            events, evoked_window_s, args.n_rec_channels, n_timesteps,
        )
        with torch.no_grad():
            x = torch.from_numpy(binned).unsqueeze(0)  # (1, n_t, n_in)
            spike_counts, _ = snn(x)
            pred_class = int(spike_counts.argmax(dim=1).item())

        true_class = digit_to_class.get(digit, digit % n_classes)
        is_correct = (pred_class == true_class)

        t_trial_end = time.perf_counter() - t_origin
        latency_ms = (t_trial_end - t_trial_start) * 1000.0
        latencies.append(latency_ms)

        results.append({
            "trial": trial_idx,
            "digit": digit,
            "true_class": true_class,
            "pred_class": pred_class,
            "is_correct": is_correct,
            "n_events": int(events.shape[0]),
            "evk_bytes": int(evk_bytes),
            "latency_ms": round(latency_ms, 2),
            "spike_counts": spike_counts[0].tolist(),
        })

        flag = "+" if is_correct else "-"
        print(f"{trial_idx:>5}  {digit:>4}  {pred_class:>3}{flag}  "
              f"{events.shape[0]:>9}  {latency_ms:>11.1f}  {evk_bytes:>10}")

        # ISI: account for time already spent in this trial
        sleep_left = args.isi - (time.perf_counter() - t_origin - t_trial_start)
        if sleep_left > 0:
            time.sleep(sleep_left)

    # ── Wrap up ─────────────────────────────────────────────────
    reader.stop()
    client.disconnect()

    n_correct = sum(1 for r in results if r["is_correct"])
    acc = n_correct / max(len(results), 1)

    summary = {
        "n_trials": len(results),
        "n_correct": n_correct,
        "accuracy": acc,
        "chance": 1.0 / n_classes,
        "latency_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
        "latency_ms_max": float(np.max(latencies)) if latencies else 0.0,
        "isi_s": args.isi,
        "evoked_window_ms": args.evoked_window_ms,
        "encoding": args.encoding,
        "snn_checkpoint": args.snn_checkpoint,
        "cnn_checkpoint": args.cnn_checkpoint,
        "total_spike_bytes": reader.raw_bytes,
        "dry_run": not args.upload_stim,
        "upload_stim": bool(args.upload_stim),
        "trials": results,
    }
    (out_dir / "openloop_results.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 64)
    print(f"  {len(results)} trials done, {n_correct} correct "
          f"({acc:.1%}, chance {1/n_classes:.1%})")
    if latencies:
        print(f"  latency: mean {np.mean(latencies):.1f} ms, "
              f"max {np.max(latencies):.1f} ms")
    print(f"  total 5002 bytes received: {reader.raw_bytes}")
    print(f"  results: {out_dir / 'openloop_results.json'}")
    print("=" * 64)

    log_experiment(
        script="realtime_openloop.py",
        args={
            "digits": args.digits,
            "n_trials": args.n_trials,
            "isi_s": args.isi,
            "evoked_window_ms": args.evoked_window_ms,
            "n_stim_channels": args.n_stim_channels,
            "n_rec_channels": args.n_rec_channels,
            "encoding": args.encoding,
            "snn_checkpoint": args.snn_checkpoint,
            "cnn_checkpoint": args.cnn_checkpoint,
        },
        verdict="COMPLETE" if results else "NO_TRIALS",
        summary={
            "n_trials_done": len(results),
            "n_correct": n_correct,
            "accuracy": round(acc, 4),
            "chance": round(1.0 / n_classes, 4),
            "latency_ms_mean": round(summary["latency_ms_mean"], 2),
            "latency_ms_max": round(summary["latency_ms_max"], 2),
            "total_spike_bytes": reader.raw_bytes,
            "dry_run": not args.upload_stim,
            "upload_stim": bool(args.upload_stim),
        },
        data=str(out_dir),
    )


if __name__ == "__main__":
    main()
