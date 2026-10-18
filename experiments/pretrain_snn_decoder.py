"""
Pre-train the SNN decoder on recorded organoid data.

Typical workflow:
  1. python experiments/run_2d_experiment.py --phase classify
     -> Intan saves amplifier.dat; we save trial_log.json.
  2. python experiments/pretrain_snn_decoder.py \
         --data-dir data/classify_<timestamp>/ \
         --out-checkpoint checkpoints/snn_<timestamp>.pt
  3. python run_with_dashboard.py --snn-checkpoint checkpoints/snn_<timestamp>.pt ...

Sanity-check (no recording needed):
  python experiments/pretrain_snn_decoder.py --synthetic

The checkpoint file stores enough metadata (n_input, n_classes, n_timesteps,
class-label mapping) to reconstruct the model in the realtime loop.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torch_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoders.snn_decoder import SNNDecoder, train_snn_decoder
from utils.experiment_log import log_experiment


# ── Spike detection & binning (matches realtime_loop's per-channel thresholding) ──

def detect_spikes_per_channel(waveform, threshold_factor=5.5):
    """
    Negative-threshold spike detection on a (n_channels, n_samples) µV array.
    Returns a binary raster of the same shape (1 = rising edge below -k*sigma).
    """
    n_ch, _ = waveform.shape
    raster = np.zeros_like(waveform, dtype=np.float32)
    for ch in range(n_ch):
        noise_std = np.median(np.abs(waveform[ch])) / 0.6745
        threshold = -threshold_factor * noise_std
        crossings = np.where(np.diff((waveform[ch] < threshold).astype(int)) == 1)[0]
        raster[ch, crossings] = 1.0
    return raster


def bin_raster(raster, n_timesteps):
    """
    (n_channels, n_samples) raster -> (n_timesteps, n_channels) binned counts.

    Matches the binning in RealtimeLoop.decode_snn: equal-width time bins,
    last bin absorbs any remainder. SNNDecoder.forward expects time as axis 1.
    """
    n_ch, n_samples = raster.shape
    bin_size = max(1, n_samples // n_timesteps)
    binned = np.zeros((n_timesteps, n_ch), dtype=np.float32)
    for t in range(n_timesteps):
        s = t * bin_size
        e = min(s + bin_size, n_samples)
        binned[t] = raster[:, s:e].sum(axis=1)
    return binned


# ── Data loaders ──────────────────────────────────────────────────

def load_trials_from_recording(data_dir, n_timesteps=40):
    """
    Parse amplifier.dat + trial_log.json into per-trial binned spike rasters.

    Intan must have been configured to save 'One File Per Signal Type'
    (gives us amplifier.dat as int16, channels interleaved per sample).

    Returns (X, y, digit_to_class):
        X.shape == (n_trials, n_timesteps, n_channels) float32
        y.shape == (n_trials,) int64 in [0, n_classes)
        digit_to_class: {original_digit: class_index} for label interpretation
    """
    data_dir = Path(data_dir)
    with open(data_dir / "config.json") as f:
        config = json.load(f)
    with open(data_dir / "trial_log.json") as f:
        trial_log = json.load(f)

    sr = config["sample_rate_hz"]
    evoked_samples = int(config["evoked_window_ms"] * sr / 1000)
    blank_samples = int(config["artifact_blanking_ms"] * sr / 1000)
    isi_samples = int(config["inter_trial_interval_s"] * sr)
    n_rec = config["n_rec_channels"]

    amp_file = data_dir / "amplifier.dat"
    if not amp_file.exists():
        raise FileNotFoundError(
            f"{amp_file} not found. In Intan RHX, set File Format to "
            f"'One File Per Signal Type' before running the classify phase."
        )

    print(f"Loading {amp_file} ({amp_file.stat().st_size / 1e6:.1f} MB)...")
    raw = np.fromfile(str(amp_file), dtype=np.int16)
    n_samples_total = len(raw) // n_rec
    raw = raw[:n_samples_total * n_rec].reshape(n_samples_total, n_rec).T
    data_uv = raw.astype(np.float32) * 0.195
    print(f"  Total: {n_samples_total / sr:.1f}s @ {n_rec} channels")

    digits = sorted({int(t["digit"]) for t in trial_log})
    digit_to_class = {d: i for i, d in enumerate(digits)}
    print(f"  Classes: {digit_to_class}")

    X_list, y_list, skipped = [], [], 0
    for trial in trial_log:
        ti = trial["trial"]
        t_start = ti * isi_samples + blank_samples
        t_end = t_start + evoked_samples
        if t_end > data_uv.shape[1]:
            skipped += 1
            continue
        seg = data_uv[:, t_start:t_end]
        raster = detect_spikes_per_channel(seg)
        binned = bin_raster(raster, n_timesteps)
        X_list.append(binned)
        y_list.append(digit_to_class[int(trial["digit"])])

    if skipped:
        print(f"  [WARN] Skipped {skipped} trials (recording shorter than expected)")
    if not X_list:
        raise RuntimeError("No usable trials extracted — check trial_log and recording length.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"  X={X.shape}, y={y.shape}, per-class counts={np.bincount(y).tolist()}")
    return X, y, digit_to_class


def make_synthetic_data(n_classes=3, n_per_class=80, n_channels=8,
                        n_timesteps=40, seed=0):
    """Linearly separable spike trains for sanity-checking the training loop."""
    rng = np.random.default_rng(seed)
    n_total = n_classes * n_per_class
    X = (rng.random((n_total, n_timesteps, n_channels)) < 0.05).astype(np.float32)
    y = np.zeros(n_total, dtype=np.int64)
    ch_per = max(1, n_channels // n_classes)
    for cls in range(n_classes):
        idxs = slice(cls * n_per_class, (cls + 1) * n_per_class)
        ch_lo, ch_hi = cls * ch_per, min((cls + 1) * ch_per, n_channels)
        boost = (rng.random((n_per_class, n_timesteps, ch_hi - ch_lo)) < 0.30).astype(np.float32)
        X[idxs, :, ch_lo:ch_hi] = np.minimum(X[idxs, :, ch_lo:ch_hi] + boost, 1.0)
        y[idxs] = cls
    return X, y


# ── Training ──────────────────────────────────────────────────────

def train(X, y, n_classes, n_epochs, lr, val_frac, batch_size, device, seed=42):
    n_total, n_timesteps, n_channels = X.shape
    n_val = max(1, int(n_total * val_frac))
    n_train = n_total - n_val

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    Xs, ys = X[perm], y[perm]

    train_ds = torch_data.TensorDataset(
        torch.from_numpy(Xs[:n_train]), torch.from_numpy(ys[:n_train])
    )
    val_ds = torch_data.TensorDataset(
        torch.from_numpy(Xs[n_train:]), torch.from_numpy(ys[n_train:])
    )
    train_loader = torch_data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch_data.DataLoader(val_ds, batch_size=batch_size)

    model = SNNDecoder(n_input=n_channels, n_classes=n_classes)
    model, history = train_snn_decoder(
        model, train_loader, val_loader,
        n_epochs=n_epochs, lr=lr, device=device,
    )
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",
                        help="data/classify_<ts>/ from run_2d_experiment.py")
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip --data-dir, use separable synthetic data (sanity-check)")
    parser.add_argument("--out-checkpoint", default=None,
                        help="Output .pt path. Default: checkpoints/snn_<ts>.pt")
    parser.add_argument("--n-timesteps", type=int, default=40)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.synthetic:
        print("Mode: synthetic separable data (sanity-check)")
        X, y = make_synthetic_data(n_timesteps=args.n_timesteps)
        n_classes = int(y.max()) + 1
        digit_to_class = None
        source = "synthetic"
    else:
        if not args.data_dir:
            parser.error("--data-dir is required unless --synthetic is set")
        print(f"Mode: real recording — {args.data_dir}")
        X, y, digit_to_class = load_trials_from_recording(args.data_dir, args.n_timesteps)
        n_classes = int(y.max()) + 1
        source = str(args.data_dir)

    print(f"\nTraining SNN: n_classes={n_classes}, X={X.shape}, "
          f"epochs={args.n_epochs}, lr={args.lr}\n")

    model, history = train(
        X, y, n_classes,
        n_epochs=args.n_epochs, lr=args.lr,
        val_frac=args.val_frac, batch_size=args.batch_size,
        device=args.device,
    )

    final_train_acc = history["train_acc"][-1] if history["train_acc"] else 0.0
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    chance = 1.0 / n_classes
    print(f"\nTraining complete.")
    print(f"  Final train acc: {final_train_acc:.1%}")
    print(f"  Final val acc:   {final_val_acc:.1%}")
    print(f"  Chance level:    {chance:.1%}")

    out_path = args.out_checkpoint
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"checkpoints/snn_{ts}.pt"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "n_input": model.n_input,
        "n_classes": n_classes,
        "n_timesteps": args.n_timesteps,
        "digit_to_class": digit_to_class,
        "history": history,
        "trained_at": datetime.now().isoformat(),
        "source": source,
    }
    torch.save(checkpoint, out_path)
    print(f"  Checkpoint: {out_path}")

    log_experiment(
        script="pretrain_snn_decoder.py",
        args=vars(args),
        verdict="PASS",
        summary={
            "mode": "synthetic" if args.synthetic else "recording",
            "n_classes": int(n_classes),
            "n_train_trials": int(X.shape[0] - max(1, int(X.shape[0] * args.val_frac))),
            "n_val_trials": max(1, int(X.shape[0] * args.val_frac)),
            "n_timesteps": int(args.n_timesteps),
            "n_channels": int(X.shape[2]),
            "final_train_acc": round(float(final_train_acc), 4),
            "final_val_acc": round(float(final_val_acc), 4),
            "chance_level": round(float(chance), 4),
            "n_epochs_actual": len(history["train_acc"]),
        },
        data=str(out_path),
        source=source,
    )


if __name__ == "__main__":
    main()
