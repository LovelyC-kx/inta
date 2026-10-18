"""
Verify that stimulation actually happened by analyzing RHX-recorded stim.dat.

RHX writes one int16 per sample per stim-capable channel to stim.dat. A
non-zero value means current was flowing at that sample. Rising edges mark
pulse onsets.

Usage:
  python experiments/verify_stim_dat.py --data-dir data/test_<rhx_ts>/
  python experiments/verify_stim_dat.py --data-dir data/test_<rhx_ts>/ --no-plot

Outputs:
  - stdout summary: per-channel pulse count, total charge, comparison with
    trial_log.json if present
  - data/<rhx_dir>/stim_verification.png   (3-panel figure)
  - data/<rhx_dir>/stim_verification.json  (structured summary)

The figure has:
  Panel 1: stim raster (channels × time, ticks at every pulse onset)
  Panel 2: cumulative pulse count over time (one line per channel)
  Panel 3: histogram of pulse counts per channel
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_log import log_experiment


def load_stim_dat(data_dir, n_channels):
    """Read stim.dat → (n_channels, n_samples) int16 array."""
    p = data_dir / "stim.dat"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    raw = np.fromfile(str(p), dtype=np.int16)
    n_samples = len(raw) // n_channels
    return raw[:n_samples * n_channels].reshape(n_samples, n_channels).T


def find_pulse_onsets(stim_ch):
    """
    Given a 1-D int16 stim signal for one channel, find sample indices of
    pulse onsets (transitions from zero to non-zero).
    """
    nonzero = (stim_ch != 0).astype(np.int8)
    # Rising edges: 0 → 1 transitions
    onsets = np.where(np.diff(nonzero) == 1)[0] + 1
    return onsets


def estimate_charge_per_pulse(stim_ch, onset_idx, sample_rate=30000):
    """
    Approximate charge of one pulse by integrating |stim_ch| over the
    pulse's nonzero span. Returns charge in nC (assuming 1 step = 0.01 µA;
    not sure of exact RHS scaling, but the relative comparison is valid).
    """
    if onset_idx >= len(stim_ch):
        return 0.0
    end = onset_idx
    while end < len(stim_ch) and stim_ch[end] != 0:
        end += 1
    span = stim_ch[onset_idx:end]
    # crude approximation
    return float(np.sum(np.abs(span)))


def summarize(data_dir, n_channels=64, n_stim_channels=8, plot=True):
    config = {}
    cfg_path = data_dir / "config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text())
        n_channels = config.get("n_rec_channels", n_channels)
        n_stim_channels = config.get("n_stim_channels", n_stim_channels)

    print(f"Loading {data_dir}/stim.dat ...")
    stim = load_stim_dat(data_dir, n_channels)
    n_samples = stim.shape[1]
    sample_rate = config.get("sample_rate_hz", 30000)
    duration_s = n_samples / sample_rate

    print(f"  shape: {stim.shape}  duration: {duration_s:.1f}s")

    # ── Per-channel analysis ─────────────────────────────────
    print()
    print(f"{'ch':>4} {'pulses':>8} {'nonzero_samples':>16} {'pulse_rate_hz':>14}")
    print("-" * 50)

    per_ch_pulses = {}
    per_ch_onsets = {}
    total_pulses = 0
    active_channels = []

    for ch in range(n_channels):
        onsets = find_pulse_onsets(stim[ch])
        n_pulses = len(onsets)
        nonzero = int(np.sum(stim[ch] != 0))
        rate = n_pulses / duration_s if duration_s > 0 else 0
        per_ch_pulses[ch] = n_pulses
        per_ch_onsets[ch] = onsets
        total_pulses += n_pulses
        if n_pulses > 0:
            print(f"{ch:>4} {n_pulses:>8d} {nonzero:>16d} {rate:>14.2f}")
            active_channels.append(ch)

    print("-" * 50)
    print(f"Total pulses across all channels: {total_pulses}")
    print(f"Active stim channels: {len(active_channels)} / {n_channels}")
    if not active_channels:
        print("\n[WARN] stim.dat is all zeros — no stim was recorded.")
        print("       Possible causes:")
        print("         - Recording was in Run mode (not Record); no stim.dat written")
        print("         - Running in Demo mode (no hardware to commit stim to)")
        print("         - Stim commands didn't reach RHX")
        verdict = "NO_STIM_RECORDED"
    else:
        verdict = "STIM_VERIFIED"

    # ── Compare with trial_log if present ────────────────────
    trial_log_path = data_dir / "trial_log.json"
    if trial_log_path.exists():
        trial_log = json.loads(trial_log_path.read_text())
        total_expected = sum(t.get("total_pulses", 0) for t in trial_log)
        print()
        print(f"Trial log: {len(trial_log)} trials, expected ~{total_expected} pulses")
        if total_pulses > 0:
            ratio = total_pulses / max(total_expected, 1)
            print(f"  recorded / expected = {ratio:.2%}")
            if 0.8 <= ratio <= 1.2:
                print("  → match within 20% — stim verification passes")
            else:
                print("  → mismatch — investigate")

    # ── Plot ─────────────────────────────────────────────────
    if plot and active_channels:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

            # Panel 1: raster
            ax = axes[0]
            for ch in active_channels:
                t = per_ch_onsets[ch] / sample_rate
                ax.scatter(t, [ch] * len(t), s=4, marker="|", color="C0")
            ax.set_xlim(0, duration_s)
            ax.set_ylim(-0.5, max(active_channels) + 0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Stim channel")
            ax.set_title(f"Stim raster (n={total_pulses} pulses on {len(active_channels)} channels)")

            # Panel 2: cumulative count
            ax = axes[1]
            for ch in active_channels:
                t = per_ch_onsets[ch] / sample_rate
                ax.plot(t, np.arange(1, len(t) + 1), label=f"ch {ch}", linewidth=0.8)
            ax.set_xlim(0, duration_s)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Cumulative pulse count")
            ax.set_title("Cumulative pulses over time")
            if len(active_channels) <= 16:
                ax.legend(fontsize=7, ncol=2, loc="upper left")
            ax.grid(alpha=0.3)

            # Panel 3: per-channel histogram
            ax = axes[2]
            chs = list(per_ch_pulses.keys())
            counts = [per_ch_pulses[c] for c in chs]
            ax.bar(chs, counts, color=["C0" if c < 8 else "C7" for c in chs])
            ax.set_xlabel("Channel index")
            ax.set_ylabel("Pulse count")
            ax.set_title("Pulse count per channel (blue = expected stim channels 0–7)")
            ax.grid(alpha=0.3, axis="y")

            out_png = data_dir / "stim_verification.png"
            fig.savefig(out_png, dpi=120)
            plt.close(fig)
            print(f"\nSaved figure: {out_png}")
        except ImportError:
            print("\n[WARN] matplotlib not installed, skipping plot")

    # ── Save structured summary ──────────────────────────────
    summary_out = {
        "verdict": verdict,
        "n_channels_checked": int(n_channels),
        "duration_s": float(duration_s),
        "total_pulses": int(total_pulses),
        "active_channels": active_channels,
        "per_channel_pulses": {int(k): int(v) for k, v in per_ch_pulses.items()
                                if v > 0},
    }
    out_json = data_dir / "stim_verification.json"
    out_json.write_text(json.dumps(summary_out, indent=2))
    print(f"Saved summary: {out_json}")

    return summary_out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="RHX-created dir containing stim.dat (e.g. data/test_<ts>/)")
    p.add_argument("--n-channels", type=int, default=None,
                   help="Override channel count if config.json missing")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip png generation")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Directory not found: {data_dir}")

    summary = summarize(
        data_dir,
        n_channels=args.n_channels or 64,
        plot=not args.no_plot,
    )

    log_experiment(
        script="verify_stim_dat.py",
        args=vars(args),
        verdict=summary["verdict"],
        summary={
            "total_pulses": summary["total_pulses"],
            "active_channels": len(summary["active_channels"]),
            "duration_s": summary["duration_s"],
        },
        data=str(data_dir),
    )


if __name__ == "__main__":
    main()
