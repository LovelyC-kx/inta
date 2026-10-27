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
    """Read little-endian uint16 bitfields without copying the entire file."""
    if isinstance(n_channels, bool) or not isinstance(n_channels, int) or n_channels <= 0:
        raise ValueError("n_channels must be a positive integer")
    path = Path(data_dir) / "stim.dat"
    size = path.stat().st_size
    if size % (2 * n_channels):
        raise ValueError("stim.dat is truncated or its saved channel count is incorrect")
    if size == 0:
        return np.zeros((n_channels, 0), dtype=np.uint16)
    return np.memmap(path, mode="r", dtype="<u2").reshape(-1, n_channels).T


def decode_stim_words(stim):
    """RHS bit layout from official filepersignaltypemanager.cpp.

    Bits 0..7 magnitude; bit8 negative polarity; bit13 amp settle;
    bit14 charge recovery; bit15 compliance limit. Flags are not current.
    """
    words = np.asarray(stim).astype(np.uint16, copy=False)
    magnitude = words & 0xFF
    return {"magnitude": magnitude, "negative": (words & 0x100) != 0,
            "amp_settle": (words & 0x2000) != 0,
            "charge_recovery": (words & 0x4000) != 0,
            "compliance_limit": (words & 0x8000) != 0,
            "unknown_flags": (words & 0x1E00) != 0}


def extract_biphasic_pulses(stim_ch, sample_rate=30000):
    """Pair opposite-polarity phases, allowing RHS's <=5ms interphase gap.

    Return complete symmetric-representation pulses plus unpaired phase count.
    Pulse counts refer to two-phase events, not every nonzero run.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    decoded = decode_stim_words(stim_ch)
    magnitude, negative = decoded["magnitude"], decoded["negative"]
    active = magnitude != 0
    prior_active = np.r_[False, active[:-1]]
    next_active = np.r_[active[1:], False]
    prior_negative = np.r_[False, negative[:-1]]
    next_negative = np.r_[negative[1:], False]
    starts = np.flatnonzero(active & (~prior_active | (negative != prior_negative)))
    ends = np.flatnonzero(active & (~next_active | (negative != next_negative))) + 1
    phases = []
    for a, b in zip(starts, ends):
        values = magnitude[a:b]
        phases.append({"start": int(a), "end": int(b), "negative": bool(negative[a]),
                       "amplitude_code": int(values[0]),
                       "constant_amplitude": bool(np.all(values == values[0])),
                       "charge_code_samples": int(values.sum(dtype=np.uint64))})
    pulses = []
    unpaired = 0
    i = 0
    while i < len(phases):
        first = phases[i]
        if i + 1 < len(phases):
            second = phases[i + 1]
            gap = second["start"] - first["end"]
            if first["negative"] != second["negative"] and gap <= int(np.ceil(.005 * sample_rate)):
                pulses.append({"onset_sample": first["start"], "end_sample": second["end"],
                               "interphase_samples": gap, "phase1": first, "phase2": second,
                               "charge_balanced_codes": first["charge_code_samples"] == second["charge_code_samples"]})
                i += 2
                continue
        unpaired += 1
        i += 1
    return pulses, unpaired


def find_pulse_onsets(stim_ch, sample_rate=30000):
    pulses, _ = extract_biphasic_pulses(stim_ch, sample_rate)
    return np.array([p["onset_sample"] for p in pulses], dtype=np.int64)


def estimate_charge_per_pulse(stim_ch, onset_idx, sample_rate=30000, stim_step_ua=None):
    """Absolute two-phase charge in nC; never guess RHS current step size."""
    if stim_step_ua is None or not np.isfinite(stim_step_ua) or stim_step_ua <= 0:
        raise ValueError("An explicit positive stim_step_ua is required for charge in nC")
    pulses, _ = extract_biphasic_pulses(stim_ch, sample_rate)
    for pulse in pulses:
        if pulse["onset_sample"] == onset_idx:
            codes = pulse["phase1"]["charge_code_samples"] + pulse["phase2"]["charge_code_samples"]
            return codes * stim_step_ua * 1000.0 / sample_rate
    raise ValueError("onset_idx is not a complete biphasic pulse")


def summarize(data_dir, n_channels=64, n_stim_channels=8, plot=True, stim_step_ua=None):
    data_dir = Path(data_dir)
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
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Invalid sample rate")
    stim_step_ua = stim_step_ua if stim_step_ua is not None else config.get("stim_step_ua")
    if stim_step_ua is not None and (not np.isfinite(stim_step_ua) or stim_step_ua <= 0):
        raise ValueError("Invalid stim_step_ua")
    duration_s = n_samples / sample_rate
    per_ch_pulses, per_ch_onsets, pulse_records = {}, {}, {}
    flag_counts = {k: 0 for k in ("amp_settle", "charge_recovery", "compliance_limit", "unknown_flags")}
    unpaired = unbalanced = variable_amplitude = 0
    active_channels = []
    for ch in range(n_channels):
        decoded = decode_stim_words(stim[ch])
        for key in flag_counts:
            flag_counts[key] += int(decoded[key].sum())
        pulses, missing = extract_biphasic_pulses(stim[ch], sample_rate)
        unpaired += missing
        unbalanced += sum(not p["charge_balanced_codes"] for p in pulses)
        variable_amplitude += sum(not (p["phase1"]["constant_amplitude"] and
                                      p["phase2"]["constant_amplitude"]) for p in pulses)
        onsets = np.array([p["onset_sample"] for p in pulses], dtype=np.int64)
        per_ch_onsets[ch] = onsets
        per_ch_pulses[ch] = len(pulses)
        if pulses:
            active_channels.append(ch)
            pulse_records[ch] = pulses
    total_pulses = sum(per_ch_pulses.values())
    timestamp_source = "recording_sample_index"
    time_path = data_dir / "time.dat"
    if time_path.exists():
        if time_path.stat().st_size != 4 * n_samples:
            raise ValueError("time.dat and stim.dat sample counts do not match")
        if n_samples:
            times = np.memmap(time_path, mode="r", dtype="<u4")
            for pulses in pulse_records.values():
                for pulse in pulses:
                    pulse["raw_rhx_timestamp"] = int(times[pulse["onset_sample"]])
        timestamp_source = "rhx_time_dat_uint32"
    if stim_step_ua is not None:
        for pulses in pulse_records.values():
            for pulse in pulses:
                codes = pulse["phase1"]["charge_code_samples"] + pulse["phase2"]["charge_code_samples"]
                pulse["absolute_charge_nc_from_codes"] = codes * stim_step_ua * 1000 / sample_rate
    expected = None
    trial_log_path = data_dir / "trial_log.json"
    if trial_log_path.exists():
        trial_log = json.loads(trial_log_path.read_text())
        if all(isinstance(t.get("total_pulses"), int) and t["total_pulses"] >= 0 for t in trial_log):
            expected = sum(t["total_pulses"] for t in trial_log)
    if flag_counts["compliance_limit"]:
        verdict = "COMPLIANCE_LIMIT_DETECTED"
    elif unpaired or unbalanced or variable_amplitude or flag_counts["unknown_flags"]:
        verdict = "INVALID_STIM_DATA"
    elif total_pulses == 0:
        verdict = "NO_STIM_RECORDED"
    elif expected is not None and total_pulses != expected:
        verdict = "STIM_COUNT_MISMATCH"
    elif expected is not None:
        verdict = "STIM_RECORDED_COUNT_MATCH"
    else:
        verdict = "STIM_RECORDED_UNVERIFIED"
    print(f"Digital record: {total_pulses} complete biphasic pulses; {verdict}")
    print("Electrical delivery/timing is not verified by a nonzero digital record.")

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
        "physical_delivery_verified": False,
        "n_channels_checked": int(n_channels),
        "sample_rate_hz": sample_rate,
        "stim_step_ua": stim_step_ua,
        "duration_s": float(duration_s),
        "total_pulses": int(total_pulses),
        "expected_pulses": expected,
        "active_channels": active_channels,
        "per_channel_pulses": {int(k): int(v) for k, v in per_ch_pulses.items() if v > 0},
        "flag_sample_counts": flag_counts,
        "unpaired_phases": unpaired,
        "unbalanced_pulses": unbalanced,
        "variable_amplitude_pulses": variable_amplitude,
        "pulse_records": pulse_records,
        "timestamp_source": timestamp_source,
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
    p.add_argument("--stim-step-ua", type=float, help="Actual RHS current step; never inferred")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Directory not found: {data_dir}")

    summary = summarize(
        data_dir,
        n_channels=args.n_channels or 64,
        plot=not args.no_plot,
        stim_step_ua=args.stim_step_ua,
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
