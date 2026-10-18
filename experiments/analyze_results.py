"""
Offline analysis of recorded experiment data.

Run after run_2d_experiment.py completes.
Reads Intan-saved data files + trial logs, then:
  1. Spike detection on raw waveforms
  2. Feature extraction (time-bin, RMS, power)
  3. Decoder comparison (LR, SVM, SNN) with 5-fold CV
  4. Functional connectivity analysis (STTC)
  5. Results summary and figures

Usage:
  python analyze_results.py --data-dir data/classify_20260509_120000/
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction.extractors import (
    extract_timebin_features,
    extract_rms_features,
    extract_power_features,
    reduce_with_pca,
)
from decoders.conventional_decoders import (
    LogisticRegressionDecoder,
    SVMDecoder,
)
from utils.experiment_log import log_experiment


def load_trial_log(data_dir):
    with open(data_dir / "trial_log.json") as f:
        return json.load(f)


def load_config(data_dir):
    with open(data_dir / "config.json") as f:
        return json.load(f)


def detect_spikes(waveform, threshold_factor=5.5):
    """Simple negative-threshold spike detection."""
    noise_std = np.median(np.abs(waveform)) / 0.6745  # robust noise estimate
    threshold = -threshold_factor * noise_std
    crossings = np.where(np.diff((waveform < threshold).astype(int)) == 1)[0]
    return crossings


def extract_evoked_features(data_dir, trial_log, config, method="timebin"):
    """
    Extract feature vectors from recorded evoked responses.

    This function attempts to load Intan-saved data files.
    If files are not available (e.g., running offline without data),
    it returns None.
    """
    sr = config["sample_rate_hz"]
    evoked_window_samples = int(config["evoked_window_ms"] * sr / 1000)
    blank_samples = int(config["artifact_blanking_ms"] * sr / 1000)
    n_rec = config["n_rec_channels"]

    # Try to load amplifier data
    amp_file = data_dir / "amplifier.dat"
    if not amp_file.exists():
        print(f"  [WARN] {amp_file} not found.")
        print("  Looking for .rhs files...")
        rhs_files = sorted(data_dir.glob("*.rhs"))
        if not rhs_files:
            print("  [WARN] No data files found. Cannot extract features.")
            return None, None
        print(f"  Found {len(rhs_files)} .rhs files.")
        print("  Use official Intan read_Intan_RHS2000_file.m to extract data.")
        return None, None

    print(f"  Loading {amp_file} ({amp_file.stat().st_size / 1e6:.1f} MB)")
    raw = np.fromfile(str(amp_file), dtype=np.int16)
    n_samples_total = len(raw) // n_rec
    raw = raw[:n_samples_total * n_rec].reshape(n_samples_total, n_rec).T

    # Convert to microvolts
    data_uv = raw.astype(np.float32) * 0.195

    # Spike detection per channel
    print("  Detecting spikes...")
    spike_raster = np.zeros_like(data_uv, dtype=np.float32)
    for ch in range(n_rec):
        spike_times = detect_spikes(data_uv[ch])
        spike_raster[ch, spike_times] = 1.0

    active_channels = np.sum(spike_raster.sum(axis=1) > 0)
    print(f"  Active channels: {active_channels}/{n_rec}")

    # Extract features per trial
    isi_samples = int(config["inter_trial_interval_s"] * sr)
    features_list = []
    labels = []

    for trial in trial_log:
        trial_idx = trial["trial"]
        trial_start = trial_idx * isi_samples + blank_samples
        trial_end = trial_start + evoked_window_samples

        if trial_end > spike_raster.shape[1]:
            continue

        segment = spike_raster[:, trial_start:trial_end]

        if method == "timebin":
            feat = extract_timebin_features(segment, bin_size_ms=200.0,
                                            sample_rate_hz=sr)
        elif method == "rms":
            seg_uv = data_uv[:, trial_start:trial_end]
            feat = extract_rms_features(seg_uv, window_ms=50.0,
                                        sample_rate_hz=sr)
        elif method == "power":
            seg_uv = data_uv[:, trial_start:trial_end]
            feat = extract_power_features(seg_uv, window_ms=50.0,
                                          sample_rate_hz=sr)
        else:
            raise ValueError(f"Unknown method: {method}")

        features_list.append(feat)
        labels.append(trial["digit"])

    if not features_list:
        return None, None

    X = np.stack(features_list)
    y = np.array(labels)
    return X, y


def run_decoder_comparison(X, y, n_folds=5):
    """Run all decoders with k-fold CV."""
    results = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for name, DecoderClass, kwargs in [
        ("LR (timebin)", LogisticRegressionDecoder, {"n_components": 0.95}),
        ("SVM (linear)", SVMDecoder, {"kernel": "linear"}),
    ]:
        accs = []
        for train_idx, test_idx in skf.split(X, y):
            decoder = DecoderClass(**kwargs)
            decoder.fit(X[train_idx], y[train_idx])
            acc = decoder.score(X[test_idx], y[test_idx])
            accs.append(acc)
        results[name] = np.array(accs)

    return results


def compute_sttc(spike_raster, delta_ms=10.0, sample_rate_hz=30000.0):
    """
    Compute pairwise Spike Time Tiling Coefficient.
    Returns (n_channels, n_channels) STTC matrix.
    """
    n_ch, n_samples = spike_raster.shape
    delta_samples = int(delta_ms * sample_rate_hz / 1000.0)
    sttc_matrix = np.zeros((n_ch, n_ch), dtype=np.float32)

    for i in range(n_ch):
        spike_times_i = np.where(spike_raster[i] > 0)[0]
        if len(spike_times_i) == 0:
            continue

        # T_i: fraction of total time within ±delta of any spike from i
        tiled_i = np.zeros(n_samples, dtype=bool)
        for t in spike_times_i:
            start = max(0, t - delta_samples)
            end = min(n_samples, t + delta_samples + 1)
            tiled_i[start:end] = True
        T_i = tiled_i.sum() / n_samples

        for j in range(i + 1, n_ch):
            spike_times_j = np.where(spike_raster[j] > 0)[0]
            if len(spike_times_j) == 0:
                continue

            tiled_j = np.zeros(n_samples, dtype=bool)
            for t in spike_times_j:
                start = max(0, t - delta_samples)
                end = min(n_samples, t + delta_samples + 1)
                tiled_j[start:end] = True
            T_j = tiled_j.sum() / n_samples

            # P_i: fraction of spikes from i within ±delta of any spike from j
            P_i = np.mean(tiled_j[spike_times_i]) if len(spike_times_i) > 0 else 0
            P_j = np.mean(tiled_i[spike_times_j]) if len(spike_times_j) > 0 else 0

            # STTC
            term1 = (P_i - T_j) / (1 - P_i * T_j) if (1 - P_i * T_j) != 0 else 0
            term2 = (P_j - T_i) / (1 - P_j * T_i) if (1 - P_j * T_i) != 0 else 0
            sttc_matrix[i, j] = 0.5 * (term1 + term2)
            sttc_matrix[j, i] = sttc_matrix[i, j]

    return sttc_matrix


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--data-dir", required=True, help="Path to experiment data")
    parser.add_argument("--feature", default="timebin",
                        choices=["timebin", "rms", "power"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        return

    config = load_config(data_dir)
    print(f"Config: {json.dumps(config, indent=2)}")

    trial_log_path = data_dir / "trial_log.json"
    if not trial_log_path.exists():
        print("[ERROR] No trial_log.json found. Is this a classify experiment?")
        return

    trial_log = load_trial_log(data_dir)
    print(f"Loaded {len(trial_log)} trials")

    # Extract features
    print(f"\nExtracting {args.feature} features...")
    X, y = extract_evoked_features(data_dir, trial_log, config, method=args.feature)

    if X is None:
        print("\n[INFO] No raw data files found for offline analysis.")
        print("To analyze, either:")
        print("  1. Use 'One File Per Signal Type' format in Intan")
        print("  2. Convert .rhs files with official Intan MATLAB/Python code")
        return

    print(f"Feature matrix: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}, counts: {np.bincount(y)}")

    # PCA
    X_pca, _, pca = reduce_with_pca(X, n_components=0.95)
    print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} dims")

    # Decoder comparison
    print("\nRunning 5-fold cross-validation...")
    results = run_decoder_comparison(X_pca, y)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    chance = 1.0 / len(np.unique(y))
    print(f"Chance level: {chance:.2%}")
    for name, accs in sorted(results.items(), key=lambda x: -np.mean(x[1])):
        fold_str = ", ".join(f"{a:.1%}" for a in accs)
        print(f"  {name:20s}: {np.mean(accs):.2%} ± {np.std(accs):.2%}  [{fold_str}]")

    # Save results
    result_summary = {
        "feature_method": args.feature,
        "n_trials": len(trial_log),
        "n_features_raw": int(X.shape[1]),
        "n_features_pca": int(X_pca.shape[1]),
        "chance_level": chance,
    }
    for name, accs in results.items():
        result_summary[name] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "folds": accs.tolist(),
        }

    with open(data_dir / "analysis_results.json", "w") as f:
        json.dump(result_summary, f, indent=2)

    print(f"\nResults saved to {data_dir / 'analysis_results.json'}")

    log_summary = {
        "feature_method": args.feature,
        "n_trials": len(trial_log),
        "n_features_raw": int(X.shape[1]),
        "n_features_pca": int(X_pca.shape[1]),
        "chance_level": round(float(chance), 4),
        "n_classes": int(len(np.unique(y))),
    }
    for name, accs in results.items():
        log_summary[f"acc_mean_{name}"] = round(float(np.mean(accs)), 4)
        log_summary[f"acc_std_{name}"] = round(float(np.std(accs)), 4)

    log_experiment(
        script="analyze_results.py",
        args=vars(args),
        verdict="COMPLETE",
        summary=log_summary,
        data=str(data_dir / "analysis_results.json"),
    )


if __name__ == "__main__":
    main()
