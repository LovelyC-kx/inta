"""Tests for feature extraction module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction.extractors import (
    extract_timebin_features,
    extract_rms_features,
    extract_power_features,
    extract_spike_rate_features,
    extract_features_batch,
    reduce_with_pca,
)


def test_timebin_features_shape():
    """Time-bin features should have correct dimensions."""
    n_channels, n_samples = 8, 60000  # 2s at 30kHz
    sample_rate = 30000.0
    bin_ms = 200.0

    spike_trains = (np.random.rand(n_channels, n_samples) < 0.01).astype(np.float32)
    features = extract_timebin_features(spike_trains, bin_ms, sample_rate)

    expected_bins = int(np.ceil(n_samples / (bin_ms * sample_rate / 1000.0)))
    expected_len = n_channels * expected_bins
    assert len(features) == expected_len, \
        f"Expected {expected_len}, got {len(features)}"

    print(f"[PASS] Time-bin features: {len(features)} dims "
          f"({n_channels} ch x {expected_bins} bins)")


def test_rms_power_consistency():
    """RMS^2 should equal power for same data."""
    n_channels, n_samples = 4, 30000
    waveforms = np.random.randn(n_channels, n_samples).astype(np.float32)

    rms = extract_rms_features(waveforms, window_ms=50.0, sample_rate_hz=30000.0)
    power = extract_power_features(waveforms, window_ms=50.0, sample_rate_hz=30000.0)

    np.testing.assert_allclose(rms ** 2, power, rtol=1e-5)
    print("[PASS] RMS^2 == Power verified")


def test_spike_rate_units():
    """Spike rate should be in Hz (spikes per second)."""
    n_channels = 4
    sample_rate = 30000.0
    window_ms = 100.0
    samples_per_window = int(window_ms * sample_rate / 1000.0)

    # Put exactly 10 spikes per window per channel
    n_samples = samples_per_window * 2  # 2 windows
    spike_trains = np.zeros((n_channels, n_samples), dtype=np.float32)
    for ch in range(n_channels):
        spike_positions = np.linspace(0, samples_per_window - 1, 10, dtype=int)
        spike_trains[ch, spike_positions] = 1.0

    features = extract_spike_rate_features(spike_trains, window_ms, sample_rate)
    # First window: 10 spikes / 0.1 sec = 100 Hz
    expected_rate = 10.0 / (window_ms / 1000.0)
    assert abs(features[0] - expected_rate) < 1e-3, \
        f"Expected rate {expected_rate}, got {features[0]}"

    print(f"[PASS] Spike rate: {features[0]:.1f} Hz (expected {expected_rate:.1f} Hz)")


def test_batch_extraction():
    """Batch extraction should produce correct matrix shape."""
    n_samples_list = 10
    n_channels, n_timepoints = 8, 60000
    data_list = [
        (np.random.rand(n_channels, n_timepoints) < 0.01).astype(np.float32)
        for _ in range(n_samples_list)
    ]

    feature_matrix = extract_features_batch(
        data_list, extract_timebin_features,
        bin_size_ms=200.0, sample_rate_hz=30000.0
    )

    assert feature_matrix.shape[0] == n_samples_list
    assert feature_matrix.ndim == 2
    print(f"[PASS] Batch extraction: shape {feature_matrix.shape}")


def test_pca_reduction():
    """PCA should reduce dimensions while preserving variance."""
    np.random.seed(42)
    X_train = np.random.randn(100, 50).astype(np.float32)
    X_test = np.random.randn(20, 50).astype(np.float32)

    X_train_pca, X_test_pca, pca = reduce_with_pca(X_train, X_test, n_components=0.95)

    assert X_train_pca.shape[0] == 100
    assert X_test_pca.shape[0] == 20
    assert X_train_pca.shape[1] < 50, "PCA should reduce dimensions"
    assert X_train_pca.shape[1] == X_test_pca.shape[1]

    variance_explained = np.sum(pca.explained_variance_ratio_)
    assert variance_explained >= 0.95 - 0.01

    print(f"[PASS] PCA: {50} -> {X_train_pca.shape[1]} dims, "
          f"variance={variance_explained:.2%}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running feature extraction tests...")
    print("=" * 60)

    test_timebin_features_shape()
    test_rms_power_consistency()
    test_spike_rate_units()
    test_batch_extraction()
    test_pca_reduction()

    print("=" * 60)
    print("All feature extraction tests passed!")
    print("=" * 60)
