"""
Feature extraction functions for neural spike train and waveform data.

All functions accept multichannel data and return feature vectors suitable
for input to conventional decoders (LR, SVM, MLP).

References:
    Wang et al. (2025): time-bin firing counts + PCA
    Donati et al. (2026): RMS, power, spike rate in 50ms sub-windows
"""

import numpy as np
from sklearn.decomposition import PCA


def extract_timebin_features(spike_trains, bin_size_ms=200.0, sample_rate_hz=30000.0):
    """
    Extract time-bin firing count features (Wang et al.).

    Divides each channel's spike train into fixed time windows and counts
    the number of spikes in each window. Feature vectors from all channels
    are concatenated.

    Args:
        spike_trains: (n_channels, n_samples) binary spike raster
                      (1 = spike, 0 = no spike).
        bin_size_ms: Time window size in ms. Wang: 200 ms.
        sample_rate_hz: Sampling rate.

    Returns:
        feature_vector: 1D array of shape (n_channels * n_bins,).
    """
    n_channels, n_samples = spike_trains.shape
    samples_per_bin = int(bin_size_ms * sample_rate_hz / 1000.0)
    n_bins = int(np.ceil(n_samples / samples_per_bin))

    features = np.zeros((n_channels, n_bins), dtype=np.float32)

    for ch in range(n_channels):
        for b in range(n_bins):
            start = b * samples_per_bin
            end = min(start + samples_per_bin, n_samples)
            features[ch, b] = np.sum(spike_trains[ch, start:end])

    return features.flatten()


def extract_rms_features(waveforms, window_ms=50.0, sample_rate_hz=30000.0):
    """
    Extract RMS features (Donati et al.).

    Computes root mean square in non-overlapping sub-windows,
    then concatenates across channels.

    Args:
        waveforms: (n_channels, n_samples) continuous voltage data.
        window_ms: Sub-window size in ms. Donati: 50 ms.
        sample_rate_hz: Sampling rate.

    Returns:
        feature_vector: 1D array of shape (n_channels * n_windows,).
    """
    n_channels, n_samples = waveforms.shape
    samples_per_window = int(window_ms * sample_rate_hz / 1000.0)
    n_windows = int(np.ceil(n_samples / samples_per_window))

    features = np.zeros((n_channels, n_windows), dtype=np.float32)

    for ch in range(n_channels):
        for w in range(n_windows):
            start = w * samples_per_window
            end = min(start + samples_per_window, n_samples)
            segment = waveforms[ch, start:end]
            features[ch, w] = np.sqrt(np.mean(segment ** 2))

    return features.flatten()


def extract_power_features(waveforms, window_ms=50.0, sample_rate_hz=30000.0):
    """
    Extract power features (Donati et al.).

    Power = (1/N) * sum(|X|^2), computed in non-overlapping windows.

    Args:
        waveforms: (n_channels, n_samples) continuous voltage data.
        window_ms: Sub-window size.
        sample_rate_hz: Sampling rate.

    Returns:
        feature_vector: 1D array of shape (n_channels * n_windows,).
    """
    n_channels, n_samples = waveforms.shape
    samples_per_window = int(window_ms * sample_rate_hz / 1000.0)
    n_windows = int(np.ceil(n_samples / samples_per_window))

    features = np.zeros((n_channels, n_windows), dtype=np.float32)

    for ch in range(n_channels):
        for w in range(n_windows):
            start = w * samples_per_window
            end = min(start + samples_per_window, n_samples)
            segment = waveforms[ch, start:end]
            features[ch, w] = np.mean(segment ** 2)

    return features.flatten()


def extract_spike_rate_features(spike_trains, window_ms=50.0, sample_rate_hz=30000.0):
    """
    Extract spike rate features (Donati et al.).

    Spike rate = number of spikes / window duration.
    Donati et al. found this performed worst among feature types.

    Args:
        spike_trains: (n_channels, n_samples) binary spike raster.
        window_ms: Sub-window size.
        sample_rate_hz: Sampling rate.

    Returns:
        feature_vector: 1D array of shape (n_channels * n_windows,).
    """
    n_channels, n_samples = spike_trains.shape
    samples_per_window = int(window_ms * sample_rate_hz / 1000.0)
    n_windows = int(np.ceil(n_samples / samples_per_window))
    window_sec = window_ms / 1000.0

    features = np.zeros((n_channels, n_windows), dtype=np.float32)

    for ch in range(n_channels):
        for w in range(n_windows):
            start = w * samples_per_window
            end = min(start + samples_per_window, n_samples)
            n_spikes = np.sum(spike_trains[ch, start:end])
            features[ch, w] = n_spikes / window_sec

    return features.flatten()


def extract_features_batch(data_list, extract_fn, **kwargs):
    """
    Apply a feature extraction function to a list of data samples.

    Args:
        data_list: List of (n_channels, n_samples) arrays.
        extract_fn: One of the extraction functions above.
        **kwargs: Additional arguments for extract_fn.

    Returns:
        feature_matrix: (n_samples, n_features) array.
    """
    features = [extract_fn(d, **kwargs) for d in data_list]
    return np.stack(features)


def reduce_with_pca(X_train, X_test=None, n_components=0.95):
    """
    Apply PCA dimensionality reduction.

    Args:
        X_train: (n_train, n_features) training feature matrix.
        X_test: (n_test, n_features) test feature matrix, or None.
        n_components: Number of components or variance ratio to retain.

    Returns:
        X_train_pca: Reduced training features.
        X_test_pca: Reduced test features (or None).
        pca: Fitted PCA object (for inspection).
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)

    X_test_pca = None
    if X_test is not None:
        X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, pca
