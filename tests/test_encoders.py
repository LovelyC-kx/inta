"""
Tests for encoder modules.

Validates:
1. CNN feature extractor: output shape, normalization, training loop
2. Delayed phase encoder: firing time monotonicity, constraint compliance,
   binary matrix correctness, inter-class distance
3. LIF encoder: spike generation, rate-intensity relationship
4. Rate encoder: rate proportionality, Poisson statistics
5. Full pipeline: end-to-end encoding from image to spike sequence
"""

import numpy as np
import torch
import torch.utils.data as data
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.cnn_feature_extractor import CNNFeatureExtractor, train_feature_extractor
from encoders.delayed_phase_encoder import DelayedPhaseEncoder, PulseEvent
from encoders.lif_encoder import LIFEncoder
from encoders.rate_encoder import RateEncoder
from encoders.pipeline import EncodingPipeline


def test_cnn_feature_extractor_shapes():
    """Verify CNN output dimensions match expectations."""
    model = CNNFeatureExtractor(n_channels=8, n_classes=10)
    x = torch.randn(4, 1, 28, 28)

    # Feature extraction
    features = model.extract_features(x)
    assert features.shape == (4, 8, 3, 3), f"Expected (4,8,3,3), got {features.shape}"

    # Classification forward
    logits = model(x)
    assert logits.shape == (4, 10), f"Expected (4,10), got {logits.shape}"

    # Feature normalization: values should be in [0, 1]
    assert features.min() >= 0.0, f"Features min {features.min()} < 0"
    assert features.max() <= 1.0 + 1e-6, f"Features max {features.max()} > 1"

    print("[PASS] CNN feature extractor shapes and normalization correct")


def test_cnn_training_loop():
    """Verify CNN training converges on synthetic data."""
    n_samples = 200
    images = torch.randn(n_samples, 1, 28, 28)
    labels = torch.randint(0, 3, (n_samples,))

    dataset = data.TensorDataset(images, labels)
    train_set, val_set = data.random_split(dataset, [160, 40])
    train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=32)

    model, history = train_feature_extractor(
        train_loader, val_loader,
        n_channels=8, n_classes=3, n_epochs=10, lr=1e-3
    )

    assert len(history["train_loss"]) > 0, "No training history recorded"
    # Loss should decrease (at least slightly on random data with enough capacity)
    assert history["train_loss"][-1] <= history["train_loss"][0] + 0.5, \
        "Training loss did not decrease"

    print("[PASS] CNN training loop runs without errors")


def test_delayed_phase_firing_times():
    """Verify pixel-to-time mapping: higher pixel -> earlier firing."""
    encoder = DelayedPhaseEncoder(t_max_ms=2000.0, delta=1.0)

    # Higher pixel values should produce earlier (smaller) firing times
    t_low = encoder._pixel_to_firing_time(0.1)
    t_high = encoder._pixel_to_firing_time(0.9)
    assert t_high < t_low, \
        f"Higher pixel should fire earlier: t(0.9)={t_high} >= t(0.1)={t_low}"

    # Boundary: pixel=0 should give t_max
    t_zero = encoder._pixel_to_firing_time(0.0)
    assert abs(t_zero - 2000.0) < 1e-6, \
        f"pixel=0 should give t_max, got {t_zero}"

    print("[PASS] Delayed phase firing time monotonicity correct")


def test_delayed_phase_constraints():
    """Verify pulse sequences satisfy biological constraints."""
    encoder = DelayedPhaseEncoder(
        t_max_ms=2000.0, delta=1.0, min_interval_ms=200.0
    )

    # Create synthetic feature maps (8 channels, 3x3)
    np.random.seed(42)
    feature_maps = np.random.rand(8, 3, 3).astype(np.float32)

    result = encoder.encode(feature_maps)

    # Check total duration constraint
    for pulse in result.pulses:
        assert 0 <= pulse.time_ms <= 2000.0, \
            f"Pulse at {pulse.time_ms}ms exceeds duration"

    # Check minimum interval per channel
    for ch in range(8):
        ch_times = sorted([p.time_ms for p in result.pulses if p.channel == ch])
        for i in range(1, len(ch_times)):
            interval = ch_times[i] - ch_times[i - 1]
            assert interval >= 200.0 - 1e-6, \
                f"Ch {ch}: interval {interval}ms < 200ms minimum"

    # Check binary matrix shape
    assert result.binary_matrix.shape == (8, 10), \
        f"Expected (8,10), got {result.binary_matrix.shape}"

    # Binary matrix should only contain 0 and 1
    assert set(np.unique(result.binary_matrix)).issubset({0, 1}), \
        "Binary matrix contains non-binary values"

    print(f"[PASS] Delayed phase constraints satisfied "
          f"({len(result.pulses)} pulses across 8 channels)")


def test_delayed_phase_inter_class_distance():
    """Different images should produce different encodings (larger distance)."""
    encoder = DelayedPhaseEncoder(t_max_ms=2000.0, delta=1.0)

    np.random.seed(0)
    # Two distinct feature maps
    feat_a = np.ones((8, 3, 3), dtype=np.float32) * 0.8  # Bright
    feat_b = np.ones((8, 3, 3), dtype=np.float32) * 0.2  # Dark

    # Same feature map twice (should be identical)
    feat_a2 = feat_a.copy()

    seq_a = encoder.encode(feat_a)
    seq_b = encoder.encode(feat_b)
    seq_a2 = encoder.encode(feat_a2)

    dist_same = DelayedPhaseEncoder.compute_distance(seq_a, seq_a2)
    dist_diff = DelayedPhaseEncoder.compute_distance(seq_a, seq_b)

    assert dist_same == 0.0, f"Same image distance should be 0, got {dist_same}"
    assert dist_diff > 0.0, f"Different images should have distance > 0"

    print(f"[PASS] Inter-class distance: same={dist_same:.2f}, diff={dist_diff:.2f}")


def test_lif_encoder_spike_generation():
    """LIF encoder should produce spikes for sufficient input."""
    encoder = LIFEncoder(tau_ms=10.0, threshold=350.0, dt_ms=0.1)

    # Test _simulate_single_channel directly with constant drive
    duration_ms = 100.0
    n_steps = int(duration_ms / encoder.dt_ms)
    strong_drive = np.ones(n_steps) * 500.0
    spikes = encoder._simulate_single_channel(strong_drive, duration_ms)
    assert len(spikes) > 0, "LIF should spike for strong constant drive"

    zero_drive = np.zeros(n_steps)
    spikes_zero = encoder._simulate_single_channel(zero_drive, duration_ms)
    assert len(spikes_zero) == 0, "LIF should not spike for zero drive"

    # Test encode_signal with a varying signal (z-score needs variance)
    duration_ms = 500.0
    n_steps = int(duration_ms / encoder.dt_ms)
    np.random.seed(42)
    varying_signal = np.random.randn(1, n_steps) * 200.0 + 100.0
    result = encoder.encode_signal(varying_signal, duration_ms)
    assert len(result.spike_times[0]) > 0, \
        "LIF should produce spikes for varying signal with sufficient amplitude"

    print(f"[PASS] LIF encoder: {len(spikes)} spikes for strong drive, "
          f"0 for zero drive, {len(result.spike_times[0])} for varying signal")


def test_lif_encoder_rate_intensity():
    """Higher drive current should produce higher spike rate (direct test)."""
    encoder = LIFEncoder(tau_ms=10.0, threshold=350.0, dt_ms=0.1)

    duration_ms = 500.0
    n_steps = int(duration_ms / encoder.dt_ms)

    low_drive = np.ones(n_steps) * 400.0
    high_drive = np.ones(n_steps) * 800.0

    spikes_low = encoder._simulate_single_channel(low_drive, duration_ms)
    spikes_high = encoder._simulate_single_channel(high_drive, duration_ms)

    n_low = len(spikes_low)
    n_high = len(spikes_high)

    assert n_high > n_low, \
        f"Higher drive should produce more spikes: {n_high} <= {n_low}"

    print(f"[PASS] LIF rate-intensity: low={n_low} spikes, high={n_high} spikes")


def test_lif_encode_features():
    """LIF encoding of image features should produce valid spike trains."""
    encoder = LIFEncoder(tau_ms=10.0, threshold=350.0, dt_ms=0.1, bin_size_ms=2.5)

    feature_maps = np.random.rand(8, 3, 3).astype(np.float32)
    result = encoder.encode_features(feature_maps, duration_ms=2000.0)

    assert result.n_channels == 8
    assert result.duration_ms == 2000.0
    assert result.spike_trains_binned.shape[0] == 8
    expected_bins = int(np.ceil(2000.0 / 2.5))
    assert result.spike_trains_binned.shape[1] == expected_bins

    total_spikes = sum(len(st) for st in result.spike_times)
    assert total_spikes > 0, "Should produce at least some spikes"

    print(f"[PASS] LIF feature encoding: {total_spikes} total spikes, "
          f"binned shape {result.spike_trains_binned.shape}")


def test_rate_encoder():
    """Rate encoder should produce rates proportional to pixel intensity."""
    encoder = RateEncoder(f_max_hz=50.0, dt_ms=1.0, seed=42)

    # All-high vs all-low feature maps
    high_features = np.ones((8, 3, 3), dtype=np.float32) * 0.9
    low_features = np.ones((8, 3, 3), dtype=np.float32) * 0.1

    result_high = encoder.encode_features(high_features, duration_ms=2000.0)
    result_low = encoder.encode_features(low_features, duration_ms=2000.0)

    assert result_high.mean_rate_hz > result_low.mean_rate_hz, \
        f"High features should have higher rate: " \
        f"{result_high.mean_rate_hz:.1f} <= {result_low.mean_rate_hz:.1f}"

    print(f"[PASS] Rate encoder: high={result_high.mean_rate_hz:.1f} Hz, "
          f"low={result_low.mean_rate_hz:.1f} Hz")


def test_full_pipeline():
    """End-to-end: image tensor -> StimulusProtocol."""
    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)

    for enc_type in ("delayed_phase", "lif", "rate"):
        pipeline = EncodingPipeline(cnn, encoder_type=enc_type)
        image = torch.randn(1, 1, 28, 28)
        proto = pipeline.encode_to_protocol(image)
        assert proto is not None, f"Pipeline {enc_type} returned None"
        proto.validate()

    print("[PASS] Full pipeline works for all 3 encoder types -> StimulusProtocol")


def test_encoding_reproducibility():
    """Same input should produce same output (deterministic encoders)."""
    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)
    cnn.eval()

    image = torch.randn(1, 1, 28, 28)

    # Delayed phase is deterministic
    pipeline = EncodingPipeline(cnn, encoder_type="delayed_phase")
    r1 = pipeline.encode_to_protocol(image)
    r2 = pipeline.encode_to_protocol(image)
    assert np.array_equal(r1.to_binary_matrix(10), r2.to_binary_matrix(10)), \
        "Delayed phase encoding should be deterministic"

    # LIF is deterministic (no randomness in dynamics)
    pipeline_lif = EncodingPipeline(cnn, encoder_type="lif")
    r3 = pipeline_lif.encode_to_protocol(image)
    r4 = pipeline_lif.encode_to_protocol(image)
    assert np.array_equal(r3.to_binary_matrix(10), r4.to_binary_matrix(10)), \
        "LIF encoding should be deterministic"

    print("[PASS] Deterministic encoders produce reproducible results")


if __name__ == "__main__":
    print("=" * 60)
    print("Running encoder tests...")
    print("=" * 60)

    test_cnn_feature_extractor_shapes()
    test_cnn_training_loop()
    test_delayed_phase_firing_times()
    test_delayed_phase_constraints()
    test_delayed_phase_inter_class_distance()
    test_lif_encoder_spike_generation()
    test_lif_encoder_rate_intensity()
    test_lif_encode_features()
    test_rate_encoder()
    test_full_pipeline()
    test_encoding_reproducibility()

    print("=" * 60)
    print("All encoder tests passed!")
    print("=" * 60)
