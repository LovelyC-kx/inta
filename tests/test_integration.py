"""
Offline closed-loop integration test.

Purpose: verify the FULL pipeline works end-to-end without crashing,
with correct interfaces, dimensions, and timing at every stage.

This is NOT a performance evaluation. It does NOT claim any method is
better than another. It only checks:

    1. Interface: each stage accepts the previous stage's output type
    2. Dimensions: array shapes are consistent through the chain
    3. Timing: all StimulusProtocol invariants hold for every image
    4. Stability: the pipeline can process 1000 images without error

Pipeline under test:
    MNIST image
    -> CNN feature extraction
    -> Spike encoding (all 3 types)
    -> StimulusProtocol (validated)
    -> Fake reservoir (deterministic, NOT tuned for accuracy)
    -> Feature extraction
    -> Decoder input shape check

The fake reservoir here is a TRIVIAL linear mapping, deliberately NOT
designed to be separable. It exists only to produce output of the right
shape so we can test feature extraction and decoder input wiring.
"""

import numpy as np
import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.cnn_feature_extractor import CNNFeatureExtractor
from encoders.pipeline import EncodingPipeline
from encoders.stimulus_schema import (
    StimulusProtocol,
    MAX_STIM_CHANNELS,
    MAX_DURATION_MS,
    MIN_INTER_PULSE_MS,
)
from feature_extraction.extractors import (
    extract_timebin_features,
    extract_rms_features,
)


# ── Trivial fake reservoir ─────────────────────────────────────────
# NOT tuned. NOT designed to produce separable classes.
# Exists only to produce output of correct shape for downstream testing.

def trivial_fake_reservoir(protocol, n_rec_channels=32, bin_size_ms=10.0):
    """
    Minimal deterministic mapping: StimulusProtocol -> recording-shaped output.

    Returns (n_rec_channels, n_time_bins) array.
    This is a hash-like function of the protocol, not a reservoir model.
    """
    raster = protocol.to_spike_raster(bin_size_ms)  # (n_stim, n_bins)
    n_bins = raster.shape[1]

    # Fixed random projection (seeded, deterministic)
    rng = np.random.default_rng(seed=0)
    W = rng.standard_normal((n_rec_channels, protocol.n_channels)).astype(np.float32)

    # Project and add fixed noise
    output = np.tanh(W @ raster)
    noise = rng.standard_normal(output.shape).astype(np.float32) * 0.1
    output = (output + noise > 0.3).astype(np.float32)

    return output  # (n_rec_channels, n_bins)


# ── Integration tests ──────────────────────────────────────────────

def test_single_image_pipeline():
    """One image through all 3 encoders, full chain."""
    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)
    image = torch.rand(1, 1, 28, 28)

    for enc_type in ("delayed_phase", "lif", "rate"):
        pipe = EncodingPipeline(cnn, encoder_type=enc_type)

        # Stage 1: Image -> StimulusProtocol
        proto = pipe.encode_to_protocol(image)
        assert isinstance(proto, StimulusProtocol)
        proto.validate()  # Must not raise
        assert proto.n_channels <= MAX_STIM_CHANNELS
        assert proto.duration_ms <= MAX_DURATION_MS

        # Stage 2: Protocol -> Fake reservoir output
        rec_output = trivial_fake_reservoir(proto, n_rec_channels=32)
        assert rec_output.shape[0] == 32
        assert rec_output.ndim == 2

        # Stage 3: Reservoir output -> Features
        n_bins = rec_output.shape[1]
        timebin_feat = extract_timebin_features(
            rec_output, bin_size_ms=200.0, sample_rate_hz=100.0
        )
        assert timebin_feat.ndim == 1
        assert len(timebin_feat) > 0

        rms_feat = extract_rms_features(
            rec_output, window_ms=50.0, sample_rate_hz=100.0
        )
        assert rms_feat.ndim == 1
        assert len(rms_feat) > 0

    print("[PASS] Single image pipeline: all 3 encoders, full chain")


def test_protocol_invariants_batch():
    """Verify protocol invariants hold for a batch of random images."""
    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)
    n_images = 50

    for enc_type in ("delayed_phase", "lif", "rate"):
        pipe = EncodingPipeline(cnn, encoder_type=enc_type)
        for i in range(n_images):
            image = torch.rand(1, 1, 28, 28)
            proto = pipe.encode_to_protocol(image)
            try:
                proto.validate()
            except ValueError as e:
                raise AssertionError(
                    f"Protocol validation failed for {enc_type}, image {i}: {e}"
                )

    print(f"[PASS] Protocol invariants hold for {n_images} images x 3 encoders")


def test_dimension_consistency():
    """Verify all dimension handoffs are consistent."""
    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)
    pipe = EncodingPipeline(cnn, encoder_type="delayed_phase")

    # Feature map shape
    image = torch.rand(1, 1, 28, 28)
    fmaps = pipe._extract_features(image)
    assert fmaps.shape == (8, 3, 3), f"Expected (8,3,3), got {fmaps.shape}"

    # Protocol channels must match feature map channels
    proto = pipe.encode_to_protocol(image)
    assert proto.n_channels == 8

    # Raster shape must be (n_channels, n_bins)
    raster = proto.to_spike_raster(bin_size_ms=10.0)
    expected_bins = int(np.ceil(2000.0 / 10.0))
    assert raster.shape == (8, expected_bins), \
        f"Expected (8,{expected_bins}), got {raster.shape}"

    # Binary matrix shape
    bm = proto.to_binary_matrix(10)
    assert bm.shape == (8, 10)

    # Fake reservoir output shape
    rec = trivial_fake_reservoir(proto, n_rec_channels=32, bin_size_ms=10.0)
    assert rec.shape == (32, expected_bins)

    print("[PASS] Dimension consistency verified through entire chain")


def test_stability_1000_runs():
    """Run 1000 images through the pipeline without error."""
    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)
    # Use delayed_phase as the most constrained encoder
    pipe = EncodingPipeline(cnn, encoder_type="delayed_phase")

    t0 = time.time()
    for i in range(1000):
        image = torch.rand(1, 1, 28, 28)
        proto = pipe.encode_to_protocol(image)
        proto.validate()
        rec = trivial_fake_reservoir(proto)
        feat = extract_timebin_features(rec, bin_size_ms=200.0, sample_rate_hz=100.0)
        assert feat.ndim == 1

    elapsed = time.time() - t0
    print(f"[PASS] 1000 runs, no crashes, {elapsed:.1f}s total "
          f"({elapsed / 1000 * 1000:.1f}ms/image)")


def test_inter_pulse_interval_enforcement():
    """Explicitly verify no per-channel interval violation across many images."""
    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)

    for enc_type in ("delayed_phase", "lif", "rate"):
        pipe = EncodingPipeline(cnn, encoder_type=enc_type)
        violation_count = 0
        for _ in range(100):
            image = torch.rand(1, 1, 28, 28)
            proto = pipe.encode_to_protocol(image)
            for ch in range(proto.n_channels):
                times = sorted(p.time_ms for p in proto.pulses if p.channel == ch)
                for j in range(1, len(times)):
                    if times[j] - times[j - 1] < MIN_INTER_PULSE_MS - 1e-6:
                        violation_count += 1

        assert violation_count == 0, \
            f"{enc_type}: {violation_count} interval violations in 100 images"

    print("[PASS] Zero inter-pulse interval violations across 300 trials")


if __name__ == "__main__":
    print("=" * 60)
    print("Integration tests (interface/dimension/timing/stability)")
    print("These tests make NO claims about method quality.")
    print("=" * 60)

    test_single_image_pipeline()
    test_protocol_invariants_batch()
    test_dimension_consistency()
    test_inter_pulse_interval_enforcement()
    test_stability_1000_runs()

    print("=" * 60)
    print("All integration tests passed.")
    print("=" * 60)
