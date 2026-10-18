"""
Tests for decoder modules.

Validates:
1. SNN decoder: shape correctness, training convergence on separable data
2. Logistic Regression: fitting and prediction, PCA info
3. SVM: fitting and prediction
4. MLP: fitting and prediction
5. All decoders: above-chance accuracy on linearly separable synthetic data
"""

import numpy as np
import torch
import torch.utils.data as data
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoders.snn_decoder import SNNDecoder, train_snn_decoder
from decoders.conventional_decoders import (
    LogisticRegressionDecoder,
    SVMDecoder,
    MLPDecoder,
)


def make_separable_spike_data(n_classes=3, n_samples_per_class=50,
                               n_channels=16, n_timesteps=40, seed=42):
    """
    Generate synthetic spike train data that is linearly separable.

    Each class has a distinct firing pattern: class k has higher activity
    in channels [k*n_ch_per_class : (k+1)*n_ch_per_class].
    """
    rng = np.random.default_rng(seed)
    n_total = n_classes * n_samples_per_class
    ch_per_class = n_channels // n_classes

    spike_trains = np.zeros((n_total, n_timesteps, n_channels), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int64)

    for cls in range(n_classes):
        for i in range(n_samples_per_class):
            idx = cls * n_samples_per_class + i
            labels[idx] = cls
            # Background low activity
            spike_trains[idx] = (rng.random((n_timesteps, n_channels)) < 0.05).astype(np.float32)
            # Class-specific high activity in designated channels
            ch_start = cls * ch_per_class
            ch_end = ch_start + ch_per_class
            spike_trains[idx, :, ch_start:ch_end] = (
                rng.random((n_timesteps, ch_per_class)) < 0.5
            ).astype(np.float32)

    return spike_trains, labels


def make_separable_feature_data(n_classes=3, n_samples_per_class=50,
                                 n_features=32, seed=42):
    """Generate linearly separable feature vectors for conventional decoders."""
    rng = np.random.default_rng(seed)
    n_total = n_classes * n_samples_per_class
    X = rng.standard_normal((n_total, n_features)).astype(np.float32)
    y = np.zeros(n_total, dtype=np.int64)

    for cls in range(n_classes):
        start = cls * n_samples_per_class
        end = start + n_samples_per_class
        y[start:end] = cls
        # Shift mean of designated feature dimensions
        feat_start = cls * (n_features // n_classes)
        feat_end = feat_start + (n_features // n_classes)
        X[start:end, feat_start:feat_end] += 3.0

    return X, y


def test_snn_decoder_shapes():
    """Verify SNN decoder output shapes."""
    model = SNNDecoder(n_input=16, n_classes=3)
    batch = torch.randn(4, 40, 16)  # (batch, timesteps, channels)

    spike_counts, mem_trace = model(batch)
    assert spike_counts.shape == (4, 3), f"Expected (4,3), got {spike_counts.shape}"
    assert mem_trace.shape == (4, 40, 3), f"Expected (4,40,3), got {mem_trace.shape}"

    predictions = model.predict(batch)
    assert predictions.shape == (4,), f"Expected (4,), got {predictions.shape}"

    print("[PASS] SNN decoder shapes correct")


def test_snn_decoder_training():
    """SNN decoder should converge on separable data."""
    torch.manual_seed(42)
    np.random.seed(42)

    spike_data, labels = make_separable_spike_data(
        n_classes=3, n_samples_per_class=60, n_channels=16, n_timesteps=40
    )

    # Split train/val
    n_train = 150
    train_spikes = torch.from_numpy(spike_data[:n_train])
    train_labels = torch.from_numpy(labels[:n_train])
    val_spikes = torch.from_numpy(spike_data[n_train:])
    val_labels = torch.from_numpy(labels[n_train:])

    train_ds = data.TensorDataset(train_spikes, train_labels)
    val_ds = data.TensorDataset(val_spikes, val_labels)
    train_loader = data.DataLoader(train_ds, batch_size=30, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=30)

    model = SNNDecoder(n_input=16, n_classes=3)
    model, history = train_snn_decoder(
        model, train_loader, val_loader, n_epochs=200, lr=5e-3
    )

    # Take best val accuracy across training
    best_val_acc = max(history["val_acc"])
    chance = 1.0 / 3.0

    assert best_val_acc > chance + 0.1, \
        f"SNN best val accuracy {best_val_acc:.2%} not above chance+10% ({chance+0.1:.2%})"

    print(f"[PASS] SNN decoder training: best_val_acc={best_val_acc:.2%} "
          f"(chance={chance:.2%})")


def test_logistic_regression_decoder():
    """LR decoder should achieve high accuracy on separable data."""
    X, y = make_separable_feature_data(n_classes=3, n_samples_per_class=50)
    n_train = 120

    decoder = LogisticRegressionDecoder(n_components=0.95)
    decoder.fit(X[:n_train], y[:n_train])

    acc = decoder.score(X[n_train:], y[n_train:])
    pca_info = decoder.get_pca_info()

    assert acc > 0.8, f"LR accuracy {acc:.2%} < 80% on separable data"
    assert pca_info["total_explained"] >= 0.90, \
        f"PCA explained variance {pca_info['total_explained']:.2%} < 90%"

    print(f"[PASS] LR decoder: acc={acc:.2%}, "
          f"PCA components={pca_info['n_components']}, "
          f"variance explained={pca_info['total_explained']:.2%}")


def test_svm_decoder():
    """SVM decoder should achieve high accuracy on separable data."""
    X, y = make_separable_feature_data(n_classes=3, n_samples_per_class=50)
    n_train = 120

    decoder = SVMDecoder(kernel="linear")
    decoder.fit(X[:n_train], y[:n_train])
    acc = decoder.score(X[n_train:], y[n_train:])

    assert acc > 0.8, f"SVM accuracy {acc:.2%} < 80%"
    print(f"[PASS] SVM decoder: acc={acc:.2%}")


def test_mlp_decoder():
    """MLP decoder should achieve high accuracy on separable data."""
    X, y = make_separable_feature_data(n_classes=3, n_samples_per_class=50)
    n_train = 120

    decoder = MLPDecoder(hidden_ratio=2.1, max_iter=200)
    decoder.fit(X[:n_train], y[:n_train])
    acc = decoder.score(X[n_train:], y[n_train:])

    assert acc > 0.7, f"MLP accuracy {acc:.2%} < 70%"
    print(f"[PASS] MLP decoder: acc={acc:.2%}")


def test_decoder_comparison():
    """Compare all decoders on the same data to verify relative performance."""
    X_feat, y = make_separable_feature_data(
        n_classes=3, n_samples_per_class=80, n_features=32
    )
    spike_data, y_spike = make_separable_spike_data(
        n_classes=3, n_samples_per_class=80, n_channels=16, n_timesteps=40
    )

    n_train = 180
    results = {}

    # Conventional decoders on feature data
    for name, decoder in [
        ("LR", LogisticRegressionDecoder()),
        ("SVM", SVMDecoder()),
        ("MLP", MLPDecoder(max_iter=200)),
    ]:
        decoder.fit(X_feat[:n_train], y[:n_train])
        results[name] = decoder.score(X_feat[n_train:], y[n_train:])

    # SNN on spike data
    train_ds = data.TensorDataset(
        torch.from_numpy(spike_data[:n_train]),
        torch.from_numpy(y_spike[:n_train]),
    )
    val_ds = data.TensorDataset(
        torch.from_numpy(spike_data[n_train:]),
        torch.from_numpy(y_spike[n_train:]),
    )
    train_loader = data.DataLoader(train_ds, batch_size=30, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=30)

    snn_model = SNNDecoder(n_input=16, n_classes=3)
    snn_model, snn_hist = train_snn_decoder(
        snn_model, train_loader, val_loader, n_epochs=100
    )
    results["SNN"] = snn_hist["val_acc"][-1]

    print("\n[Decoder Comparison on Separable Data]")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:5s}: {acc:.2%}")

    # All should be above chance
    chance = 1.0 / 3.0
    for name, acc in results.items():
        assert acc > chance, f"{name} accuracy {acc:.2%} not above chance"

    print("[PASS] All decoders above chance level")


if __name__ == "__main__":
    print("=" * 60)
    print("Running decoder tests...")
    print("=" * 60)

    test_snn_decoder_shapes()
    test_snn_decoder_training()
    test_logistic_regression_decoder()
    test_svm_decoder()
    test_mlp_decoder()
    test_decoder_comparison()

    print("=" * 60)
    print("All decoder tests passed!")
    print("=" * 60)
