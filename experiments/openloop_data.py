"""Shared real-event feature contract, portable frozen decoders and training.

Training and online inference call the SAME window builder. No waveform spike
detection, PCA fit on test data, nominal trial-index timing, or pickle loading.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from intan_interface.spike_windows import build_spike_window


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class FeatureSpec:
    duration_ms: float = 2500
    offset_ms: float = 0
    pre_blank_ms: float = 1
    post_blank_ms: float = 10
    n_bins: int = 50
    min_valid_fraction: float = .2

    def __post_init__(self):
        for name in ("duration_ms", "offset_ms", "pre_blank_ms", "post_blank_ms", "min_valid_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Invalid {name}")
        if (not 0 < self.duration_ms <= 10000 or self.offset_ms < 0 or
                self.pre_blank_ms < 0 or self.post_blank_ms < 0 or
                not 0 < self.min_valid_fraction <= 1 or type(self.n_bins) is not int or
                not 1 <= self.n_bins <= 1000):
            raise ValueError("Invalid observation window/binning/artifact policy")

    def samples(self, rate, field):
        return int(round(getattr(self, field) * rate / 1000))


def make_contract(channels, rate, spec, encoder, neural_settings, *, simulated):
    return {
        "schema": "inta-rhx-features-v1", "event_source": "RHX-5002",
        "simulated": bool(simulated), "channel_names": list(channels),
        "sample_rate_hz": rate, "features": asdict(spec), "encoder": encoder,
        "neural_settings": neural_settings,
        "spike_id_policy": "0x01..0x0f or 0x40; exclude artifact/unknown bits",
        "flatten_order": "time_bin,channel", "artifact_scope": "all_recording_channels",
    }


def extract_observation(observation, contract, epoch):
    spec = FeatureSpec(**contract["features"])
    rate = contract["sample_rate_hz"]
    start = observation["anchor"] + spec.samples(rate, "offset_ms")
    end = start + spec.samples(rate, "duration_ms")
    if end - start < spec.n_bins:
        raise ValueError("More bins than observed hardware samples")
    masks = [(p.start_sample - spec.samples(rate, "pre_blank_ms"),
              p.end_sample + spec.samples(rate, "post_blank_ms")) for p in observation["markers"]]
    window = build_spike_window(observation["events"], channel_names=contract["channel_names"],
                                start_sample=start, end_sample=end, acquisition_epoch=epoch,
                                watermark=observation["watermark"], artifact_intervals={"*": masks})
    status = window.status
    if min(window.valid_samples_per_channel) < (end - start) * spec.min_valid_fraction:
        status = "INVALID"
    counts = window.bin_counts(spec.n_bins) if status != "INVALID" else None
    return window, counts, status


class FrozenDecoder:
    def __init__(self, payload):
        if payload.get("schema") != "inta-frozen-decoder-v1":
            raise ValueError("Unsupported decoder artifact")
        self.payload = payload
        self.contract = payload["contract"]
        self.classes = payload["classes"]
        self.mean = np.asarray(payload["mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scale"], dtype=np.float64)
        self.weights = [np.asarray(w, dtype=np.float64) for w in payload["weights"]]
        self.biases = [np.asarray(b, dtype=np.float64) for b in payload["biases"]]
        expected = len(self.contract["channel_names"]) * self.contract["features"]["n_bins"]
        if (len(self.classes) < 2 or len(set(self.classes)) != len(self.classes) or
                self.mean.shape != (expected,) or self.scale.shape != (expected,) or
                not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or
                np.any(self.scale <= 0) or len(self.weights) != len(self.biases) or not self.weights):
            raise ValueError("Malformed decoder dimensions/classes/scaling")
        width = expected
        for weight, bias in zip(self.weights, self.biases):
            if (weight.ndim != 2 or weight.shape[0] != width or bias.shape != (weight.shape[1],) or
                    not np.isfinite(weight).all() or not np.isfinite(bias).all()):
                raise ValueError("Malformed decoder layer")
            width = weight.shape[1]
        if width != (1 if len(self.classes) == 2 else len(self.classes)):
            raise ValueError("Decoder output/class mismatch")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def require_contract(self, contract):
        if canonical(contract) != canonical(self.contract):
            raise ValueError("Decoder contract differs from acquisition/encoder/window settings")

    def predict(self, counts):
        x = np.asarray(counts, dtype=np.float64).reshape(-1)
        if x.shape != self.mean.shape or not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError("Invalid decoder features")
        x = (x - self.mean) / self.scale
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            if i + 1 < len(self.weights):
                x = np.maximum(x, 0)
        if len(self.classes) == 2:
            p = 1 / (1 + np.exp(-np.clip(x[0], -700, 700)))
            probabilities = np.array([1 - p, p])
        else:
            probabilities = np.exp(x - x.max())
            probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        return {"label": self.classes[index], "confidence": float(probabilities[index]),
                "probabilities": dict(zip(self.classes, probabilities.tolist())),
                "confidence_calibrated": False}


def read_session(path):
    path = Path(path)
    meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    if meta.get("status") != "COMPLETE" or meta.get("schema") != "inta-openloop-v1":
        raise ValueError(f"Session is incomplete/invalid: {path}")
    if set(meta.get("file_sha256", {})) != {"events.jsonl", "trials.jsonl", "markers.raw", "spikes.raw"}:
        raise ValueError("Missing required session journals/hashes")
    for name, sha in meta["file_sha256"].items():
        if file_hash(path / name) != sha:
            raise ValueError(f"Session journal hash mismatch: {name}")
    rows = [json.loads(line) for line in (path / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    return meta, rows


def train_from_sessions(paths, output, test_groups, *, kind="lr", seed=42):
    """An explicit held-out group split; scaler/model only see training rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if kind not in ("lr", "mlp") or not test_groups:
        raise ValueError("Choose lr/mlp and explicit held-out test groups")
    contract = None
    features, labels, groups, images, provenance = [], [], [], [], []
    excluded = 0
    timing_validated = True
    for path in paths:
        meta, rows = read_session(path)
        if contract is not None and canonical(meta["contract"]) != canonical(contract):
            raise ValueError("Cannot train across incompatible feature contracts")
        contract = meta["contract"]
        timing_validated = timing_validated and meta.get("acquisition", {}).get("timing", {}).get("validated") is True
        provenance.append({"session_id": meta["session_id"], "files": meta["file_sha256"]})
        for row in rows:
            if row["status"] != "VALID":
                excluded += 1
                continue
            if row.get("label") is None:
                raise ValueError("Training sessions require ground-truth labels")
            features.append(np.asarray(row["counts"], dtype=np.float64).ravel())
            labels.append(str(row["label"]))
            groups.append(row["group"])
            images.append(row["image_sha256"])
    if not features:
        raise ValueError("No valid labeled trials")
    X, y = np.stack(features), np.array(labels)
    test = np.array([group in set(test_groups) for group in groups])
    if not test.any() or test.all() or set(test_groups) - set(groups):
        raise ValueError("Held-out groups must exist and leave nonempty train/test sets")
    if set(np.array(images)[test]) & set(np.array(images)[~test]):
        raise ValueError("Identical input images cross the train/test boundary")
    if len(set(y[~test])) < 2 or set(y[test]) - set(y[~test]):
        raise ValueError("Training needs >=2 classes and must cover all test classes")
    scaler = StandardScaler().fit(X[~test])
    model = (LogisticRegression(max_iter=1000, random_state=seed) if kind == "lr" else
             MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=seed))
    model.fit(scaler.transform(X[~test]), y[~test])
    if kind == "lr":
        weights, biases = [model.coef_.T.tolist()], [model.intercept_.tolist()]
    else:
        weights, biases = [w.tolist() for w in model.coefs_], [b.tolist() for b in model.intercepts_]
    report = {
        "train_trials": int((~test).sum()), "test_trials": int(test.sum()),
        "excluded_silent_or_invalid_trials": excluded, "held_out_groups": sorted(set(test_groups)),
        "train_accuracy": float(model.score(scaler.transform(X[~test]), y[~test])),
        "test_accuracy": float(model.score(scaler.transform(X[test]), y[test])),
        "evaluation_scope": "synthetic interface only" if contract["simulated"] else "held-out supplied groups",
        "confidence_calibrated": False,
    }
    payload = {"schema": "inta-frozen-decoder-v1", "kind": kind, "contract": contract,
               "classes": model.classes_.tolist(), "mean": scaler.mean_.tolist(),
               "scale": scaler.scale_.tolist(), "weights": weights, "biases": biases,
               "training_sessions": provenance, "timing_validated": timing_validated, "report": report}
    decoder = FrozenDecoder(payload)
    # Ensure the portable numpy implementation preserves fitted probabilities.
    for i in np.flatnonzero(test):
        actual = list(decoder.predict(X[i])["probabilities"].values())
        np.testing.assert_allclose(actual, model.predict_proba(scaler.transform(X[i:i+1]))[0], atol=1e-8)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing decoder")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return report
