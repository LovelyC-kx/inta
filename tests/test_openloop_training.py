import json

import numpy as np
import pytest

from experiments.openloop_data import FeatureSpec, FrozenDecoder, file_hash, make_contract, train_from_sessions


def training_fixture(path, *, duplicate=False, unknown_class=False):
    path.mkdir()
    contract = make_contract(["A-000"], 30000, FeatureSpec(duration_ms=100, n_bins=2), {}, {}, simulated=True)
    rows = []
    for i, (label, group, counts) in enumerate([
            ("a", "train", [[1], [2]]), ("b", "train", [[4], [5]]),
            ("a", "train", [[2], [3]]), ("b", "train", [[5], [6]]),
            ("a", "test", [[999], [1000]]), ("c" if unknown_class else "b", "test", [[1001], [1002]])]):
        rows.append({"status": "VALID", "counts": counts, "label": label, "group": group,
                     "image_sha256": "0" if duplicate and i == 4 else str(i)})
    (path / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for name in ("events.jsonl", "markers.raw", "spikes.raw"):
        (path / name).write_bytes(b"")
    meta = {"schema": "inta-openloop-v1", "status": "COMPLETE", "session_id": "fixture",
            "contract": contract, "acquisition": {}, "file_sha256": {
                n: file_hash(path / n) for n in ("trials.jsonl", "events.jsonl", "markers.raw", "spikes.raw")}}
    (path / "session.json").write_text(json.dumps(meta))


def test_scaling_is_fit_only_on_explicit_training_groups(tmp_path):
    training_fixture(tmp_path / "data")
    result = train_from_sessions([tmp_path / "data"], tmp_path / "model.json", ["test"])
    decoder = FrozenDecoder.load(tmp_path / "model.json")
    np.testing.assert_allclose(decoder.mean, [3, 4])
    assert result["train_trials"] == 4 and result["test_trials"] == 2
    assert not decoder.payload["timing_validated"]


@pytest.mark.parametrize("duplicate,unknown,groups,match", [
    (True, False, ["test"], "Identical input images"),
    (False, True, ["test"], "cover all test classes"),
    (False, False, ["missing"], "Held-out groups"),
    (False, False, ["train", "test"], "Held-out groups"),
])
def test_invalid_data_splits_are_rejected(tmp_path, duplicate, unknown, groups, match):
    training_fixture(tmp_path / "data", duplicate=duplicate, unknown_class=unknown)
    with pytest.raises(ValueError, match=match):
        train_from_sessions([tmp_path / "data"], tmp_path / "model.json", groups)
    assert not (tmp_path / "model.json").exists()


def test_model_file_is_not_overwritten(tmp_path):
    training_fixture(tmp_path / "data")
    (tmp_path / "model.json").write_text("existing model")
    with pytest.raises(FileExistsError):
        train_from_sessions([tmp_path / "data"], tmp_path / "model.json", ["test"])
    assert (tmp_path / "model.json").read_text() == "existing model"
