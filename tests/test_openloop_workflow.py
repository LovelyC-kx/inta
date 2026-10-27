import copy
import json
import threading
import time

import numpy as np
import pytest

from encoders.stimulus_schema import Pulse, StimulusProtocol
from experiments.openloop_arm import ArmTaskWorker
from experiments.openloop_backend import SyntheticBackend
from experiments.openloop_data import FeatureSpec, FrozenDecoder, read_session, train_from_sessions
from experiments.openloop_engine import InputTrial, ImageEncoder, run_session
from experiments.openloop_replay import replay_session
from experiments.run_openloop import demo, main


@pytest.fixture(scope="module")
def completed_demo(tmp_path_factory):
    path = tmp_path_factory.mktemp("demo-parent") / "demo"
    result = demo(path)
    return path, result


def test_image_to_training_frozen_inference_arm_and_raw_replay(completed_demo):
    path, result = completed_demo
    assert result["collection"] == result["inference"] == "COMPLETE"
    assert result["replay"] == "REPLAY_MATCH"
    assert result["arm_task_count"] == 6
    assert result["physical_task_success"] is None
    assert result["training"]["train_trials"] == 12
    assert result["training"]["test_trials"] == 6
    meta, rows = read_session(path / "inference")
    assert all(r["status"] == "VALID" and r["decision"]["status"] == "QUEUED" for r in rows)
    assert meta["contract"]["simulated"]
    assert all(not a["position_verified"] for a in meta["arm_tasks"])
    decoder = FrozenDecoder.load(path / "decoder.json")
    replayed = replay_session(path / "inference", decoder=decoder)
    assert [r["prediction"] for r in replayed["predictions"]] == [r["prediction"] for r in rows]


def test_hash_mismatch_prevents_training_or_replay(completed_demo, tmp_path):
    import shutil
    path, _ = completed_demo
    target = tmp_path / "altered"
    shutil.copytree(path / "inference", target)
    with (target / "spikes.raw").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_session(target)


def test_mlp_uses_the_same_portable_inference_contract(completed_demo, tmp_path):
    path, _ = completed_demo
    result = train_from_sessions([path / "collection"], tmp_path / "mlp.json", ["heldout"], kind="mlp")
    assert result["confidence_calibrated"] is False
    assert replay_session(path / "inference", decoder=FrozenDecoder.load(tmp_path / "mlp.json"))["status"] == "REPLAY_MATCH"


def test_contract_change_fails_before_stimulation_or_arm(completed_demo, tmp_path):
    path, _ = completed_demo
    decoder = FrozenDecoder.load(path / "decoder.json")
    proto = StimulusProtocol([Pulse(0, 0)], 8, 200)
    trials = [InputTrial("one", "digest", "test", None, proto)]
    with pytest.raises(ValueError, match="contract differs"):
        run_session(trials, SyntheticBackend, tmp_path / "mismatch", ["A-000"], {"different": True},
                    spec=FeatureSpec(duration_ms=500, n_bins=20), decoder=decoder,
                    action_map={"0": "left", "1": "right", "2": "idle"})
    meta = json.loads((tmp_path / "mismatch" / "session.json").read_text())
    assert meta["status"] == "FAILED" and meta["processed_trials"] == 0
    assert (tmp_path / "mismatch" / "markers.raw").stat().st_size == 0


def test_synthetic_decoder_cannot_be_used_with_physical_contract(completed_demo):
    path, _ = completed_demo
    decoder = FrozenDecoder.load(path / "decoder.json")
    contract = copy.deepcopy(decoder.contract)
    contract["simulated"] = False
    with pytest.raises(ValueError, match="contract differs"):
        decoder.require_contract(contract)


def test_single_input_may_use_only_a_subset_of_the_trained_stimulus_bank(completed_demo, tmp_path):
    path, _ = completed_demo
    decoder = FrozenDecoder.load(path / "decoder.json")
    encoder = {k: v for k, v in decoder.contract["encoder"].items()
               if k not in ("stim_prefix", "stim_profiles", "stim_step_ua")}
    trials = [InputTrial("one", "new-image", "unseen", None, StimulusProtocol([Pulse(0, 0)], 8, 200))]
    result = run_session(trials, SyntheticBackend, tmp_path / "one-input", decoder.contract["channel_names"], encoder,
                         spec=FeatureSpec(**decoder.contract["features"]), decoder=decoder,
                         action_map={"0": "idle", "1": "idle", "2": "idle"}, min_confidence=0)
    assert result["status"] == "COMPLETE" and len(result["arm_tasks"]) == 1


def test_silence_does_not_produce_an_action(tmp_path):
    class SilentBackend(SyntheticBackend):
        def observe(self, protocol):
            observation = super().observe(protocol)
            observation["events"] = []
            return observation
    trials = [InputTrial("one", "image", "g", "a", StimulusProtocol([Pulse(0, 0)], 8, 200))]
    result = run_session(trials, SilentBackend, tmp_path / "silent", ["A-000"], {},
                         spec=FeatureSpec(duration_ms=100, n_bins=5))
    assert result["status"] == "NO_VALID_TRIALS"
    rows = [json.loads(s) for s in (tmp_path / "silent" / "trials.jsonl").read_text().splitlines()]
    assert rows[0]["decision"] == {"status": "REJECTED", "reason": "silent"}
    assert not result["arm_tasks"]


def test_empty_protocol_is_recorded_as_invalid_not_decoded(tmp_path):
    trials = [InputTrial("empty", "i0", "g", "a", StimulusProtocol([], 8, 200)),
              InputTrial("valid", "i1", "g", "a", StimulusProtocol([Pulse(0, 0)], 8, 200))]
    result = run_session(trials, SyntheticBackend, tmp_path / "session", ["A-000"], {},
                         spec=FeatureSpec(duration_ms=100, n_bins=5))
    assert result["trial_status_counts"] == {"INVALID": 1, "VALID": 1}


def test_no_implicit_physical_output_and_no_output_overwrite(tmp_path, completed_demo):
    assert main(["collect", "--manifest", "absent.json", "--out", str(tmp_path / "none")]) == 1
    assert not (tmp_path / "none").exists()
    path, _ = completed_demo
    with pytest.raises(FileExistsError):
        demo(path)


def test_rate_protocol_is_deterministic_per_image_independent_of_input_order(tmp_path):
    first, second = tmp_path / "first.npy", tmp_path / "second.npy"
    np.save(first, np.random.default_rng(1).random((28, 28)).astype(np.float32))
    np.save(second, np.random.default_rng(2).random((28, 28)).astype(np.float32))
    encoder = ImageEncoder(encoding="rate", simulated=True)
    before, sha = encoder.encode(first)
    encoder.encode(second)
    after, sha_after = encoder.encode(first)
    assert sha == sha_after and before.pulses == after.pulses


def test_actual_pulse_profile_is_applied_before_charge_trimming(tmp_path):
    image = tmp_path / "image.npy"
    np.save(image, np.random.default_rng(3).random((28, 28)).astype(np.float32))
    encoder = ImageEncoder(simulated=True, pulse_profile={"amplitude_ua": 20, "phase_duration_us": 480})
    protocol, _ = encoder.encode(image)
    assert all(p.amplitude_ua == 20 and p.phase_duration_us == 480 for p in protocol.pulses)
    assert protocol.safety_report()["total_charge_nc"] <= 180


def test_too_short_window_fails_before_backend_construction(tmp_path):
    trials = [InputTrial("one", "image", "g", None,
                         StimulusProtocol([Pulse(0, 0), Pulse(0, 500)], 8, 1000))]
    def no_backend(*args):
        raise AssertionError("Backend must not be constructed")
    with pytest.raises(ValueError, match="before the planned stimulus"):
        run_session(trials, no_backend, tmp_path / "out", ["A-000"], {},
                    spec=FeatureSpec(duration_ms=100, n_bins=5))
    assert not (tmp_path / "out").exists()


class SlowArm:
    ACTIONS = {"left", "right", "idle", "home", "stop"}
    dry_run = True
    def __init__(self):
        self.started, self.release = threading.Event(), threading.Event()
        self.stopped = False
    def home(self):
        return {"status": "simulated"}
    def execute(self, action):
        self.started.set()
        self.release.wait(1)
        if self.stopped:
            raise RuntimeError("Stopped")
        return {"status": "simulated", "action": action}
    def stop(self):
        self.stopped = True
        self.release.set()
    def close(self):
        pass


def test_arm_is_nonblocking_and_rejects_duplicate_busy_low_confidence_and_expired():
    arm = SlowArm()
    worker = ArmTaskWorker({"a": "left"}, lambda *_: None, arm=arm)
    prediction = {"label": "a", "confidence": .99}
    try:
        assert worker.submit("one", prediction)["status"] == "QUEUED"
        assert arm.started.wait(1)
        assert worker.submit("one", prediction)["reason"] == "duplicate_trial"
        assert worker.submit("two", prediction)["reason"] == "arm_busy"
        assert worker.submit("three", {"label": "a", "confidence": .1})["reason"] == "low_confidence"
        assert worker.submit("four", prediction, decision_time=time.time() - 10)["reason"] == "expired_or_future_decision"
        arm.release.set()
        worker.wait()
    finally:
        worker.close()


def test_acquisition_fault_can_interrupt_an_in_progress_arm_macro():
    arm = SlowArm()
    worker = ArmTaskWorker({"a": ["left", "right"]}, lambda *_: None, arm=arm)
    worker.submit("one", {"label": "a", "confidence": 1})
    assert arm.started.wait(1)
    worker.fail()
    worker.close(failed=True)
    assert arm.stopped and not worker.results


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), True])
def test_invalid_decision_confidence_is_rejected_before_queueing(confidence):
    worker = ArmTaskWorker({"a": "idle"}, lambda *_: None)
    try:
        with pytest.raises(ValueError, match="finite numbers"):
            worker.submit("one", {"label": "a", "confidence": confidence})
        assert not worker.results
    finally:
        worker.close()


@pytest.mark.parametrize("phase", ["probe", "classify"])
def test_legacy_stimulation_phases_stop_before_creating_a_session(monkeypatch, phase):
    from experiments import run_2d_experiment as old
    import sys
    monkeypatch.setattr(sys, "argv", ["run_2d_experiment.py", "--phase", phase])
    def no_output(*args):
        raise AssertionError("Retired phase must not create files or connect hardware")
    monkeypatch.setattr(old, "create_output_dir", no_output)
    with pytest.raises(SystemExit) as exc:
        old.main()
    assert exc.value.code == 2
