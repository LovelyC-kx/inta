"""One open-loop owner: input -> protocol -> RHX events -> decoder -> arm task."""
from collections import Counter
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import torch

from encoders.cnn_feature_extractor import CNNFeatureExtractor
from encoders.pipeline import EncodingPipeline
from encoders.stimulus_schema import Pulse, StimulusProtocol, DEFAULT_TRIM_TARGET_NC, trim_to_charge_budget
from intan_interface.spike_stream import CHANNEL_NAME

from experiments.openloop_arm import ArmTaskWorker
from experiments.openloop_backend import protocol_dict
from experiments.openloop_data import FeatureSpec, canonical, digest, extract_observation, file_hash, make_contract


@dataclass
class InputTrial:
    trial_id: str
    image_sha256: str
    group: str
    label: str | None
    protocol: object


class ImageEncoder:
    def __init__(self, checkpoint=None, *, encoding="delayed_phase", seed=42, simulated=False,
                 encoder_params=None, pulse_profile=None):
        self.encoding, self.seed = encoding, seed
        self.encoder_params = dict(encoder_params or {})
        self.pulse_profile = dict(pulse_profile or {})
        if set(self.pulse_profile) - {"amplitude_ua", "phase_duration_us", "interphase_delay_us", "polarity", "waveform"}:
            raise ValueError("Unsupported pulse profile field")
        StimulusProtocol([Pulse(0, 0, **self.pulse_profile)], 1, 2000).validate_safety()
        if checkpoint is None:
            if not simulated:
                raise ValueError("Real input encoding requires a trained --cnn-checkpoint")
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                cnn = CNNFeatureExtractor(8, 10, 28)
            self.size = 28
            h = hashlib.sha256()
            for name, value in sorted(cnn.state_dict().items()):
                h.update(name.encode())
                h.update(value.numpy().tobytes())
            checkpoint_hash = h.hexdigest()
        else:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.size = ckpt.get("input_size", 28)
            if type(self.size) is not int or not 8 <= self.size <= 512:
                raise ValueError("Unsupported CNN input size")
            cnn = CNNFeatureExtractor(ckpt["n_channels"], ckpt["n_classes"], self.size)
            cnn.load_state_dict(ckpt["state_dict"], strict=True)
            checkpoint_hash = file_hash(checkpoint)
        self.pipeline = EncodingPipeline(cnn, encoder_type=encoding, encoder_params=self.encoder_params,
                                         trim_warn=False, trim_target_nc=None)
        self.metadata = {"cnn_sha256": checkpoint_hash, "encoding": encoding, "seed": seed,
                         "encoder_params": self.encoder_params, "input_size": self.size,
                         "preprocessing": "PIL-L-or-npy-float01_bilinear_resize-v1",
                         "trim_target_nc": DEFAULT_TRIM_TARGET_NC, "pulse_profile": self.pulse_profile}

    def encode(self, path):
        from PIL import Image
        path = Path(path)
        if path.suffix.lower() == ".npy":
            array = np.load(path, allow_pickle=False)
            if (array.ndim != 2 or not np.issubdtype(array.dtype, np.number) or
                    not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1)):
                raise ValueError("Numpy images must be finite 2D grayscale values in [0,1]")
            image = Image.fromarray(array.astype(np.float32))
            array = np.asarray(image.resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32).copy()
        else:
            with Image.open(path) as image:
                array = np.asarray(image.convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR),
                                   dtype=np.float32).copy() / 255
        image_hash = hashlib.sha256(array.astype("<f4").tobytes()).hexdigest()
        # Rate encoding is repeatable per input, independent of manifest order.
        if self.encoding == "rate":
            self.pipeline.encoder.rng = np.random.default_rng(int(image_hash[:16], 16) ^ self.seed)
        proto = self.pipeline.encode_to_protocol(array)
        proto = StimulusProtocol([replace(p, **self.pulse_profile) for p in proto.pulses],
                                  proto.n_channels, proto.duration_ms)
        proto = trim_to_charge_budget(proto, DEFAULT_TRIM_TARGET_NC)
        proto.validate_safety()
        return proto, image_hash

    def manifest(self, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["trials"] if isinstance(data, dict) else data
        if not isinstance(entries, list) or not entries:
            raise ValueError("Manifest requires a nonempty list of trials")
        result, ids = [], set()
        for entry in entries:
            trial_id = entry.get("trial_id")
            if not isinstance(trial_id, str) or not trial_id or trial_id in ids:
                raise ValueError("Each trial needs a unique nonempty string trial_id")
            ids.add(trial_id)
            image = Path(entry["image"])
            if not image.is_absolute():
                image = path.parent / image
            proto, sha = self.encode(image)
            label, group = entry.get("label"), entry.get("group", sha)
            if label is not None and (not isinstance(label, str) or not label):
                raise ValueError("Labels must be nonempty strings, or omitted during inference")
            if not isinstance(group, str) or not group:
                raise ValueError("Each trial requires a nonempty group")
            result.append(InputTrial(trial_id, sha, group, label, proto))
        return result


class Journal:
    def __init__(self, path):
        self.events = (path / "events.jsonl").open("x", encoding="utf-8")
        self.trials = (path / "trials.jsonl").open("x", encoding="utf-8")
        self.lock = threading.Lock()

    def __call__(self, kind, payload):
        with self.lock:
            self.events.write(canonical({"kind": kind, "host_monotonic_ns": time.monotonic_ns(),
                                         "host_wall_time": time.time(), "payload": payload}) + "\n")
            self.events.flush()

    def trial(self, row):
        with self.lock:
            self.trials.write(canonical(row) + "\n")
            self.trials.flush()

    def close(self):
        self.events.close()
        self.trials.close()


def run_session(trials, backend_factory, output, channels, encoder_metadata, *, spec=None,
                decoder=None, action_map=None, live_arm=False, arm_port=None, home=False,
                min_confidence=.8, arm_factory=ArmTaskWorker):
    """Run collect-only or frozen inference. Labels never enter a backend/arm policy."""
    channels = tuple(channels)
    if (not channels or len(set(channels)) != len(channels) or
            any(not CHANNEL_NAME.fullmatch(n) for n in channels)):
        raise ValueError("Recording channels must be unique full RHX names in model order")
    if not trials or len({t.trial_id for t in trials}) != len(trials):
        raise ValueError("A nonempty unique trial plan is required")
    for trial in trials:
        trial.protocol.validate_safety()
    spec = spec or FeatureSpec()
    for trial in trials:
        if trial.protocol.pulses:
            pulses = trial.protocol.pulses
            planned_end = max(p.time_ms + (p.phase_duration_us * 2 + p.interphase_delay_us) / 1000
                              for p in pulses) - min(p.time_ms for p in pulses)
            if planned_end > spec.offset_ms + spec.duration_ms:
                raise ValueError("Observation window ends before the planned stimulus; enlarge duration/offset")
    if (decoder is None) != (action_map is None):
        raise ValueError("Inference requires both a frozen decoder and an explicit action map")
    if decoder is not None and set(action_map) != set(decoder.classes):
        raise ValueError("Action map must cover exactly the frozen model's class labels")
    if not isinstance(min_confidence, (int, float)) or not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0,1]")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output)
    epoch = uuid.uuid4().hex
    backend = worker = None
    failure = None
    meta = {"schema": "inta-openloop-v1", "session_id": epoch, "status": "PREPARING",
            "physical_delivery_verified": False, "physical_task_success": None,
            "intended_trials": len(trials), "min_confidence": min_confidence}
    statuses, processed = Counter(), 0
    try:
        backend = backend_factory(output, channels, epoch)
        prepared_protocols = [t.protocol for t in trials]
        # Keep the trained profile bank stable even when this particular input
        # uses only a subset of electrodes. These are configuration rows, not
        # additional executed stimuli; each bank row is validated separately.
        if decoder is not None:
            bank = decoder.contract["encoder"]
            for name, p in bank["stim_profiles"].items():
                ch = int(name.split("-")[1])
                pulse = Pulse(ch, 0, p["amplitude_code"] * bank["stim_step_ua"],
                              p["phase_samples"] * 1e6 / decoder.contract["sample_rate_hz"],
                              p["interphase_samples"] * 1e6 / decoder.contract["sample_rate_hz"],
                              "NegativeFirst" if p["negative_first"] else "PositiveFirst", p["waveform"])
                prepared_protocols.append(StimulusProtocol([pulse], max(t.protocol.n_channels for t in trials), 2000))
        neural = backend.prepare(prepared_protocols, spec)
        encoder_contract = {**encoder_metadata, "stim_prefix": backend.prefix,
                            "stim_profiles": backend.profiles, "stim_step_ua": backend.hardware.stim_step_ua}
        contract = make_contract(channels, backend.rate, spec, encoder_contract, neural, simulated=backend.simulated)
        meta.update(contract=contract, acquisition=backend.metadata,
                    data_scope="synthetic interface only" if backend.simulated else "physical RHX recording")
        if decoder is not None:
            decoder.require_contract(contract)
            meta["decoder_sha256"] = digest(decoder.payload)
        if live_arm and (backend.simulated or backend.timing.get("validated") is not True or
                         decoder is None or decoder.payload.get("timing_validated") is not True):
            raise ValueError("Live arm requires a physical backend and explicitly validated timing calibration")
        if action_map is not None:
            worker = arm_factory(action_map, journal, live=live_arm, port=arm_port, home=home,
                                 min_confidence=min_confidence)
            backend.fault_callback = worker.fail
        journal("session_prepared", {"contract_sha256": digest(contract), "acquisition": backend.metadata})
        backend.start()
        meta["status"] = "RUNNING"
        for trial in trials:
            backend.check()
            if worker is not None and worker.error is not None:
                raise RuntimeError("Arm worker failed") from worker.error
            journal("input", {"trial_id": trial.trial_id, "image_sha256": trial.image_sha256,
                              "protocol": protocol_dict(trial.protocol)})
            row = {"trial_id": trial.trial_id, "image_sha256": trial.image_sha256,
                   "group": trial.group, "label": trial.label, "protocol": protocol_dict(trial.protocol),
                   "status": "INVALID", "counts": None, "decision": None}
            if not trial.protocol.pulses:
                row["decision"] = {"status": "REJECTED", "reason": "empty_protocol"}
            else:
                observation = backend.observe(trial.protocol)
                window, counts, status = extract_observation(observation, contract, epoch)
                row.update(status=status, counts=None if counts is None else counts.tolist(),
                           anchor_sample=observation["anchor"], start_sample=window.start_sample,
                           end_sample=window.end_sample, markers=[asdict(m) for m in observation["markers"]],
                           valid_samples_per_channel=window.valid_samples_per_channel,
                           excluded_id_events=window.excluded_id_events, masked_events=window.masked_events,
                           completeness=observation["completeness"],
                           ttl_edges=observation.get("ttl_edges", []),
                           host_trigger_timing=observation["host_trigger_timing"])
                journal("response", {"trial_id": trial.trial_id, "status": status,
                                     "start_sample": window.start_sample, "end_sample": window.end_sample,
                                     "markers": row["markers"], "completeness": observation["completeness"]})
                if status != "VALID":
                    row["decision"] = {"status": "REJECTED", "reason": status.lower()}
                elif decoder is not None:
                    prediction = decoder.predict(counts)
                    row["prediction"] = prediction
                    backend.check()
                    row["decision"] = worker.submit(trial.trial_id, prediction)
                    journal("decision", {"trial_id": trial.trial_id, "prediction": prediction,
                                         "dispatch": row["decision"]})
            statuses[row["status"]] += 1
            processed += 1
            journal.trial(row)
        if worker is not None:
            worker.wait()
        backend.check()
        meta["status"] = "COMPLETE" if statuses["VALID"] else "NO_VALID_TRIALS"
    except BaseException as exc:
        failure = exc
        meta.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        if backend is not None:
            backend.cancel.set()
        if worker is not None:
            worker.fail()
        journal("fault", {"error": meta["error"]})
    finally:
        for obj, close in ((backend, lambda b: b.close()),
                           (worker, lambda w: w.close(failed=failure is not None))):
            if obj is not None:
                try:
                    close(obj)
                except Exception as exc:
                    failure = failure or exc
                    meta.update(status="FAILED", cleanup_error=f"{type(exc).__name__}: {exc}")
        meta.update(processed_trials=processed, trial_status_counts=dict(statuses),
                    arm_tasks=[] if worker is None else worker.results)
        journal("session_end", {"status": meta["status"], "trial_status_counts": dict(statuses)})
        journal.close()
        names = ["events.jsonl", "trials.jsonl", "markers.raw", "spikes.raw"]
        meta["file_sha256"] = {name: file_hash(output / name) for name in names if (output / name).exists()}
        (output / "session.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    if failure is not None:
        raise failure
    return meta
