"""Live-backend tests use socketpairs and a fake controller; no lab connections."""
import contextlib
import json
import queue
import socket
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from encoders.stimulus_schema import Pulse, StimulusProtocol
from experiments.openloop_backend import NEURAL_GLOBALS, RHXBackend, encode_marker_bytes
from experiments.openloop_data import FeatureSpec, FrozenDecoder
from experiments.openloop_engine import InputTrial, run_session
from intan_interface.marker_stream import MarkerBlock
from intan_interface.spike_stream import SPIKE_MAGIC, SpikeStreamError
from intan_interface.tcp_client import IntanTCPClient
from tests.test_rhx_session import FakeRHXSocket

TIMING = {"max_cross_socket_delay_ms": 20, "max_stream_age_ms": 2000,
          "max_schedule_error_ms": 100, "validated": False}


class SocketPairRHX(IntanTCPClient):
    def __init__(self):
        super().__init__(execution_mode="live")
        self.INTER_COMMAND_DELAY_S = 0
        self.triggers = queue.Queue()
        self.sample = 0
        self.thread = None
        self.stop_producer = threading.Event()
        self.drop_trigger = False
        self.clock_origin = None

    def connect_command(self, *args, **kwargs):
        sock = FakeRHXSocket()
        sock.values["synthetic"] = "False"
        for p in NEURAL_GLOBALS:
            sock.values.setdefault(p.lower(), "1")
        for ch in range(8):
            name = f"a-{ch:03d}"
            sock.values.update({f"{name}.reference": "Hardware", f"{name}.spikethresholdmicrovolts": "-70",
                                f"{name}.enabled": "True"})
            for band in ("", "Low", "High", "Spike", "DC", "Stim"):
                sock.values[f"{name}.tcpdataoutputenabled{band.lower()}"] = "False"
        original = sock.sendall

        def send(data):
            original(data)
            if data.startswith(b"Execute ManualStimTriggerPulse") and not self.drop_trigger:
                key = data.decode().split()[-1].rstrip(";").lower()
                # A hardware trigger is dated by the acquisition clock, even
                # when the fake output thread has not yet flushed older data.
                now_sample = int((time.monotonic() - self.clock_origin) * 30000)
                onset = (max(self.sample, now_sample) // 128 + 1) * 128
                self.triggers.put((onset, key))

        sock.sendall = send
        self._command_socket = sock
        return self

    def connect_waveform(self, *args, **kwargs):
        self._waveform_socket, self.marker_tx = socket.socketpair()
        return self

    def connect_spike(self, *args, **kwargs):
        self._spike_socket, self.spike_tx = socket.socketpair()
        return self

    def record(self, verify=False):
        result = super().record(verify=verify)
        self.clock_origin = time.monotonic()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()
        return result

    def _produce(self):
        active = []
        # Real hardware keeps sampling independently of Python scheduling.
        # Flush every elapsed block in one batch instead of making source
        # time advance by only one block per Python loop under runner load.
        names = sorted(k.split(".")[0].upper() for k, v in self._command_socket.values.items()
                       if k.endswith(".tcpdataoutputenabledstim") and v.lower() == "true")
        while not self.stop_producer.is_set():
            while True:
                try:
                    sample, key = self.triggers.get_nowait()
                    amp, phase, gap, polarity, _ = self._prepared_profiles[key]
                    active.append((sample, int(key[1:]) - 1, round(amp / .5), round(phase * .03),
                                   round(gap * .03), polarity == "NegativeFirst"))
                except queue.Empty:
                    break
            elapsed_samples = int((time.monotonic() - self.clock_origin) * 30000)
            end = max(self.sample + 128, (elapsed_samples // 128 + 1) * 128)
            words = np.zeros((end - self.sample, len(names)), dtype="<u2")
            ts = np.arange(self.sample, end)
            spikes = []
            for onset, ch, amp, phase, gap, negative in active:
                col = names.index(f"A-{ch:03d}")
                words[(ts >= onset) & (ts < onset + phase), col] = amp | (0x100 if negative else 0)
                words[(ts >= onset + phase + gap) & (ts < onset + phase * 2 + gap), col] = amp | (0 if negative else 0x100)
                response = onset + phase * 2 + gap + 120
                if self.sample <= response < end:
                    spikes.append(SPIKE_MAGIC + b"A-000" + struct.pack("<I", response) + b"\x40")
            try:
                self.marker_tx.sendall(encode_marker_bytes(self.sample, words))
                if spikes:
                    self.spike_tx.sendall(b"".join(spikes))
            except OSError:
                return
            self.sample = end
            active = [p for p in active if p[0] + p[3] * 2 + p[4] + 120 >= self.sample]
            self.stop_producer.wait(.02)

    def stop(self, verify=False):
        result = super().stop(verify=verify)
        self.stop_producer.set()
        if self.thread is not None:
            self.thread.join(1)
        return result

    def disconnect(self):
        self.stop_producer.set()
        if self.thread is not None:
            self.thread.join(1)
        super().disconnect()
        for name in ("marker_tx", "spike_tx"):
            stream = getattr(self, name, None)
            if stream is not None:
                stream.close()


def backend(output, channels, epoch, *, timeout=5):
    b = RHXBackend(output, channels, epoch, timing=TIMING, rhx_path="/rhx/test",
                   timeout_s=timeout, preparation_id="fake-controller-test")
    b.client = SocketPairRHX()
    return b


def test_physical_path_keeps_collecting_through_blocking_stimulation_schedule(tmp_path):
    proto = StimulusProtocol([Pulse(0, 0), Pulse(0, 250)], 8, 400)
    trials = [InputTrial("one", "i1", "g", "a", proto),
              InputTrial("two", "i2", "g", "b", StimulusProtocol([Pulse(0, 0)], 8, 400))]
    result = run_session(trials, backend, tmp_path / "session", ["A-000"], {},
                         spec=FeatureSpec(duration_ms=400, n_bins=10, post_blank_ms=2))
    assert result["status"] == "COMPLETE"
    assert result["trial_status_counts"] == {"VALID": 2}
    assert result["contract"]["event_source"] == "RHX-5002"
    rows = [json.loads(s) for s in (tmp_path / "session" / "trials.jsonl").read_text().splitlines()]
    assert len(rows[0]["markers"]) == 2
    assert np.asarray(rows[0]["counts"]).sum() == 2
    assert rows[1]["start_sample"] > rows[0]["end_sample"]
    assert not result["physical_delivery_verified"]
    assert result["acquisition"]["rhx_stop_confirmed"]


def test_missing_stim_marker_fails_session_instead_of_decoding_zeros(tmp_path):
    def dropped(output, channels, epoch):
        b = backend(output, channels, epoch, timeout=.2)
        b.client.drop_trigger = True
        return b
    trials = [InputTrial("one", "image", "g", "a", StimulusProtocol([Pulse(0, 0)], 8, 200))]
    with pytest.raises(TimeoutError, match="hardware markers"):
        run_session(trials, dropped, tmp_path / "failed", ["A-000"], {},
                    spec=FeatureSpec(duration_ms=60, n_bins=6, post_blank_ms=2))
    meta = json.loads((tmp_path / "failed" / "session.json").read_text())
    assert meta["status"] == "FAILED" and meta["processed_trials"] == 0
    assert not meta["arm_tasks"]


def test_advancing_marker_data_cannot_hide_excessive_source_clock_lag(tmp_path):
    b = backend(tmp_path, ["A-000"], "lag-test")
    b.rate = 30000
    b.metadata["max_observed_clock_lag_ms"] = 0
    received = time.monotonic_ns()
    b._record_host_ns = received - int((TIMING["max_stream_age_ms"] + 100) * 1e6)
    block = MarkerBlock(0, np.zeros((128, 1), dtype="<u2"), received)
    b.marker_reader = SimpleNamespace(drain=lambda: [block])
    # Bytes have just arrived and samples advance, but their acquisition time
    # is already too old. Keep this fault check independent of runner speed.
    with pytest.raises(SpikeStreamError, match="clock fell behind"):
        b._drain_once()
    assert b.metadata["max_observed_clock_lag_ms"] > TIMING["max_stream_age_ms"]


def test_late_spike_invalidates_closed_window_and_invokes_fault_stop(tmp_path):
    b = backend(tmp_path, ["A-000"], "test-epoch")
    proto = StimulusProtocol([Pulse(0, 0)], 8, 200)
    stop = threading.Event()
    b.fault_callback = stop.set
    try:
        b.prepare([proto], FeatureSpec(duration_ms=60, n_bins=6, post_blank_ms=2))
        b.start()
        observation = b.observe(proto)
        end = b.closed_through
        b.client.spike_tx.sendall(SPIKE_MAGIC + b"A-000" + struct.pack("<I", end - 1) + b"\x40")
        assert stop.wait(1)
        with pytest.raises(SpikeStreamError, match="Late spike"):
            b.check()
    finally:
        with contextlib.suppress(Exception):
            b.close()


def test_spike_eof_invokes_fault_stop_even_when_main_is_not_polling(tmp_path):
    b = backend(tmp_path, ["A-000"], "test-epoch")
    stop = threading.Event()
    b.fault_callback = stop.set
    try:
        b.prepare([StimulusProtocol([Pulse(0, 0)], 8, 200)], FeatureSpec(duration_ms=60, n_bins=6))
        b.start()
        b.client.spike_tx.shutdown(socket.SHUT_WR)
        assert stop.wait(1)
        with pytest.raises(SpikeStreamError, match="EOF"):
            b.check()
    finally:
        with contextlib.suppress(Exception):
            b.close()


def test_stalled_marker_stream_invokes_stop_without_waiting_for_a_trial(tmp_path):
    b = backend(tmp_path, ["A-000"], "epoch")
    stop = threading.Event()
    b.fault_callback = stop.set
    try:
        b.prepare([StimulusProtocol([Pulse(0, 0)], 8, 200)], FeatureSpec(duration_ms=60, n_bins=6))
        b.start()
        b.client.stop_producer.set()
        b.client.thread.join(1)
        b.timing["max_stream_age_ms"] = 30
        assert stop.wait(1)
        with pytest.raises(SpikeStreamError, match="stalled|lag bound"):
            b.check()
    finally:
        with contextlib.suppress(Exception):
            b.close()


def test_frozen_inference_reaches_arm_adapter_with_real_backend_socket_fixture(tmp_path):
    from experiments.openloop_replay import replay_session
    trials = [InputTrial("one", "image", "g", None, StimulusProtocol([Pulse(0, 0)], 8, 200))]
    spec = FeatureSpec(duration_ms=60, n_bins=6, post_blank_ms=2)
    collected = run_session(trials, backend, tmp_path / "collect", ["A-000"], {}, spec=spec)
    decoder = FrozenDecoder({"schema": "inta-frozen-decoder-v1", "kind": "test-fixture",
        "contract": collected["contract"], "classes": ["a", "b"], "mean": [0] * 6,
        "scale": [1] * 6, "weights": [[[0]] * 6], "biases": [[10]], "timing_validated": False})
    result = run_session(trials, backend, tmp_path / "infer", ["A-000"], {}, spec=spec,
                         decoder=decoder, action_map={"a": "left", "b": ["right", "grab"]})
    assert result["status"] == "COMPLETE" and len(result["arm_tasks"]) == 1
    assert result["arm_tasks"][0]["actions"][0]["action"] == "right"
    assert replay_session(tmp_path / "infer", decoder=decoder)["status"] == "REPLAY_MATCH"


@pytest.mark.parametrize("timing", [{**TIMING, "validated": "false"},
                                    {**TIMING, "validated": True},
                                    {**TIMING, "max_cross_socket_delay_ms": float("nan")}])
def test_live_timing_contract_rejects_invalid_or_unsubstantiated_values(tmp_path, timing):
    with pytest.raises(ValueError):
        RHXBackend(tmp_path, ["A-000"], "epoch", timing=timing, rhx_path="/rhx/test", preparation_id="test")


@pytest.mark.parametrize("unvalidated", ["runtime", "training"])
def test_physical_arm_requires_validated_training_and_runtime_before_homing(tmp_path, unvalidated):
    trials = [InputTrial("one", "image", "g", None, StimulusProtocol([Pulse(0, 0)], 8, 200))]
    spec = FeatureSpec(duration_ms=60, n_bins=6, post_blank_ms=2)
    collected = run_session(trials, backend, tmp_path / "collect", ["A-000"], {}, spec=spec)
    decoder = FrozenDecoder({"schema": "inta-frozen-decoder-v1", "kind": "test-fixture",
        "contract": collected["contract"], "classes": ["a", "b"], "mean": [0] * 6,
        "scale": [1] * 6, "weights": [[[0]] * 6], "biases": [[10]],
        "timing_validated": unvalidated != "training"})
    created = []
    def factory(*args):
        b = backend(*args)
        b.timing.update(validated=unvalidated != "runtime", evidence="test-fixture declaration")
        created.append(b)
        return b
    def no_arm(*args, **kwargs):
        raise AssertionError("Arm must not initialize before timing validation")
    with pytest.raises(ValueError, match="validated timing calibration"):
        run_session(trials, factory, tmp_path / "infer", ["A-000"], {}, spec=spec,
                    decoder=decoder, action_map={"a": "idle", "b": "idle"}, live_arm=True,
                    home=True, arm_factory=no_arm)
    assert not created[0].started
    assert (tmp_path / "infer" / "markers.raw").stat().st_size == 0
