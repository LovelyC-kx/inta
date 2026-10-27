"""Continuous RHX acquisition and an explicitly synthetic interface backend."""
from dataclasses import asdict
from pathlib import Path
import struct
import threading
import time

import numpy as np

from encoders.stimulus_schema import Pulse, StimulusProtocol
from intan_interface.hardware_profile import RHSProfile
from intan_interface.marker_stream import MAGIC, MarkerParser, MarkerStreamReader, StimMarkerTracker
from intan_interface.spike_stream import SPIKE_MAGIC, SpikePacketParser, SpikeStreamError, SpikeStreamReader
from intan_interface.spike_windows import ProgressMarker
from intan_interface.tcp_client import IntanTCPClient

from experiments.openloop_data import canonical

NEURAL_GLOBALS = ("Version", "DSPEnabled", "ActualDSPCutoffFreqHertz", "ActualLowerBandwidthHertz",
    "ActualUpperBandwidthHertz", "NotchFilterFreqHertz", "HighpassFilterOrder", "HighpassFilterType",
    "HighpassFilterCutoffFreqHertz", "ArtifactSuppressionEnabled", "ArtifactSuppressionThresholdMicroVolts",
    "UseMedianReference")


def protocol_dict(protocol):
    return {"pulses": [asdict(p) for p in protocol.pulses],
            "n_channels": protocol.n_channels, "duration_ms": protocol.duration_ms}


def protocol_from_dict(data):
    result = StimulusProtocol([Pulse(**p) for p in data["pulses"]], data["n_channels"], data["duration_ms"])
    result.validate_safety()
    return result


def compile_profiles(protocols, prefix, hardware):
    result = {}
    for proto in protocols:
        proto.validate_safety()
        for p in proto.pulses:
            name = f"{prefix}-{p.channel:03d}"
            value = {**hardware.validate_pulse(p), "negative_first": p.polarity == "NegativeFirst",
                     "waveform": p.waveform}
            if name in result and result[name] != value:
                raise ValueError("Per-channel pulse profiles must be constant throughout the session")
            result[name] = value
    if not result:
        raise ValueError("No stimulation pulses in the entire input plan")
    return dict(sorted(result.items()))


def check_markers(markers, protocol, prefix, rate, max_error_ms):
    if len(markers) != len(protocol.pulses):
        raise SpikeStreamError(f"Expected {len(protocol.pulses)} digital pulses, received {len(markers)}")
    first_actual = min(p.start_sample for p in markers)
    first_planned = min(p.time_ms for p in protocol.pulses)
    for ch in range(protocol.n_channels):
        planned = [p for p in protocol.pulses if p.channel == ch]
        actual = sorted((m for m in markers if m.native_channel == f"{prefix}-{ch:03d}"), key=lambda m: m.start_sample)
        if len(planned) != len(actual):
            raise SpikeStreamError("Per-channel pulse count mismatch")
        for p, m in zip(planned, actual):
            error_ms = (m.start_sample - first_actual) * 1000 / rate - (p.time_ms - first_planned)
            if abs(error_ms) > max_error_ms:
                raise SpikeStreamError(f"Digital stimulation schedule error {error_ms:.3f}ms exceeds configured bound")
    return first_actual


def encode_marker_bytes(start, words):
    """Synthetic fixture producer, also used to validate the real wire parser."""
    if len(words) % 128:
        raise ValueError("Whole RHX data blocks required")
    dtype = np.dtype([("ts", "<u4"), ("words", "<u2", (words.shape[1],))])
    frames = np.zeros(len(words), dtype=dtype)
    frames["ts"] = (np.arange(len(words), dtype=np.uint64) + start).astype("<u4")
    frames["words"] = words
    return b"".join(MAGIC + frames[i:i + 128].tobytes() for i in range(0, len(words), 128))


class SyntheticBackend:
    """Protocol-driven fake responses for plumbing only. Never receives labels."""
    simulated = True

    def __init__(self, output, channels, epoch, *, prefix="A", rate=30000):
        self.output, self.channels, self.epoch = Path(output), tuple(channels), epoch
        self.prefix, self.rate = prefix, rate
        self.hardware = RHSProfile("synthetic-interface-v1", "ControllerStimRecord", True, False, rate, .5, False)
        self.next_sample = 0
        self.cancel = threading.Event()
        self.fault_callback = None
        self.metadata = {}
        self.files = []

    def prepare(self, protocols, spec):
        self.spec = spec
        self.profiles = compile_profiles(protocols, self.prefix, self.hardware)
        self.names = tuple(self.profiles)
        self.marker_parser = MarkerParser(self.names)
        self.tracker = StimMarkerTracker(self.names, self.profiles)
        self.spike_parser = SpikePacketParser(self.epoch)
        self.files = [(self.output / name).open("wb", buffering=0) for name in ("markers.raw", "spikes.raw")]
        self.metadata = {"hardware": self.hardware.as_dict(), "stim_profiles": self.profiles,
                         "marker_channels": self.names, "completeness": "synthetic_exact",
                         "physical_delivery_verified": False}
        return {"synthetic_response_generator": "protocol-driven-v1"}

    def start(self):
        pass

    def observe(self, protocol):
        if self.cancel.is_set():
            raise InterruptedError("Acquisition cancelled")
        onset = self.next_sample + int(.1 * self.rate)
        times = [(p, onset + round(p.time_ms * self.rate / 1000)) for p in protocol.pulses]
        anchor = min(t for _, t in times)
        window_end = anchor + self.spec.samples(self.rate, "offset_ms") + self.spec.samples(self.rate, "duration_ms")
        last_end = max(t + self.profiles[f"{self.prefix}-{p.channel:03d}"]["phase_samples"] * 2 +
                       self.profiles[f"{self.prefix}-{p.channel:03d}"]["interphase_samples"] for p, t in times)
        if last_end > window_end:
            raise SpikeStreamError("Observation window ends before stimulation completes")
        end = ((window_end + 127) // 128 + 1) * 128
        words = np.zeros((end - self.next_sample, len(self.names)), dtype="<u2")
        spike_specs = []
        for p, sample in times:
            name = f"{self.prefix}-{p.channel:03d}"
            code = self.profiles[name]
            a = sample - self.next_sample
            phase, gap = code["phase_samples"], code["interphase_samples"]
            first, second = code["amplitude_code"], code["amplitude_code"]
            if code["negative_first"]:
                first |= 0x100
            else:
                second |= 0x100
            col = self.names.index(name)
            words[a:a + phase, col] = first
            words[a + phase + gap:a + 2 * phase + gap, col] = second
            # Deliberately simple response to protocol channel/timing, no label access.
            for j in range(3):
                response = sample + phase * 2 + gap + self.spec.samples(self.rate, "post_blank_ms") + 30 + j * 61
                if response < window_end:
                    spike_specs.append((response, self.channels[(p.channel * 3 + j) % len(self.channels)]))
        marker_bytes = encode_marker_bytes(self.next_sample, words)
        spike_bytes = b"".join(SPIKE_MAGIC + name.encode() + struct.pack("<I", sample & 0xffffffff) + b"\x40"
                               for sample, name in sorted(spike_specs))
        self.files[0].write(marker_bytes)
        self.files[1].write(spike_bytes)
        markers = []
        for block in self.marker_parser.feed(marker_bytes):
            got, _ = self.tracker.feed(block)
            markers.extend(got)
        events = self.spike_parser.feed(spike_bytes)
        check_markers(markers, protocol, self.prefix, self.rate, 1)
        self.next_sample = end
        return {"anchor": anchor, "markers": markers, "events": events,
                "watermark": ProgressMarker(end, self.epoch, "stim"),
                "completeness": "synthetic_exact", "host_trigger_timing": []}

    def check(self):
        if self.cancel.is_set():
            raise InterruptedError("Session cancelled")

    def close(self):
        for stream in self.files:
            stream.close()


class RHXBackend:
    simulated = False

    def __init__(self, output, channels, epoch, *, timing, rhx_path, prefix="A", host="127.0.0.1",
                 command_port=5000, waveform_port=5001, spike_port=5002, timeout_s=15, ttl_bit=None,
                 preparation_id=None):
        self.output, self.channels, self.epoch = Path(output), tuple(channels), epoch
        self.prefix, self.timing, self.rhx_path = prefix, dict(timing), rhx_path
        if not isinstance(preparation_id, str) or not preparation_id:
            raise ValueError("A preparation_id is required to bind a decoder to this culture/organoid")
        self.preparation_id = preparation_id
        if type(timing.get("validated", False)) is not bool:
            raise ValueError("timing.validated must be a JSON boolean")
        if timing.get("validated") and not isinstance(timing.get("evidence"), str):
            raise ValueError("Validated timing requires recorded measurement evidence")
        if timing.get("validated") and not timing["evidence"].strip():
            raise ValueError("Validated timing requires nonempty measurement evidence")
        for field in ("max_cross_socket_delay_ms", "max_stream_age_ms", "max_schedule_error_ms"):
            value = timing.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or
                    not 0 < value <= 10000):
                raise ValueError(f"An explicit positive timing bound is required: {field}")
        if (not isinstance(rhx_path, str) or not rhx_path or not rhx_path.isascii() or
                any(ch.isspace() for ch in rhx_path)):
            raise ValueError("RHX save path must be nonempty ASCII without whitespace (RHX TCP syntax)")
        self.client = IntanTCPClient(host, command_port, waveform_port, spike_port, execution_mode="live")
        self.timeout_s, self.ttl_bit = timeout_s, ttl_bit
        self.cancel = threading.Event()
        self.fault_callback = None
        self.files, self.readers = [], []
        self.events, self.markers, self.ttl_edges = [], [], []
        self.progress = None
        self.closed_through = -1
        self.last_progress_ns = None
        self._record_host_ns = None
        self._clock_origin = None
        self.metadata = {}
        self._watchdog = None
        self._watch_stop = threading.Event()
        self._collector_stop = threading.Event()
        self._collector = None
        self._state_lock = threading.RLock()
        self._fault = None
        self.started = False

    def snapshot_neural(self):
        c = self.client
        return {"preparation_id": self.preparation_id,
                "global": {p: c._get_value(p) for p in NEURAL_GLOBALS},
                "channels": {n: {p: c._get_value(f"{n}.{p}") for p in
                                      ("Reference", "SpikeThresholdMicroVolts", "Enabled")} for n in self.channels}}

    def _set_checked(self, parameter, value):
        self.client.set(parameter, value)
        if self.client._get_value(parameter).casefold() != str(value).casefold():
            raise RuntimeError(f"RHX ignored {parameter}={value}")

    def prepare(self, protocols, spec):
        self.spec = spec
        c = self.client
        c.connect_command()
        if c.get("RunMode") != "Return: RunMode Stop":
            raise RuntimeError("Stop RHX before starting a new owned open-loop session")
        initial_hardware = RHSProfile.read(c._get_value)
        if initial_hardware.synthetic:
            raise ValueError("Real collection requires a physical RHS controller; use simulate for interface tests")
        for name in self.channels:
            port, index = name.split("-")
            if port not in "ABCD" or int(index) >= int(c._get_value(f"{port}.NumberAmplifierChannels")):
                raise ValueError(f"Recording channel is absent from RHS inventory: {name}")
        # TCP hardware filter state must be explicitly uploaded, not merely
        # reported as an 'Actual...' value computed in the GUI state object.
        c.wait_for_upload()
        c.send_command("Execute UploadBandwidthSettings", expect_response=False)
        c.wait_for_upload()
        c.prepare_protocols(protocols, self.prefix)
        session = c.prepared_session
        if session["hardware"]["synthetic"] or session["hardware"]["playback"]:
            raise ValueError("Real collection requires a physical RHS controller; use simulate for interface tests")
        self.hardware = RHSProfile(**session["hardware"])
        self.rate = self.hardware.sample_rate_hz
        self.profiles = compile_profiles(protocols, self.prefix, self.hardware)
        self.names = tuple(self.profiles)
        self.neural = self.snapshot_neural()
        if any(v["Enabled"].casefold() != "true" for v in self.neural["channels"].values()):
            raise ValueError("Every selected RHX spike recording channel must be enabled")
        c.send_command("Execute ClearAllDataOutputs", expect_response=False)
        self._set_checked("TCPNumberDataBlocksPerWrite", 1)
        for name in self.names:
            self._set_checked(f"{name}.TCPDataOutputEnabledStim", "True")
        for name in self.channels:
            self._set_checked(f"{name}.TCPDataOutputEnabledSpike", "True")
        if self.ttl_bit is not None:
            limit = 16 if self.hardware.expander_connected else 2
            if not 0 <= self.ttl_bit < limit:
                raise ValueError("TTL input bit is absent from this controller/expander inventory")
            self._set_checked(f"DIGITAL-IN-{self.ttl_bit + 1:02d}.TCPDataOutputEnabled", "True")
        # Verify that no old amplifier payload changes the declared word layout.
        for port in "ABCD":
            count = int(c._get_value(f"{port}.NumberAmplifierChannels"))
            for ch in range(count):
                name = f"{port}-{ch:03d}"
                for band in ("", "Low", "High", "Spike", "DC", "Stim"):
                    expected = (band == "Stim" and name in self.names) or (band == "Spike" and name in self.channels)
                    actual = c._get_value(f"{name}.TCPDataOutputEnabled{band}").casefold()
                    if actual != str(expected).casefold():
                        raise ValueError(f"Unexpected TCP stream selection: {name}/{band}")
        self._set_checked("FileFormat", "OneFilePerSignalType")
        self._set_checked("CreateNewDirectory", "True")
        c.set_filename(self.rhx_path, f"inta_{self.epoch}")
        if c._get_value("Filename.Path") != self.rhx_path or c._get_value("Filename.BaseFilename") != f"inta_{self.epoch}":
            raise ValueError("RHX recording filename/path readback mismatch")
        c.connect_waveform()
        c.connect_spike()
        self.files = [(self.output / name).open("wb", buffering=0) for name in ("markers.raw", "spikes.raw")]
        self.marker_parser = MarkerParser(self.names, digital_in=self.ttl_bit is not None)
        self.tracker = StimMarkerTracker(self.names, self.profiles, ttl_bit=self.ttl_bit)
        self.marker_reader = MarkerStreamReader(c._waveform_socket, self.marker_parser, self.files[0])
        self.spike_reader = SpikeStreamReader(c._spike_socket, acquisition_epoch=self.epoch, raw_sink=self.files[1])
        self.readers = [self.marker_reader, self.spike_reader]
        self.metadata = {**session, "stim_profiles": self.profiles, "marker_channels": self.names,
                         "ttl_bit": self.ttl_bit, "timing": self.timing,
                         "completeness": "bounded_cross_socket_delay_assumption",
                         "rhx_filename_path": self.rhx_path, "rhx_base_filename": f"inta_{self.epoch}",
                         "max_observed_clock_lag_ms": 0.0, "rhx_stop_confirmed": True}
        return self.neural

    def start(self):
        for reader in self.readers:
            reader.start()
        self._collector = threading.Thread(target=self._collect, name="RHX-continuous-collector", daemon=True)
        self._collector.start()
        self._record_host_ns = time.monotonic_ns()
        self.started = True
        self.metadata["rhx_stop_confirmed"] = False
        self.client.record(verify=True)
        self._watchdog = threading.Thread(target=self._watch, name="RHX-fault-watchdog", daemon=True)
        self._watchdog.start()
        self._until(lambda: self.progress is not None)

    def _watch(self):
        while not self._watch_stop.wait(.005):
            error = next((r.error for r in self.readers if r.error is not None), None)
            if self.last_progress_ns is not None and time.monotonic_ns() - self.last_progress_ns > self.timing["max_stream_age_ms"] * 1e6:
                error = SpikeStreamError("5001 stream stalled beyond configured age bound")
            if error is not None:
                self._fault = error
                self.cancel.set()
                if self.fault_callback is not None:
                    self.fault_callback()
                return

    def check(self):
        if self._fault is not None:
            raise SpikeStreamError(str(self._fault)) from self._fault
        for reader in self.readers:
            if reader.error is not None:
                raise SpikeStreamError(str(reader.error)) from reader.error
        if self.cancel.is_set():
            raise InterruptedError("Session cancelled")
        if self.progress is not None and self.progress > self.rate * 3600:
            raise SpikeStreamError("One-hour epoch limit reached; finish and start a new session")

    def _collect(self):
        try:
            while not self._collector_stop.is_set():
                self._drain_once()
                self._collector_stop.wait(.002)
        except Exception as exc:
            self._fault = exc
            self.cancel.set()
            if self.fault_callback is not None:
                self.fault_callback()

    def _pump(self):
        self.check()

    def _drain_once(self):
        with self._state_lock:
            self._drain_locked()

    def _drain_locked(self):
        self.check()
        for block in self.marker_reader.drain():
            if self.progress is None and block.start_sample != 0:
                raise SpikeStreamError("New session did not begin at hardware sample zero; stale stream/epoch")
            if self._record_host_ns is None:
                raise SpikeStreamError("Unexpected marker data before Record; stale stream")
            if self._clock_origin is None:
                self._clock_origin = (block.start_sample, block.host_rx_ns)
            source_elapsed_ms = (block.end_sample - self._clock_origin[0]) * 1000 / self.rate
            host_elapsed_ms = (block.host_rx_ns - self._record_host_ns) / 1e6
            lag_ms = host_elapsed_ms - source_elapsed_ms
            self.metadata["max_observed_clock_lag_ms"] = max(self.metadata["max_observed_clock_lag_ms"], lag_ms)
            if lag_ms > self.timing["max_stream_age_ms"]:
                raise SpikeStreamError("Hardware clock fell behind host time beyond the configured lag bound")
            if time.monotonic_ns() - block.host_rx_ns > self.timing["max_stream_age_ms"] * 1e6:
                raise SpikeStreamError("Marker processing backlog exceeded configured bound")
            markers, edges = self.tracker.feed(block)
            self.markers.extend(markers)
            self.ttl_edges.extend(edges)
            self.progress = block.end_sample
            self.last_progress_ns = block.host_rx_ns
        new_events = self.spike_reader.drain()
        for event in new_events:
            if event.sample_index < self.closed_through:
                raise SpikeStreamError("Late spike arrived after a window was closed; session invalid")
            if time.monotonic_ns() - event.host_rx_ns > self.timing["max_stream_age_ms"] * 1e6:
                raise SpikeStreamError("Spike processing backlog exceeded configured bound")
        self.events.extend(new_events)
        if len(self.events) > 1000000 or len(self.markers) > 10000:
            raise SpikeStreamError("Unconsumed acquisition buffers exceeded session limits")

    def _until(self, condition):
        deadline = time.monotonic() + self.timeout_s
        while True:
            self._pump()
            if condition():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for hardware markers/window progress")
            self.cancel.wait(.002)

    def observe(self, protocol):
        if not protocol.pulses:
            raise ValueError("An empty stimulation protocol has no response anchor")
        if canonical(self.snapshot_neural()) != canonical(self.neural):
            raise ValueError("RHX spike filter/reference/threshold settings changed during session")
        self._pump()
        if self.markers:
            raise SpikeStreamError("Unassigned stimulation markers before next input")
        timing = self.client.play_prepared_protocol(protocol, self.cancel)
        self._until(lambda: len(self.markers) >= len(protocol.pulses))
        with self._state_lock:
            anchor = check_markers(self.markers, protocol, self.prefix, self.rate, self.timing["max_schedule_error_ms"])
        end = anchor + self.spec.samples(self.rate, "offset_ms") + self.spec.samples(self.rate, "duration_ms")
        if max(m.end_sample for m in self.markers) > end:
            raise SpikeStreamError("Observation window ends before the final stimulus")
        self._until(lambda: self.progress >= end)
        # This is a declared, experimentally bounded assumption, NOT a spike
        # heartbeat or proof from socket silence. Any later violation aborts.
        deadline = time.monotonic() + self.timing["max_cross_socket_delay_ms"] / 1000
        while time.monotonic() < deadline:
            self._pump()
            self.cancel.wait(.002)
        self._pump()
        with self._state_lock:
            check_markers(self.markers, protocol, self.prefix, self.rate, self.timing["max_schedule_error_ms"])
            self.closed_through = end
            observation = {"anchor": anchor, "markers": list(self.markers), "events": list(self.events),
                           "watermark": ProgressMarker(self.progress, self.epoch, "stim"),
                           "completeness": "bounded_cross_socket_delay_assumption", "host_trigger_timing": timing,
                           "ttl_edges": [t for t in self.ttl_edges if anchor <= t < end]}
            self.markers = []
            self.events = [e for e in self.events if e.sample_index >= end]
            self.ttl_edges = [t for t in self.ttl_edges if t >= end]
        if canonical(self.snapshot_neural()) != canonical(self.neural):
            raise ValueError("RHX spike settings changed during the response window")
        return observation

    def close(self):
        errors = []
        self._watch_stop.set()
        if self._watchdog is not None:
            self._watchdog.join(2)
        try:
            if self.started and self.client.is_command_connected:
                self.client.stop(verify=True)
                self.metadata["rhx_stop_confirmed"] = True
                deadline = time.monotonic() + self.timing["max_cross_socket_delay_ms"] / 1000 + .05
                while time.monotonic() < deadline:
                    self._pump()
                    time.sleep(.002)
            elif self.started:
                raise RuntimeError("Command connection lost; RHX Stop could not be confirmed")
        except Exception as exc:
            errors.append(exc)
        for reader in self.readers:
            try:
                reader.stop()
            except Exception as exc:
                errors.append(exc)
        try:
            if self.started:
                self._drain_once()
        except Exception as exc:
            errors.append(exc)
        self._collector_stop.set()
        if self._collector is not None:
            self._collector.join(2)
            if self._collector.is_alive():
                errors.append(TimeoutError("Continuous collector failed to stop"))
        self.client.disconnect()
        for stream in self.files:
            stream.close()
        if self.readers and (self.marker_parser.buffer or self.spike_reader.parser.buffer or self.tracker.active or self.markers):
            errors.append(SpikeStreamError("Recording ended with incomplete wire packet/pulse"))
        if errors:
            raise RuntimeError("RHX session cleanup/integrity failed: " + "; ".join(str(e) for e in errors))
