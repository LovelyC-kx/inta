"""
TCP Client for Intan RHX Software.

Implements the three TCP connection types documented in the Intan RHX manual:
1. Command: Send ASCII commands (Set/Get) to control all software parameters
2. Waveform Output: Receive streaming waveform data
3. Spike Output: Receive spike raster event data

Usage:
    1. Start Intan RHX in Demo or Hardware mode
    2. Open Network -> Remote TCP Control
    3. Set host/port for each connection type and click Connect (Pending)
    4. Run this client to establish connections

Reference: Intan RHX TCP Guide (available on Intan website)

VERIFIED waveform stream format (RHS Demo, 8ch WIDE, 30 kS/s):
    Each block starts with a 4-byte magic followed by 128 frames.
    Each subsequent frame:
        [4 bytes: timestamp, little-endian uint32]
        [n_channels * 2 bytes: samples, little-endian uint16, offset-binary]
    Frame size = 4 + 2 * n_channels bytes
    Timestamp increments by 1 per frame (= 1/sample_rate seconds)
    Microvolts conversion: uv = (uint16_value - 32768) * 0.195
"""

import socket
import struct
import threading
import time
import math
from numbers import Integral
from collections import deque
import numpy as np
from typing import Optional

from .command_channel import RHXCommandError, receive_fenced_reply, validate_command
from .hardware_profile import RHSProfile, rhx_bool


# ── Verified Intan waveform stream format constants ────────────
# Verified from RHS Demo 8ch WIDE 30 kS/s TCP capture + block boundary analysis:
#
# Stream = repeating blocks:
#   [BLOCK_HEADER_SIZE bytes: magic/marker (e.g., 08 7a f0 2e)]
#   [SAMPLES_PER_BLOCK frames, each: 4B timestamp(LE u32) + n_ch*2B(LE u16 offset-binary)]
#
# Block size = BLOCK_HEADER_SIZE + SAMPLES_PER_BLOCK * (4 + n_ch * 2)
# No separate "one-time stream header" — every block starts with the marker.
#
BLOCK_HEADER_SIZE = 4                # bytes; magic marker per block
SAMPLES_PER_BLOCK = 128              # frames per block (Intan default at 30 kS/s)
WAVEFORM_TIMESTAMP_BYTES = 4
WAVEFORM_SAMPLE_BYTES = 2            # uint16 per sample per channel
AMPLIFIER_UV_PER_BIT = 0.195
AMPLIFIER_OFFSET_BINARY_ZERO = 32768 # 0x8000
WAVEFORM_MAGIC = struct.pack("<I", 0x2ef07a08)


class WaveformStreamError(RuntimeError):
    """Waveform framing/continuity failed; do not decode this acquisition."""


class WaveformFrameReader:
    """
    Stateful block-aware parser for Intan RHX waveform TCP stream.

    Verified format (RHS Demo, 8ch WIDE, 30 kS/s):
        Stream = repeating blocks of:
            [4-byte block header (magic)]
            [128 frames, each: 4B timestamp(u32 LE) + n_ch × 2B sample(u16 LE)]

        Block total = 4 + 128 × (4 + n_ch × 2) bytes

    The parser:
        1. Accumulates raw TCP bytes
        2. Validates magic and timestamp continuity, then parses complete blocks
        3. Stores parsed frames in an internal buffer
        4. read_frames(n) returns n frames from the parsed buffer

    Usage:
        reader = WaveformFrameReader(n_channels=8)
        while True:
            chunk = sock.recv(65536)
            reader.add_data(chunk)
            timestamps, wave_uv = reader.read_frames(n_frames=3000)
    """

    def __init__(self, n_channels, block_size=SAMPLES_PER_BLOCK,
                 block_header_size=BLOCK_HEADER_SIZE):
        if isinstance(n_channels, bool) or not isinstance(n_channels, Integral) or n_channels < 1:
            raise ValueError("n_channels must be a positive integer")
        if block_size != SAMPLES_PER_BLOCK or block_header_size != BLOCK_HEADER_SIZE:
            raise ValueError("RHX TCP requires 128-frame blocks with a 4-byte magic")
        self.n_channels = n_channels
        self.frame_bytes = WAVEFORM_TIMESTAMP_BYTES + n_channels * WAVEFORM_SAMPLE_BYTES
        self.block_size = block_size
        self.block_header_size = block_header_size
        self.block_total_bytes = block_header_size + block_size * self.frame_bytes

        self._raw_buffer = bytearray()
        self._parsed_ts = []    # list of uint32 arrays
        self._parsed_uv = []    # list of (n_ch, n_frames) float32 arrays
        self._total_parsed = 0
        self._last_timestamp = None
        self._error = None

    def add_data(self, data):
        """Append raw bytes received from the waveform socket."""
        self._raise_if_failed()
        self._raw_buffer.extend(data)
        try:
            self._parse_available_blocks()
        except WaveformStreamError as exc:
            self._error = exc
            raise

    def _raise_if_failed(self):
        if self._error is not None:
            raise self._error

    def reset(self):
        """Reset all state. Call when reconnecting."""
        self._raw_buffer.clear()
        self._parsed_ts.clear()
        self._parsed_uv.clear()
        self._total_parsed = 0
        self._last_timestamp = None
        self._error = None

    def _parse_available_blocks(self):
        """Parse as many complete blocks as possible from the raw buffer."""
        while len(self._raw_buffer) >= self.block_total_bytes:
            if self._raw_buffer[:self.block_header_size] != WAVEFORM_MAGIC:
                raise WaveformStreamError("Invalid RHX waveform magic; check stream layout/alignment")
            # Validate every block; never skip arbitrary bytes as a header.
            del self._raw_buffer[:self.block_header_size]

            # Read one block of frames
            data_bytes = self.block_size * self.frame_bytes
            chunk = bytes(self._raw_buffer[:data_bytes])
            del self._raw_buffer[:data_bytes]

            arr = np.frombuffer(chunk, dtype=np.uint8).reshape(
                self.block_size, self.frame_bytes
            )
            timestamps = arr[:, :WAVEFORM_TIMESTAMP_BYTES].copy().view("<u4").reshape(-1)
            deltas = (timestamps[1:].astype(np.uint64) - timestamps[:-1].astype(np.uint64)) & 0xffffffff
            if (np.any(deltas != 1) or (self._last_timestamp is not None and
                    int(timestamps[0]) != ((self._last_timestamp + 1) & 0xffffffff))):
                raise WaveformStreamError("RHX waveform timestamp gap/reset; start a new acquisition")
            self._last_timestamp = int(timestamps[-1])
            wave_u16 = (
                arr[:, WAVEFORM_TIMESTAMP_BYTES:]
                .copy()
                .view("<u2")
                .reshape(self.block_size, self.n_channels)
            )
            wave_uv = (
                wave_u16.astype(np.int32) - AMPLIFIER_OFFSET_BINARY_ZERO
            ).astype(np.float32) * AMPLIFIER_UV_PER_BIT
            wave_uv = wave_uv.T  # (n_channels, block_size)

            self._parsed_ts.append(timestamps)
            self._parsed_uv.append(wave_uv)
            self._total_parsed += self.block_size

    @property
    def available_frames(self):
        """Number of parsed frames ready to read."""
        return sum(len(ts) for ts in self._parsed_ts)

    def read_frames(self, n_frames=None):
        """
        Return parsed frames.

        SHAPE CONTRACT (frozen):
            timestamps.shape == (n_returned,)        dtype=uint32
            wave_uv.shape    == (n_channels, n_returned)  dtype=float32, µV

        Args:
            n_frames: Max frames to return. None = all available.

        Returns:
            (timestamps, wave_uv) tuple. May return fewer than n_frames
            if not enough data has been received yet.
        """
        self._raise_if_failed()
        if n_frames is not None and (isinstance(n_frames, bool) or
                not isinstance(n_frames, Integral) or n_frames < 1):
            raise ValueError("n_frames must be a positive integer or None")
        if not self._parsed_ts:
            return (
                np.array([], dtype=np.uint32),
                np.zeros((self.n_channels, 0), dtype=np.float32),
            )

        all_ts = np.concatenate(self._parsed_ts)
        all_uv = np.concatenate(self._parsed_uv, axis=1)

        if n_frames is None or n_frames >= len(all_ts):
            self._parsed_ts.clear()
            self._parsed_uv.clear()
            return all_ts, all_uv

        ts_out = all_ts[:n_frames]
        uv_out = all_uv[:, :n_frames]
        ts_remain = all_ts[n_frames:]
        uv_remain = all_uv[:, n_frames:]

        if len(ts_remain) > 0:
            self._parsed_ts = [ts_remain]
            self._parsed_uv = [uv_remain]
        else:
            self._parsed_ts.clear()
            self._parsed_uv.clear()

        return ts_out, uv_out


class IntanTCPClient:
    """
    Client for Intan RHX TCP remote control.

    Three independent connections:
    - command_socket: For Set/Get ASCII commands
    - waveform_socket: For receiving streaming waveform data (uses WaveformFrameReader)
    - spike_socket: For receiving spike event data
    """

    def __init__(
        self,
        host="127.0.0.1",
        command_port=5000,
        waveform_port=5001,
        spike_port=5002,
        execution_mode="acquire_only",
    ):
        """
        Args:
            host: IP address of the Intan RHX host.
            command_port: TCP port for command interface.
            waveform_port: TCP port for waveform data output.
            spike_port: TCP port for spike event output.
        """
        self.host = host
        self.command_port = command_port
        self.waveform_port = waveform_port
        self.spike_port = spike_port
        if execution_mode not in ("simulate", "acquire_only", "live"):
            raise ValueError("execution_mode must be simulate, acquire_only, or live")
        self.execution_mode = execution_mode
        self._command_lock = threading.RLock()
        self._command_buffer = bytearray()
        self._prepared_profiles = {}
        self.prepared_session = None
        self._trigger_history = deque()

        self._command_socket: Optional[socket.socket] = None
        self._waveform_socket: Optional[socket.socket] = None
        self._spike_socket: Optional[socket.socket] = None
        self._waveform_reader: Optional[WaveformFrameReader] = None

    # ── Connection Management ──────────────────────────────────────

    def connect_command(self, timeout=5.0):
        """Establish command TCP connection."""
        if self._command_socket is not None:
            raise RuntimeError("Command connection already has an owner")
        self._command_socket = self._connect(self.command_port, timeout)
        self._command_buffer.clear()
        self._prepared_profiles.clear()
        self.prepared_session = None
        return self

    def connect_waveform(self, timeout=5.0):
        """Establish waveform data TCP connection."""
        if self._waveform_socket is not None:
            raise RuntimeError("Waveform connection already has an owner")
        self._waveform_socket = self._connect(self.waveform_port, timeout)
        self._waveform_reader = None
        return self

    def connect_spike(self, timeout=5.0):
        """Establish spike data TCP connection."""
        if self._spike_socket is not None:
            raise RuntimeError("Spike connection already has an owner")
        self._spike_socket = self._connect(self.spike_port, timeout)
        return self

    def connect_all(self, timeout=5.0):
        """Establish all three TCP connections."""
        self.connect_command(timeout)
        self.connect_waveform(timeout)
        self.connect_spike(timeout)
        return self

    def disconnect(self):
        """Close all TCP connections."""
        for sock in [self._command_socket, self._waveform_socket, self._spike_socket]:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._command_socket = None
        self._waveform_socket = None
        self._spike_socket = None
        self._command_buffer.clear()
        self._prepared_profiles.clear()
        self.prepared_session = None
        self._waveform_reader = None

    def _connect(self, port, timeout):
        """Create and connect a TCP socket."""
        if self.execution_mode == "simulate":
            raise RuntimeError("simulate never opens hardware sockets; use an offline/replay backend")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((self.host, port))
        except Exception:
            sock.close()
            raise
        return sock

    # ── Command Interface ──────────────────────────────────────────

    # Minimum gap between consecutive Set commands. Sending too fast causes
    # RHX to silently drop some commands AND blocks the next Get with a
    # backlog timeout. 50 ms empirically lets all params land while keeping
    # an 80-cmd batch under ~5 s. Do NOT set below 20 ms.
    INTER_COMMAND_DELAY_S = 0.05

    def send_command(self, command, expect_response=True, timeout=2.0, *, _allow_warnings=False):
        """
        Send an ASCII command and optionally read the response.

        Args:
            command: Command string (e.g., "Set RunMode Run").
            expect_response: Whether to wait for a response.
            timeout: Response timeout in seconds.

        Returns:
            Response string if expect_response, else None.

        Raises:
            ConnectionError: If command socket is not connected.
            TimeoutError: If no response received within timeout.
        """
        command = validate_command(command)
        with self._command_lock:
            self._guard_stimulation_command(command)
            if self._command_socket is None:
                raise ConnectionError("Command socket not connected. Call connect_command() first.")
            # RHX splits on semicolons. An ordered Get fences replies even
            # on versions which do not terminate responses with a newline.
            msg = command + (";Get RunMode;" if expect_response else ";")
            sock = self._command_socket
            previous_timeout = sock.gettimeout()
            try:
                sock.sendall(msg.encode("ascii"))
                if not expect_response:
                    return None
                return receive_fenced_reply(
                    sock, self._command_buffer, timeout=timeout,
                    run_mode_query=command.lower() == "get runmode",
                    allow_warnings=_allow_warnings,
                )
            except RHXCommandError:
                self._prepared_profiles.clear()
                self.prepared_session = None
                raise
            except (OSError, TimeoutError, ConnectionError):
                # A late reply must never satisfy a later request.
                sock.close()
                self._command_socket = None
                self._prepared_profiles.clear()
                self.prepared_session = None
                self._command_buffer.clear()
                raise
            finally:
                if self._command_socket is sock:
                    sock.settimeout(previous_timeout)

    def _guard_stimulation_command(self, command):
        words = command.lower().split()
        action = words[1] if len(words) > 1 else ""
        stim_set = (words[0] == "set" and "." in action and
                    action.split(".")[-1] in self.STIM_PARAMETER_NAMES)
        stim_execute = words[0] == "execute" and action.startswith((
            "manualstimtrigger", "uploadstim", "uploadampsettle", "uploadchargerecovery",
            "loadsettingsfile", "loadstimulationsettingsfile"))
        global_stim_set = words[0] == "set" and action in self.GLOBAL_STIM_PARAMETERS
        if (stim_set or stim_execute or global_stim_set) and self.execution_mode != "live":
            raise PermissionError(f"{self.execution_mode} does not send stimulation commands")
        if stim_set or global_stim_set or (words[0] == "execute" and action in {
                "uploadstimparameters", "uploadampsettlesettings", "uploadchargerecoverysettings",
                "loadsettingsfile", "loadstimulationsettingsfile", "rescanports"}):
            self._prepared_profiles.clear()
            self.prepared_session = None
        if words[0] == "execute" and action == "manualstimtriggeron":
            raise RuntimeError("Latched stimulation is unsupported; use verified pulse scheduling")
        if words[0] == "execute" and action == "manualstimtriggerpulse":
            if len(words) != 3 or words[2] not in self._prepared_profiles:
                raise RuntimeError("Trigger has no verified uploaded profile; prepare it while stopped")
            # One budget covers sensory, feedback and direct manual pulses.
            from encoders.stimulus_schema import (MIN_INTER_PULSE_MS,
                                                 SAFETY_MAX_TOTAL_CHARGE_PER_TRIAL_NC)
            now = time.monotonic()
            while self._trigger_history and self._trigger_history[0][0] <= now - 2.0:
                self._trigger_history.popleft()
            key = words[2]
            if any(k == key and now - t < MIN_INTER_PULSE_MS / 1000.0
                   for t, k, _ in self._trigger_history):
                raise RuntimeError("Trigger would violate the session inter-pulse interval")
            amp, phase, *_ = self._prepared_profiles[key]
            charge = 2.0 * amp * phase / 1000.0
            if sum(q for _, _, q in self._trigger_history) + charge > SAFETY_MAX_TOTAL_CHARGE_PER_TRIAL_NC:
                raise RuntimeError("Trigger would exceed the rolling 2-second charge budget")
            # Reserve on attempt; failed writes are not automatically retried.
            self._trigger_history.append((now, key, charge))

    STIM_PARAMETER_NAMES = frozenset({
        "stimenabled", "source", "shape", "polarity", "pulseortrain",
        "triggeredgeorlevel", "triggerhighorlow", "posttriggerdelaymicroseconds",
        "refractoryperiodmicroseconds", "firstphasedurationmicroseconds",
        "secondphasedurationmicroseconds", "interphasedelaymicroseconds",
        "firstphaseamplitudemicroamps", "secondphaseamplitudemicroamps",
        "numberofstimpulses", "pulsetrainperiodmicroseconds", "enableampsettle",
        "maintainampsettle", "enablechargerecovery", "prestimampsettlemicroseconds",
        "poststimampsettlemicroseconds", "poststimchargerecovonmicroseconds",
        "poststimchargerecovoffmicroseconds",
        "firstphaseamplitudevolts", "secondphaseamplitudevolts", "baselinevoltagevolts",
    })

    GLOBAL_STIM_PARAMETERS = frozenset({
        "usefastsettle", "headstageglobalsettle", "desiredlowersettlebandwidthhertz",
        "chargerecovery", "chargerecoverycurrentlimitmicroamps", "chargerecoverytargetvoltagevolts",
    })

    def _get_value(self, parameter):
        # RHX returns a bare field name for channel/port StateItems, e.g.
        # 'Return: StimEnabled True', not 'Return: A-000.StimEnabled True'.
        response = self.get(parameter)
        words = response.split(maxsplit=2)
        if len(words) != 3 or words[0] != "Return:":
            raise RHXCommandError(f"Malformed RHX response for {parameter}: {response}")
        return words[2]

    def get(self, parameter, timeout=2.0):
        """
        Query a parameter value.

        Args:
            parameter: Parameter name (e.g., "RunMode", "SampleRateHertz").

        Returns:
            Response string (e.g., "Return: RunMode Stop").
        """
        return self.send_command(f"Get {parameter}", timeout=timeout)

    def set(self, parameter, value, expect_response=False, verify=False):
        """
        Set a parameter value.

        Args:
            parameter: Parameter name (e.g., "RunMode").
            value: Value to set (e.g., "Run", "Stop").
            expect_response: If True, wait for and return Set response.
                             VERIFIED: many Set commands do NOT echo a response.
                             Default False to avoid blocking on recv().
            verify: If True, send Get after Set and return that response.

        Returns:
            None (default), Set response, or Get response (if verify=True).
        """
        self.send_command(f"Set {parameter} {value}", expect_response=expect_response)
        if verify:
            return self.get(parameter)
        return None

    def wait_for_run_mode(self, mode, timeout=10.0):
        """Poll confirmed state; never substitute the requested state on error."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.get("RunMode", timeout=max(.01, min(2.0, deadline - time.monotonic())))
            if response == f"Return: RunMode {mode}":
                return response
            time.sleep(0.05)
        raise TimeoutError(f"RHX did not enter {mode}")

    def wait_for_upload(self, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.get("UploadInProgress", timeout=max(.01, deadline - time.monotonic()))
            if not rhx_bool(response.split()[-1], "UploadInProgress"):
                return
            time.sleep(0.05)
        raise TimeoutError("RHX stimulation upload did not finish")

    # ── Convenience Commands ───────────────────────────────────────

    def run(self, verify=False):
        """Start Run from Stop after uploads; repeated Run is a no-op."""
        return self._transition_run_mode("Run", verify)

    def stop(self, verify=False):
        """Stop if running; RHX rejects a raw Stop when already stopped."""
        return self._transition_run_mode("Stop", verify)

    def record(self, verify=False):
        """Start Record from Stop after uploads; do not silently restart Run."""
        return self._transition_run_mode("Record", verify)

    def _transition_run_mode(self, mode, verify):
        with self._command_lock:
            current = self.get("RunMode")
            if current == f"Return: RunMode {mode}":
                return current if verify else None
            if mode != "Stop":
                if current != "Return: RunMode Stop":
                    raise RuntimeError(f"Stop RHX before changing acquisition mode to {mode}")
                self.wait_for_upload()
            self.set("RunMode", mode)
            return self.wait_for_run_mode(mode) if verify else None

    def get_sample_rate(self):
        """Query current sample rate in Hz."""
        response = self.get("SampleRateHertz")
        # Expected: "Return: SampleRateHertz 30000"
        try:
            return int(response.split()[-1])
        except (ValueError, IndexError):
            return response

    def set_filename(self, path, base_filename):
        """Set recording file path and base filename."""
        r1 = self.set("Filename.Path", path)
        r2 = self.set("Filename.BaseFilename", base_filename)
        return r1, r2

    # ── Stimulation Commands (RHS only) ────────────────────────────

    def configure_stim_channel(self, channel, params):
        """
        Configure stimulation parameters for a single channel.

        Args:
            channel: Channel name (e.g., "A-000").
            params: Dict of parameter names and values. Example:
                {
                    "stimenabled": "true",
                    "source": "keypressf1",
                    "shape": "Biphasic",
                    "polarity": "NegativeFirst",
                    "firstphasedurationmicroseconds": "500",
                    "secondphasedurationmicroseconds": "500",
                    "firstphaseamplitudemicroamps": "10",
                }

        Returns:
            Dict mapping parameter names to (set_response, get_response).
        """
        results = {}
        for param, value in params.items():
            set_resp = self.set(f"{channel}.{param}", value)
            get_resp = self.get(f"{channel}.{param}")
            results[param] = (set_resp, get_resp)
        return results

    # ── StimulusProtocol -> Intan RHS commands ───────────────────────

    def protocol_to_commands(self, protocol, channel_prefix="A"):
        """
        Convert a StimulusProtocol into Intan RHS TCP command strings.

        VERIFIED: Intan RHS uses CURRENT-mode stimulation (microamps).
        amplitude_ua is mapped to firstphaseamplitudemicroamps and
        secondphaseamplitudemicroamps for biphasic charge balance.

        Limitations:
            - PostTriggerDelay capped at 500 ms by RHS hardware.
              Pulses beyond 500 ms after a trigger require separate triggers.
            - Each channel uses one parameter set (first pulse's params).
              Schema currently enforces uniform amplitude/duration per channel.

        Returns:
            List of (command_string, description) tuples.
        """
        protocol.validate()
        if channel_prefix not in ("A", "B", "C", "D"):
            raise ValueError("RHS channel prefix must be A, B, C, or D")
        commands = []
        by_channel = {}
        for pulse in protocol.pulses:
            signature = self._pulse_signature(pulse)
            if pulse.channel in by_channel and by_channel[pulse.channel] != signature:
                raise ValueError("A channel cannot change pulse parameters inside a prepared session")
            by_channel[pulse.channel] = signature
        for ch in range(protocol.n_channels):
            name = f"{channel_prefix}-{ch:03d}"
            if ch not in by_channel:
                commands.append((f"Set {name}.stimenabled false", "Disable unused channel"))
                continue
            amp, phase, delay, polarity, waveform = by_channel[ch]
            params = {
                "stimenabled": "true", "source": f"KeyPressF{ch + 1}",
                "shape": waveform, "polarity": polarity,
                "firstphasedurationmicroseconds": f"{phase:g}",
                "secondphasedurationmicroseconds": f"{phase:g}",
                "interphasedelaymicroseconds": f"{delay:g}",
                "firstphaseamplitudemicroamps": f"{amp:g}",
                "secondphaseamplitudemicroamps": f"{amp:g}",
                "pulseortrain": "SinglePulse", "numberofstimpulses": "1",
                "triggeredgeorlevel": "Edge", "triggerhighorlow": "High",
                "posttriggerdelaymicroseconds": "0",
                "enableampsettle": "true", "maintainampsettle": "false",
                "enablechargerecovery": "false",
                "prestimampsettlemicroseconds": "0",
                "poststimampsettlemicroseconds": "1000",
                "poststimchargerecovonmicroseconds": "0",
                "poststimchargerecovoffmicroseconds": "0",
                "refractoryperiodmicroseconds": "1000",
                "pulsetrainperiodmicroseconds": "1000000",
            }
            commands.extend((f"Set {name}.{k} {v}", k) for k, v in params.items())
        return commands

    @staticmethod
    def _pulse_signature(pulse):
        return (pulse.amplitude_ua, pulse.phase_duration_us, pulse.interphase_delay_us,
                pulse.polarity, pulse.waveform)

    def send_protocol_config(self, protocol, channel_prefix="A",
                             skip_safety_check=False, upload_to_hardware=False):
        """Stage parameters while stopped; optionally upload and verify them.

        No upload is not a dry-run: old parameters remain in the controller.
        Non-live clients validate the protocol but send no stimulation commands.
        Use prepare_protocols() once before continuous acquisition.
        """
        protocol.validate_safety()
        if skip_safety_check:
            raise ValueError("Skipping stimulation validation is no longer supported")
        return self._upload_protocol_profile(protocol, channel_prefix, upload_to_hardware)

    def _upload_protocol_profile(self, protocol, channel_prefix, upload_to_hardware):
        """Compile profiles only; callers validate each actually executed trial."""
        commands = self.protocol_to_commands(protocol, channel_prefix)
        if self.execution_mode != "live":
            return []
        with self._command_lock:
            if self.get("RunMode") != "Return: RunMode Stop":
                raise RuntimeError("Stop RHX before configuring/uploading stimulation")
            self.wait_for_upload()
            self._prepared_profiles.clear()
            self.prepared_session = None
            hardware = RHSProfile.read(self._get_value)
            pulse_codes = {f"{channel_prefix}-{p.channel:03d}": hardware.validate_pulse(p)
                           for p in protocol.pulses}
            inventory = {}
            for port in "ABCD":
                count = int(self._get_value(f"{port}.NumberAmplifierChannels"))
                if not 0 <= count <= 32:
                    raise RuntimeError("Expected an RHS controller with at most 32 channels per port")
                inventory[port] = count
            if protocol.n_channels > inventory[channel_prefix]:
                raise RuntimeError("Requested stimulation channels are absent from the RHS inventory")
            global_parameters = (
                "UseFastSettle", "HeadstageGlobalSettle", "DesiredLowerSettleBandwidthHertz",
                "ActualLowerSettleBandwidthHertz", "ChargeRecovery",
                "ChargeRecoveryCurrentLimitMicroAmps", "ChargeRecoveryTargetVoltageVolts")
            global_settings = {p: self._get_value(p) for p in global_parameters}
            # Disable all amplifier channels first: F1..F8 triggers are global,
            # and affect analog/digital outputs as well as other headstages.
            disabled_channels = []
            for port, count in inventory.items():
                for ch in range(count):
                    disabled_channels.append(f"{port}-{ch:03d}")
            for port, count in (("ANALOG-OUT", 8 if hardware.expander_connected else 2),
                                ("DIGITAL-OUT", 16 if hardware.expander_connected else 2)):
                disabled_channels.extend(f"{port}-{ch}" for ch in range(1, count + 1))
            for name in disabled_channels:
                self.set(f"{name}.stimenabled", "false")
                time.sleep(self.INTER_COMMAND_DELAY_S)
            results = []
            for cmd, _ in commands:
                self.send_command(cmd, expect_response=False)
                results.append((cmd, None))
                time.sleep(self.INTER_COMMAND_DELAY_S)
            # RHX warns after EACH dependency-related Set, even while a stopped
            # profile is still being assembled. Preserve these warnings; hard
            # errors still abort. The final complete profile is read back below.
            staged = self.send_command("Get RunMode", _allow_warnings=True)
            warning_text = staged[:staged.rfind("Return:")].strip()
            if not upload_to_hardware:
                return results
            # Verify software profile, including mode/delay fields that used to
            # depend silently on whatever was last selected in the GUI.
            for cmd, _ in commands:
                _, param, expected = cmd.split()
                actual = self.get(param).split()[-1]
                try:
                    matches = float(actual) == float(expected)
                except ValueError:
                    matches = actual.casefold() == expected.casefold()
                if not matches:
                    raise RuntimeError(f"RHX profile mismatch: {param}={actual}, expected {expected}")
            active_names = {f"{channel_prefix}-{p.channel:03d}" for p in protocol.pulses}
            for name in disabled_channels:
                expected = "true" if name in active_names else "false"
                if self.get(f"{name}.stimenabled").split()[-1].lower() != expected:
                    raise RuntimeError(f"RHX profile mismatch: {name}.stimenabled")
            if active_names - set(disabled_channels):
                raise RuntimeError("A requested stimulation channel is absent from the RHS inventory")
            # Only a complete, verified software profile may be uploaded.
            # Apply the selected GLOBAL settle mode too; channel StimParameters
            # upload alone does not apply these chip-level settings (TCP p.60).
            self.send_command("Execute UploadAmpSettleSettings", expect_response=False)
            self.wait_for_upload()
            global_settings = {p: self._get_value(p) for p in global_parameters}
            self.send_command("Execute UploadStimParameters", expect_response=False)
            self.wait_for_upload()
            for pulse in protocol.pulses:
                self._prepared_profiles[f"f{pulse.channel + 1}"] = self._pulse_signature(pulse)
            self.prepared_session = {
                "hardware": hardware.as_dict(), "pulse_codes": pulse_codes,
                "disabled_output_names": [n for n in disabled_channels if n not in active_names],
                "global_stimulation_settings": global_settings,
                "configuration_warnings": warning_text,
                "physical_delivery_verified": False,
            }
            results.extend((("Execute UploadAmpSettleSettings", None),
                            ("Execute UploadStimParameters", None)))
            return results

    def prepare_protocols(self, protocols, channel_prefix="A"):
        """Upload the union of fixed channel profiles once, before Run/Record.

        Trial timing can vary freely. Per-channel current/shape must be constant
        across all sensory and feedback protocols within this session.
        """
        from encoders.stimulus_schema import Pulse, StimulusProtocol
        protocols = list(protocols)
        if not protocols:
            raise ValueError("At least one protocol is required")
        profiles = {}
        n_channels = max(p.n_channels for p in protocols)
        for protocol in protocols:
            protocol.validate_safety()
            for pulse in protocol.pulses:
                signature = self._pulse_signature(pulse)
                if pulse.channel in profiles and profiles[pulse.channel] != signature:
                    raise ValueError("All protocols must use the same pulse profile per channel")
                profiles[pulse.channel] = signature
        pulses = [Pulse(ch, 0.0, *sig) for ch, sig in sorted(profiles.items())]
        union = StimulusProtocol(pulses, n_channels, 2000.0)
        # The union is a configuration table, not an executed trial. Each
        # actual protocol was validated above and is checked again at playback.
        return self._upload_protocol_profile(union, channel_prefix, True)

    def play_prepared_protocol(self, protocol, cancel_event=None):
        """Trigger every event using an already uploaded profile.

        Times below measure host send scheduling, NOT electrode current onset.
        Hardware marker measurements are required for electrical timing claims.
        """
        protocol.validate_safety()
        for pulse in protocol.pulses:
            if self._prepared_profiles.get(f"f{pulse.channel + 1}") != self._pulse_signature(pulse):
                raise RuntimeError("Protocol does not match the verified uploaded profile")
        if self.execution_mode != "live":
            raise PermissionError("Only live mode can execute prepared stimulation")
        if self.get("RunMode") not in ("Return: RunMode Run", "Return: RunMode Record"):
            raise RuntimeError("RHX must be running before triggering a prepared protocol")
        cancel_event = cancel_event or threading.Event()
        t0 = time.monotonic()
        timing = []
        for pulse in protocol.pulses:
            from encoders.stimulus_schema import MIN_INTER_PULSE_MS
            target = t0 + pulse.time_ms / 1000.0
            key = f"f{pulse.channel + 1}"
            target = max([target] + [t + MIN_INTER_PULSE_MS / 1000.0
                                    for t, k, _ in self._trigger_history if k == key])
            while True:
                if cancel_event.is_set():
                    raise InterruptedError("Stimulation cancelled before next trigger")
                remaining = target - time.monotonic()
                if remaining <= 0:
                    break
                cancel_event.wait(min(remaining, 0.01))
            sent_ms = (time.monotonic() - t0) * 1000.0
            self.send_command(f"Execute ManualStimTriggerPulse F{pulse.channel + 1}",
                              expect_response=False)
            timing.append({"target_ms": float(pulse.time_ms), "actual_ms": sent_ms,
                           "drift_ms": sent_ms - pulse.time_ms, "channel": int(pulse.channel),
                           "host_send_only": True, "hardware_sample": None})
        return timing

    def play_protocol_in_time(self, protocol, channel_prefix="A",
                              skip_safety_check=False, upload_to_hardware=False):
        """Compatibility entry point; default is strictly no-output.

        For continuous acquisition use prepare_protocols(), start Run/Record,
        then play_prepared_protocol(). This one-shot live path starts Run after
        preparing while stopped. No electrical timing precision is guaranteed.
        """
        protocol.validate_safety()
        if skip_safety_check:
            raise ValueError("Skipping stimulation validation is no longer supported")
        if not upload_to_hardware:
            return [{"target_ms": float(p.time_ms), "actual_ms": None, "drift_ms": None,
                     "channel": int(p.channel), "sent": False} for p in protocol.pulses]
        if self.execution_mode != "live":
            raise PermissionError("Set execution_mode='live' to upload and trigger")
        self.send_protocol_config(protocol, channel_prefix, upload_to_hardware=True)
        self.run()
        self.wait_for_run_mode("Run")
        return self.play_prepared_protocol(protocol)

    # ── Waveform Data Reception ────────────────────────────────────

    def read_waveform_data(self, n_bytes):
        """
        Read raw bytes from waveform output socket.

        Raw RHX block bytes, including magic and timestamps. Payload types and
        scaling depend on enabled signal types; they are not signed int16 µV.

        Args:
            n_bytes: Number of bytes to read.

        Returns:
            Raw bytes.
        """
        if self._waveform_socket is None:
            raise ConnectionError("Waveform socket not connected.")

        data = b""
        while len(data) < n_bytes:
            chunk = self._waveform_socket.recv(min(n_bytes - len(data), 65536))
            if not chunk:
                raise ConnectionError("Waveform socket closed before requested bytes arrived")
            data += chunk
        return data

    def init_waveform_reader(self, n_channels):
        """
        Initialize or reset the waveform frame reader.

        Call this once after connect_waveform() and before reading data.
        This reader supports amplifier voltage words (WIDE/LOW/HIGH) only.
        Pass the exact number of enabled voltage words in native stream order.
        DC/STIM/TTL/auxiliary/board analog signals require a typed layout parser;
        they cannot be converted using the amplifier's 0.195 µV/bit scale.
        """
        if self._waveform_reader is not None:
            raise RuntimeError("Reconnect before resetting an active waveform parser")
        self._waveform_reader = WaveformFrameReader(n_channels)
        return self._waveform_reader

    def read_waveform_samples(self, n_samples, n_channels, timeout=10.0):
        """
        Read exactly n_samples worth of waveform frames.

        SHAPE CONTRACT (frozen):
            Returns (timestamps, wave_uv) where
                timestamps.shape == (n_samples,)         dtype=uint32
                wave_uv.shape    == (n_channels, n_samples)  dtype=float32, units=µV

            Channels-first ordering. Time is the second axis.
            DO NOT transpose. All downstream code (spike detection, feature
            extraction, decoder input) assumes this layout.

        Implements the verified Intan RHX waveform stream format:
            repeated blocks: 4B magic + 128 frames of [timestamp(4B) + samples(2B*n_ch)]
            Samples are uint16 offset-binary, converted to microvolts.

        Across multiple calls, timestamps stay continuous: the last
        timestamp of one call equals (first timestamp of next call - 1).
        Use that property to detect dropped frames.

        Args:
            n_samples: Number of time samples (frames) to return.
            n_channels: Number of channels in the stream (must match Intan TCP config).
            timeout: Total read deadline in seconds, including partial blocks.

        Returns:
            timestamps: (n_samples,) uint32 array.
            wave_uv: (n_channels, n_samples) float32 array in microvolts.

        Raises:
            ConnectionError: If waveform socket is not connected.
            TimeoutError: If not enough data arrives within the timeout.
        """
        if self._waveform_socket is None:
            raise ConnectionError("Waveform socket not connected.")

        if (isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1):
            raise ValueError("n_samples must be a positive integer")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if self._waveform_reader is not None and self._waveform_reader.n_channels != n_channels:
            raise ValueError("Do not change waveform layout on an active stream; reconnect first")
        if self._waveform_reader is None:
            self._waveform_reader = WaveformFrameReader(n_channels)
        sock = self._waveform_socket
        previous_timeout = sock.gettimeout()
        deadline = time.monotonic() + timeout
        try:
            ts, wave = self._waveform_reader.read_frames(n_samples)
            while ts.shape[0] < n_samples:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Waveform read exceeded its total deadline")
                sock.settimeout(remaining)
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("Waveform socket closed before a complete sample window")
                self._waveform_reader.add_data(chunk)
                new_ts, new_wave = self._waveform_reader.read_frames(n_samples - ts.shape[0])
                if new_ts.shape[0] > 0:
                    ts = np.concatenate([ts, new_ts])
                    wave = np.concatenate([wave, new_wave], axis=1)
            return ts, wave
        except (OSError, TimeoutError, WaveformStreamError):
            # Partial windows cannot be silently reused after a failed read.
            sock.close()
            self._waveform_socket = None
            self._waveform_reader = None
            raise
        finally:
            if self._waveform_socket is sock:
                sock.settimeout(previous_timeout)

    def read_waveform_samples_uv_only(self, n_samples, n_channels, timeout=10.0):
        """Convenience: same as read_waveform_samples but returns only the µV array."""
        _, wave_uv = self.read_waveform_samples(n_samples, n_channels, timeout=timeout)
        return wave_uv

    # ── Spike Data Reception ───────────────────────────────────────

    def read_spike_data(self, n_bytes):
        """
        Read raw bytes from spike output socket.

        Spike data format: timestamped events with channel ID.

        Args:
            n_bytes: Number of bytes to read.

        Returns:
            Raw bytes.
        """
        if self._spike_socket is None:
            raise ConnectionError("Spike socket not connected.")

        data = b""
        while len(data) < n_bytes:
            chunk = self._spike_socket.recv(min(n_bytes - len(data), 65536))
            if not chunk:
                raise ConnectionError("Spike socket closed before requested bytes arrived")
            data += chunk
        return data

    # ── Context Manager ────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    @property
    def is_command_connected(self):
        return self._command_socket is not None

    @property
    def is_waveform_connected(self):
        return self._waveform_socket is not None

    @property
    def is_spike_connected(self):
        return self._spike_socket is not None
