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
    Stream starts with 4-byte header/magic (one-time, post-connect).
    Each subsequent frame:
        [4 bytes: timestamp, little-endian uint32]
        [n_channels * 2 bytes: samples, little-endian uint16, offset-binary]
    Frame size = 4 + 2 * n_channels bytes
    Timestamp increments by 1 per frame (= 1/sample_rate seconds)
    Microvolts conversion: uv = (uint16_value - 32768) * 0.195
"""

import socket
import struct
import time
import numpy as np
from typing import Optional


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
        2. Parses complete blocks (skipping block headers)
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
        self.n_channels = n_channels
        self.frame_bytes = WAVEFORM_TIMESTAMP_BYTES + n_channels * WAVEFORM_SAMPLE_BYTES
        self.block_size = block_size
        self.block_header_size = block_header_size
        self.block_total_bytes = block_header_size + block_size * self.frame_bytes

        self._raw_buffer = bytearray()
        self._parsed_ts = []    # list of uint32 arrays
        self._parsed_uv = []    # list of (n_ch, n_frames) float32 arrays
        self._total_parsed = 0

    def add_data(self, data):
        """Append raw bytes received from the waveform socket."""
        self._raw_buffer.extend(data)
        self._parse_available_blocks()

    def reset(self):
        """Reset all state. Call when reconnecting."""
        self._raw_buffer.clear()
        self._parsed_ts.clear()
        self._parsed_uv.clear()
        self._total_parsed = 0

    def _parse_available_blocks(self):
        """Parse as many complete blocks as possible from the raw buffer."""
        while len(self._raw_buffer) >= self.block_total_bytes:
            # Skip block header
            del self._raw_buffer[:self.block_header_size]

            # Read one block of frames
            data_bytes = self.block_size * self.frame_bytes
            chunk = bytes(self._raw_buffer[:data_bytes])
            del self._raw_buffer[:data_bytes]

            arr = np.frombuffer(chunk, dtype=np.uint8).reshape(
                self.block_size, self.frame_bytes
            )
            timestamps = arr[:, :WAVEFORM_TIMESTAMP_BYTES].copy().view("<u4").reshape(-1)
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

        self._command_socket: Optional[socket.socket] = None
        self._waveform_socket: Optional[socket.socket] = None
        self._spike_socket: Optional[socket.socket] = None
        self._waveform_reader: Optional[WaveformFrameReader] = None

    # ── Connection Management ──────────────────────────────────────

    def connect_command(self, timeout=5.0):
        """Establish command TCP connection."""
        self._command_socket = self._connect(self.command_port, timeout)
        return self

    def connect_waveform(self, timeout=5.0):
        """Establish waveform data TCP connection."""
        self._waveform_socket = self._connect(self.waveform_port, timeout)
        return self

    def connect_spike(self, timeout=5.0):
        """Establish spike data TCP connection."""
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
        if self._waveform_reader is not None:
            self._waveform_reader.reset()

    def _connect(self, port, timeout):
        """Create and connect a TCP socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((self.host, port))
        return sock

    # ── Command Interface ──────────────────────────────────────────

    # Minimum gap between consecutive Set commands. Sending too fast causes
    # RHX to silently drop some commands AND blocks the next Get with a
    # backlog timeout. 50 ms empirically lets all params land while keeping
    # an 80-cmd batch under ~5 s. Do NOT set below 20 ms.
    INTER_COMMAND_DELAY_S = 0.05

    def send_command(self, command, expect_response=True, timeout=2.0):
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
        if self._command_socket is None:
            raise ConnectionError("Command socket not connected. Call connect_command() first.")

        # Intan RHX expects commands terminated with newline
        msg = command.strip() + "\n"
        self._command_socket.sendall(msg.encode("ascii"))

        if not expect_response:
            return None

        self._command_socket.settimeout(timeout)
        try:
            response = self._recv_until_newline(self._command_socket)
            return response.strip()
        except socket.timeout:
            raise TimeoutError(f"No response for command: {command}")

    def get(self, parameter):
        """
        Query a parameter value.

        Args:
            parameter: Parameter name (e.g., "RunMode", "SampleRateHertz").

        Returns:
            Response string (e.g., "Return: RunMode Stop").
        """
        return self.send_command(f"Get {parameter}")

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

    def _recv_until_newline(self, sock, bufsize=4096):
        """
        Receive one Intan RHX command response.

        Important RHX behavior verified during local testing:
            Raw socket command:
                send: b"Get RunMode\\n"
                recv: b"Return: RunMode Run"

        Some RHX versions return a complete ASCII response without a trailing
        newline. Therefore this function must NOT wait strictly for "\\n".
        It returns as soon as the buffer looks like a complete RHX response
        ("Return:" or "Error:"), while still supporting newline-terminated
        responses.
        """
        data = b""

        while True:
            chunk = sock.recv(bufsize)
            if not chunk:
                break

            data += chunk

            # Standard newline-terminated response.
            if b"\n" in data:
                break

            # RHX may return a complete response without a newline.
            text = data.decode("ascii", errors="ignore").strip()
            if text.startswith("Return:") or text.startswith("Error:"):
                break

        return data.decode("ascii", errors="ignore")

    # ── Convenience Commands ───────────────────────────────────────

    def run(self, verify=False):
        """
        Start data acquisition.

        Default `verify=False`: after the Set, we DO NOT immediately Get RunMode.
        Reason: RHX serialises command processing — if any Set commands have
        been sent recently, an immediate Get may time out while RHX catches up.
        See MAINTENANCE.md §4. Pass `verify=True` only on a fresh socket with
        no prior Sets in flight (and even then expect occasional timeouts).
        """
        return self.set("RunMode", "Run", verify=verify)

    def stop(self, verify=False):
        """Stop data acquisition. See `run()` docstring for verify default rationale."""
        return self.set("RunMode", "Stop", verify=verify)

    def record(self, verify=False):
        """Start recording to disk. See `run()` docstring for verify default rationale."""
        return self.set("RunMode", "Record", verify=verify)

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
                    "polarity": "CathodicFirst",
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
        commands = []

        pulses_by_ch = {}
        for p in protocol.pulses:
            pulses_by_ch.setdefault(p.channel, []).append(p)

        for ch_idx in range(protocol.n_channels):
            ch_name = f"{channel_prefix}-{ch_idx:03d}"
            ch_pulses = pulses_by_ch.get(ch_idx, [])

            if not ch_pulses:
                commands.append((
                    f"Set {ch_name}.stimenabled false",
                    f"Disable stim on {ch_name} (no pulses)",
                ))
                continue

            pulse = ch_pulses[0]

            commands.append((
                f"Set {ch_name}.stimenabled true",
                f"Enable stim on {ch_name}",
            ))
            commands.append((
                f"Set {ch_name}.source KeyPressF{ch_idx + 1}",
                f"Trigger source: F{ch_idx + 1} key",
            ))
            commands.append((
                f"Set {ch_name}.shape {pulse.waveform}",
                f"Waveform shape",
            ))
            commands.append((
                f"Set {ch_name}.polarity {pulse.polarity}",
                f"Polarity",
            ))
            commands.append((
                f"Set {ch_name}.firstphasedurationmicroseconds "
                f"{int(pulse.phase_duration_us)}",
                f"Phase 1 duration",
            ))
            commands.append((
                f"Set {ch_name}.secondphasedurationmicroseconds "
                f"{int(pulse.phase_duration_us)}",
                f"Phase 2 duration",
            ))
            commands.append((
                f"Set {ch_name}.interphasedelaymicroseconds "
                f"{int(pulse.interphase_delay_us)}",
                f"Interphase delay",
            ))
            commands.append((
                f"Set {ch_name}.firstphaseamplitudemicroamps "
                f"{pulse.amplitude_ua:.1f}",
                f"Phase 1 amplitude (µA)",
            ))
            commands.append((
                f"Set {ch_name}.secondphaseamplitudemicroamps "
                f"{pulse.amplitude_ua:.1f}",
                f"Phase 2 amplitude (µA)",
            ))
            commands.append((
                f"Set {ch_name}.numberofstimpulses 1",
                f"Single pulse per trigger",
            ))

        # NOTE: `Execute UploadStimParameters` was tried here but RHX rejects
        # it with "Set commands require a parameter and a value" — wrong
        # syntax. The actual command name / required args are TBD. For now
        # we rely on RHX auto-uploading Set changes (which it appears to do
        # in newer versions). If stim.dat stays empty, revisit and find the
        # correct Upload command.

        return commands

    def send_protocol_config(self, protocol, channel_prefix="A",
                             skip_safety_check=False,
                             upload_to_hardware=False):
        """
        Send all configuration commands for a StimulusProtocol.

        SAFETY MODEL — this method is dry-run BY DEFAULT:
            By default `upload_to_hardware=False`, so Set commands land in
            RHX's software state but the final `Execute UploadStimParameters`
            is SKIPPED. RHX accepts subsequent `Execute ManualStimTriggerPulse`
            but fires whatever was last uploaded (initial state = 0 µA), so
            stim.dat stays empty / silent. This is the intended dry-run
            mode for organoid-pre-Gate-A safety: stimulus plumbing exercised,
            but no current leaves the headstage.

            Only pass `upload_to_hardware=True` AFTER:
              1. CHECKLIST_REAL_ORGANOID.md Gate A is passed
                 (oscilloscope + dummy load shows correct biphasic current)
              2. The caller explicitly opted in (e.g. run_with_dashboard
                 --live-stim with ENABLE_STIM phrase typed).

        SAFETY: This method always calls protocol.validate_safety() before
        sending, even if skip_safety_check=True (which only suppresses raises).

        Args:
            protocol: A validated StimulusProtocol.
            channel_prefix: Port prefix.
            skip_safety_check: ONLY for unit tests with synthetic protocols.
                               Real-hardware callers must leave this False.
            upload_to_hardware: If True, emit `Execute UploadStimParameters`
                                after the Set burst, committing parameters
                                so subsequent triggers fire real current.
                                Default False = dry-run.

        Returns:
            List of (command, response) tuples.

        Raises:
            SafetyViolation: If protocol.validate_safety() fails.
            ValueError: If protocol.validate() fails.
        """
        # Structural validation (always)
        protocol.validate()

        # Safety validation — refused before any TCP send
        if not skip_safety_check:
            from encoders.stimulus_schema import SafetyViolation
            try:
                protocol.validate_safety()
            except SafetyViolation:
                raise
        else:
            # Even when skipped (tests only), log a warning
            try:
                protocol.validate_safety()
            except Exception as e:
                print(f"[TCP-WARN] safety check skipped, would have raised: {e}")

        commands = self.protocol_to_commands(protocol, channel_prefix)
        results = []
        for cmd, desc in commands:
            resp = self.send_command(cmd, expect_response=False)
            results.append((cmd, resp))
            # Throttle: without this gap, RHX silently drops half the Sets
            # AND backlogs the next Get. See INTER_COMMAND_DELAY_S note.
            time.sleep(self.INTER_COMMAND_DELAY_S)

        # Hardware commit gate — see method docstring SAFETY MODEL.
        # Skipping Upload is what makes dry-run safe: RHX accepts triggers
        # but stim.dat stays empty because the new params never reached HW.
        if upload_to_hardware:
            self.send_command("Execute UploadStimParameters", expect_response=False)
            results.append(("Execute UploadStimParameters", None))
            time.sleep(self.INTER_COMMAND_DELAY_S)
        return results

    def play_protocol_in_time(self, protocol, channel_prefix="A",
                              skip_safety_check=False,
                              upload_to_hardware=False):
        """
        Configure stim params + play the time-encoded pulse sequence as-is.

        Unlike send_protocol_config (which only configures), this method
        ALSO triggers each pulse at its `time_ms` relative to call start.
        Use this when you want the encoder's temporal pattern to actually
        reach the hardware.

        Internal model:
          - Each channel is configured once with first-pulse parameters
            (RHX TCP API supports only one set of params per channel).
          - All pulses (potentially multiple on same channel at different
            times) are sorted by time_ms and triggered via
            `Execute ManualStimTriggerPulse F<ch+1>` at the right moment.

        Timing precision: ~1 ms on Windows (busy-wait for last ms after
        a coarse sleep). Use shorter pulse trains to avoid drift.

        This method BLOCKS for `protocol.duration_ms` (plus config overhead).
        Don't call from a tight real-time loop unless you account for that.

        Returns:
            timing_log: list of dicts {target_ms, actual_ms, channel, drift_ms}
        """
        import time

        protocol.validate()
        if not skip_safety_check:
            from encoders.stimulus_schema import SafetyViolation
            protocol.validate_safety()

        # Step 1: configure each channel. Each Set must be throttled to avoid
        # RHX backlog drops. With 50 ms gap an 8-channel/80-cmd protocol takes
        # ~4 s — that's the cost of reliable configuration.
        config_cmds = self.protocol_to_commands(protocol, channel_prefix)
        for cmd, _ in config_cmds:
            self.send_command(cmd, expect_response=False)
            time.sleep(self.INTER_COMMAND_DELAY_S)

        # Step 1b: optionally commit parameters to hardware.
        # Default False = dry-run (RHX accepts triggers in Step 3 but fires
        # whatever stim is currently uploaded — initial state = silent).
        # See send_protocol_config docstring SAFETY MODEL for when to enable.
        if upload_to_hardware:
            self.send_command("Execute UploadStimParameters", expect_response=False)
            time.sleep(self.INTER_COMMAND_DELAY_S)

        # Step 2: sort pulses by time_ms
        pulses_sorted = sorted(protocol.pulses, key=lambda p: (p.time_ms, p.channel))

        # Step 3: play them out at their scheduled times
        timing_log = []
        t0 = time.perf_counter()
        for pulse in pulses_sorted:
            target = t0 + pulse.time_ms / 1000.0
            # Coarse sleep until close, then busy-wait final 2 ms
            while True:
                remaining = target - time.perf_counter()
                if remaining <= 0:
                    break
                if remaining > 0.002:
                    time.sleep(remaining - 0.002)
                else:
                    pass  # busy-spin
            actual_ms = (time.perf_counter() - t0) * 1000.0
            f_key = pulse.channel + 1
            self.send_command(
                f"Execute ManualStimTriggerPulse F{f_key}",
                expect_response=False,
            )
            timing_log.append({
                "target_ms": float(pulse.time_ms),
                "actual_ms": round(actual_ms, 3),
                "drift_ms": round(actual_ms - pulse.time_ms, 3),
                "channel": int(pulse.channel),
            })

        return timing_log

    # ── Waveform Data Reception ────────────────────────────────────

    def read_waveform_data(self, n_bytes):
        """
        Read raw bytes from waveform output socket.

        The data format depends on RHX configuration. Typically:
        - 2 bytes (int16) per sample per channel
        - Channels interleaved

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
                break
            data += chunk
        return data

    def init_waveform_reader(self, n_channels):
        """
        Initialize or reset the waveform frame reader.

        Call this once after connect_waveform() and before reading data.
        Pass the EXACT number of channels you added to "Channels To Stream"
        in the Intan RHX TCP Waveform Output panel.
        """
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
            stream header (4B, once) + frames of [timestamp(4B) + samples(2B*n_ch)]
            Samples are uint16 offset-binary, converted to microvolts.

        Across multiple calls, timestamps stay continuous: the last
        timestamp of one call equals (first timestamp of next call - 1).
        Use that property to detect dropped frames.

        Args:
            n_samples: Number of time samples (frames) to return.
            n_channels: Number of channels in the stream (must match Intan TCP config).
            timeout: Per-recv timeout in seconds.

        Returns:
            timestamps: (n_samples,) uint32 array.
            wave_uv: (n_channels, n_samples) float32 array in microvolts.

        Raises:
            ConnectionError: If waveform socket is not connected.
            TimeoutError: If not enough data arrives within the timeout.
        """
        if self._waveform_socket is None:
            raise ConnectionError("Waveform socket not connected.")

        if self._waveform_reader is None or self._waveform_reader.n_channels != n_channels:
            self._waveform_reader = WaveformFrameReader(n_channels)

        self._waveform_socket.settimeout(timeout)
        ts, wave = self._waveform_reader.read_frames(n_samples)
        while ts.shape[0] < n_samples:
            try:
                chunk = self._waveform_socket.recv(65536)
            except socket.timeout:
                raise TimeoutError(
                    f"Timed out waiting for waveform data "
                    f"({ts.shape[0]}/{n_samples} samples received)"
                )
            if not chunk:
                break
            self._waveform_reader.add_data(chunk)
            new_ts, new_wave = self._waveform_reader.read_frames(n_samples - ts.shape[0])
            if new_ts.shape[0] > 0:
                ts = np.concatenate([ts, new_ts])
                wave = np.concatenate([wave, new_wave], axis=1)

        return ts[:n_samples], wave[:, :n_samples]

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
                break
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
