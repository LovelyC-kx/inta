"""Typed RHS 5001 STIM/TTL stream, on the acquisition sample clock.

Wire order: Intan TCPDataOutputThread::run (pinned in docs/OPENLOOP.md).
Only explicitly configured STIM words and an optional DIGITAL-IN word are
accepted here. These uint16 words are never converted to amplifier microvolts.
"""
from dataclasses import dataclass
import queue
import socket
import struct
import threading
import time

import numpy as np

from .spike_stream import CHANNEL_NAME, SpikeStreamError, TimestampUnwrapper

MAGIC = struct.pack("<I", 0x2ef07a08)
COUNTER = 1 << 32


@dataclass(frozen=True)
class MarkerBlock:
    start_sample: int
    words: np.ndarray  # (128, n_words), raw uint16
    host_rx_ns: int

    @property
    def end_sample(self):
        return self.start_sample + len(self.words)


class MarkerParser:
    def __init__(self, stim_channels, *, digital_in=False):
        self.stim_channels = tuple(stim_channels)
        if (not self.stim_channels or tuple(sorted(set(self.stim_channels))) != self.stim_channels or
                any(not CHANNEL_NAME.fullmatch(n) for n in self.stim_channels)):
            raise ValueError("Use sorted unique native STIM channel names")
        self.digital_in = bool(digital_in)
        self.n_words = len(self.stim_channels) + self.digital_in
        self.frame_bytes = 4 + self.n_words * 2
        self.block_bytes = 4 + 128 * self.frame_bytes
        self.buffer = bytearray()
        self.clock = TimestampUnwrapper()
        self.previous = None
        self.failed = False

    def feed(self, data, host_rx_ns=None):
        if self.failed:
            raise SpikeStreamError("Marker parser previously failed")
        self.buffer.extend(data)
        host_rx_ns = time.monotonic_ns() if host_rx_ns is None else host_rx_ns
        result = []
        try:
            while len(self.buffer) >= self.block_bytes:
                if self.buffer[:4] != MAGIC:
                    raise SpikeStreamError("5001 STIM/TTL magic/layout mismatch")
                frames = np.frombuffer(bytes(self.buffer[4:self.block_bytes]), dtype=np.uint8).reshape(128, -1)
                raw_ts = frames[:, :4].copy().view("<u4").ravel()
                if (np.any(np.diff(raw_ts) != 1) or (self.previous is not None and
                        int(raw_ts[0]) != ((self.previous + 1) & 0xffffffff))):
                    raise SpikeStreamError("5001 acquisition timestamp gap/reset")
                start = self.clock.update(int(raw_ts[0]))
                self.clock.update(int(raw_ts[-1]))
                self.previous = int(raw_ts[-1])
                words = frames[:, 4:].copy().view("<u2").reshape(128, self.n_words)
                result.append(MarkerBlock(start, words, host_rx_ns))
                del self.buffer[:self.block_bytes]
        except Exception:
            self.failed = True
            raise
        return result


@dataclass(frozen=True)
class StimMarker:
    native_channel: str
    start_sample: int
    end_sample: int
    amplitude_code: int
    phase_samples: int
    interphase_samples: int
    negative_first: bool


class StimMarkerTracker:
    """Validate every digital phase against the prepared pulse profile.

    Work on constant-word runs, so a quiet 30kHz stream does not require a
    Python callback per sample. A marker denotes a digital controller record,
    not measured electrode current. Global pre/post artifact masks are applied
    separately by the feature contract.
    """
    def __init__(self, stim_channels, profiles, *, ttl_bit=None):
        self.channels = tuple(stim_channels)
        if set(self.channels) != set(profiles):
            raise ValueError("One verified pulse profile is required per STIM word")
        self.profiles = profiles
        self.active = {}
        self.next_sample = None
        self.ttl_bit = ttl_bit
        if ttl_bit is not None and (type(ttl_bit) is not int or not 0 <= ttl_bit < 16):
            raise ValueError("ttl_bit must be 0..15")
        self.previous_ttl = None

    def feed(self, block):
        if self.next_sample is not None and self.next_sample != block.start_sample:
            raise SpikeStreamError("Marker tracker received a discontinuous block")
        self.next_sample = block.end_sample
        markers = []
        for col, name in enumerate(self.channels):
            words = block.words[:, col]
            if np.any(words & 0x8000):
                raise SpikeStreamError(f"Compliance limit on {name}")
            if np.any(words & 0x1e00):
                raise SpikeStreamError(f"Unknown STIM flags on {name}")
            # Ignore settle/recovery flags for current detection, retain polarity.
            values = words & 0x1ff
            changes = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
            for a, b in zip(changes[:-1], changes[1:]):
                start, end, word = block.start_sample + int(a), block.start_sample + int(b), int(values[a])
                while start < end:
                    state = self.active.get(name)
                    if state is None:
                        if not word & 0xff:
                            break
                        p = self.profiles[name]
                        first = p["amplitude_code"] | (0x100 if p["negative_first"] else 0)
                        if word != first:
                            raise SpikeStreamError(f"Unexpected first phase on {name}")
                        phase, gap = p["phase_samples"], p["interphase_samples"]
                        segments = [(start + phase, first)]
                        if gap:
                            segments.append((start + phase + gap, 0))
                        second = p["amplitude_code"] | (0 if p["negative_first"] else 0x100)
                        segments.append((start + phase * 2 + gap, second))
                        state = {"start": start, "segments": segments, "index": 0}
                        self.active[name] = state
                    boundary, expected = state["segments"][state["index"]]
                    # Zero current may retain a polarity bit in the RHS record.
                    matches = (word & 0xff) == 0 if expected == 0 else word == expected
                    if not matches:
                        raise SpikeStreamError(f"STIM amplitude/phase/timing mismatch on {name}")
                    start = min(end, boundary)
                    if start == boundary:
                        state["index"] += 1
                        if state["index"] == len(state["segments"]):
                            p = self.profiles[name]
                            markers.append(StimMarker(name, state["start"], boundary,
                                                      p["amplitude_code"], p["phase_samples"],
                                                      p["interphase_samples"], p["negative_first"]))
                            del self.active[name]
        edges = []
        if self.ttl_bit is not None:
            states = ((block.words[:, -1] >> self.ttl_bit) & 1).astype(bool)
            previous = np.r_[states[0] if self.previous_ttl is None else self.previous_ttl, states[:-1]]
            edges = (np.flatnonzero(states & ~previous) + block.start_sample).tolist()
            self.previous_ttl = bool(states[-1])
        return sorted(markers, key=lambda m: (m.start_sample, m.native_channel)), edges


class MarkerStreamReader:
    """Single socket owner; bounded queue and raw journal, like the 5002 reader."""
    def __init__(self, sock, parser, raw_sink, *, max_blocks=4096):
        self.sock, self.parser, self.raw_sink = sock, parser, raw_sink
        self.queue = queue.Queue(maxsize=max_blocks)
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Marker reader can only start once")
        self.sock.settimeout(.05)
        self._thread = threading.Thread(target=self._run, name="RHX-5001-STIM", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    data = self.sock.recv(65536)
                except socket.timeout:
                    continue
                if not data:
                    raise SpikeStreamError("Marker socket EOF")
                host_rx_ns = time.monotonic_ns()
                if self.raw_sink.write(data) != len(data):
                    raise SpikeStreamError("Incomplete marker journal write")
                for block in self.parser.feed(data, host_rx_ns=host_rx_ns):
                    try:
                        self.queue.put_nowait(block)
                    except queue.Full as exc:
                        raise SpikeStreamError("Marker queue overflow") from exc
        except Exception as exc:
            if not self._stop.is_set():
                self.error = exc

    def drain(self):
        if self.error is not None:
            raise SpikeStreamError(str(self.error)) from self.error
        result = []
        while True:
            try:
                result.append(self.queue.get_nowait())
            except queue.Empty:
                return result

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)
            if self._thread.is_alive():
                raise TimeoutError("Marker receiver failed to stop")
