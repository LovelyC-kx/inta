"""Loss-aware RHX 5002 acquisition. All times used for analysis are samples.

Format and ID bits: Intan-RHX tcpdataoutputthread.cpp and waveformfifo.h.
Receiver EOF, timestamp resets, disk errors and queue overflow are failures,
not a silent neural window. Start a new receiver/epoch after acquisition resets.
"""
from dataclasses import dataclass
import queue
import re
import socket
import struct
import threading
import time


SPIKE_MAGIC = struct.pack("<I", 0x3AE2710F)
SPIKE_PACKET_BYTES = 14
CHANNEL_NAME = re.compile(r"[A-H]-[0-9]{3}\Z")
LIKELY_ARTIFACT = 0x80
UNCLASSIFIED = 0x40
VALID_SPIKE_TYPES = 0x0F


class SpikeStreamError(RuntimeError):
    """Acquisition is invalid; downstream control must not use its output."""


@dataclass(frozen=True)
class SpikeEvent:
    native_channel: str
    raw_sample_index: int
    sample_index: int
    spike_id: int
    host_rx_ns: int
    acquisition_epoch: str

    @property
    def usable(self):
        """Default policy: include unsorted/type spikes, exclude artifacts/unknown bits."""
        return bool(self.spike_id & (UNCLASSIFIED | VALID_SPIKE_TYPES)) and not (
            self.spike_id & ~(UNCLASSIFIED | VALID_SPIKE_TYPES))


class TimestampUnwrapper:
    def __init__(self):
        self.previous = None
        self.unwrapped = None

    def update(self, raw):
        if self.previous is None:
            self.previous = raw
            self.unwrapped = raw
            return raw
        delta = (raw - self.previous) & 0xFFFFFFFF
        # This stream is sample-ordered. A backwards/reset timestamp is not
        # repaired by guessing a new epoch. Gaps >= half the counter range
        # cannot be unwrapped from sparse spike events alone.
        if delta >= 0x80000000:
            raise SpikeStreamError("RHX timestamp moved backwards/reset; start a new epoch")
        self.previous = raw
        self.unwrapped += delta
        return self.unwrapped


class SpikePacketParser:
    def __init__(self, acquisition_epoch):
        if not isinstance(acquisition_epoch, str) or not acquisition_epoch:
            raise ValueError("An explicit nonempty acquisition_epoch is required")
        self.acquisition_epoch = acquisition_epoch
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.malformed_packets = 0
        self.clock = TimestampUnwrapper()

    def feed(self, data, host_rx_ns=None):
        self.buffer.extend(data)
        host_rx_ns = time.monotonic_ns() if host_rx_ns is None else host_rx_ns
        events = []
        while True:
            pos = self.buffer.find(SPIKE_MAGIC)
            if pos < 0:
                discarded = max(0, len(self.buffer) - len(SPIKE_MAGIC) + 1)
                self.discarded_bytes += discarded
                del self.buffer[:discarded]
                break
            if pos:
                self.discarded_bytes += pos
                del self.buffer[:pos]
            if len(self.buffer) < SPIKE_PACKET_BYTES:
                break
            name = bytes(self.buffer[4:9]).decode("ascii", errors="replace")
            raw = struct.unpack_from("<I", self.buffer, 9)[0]
            spike_id = self.buffer[13]
            if not CHANNEL_NAME.fullmatch(name) or spike_id == 0:
                self.malformed_packets += 1
                self.discarded_bytes += 1
                del self.buffer[:1]
                continue
            sample = self.clock.update(raw)
            events.append(SpikeEvent(name, raw, sample, spike_id, host_rx_ns,
                                     self.acquisition_epoch))
            del self.buffer[:SPIKE_PACKET_BYTES]
        return events


class SpikeStreamReader:
    """Single reader for a supplied connected socket; never opens a connection.

    raw_sink is an optional binary file owned by the caller. Close it only after
    stop() joins the receiver. Full events, including artifacts, remain available
    for audit; filtering happens at window construction, not inside the parser.
    """
    def __init__(self, sock, *, acquisition_epoch, max_events=100000, raw_sink=None):
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        self.sock = sock
        self.parser = SpikePacketParser(acquisition_epoch)
        self.queue = queue.Queue(maxsize=max_events)
        self.raw_sink = raw_sink
        self.raw_bytes = 0
        self.events_received = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("A reader may only be started once")
        self.sock.settimeout(0.05)
        self._thread = threading.Thread(target=self._run, name="RHX-5002", daemon=True)
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
                    raise SpikeStreamError("RHX spike socket closed (EOF)")
                host_rx_ns = time.monotonic_ns()
                self.raw_bytes += len(data)
                if self.raw_sink is not None:
                    count = self.raw_sink.write(data)
                    if count is not None and count != len(data):
                        raise SpikeStreamError("Incomplete raw spike journal write")
                events = self.parser.feed(data, host_rx_ns=host_rx_ns)
                if self.parser.discarded_bytes or self.parser.malformed_packets:
                    raise SpikeStreamError("Corrupt/misaligned spike bytes; raw journal retained")
                for event in events:
                    try:
                        self.queue.put_nowait(event)
                    except queue.Full as exc:
                        raise SpikeStreamError("Spike event queue overflow; acquisition invalid") from exc
                    self.events_received += 1
        except Exception as exc:
            if not self._stop.is_set():
                self.error = exc

    def raise_if_failed(self):
        if self.error is not None:
            raise SpikeStreamError(str(self.error)) from self.error

    def drain(self):
        self.raise_if_failed()
        events = []
        while True:
            try:
                events.append(self.queue.get_nowait())
            except queue.Empty:
                return events

    def stop(self, timeout=2.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("Spike receiver did not stop; raw journal must remain open")

