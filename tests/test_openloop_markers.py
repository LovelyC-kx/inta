import struct

import numpy as np
import pytest

from experiments.openloop_backend import encode_marker_bytes
from intan_interface.marker_stream import MarkerParser, StimMarkerTracker
from intan_interface.spike_stream import SpikeStreamError


def profile(negative=True, gap=0):
    return {"A-000": {"amplitude_code": 20, "phase_samples": 15,
                      "interphase_samples": gap, "negative_first": negative}}


def wire(negative=True, gap=0, start=120, ttl=False):
    words = np.zeros((256, 2 if ttl else 1), dtype="<u2")
    first, second = ((0x114, 20) if negative else (20, 0x114))
    words[start:start + 15, 0] = first | 0x2000
    words[start + 15 + gap:start + 30 + gap, 0] = second | 0x2000
    if ttl:
        words[100:110, 1] = 1 << 3
    return encode_marker_bytes(0, words)


@pytest.mark.parametrize("negative,gap", [(True, 0), (False, 15), (True, 10)])
def test_typed_stim_phases_and_ttl_survive_tcp_and_block_splits(negative, gap):
    raw = wire(negative, gap, ttl=True)
    parser = MarkerParser(["A-000"], digital_in=True)
    tracker = StimMarkerTracker(["A-000"], profile(negative, gap), ttl_bit=3)
    markers, edges = [], []
    for i in range(0, len(raw), 7):
        for block in parser.feed(raw[i:i + 7]):
            got, ttl = tracker.feed(block)
            markers.extend(got)
            edges.extend(ttl)
    assert len(markers) == 1
    assert (markers[0].start_sample, markers[0].end_sample) == (120, 150 + gap)
    assert edges == [100]
    assert not tracker.active and not parser.buffer


@pytest.mark.parametrize("kind", ["amplitude", "duration", "compliance", "unknown_flag", "magic", "timestamp"])
def test_marker_corruption_invalidates_acquisition(kind):
    raw = bytearray(wire())
    if kind == "magic":
        raw[:4] = b"oops"
    elif kind == "timestamp":
        raw[4 + 6 * 10:8 + 6 * 10] = struct.pack("<I", 77)
    else:
        words = np.zeros((256, 1), dtype="<u2")
        words[120:135, 0] = 0x114
        words[135:150, 0] = 20
        if kind == "amplitude":
            words[124, 0] = 0x115
        elif kind == "duration":
            words[134, 0] = 20
        elif kind == "compliance":
            words[200, 0] = 0x8000
        else:
            words[20, 0] = 0x200
        raw = encode_marker_bytes(0, words)
    parser, tracker = MarkerParser(["A-000"]), StimMarkerTracker(["A-000"], profile())
    with pytest.raises(SpikeStreamError):
        for block in parser.feed(raw):
            tracker.feed(block)


def test_status_only_words_are_not_pulses_and_wrap_is_continuous():
    parser, tracker = MarkerParser(["A-000"]), StimMarkerTracker(["A-000"], profile())
    raw = encode_marker_bytes(0xffffff80, np.full((256, 1), 0x6000, dtype="<u2"))
    blocks = parser.feed(raw)
    assert blocks[1].start_sample == 1 << 32
    assert all(tracker.feed(b)[0] == [] for b in blocks)


@pytest.mark.parametrize("names", [[], ["A-000", "A-000"], ["B-000", "A-000"], ["000"]])
def test_marker_layout_is_explicit_and_canonical(names):
    with pytest.raises(ValueError):
        MarkerParser(names)
