import io
import socket
import struct
import time

import numpy as np
import pytest

from intan_interface.spike_stream import (
    SPIKE_MAGIC, SpikeEvent, SpikePacketParser, SpikeStreamReader, SpikeStreamError,
)
from intan_interface.spike_windows import ProgressMarker, build_spike_window


def packet(name="A-000", sample=10, spike_id=1):
    return SPIKE_MAGIC + name.encode() + struct.pack("<I", sample) + bytes([spike_id])


def event(sample, name="A-000", spike_id=1, epoch="session-1"):
    return SpikeEvent(name, sample & 0xFFFFFFFF, sample, spike_id, 999999, epoch)


def window(events, **kwargs):
    options = dict(channel_names=["A-000", "B-000"], start_sample=100,
                   end_sample=200, acquisition_epoch="session-1",
                   watermark=ProgressMarker(200, "session-1", "rhx_stream"))
    options.update(kwargs)
    return build_spike_window(events, **options)


def test_all_packet_split_boundaries_and_full_channel_id():
    raw = packet("A-000", 1) + packet("B-000", 2, 0x80) + packet("D-031", 3, 0x40)
    for split in range(len(raw) + 1):
        parser = SpikePacketParser("session-1")
        got = parser.feed(raw[:split], 11) + parser.feed(raw[split:], 22)
        assert [e.native_channel for e in got] == ["A-000", "B-000", "D-031"]
        assert [e.sample_index for e in got] == [1, 2, 3]
        assert [e.spike_id for e in got] == [1, 0x80, 0x40]
        assert [e.usable for e in got] == [True, False, True]


def test_byte_at_a_time_coalesced_packets():
    parser = SpikePacketParser("session-1")
    got = []
    for b in packet(sample=4) + packet(sample=4, name="B-000"):
        got.extend(parser.feed(bytes([b])))
    assert len(got) == 2 and got[0].sample_index == got[1].sample_index


def test_parser_resynchronization_is_measurable_not_silent():
    parser = SpikePacketParser("session-1")
    got = parser.feed(b"bad bytes" + packet())
    assert len(got) == 1
    assert parser.discarded_bytes == len(b"bad bytes")


def test_invalid_native_name_resynchronizes_without_aliasing():
    parser = SpikePacketParser("session-1")
    got = parser.feed(packet("Q-000") + packet("A-000"))
    assert [e.native_channel for e in got] == ["A-000"]
    assert parser.malformed_packets == 1


def test_counter_wrap_and_reset_are_distinct():
    parser = SpikePacketParser("session-1")
    got = parser.feed(packet(sample=0xFFFFFFF0) + packet(sample=0x10))
    assert got[1].sample_index == 0x100000010
    with pytest.raises(SpikeStreamError, match="backwards/reset"):
        parser.feed(packet(sample=2))


def test_hardware_window_is_independent_of_arrival_time():
    result = window([event(99), event(100), event(150, "B-000"), event(199), event(200)])
    np.testing.assert_array_equal(result.bin_counts(2), [[1, 0], [1, 1]])
    assert result.status == "VALID"


def test_artifact_bits_and_each_pulse_mask_are_applied():
    result = window([event(105), event(115), event(135), event(155), event(170, spike_id=0x80),
                     event(180, "B-000", 0x40)],
                    artifact_intervals={"*": [(100, 110), (130, 140)],
                                        "A-000": [(105, 120)]})
    assert result.masked_events == 3
    assert result.excluded_id_events == 1
    assert [e.sample_index for e in result.events] == [155, 180]
    assert result.valid_samples_per_channel == (70, 80)


def test_silence_needs_progress_and_full_mask_is_invalid():
    assert window([]).status == "SILENT"
    with pytest.raises(SpikeStreamError, match="advanced"):
        window([], watermark=ProgressMarker(199, "session-1", "rhx_stream"))
    invalid = window([], artifact_intervals={"*": [(0, 1000)]})
    assert invalid.status == "INVALID"
    with pytest.raises(SpikeStreamError):
        invalid.bin_counts(40)


@pytest.mark.parametrize("marker", [None, ProgressMarker(200, "wrong", "rhx_stream"),
                                    ProgressMarker(200, "session-1", "host_clock")])
def test_no_host_clock_or_mixed_epoch_watermarks(marker):
    with pytest.raises(SpikeStreamError):
        window([], watermark=marker)


def test_future_events_are_not_consumed_and_epochs_never_mix():
    events = [event(150), event(250)]
    assert len(window(events).events) == 1 and len(events) == 2
    with pytest.raises(SpikeStreamError, match="epochs"):
        window(events + [event(175, epoch="another-session")])


@pytest.mark.parametrize("names", [["A-000", "A-000"], ["000"], []])
def test_channel_map_must_be_explicit_and_unique(names):
    with pytest.raises(ValueError):
        window([], channel_names=names)


def wait_error(reader):
    deadline = time.monotonic() + 2
    while reader.error is None and time.monotonic() < deadline:
        time.sleep(.001)
    assert reader.error is not None


def test_receiver_eof_invalidates_data_instead_of_busy_looping():
    rx, tx = socket.socketpair()
    reader = SpikeStreamReader(rx, acquisition_epoch="session-1").start()
    try:
        tx.close()
        wait_error(reader)
        with pytest.raises(SpikeStreamError, match="EOF"):
            reader.drain()
    finally:
        reader.stop()
        rx.close()


def test_receiver_overflow_keeps_raw_bytes_and_invalidates_control():
    rx, tx = socket.socketpair()
    raw = io.BytesIO()
    reader = SpikeStreamReader(rx, acquisition_epoch="session-1", max_events=1, raw_sink=raw).start()
    try:
        payload = packet(sample=10) + packet(sample=11)
        tx.sendall(payload)
        wait_error(reader)
        with pytest.raises(SpikeStreamError, match="overflow"):
            reader.drain()
        assert raw.getvalue() == payload
    finally:
        reader.stop()
        rx.close()
        tx.close()


def test_receiver_corrupt_stream_is_failure_even_if_parser_can_resync():
    rx, tx = socket.socketpair()
    reader = SpikeStreamReader(rx, acquisition_epoch="session-1").start()
    try:
        tx.sendall(b"garbage" + packet())
        wait_error(reader)
        with pytest.raises(SpikeStreamError, match="Corrupt"):
            reader.drain()
    finally:
        reader.stop()
        rx.close()
        tx.close()


def test_threshold_demo_uses_native_names_samples_and_excludes_artifacts():
    from experiments.spike_triggered_stim_demo import SpikeThresholdPolicy
    policy = SpikeThresholdPolicy(["A-000", "B-000"], 1000, 1, 1, 500)
    assert not policy.observe(event(10, "A-000"))
    assert not policy.observe(event(11, "B-000"))
    assert not policy.observe(event(12, "A-000", 0x80))
    assert policy.observe(event(13, "A-000"))
    assert not policy.observe(event(14, "A-000"))
    assert not policy.observe(event(15, "A-000"))
    assert policy.observe(event(520, "A-000"))


def test_threshold_demo_expires_old_hardware_events():
    from experiments.spike_triggered_stim_demo import SpikeThresholdPolicy
    policy = SpikeThresholdPolicy(["A-000"], 1000, 1, 1, 500)
    assert not policy.observe(event(10))
    assert not policy.observe(event(1010))
    assert policy.observe(event(1011))
