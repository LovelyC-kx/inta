"""Regressions grounded in the supplied RHX 3.5.0 manuals and Intan source."""
import socket
import struct

import pytest

from intan_interface.command_channel import RHXCommandError
from intan_interface.tcp_client import IntanTCPClient, WaveformFrameReader, WaveformStreamError
from tests.test_rhx_session import client, protocol
from tests.test_waveform_format import make_synthetic_stream


@pytest.mark.parametrize("values,match", [
    ({"type": "ControllerRecordUSB3"}, "requires RHS"),
    ({"playback": "True"}, "Playback"),
    ({"synthetic": "unknown"}, "Synthetic"),
    ({"stimstepsizemicroamps": "nan"}, "StimStep"),
    ({"sampleratehertz": "25000"}, "phase_duration_us"),
    ({"stimstepsizemicroamps": "0.02"}, "amplitude_ua"),  # 10uA > 255 steps
    ({"stimstepsizemicroamps": "0.2"}, None),
])
def test_hardware_representability_is_checked_before_any_mutation(values, match):
    c = client()
    c._command_socket.values.update(values)
    if match is None:
        c.prepare_protocols([protocol()])
        assert c.prepared_session["pulse_codes"]["A-000"]["amplitude_code"] == 50
    else:
        with pytest.raises(ValueError, match=match):
            c.prepare_protocols([protocol()])
        assert not any(msg.startswith((b"Set ", b"Execute ")) for msg in c._command_socket.sent)
        assert not c._prepared_profiles


def test_fractional_current_and_interphase_ticks_are_not_silently_rounded():
    from encoders.stimulus_schema import Pulse, StimulusProtocol
    for pulse, match in [
        (Pulse(0, 0, amplitude_ua=10.1), "amplitude_ua"),
        (Pulse(0, 0, interphase_delay_us=50, waveform="BiphasicWithInterphaseDelay"),
         "interphase_delay_us"),
    ]:
        c = client()
        with pytest.raises(ValueError, match=match):
            c.prepare_protocols([StimulusProtocol([pulse], 8, 2000)])
        assert not any(msg.startswith(b"Set ") for msg in c._command_socket.sent)


@pytest.mark.parametrize("expander,analog,digital", [(False, 2, 2), (True, 8, 16)])
def test_all_legacy_output_routes_are_disabled_and_read_back(expander, analog, digital):
    c = client()
    sock = c._command_socket
    sock.values["expanderconnected"] = str(expander)
    sock.values["analog-out-1.stimenabled"] = "true"
    sock.values["digital-out-01.stimenabled"] = "true"
    c.prepare_protocols([protocol()])
    sent = b"".join(sock.sent)
    for prefix, count in [("ANALOG-OUT", analog), ("DIGITAL-OUT", digital)]:
        for ch in range(1, count + 1):
            name = f"{prefix}-{ch:02d}" if prefix == "DIGITAL-OUT" else f"{prefix}-{ch}"
            assert f"Set {name}.stimenabled false;".encode() in sent
            assert f"Get {name}.stimenabled;".encode() in sent
    assert c.prepared_session["hardware"]["synthetic"] is True
    assert c.prepared_session["physical_delivery_verified"] is False


def test_silent_analog_disable_drop_prevents_upload():
    c = client()
    sock = c._command_socket
    sock.values["analog-out-1.stimenabled"] = "true"
    send = sock.sendall
    sock.sendall = lambda data: None if data == b"Set ANALOG-OUT-1.stimenabled false;" else send(data)
    with pytest.raises(RuntimeError, match="ANALOG-OUT-1.stimenabled"):
        c.prepare_protocols([protocol()])
    assert b"Execute UploadStimParameters;" not in sock.sent
    assert not c._prepared_profiles


def test_silent_digital_disable_drop_still_prevents_upload():
    c = client()
    sock = c._command_socket
    sock.values["digital-out-01.stimenabled"] = "true"
    send = sock.sendall
    sock.sendall = lambda data: None if data == b"Set DIGITAL-OUT-01.stimenabled false;" else send(data)
    with pytest.raises(RuntimeError, match="DIGITAL-OUT-01.stimenabled=.*expected false"):
        c.prepare_protocols([protocol()])
    assert b"Execute UploadStimParameters;" not in sock.sent
    assert not c._prepared_profiles


@pytest.mark.parametrize("name", ["DIGITAL-OUT-1", "DIGITAL-IN-1"])
def test_controller_fixture_rejects_unpadded_digital_native_names(name):
    from intan_interface.command_channel import RHXCommandError
    c = client()
    with pytest.raises(RHXCommandError, match="two digits"):
        c.get(f"{name}.StimEnabled")


def test_settle_recovery_and_train_period_do_not_inherit_gui_values():
    c = client()
    c._command_socket.values.update({"a-000.enableampsettle": "false",
                                     "a-000.enablechargerecovery": "true",
                                     "a-000.poststimampsettlemicroseconds": "500000"})
    c.prepare_protocols([protocol()])
    values = c._command_socket.values
    assert values["a-000.enableampsettle"] == "true"
    assert values["a-000.enablechargerecovery"] == "false"
    assert values["a-000.poststimampsettlemicroseconds"] == "1000"
    assert values["a-000.pulsetrainperiodmicroseconds"] == "1000000"
    sent = c._command_socket.sent
    settle = sent.index(b"Execute UploadAmpSettleSettings;")
    stim = sent.index(b"Execute UploadStimParameters;")
    assert settle < stim
    assert any(b"Get UploadInProgress;" in msg for msg in sent[settle + 1:stim])


def test_stop_and_repeated_run_are_idempotent_and_record_requires_stop():
    c = client()
    c.stop(verify=True)
    assert b"Set RunMode Stop;" not in c._command_socket.sent
    c.run(verify=True)
    c.run(verify=True)
    assert c._command_socket.sent.count(b"Get RunMode;Get RunMode;Set RunMode Run;") == 1
    with pytest.raises(RuntimeError, match="Stop RHX"):
        c.record()
    c.stop(verify=True)
    c.record(verify=True)
    assert c.get("RunMode") == "Return: RunMode Record"


def test_transient_configuration_warnings_are_retained_but_errors_abort():
    for message, succeeds in [(b"Warning: temporary dependency", True),
                              (b"Error: invalid current", False)]:
        c = client()
        sock = c._command_socket
        send = sock.sendall

        def inject(data):
            send(data)
            if data.startswith(b"Set A-000.firstphaseduration"):
                sock.pending.extend(message)

        sock.sendall = inject
        if succeeds:
            c.prepare_protocols([protocol()])
            assert "temporary dependency" in c.prepared_session["configuration_warnings"]
        else:
            with pytest.raises(RHXCommandError, match="invalid current"):
                c.prepare_protocols([protocol()])
            assert b"Execute UploadStimParameters;" not in sock.sent


@pytest.mark.parametrize("command", ["Execute LoadSettingsFile config.xml",
    "Execute LoadStimulationSettingsFile stim.xml", "Set UseFastSettle True",
    "Set ANALOG-OUT-1.BaselineVoltageVolts 1"])
def test_nonlive_cannot_change_stimulation_indirectly(command):
    c = client("acquire_only")
    with pytest.raises(PermissionError):
        c.send_command(command, expect_response=False)
    assert not c._command_socket.sent


@pytest.mark.parametrize("command", ["Execute LoadSettingsFile config.xml",
    "Execute LoadStimulationSettingsFile stim.xml", "Execute RescanPorts",
    "Execute UploadAmpSettleSettings", "Set HeadstageGlobalSettle True"])
def test_configuration_changes_invalidate_verified_trigger_cache(command):
    c = client()
    c.prepare_protocols([protocol()])
    c.send_command(command, expect_response=False)
    assert not c._prepared_profiles and c.prepared_session is None


def test_waveform_magic_and_timestamp_corruption_poison_the_reader():
    raw, _ = make_synthetic_stream(2, 256)
    for index, replacement in [(0, b"oops"), (4 + 128 * 8, b"oops"),
                                (4 + 8, struct.pack("<I", 10))]:
        bad = bytearray(raw)
        bad[index:index + 4] = replacement
        reader = WaveformFrameReader(2)
        with pytest.raises(WaveformStreamError):
            reader.add_data(bad)
        with pytest.raises(WaveformStreamError):
            reader.read_frames()


def test_waveform_wrap_is_valid_but_cross_block_gap_is_not():
    raw, _ = make_synthetic_stream(2, 128, start_timestamp=0xffffff80)
    next_raw, _ = make_synthetic_stream(2, 128, start_timestamp=0)
    reader = WaveformFrameReader(2)
    reader.add_data(raw + next_raw)
    ts, _ = reader.read_frames()
    assert ts[127] == 0xffffffff and ts[128] == 0
    next_raw, _ = make_synthetic_stream(2, 128, start_timestamp=129)
    with pytest.raises(WaveformStreamError, match="gap/reset"):
        reader.add_data(next_raw)


def test_eof_does_not_return_a_short_waveform_window():
    rx, tx = socket.socketpair()
    c = IntanTCPClient()
    c._waveform_socket = rx
    raw, _ = make_synthetic_stream(2, 128)
    tx.sendall(raw)
    tx.close()
    with pytest.raises(ConnectionError, match="complete sample window"):
        c.read_waveform_samples(256, 2, timeout=.5)
    assert not c.is_waveform_connected and c._waveform_reader is None


def test_waveform_total_deadline_invalidates_partial_window():
    rx, tx = socket.socketpair()
    c = IntanTCPClient()
    c._waveform_socket = rx
    try:
        tx.sendall(b"\x08\x7a")
        with pytest.raises(TimeoutError):
            c.read_waveform_samples(1, 2, timeout=.02)
        assert not c.is_waveform_connected
    finally:
        tx.close()


def test_wrong_channel_count_is_detected_at_a_block_boundary():
    raw, _ = make_synthetic_stream(4, 256)
    with pytest.raises(WaveformStreamError):
        WaveformFrameReader(2).add_data(raw)


@pytest.mark.parametrize("count", [True, 0, -1, 1.5])
def test_invalid_waveform_layout_is_rejected(count):
    with pytest.raises(ValueError):
        WaveformFrameReader(count)
