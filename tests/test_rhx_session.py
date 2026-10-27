"""Protocol/transport regressions using a fake RHX, never laboratory hardware."""
import socket
import re

import pytest

from encoders.stimulus_schema import Pulse, StimulusProtocol, SafetyViolation
from intan_interface.command_channel import RHXCommandError
from intan_interface.tcp_client import IntanTCPClient


class FakeRHXSocket:
    def __init__(self):
        self.values = {"runmode": "Stop", "uploadinprogress": "False",
                       "type": "ControllerStimRecord", "version": "3.5.0",
                       "synthetic": "True", "playback": "False",
                       "sampleratehertz": "30000", "stimstepsizemicroamps": "0.5",
                       "expanderconnected": "False", "usefastsettle": "False",
                       "headstageglobalsettle": "False", "desiredlowersettlebandwidthhertz": "1000",
                       "actuallowersettlebandwidthhertz": "1205.9", "chargerecovery": "CurrentLimited",
                       "chargerecoverycurrentlimitmicroamps": "0.01",
                       "chargerecoverytargetvoltagevolts": "0",
                       **{f"{p}.numberamplifierchannels": "8" if p == "a" else "0"
                          for p in "abcd"}}
        self.pending = bytearray()
        self.sent = []
        self.timeout = 2.0
        self.closed = False

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value

    def sendall(self, data):
        self.sent.append(data)
        for cmd in data.decode().split(";"):
            if not cmd:
                continue
            words = cmd.split()
            if len(words) > 1 and words[1].lower().startswith(("digital-in-", "digital-out-")):
                name = words[1].split(".")[0].lower()
                if not re.fullmatch(r"digital-(?:in|out)-(?:0[1-9]|1[0-6])", name):
                    self.pending.extend(b"Error: Unrecognized digital native channel; use two digits")
                    continue
            if words[0].lower() == "set":
                if words[1].lower() == "runmode":
                    current = self.values["runmode"]
                    if (words[2] == "Stop" and current == "Stop") or (
                            words[2] != "Stop" and current != "Stop"):
                        self.pending.extend(b"Error: Board is in the wrong run mode")
                        continue
                self.values[words[1].lower()] = words[2]
            elif words[0].lower() == "get":
                key = words[1]
                if key.lower() not in self.values:
                    self.pending.extend(b"Error: Unrecognized parameter")
                else:
                    value = self.values[key.lower()]
                    # RHX StateItem replies omit channel/port prefixes.
                    key = "RunMode" if key.lower() == "runmode" else key.split(".")[-1]
                    self.pending.extend(f"Return: {key} {value}".encode())

    def recv(self, count):
        if not self.pending:
            raise socket.timeout()
        # Split even the known state value across TCP recv calls.
        chunk = bytes(self.pending[:3])
        del self.pending[:3]
        return chunk

    def close(self):
        self.closed = True


def client(mode="live"):
    c = IntanTCPClient(execution_mode=mode)
    c._command_socket = FakeRHXSocket()
    c.INTER_COMMAND_DELAY_S = 0
    return c


def protocol(times=(0, 250, 500)):
    return StimulusProtocol([Pulse(0, t) for t in times], 8, 2000)


class CoalescedStartSocket(FakeRHXSocket):
    """Model Qt receiving a bare start and the next query in one read batch.

    Set Run enters runController synchronously; later commands in that batch
    cannot execute until Stop. A preceding reply proves the batch was read.
    """
    def __init__(self):
        super().__init__()
        self.unparsed_start = None

    def sendall(self, data):
        if data.startswith((b"Set RunMode Run;", b"Set RunMode Record;")):
            self.unparsed_start = data
            return
        if self.unparsed_start is not None:
            super().sendall(self.unparsed_start)
            self.unparsed_start = None
            # The new query was coalesced behind the blocking startup Set.
            return
        super().sendall(data)


def test_legacy_start_then_query_can_be_stranded_in_same_rhx_batch():
    c = client()
    c._command_socket = CoalescedStartSocket()
    c.send_command("Set RunMode Run", expect_response=False)
    with pytest.raises(TimeoutError, match="Get RunMode"):
        c.get("RunMode", timeout=.02)
    assert not c.is_command_connected


@pytest.mark.parametrize("mode", ["Run", "Record"])
def test_start_prefence_keeps_confirmation_out_of_blocking_start_batch(mode):
    c = client()
    sock = CoalescedStartSocket()
    c._command_socket = sock
    method = c.run if mode == "Run" else c.record
    assert method(verify=True) == f"Return: RunMode {mode}"
    assert c.is_command_connected
    packet = f"Get RunMode;Get RunMode;Set RunMode {mode};".encode()
    assert packet in sock.sent
    # No query follows the blocking Set inside its already-received batch.
    assert packet.endswith(f"Set RunMode {mode};".encode())


def test_mode_confirmation_uses_remaining_total_budget_not_two_second_cap():
    c = client()
    observed = []
    def get(parameter, timeout):
        observed.append(timeout)
        return "Return: RunMode Run"
    c.get = get
    assert c.wait_for_run_mode("Run", timeout=10) == "Return: RunMode Run"
    assert 9 < observed[0] <= 10


def test_fragmented_delimiter_free_response_and_state_fence():
    c = client()
    assert c.get("RunMode") == "Return: RunMode Stop"
    c.set("RunMode", "Record")
    assert c.get("RunMode") == "Return: RunMode Record"
    assert all(msg.endswith(b";") for msg in c._command_socket.sent)


def test_prior_set_error_is_not_silently_consumed_as_get_response():
    c = client()
    c._command_socket.pending.extend(b"Error: bad stimulation parameter")
    with pytest.raises(RHXCommandError, match="bad stimulation"):
        c.get("RunMode")
    assert c.get("RunMode") == "Return: RunMode Stop"


def test_timeout_invalidates_socket_instead_of_reusing_late_reply():
    c = client()
    sock = c._command_socket
    sock.sendall = lambda data: None
    with pytest.raises(TimeoutError):
        c.get("RunMode")
    assert sock.closed and not c.is_command_connected


@pytest.mark.parametrize("mode", ["simulate", "acquire_only"])
def test_nonlive_modes_cannot_send_raw_triggers_or_upload(mode):
    c = client(mode)
    for cmd in ("Execute ManualStimTriggerPulse F1", "Execute UploadStimParameters",
                "Set A-000.stimenabled true"):
        with pytest.raises(PermissionError):
            c.send_command(cmd, expect_response=False)
    assert c.send_protocol_config(protocol()) == []
    timing = c.play_protocol_in_time(protocol())
    assert all(not row["sent"] for row in timing)
    assert c._command_socket.sent == []


def test_running_upload_refused_before_stim_sets():
    c = client()
    c._command_socket.values["runmode"] = "Record"
    with pytest.raises(RuntimeError, match="Stop RHX"):
        c.send_protocol_config(protocol(), upload_to_hardware=True)
    assert not any(b"Set " in data for data in c._command_socket.sent)


def test_prepared_profile_executes_all_pulses_and_checks_cancellation():
    c = client()
    c.prepare_protocols([protocol()])
    c.run()
    timing = c.play_prepared_protocol(protocol())
    assert len(timing) == 3
    assert sum(b"ManualStimTriggerPulse" in cmd for cmd in c._command_socket.sent) == 3
    assert all(row["hardware_sample"] is None for row in timing)
    import threading
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        c.play_prepared_protocol(protocol(), cancel_event=cancel)


def test_profile_mismatch_refused_before_upload():
    c = client()
    mixed = StimulusProtocol([Pulse(0, 0), Pulse(0, 250, amplitude_ua=5)], 8, 2000)
    with pytest.raises(ValueError, match="same pulse profile"):
        c.prepare_protocols([mixed])
    assert c._command_socket.sent == []


def test_new_profile_disables_other_headstages_and_sets_trigger_mode():
    c = client()
    c._command_socket.values["b.numberamplifierchannels"] = "1"
    c.prepare_protocols([protocol()])
    messages = b"".join(c._command_socket.sent)
    assert b"Set B-000.stimenabled false;" in messages
    assert b"Set A-000.pulseortrain SinglePulse;" in messages
    assert b"Set A-000.posttriggerdelaymicroseconds 0;" in messages


def test_cross_protocol_interval_and_total_budget_are_shared():
    c = client()
    c.prepare_protocols([protocol((0,))])
    c.run()
    c.play_prepared_protocol(protocol((0,)))
    with pytest.raises(RuntimeError, match="session inter-pulse"):
        c.send_command("Execute ManualStimTriggerPulse F1", expect_response=False)


@pytest.mark.parametrize("field,value", [
    ("amplitude_ua", -1), ("amplitude_ua", float("nan")),
    ("amplitude_ua", float("inf")), ("phase_duration_us", -1),
    ("phase_duration_us", "500"), ("interphase_delay_us", -1),
    ("polarity", "CathodicFirst"), ("waveform", "Triphasic"),
])
def test_invalid_physical_values_refused(field, value):
    with pytest.raises(SafetyViolation):
        StimulusProtocol([Pulse(0, 0, **{field: value})], 8, 2000).validate_safety()


@pytest.mark.parametrize("time", [float("nan"), float("inf"), True])
def test_invalid_event_time_refused(time):
    with pytest.raises(ValueError):
        StimulusProtocol([Pulse(0, time)], 8, 2000).validate()


def test_command_injection_cannot_bypass_mode():
    c = client("acquire_only")
    with pytest.raises(ValueError):
        c.send_command("Get RunMode;Execute ManualStimTriggerPulse F1")
    assert c._command_socket.sent == []


def test_configuration_union_does_not_count_as_an_executed_trial_budget():
    c = client()
    protocols = [StimulusProtocol([Pulse(ch, 0, amplitude_ua=40)], 8, 2000)
                 for ch in range(8)]
    c.prepare_protocols(protocols)
    c.run()
    for key in range(1, 6):
        c.send_command(f"Execute ManualStimTriggerPulse F{key}", expect_response=False)
    with pytest.raises(RuntimeError, match="rolling 2-second charge"):
        c.send_command("Execute ManualStimTriggerPulse F6", expect_response=False)


def test_upload_invalidates_previously_verified_trigger_cache():
    c = client()
    c.prepare_protocols([protocol()])
    c.send_command("Execute UploadStimParameters", expect_response=False)
    with pytest.raises(RuntimeError, match="no verified uploaded"):
        c.send_command("Execute ManualStimTriggerPulse F1", expect_response=False)


def test_silent_disable_drop_on_other_headstage_is_detected():
    c = client()
    sock = c._command_socket
    sock.values.update({"b.numberamplifierchannels": "1", "b-000.stimenabled": "true"})
    send = sock.sendall

    def drop_one_set(data):
        if data != b"Set B-000.stimenabled false;":
            send(data)

    sock.sendall = drop_one_set
    with pytest.raises(RuntimeError, match="B-000.stimenabled"):
        c.prepare_protocols([protocol()])
    assert not c._prepared_profiles
