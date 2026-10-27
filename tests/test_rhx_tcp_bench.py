"""Offline checks of the actual-connection CLI, including closed-port failure."""
import json
import socket
import struct
import threading
import time

import pytest

from experiments import rhx_tcp_bench as cli
from intan_interface.spike_stream import SPIKE_MAGIC
from tests.test_openloop_live_transport import SocketPairRHX
from tests.test_rhx_session import client


def test_closed_actual_tcp_port_fails_instead_of_falling_back(tmp_path):
    # A bound, non-listening local socket reliably refuses connection; no
    # configured RHX ports or lab instruments are contacted by this test.
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
        result = cli.main(["--demo", "--cmd-port", str(port), "--out", str(tmp_path / "closed")])
    report = json.loads((tmp_path / "closed" / "session.json").read_text())
    assert result == 1 and report["status"] == "FAILED"
    assert report["connected"] is False and report["triggers_sent"] == 0


def test_demo_bench_refuses_physical_hardware_before_writing(monkeypatch, tmp_path):
    c = client()
    c._command_socket.values["synthetic"] = "False"
    c.connect_command = lambda **_: c
    sock = c._command_socket
    monkeypatch.setattr(cli, "IntanTCPClient", lambda **_: c)
    assert cli.main(["--demo", "--out", str(tmp_path / "physical")]) == 1
    assert not any(message.startswith((b"Set ", b"Execute ")) for message in sock.sent)


class DemoFixture(SocketPairRHX):
    def __init__(self, markers):
        super().__init__()
        self.drop_trigger = not markers

    def connect_command(self, *args, **kwargs):
        super().connect_command(*args, **kwargs)
        self._command_socket.values["synthetic"] = "True"
        return self

    def run(self, verify=False):
        result = super().run(verify=verify)
        self.clock_origin = time.monotonic()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()
        # A spontaneous Demo spike is independent of any manual trigger.
        self.spike_tx.sendall(SPIKE_MAGIC + b"A-000" + struct.pack("<I", 0) + b"\x40")
        return result

    def play_prepared_protocol(self, protocol, cancel_event=None):
        if self.drop_trigger:
            return super().play_prepared_protocol(protocol, cancel_event)
        # This case tests the report for exactly matched digital records, not
        # OS scheduling accuracy (the command scheduler has separate tests).
        now = int((time.monotonic() - self.clock_origin) * 30000)
        base = (max(self.sample, now) // 128 + 2) * 128
        for pulse in protocol.pulses:
            self.triggers.put((base + round(pulse.time_ms * 30), "f1"))
        return [{"target_ms": p.time_ms, "actual_ms": p.time_ms, "host_send_only": True}
                for p in protocol.pulses]


@pytest.mark.parametrize("markers", [False, True])
def test_demo_report_distinguishes_tcp_reading_from_stim_records(monkeypatch, tmp_path, markers):
    monkeypatch.setattr(cli, "IntanTCPClient", lambda **_: DemoFixture(markers))
    path = tmp_path / "demo"
    assert cli.main(["--demo", "--seconds", "3", "--out", str(path)]) == 0
    report = json.loads((path / "session.json").read_text())
    assert report["status"] == "DEMO_TCP_OK" and report["triggers_sent"] == 3
    assert report["waveform_blocks"] > 0 and report["spike_events"] > 0
    assert report["stim_markers"] == (3 if markers else 0)
    assert report["stim_record"] == ("DIGITAL_RECORD_MATCH_ONLY" if markers else
                                      "NOT_VERIFIED_DEMO_DOES_NOT_SIMULATE_STIM")
    assert report["stop_confirmed"] and not report["physical_delivery_verified"]
    assert not report["classification_tested"] and not report["evoked_response_verified"]
