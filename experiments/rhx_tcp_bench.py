"""Connect to the actual RHX RHS Demo app; never substitute a fake server.

Demo-only by design. Official synthetic controller ignores manual stimulation
and generates zero stim-on/off words. Command/control and acquisition can be
checked, but zero markers do not prove stimulation or an evoked response.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoders.stimulus_schema import Pulse, StimulusProtocol
from experiments.openloop_backend import check_markers
from intan_interface.hardware_profile import RHSProfile
from intan_interface.marker_stream import MarkerParser, MarkerStreamReader, StimMarkerTracker
from intan_interface.spike_stream import SpikeStreamReader
from intan_interface.tcp_client import IntanTCPClient


def bench(args):
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "STARTING", "source": "ACTUAL_RHX_TCP_CONNECTION",
              "connected": False, "physical_delivery_verified": False,
              "evoked_response_verified": False, "classification_tested": False,
              "triggers_sent": 0, "stim_markers": 0, "waveform_blocks": 0,
              "spike_events": 0, "stop_confirmed": False}
    c = IntanTCPClient(host=args.host, command_port=args.cmd_port,
                       waveform_port=args.wave_port, spike_port=args.spike_port,
                       execution_mode="live")
    readers, streams, markers, errors = [], [], [], []
    cancel, worker = threading.Event(), None
    started = False
    epoch = uuid.uuid4().hex
    protocol = StimulusProtocol([Pulse(0, t, 10, 500) for t in (0, 500, 1000)], 1, 1500)
    log = (output / "events.jsonl").open("x", encoding="utf-8")

    def emit(kind, value):
        log.write(json.dumps({"kind": kind, "host_ns": time.monotonic_ns(), "value": value}) + "\n")
        log.flush()

    def set_checked(parameter, value):
        c.set(parameter, value)
        if c._get_value(parameter).casefold() != str(value).casefold():
            raise RuntimeError(f"RHX did not apply {parameter}={value}")

    def drain():
        for block in marker_reader.drain():
            found, _ = tracker.feed(block)
            markers.extend(found)
            report["waveform_blocks"] += 1
            report["last_sample"] = block.end_sample
        for event in spike_reader.drain():
            if event.native_channel not in names:
                raise RuntimeError("Unexpected spike channel in declared layout")
            report["spike_events"] += 1
            emit("spike", asdict(event))

    try:
        report["phase"] = "CONNECT_COMMAND"
        print(f"CONNECTING to actual RHX {args.host}:{args.cmd_port}", flush=True)
        c.connect_command(timeout=3)
        report["connected"] = True
        report["identity"] = {p: c._get_value(p) for p in ("Version", "Type", "Synthetic", "Playback")}
        print(json.dumps(report["identity"]), flush=True)
        # This guard precedes EVERY Set/Upload/Trigger. Never actuate hardware
        # through a command advertised as a Demo test.
        if report["identity"]["Synthetic"].casefold() != "true":
            raise ValueError("Demo-only bench refuses physical hardware before any configuration write")
        hardware = RHSProfile.read(c._get_value)
        if c.get("RunMode") != "Return: RunMode Stop":
            raise ValueError("Click Stop in RHX before starting this owned bench session")
        if hardware.sample_rate_hz != 30000:
            raise ValueError("For this explicit 500us bench profile select 30 kS/s in RHX")
        report["hardware"] = hardware.as_dict()
        print("PREPARING: configure and read back three A-000 pulse commands (may take several seconds)", flush=True)
        report["phase"] = "PREPARE_STIMULATION"
        c.prepare_protocols([protocol])
        report["prepared_session"] = c.prepared_session
        c.send_command("Execute ClearAllDataOutputs", expect_response=False)
        set_checked("TCPNumberDataBlocksPerWrite", 1)
        names = ["A-000", "A-001"]
        if int(c._get_value("A.NumberAmplifierChannels")) < 2:
            raise ValueError("RHS Demo must expose A-000 and A-001")
        set_checked("A-000.TCPDataOutputEnabledStim", "True")
        for name in names:
            set_checked(f"{name}.TCPDataOutputEnabledSpike", "True")
        for port in "ABCD":
            for index in range(int(c._get_value(f"{port}.NumberAmplifierChannels"))):
                name = f"{port}-{index:03d}"
                for band in ("", "Low", "High", "DC", "Stim", "Spike"):
                    expected = (band == "Stim" and name == "A-000") or (band == "Spike" and name in names)
                    if c._get_value(f"{name}.TCPDataOutputEnabled{band}").casefold() != str(expected).casefold():
                        raise RuntimeError("Unexpected TCP data layout")
        c.connect_waveform(timeout=3)
        c.connect_spike(timeout=3)
        print(f"CONNECTED: actual TCP ports {args.cmd_port}/{args.wave_port}/{args.spike_port}; beginning Run", flush=True)
        streams = [(output / name).open("xb", buffering=0) for name in ("markers.raw", "spikes.raw")]
        parser = MarkerParser(["A-000"])
        profiles = {"A-000": {**hardware.validate_pulse(protocol.pulses[0]),
                              "negative_first": True, "waveform": "Biphasic"}}
        tracker = StimMarkerTracker(["A-000"], profiles)
        marker_reader = MarkerStreamReader(c._waveform_socket, parser, streams[0])
        spike_reader = SpikeStreamReader(c._spike_socket, acquisition_epoch=epoch, raw_sink=streams[1])
        readers = [marker_reader, spike_reader]
        for reader in readers:
            reader.start()
        started = True
        report["phase"] = "ENTER_RUN"
        def wait_for_stream():
            until = time.monotonic() + 10
            while not report["waveform_blocks"]:
                drain()
                if time.monotonic() >= until:
                    raise TimeoutError("No 5001 acquisition data after Run request")
                time.sleep(.01)
        c.run(verify=True, startup_barrier=wait_for_stream)
        report["phase"] = "TRIGGER_AND_RECEIVE"
        print("RUN CONFIRMED: sending three trigger commands and receiving data", flush=True)

        def trigger():
            try:
                report["host_trigger_timing"] = c.play_prepared_protocol(protocol, cancel)
                # Fence the write-only trigger commands and surface RHX errors.
                if c.get("RunMode") != "Return: RunMode Run":
                    raise RuntimeError("RHX left Run during trigger commands")
                report["triggers_sent"] = len(report["host_trigger_timing"])
            except BaseException as exc:
                errors.append(exc)
                cancel.set()

        worker = threading.Thread(target=trigger, name="bench-trigger", daemon=True)
        worker.start()
        deadline, next_print = time.monotonic() + args.seconds, time.monotonic()
        while time.monotonic() < deadline:
            if errors:
                raise RuntimeError(str(errors[0])) from errors[0]
            drain()
            if time.monotonic() >= next_print:
                print(f"RECEIVING blocks={report['waveform_blocks']} spikes={report['spike_events']} "
                      f"stim_markers={len(markers)} triggers_confirmed={report['triggers_sent']}", flush=True)
                next_print = time.monotonic() + 1
            cancel.wait(.01)
        worker.join(3)
        if worker.is_alive() or errors:
            raise RuntimeError("Trigger worker did not finish cleanly")
        if not report["waveform_blocks"] or report["triggers_sent"] != 3:
            raise RuntimeError("No waveform clock received or trigger sequence incomplete")
        report["status"] = "DEMO_TCP_OK" if report["spike_events"] else "DEMO_TCP_NO_SPIKES"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        cancel.set()
        if worker is not None:
            worker.join(3)
        try:
            if started:
                c.stop(verify=True)
                report["stop_confirmed"] = True
                # Allow already buffered data to reach both readers after Stop.
                until = time.monotonic() + .2
                while time.monotonic() < until:
                    drain()
                    time.sleep(.01)
        except Exception as exc:
            report.update(status="FAILED", cleanup_error=str(exc))
        joined = True
        for reader in readers:
            try:
                reader.stop()
            except Exception as exc:
                joined = False
                report.update(status="FAILED", reader_error=str(exc))
        try:
            if started and joined:
                drain()
                if parser.buffer or spike_reader.parser.buffer or tracker.active:
                    raise RuntimeError("Incomplete packet or stimulation phase at end of capture")
                report["stim_markers"] = len(markers)
                report["markers"] = [asdict(m) for m in markers]
                if markers:
                    check_markers(markers, protocol, "A", hardware.sample_rate_hz, 100)
                    report["stim_record"] = "DIGITAL_RECORD_MATCH_ONLY"
                else:
                    report["stim_record"] = "NOT_VERIFIED_DEMO_DOES_NOT_SIMULATE_STIM"
                if report["status"] in ("DEMO_TCP_OK", "DEMO_TCP_NO_SPIKES"):
                    report["status"] = "DEMO_TCP_OK" if report["spike_events"] else "DEMO_TCP_NO_SPIKES"
        except Exception as exc:
            report.update(status="FAILED", integrity_error=str(exc))
        c.disconnect()
        if joined:
            for stream in streams:
                stream.close()
        report["raw_bytes"] = {p.name: p.stat().st_size for p in output.glob("*.raw")}
        emit("summary", report)
        log.close()
        (output / "session.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = ("status", "phase", "connected", "triggers_sent", "waveform_blocks", "spike_events",
            "stim_markers", "stim_record", "stop_confirmed", "raw_bytes",
            "physical_delivery_verified", "evoked_response_verified", "classification_tested",
            "error", "cleanup_error", "reader_error", "integrity_error")
    print(json.dumps({k: report[k] for k in keys if k in report}, indent=2), flush=True)
    print(f"Report: {output / 'session.json'}", flush=True)
    return 0 if report["status"] == "DEMO_TCP_OK" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True,
                        help="Require actual RHX Synthetic=True; refuse physical hardware")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cmd-port", type=int, default=5000)
    parser.add_argument("--wave-port", type=int, default=5001)
    parser.add_argument("--spike-port", type=int, default=5002)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--out", required=True, help="New directory for actual received bytes and report")
    args = parser.parse_args(argv)
    if not 3 <= args.seconds <= 300:
        parser.error("--seconds must be finite and between 3 and 300")
    try:
        return bench(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
