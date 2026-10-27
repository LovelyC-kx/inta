"""RHX 5002 threshold-response engineering demo / 事件触发工程演示.

Default: acquisition only; no stimulation commands. --live-stim explicitly
prepares fixed profiles while stopped, then runs continuous acquisition.
This is not a classification or Pong learning experiment. Full native channel
names, hardware sample timestamps and spike IDs are retained in the journal.
"""
import argparse
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime
import json
import math
from numbers import Real
from pathlib import Path
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoders.stimulus_schema import Pulse, StimulusProtocol, MIN_INTER_PULSE_MS
from intan_interface.spike_stream import CHANNEL_NAME, SpikeStreamReader, SpikeStreamError
from intan_interface.tcp_client import IntanTCPClient


class SpikeThresholdPolicy:
    """Hardware event-time policy; each event may produce one candidate."""
    def __init__(self, channels, sample_rate, threshold, window_s, refractory_ms):
        self.channels = tuple(channels)
        if not self.channels or len(set(self.channels)) != len(self.channels) or any(
                not CHANNEL_NAME.fullmatch(n) for n in self.channels):
            raise ValueError("Use unique full native channel names")
        if (any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                for v in (sample_rate, threshold, window_s, refractory_ms)) or
                int(threshold) != threshold or sample_rate <= 0 or threshold < 1 or
                window_s * sample_rate < 1 or refractory_ms < MIN_INTER_PULSE_MS):
            raise ValueError("Invalid sample rate/threshold/window or refractory below protocol minimum")
        self.threshold = threshold
        self.window_samples = int(window_s * sample_rate)
        self.refractory_samples = int(refractory_ms * sample_rate / 1000)
        self.history = defaultdict(deque)
        self.last_candidate = {}

    def observe(self, event):
        if not event.usable or event.native_channel not in self.channels:
            return False
        values = self.history[event.native_channel]
        cutoff = event.sample_index - self.window_samples
        while values and values[0] <= cutoff:
            values.popleft()
        values.append(event.sample_index)
        last = self.last_candidate.get(event.native_channel)
        if len(values) > self.threshold and (last is None or
                event.sample_index - last >= self.refractory_samples):
            values.clear()
            self.last_candidate[event.native_channel] = event.sample_index
            return True
        return False


def run_demo(host, cmd_port, spike_port, n_channels, threshold, window_s,
             refractory_ms, amplitude_ua, phase_us, prefix, dry_run,
             channel_names=None, out_dir=None):
    names = list(channel_names or [f"{prefix}-{ch:03d}" for ch in range(n_channels)])
    if not channel_names and not 1 <= n_channels <= 32:
        raise ValueError("For multiple headstages pass --channels A-000 B-000 ... explicitly")
    # Validate user arguments before opening any laboratory connection.
    SpikeThresholdPolicy(names, 30000, threshold, window_s, refractory_ms)
    policy_channels = names[:8]
    protocols = {name: StimulusProtocol(
        [Pulse(i, 0, amplitude_ua=amplitude_ua, phase_duration_us=phase_us)], 8, 2000)
        for i, name in enumerate(policy_channels)}
    for proto in protocols.values():
        proto.validate_safety()
    output = Path(out_dir or f"data/spike_response_{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=True)
    epoch = uuid.uuid4().hex
    mode = "acquire_only" if dry_run else "live"
    client = IntanTCPClient(host=host, command_port=cmd_port, spike_port=spike_port,
                            execution_mode=mode)
    reader = None
    cancel = threading.Event()
    summary = {"status": "STARTING", "execution_mode": mode, "acquisition_epoch": epoch,
               "channel_names": names, "candidates": 0, "triggers_sent": 0,
               "physical_delivery_verified": False}
    print(f"Mode={mode}; journal={output}; Ctrl+C to stop.")
    raw = (output / "spikes.raw").open("wb")
    events_file = (output / "events.jsonl").open("w", encoding="utf-8")
    exit_code = 0
    try:
        client.connect_command()
        client.connect_spike()
        rate = client.get_sample_rate()
        if not isinstance(rate, int) or rate <= 0:
            raise RuntimeError(f"Invalid RHX sample rate: {rate}")
        policy = SpikeThresholdPolicy(names, rate, threshold, window_s, refractory_ms)
        summary.update(sample_rate_hz=rate,
                       stimulus_mapping={name: f"{prefix}-{i:03d}" for i, name in enumerate(policy_channels)})
        if not dry_run:
            client.stop()
            client.wait_for_run_mode("Stop")
            client.prepare_protocols(protocols.values(), channel_prefix=prefix)
        for name in names:
            client.set(f"{name}.TCPDataOutputEnabledSpike", "True", verify=True)
        if not dry_run:
            client.run()
            client.wait_for_run_mode("Run")
        elif client.get("RunMode") not in ("Return: RunMode Run", "Return: RunMode Record"):
            raise RuntimeError("Acquisition-only requires RHX already in Run/Record")
        reader = SpikeStreamReader(client._spike_socket, acquisition_epoch=epoch,
                                    raw_sink=raw).start()
        summary["status"] = "RUNNING"
        while True:
            for event in reader.drain():
                events_file.write(json.dumps({"kind": "spike", **asdict(event)}) + "\n")
                # Queue age is a processing-lag check, not a hardware clock.
                if time.monotonic_ns() - event.host_rx_ns > 500_000_000:
                    raise SpikeStreamError("Neural events spent more than 500ms in the processing queue")
                if policy.observe(event):
                    reader.raise_if_failed()
                    summary["candidates"] += 1
                    timing = []
                    if not dry_run and event.native_channel in protocols:
                        timing = client.play_prepared_protocol(protocols[event.native_channel], cancel)
                        summary["triggers_sent"] += len(timing)
                    events_file.write(json.dumps({"kind": "candidate",
                        "native_channel": event.native_channel, "sample_index": event.sample_index,
                        "execution_mode": mode, "trigger_sent": bool(timing), "timing": timing}) + "\n")
                    print(f"sample={event.sample_index} channel={event.native_channel} "
                          f"{'trigger sent; marker unverified' if timing else 'candidate only'}")
            time.sleep(.002)
    except KeyboardInterrupt:
        summary["status"] = "STOPPED"
    except Exception as exc:
        summary.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        print(summary["error"], file=sys.stderr)
        exit_code = 1
    finally:
        cancel.set()
        reader_stopped = True
        try:
            if reader is not None:
                try:
                    reader.stop()
                except TimeoutError as exc:
                    reader_stopped = False
                    summary.update(status="FAILED", journal_incomplete=True, error=str(exc))
                    exit_code = 1
                summary.update(raw_bytes=reader.raw_bytes, events_received=reader.events_received)
        finally:
            try:
                if not dry_run and client.is_command_connected:
                    client.stop()
                    client.wait_for_run_mode("Stop")
            except Exception as exc:
                summary.update(status="FAILED", stop_error=str(exc))
                exit_code = 1
            finally:
                client.disconnect()
                if reader_stopped:
                    raw.close()
                events_file.close()
                (output / "session.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cmd-port", type=int, default=5000)
    parser.add_argument("--spike-port", type=int, default=5002)
    parser.add_argument("--n-channels", type=int, default=8)
    parser.add_argument("--channels", nargs="+", help="Full ordered native names, including headstage prefix")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--window", type=float, default=1)
    parser.add_argument("--refractory", type=float, default=500)
    parser.add_argument("--amplitude", type=float, default=10)
    parser.add_argument("--phase-duration", type=float, default=200)
    parser.add_argument("--prefix", choices=list("ABCD"), default="A")
    parser.add_argument("--out-dir")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live-stim", action="store_true", help="Explicitly prepare/upload and trigger real stimulation")
    mode.add_argument("--dry-run", action="store_true", help="Acquisition only (also the default)")
    args = parser.parse_args()
    return run_demo(args.host, args.cmd_port, args.spike_port, args.n_channels,
                    args.threshold, args.window, args.refractory, args.amplitude,
                    args.phase_duration, args.prefix, not args.live_stim,
                    channel_names=args.channels, out_dir=args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
