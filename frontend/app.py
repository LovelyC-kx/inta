"""
Flask control console for the Intan RHX experiment stack.

The server exposes a browser UI for the operations we currently trust:
status checks, safe single-stimulus verification, recording discovery, and
stim.dat analysis. Long-running hardware actions run in a background thread
and stream progress over Server-Sent Events.
"""

import argparse
import json
import os
import queue
import threading
import time
from functools import wraps
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from flask import Flask, Response, jsonify, render_template, request
except ImportError:
    raise ImportError("Flask not installed. Install with: pip install flask")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.cnn_feature_extractor import CNNFeatureExtractor
from encoders.pipeline import EncodingPipeline
from encoders.stimulus_schema import (
    DEFAULT_INTERPHASE_DELAY_US,
    DEFAULT_PHASE_DURATION_US,
    DEFAULT_POLARITY,
    DEFAULT_WAVEFORM,
    Pulse,
    StimulusProtocol,
)
from experiments.verify_stim_dat import summarize as summarize_stim_dat
from intan_interface.tcp_client import IntanTCPClient


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
DEFAULT_CNN_CHECKPOINT = ROOT / "checkpoints" / "cnn_mnist_20260512_191909.pt"

app = Flask(__name__, template_folder="templates", static_folder="static")

event_queue = queue.Queue(maxsize=2000)
state_lock = threading.Lock()
operation_lock = threading.Lock()
stim_cancel_event = threading.Event()

experiment_state = {
    "status": "idle",
    "operation": None,
    "config": {},
    "trials": [],
    "timing": [],
    "current_trial": 0,
    "total_trials": 0,
    "accuracy": 0.0,
    "latest_protocol": None,
    "latest_recording": None,
    "tcp_command_connected": False,
    "tcp_waveform_connected": False,
    "run_mode": "Unknown",
    "sample_rate": 0,
    "n_channels_streamed": 0,
    "buffer_lag_frames": 0,
    "timestamp_gaps": 0,
    "last_safety_violation": None,
    "total_stims_sent": 0,
    "total_charge_nc": 0.0,
    "dry_run": True,
    "emergency_stop": False,
}


def push_event(event_type, data):
    event = {"type": event_type, "data": data, "time": time.time()}
    try:
        event_queue.put_nowait(event)
    except queue.Full:
        try:
            event_queue.get_nowait()
        except queue.Empty:
            pass
        event_queue.put_nowait(event)


def update_state(**kwargs):
    with state_lock:
        experiment_state.update(kwargs)


def log_event(message, level="info"):
    payload = {
        "message": message,
        "level": level,
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    push_event("operation_log", payload)


def _parse_return_value(response):
    if not response:
        return None
    parts = response.split(maxsplit=2)
    return parts[-1] if parts else response


def _connect_command(host="127.0.0.1", port=5000, timeout=8.0, execution_mode="acquire_only"):
    client = IntanTCPClient(host=host, command_port=port, execution_mode=execution_mode)
    client.connect_command(timeout=timeout)
    return client


def _query_rhx_status(host="127.0.0.1", port=5000):
    client = _connect_command(host, port, timeout=5.0)
    try:
        fields = {}
        for param in [
            "RunMode",
            "Type",
            "SampleRateHertz",
            "Filename.Path",
            "Filename.BaseFilename",
            "Filename.ActiveFileTimestamp",
            "FileFormat",
            "CreateNewDirectory",
            "UploadInProgress",
        ]:
            try:
                fields[param] = client.get(param)
            except Exception as exc:
                fields[param] = f"ERROR: {type(exc).__name__}: {exc}"
        run_mode = _parse_return_value(fields.get("RunMode")) or "Unknown"
        sample_rate = _parse_return_value(fields.get("SampleRateHertz"))
        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            sample_rate = 0
        update_state(
            tcp_command_connected=True,
            run_mode=run_mode,
            sample_rate=sample_rate,
        )
        return fields
    finally:
        client.disconnect()


def _exclusive_hardware_route(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not operation_lock.acquire(blocking=False):
            return jsonify({"ok": False, "error": "Hardware session is busy."}), 409
        try:
            return fn(*args, **kwargs)
        finally:
            operation_lock.release()
    return wrapped


def _load_mnist_image(digit, root):
    from torchvision import datasets, transforms

    mnist = datasets.MNIST(root=str(root), train=True, download=False,
                           transform=transforms.ToTensor())
    for img, label in mnist:
        if int(label) == int(digit):
            return img
    raise ValueError(f"No MNIST image found for digit {digit}")


def _build_protocol(config):
    import torch

    ckpt_path = Path(config.get("cnn_checkpoint") or DEFAULT_CNN_CHECKPOINT)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cnn = CNNFeatureExtractor(
        n_channels=ckpt["n_channels"],
        n_classes=ckpt["n_classes"],
        input_size=ckpt.get("input_size", 28),
    )
    cnn.load_state_dict(ckpt["state_dict"])
    cnn.eval()

    digit = int(config.get("digit", 0))
    encoding = config.get("encoding", "lif")
    img = _load_mnist_image(digit, ROOT / "data" / "mnist")
    pipe = EncodingPipeline(cnn, encoder_type=encoding, trim_warn=False)
    proto = pipe.encode_to_protocol(img)
    proto.validate()
    proto.validate_safety()
    report = proto.safety_report()
    summary = {
        "digit": digit,
        "encoding": encoding,
        "total_pulses": int(proto.total_pulses),
        "pulses_per_channel": [int(x) for x in proto.pulses_per_channel],
        "total_charge_nc": round(float(report["total_charge_nc"]), 3),
        "max_amplitude_ua": round(float(report["max_amplitude_ua"]), 3),
        "max_rate_hz": round(float(report["max_rate_hz"]), 3),
        "duration_ms": float(proto.duration_ms),
        "binary_matrix": proto.to_binary_matrix(10).tolist(),
    }
    return proto, summary


def _build_manual_protocol(config):
    channel_spec = str(config.get("channels", config.get("channel", "0"))).strip()
    if not channel_spec:
        raise ValueError("At least one channel is required.")
    channels = []
    for chunk in channel_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        channels.append(int(chunk))
    if not channels:
        raise ValueError("At least one channel is required.")
    amplitude_ua = float(config.get("amplitude_ua", 10.0))
    phase_duration_us = float(config.get("phase_duration_us", DEFAULT_PHASE_DURATION_US))
    interphase_delay_us = float(config.get("interphase_delay_us", DEFAULT_INTERPHASE_DELAY_US))
    polarity = config.get("polarity", DEFAULT_POLARITY)
    waveform = config.get("waveform", DEFAULT_WAVEFORM)
    trigger_delay_ms = float(config.get("trigger_delay_ms", 0.0))
    pulse_count = int(config.get("pulse_count", 1))
    pulse_interval_ms = float(config.get("pulse_interval_ms", 250.0))
    channel_interval_ms = float(config.get("channel_interval_ms", 0.0))

    if pulse_count < 1:
        raise ValueError("pulse_count must be >= 1")
    if pulse_interval_ms < 0:
        raise ValueError("pulse_interval_ms must be >= 0")
    if channel_interval_ms < 0:
        raise ValueError("channel_interval_ms must be >= 0")

    pulses = []
    for pulse_idx in range(pulse_count):
        base_ms = trigger_delay_ms + pulse_idx * pulse_interval_ms
        for channel_idx, channel in enumerate(channels):
            pulses.append(
                Pulse(
                    channel=channel,
                    time_ms=base_ms + channel_idx * channel_interval_ms,
                    amplitude_ua=amplitude_ua,
                    phase_duration_us=phase_duration_us,
                    interphase_delay_us=interphase_delay_us,
                    polarity=polarity,
                    waveform=waveform,
                )
            )
    pulses.sort(key=lambda p: (p.time_ms, p.channel))
    last_time_ms = max((pulse.time_ms for pulse in pulses), default=0.0)
    proto = StimulusProtocol(
        pulses=pulses,
        n_channels=8,
        duration_ms=max(200.0, last_time_ms + 200.0),
    )
    proto.validate()
    proto.validate_safety()
    report = proto.safety_report()
    summary = {
        "mode": "manual_control",
        "channels": channels,
        "amplitude_ua": amplitude_ua,
        "phase_duration_us": phase_duration_us,
        "interphase_delay_us": interphase_delay_us,
        "polarity": polarity,
        "waveform": waveform,
        "trigger_delay_ms": trigger_delay_ms,
        "pulse_count": pulse_count,
        "pulse_interval_ms": pulse_interval_ms,
        "channel_interval_ms": channel_interval_ms,
        "total_pulses": int(proto.total_pulses),
        "pulses_per_channel": [int(x) for x in proto.pulses_per_channel],
        "total_charge_nc": round(float(report["total_charge_nc"]), 3),
        "max_amplitude_ua": round(float(report["max_amplitude_ua"]), 3),
        "max_rate_hz": round(float(report["max_rate_hz"]), 3),
        "duration_ms": float(proto.duration_ms),
        "binary_matrix": proto.to_binary_matrix(10).tolist(),
    }
    return proto, summary


def _recording_snapshot(data_root):
    data_root.mkdir(parents=True, exist_ok=True)
    return set(os.listdir(data_root))


def _new_recording_dirs(data_root, before):
    after = set(os.listdir(data_root))
    dirs = []
    for name in sorted(after - before):
        path = data_root / name
        if path.is_dir():
            dirs.append(path)
    return dirs


def _file_info(path):
    return {
        "name": path.name,
        "path": str(path.relative_to(ROOT)),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "files": {
            fn: ((path / fn).stat().st_size if (path / fn).exists() else None)
            for fn in ["stim.dat", "amplifier.dat", "time.dat", "info.rhs", "settings.xml"]
        },
        "stim_summary": _read_json(path / "stim_verification.json"),
    }


def _read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _list_recordings(limit=20):
    if not DATA_ROOT.exists():
        return []
    dirs = [p for p in DATA_ROOT.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [_file_info(p) for p in dirs[:limit]]


def _set_file_settings(client):
    client.set("FileFormat", "OneFilePerSignalType")
    time.sleep(0.1)
    client.set("CreateNewDirectory", "True")
    time.sleep(0.1)


def _wait_upload_done(client, timeout_s=10.0):
    deadline = time.perf_counter() + timeout_s
    last = None
    while time.perf_counter() < deadline:
        last = client.get("UploadInProgress")
        log_event(last)
        if "False" in last:
            return last
        time.sleep(0.5)
    raise TimeoutError(f"UploadInProgress did not become False: {last}")


def _set_rhx_run_mode(host, port, action):
    client = _connect_command(host, port, timeout=8.0)
    try:
        action_map = {
            "run": ("Starting Run.", client.run, "Run"),
            "stop": ("Stopping RHX.", client.stop, "Stop"),
            "record": ("Starting Record.", client.record, "Record"),
        }
        if action not in action_map:
            raise ValueError(f"Unsupported RHX action: {action}")
        message, command_fn, mode = action_map[action]
        log_event(message)
        command_fn()
        time.sleep(1.0 if action != "stop" else 2.0)
        run_mode = _parse_return_value(client.wait_for_run_mode(mode))
        update_state(tcp_command_connected=True, run_mode=run_mode)
        return {"action": action, "run_mode": run_mode}
    finally:
        client.disconnect()
        update_state(tcp_command_connected=False)


def _run_single_stim_job(config):
    if not operation_lock.acquire(blocking=False):
        push_event("operation_error", {"message": "Another hardware operation is running."})
        return

    client = None
    stim_cancel_event.clear()
    try:
        update_state(
            status="running",
            operation="single_stim_record",
            dry_run=False,
            emergency_stop=False,
            last_safety_violation=None,
        )
        push_event("operation_start", {"operation": "single_stim_record"})

        host = config.get("host", "127.0.0.1")
        port = int(config.get("command_port", 5000))
        n_channels = int(config.get("n_channels", 64))

        log_event("Building protocol.")
        proto, protocol_summary = _build_protocol(config)
        update_state(latest_protocol=protocol_summary)
        push_event("protocol", protocol_summary)
        log_event(
            "Protocol: {total_pulses} pulse(s), {total_charge_nc} nC, {max_amplitude_ua} uA.".format(
                **protocol_summary
            )
        )

        before = _recording_snapshot(DATA_ROOT)

        log_event(f"Connecting command TCP {host}:{port}.")
        client = _connect_command(host, port, timeout=8.0, execution_mode="live")
        update_state(tcp_command_connected=True)

        try:
            initial_mode = client.get("RunMode")
            log_event(initial_mode)
            update_state(run_mode=_parse_return_value(initial_mode))
        except Exception as exc:
            log_event(f"Initial RunMode read failed: {type(exc).__name__}: {exc}", "warn")

        log_event("Stopping RHX before changing upload/file settings.")
        client.stop()
        client.wait_for_run_mode("Stop")

        log_event("Ensuring OneFilePerSignalType and timestamp directories.")
        _set_file_settings(client)
        for param in ["Filename.Path", "Filename.BaseFilename", "FileFormat", "CreateNewDirectory"]:
            try:
                log_event(client.get(param))
            except Exception as exc:
                log_event(f"Get {param} failed: {type(exc).__name__}: {exc}", "warn")

        log_event("Uploading stimulation parameters while stopped.")
        client.send_protocol_config(proto, upload_to_hardware=True)
        _wait_upload_done(client)

        if experiment_state.get("emergency_stop"):
            raise RuntimeError("Emergency stop requested before Record.")

        log_event("Starting Record.")
        client.record()
        time.sleep(1.5)
        try:
            mode = client.get("RunMode")
            log_event(mode)
            update_state(run_mode=_parse_return_value(mode))
        except Exception as exc:
            log_event(f"RunMode read after Record failed: {type(exc).__name__}: {exc}", "warn")

        log_event("Triggering timed pulses.")
        t0 = time.perf_counter()
        timing = []
        timing = client.play_prepared_protocol(proto, cancel_event=stim_cancel_event)
        for entry in timing:
            push_event("timing", entry)
        time.sleep(float(config.get("post_trigger_s", 2.0)))
        log_event("Stopping Record.")
        client.stop()
        time.sleep(4.0)
        try:
            mode = client.get("RunMode")
            log_event(mode)
            update_state(run_mode=_parse_return_value(mode))
        except Exception as exc:
            log_event(f"RunMode read after Stop failed: {type(exc).__name__}: {exc}", "warn")

        if client:
            client.disconnect()
            client = None
            update_state(tcp_command_connected=False)

        dirs = _new_recording_dirs(DATA_ROOT, before)
        if not dirs:
            dirs = sorted(
                [p for p in DATA_ROOT.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:1]
            log_event("No new directory found by snapshot; using most recent directory.", "warn")
        if not dirs:
            raise RuntimeError("No recording directory found.")

        recording = dirs[-1]
        log_event(f"Recording directory: {recording.relative_to(ROOT)}")
        if not (recording / "stim.dat").exists():
            raise RuntimeError(f"stim.dat not found in {recording}")

        log_event("Analyzing stim.dat.")
        stim_summary = summarize_stim_dat(recording, n_channels=n_channels, plot=False)
        info = _file_info(recording)
        update_state(
            status="stopped",
            operation=None,
            latest_recording=info,
            total_stims_sent=experiment_state.get("total_stims_sent", 0) + protocol_summary["total_pulses"],
            total_charge_nc=experiment_state.get("total_charge_nc", 0.0) + protocol_summary["total_charge_nc"],
            dry_run=True,
        )
        push_event("operation_done", {
            "recording": info,
            "stim_summary": stim_summary,
            "protocol": protocol_summary,
            "timing": timing,
        })
        log_event("Stim verification complete.")
    except Exception as exc:
        update_state(status="stopped", operation=None, dry_run=True)
        push_event("operation_error", {"message": f"{type(exc).__name__}: {exc}"})
        log_event(f"{type(exc).__name__}: {exc}", "error")
    finally:
        if client is not None:
            try:
                client.stop()
                client.wait_for_run_mode("Stop")
            except Exception as stop_error:
                log_event(f"Stop could not be confirmed: {stop_error}", "error")
                update_state(run_mode="Unknown")
            finally:
                client.disconnect()
        update_state(tcp_command_connected=False)
        operation_lock.release()


def _run_manual_single_pulse_job(config):
    if not operation_lock.acquire(blocking=False):
        push_event("operation_error", {"message": "Another hardware operation is running."})
        return

    client = None
    stim_cancel_event.clear()
    try:
        update_state(
            status="running",
            operation="manual_single_pulse",
            dry_run=False,
            emergency_stop=False,
            last_safety_violation=None,
        )
        push_event("operation_start", {"operation": "manual_single_pulse"})

        host = config.get("host", "127.0.0.1")
        port = int(config.get("command_port", 5000))
        n_channels = int(config.get("n_channels", 64))
        record_to_disk = bool(config.get("record_to_disk", False))

        log_event("Building manual single-pulse protocol.")
        proto, protocol_summary = _build_manual_protocol(config)
        update_state(latest_protocol=protocol_summary)
        push_event("protocol", protocol_summary)
        log_event(
            "Manual protocol: {total_pulses} pulse(s), channels {channels}, {amplitude_ua} uA.".format(
                channels=",".join(f"A-{ch:03d}" for ch in protocol_summary["channels"]),
                total_pulses=protocol_summary["total_pulses"],
                amplitude_ua=protocol_summary["amplitude_ua"],
            )
        )

        before = _recording_snapshot(DATA_ROOT) if record_to_disk else None

        log_event(f"Connecting command TCP {host}:{port}.")
        client = _connect_command(host, port, timeout=8.0, execution_mode="live")
        update_state(tcp_command_connected=True)

        log_event("Stopping RHX before upload.")
        client.stop()
        client.wait_for_run_mode("Stop")

        if record_to_disk:
            log_event("Ensuring OneFilePerSignalType and timestamp directories.")
            _set_file_settings(client)

        log_event("Uploading single-pulse parameters.")
        client.send_protocol_config(proto, upload_to_hardware=True)
        _wait_upload_done(client)

        if experiment_state.get("emergency_stop"):
            raise RuntimeError("Emergency stop requested before trigger.")

        if record_to_disk:
            log_event("Starting Record.")
            client.record()
        else:
            log_event("Starting Run.")
            client.run()
        time.sleep(1.0)

        log_event("Triggering manual pulse sequence.")
        t0 = time.perf_counter()
        timing = []
        timing = client.play_prepared_protocol(proto, cancel_event=stim_cancel_event)
        for entry in timing:
            push_event("timing", entry)
        time.sleep(float(config.get("post_trigger_s", 1.0)))

        log_event("Stopping RHX.")
        client.stop()
        time.sleep(4.0)

        result_payload = {
            "protocol": protocol_summary,
            "timing": timing,
            "recording": None,
            "stim_summary": None,
        }

        if record_to_disk:
            dirs = _new_recording_dirs(DATA_ROOT, before)
            if not dirs:
                dirs = sorted(
                    [p for p in DATA_ROOT.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[:1]
                log_event("No new directory found by snapshot; using most recent directory.", "warn")
            if not dirs:
                raise RuntimeError("No recording directory found.")

            recording = dirs[-1]
            log_event(f"Recording directory: {recording.relative_to(ROOT)}")
            if not (recording / "stim.dat").exists():
                raise RuntimeError(f"stim.dat not found in {recording}")

            log_event("Analyzing stim.dat.")
            stim_summary = summarize_stim_dat(recording, n_channels=n_channels, plot=False)
            info = _file_info(recording)
            result_payload["recording"] = info
            result_payload["stim_summary"] = stim_summary
            latest_recording = info
        else:
            latest_recording = experiment_state.get("latest_recording")

        update_state(
            status="stopped",
            operation=None,
            latest_recording=latest_recording,
            total_stims_sent=experiment_state.get("total_stims_sent", 0) + 1,
            total_charge_nc=experiment_state.get("total_charge_nc", 0.0) + protocol_summary["total_charge_nc"],
            dry_run=True,
        )
        push_event("operation_done", result_payload)
        log_event("Manual single pulse complete.")
    except Exception as exc:
        update_state(status="stopped", operation=None, dry_run=True)
        push_event("operation_error", {"message": f"{type(exc).__name__}: {exc}"})
        log_event(f"{type(exc).__name__}: {exc}", "error")
    finally:
        if client is not None:
            try:
                client.stop()
                client.wait_for_run_mode("Stop")
            except Exception as stop_error:
                log_event(f"Stop could not be confirmed: {stop_error}", "error")
                update_state(run_mode="Unknown")
            finally:
                client.disconnect()
        update_state(tcp_command_connected=False)
        operation_lock.release()


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                event = event_queue.get(timeout=1.0)
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
            except queue.Empty:
                yield "event: heartbeat\ndata: {}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/state")
def get_state():
    with state_lock:
        return jsonify(dict(experiment_state))


@app.route("/api/system_status")
def get_system_status():
    with state_lock:
        return jsonify({
            "status": experiment_state["status"],
            "operation": experiment_state["operation"],
            "tcp_command_connected": experiment_state["tcp_command_connected"],
            "tcp_waveform_connected": experiment_state["tcp_waveform_connected"],
            "run_mode": experiment_state["run_mode"],
            "sample_rate": experiment_state["sample_rate"],
            "n_channels_streamed": experiment_state["n_channels_streamed"],
            "buffer_lag_frames": experiment_state["buffer_lag_frames"],
            "timestamp_gaps": experiment_state["timestamp_gaps"],
            "total_stims_sent": experiment_state["total_stims_sent"],
            "total_charge_nc": experiment_state["total_charge_nc"],
            "dry_run": experiment_state["dry_run"],
            "last_safety_violation": experiment_state["last_safety_violation"],
            "emergency_stop": experiment_state["emergency_stop"],
            "latest_protocol": experiment_state["latest_protocol"],
            "latest_recording": experiment_state["latest_recording"],
        })


@app.route("/api/rhx/status", methods=["POST"])
@_exclusive_hardware_route
def rhx_status():
    payload = request.json or {}
    try:
        fields = _query_rhx_status(
            payload.get("host", "127.0.0.1"),
            int(payload.get("command_port", 5000)),
        )
        return jsonify({"ok": True, "fields": fields})
    except Exception as exc:
        update_state(tcp_command_connected=False, run_mode="Unknown")
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 503


@app.route("/api/rhx/control", methods=["POST"])
@_exclusive_hardware_route
def rhx_control():
    payload = request.json or {}
    try:
        result = _set_rhx_run_mode(
            payload.get("host", "127.0.0.1"),
            int(payload.get("command_port", 5000)),
            str(payload.get("action", "")).strip().lower(),
        )
        return jsonify({"ok": True, **result})
    except Exception as exc:
        update_state(tcp_command_connected=False)
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 400


@app.route("/api/protocol/preview", methods=["POST"])
def protocol_preview():
    try:
        _, summary = _build_protocol(request.json or {})
        return jsonify({"ok": True, "protocol": summary})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 400


@app.route("/api/stim/manual_preview", methods=["POST"])
def manual_protocol_preview():
    try:
        _, summary = _build_manual_protocol(request.json or {})
        return jsonify({"ok": True, "protocol": summary})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 400


@app.route("/api/stim/single_record", methods=["POST"])
def single_record_stim():
    payload = request.json or {}
    if payload.get("confirm_text") != "ENABLE_STIM":
        return jsonify({"ok": False, "error": "Type ENABLE_STIM to allow live stimulation."}), 400
    t = threading.Thread(target=_run_single_stim_job, args=(payload,), daemon=True)
    t.start()
    return jsonify({"ok": True, "status": "started"})


@app.route("/api/stim/manual_single_pulse", methods=["POST"])
def manual_single_pulse():
    payload = request.json or {}
    if payload.get("confirm_text") != "ENABLE_STIM":
        return jsonify({"ok": False, "error": "Type ENABLE_STIM to allow live stimulation."}), 400
    t = threading.Thread(target=_run_manual_single_pulse_job, args=(payload,), daemon=True)
    t.start()
    return jsonify({"ok": True, "status": "started"})


@app.route("/api/recordings")
def recordings():
    limit = int(request.args.get("limit", 20))
    return jsonify({"recordings": _list_recordings(limit)})


@app.route("/api/recordings/<name>/verify", methods=["POST"])
def verify_recording(name):
    path = DATA_ROOT / name
    if not path.exists() or not path.is_dir():
        return jsonify({"ok": False, "error": "Recording directory not found."}), 404
    payload = request.json or {}
    try:
        summary = summarize_stim_dat(
            path,
            n_channels=int(payload.get("n_channels", 64)),
            plot=bool(payload.get("plot", False)),
        )
        return jsonify({"ok": True, "summary": summary, "recording": _file_info(path)})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 400


@app.route("/api/start", methods=["POST"])
def start_experiment():
    config = request.json or {}
    update_state(config=config, status="pending_start")
    push_event("control", {"action": "start", "config": config})
    return jsonify({"status": "start_requested"})


@app.route("/api/stop", methods=["POST"])
def stop_experiment():
    update_state(status="stop_requested")
    push_event("control", {"action": "stop"})
    return jsonify({"status": "stop_requested"})


@app.route("/api/emergency_stop", methods=["POST"])
def emergency_stop():
    stim_cancel_event.set()
    update_state(emergency_stop=True, status="emergency_stopped")
    push_event("control", {"action": "emergency_stop"})
    log_event("Emergency stop requested.", "error")
    return jsonify({"status": "emergency_stopped"})


@app.route("/api/trials")
def get_trials():
    with state_lock:
        return jsonify(experiment_state["trials"][-50:])


@app.route("/api/timing")
def get_timing():
    with state_lock:
        return jsonify(experiment_state["timing"][-200:])


class DashboardBridge:
    """Compatibility bridge used by existing realtime scripts."""

    @staticmethod
    def on_experiment_start(config, total_trials):
        update_state(status="running", config=config, total_trials=total_trials, trials=[], timing=[])
        push_event("experiment_start", {"total_trials": total_trials, "config": config})

    @staticmethod
    def on_trial_complete(result):
        with state_lock:
            experiment_state["trials"].append(result)
            experiment_state["current_trial"] = result.get("trial", 0) + 1
            n_correct = sum(1 for t in experiment_state["trials"] if t.get("is_correct"))
            n_total = len(experiment_state["trials"])
            experiment_state["accuracy"] = n_correct / n_total if n_total else 0.0
        push_event("trial", result)

    @staticmethod
    def on_timing(timing_entry):
        with state_lock:
            experiment_state["timing"].append(timing_entry)
        push_event("timing", timing_entry)

    @staticmethod
    def on_spikes(spike_raster_summary):
        push_event("spikes", spike_raster_summary)

    @staticmethod
    def on_protocol(protocol_summary):
        update_state(latest_protocol=protocol_summary)
        push_event("protocol", protocol_summary)

    @staticmethod
    def on_experiment_end(summary):
        update_state(status="stopped")
        push_event("experiment_end", summary)

    @staticmethod
    def should_stop():
        with state_lock:
            return experiment_state["status"] in ("stop_requested", "emergency_stopped")

    @staticmethod
    def should_emergency_stop():
        with state_lock:
            return bool(experiment_state["emergency_stop"])

    @staticmethod
    def update_system_status(**kwargs):
        update_state(**{k: v for k, v in kwargs.items() if k in experiment_state})

    @staticmethod
    def report_safety_violation(message):
        update_state(last_safety_violation=message)
        push_event("safety_violation", {"message": message, "time": datetime.now().isoformat()})


def run_server(port=8050, debug=False):
    print(f"Dashboard: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    run_server(port=args.port, debug=True)
