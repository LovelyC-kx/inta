"""Rebuild saved windows from raw 5001/5002 bytes before replaying decisions."""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from intan_interface.marker_stream import MarkerParser, StimMarkerTracker
from intan_interface.spike_stream import SpikePacketParser
from intan_interface.spike_windows import ProgressMarker
from experiments.openloop_arm import ArmTaskWorker
from experiments.openloop_data import canonical, extract_observation, read_session


def spike_events(path, epoch):
    parser = SpikePacketParser(epoch)
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(65536), b""):
            for event in parser.feed(data):
                yield event
    if parser.buffer or parser.discarded_bytes or parser.malformed_packets:
        raise ValueError("Incomplete/corrupt raw spike journal")


def replay_session(path, *, decoder=None, action_map=None, min_confidence=.8):
    path = Path(path)
    meta, rows = read_session(path)
    contract, acquisition = meta["contract"], meta["acquisition"]
    if decoder is not None:
        decoder.require_contract(contract)
    if action_map is not None and (decoder is None or set(action_map) != set(decoder.classes)):
        raise ValueError("Replay actions require a decoder and its exact class mapping")
    parser = MarkerParser(acquisition["marker_channels"], digital_in=acquisition.get("ttl_bit") is not None)
    tracker = StimMarkerTracker(acquisition["marker_channels"], acquisition["stim_profiles"],
                                ttl_bit=acquisition.get("ttl_bit"))
    markers, progress = [], None
    with (path / "markers.raw").open("rb") as stream:
        for data in iter(lambda: stream.read(65536), b""):
            for block in parser.feed(data):
                found, _ = tracker.feed(block)
                markers.extend(found)
                progress = block.end_sample
    if parser.buffer or tracker.active or progress is None:
        raise ValueError("Incomplete raw marker journal")
    expected_markers = [m for row in rows for m in row.get("markers", [])]
    actual_markers = [asdict(m) for m in sorted(markers, key=lambda m: (m.start_sample, m.native_channel))]
    expected_markers.sort(key=lambda m: (m["start_sample"], m["native_channel"]))
    if canonical(actual_markers) != canonical(expected_markers):
        raise ValueError("Raw STIM journal does not match trial marker assignments")
    events = iter(spike_events(path / "spikes.raw", meta["session_id"]))
    pending = next(events, None)
    predictions, matched = [], 0
    worker = None
    try:
        if action_map is not None:
            worker = ArmTaskWorker(action_map, lambda *_: None, min_confidence=min_confidence)
        for row in rows:
            if "anchor_sample" not in row:
                continue
            trial_events = []
            while pending is not None and pending.sample_index < row["end_sample"]:
                if pending.sample_index >= row["start_sample"]:
                    trial_events.append(pending)
                pending = next(events, None)
            chosen = [m for m in markers if m.start_sample >= row["anchor_sample"] and m.end_sample <= row["end_sample"]]
            observation = {"anchor": row["anchor_sample"], "markers": chosen, "events": trial_events,
                           "watermark": ProgressMarker(progress, meta["session_id"], "stim")}
            _, counts, status = extract_observation(observation, contract, meta["session_id"])
            if status != row["status"] or (counts is None) != (row["counts"] is None):
                raise ValueError("Replayed trial validity differs from saved result")
            if counts is not None:
                np.testing.assert_array_equal(counts, row["counts"])
            matched += 1
            if decoder is not None and status == "VALID":
                prediction = decoder.predict(counts)
                result = {"trial_id": row["trial_id"], "prediction": prediction}
                if worker is not None:
                    result["dispatch"] = worker.submit(row["trial_id"], prediction)
                    worker.wait()  # Replay is not an online latency/busy benchmark.
                predictions.append(result)
        # Exhaust the source to validate trailing bytes and timestamp order.
        for _ in events:
            pass
        if worker is not None:
            worker.wait()
        return {"status": "REPLAY_MATCH", "matched_windows": matched,
                "predictions": predictions, "simulated_source": contract["simulated"],
                "arm_tasks": [] if worker is None else worker.results,
                "physical_task_success": None, "physical_delivery_verified": False}
    finally:
        if worker is not None:
            worker.close()
