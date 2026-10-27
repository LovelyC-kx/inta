"""Hardware-time windows with explicit progress evidence and artifact masks.

Never use the latest sparse spike or host receive time as a watermark. The
caller must supply an acquisition-clock progress marker and account for the
relative delivery latency of the marker and spike streams before closing.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .spike_stream import CHANNEL_NAME, SpikeStreamError


@dataclass(frozen=True)
class ProgressMarker:
    sample_index: int
    acquisition_epoch: str
    source: str


@dataclass(frozen=True)
class SpikeWindow:
    events: tuple
    channel_names: tuple
    start_sample: int
    end_sample: int
    valid_samples_per_channel: tuple
    excluded_id_events: int
    masked_events: int
    status: str

    def bin_counts(self, n_bins):
        if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins <= 0:
            raise ValueError("n_bins must be a positive integer")
        if self.status == "INVALID":
            raise SpikeStreamError("A fully masked window cannot be decoded")
        result = np.zeros((n_bins, len(self.channel_names)), dtype=np.float32)
        channels = {name: i for i, name in enumerate(self.channel_names)}
        duration = self.end_sample - self.start_sample
        for event in self.events:
            # Half-open sample windows: out-of-window spikes never get clipped.
            index = (event.sample_index - self.start_sample) * n_bins // duration
            result[index, channels[event.native_channel]] += 1
        return result


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def build_spike_window(events, *, channel_names, start_sample, end_sample,
                       acquisition_epoch, watermark, artifact_intervals=None):
    """Pure, non-consuming extraction: future events remain in the caller's buffer.

    artifact_intervals maps full channel names (or '*' for all channels) to
    half-open sample intervals around EVERY measured stimulus event.
    """
    names = tuple(channel_names)
    if not isinstance(acquisition_epoch, str) or not acquisition_epoch:
        raise ValueError("A nonempty acquisition_epoch is required")
    if not names or len(set(names)) != len(names) or any(
            not isinstance(n, str) or not CHANNEL_NAME.fullmatch(n) for n in names):
        raise ValueError("channel_names must be unique full RHX native names")
    if not _integer(start_sample) or not _integer(end_sample) or not 0 <= start_sample < end_sample:
        raise ValueError("Window bounds must be increasing nonnegative hardware samples")
    if (not isinstance(watermark, ProgressMarker) or
            not _integer(watermark.sample_index) or
            watermark.acquisition_epoch != acquisition_epoch or
            watermark.source not in ("rhx_stream", "ttl", "stim")):
        raise SpikeStreamError("A matching acquisition-clock progress marker is required")
    if watermark.sample_index < end_sample:
        raise SpikeStreamError("Acquisition has not yet advanced beyond this window")
    masks = artifact_intervals or {}
    if set(masks) - set(names) - {"*"}:
        raise ValueError("Artifact masks contain unknown channel names")
    merged = {}
    exposure = []
    for name in names:
        intervals = []
        for start, end in list(masks.get("*", ())) + list(masks.get(name, ())):
            if not _integer(start) or not _integer(end) or start > end:
                raise ValueError("Artifact intervals must be ordered integer sample bounds")
            a, b = max(start_sample, start), min(end_sample, end)
            if a < b:
                intervals.append((a, b))
        joined = []
        for a, b in sorted(intervals):
            if joined and a <= joined[-1][1]:
                joined[-1] = (joined[-1][0], max(b, joined[-1][1]))
            else:
                joined.append((a, b))
        merged[name] = joined
        exposure.append(end_sample - start_sample - sum(b - a for a, b in joined))
    selected = []
    excluded = masked = 0
    for event in events:
        if event.acquisition_epoch != acquisition_epoch:
            raise SpikeStreamError("Events from different acquisition epochs cannot be mixed")
        if event.native_channel not in names or not start_sample <= event.sample_index < end_sample:
            continue
        if not event.usable:
            excluded += 1
        elif any(a <= event.sample_index < b for a, b in merged[event.native_channel]):
            masked += 1
        else:
            selected.append(event)
    status = "INVALID" if not any(exposure) else "VALID" if selected else "SILENT"
    return SpikeWindow(tuple(selected), names, start_sample, end_sample, tuple(exposure),
                       excluded, masked, status)
