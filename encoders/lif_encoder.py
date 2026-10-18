"""
LIF Neuron-based Spike Encoder.

Converts continuous signals into spike trains by driving a Leaky Integrate-and-Fire
(LIF) neuron with the input signal as injected current.

Reference: Donati/Valle et al. (2026), Nature Communications
    - Used for encoding ENG signals before SNN decoding
    - Extracts features related to power content and temporal variation
    - Biologically inspired, similar to cochlear processing

This module provides two use cases:
1. Encoding analog signals (e.g., from MEA recordings) into spike trains
2. Encoding image feature values into spike trains for stimulation

Key parameters from Donati et al.:
    - tau = 10 ms (membrane time constant)
    - threshold = 350 mV (spike threshold, after signal scaling)
    - U_reset = 0 mV (reset potential)
    - refractory_period = 1 ms
"""

import numpy as np
from dataclasses import dataclass
from .stimulus_schema import Pulse, StimulusProtocol, MIN_INTER_PULSE_MS


@dataclass
class SpikeTrainOutput:
    """Output of LIF encoding."""

    spike_times: list  # List of arrays, one per channel: spike times in ms
    spike_trains_binned: np.ndarray  # (n_channels, n_time_bins) binned counts
    n_channels: int
    duration_ms: float
    bin_size_ms: float


class LIFEncoder:
    """
    Encodes continuous signals into spike trains via LIF neuron dynamics.

    The membrane potential follows:
        dU/dt = (-U[t-1] + U_drive[t]) / tau

    When U >= threshold, a spike is emitted and U resets to U_reset.

    For image encoding: pixel values are converted to constant driving currents
    over a specified duration, producing spike trains whose rates reflect
    the input intensity.
    """

    def __init__(
        self,
        tau_ms=10.0,
        threshold=350.0,
        u_reset=0.0,
        refractory_ms=1.0,
        dt_ms=0.1,
        bin_size_ms=2.5,
    ):
        """
        Args:
            tau_ms: Membrane time constant in ms.
            threshold: Spike threshold (in same units as input after scaling).
            u_reset: Reset potential after spike.
            refractory_ms: Refractory period in ms.
            dt_ms: Simulation time step in ms.
            bin_size_ms: Bin size for binned spike train output.
                         Donati et al. used 2.5 ms.
        """
        self.tau_ms = tau_ms
        self.threshold = threshold
        self.u_reset = u_reset
        self.refractory_ms = refractory_ms
        self.dt_ms = dt_ms
        self.bin_size_ms = bin_size_ms

    def _simulate_single_channel(self, drive_signal, duration_ms):
        """
        Simulate one LIF neuron driven by a continuous input signal.

        Args:
            drive_signal: 1D array of driving current values,
                          one per time step.
            duration_ms: Total simulation duration.

        Returns:
            spike_times: Array of spike onset times in ms.
        """
        n_steps = len(drive_signal)
        u = self.u_reset
        spike_times = []
        refractory_counter = 0.0

        for i in range(n_steps):
            if refractory_counter > 0:
                refractory_counter -= self.dt_ms
                continue

            # LIF dynamics: Euler integration
            du = (-u + drive_signal[i]) / self.tau_ms * self.dt_ms
            u += du

            # Spike generation
            if u >= self.threshold:
                spike_times.append(i * self.dt_ms)
                u = self.u_reset
                refractory_counter = self.refractory_ms

        return np.array(spike_times)

    def encode_signal(self, signals, duration_ms):
        """
        Encode multi-channel continuous signals into spike trains.

        This is the primary method from Donati et al.: raw analog signal
        is rectified, z-score normalized, and fed as drive current.

        Args:
            signals: (n_channels, n_samples) array of analog signals.
            duration_ms: Duration of the recording in ms.

        Returns:
            SpikeTrainOutput with spike times and binned representation.
        """
        n_channels, n_samples = signals.shape
        n_steps = int(duration_ms / self.dt_ms)

        all_spike_times = []

        for ch in range(n_channels):
            # Full-wave rectification (Donati et al.)
            rectified = np.abs(signals[ch])

            # Z-score normalization
            mean_val = np.mean(rectified)
            std_val = np.std(rectified)
            if std_val > 1e-10:
                normalized = (rectified - mean_val) / std_val
            else:
                normalized = np.zeros_like(rectified)

            # Scale so the driving current can sustain membrane potential
            # above threshold. After z-score and rectification (clip to >0),
            # positive values average ~0.8 SD. For the LIF to spike,
            # the drive must exceed threshold sustainably. We scale so
            # that moderate positive fluctuations (~1 SD) can reach threshold
            # through temporal integration over several tau periods.
            # Factor of 2.0 ensures values at +0.5 SD sustain ~threshold-level drive.
            scaled = np.clip(normalized, 0, None) * self.threshold * 2.0

            # Resample to simulation time step if needed
            if len(scaled) != n_steps:
                indices = np.linspace(0, len(scaled) - 1, n_steps).astype(int)
                drive = scaled[indices]
            else:
                drive = scaled

            spike_times = self._simulate_single_channel(drive, duration_ms)
            all_spike_times.append(spike_times)

        # Build binned representation
        binned = self._bin_spike_trains(all_spike_times, n_channels, duration_ms)

        return SpikeTrainOutput(
            spike_times=all_spike_times,
            spike_trains_binned=binned,
            n_channels=n_channels,
            duration_ms=duration_ms,
            bin_size_ms=self.bin_size_ms,
        )

    def encode_features(self, feature_maps, duration_ms=2000.0):
        """
        Encode image feature maps into spike trains for stimulation.

        Each pixel value in the feature map becomes a constant driving
        current for a corresponding time window. Higher values produce
        higher spike rates, creating a rate-coded representation.

        Args:
            feature_maps: (n_channels, H, W) normalized feature maps in [0, 1].
            duration_ms: Total encoding duration.

        Returns:
            SpikeTrainOutput with spike times and binned representation.
        """
        n_channels, h, w = feature_maps.shape
        n_pixels = h * w
        n_steps = int(duration_ms / self.dt_ms)
        steps_per_pixel = max(1, n_steps // n_pixels)

        all_spike_times = []

        for ch in range(n_channels):
            pixels = feature_maps[ch].flatten()

            # Build drive signal: each pixel drives the neuron for a sub-window
            drive = np.zeros(n_steps)
            for px_idx, px_val in enumerate(pixels):
                start = px_idx * steps_per_pixel
                end = min(start + steps_per_pixel, n_steps)
                # Scale pixel value to drive current near threshold
                drive[start:end] = px_val * self.threshold * 1.2

            spike_times = self._simulate_single_channel(drive, duration_ms)
            all_spike_times.append(spike_times)

        binned = self._bin_spike_trains(all_spike_times, n_channels, duration_ms)

        return SpikeTrainOutput(
            spike_times=all_spike_times,
            spike_trains_binned=binned,
            n_channels=n_channels,
            duration_ms=duration_ms,
            bin_size_ms=self.bin_size_ms,
        )

    @staticmethod
    def spike_output_to_protocol(output):
        """
        Convert a SpikeTrainOutput to the unified StimulusProtocol.

        Since LIF produces many more spikes than hardware can deliver
        (min 200 ms inter-pulse -> max 10 pulses/channel in 2 s),
        we subsample: keep the first spike, then the next spike that is
        >= MIN_INTER_PULSE_MS later, etc.
        """
        all_pulses = []
        for ch, times in enumerate(output.spike_times):
            if len(times) == 0:
                continue
            sorted_times = np.sort(times)
            last_kept = -MIN_INTER_PULSE_MS - 1
            for t in sorted_times:
                if t - last_kept >= MIN_INTER_PULSE_MS and t <= output.duration_ms:
                    all_pulses.append(Pulse(channel=ch, time_ms=float(t)))
                    last_kept = t

        all_pulses.sort(key=lambda p: (p.time_ms, p.channel))
        proto = StimulusProtocol(
            pulses=all_pulses,
            n_channels=output.n_channels,
            duration_ms=output.duration_ms,
        )
        proto.validate()
        return proto

    def _bin_spike_trains(self, spike_times_list, n_channels, duration_ms):
        """
        Bin spike trains into fixed time windows.

        Args:
            spike_times_list: List of spike time arrays per channel.
            n_channels: Number of channels.
            duration_ms: Total duration.

        Returns:
            (n_channels, n_bins) array of spike counts per bin.
        """
        n_bins = int(np.ceil(duration_ms / self.bin_size_ms))
        binned = np.zeros((n_channels, n_bins), dtype=np.float32)

        for ch, times in enumerate(spike_times_list):
            if len(times) == 0:
                continue
            bin_indices = np.minimum(
                (times / self.bin_size_ms).astype(int), n_bins - 1
            )
            for idx in bin_indices:
                binned[ch, idx] += 1

        return binned
