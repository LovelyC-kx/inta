"""
Rate Encoder (baseline).

Simple Poisson-process-based encoding where pixel intensity
determines the firing rate. Included as a baseline to compare
against more sophisticated encoding methods.

Warning from Wang et al. (2025): High-frequency stimulation
from rate encoding can diminish or eliminate BNN responses and
may lead to network dysfunction. Use with caution.
"""

import numpy as np
from dataclasses import dataclass
from .stimulus_schema import Pulse, StimulusProtocol, MIN_INTER_PULSE_MS


@dataclass
class RateEncodedOutput:
    """Output of rate encoding."""

    spike_trains_binned: np.ndarray  # (n_channels, n_time_bins)
    spike_times: list  # List of arrays per channel
    n_channels: int
    duration_ms: float
    bin_size_ms: float
    mean_rate_hz: float  # Average firing rate across channels


class RateEncoder:
    """
    Poisson rate encoder: pixel intensity -> firing probability per time step.

    rate_i = f_max * pixel_value_i
    P(spike in dt) = rate_i * dt
    """

    def __init__(
        self,
        f_max_hz=50.0,
        dt_ms=1.0,
        bin_size_ms=2.5,
        min_interval_ms=0.0,
        seed=None,
    ):
        """
        Args:
            f_max_hz: Maximum firing rate for pixel value = 1.0.
                      Keep this moderate to avoid BNN overactivation.
            dt_ms: Simulation time step.
            bin_size_ms: Bin size for output.
            min_interval_ms: Minimum inter-spike interval (0 = no constraint).
            seed: Random seed for reproducibility.
        """
        self.f_max_hz = f_max_hz
        self.dt_ms = dt_ms
        self.bin_size_ms = bin_size_ms
        self.min_interval_ms = min_interval_ms
        self.rng = np.random.default_rng(seed)

    def encode_features(self, feature_maps, duration_ms=2000.0):
        """
        Encode feature maps using Poisson rate coding.

        Args:
            feature_maps: (n_channels, H, W) in [0, 1].
            duration_ms: Encoding duration.

        Returns:
            RateEncodedOutput.
        """
        n_channels, h, w = feature_maps.shape
        n_steps = int(duration_ms / self.dt_ms)
        n_pixels = h * w
        steps_per_pixel = max(1, n_steps // n_pixels)

        all_spike_times = []
        total_spikes = 0

        for ch in range(n_channels):
            pixels = feature_maps[ch].flatten()
            spike_times = []

            for px_idx, px_val in enumerate(pixels):
                rate_hz = self.f_max_hz * px_val
                prob_per_step = rate_hz * (self.dt_ms / 1000.0)
                t_start = px_idx * steps_per_pixel * self.dt_ms

                last_spike_time = -self.min_interval_ms - 1

                for step in range(steps_per_pixel):
                    t = t_start + step * self.dt_ms
                    if t - last_spike_time < self.min_interval_ms:
                        continue
                    if self.rng.random() < prob_per_step:
                        spike_times.append(t)
                        last_spike_time = t

            spike_arr = np.array(spike_times)
            all_spike_times.append(spike_arr)
            total_spikes += len(spike_arr)

        # Bin
        n_bins = int(np.ceil(duration_ms / self.bin_size_ms))
        binned = np.zeros((n_channels, n_bins), dtype=np.float32)
        for ch, times in enumerate(all_spike_times):
            if len(times) == 0:
                continue
            indices = np.minimum(
                (times / self.bin_size_ms).astype(int), n_bins - 1
            )
            for idx in indices:
                binned[ch, idx] += 1

        mean_rate = total_spikes / (n_channels * duration_ms / 1000.0)

        return RateEncodedOutput(
            spike_trains_binned=binned,
            spike_times=all_spike_times,
            n_channels=n_channels,
            duration_ms=duration_ms,
            bin_size_ms=self.bin_size_ms,
            mean_rate_hz=mean_rate,
        )

    @staticmethod
    def rate_output_to_protocol(output):
        """
        Convert RateEncodedOutput to the unified StimulusProtocol.

        Same subsampling logic as LIF: keep spikes that are
        >= MIN_INTER_PULSE_MS apart per channel.
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
