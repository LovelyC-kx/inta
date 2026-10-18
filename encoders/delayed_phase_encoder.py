"""
Improved Delayed Phase Encoding.

Converts normalized feature maps into spatiotemporal pulse sequences
suitable for MEA stimulation of biological neural networks.

Reference: Wang et al. (2025), Section 2.2.1

Pipeline:
    Feature map pixels -> firing timestamps (via arctan function)
    -> temporal alignment (via SMO function)
    -> compression into pulse sequences
    -> one sequence per stimulation electrode

Key constraints from biological experiments (RHS current-mode):
    - Total stimulation duration: <= 2000 ms
    - Minimum inter-pulse interval: >= 200 ms (network recovery)
    - Each pulse: biphasic current, default 10 µA, 500 µs per phase, cathodic-first
"""

import numpy as np
from dataclasses import dataclass
from .stimulus_schema import Pulse, StimulusProtocol, MIN_INTER_PULSE_MS


@dataclass
class PulseEvent:
    """A single stimulation pulse event."""

    channel: int  # Stimulation electrode index (0-based)
    time_ms: float  # Pulse onset time in milliseconds
    amplitude_ua: float = 10.0  # Biphasic pulse amplitude in µA (RHS current mode)
    phase_duration_us: float = 500.0  # Duration per phase


@dataclass
class StimulusSequence:
    """Complete stimulus sequence for one image."""

    pulses: list  # List of PulseEvent, sorted by time
    duration_ms: float  # Total sequence duration
    n_channels: int  # Number of stimulation channels
    binary_matrix: np.ndarray  # (n_channels, n_time_bins) 0-1 representation


class DelayedPhaseEncoder:
    """
    Encodes feature maps into spatiotemporal pulse sequences.

    Each feature map channel corresponds to one stimulation electrode.
    Pixel intensities within each channel are converted to firing times,
    aligned to sub-threshold membrane oscillation (SMO) peaks,
    and compressed into a pulse sequence.
    """

    def __init__(
        self,
        t_max_ms=2000.0,
        delta=1.0,
        min_interval_ms=200.0,
        smo_frequency_hz=5.0,
        n_time_bins=10,
    ):
        """
        Args:
            t_max_ms: Maximum stimulation duration in ms.
            delta: Scaling parameter for arctan firing time function.
                   Larger delta -> more spread in firing times.
            min_interval_ms: Minimum interval between adjacent pulses (ms).
                             Wang et al. used 200 ms for BNN recovery.
            smo_frequency_hz: Sub-threshold membrane oscillation frequency.
                              Wang et al. used 5 Hz (matching t_max of 2s).
            n_time_bins: Number of time bins for binary matrix representation.
        """
        self.t_max_ms = t_max_ms
        self.delta = delta
        self.min_interval_ms = min_interval_ms
        self.smo_frequency_hz = smo_frequency_hz
        self.n_time_bins = n_time_bins

    def _pixel_to_firing_time(self, pixel_value):
        """
        Convert normalized pixel value [0,1] to firing time.

        t_i = t_max * (1 - arctan(delta * p_i) / (pi/2))

        Higher pixel values -> earlier firing times.
        This matches biological intuition: stronger features activate faster.

        Args:
            pixel_value: Normalized pixel intensity in [0, 1].

        Returns:
            Firing time in ms, in range [0, t_max].
        """
        half_pi = np.pi / 2.0
        t = self.t_max_ms * (1.0 - np.arctan(self.delta * pixel_value) / half_pi)
        return t

    def _apply_temporal_offset(self, firing_times, rf_size):
        """
        Apply temporal offset within a receptive field.

        Creates temporal differences between pixel positions:
            delta_t_i = (1 / n_rf) * i

        Args:
            firing_times: Array of firing times for pixels in one RF.
            rf_size: Total number of pixels in the RF.

        Returns:
            Offset firing times.
        """
        offsets = np.arange(len(firing_times)) / rf_size
        return firing_times + offsets

    def _align_to_smo_peaks(self, firing_times):
        """
        Align firing times to nearest SMO peak for temporal compression.

        SMO = A * cos(omega * t + phi_0)
        Peaks occur at t = k / frequency for integer k.

        Wang et al. set all pixels in same RF to same initial phase (phi_0=0)
        to enable compression.

        Args:
            firing_times: Array of firing times in ms.

        Returns:
            Aligned firing times (snapped to nearest SMO peak).
        """
        period_ms = 1000.0 / self.smo_frequency_hz
        aligned = np.round(firing_times / period_ms) * period_ms
        return np.clip(aligned, 0, self.t_max_ms)

    def _enforce_min_interval(self, times):
        """
        Remove pulses that violate minimum inter-pulse interval constraint.

        Keeps the first pulse at each time, removes subsequent pulses
        that are too close.

        Args:
            times: Sorted array of pulse times in ms.

        Returns:
            Filtered array of pulse times.
        """
        if len(times) == 0:
            return times

        result = [times[0]]
        for t in times[1:]:
            if t - result[-1] >= self.min_interval_ms:
                result.append(t)
        return np.array(result)

    def encode(self, feature_maps):
        """
        Encode feature maps into a stimulus sequence.

        Args:
            feature_maps: numpy array of shape (n_channels, H, W),
                          values in [0, 1].

        Returns:
            StimulusSequence with pulse events and binary matrix.
        """
        n_channels, h, w = feature_maps.shape
        rf_size = h * w  # Receptive field size per channel
        all_pulses = []

        for ch in range(n_channels):
            # Flatten pixels in this channel's feature map
            pixels = feature_maps[ch].flatten()

            # Step 1: Convert pixel intensities to firing times
            raw_times = np.array(
                [self._pixel_to_firing_time(p) for p in pixels]
            )

            # Step 2: Apply temporal offset within RF
            offset_times = self._apply_temporal_offset(raw_times, rf_size)

            # Step 3: Align to SMO peaks (compression)
            aligned_times = self._align_to_smo_peaks(offset_times)

            # Step 4: Deduplicate (multiple pixels may align to same peak)
            unique_times = np.unique(aligned_times)

            # Step 5: Sort and enforce minimum interval
            unique_times = np.sort(unique_times)
            valid_times = self._enforce_min_interval(unique_times)

            # Create pulse events
            for t in valid_times:
                all_pulses.append(PulseEvent(channel=ch, time_ms=float(t)))

        # Sort all pulses by time
        all_pulses.sort(key=lambda p: (p.time_ms, p.channel))

        # Build binary matrix representation (n_channels x n_time_bins)
        binary_matrix = self._build_binary_matrix(all_pulses, n_channels)

        return StimulusSequence(
            pulses=all_pulses,
            duration_ms=self.t_max_ms,
            n_channels=n_channels,
            binary_matrix=binary_matrix,
        )

    def _build_binary_matrix(self, pulses, n_channels):
        """
        Build binary matrix representation for distance computation.

        Wang et al.: 8x10 matrix where element=1 indicates pulse presence.

        Args:
            pulses: List of PulseEvent.
            n_channels: Number of stimulation channels.

        Returns:
            Binary matrix of shape (n_channels, n_time_bins).
        """
        matrix = np.zeros((n_channels, self.n_time_bins), dtype=int)
        bin_width = self.t_max_ms / self.n_time_bins

        for pulse in pulses:
            bin_idx = min(int(pulse.time_ms / bin_width), self.n_time_bins - 1)
            matrix[pulse.channel, bin_idx] = 1

        return matrix

    def encode_to_protocol(self, feature_maps):
        """
        Encode feature maps and return a StimulusProtocol.

        This is the preferred entry point. Returns the frozen unified format.
        """
        seq = self.encode(feature_maps)
        pulses = [
            Pulse(channel=p.channel, time_ms=p.time_ms)
            for p in seq.pulses
        ]
        proto = StimulusProtocol(
            pulses=pulses,
            n_channels=seq.n_channels,
            duration_ms=seq.duration_ms,
        )
        proto.validate()
        return proto

    @staticmethod
    def compute_distance(seq1, seq2):
        """
        Compute Euclidean distance between two stimulus sequences.

        Wang et al. Eq. 5: d = sqrt(sum((U_ij - V_ij)^2))

        Args:
            seq1, seq2: StimulusSequence objects.

        Returns:
            Euclidean distance between binary matrix representations.
        """
        return np.sqrt(
            np.sum((seq1.binary_matrix - seq2.binary_matrix) ** 2)
        )
