"""
Unified Stimulus Schema for Intan RHS systems.

This is the single, frozen interface between encoders and everything downstream
(Intan TCP, fake reservoir, real organoid, feature extraction).

Every encoder MUST convert its native output to this format.
Every consumer (TCP client, integration test, decoder pipeline) MUST accept
only this format.

============================================================================
IMPORTANT — RHS is CURRENT-mode stimulation
============================================================================
Intan RHS Stimulation/Recording Controller delivers controlled CURRENT
(not voltage). Default range is ±127.5 µA with 0.5 µA step (verified from
RHS Demo startup dialog).

Earlier voltage-based parameters (e.g., "500 mV biphasic") were inherited
from MaxOne / Wang et al. and are NOT directly applicable to Intan RHS.
This schema uses CURRENT in microamps as the primary parameter.

Default: charge-balanced biphasic, cathodic-first, 10 µA, 500 µs/phase.
This is a typical conservative starting point for in vitro stimulation.

Safety constraints (defaults; can be made stricter via validate_safety):
    - Max amplitude per pulse: 50 µA
    - Max charge per pulse: 25 nC (= 50 µA × 500 µs)
    - Min inter-pulse interval per channel: 200 ms
    - Max pulses per second per channel: 5
    - Required charge balance (|charge_pos - charge_neg| < tol)
    - Biphasic only (monophasic disabled by default)
============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ── Frozen hardware/protocol constants ─────────────────────────────

MAX_STIM_CHANNELS = 8                # Default; can be raised after 128ch validation
MAX_DURATION_MS = 2000.0             # Per-trial stimulus duration
MIN_INTER_PULSE_MS = 200.0           # Minimum gap between pulses on same channel
DEFAULT_AMPLITUDE_UA = 10.0          # Conservative default current
DEFAULT_PHASE_DURATION_US = 500.0
DEFAULT_INTERPHASE_DELAY_US = 0.0    # 0 for charge-balanced biphasic
DEFAULT_POLARITY = "NegativeFirst"   # RHX accepts NegativeFirst/PositiveFirst only.
                                     # CathodicFirst (the in-vitro literature term)
                                     # is rejected — RHX command channel hangs.
                                     # Cathodic == negative-current-first; same thing.
DEFAULT_WAVEFORM = "Biphasic"

# Hard safety ceilings (raise consciously; never silently)
SAFETY_MAX_AMPLITUDE_UA = 50.0       # Refuse pulses above this
SAFETY_MAX_PHASE_DURATION_US = 5000.0  # RHS allows up to 5 ms/phase
SAFETY_MAX_CHARGE_PER_PULSE_NC = 25.0  # 50 µA × 500 µs
SAFETY_MAX_TOTAL_CHARGE_PER_TRIAL_NC = 200.0  # Cumulative across all pulses
SAFETY_MAX_PULSES_PER_SEC_PER_CH = 5
SAFETY_MAX_PULSES_PER_TRIAL = 100    # Hard upper bound across all channels
SAFETY_CHARGE_BALANCE_TOL_NC = 0.5   # Phase imbalance allowed
SAFETY_ALLOW_MONOPHASIC = False      # Default: refuse monophasic (no charge recovery)


@dataclass(frozen=True)
class Pulse:
    """
    One stimulation pulse. Immutable.

    Maps directly to Intan RHS parameters. Amplitude is in microamps (µA),
    not millivolts (RHS is current-controlled).
    """

    channel: int
    time_ms: float
    amplitude_ua: float = DEFAULT_AMPLITUDE_UA
    phase_duration_us: float = DEFAULT_PHASE_DURATION_US
    interphase_delay_us: float = DEFAULT_INTERPHASE_DELAY_US
    polarity: str = DEFAULT_POLARITY
    waveform: str = DEFAULT_WAVEFORM

    @property
    def charge_per_phase_nc(self):
        """Charge per phase in nanocoulombs: I(µA) × t(µs) / 1000 = nC."""
        return self.amplitude_ua * self.phase_duration_us / 1000.0


@dataclass
class StimulusProtocol:
    """
    Complete stimulus for one trial.

    The ONLY format that leaves an encoder.
    The ONLY format that enters Intan TCP, fake reservoir, or real organoid.

    Two-tier validation:
        - validate(): structural invariants (channels, timing, sorting)
        - validate_safety(): hardware safety (currents, charge, balance)

    Real experiments MUST call BOTH before sending to hardware.
    """

    pulses: List[Pulse]
    n_channels: int
    duration_ms: float

    def validate(self):
        """
        Check structural invariants. Raises ValueError on violation.
        Call this after construction. Returns self for chaining.
        """
        if self.n_channels > MAX_STIM_CHANNELS:
            raise ValueError(
                f"n_channels={self.n_channels} exceeds MAX_STIM_CHANNELS={MAX_STIM_CHANNELS}"
            )
        if self.duration_ms > MAX_DURATION_MS:
            raise ValueError(
                f"duration_ms={self.duration_ms} exceeds MAX_DURATION_MS={MAX_DURATION_MS}"
            )

        for p in self.pulses:
            if p.channel < 0 or p.channel >= self.n_channels:
                raise ValueError(
                    f"Pulse channel={p.channel} out of range [0, {self.n_channels})"
                )
            if p.time_ms < 0 or p.time_ms > self.duration_ms:
                raise ValueError(
                    f"Pulse time_ms={p.time_ms} out of range [0, {self.duration_ms}]"
                )

        for ch in range(self.n_channels):
            ch_times = sorted(p.time_ms for p in self.pulses if p.channel == ch)
            for i in range(1, len(ch_times)):
                interval = ch_times[i] - ch_times[i - 1]
                if interval < MIN_INTER_PULSE_MS - 1e-6:
                    raise ValueError(
                        f"Channel {ch}: inter-pulse interval {interval:.1f} ms "
                        f"< minimum {MIN_INTER_PULSE_MS} ms"
                    )

        for i in range(1, len(self.pulses)):
            prev, curr = self.pulses[i - 1], self.pulses[i]
            if (curr.time_ms, curr.channel) < (prev.time_ms, prev.channel):
                raise ValueError("Pulses must be sorted by (time_ms, channel)")

        return self

    def validate_safety(
        self,
        max_amplitude_ua=SAFETY_MAX_AMPLITUDE_UA,
        max_phase_duration_us=SAFETY_MAX_PHASE_DURATION_US,
        max_charge_per_pulse_nc=SAFETY_MAX_CHARGE_PER_PULSE_NC,
        max_total_charge_per_trial_nc=SAFETY_MAX_TOTAL_CHARGE_PER_TRIAL_NC,
        max_pulses_per_sec=SAFETY_MAX_PULSES_PER_SEC_PER_CH,
        max_pulses_per_trial=SAFETY_MAX_PULSES_PER_TRIAL,
        allow_monophasic=SAFETY_ALLOW_MONOPHASIC,
        charge_balance_tol_nc=SAFETY_CHARGE_BALANCE_TOL_NC,
    ):
        """
        Check biological safety constraints. Raises SafetyViolation on any issue.

        This MUST pass before sending the protocol to real hardware.
        Returns self for chaining.
        """
        violations = []

        for p in self.pulses:
            if p.amplitude_ua > max_amplitude_ua:
                violations.append(
                    f"Pulse ch={p.channel} t={p.time_ms}ms: "
                    f"amplitude {p.amplitude_ua} µA > limit {max_amplitude_ua} µA"
                )
            if p.phase_duration_us > max_phase_duration_us:
                violations.append(
                    f"Pulse ch={p.channel} t={p.time_ms}ms: "
                    f"phase_duration {p.phase_duration_us} µs > limit "
                    f"{max_phase_duration_us} µs"
                )
            if p.charge_per_phase_nc > max_charge_per_pulse_nc:
                violations.append(
                    f"Pulse ch={p.channel} t={p.time_ms}ms: "
                    f"charge {p.charge_per_phase_nc:.2f} nC > limit "
                    f"{max_charge_per_pulse_nc} nC"
                )
            if not allow_monophasic and p.waveform.lower() == "monophasic":
                violations.append(
                    f"Pulse ch={p.channel} t={p.time_ms}ms: "
                    f"monophasic not allowed (no charge recovery)"
                )

        # Total charge per trial (sum of both phases for biphasic)
        total_charge_nc = sum(p.charge_per_phase_nc for p in self.pulses) * 2
        if total_charge_nc > max_total_charge_per_trial_nc:
            violations.append(
                f"Total trial charge {total_charge_nc:.1f} nC > limit "
                f"{max_total_charge_per_trial_nc} nC "
                f"(across {len(self.pulses)} pulses, biphasic ×2)"
            )

        # Total pulses per trial
        if len(self.pulses) > max_pulses_per_trial:
            violations.append(
                f"Total pulses {len(self.pulses)} > limit {max_pulses_per_trial}"
            )

        # Pulse rate per channel
        duration_s = self.duration_ms / 1000.0
        for ch in range(self.n_channels):
            n_ch_pulses = sum(1 for p in self.pulses if p.channel == ch)
            rate = n_ch_pulses / duration_s if duration_s > 0 else 0
            if rate > max_pulses_per_sec:
                violations.append(
                    f"Channel {ch}: {n_ch_pulses} pulses in {duration_s}s "
                    f"= {rate:.1f} Hz > limit {max_pulses_per_sec} Hz"
                )

        if violations:
            raise SafetyViolation(
                f"{len(violations)} safety violation(s):\n  " +
                "\n  ".join(violations)
            )

        return self

    def safety_report(self):
        """Return a dict summarizing safety-relevant statistics. No exceptions."""
        if not self.pulses:
            return {
                "n_pulses": 0,
                "max_amplitude_ua": 0,
                "max_charge_nc": 0,
                "total_charge_nc": 0,
                "max_rate_hz": 0,
            }
        amps = [p.amplitude_ua for p in self.pulses]
        charges = [p.charge_per_phase_nc for p in self.pulses]
        rates = []
        duration_s = self.duration_ms / 1000.0
        for ch in range(self.n_channels):
            n = sum(1 for p in self.pulses if p.channel == ch)
            rates.append(n / duration_s if duration_s > 0 else 0)
        return {
            "n_pulses": len(self.pulses),
            "max_amplitude_ua": max(amps),
            "max_charge_nc": max(charges),
            "total_charge_nc": sum(charges) * 2,  # ×2 for biphasic
            "max_rate_hz": max(rates) if rates else 0,
            "n_channels_used": sum(1 for r in rates if r > 0),
        }

    def to_binary_matrix(self, n_time_bins=10):
        """Convert to (n_channels, n_time_bins) binary matrix."""
        matrix = np.zeros((self.n_channels, n_time_bins), dtype=np.int32)
        bin_width = self.duration_ms / n_time_bins
        for p in self.pulses:
            bin_idx = min(int(p.time_ms / bin_width), n_time_bins - 1)
            matrix[p.channel, bin_idx] = 1
        return matrix

    def to_spike_raster(self, bin_size_ms=10.0):
        """Convert to (n_channels, n_bins) float32 spike raster."""
        n_bins = int(np.ceil(self.duration_ms / bin_size_ms))
        raster = np.zeros((self.n_channels, n_bins), dtype=np.float32)
        for p in self.pulses:
            bin_idx = min(int(p.time_ms / bin_size_ms), n_bins - 1)
            raster[p.channel, bin_idx] += 1.0
        return raster

    @property
    def total_pulses(self):
        return len(self.pulses)

    @property
    def pulses_per_channel(self):
        counts = [0] * self.n_channels
        for p in self.pulses:
            counts[p.channel] += 1
        return counts


class SafetyViolation(Exception):
    """Raised when a StimulusProtocol fails safety validation."""


def euclidean_distance(proto_a, proto_b, n_time_bins=10):
    """Wang et al. Eq. 5 distance between protocol binary matrices."""
    m_a = proto_a.to_binary_matrix(n_time_bins)
    m_b = proto_b.to_binary_matrix(n_time_bins)
    return float(np.sqrt(np.sum((m_a - m_b) ** 2)))


# Default trim target: 90% of the hard safety cap leaves headroom for
# floating-point rounding and any pulse-rate / per-pulse-charge limits.
DEFAULT_TRIM_TARGET_NC = SAFETY_MAX_TOTAL_CHARGE_PER_TRIAL_NC * 0.9


def trim_to_charge_budget(protocol, target_charge_nc=DEFAULT_TRIM_TARGET_NC):
    """
    Return a copy of `protocol` that fits within `target_charge_nc` of
    cumulative biphasic charge, by keeping pulses earliest-first.

    Why earliest-first:
        - delayed_phase: earlier time == higher feature intensity (arctan map)
        - LIF / rate: spike times distribute roughly uniformly; earliest
          subset is as good as any other and easiest to reason about
        - Always retains protocol[0] when at least one pulse fits

    Inter-pulse-interval invariants are preserved automatically: removing
    pulses can only INCREASE intervals between remaining pulses.

    No-op if the protocol already fits.

    Returns:
        StimulusProtocol — a freshly validated copy. The original is
        unchanged.
    """
    if not protocol.pulses:
        return protocol

    sorted_pulses = sorted(protocol.pulses, key=lambda p: (p.time_ms, p.channel))
    cumulative_nc = 0.0
    kept = []
    for p in sorted_pulses:
        # Biphasic delivers charge in both phases — count both phases.
        # Monophasic is rejected upstream by validate_safety; we still
        # count *2 for biphasic and equivalently for any waveform.
        pulse_total_nc = p.charge_per_phase_nc * 2.0
        if cumulative_nc + pulse_total_nc > target_charge_nc:
            break
        kept.append(p)
        cumulative_nc += pulse_total_nc

    if len(kept) == len(sorted_pulses):
        return protocol

    trimmed = StimulusProtocol(
        pulses=kept,
        n_channels=protocol.n_channels,
        duration_ms=protocol.duration_ms,
    )
    trimmed.validate()
    return trimmed
