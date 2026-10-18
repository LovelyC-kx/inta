import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.stimulus_schema import SafetyViolation
from frontend.app import _build_manual_protocol


def test_build_manual_protocol_safe():
    proto, summary = _build_manual_protocol({
        "channels": "3",
        "amplitude_ua": 10.0,
        "phase_duration_us": 500.0,
        "interphase_delay_us": 0.0,
        "polarity": "NegativeFirst",
        "waveform": "Biphasic",
    })
    proto.validate()
    proto.validate_safety()
    assert summary["channels"] == [3]
    assert summary["total_pulses"] == 1
    assert summary["pulses_per_channel"][3] == 1


def test_build_manual_protocol_multi_channel_sequence():
    proto, summary = _build_manual_protocol({
        "channels": "0,1,2",
        "amplitude_ua": 10.0,
        "phase_duration_us": 500.0,
        "pulse_count": 1,
        "channel_interval_ms": 250.0,
    })
    proto.validate()
    proto.validate_safety()
    assert summary["channels"] == [0, 1, 2]
    assert summary["total_pulses"] == 3
    assert [pulse.time_ms for pulse in proto.pulses] == [0.0, 250.0, 500.0]


def test_build_manual_protocol_unsafe_rejected():
    with pytest.raises(SafetyViolation):
        _build_manual_protocol({
            "channels": "0",
            "amplitude_ua": 60.0,
            "phase_duration_us": 500.0,
        })
