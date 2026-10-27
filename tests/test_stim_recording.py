import json

import numpy as np
import pytest

from experiments.verify_stim_dat import (
    decode_stim_words, estimate_charge_per_pulse, find_pulse_onsets,
    load_stim_dat, summarize,
)


def bipolar(gap=0, amplitude_code=20):
    return np.array([0] + [0x100 | amplitude_code] * 15 + [0x2000] * gap
                    + [amplitude_code] * 15 + [0], dtype=np.uint16)


def write_recording(path, data, expected=None):
    np.asarray(data, dtype="<u2").tofile(path / "stim.dat")
    if expected is not None:
        (path / "trial_log.json").write_text(json.dumps([{"total_pulses": expected}]))


@pytest.mark.parametrize("flag", [0x2000, 0x4000, 0x8000])
def test_flags_are_not_stimulation_current(flag):
    assert find_pulse_onsets(np.array([0, flag, 0], dtype=np.uint16)).size == 0


def test_recording_timestamp_gap_is_not_treated_as_contiguous(tmp_path):
    data = bipolar()
    write_recording(tmp_path, data)
    times = np.arange(len(data), dtype="<u4")
    times[10:] += 1
    times.tofile(tmp_path / "time.dat")
    with pytest.raises(ValueError, match="timestamp gap/reset"):
        summarize(tmp_path, n_channels=1, plot=False)


def test_conflicting_session_metadata_is_rejected(tmp_path):
    write_recording(tmp_path, bipolar())
    (tmp_path / "config.json").write_text(json.dumps({"sample_rate_hz": 30000}))
    with pytest.raises(ValueError, match="disagree"):
        summarize(tmp_path, n_channels=1, plot=False, sample_rate_hz=20000)


def test_bit_layout_matches_official_rhs_reader():
    result = decode_stim_words(np.array([0xE114], dtype=np.uint16))
    assert result["magnitude"][0] == 20
    assert all(result[k][0] for k in ("negative", "amp_settle", "charge_recovery", "compliance_limit"))
    assert not result["unknown_flags"][0]


@pytest.mark.parametrize("gap", [0, 15])
def test_two_phases_are_one_pulse_even_with_interphase_gap(gap):
    np.testing.assert_array_equal(find_pulse_onsets(bipolar(gap)), [1])


def test_pulse_at_first_sample_is_retained():
    np.testing.assert_array_equal(find_pulse_onsets(bipolar()[1:]), [0])


def test_charge_requires_actual_step_and_ignores_status_bits():
    with pytest.raises(ValueError, match="explicit"):
        estimate_charge_per_pulse(bipolar(), 1)
    assert estimate_charge_per_pulse(bipolar(15), 1, stim_step_ua=.5) == 10


def test_flag_only_record_is_not_verified(tmp_path):
    write_recording(tmp_path, [0, 0x2000, 0x4000, 0])
    result = summarize(tmp_path, n_channels=1, plot=False)
    assert result["total_pulses"] == 0
    assert result["verdict"] == "NO_STIM_RECORDED"
    assert not result["physical_delivery_verified"]


def test_count_mismatch_is_not_reported_as_verified(tmp_path):
    write_recording(tmp_path, bipolar(), expected=2)
    result = summarize(tmp_path, n_channels=1, plot=False)
    assert result["verdict"] == "STIM_COUNT_MISMATCH"


def test_count_match_and_hardware_timestamps_still_are_not_electrical_proof(tmp_path):
    data = bipolar(5)
    write_recording(tmp_path, data, expected=1)
    np.arange(1000, 1000 + len(data), dtype="<u4").tofile(tmp_path / "time.dat")
    result = summarize(tmp_path, n_channels=1, plot=False, stim_step_ua=.5)
    assert result["verdict"] == "STIM_RECORDED_COUNT_MATCH"
    assert result["pulse_records"][0][0]["raw_rhx_timestamp"] == 1001
    assert result["pulse_records"][0][0]["absolute_charge_nc_from_codes"] == 10
    assert not result["physical_delivery_verified"]


def test_unpaired_phase_and_compliance_are_failures(tmp_path):
    write_recording(tmp_path, [0, 20, 20, 0])
    assert summarize(tmp_path, n_channels=1, plot=False)["verdict"] == "INVALID_STIM_DATA"
    write_recording(tmp_path, bipolar() | np.uint16(0x8000))
    assert summarize(tmp_path, n_channels=1, plot=False)["verdict"] == "COMPLIANCE_LIMIT_DETECTED"


def test_truncated_records_and_wrong_saved_channel_count_are_rejected(tmp_path):
    (tmp_path / "stim.dat").write_bytes(b"\x01")
    with pytest.raises(ValueError, match="truncated"):
        load_stim_dat(tmp_path, 1)
    write_recording(tmp_path, [0, 1, 2])
    with pytest.raises(ValueError, match="channel count"):
        load_stim_dat(tmp_path, 2)


def test_time_file_length_must_match_recording(tmp_path):
    write_recording(tmp_path, bipolar())
    (tmp_path / "time.dat").write_bytes(b"\x00" * 4)
    with pytest.raises(ValueError, match="sample counts"):
        summarize(tmp_path, n_channels=1, plot=False)


def test_interphase_gap_above_rhs_limit_is_refused():
    from encoders.stimulus_schema import Pulse, StimulusProtocol, SafetyViolation
    with pytest.raises(SafetyViolation, match="interphase delay"):
        StimulusProtocol([Pulse(0, 0, interphase_delay_us=6000,
                               waveform="BiphasicWithInterphaseDelay")], 8, 2000).validate_safety()
