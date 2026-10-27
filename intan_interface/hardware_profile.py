"""RHS representability checks, separate from electrode safety limits.

RHX TCP guide 3.5.0 pp. 6-8, 54; User Guide pp. 4, 24.
Upstream ControllerInterface::setStimSequenceParameters rounds both duration
and current at upload. Software readback alone does not expose that rounding.
"""
from dataclasses import asdict, dataclass
import math


def _positive_number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid RHX {name}: {value!r}") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Invalid RHX {name}: {value!r}")
    return result


def rhx_bool(value, name):
    if value.casefold() not in ("true", "false"):
        raise ValueError(f"Invalid RHX {name}: {value!r}")
    return value.casefold() == "true"


def exact_steps(value, step, name, *, minimum=0, maximum=None):
    steps = value / step
    nearest = round(steps)
    if (not math.isclose(steps, nearest, rel_tol=0, abs_tol=1e-7) or
            nearest < minimum or (maximum is not None and nearest > maximum)):
        raise ValueError(f"{name}={value:g} cannot be represented exactly with hardware "
                         f"step {step:g}; choose an explicit representable value")
    return nearest


@dataclass(frozen=True)
class RHSProfile:
    version: str
    controller_type: str
    synthetic: bool
    playback: bool
    sample_rate_hz: int
    stim_step_ua: float
    expander_connected: bool

    @classmethod
    def read(cls, read_value):
        controller_type = read_value("Type")
        if controller_type != "ControllerStimRecord":
            raise ValueError(f"Current stimulation requires RHS; RHX Type={controller_type}")
        rate = _positive_number(read_value("SampleRateHertz"), "SampleRateHertz")
        if rate not in (20000, 25000, 30000):
            raise ValueError(f"Unsupported RHS sample rate: {rate:g}")
        step = _positive_number(read_value("StimStepSizeMicroAmps"), "StimStepSizeMicroAmps")
        if step not in (.01, .02, .05, .1, .2, .5, 1, 2, 5, 10):
            raise ValueError(f"Unsupported RHS current step: {step:g}")
        profile = cls(read_value("Version"), controller_type,
                      rhx_bool(read_value("Synthetic"), "Synthetic"),
                      rhx_bool(read_value("Playback"), "Playback"), int(rate), step,
                      rhx_bool(read_value("ExpanderConnected"), "ExpanderConnected"))
        if profile.playback:
            raise ValueError("Playback cannot execute a live stimulation protocol")
        return profile

    def validate_pulse(self, pulse):
        tick_us = 1e6 / self.sample_rate_hz
        return {
            "amplitude_code": exact_steps(pulse.amplitude_ua, self.stim_step_ua,
                                          "amplitude_ua", minimum=1, maximum=255),
            "phase_samples": exact_steps(pulse.phase_duration_us, tick_us,
                                         "phase_duration_us", minimum=1),
            "interphase_samples": exact_steps(pulse.interphase_delay_us, tick_us,
                                              "interphase_delay_us"),
        }

    def as_dict(self):
        return asdict(self)
