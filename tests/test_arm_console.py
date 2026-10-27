import pytest

from virtual_arm_test import main


@pytest.mark.parametrize("args", [["--live", "--port", "COM4"], ["--live", "--home"]])
def test_live_console_rejects_incomplete_options_before_opening_serial(args):
    def forbidden(**kwargs):
        raise AssertionError("Must not create an arm with incomplete live options")
    with pytest.raises(SystemExit) as exc:
        main(args, controller_factory=forbidden)
    assert exc.value.code == 2


def test_number_console_stop_latches_until_explicit_home(capsys):
    keys = iter(["1", "2", "0", "1", "h", "3", "4", "q"])
    assert main(["--interactive", "--step-mm", "1"], read=lambda _: next(keys)) == 0
    out = capsys.readouterr()
    assert '[-1.0, 100.0, 70.0]' in out.out
    assert 'Call home() explicitly' in out.out
    assert '"action": "grab"' in out.out and '"action": "release"' in out.out
    assert out.err.count("$DST!") == 2  # explicit stop and exit cleanup


def test_keyboard_interrupt_stops_console(capsys):
    def interrupt(_):
        raise KeyboardInterrupt
    assert main([], read=interrupt) == 0
    assert "$DST!" in capsys.readouterr().err
