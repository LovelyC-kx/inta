import pytest

from frontend.app import app, operation_lock, stim_cancel_event, _require_new_recording


def test_no_fallback_to_an_old_recording(tmp_path):
    old = tmp_path / "old_recording"
    old.mkdir()
    (old / "stim.dat").write_bytes(b"\x00\x00")
    with pytest.raises(RuntimeError, match="refusing an old"):
        _require_new_recording(tmp_path, {old.name})
    new = tmp_path / "new_recording"
    new.mkdir()
    (new / "stim.dat").write_bytes(b"\x00\x00")
    assert _require_new_recording(tmp_path, {old.name}) == new


def test_multiple_new_recordings_are_ambiguous(tmp_path):
    for name in ("one", "two"):
        p = tmp_path / name
        p.mkdir()
        (p / "stim.dat").write_bytes(b"\x00\x00")
    with pytest.raises(RuntimeError, match="ambiguous"):
        _require_new_recording(tmp_path, set())


def test_browser_status_and_control_do_not_open_another_busy_session():
    operation_lock.acquire()
    try:
        with app.test_client() as client:
            assert client.post("/api/rhx/status", json={}).status_code == 409
            assert client.post("/api/rhx/control", json={"action": "run"}).status_code == 409
    finally:
        operation_lock.release()


def test_emergency_stop_cancels_the_shared_scheduler():
    stim_cancel_event.clear()
    try:
        with app.test_client() as client:
            assert client.post("/api/emergency_stop").status_code == 200
        assert stim_cancel_event.is_set()
    finally:
        stim_cancel_event.clear()
