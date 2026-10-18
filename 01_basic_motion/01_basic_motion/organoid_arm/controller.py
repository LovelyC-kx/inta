"""Signal-source-independent AM1 control. Importing never opens hardware."""

import math
import sys
import threading
import time

from .kinematics import solve


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(name + " is outside the allowed range")
    return value


class ArmController:
    """Blocking commands with estimated timing, not measured position feedback."""

    HOME = (0.0, 100.0, 70.0)
    LIMITS = ((-120, 120), (45, 190), (20, 140))
    DIRECTIONS = {"up": (0, 0, 1), "down": (0, 0, -1),
                  "left": (-1, 0, 0), "right": (1, 0, 0),
                  "forward": (0, 1, 0), "back": (0, -1, 0)}
    ACTIONS = frozenset(DIRECTIONS) | {"home", "grab", "release", "cw", "ccw", "center", "idle", "stop"}

    def __init__(self, port=None, *, dry_run=True, step_mm=5, move_time_ms=800,
                 open_pwm=1000, close_pwm=1550):
        self.step_mm = number(step_mm, "step_mm", 0.1, 20)
        self.move_time_ms = number(move_time_ms, "move_time_ms", 100, 9999)
        if not isinstance(move_time_ms, int):
            raise ValueError("move_time_ms must be an integer")
        for name, value in (("open_pwm", open_pwm), ("close_pwm", close_pwm)):
            number(value, name, 500, 2500)
            if not isinstance(value, int):
                raise ValueError(name + " must be an integer")
        self.open_pwm, self.close_pwm = open_pwm, close_pwm
        self.dry_run = dry_run
        self.pose = None
        self.wrist_pwm = None
        self._serial = None
        self._closed = False
        self._busy = threading.Lock()
        self._io = threading.Lock()
        self._stopped = threading.Event()
        if not dry_run:
            if not port:
                raise ValueError("A controller serial port is required for live operation")
            import serial
            self._serial = serial.Serial(port, 115200, timeout=0.2, write_timeout=1)
            time.sleep(2)

    def _write(self, command):
        if self._closed:
            raise RuntimeError("Controller is closed")
        if self.dry_run:
            print(command, file=sys.stderr)
        else:
            data = command.encode("ascii")
            if self._serial.write(data) != len(data):
                raise OSError("Incomplete serial write")
            self._serial.flush()

    def _send(self, values):
        duration = self.move_time_ms
        command = "{" + "".join(f"#{i:03d}P{p:04d}T{duration:04d}!"
                                  for i, p in values) + "}"
        with self._io:
            if self._stopped.is_set():
                raise RuntimeError("Stopped: explicitly home() before moving again")
            self._write(command)
        if not self.dry_run:
            self._stopped.wait(duration / 1000 + 0.1)
        if self._stopped.is_set():
            raise RuntimeError("Motion interrupted; position is unknown")

    def _move_to(self, target):
        for value, (low, high) in zip(target, self.LIMITS):
            number(value, "coordinate", low, high)
        pwm = solve(*target)
        self._send(enumerate(pwm))
        with self._io:
            if self._stopped.is_set():
                raise RuntimeError("Motion interrupted; position is unknown")
            self.pose = tuple(target)

    def _set_wrist(self, pwm):
        with self._io:
            if self._stopped.is_set():
                raise RuntimeError("Motion interrupted; position is unknown")
            self.wrist_pwm = pwm

    def stop(self):
        """Latch stop; may be called from another thread during an action."""
        with self._io:
            self._stopped.set()
            self.pose = None
            self.wrist_pwm = None
            self._write("$DST!")
        return {"status": "stop_sent", "position_verified": False}

    def execute(self, action):
        if not isinstance(action, str) or action not in self.ACTIONS:
            raise ValueError("Unknown action: " + repr(action))
        if action == "stop":
            return self.stop()
        if not self._busy.acquire(blocking=False):
            return {"status": "ignored", "reason": "busy"}
        try:
            if self._closed:
                raise RuntimeError("Controller is closed")
            if action == "home":
                with self._io:
                    self._stopped.clear()
                self._move_to(self.HOME)
                self._send([(4, 1500)])
                self._set_wrist(1500)
            elif action != "idle":
                if self._stopped.is_set() or self.pose is None:
                    raise RuntimeError("Call home() explicitly before controlling the arm")
                if action in self.DIRECTIONS:
                    delta = self.DIRECTIONS[action]
                    self._move_to(tuple(p + d * self.step_mm for p, d in zip(self.pose, delta)))
                elif action in ("grab", "release"):
                    self._send([(5, self.close_pwm if action == "grab" else self.open_pwm)])
                else:
                    pwm = 1500 if action == "center" else self.wrist_pwm + (222 if action == "cw" else -222)
                    number(pwm, "wrist_pwm", 500, 2500)
                    self._send([(4, pwm)])
                    self._set_wrist(pwm)
            return {"status": "simulated" if self.dry_run else "sent",
                    "action": action, "pose_estimate_mm": self.pose,
                    "position_verified": False}
        except (OSError, RuntimeError):
            self.pose = None
            self.wrist_pwm = None
            self._stopped.set()
            raise
        finally:
            self._busy.release()

    def home(self):
        return self.execute("home")

    def close(self):
        with self._io:
            if self._serial is not None:
                self._serial.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SignalAdapter:
    """Accept discrete decoder events; never accepts raw electrode samples."""

    def __init__(self, arm, *, min_confidence=0.8, max_age_s=1.0):
        self.arm = arm
        self.min_confidence = number(min_confidence, "min_confidence", 0, 1)
        self.max_age_s = number(max_age_s, "max_age_s", 0.01, 60)
        self._last_seq = -1
        self._lock = threading.Lock()

    def handle(self, event):
        if not isinstance(event, dict):
            raise ValueError("Event must be an object")
        if set(event) != {"seq", "timestamp", "action", "confidence"}:
            raise ValueError("Required fields: seq, timestamp, action, confidence")
        seq = event["seq"]
        if type(seq) is not int or seq < 0:
            raise ValueError("seq must be a nonnegative integer")
        timestamp = number(event["timestamp"], "timestamp", 0, 1e12)
        confidence = number(event["confidence"], "confidence", 0, 1)
        action = event["action"]
        # Homing and recovery require an explicit operator call.
        if not isinstance(action, str) or action not in self.arm.ACTIONS - {"home"}:
            raise ValueError("Unsupported signal action")
        if not self._lock.acquire(blocking=False):
            return {"status": "ignored", "reason": "busy"}
        try:
            if seq <= self._last_seq:
                return {"status": "ignored", "reason": "duplicate_or_out_of_order"}
            self._last_seq = seq
            age = time.time() - timestamp
            if age < -0.1 or age > self.max_age_s:
                return {"status": "ignored", "reason": "expired_or_future_event"}
            if action != "stop" and confidence < self.min_confidence:
                return {"status": "ignored", "reason": "low_confidence"}
            return self.arm.execute(action)
        finally:
            self._lock.release()
