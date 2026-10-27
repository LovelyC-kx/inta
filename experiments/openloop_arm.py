"""Asynchronous class-to-task adapter; acquisition never waits on serial motion."""
import importlib
import math
from pathlib import Path
import queue
import sys
import threading
import time


def arm_types():
    root = Path(__file__).resolve().parents[1] / "01_basic_motion" / "01_basic_motion"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("organoid_arm.controller")
    return module.ArmController, module.SignalAdapter


class ArmTaskWorker:
    def __init__(self, action_map, journal, *, live=False, port=None, home=False,
                 min_confidence=.8, max_age_s=1, arm=None):
        ArmController, SignalAdapter = arm_types()
        if not 0 <= min_confidence <= 1 or not 0 < max_age_s <= 60:
            raise ValueError("Invalid decision confidence/age limits")
        self.action_map = {}
        for label, macro in action_map.items():
            macro = [macro] if isinstance(macro, str) else macro
            if (not isinstance(label, str) or not isinstance(macro, list) or not macro or
                    len(macro) > 32 or any(a not in ArmController.ACTIONS - {"home"} for a in macro)):
                raise ValueError("Action maps require 1..32 supported actions per class; home is operator-only")
            self.action_map[label] = list(macro)
        if live and not home:
            raise ValueError("Live arm requires an explicit --home initialization")
        self.journal = journal
        self.min_confidence, self.max_age_s = min_confidence, max_age_s
        self.arm = arm if arm is not None else ArmController(port=port, dry_run=not live)
        self.adapter = SignalAdapter(self.arm, min_confidence=min_confidence, max_age_s=max_age_s)
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._busy = False
        self._seen = set()
        self._stop = threading.Event()
        self.error = None
        self.results = []
        self._seq = 0
        try:
            # Simulation home has no serial output; physical home is explicit.
            self.journal("arm_home", self.arm.home())
        except Exception:
            self.arm.close()
            raise
        self._thread = threading.Thread(target=self._run, name="organoid-arm-task", daemon=True)
        self._thread.start()

    def submit(self, trial_id, prediction, *, decision_time=None):
        if self.error is not None or self._stop.is_set():
            raise RuntimeError("Arm worker is stopped/failed") from self.error
        decision_time = time.time() if decision_time is None else decision_time
        confidence = prediction.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or
                not math.isfinite(confidence) or not 0 <= confidence <= 1 or
                isinstance(decision_time, bool) or not isinstance(decision_time, (int, float)) or
                not math.isfinite(decision_time)):
            raise ValueError("Decision time/confidence must be finite numbers; confidence in [0,1]")
        if (not isinstance(trial_id, str) or not trial_id or not isinstance(prediction.get("label"), str) or
                prediction["label"] not in self.action_map):
            raise ValueError("Decision has no explicit class-to-action mapping")
        with self._lock:
            if trial_id in self._seen:
                return {"status": "REJECTED", "reason": "duplicate_trial"}
            self._seen.add(trial_id)
            age = time.time() - decision_time
            if age < -.1 or age > self.max_age_s:
                return {"status": "REJECTED", "reason": "expired_or_future_decision"}
            if prediction["confidence"] < self.min_confidence:
                return {"status": "REJECTED", "reason": "low_confidence"}
            if self._busy:
                return {"status": "REJECTED", "reason": "arm_busy"}
            self._busy = True
            self._queue.put_nowait((trial_id, dict(prediction), decision_time))
            return {"status": "QUEUED", "trial_id": trial_id}

    def _run(self):
        while not self._stop.is_set():
            try:
                trial_id, prediction, decision_time = self._queue.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                if time.time() - decision_time > self.max_age_s:
                    result = {"status": "REJECTED", "reason": "expired_before_execution"}
                else:
                    actions = []
                    for action in self.action_map[prediction["label"]]:
                        if self._stop.is_set():
                            raise InterruptedError("Task cancelled")
                        self._seq += 1
                        receipt = self.adapter.handle({"seq": self._seq, "timestamp": time.time(),
                            "action": action, "confidence": prediction["confidence"]})
                        actions.append(receipt)
                        self.journal("arm_action", {"trial_id": trial_id, "receipt": receipt})
                        if receipt["status"] not in ("sent", "simulated", "stop_sent"):
                            raise RuntimeError(f"Arm action rejected: {receipt}")
                    result = {"status": "COMMANDS_SENT", "actions": actions,
                              "physical_task_success": None, "position_verified": False}
                result["trial_id"] = trial_id
                self.results.append(result)
                self.journal("arm_task", result)
            except Exception as exc:
                self.error = exc
                self.fail()
            finally:
                with self._lock:
                    self._busy = False
                self._queue.task_done()

    def fail(self):
        self._stop.set()
        try:
            self.arm.stop()
        except Exception as exc:
            if self.error is None:
                self.error = exc

    def wait(self, timeout=60):
        deadline = time.monotonic() + timeout
        while self._busy:
            if time.monotonic() >= deadline:
                self.fail()
                raise TimeoutError("Arm task completion timeout")
            if self.error is not None:
                raise RuntimeError("Arm task failed") from self.error
            time.sleep(.002)
        if self.error is not None:
            raise RuntimeError("Arm task failed") from self.error

    def close(self, *, failed=False):
        try:
            if failed:
                self.fail()
            else:
                self.wait()
                self._stop.set()
            self._thread.join(2)
            if self._thread.is_alive():
                raise TimeoutError("Arm task worker failed to stop")
        finally:
            try:
                if not self.arm.dry_run:
                    self.arm.stop()
            finally:
                self.arm.close()
