"""
Real-time closed-loop engine.

Core loop:
  1. Encode stimulus from current task state
  2. Configure & trigger Intan stimulation via TCP
  3. Read evoked response in real-time from TCP waveform stream
  4. Spike detection on the fly
  5. Feature extraction
  6. SNN decoder inference
  7. Generate feedback stimulus based on decode result
  8. Log everything with timestamps
  9. Repeat

Architecture:
  Main thread: stimulus → wait → read → decode → feedback → next trial
  No separate reader thread needed if we control timing precisely.

Latency budget (per trial):
  - Encode stimulus:     ~3 ms (verified: 2.8ms/image in integration test)
  - TCP stim config:     ~50 ms (depends on command count)
  - Trigger + bio delay: ~200 ms (stimulus duration + neural response)
  - Read evoked data:    ~10 ms (500ms window at 30kHz = 15000 samples)
  - Spike detection:     ~5 ms
  - Feature extraction:  ~1 ms
  - SNN decode:          ~2 ms
  - Feedback stim:       ~50 ms
  Total per trial:       ~320 ms (without inter-trial interval)

Usage:
  python realtime_loop.py --mode demo       # Test with Intan Demo
  python realtime_loop.py --mode live       # Real hardware + tissue
  python realtime_loop.py --mode classify   # Classification task
  python realtime_loop.py --mode closedloop # Full closed-loop with feedback
"""

import sys
import os
import time
import json
import shutil
import argparse
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from intan_interface.tcp_client import IntanTCPClient
from encoders.stimulus_schema import StimulusProtocol, Pulse
from encoders.cnn_feature_extractor import CNNFeatureExtractor
from encoders.pipeline import EncodingPipeline
from decoders.snn_decoder import SNNDecoder
from feature_extraction.extractors import extract_timebin_features
from utils.experiment_log import log_experiment


# ── RHX directory helpers (same logic as run_2d_experiment.py) ──────
# When in Record mode, RHX auto-creates a subdir under its GUI-configured
# save path. We snapshot listdir before/after to discover that new subdir.

def _get_rhx_save_root(client, fallback="data"):
    try:
        resp = client.get("Filename.Path")
        return resp.split(maxsplit=2)[-1]
    except Exception as e:
        print(f"[WARN] Could not query Filename.Path ({e}); using fallback '{fallback}'")
        return fallback


def _find_new_rhx_subdir(rhx_root, before_snapshot):
    rhx_root = Path(rhx_root)
    if not rhx_root.exists():
        return None
    after = set(os.listdir(rhx_root))
    new = after - before_snapshot
    if not new:
        return None
    return rhx_root / sorted(new)[-1]


def _attach_metadata_to_rhx_dir(rhx_dir, our_dir,
                                 files=("config.json", "trial_log.json")):
    if rhx_dir is None:
        return
    for fname in files:
        src = Path(our_dir) / fname
        if src.exists():
            shutil.copy(src, rhx_dir / fname)


class RealtimeLoop:
    """
    Real-time closed-loop controller for Intan RHS + biological tissue.

    Manages the full cycle: stimulate → record → decode → feedback.
    Optionally pushes live updates to the web dashboard.

    Modes:
        dry_run=False: send real stimulation commands (DEFAULT — DANGEROUS
                       on real tissue without prior oscilloscope verification)
        dry_run=True : skip ALL stimulation TCP commands; only read waveform
                       and run analytics. Use this until single-channel
                       oscilloscope verification is complete.

    Safety:
        require_safety_validation=True (default) calls protocol.validate_safety()
        before EVERY stimulus. Refuses to send if it fails.
    """

    def __init__(
        self,
        host="127.0.0.1",
        command_port=5000,
        waveform_port=5001,
        sample_rate=30000,
        n_stim_channels=8,
        n_rec_channels=8,
        evoked_window_ms=500.0,
        artifact_blank_ms=10.0,
        spike_threshold_factor=5.5,
        dashboard=None,
        dry_run=True,
        require_safety_validation=True,
    ):
        self.sample_rate = sample_rate
        self.n_stim = n_stim_channels
        self.n_rec = n_rec_channels
        self.evoked_samples = int(evoked_window_ms * sample_rate / 1000)
        self.blank_samples = int(artifact_blank_ms * sample_rate / 1000)
        self.spike_thr_factor = spike_threshold_factor

        self.client = IntanTCPClient(
            host=host,
            command_port=command_port,
            waveform_port=waveform_port,
        )

        self.dashboard = dashboard
        self.dry_run = dry_run
        self.require_safety_validation = require_safety_validation

        # Stim safety counters
        self.total_stims_sent = 0
        self.total_charge_delivered_nc = 0.0

        # Noise floor estimate (updated from baseline recording)
        self.noise_std = None

        # Timing log
        self.timing_log = []

        # Output directory
        self.out_dir = None

        if dry_run:
            print("[INFO] RealtimeLoop in DRY-RUN mode: no stim commands will be sent.")
        else:
            print("[!!] RealtimeLoop in LIVE mode: stim commands WILL be sent.")
            print("[!!] Confirm oscilloscope verification has been done.")

    def connect(self):
        """Establish TCP connections."""
        self.client.connect_command(timeout=10.0)
        self.client.connect_waveform(timeout=10.0)
        rate = self.client.get_sample_rate()
        print(f"Connected. Sample rate: {rate} Hz")
        return self

    def disconnect(self):
        self.client.disconnect()

    def _timestamp(self):
        return time.perf_counter()

    def _log_timing(self, trial_id, stage, t_start):
        elapsed_ms = (self._timestamp() - t_start) * 1000
        self.timing_log.append({
            "trial": trial_id,
            "stage": stage,
            "elapsed_ms": round(elapsed_ms, 2),
            "wall_time": datetime.now().isoformat(),
        })
        return elapsed_ms

    # ── Core real-time operations ──────────────────────────────

    def estimate_noise_floor(self, duration_s=2.0):
        """
        Sample a short waveform segment to estimate per-channel noise floor.
        Must be called before spike detection.

        Important: this method does NOT control RHX run state. The caller is
        responsible for ensuring RHX is already in Run or Record mode and
        actively streaming. Previously this method toggled run/stop, but on
        real hardware the TCP `Set RunMode Record` after a Stop sometimes
        gets dropped (see MAINTENANCE.md §4c/§4e), breaking everything.
        Leaving RHX in whatever state the caller set it to is more robust.
        """
        n_samples = int(duration_s * self.sample_rate)
        self.client.init_waveform_reader(self.n_rec)
        time.sleep(duration_s + 0.1)  # let the stream buffer accumulate

        try:
            timestamps, data_uv = self.client.read_waveform_samples(n_samples, self.n_rec)
            self.noise_std = np.median(np.abs(data_uv), axis=1) / 0.6745
            print(f"Noise floor estimated: median={np.median(self.noise_std):.1f} µV "
                  f"(per-channel range: {self.noise_std.min():.1f}–{self.noise_std.max():.1f})")
        except Exception as e:
            print(f"[WARN] Noise estimation failed: {e}. Using default 30 µV.")
            self.noise_std = np.full(self.n_rec, 30.0)

        # Do NOT stop RHX here — caller controls run state.

    def stimulate(self, protocol, trial_id=0):
        """
        Send stimulus and return timestamp.

        In dry_run mode, generates commands but does NOT send them.
        Always runs safety validation if require_safety_validation=True.
        """
        # Safety gate (always check; raises SafetyViolation if dangerous)
        if self.require_safety_validation:
            from encoders.stimulus_schema import SafetyViolation
            try:
                protocol.validate_safety()
            except SafetyViolation as e:
                print(f"[SAFETY VIOLATION] Stim refused: {e}")
                if self.dashboard:
                    self.dashboard.on_timing({
                        "trial": trial_id, "stage": "safety_violation",
                        "elapsed_ms": 0.0, "error": str(e),
                    })
                raise

        t0 = self._timestamp()

        if self.dry_run:
            # Generate commands but don't send. Track stats.
            commands = self.client.protocol_to_commands(protocol)
            self._log_timing(trial_id, "stim_config_dry", t0)
            t1 = self._timestamp()
            for ch in range(protocol.n_channels):
                if any(p.channel == ch for p in protocol.pulses):
                    pass  # would have sent ManualStimTriggerPulse
            self._log_timing(trial_id, "stim_trigger_dry", t1)
        else:
            self.client.send_protocol_config(protocol)
            self._log_timing(trial_id, "stim_config", t0)
            t1 = self._timestamp()
            for ch in range(protocol.n_channels):
                if any(p.channel == ch for p in protocol.pulses):
                    self.client.send_command(
                        f"Execute ManualStimTriggerPulse F{ch + 1}",
                        expect_response=False,
                    )
            self._log_timing(trial_id, "stim_trigger", t1)

        # Update safety counters regardless of dry_run
        self.total_stims_sent += protocol.total_pulses
        report = protocol.safety_report()
        self.total_charge_delivered_nc += report["total_charge_nc"]

        return self._timestamp()

    def read_evoked_response(self, trial_id=0):
        """
        Read evoked response from waveform TCP stream.
        Blanks the artifact period and returns clean data in microvolts.

        read_waveform_samples returns (timestamps, wave_uv) where wave_uv is
        already in float32 microvolts (verified offset-binary conversion).
        """
        t0 = self._timestamp()
        total_samples = self.blank_samples + self.evoked_samples

        try:
            timestamps, data_uv = self.client.read_waveform_samples(
                total_samples, self.n_rec
            )
            clean = data_uv[:, self.blank_samples:]
        except Exception as e:
            print(f"  [WARN] Read failed: {e}. Using zeros.")
            clean = np.zeros((self.n_rec, self.evoked_samples), dtype=np.float32)

        self._log_timing(trial_id, "read_evoked", t0)
        return clean

    def detect_spikes(self, data_uv, trial_id=0):
        """Real-time spike detection using negative threshold."""
        t0 = self._timestamp()
        spike_raster = np.zeros_like(data_uv, dtype=np.float32)

        if self.noise_std is None:
            self.noise_std = np.full(self.n_rec, 30.0)

        for ch in range(data_uv.shape[0]):
            thr = -self.spike_thr_factor * self.noise_std[min(ch, len(self.noise_std)-1)]
            crossings = np.where(np.diff((data_uv[ch] < thr).astype(int)) == 1)[0]
            spike_raster[ch, crossings] = 1.0

        self._log_timing(trial_id, "spike_detect", t0)
        return spike_raster

    def extract_features(self, spike_raster, trial_id=0):
        """Real-time feature extraction."""
        t0 = self._timestamp()
        feat = extract_timebin_features(
            spike_raster, bin_size_ms=100.0, sample_rate_hz=self.sample_rate
        )
        self._log_timing(trial_id, "feature_extract", t0)
        return feat

    def decode_snn(self, spike_raster, model, trial_id=0):
        """Real-time SNN decoding."""
        t0 = self._timestamp()

        n_ch, n_samples = spike_raster.shape
        n_timesteps = 40
        bin_size = max(1, n_samples // n_timesteps)
        binned = np.zeros((n_timesteps, n_ch), dtype=np.float32)
        for t in range(n_timesteps):
            start = t * bin_size
            end = min(start + bin_size, n_samples)
            binned[t] = spike_raster[:, start:end].sum(axis=1)

        spike_input = torch.from_numpy(binned).unsqueeze(0)

        with torch.no_grad():
            spike_counts, _ = model(spike_input)
            predicted = spike_counts.argmax(dim=1).item()

        self._log_timing(trial_id, "snn_decode", t0)
        return predicted, spike_counts[0].numpy()

    def generate_feedback(self, predicted, correct_label, trial_id=0):
        """
        Generate feedback stimulus based on decode result.
        Kagan-style: correct → predictable, incorrect → unpredictable.
        """
        t0 = self._timestamp()
        is_correct = (predicted == correct_label)

        if is_correct:
            # Predictable feedback: regular pulses on channel 0
            pulses = [Pulse(channel=0, time_ms=t) for t in [0, 200, 400]]
        else:
            # Unpredictable feedback: random channels and timing
            rng = np.random.default_rng()
            pulses = [
                Pulse(
                    channel=int(rng.integers(0, self.n_stim)),
                    time_ms=float(rng.uniform(0, 600)),
                )
                for _ in range(3)
            ]
            pulses.sort(key=lambda p: (p.time_ms, p.channel))
            # Enforce minimum interval
            filtered = [pulses[0]]
            for p in pulses[1:]:
                if all(abs(p.time_ms - f.time_ms) >= 200 or p.channel != f.channel
                       for f in filtered):
                    filtered.append(p)
            pulses = filtered

        proto = StimulusProtocol(
            pulses=pulses, n_channels=self.n_stim, duration_ms=2000.0,
        )
        try:
            proto.validate()
        except ValueError:
            proto = StimulusProtocol(
                pulses=[Pulse(channel=0, time_ms=0.0)],
                n_channels=self.n_stim, duration_ms=2000.0,
            )
            proto.validate()

        self._log_timing(trial_id, "gen_feedback", t0)
        return proto, is_correct

    # ── High-level experiment modes ────────────────────────────

    def run_single_trial(self, protocol, model, correct_label, trial_id=0):
        """
        Execute one complete trial:
        stimulate → read → detect → decode → feedback.

        Returns dict with all results and timing.
        """
        t_trial_start = self._timestamp()

        # 1. Stimulate
        self.stimulate(protocol, trial_id)

        # 2. Read evoked response
        evoked = self.read_evoked_response(trial_id)

        # 3. Spike detection
        spikes = self.detect_spikes(evoked, trial_id)

        # 4. Feature extraction
        features = self.extract_features(spikes, trial_id)

        # 5. SNN decode
        predicted, spike_counts = self.decode_snn(spikes, model, trial_id)

        total_ms = (self._timestamp() - t_trial_start) * 1000

        n_spikes = int(spikes.sum())

        result = {
            "trial": trial_id,
            "correct_label": int(correct_label),
            "predicted": int(predicted),
            "is_correct": predicted == correct_label,
            "n_spikes_detected": n_spikes,
            "spike_counts": spike_counts.tolist(),
            "total_latency_ms": round(total_ms, 2),
            "timestamp": datetime.now().isoformat(),
        }

        return result, spikes

    def run_classification_loop(
        self,
        images_by_class,
        encoder_pipeline,
        snn_model,
        n_trials=30,
        isi_s=5.0,
        enable_feedback=False,
    ):
        """
        Run the main real-time classification loop.

        Args:
            images_by_class: {label: [image_tensor, ...]}
            encoder_pipeline: EncodingPipeline instance
            snn_model: trained or untrained SNNDecoder
            n_trials: total number of trials
            isi_s: inter-trial interval
            enable_feedback: whether to send feedback stimulation
        """
        self.out_dir = Path(f"data/realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        labels = list(images_by_class.keys())
        all_protocols = {}
        for label in labels:
            all_protocols[label] = []
            for img in images_by_class[label]:
                proto = encoder_pipeline.encode_to_protocol(img)
                all_protocols[label].append(proto)

        # Build trial sequence
        rng = np.random.default_rng(42)
        trial_sequence = []
        for _ in range(n_trials):
            label = int(rng.choice(labels))
            img_idx = int(rng.integers(0, len(all_protocols[label])))
            trial_sequence.append((label, img_idx))

        # Estimate noise floor
        print("Estimating noise floor...")
        self.estimate_noise_floor(duration_s=2.0)

        # ── RHX recording state is controlled by the USER (manually) ──
        # On real hardware TCP `Set RunMode Record` and `Set RunMode Stop`
        # are unreliable (see MAINTENANCE.md §4c/§4e). The robust pattern
        # is: user manually clicks ● Record before launching the script,
        # then manually clicks ■ Stop after it finishes. The script just
        # snapshots data/ at entry and diffs it at exit to find the RHX
        # subdir created during this run.
        rhx_root = _get_rhx_save_root(self.client, fallback="data")
        rhx_dir_before = set(os.listdir(rhx_root)) if Path(rhx_root).exists() else set()
        print(f"[INFO] Assuming user manually started Record mode in RHX.")
        print(f"[INFO] Will diff against {len(rhx_dir_before)} existing dirs in {rhx_root}.")

        print(f"\n{'='*60}")
        print(f"REAL-TIME LOOP: {n_trials} trials, ISI={isi_s}s, "
              f"feedback={'ON' if enable_feedback else 'OFF'}")
        print(f"{'='*60}\n")

        # Notify dashboard
        if self.dashboard:
            self.dashboard.on_experiment_start(
                config={"encoding": "snn", "isi_s": isi_s,
                        "feedback": enable_feedback},
                total_trials=n_trials,
            )

        results = []
        correct_count = 0

        for trial_idx, (label, img_idx) in enumerate(trial_sequence):
            # Check if dashboard requested stop
            if self.dashboard and self.dashboard.should_stop():
                print("\n[STOP] Dashboard requested stop.")
                break

            proto = all_protocols[label][img_idx]

            # Push protocol visualization to dashboard
            if self.dashboard:
                self.dashboard.on_protocol({
                    "binary_matrix": proto.to_binary_matrix(10).tolist(),
                    "total_pulses": proto.total_pulses,
                })

            # Run trial
            result, spikes = self.run_single_trial(
                proto, snn_model, label, trial_id=trial_idx
            )
            results.append(result)

            if result["is_correct"]:
                correct_count += 1

            # Print live status
            running_acc = correct_count / (trial_idx + 1)
            status = "+" if result["is_correct"] else "-"
            print(
                f"  Trial {trial_idx+1:3d}/{n_trials} | "
                f"true={label} pred={result['predicted']} {status} | "
                f"spikes={result['n_spikes_detected']:4d} | "
                f"latency={result['total_latency_ms']:6.1f}ms | "
                f"acc={running_acc:.1%}"
            )

            # Push to dashboard
            if self.dashboard:
                self.dashboard.on_trial_complete(result)
                # Send per-channel spike summary
                ch_spikes = spikes.sum(axis=1).tolist()
                self.dashboard.on_spikes({"channel_counts": ch_spikes})

            # Feedback stimulation
            if enable_feedback:
                fb_proto, was_correct = self.generate_feedback(
                    result["predicted"], label, trial_idx
                )
                self.stimulate(fb_proto, trial_idx)
                time.sleep(0.5)

            # Inter-trial interval
            time.sleep(isi_s)

        # Loop done. Do NOT send Stop — let the user manually click ■ Stop
        # in RHX (TCP Stop is unreliable; see MAINTENANCE.md §4e).
        print(f"\n[ACTION REQUIRED] Loop complete. Please click ■ Stop in RHX now")
        print(f"                   so the recording finalizes to disk, then ENTER.")
        try:
            input("                   Press ENTER after you've clicked Stop: ")
        except EOFError:
            time.sleep(5.0)  # non-interactive: give RHX 5s to flush

        # Now find the RHX subdir that was created during this run
        rhx_dir = _find_new_rhx_subdir(rhx_root, rhx_dir_before)
        print(f"RHX wrote to: {rhx_dir}")

        # Save results
        final_acc = correct_count / n_trials
        summary = {
            "n_trials": n_trials,
            "n_correct": correct_count,
            "accuracy": final_acc,
            "chance": 1.0 / len(labels),
            "isi_s": isi_s,
            "feedback_enabled": enable_feedback,
            "rhx_dir": str(rhx_dir) if rhx_dir else None,
            "trials": results,
            "timing_log": self.timing_log,
        }

        with open(self.out_dir / "realtime_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Also copy realtime_results.json + config into the RHX dir so
        # verify_stim_dat.py and other post-processing can find everything
        # in one place.
        _attach_metadata_to_rhx_dir(
            rhx_dir, self.out_dir,
            files=("realtime_results.json",),
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"COMPLETE: {correct_count}/{n_trials} correct ({final_acc:.1%})")
        print(f"Chance: {1.0/len(labels):.1%}")

        latencies = [r["total_latency_ms"] for r in results]
        print(f"Latency: mean={np.mean(latencies):.1f}ms, "
              f"max={np.max(latencies):.1f}ms, "
              f"min={np.min(latencies):.1f}ms")
        print(f"Results saved to {self.out_dir}")
        print(f"{'='*60}")

        if self.dashboard:
            self.dashboard.on_experiment_end({
                "accuracy": final_acc,
                "n_correct": correct_count,
                "n_trials": len(results),
            })

        log_experiment(
            script="realtime_loop.py",
            args={
                "n_trials": int(n_trials),
                "isi_s": float(isi_s),
                "enable_feedback": bool(enable_feedback),
                "n_rec_channels": int(self.n_rec),
                "n_stim_channels": int(self.n_stim),
                "dry_run": bool(self.dry_run),
            },
            verdict="COMPLETE",
            summary={
                "n_trials_done": len(results),
                "n_correct": int(correct_count),
                "accuracy": round(float(final_acc), 4),
                "chance": round(1.0 / len(labels), 4),
                "latency_mean_ms": round(float(np.mean(latencies)), 2),
                "latency_max_ms": round(float(np.max(latencies)), 2),
                "total_stims_sent": int(self.total_stims_sent),
                "total_charge_nc": round(float(self.total_charge_delivered_nc), 2),
                "rhx_dir": str(rhx_dir) if rhx_dir else None,
            },
            data=str(rhx_dir) if rhx_dir else str(self.out_dir),
        )

        return summary


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time closed-loop engine")
    parser.add_argument("--mode", required=True,
                        choices=["demo", "classify", "closedloop"],
                        help="demo: test pipeline with demo data; "
                             "classify: real classification; "
                             "closedloop: classification + feedback")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cmd-port", type=int, default=5000)
    parser.add_argument("--wave-port", type=int, default=5001)
    parser.add_argument("--n-rec", type=int, default=8,
                        help="Number of recording channels streamed via TCP")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--isi", type=float, default=5.0)
    parser.add_argument("--encoding", default="delayed_phase",
                        choices=["delayed_phase", "lif", "rate"])
    args = parser.parse_args()

    # Build components
    cnn = CNNFeatureExtractor(n_channels=8, n_classes=10)
    pipeline = EncodingPipeline(cnn, encoder_type=args.encoding)
    snn_model = SNNDecoder(n_input=args.n_rec, n_classes=3)

    # Generate test images
    images_by_class = {}
    for digit in [0, 1, 7]:
        try:
            from torchvision import datasets, transforms
            transform = transforms.Compose([transforms.ToTensor()])
            mnist = datasets.MNIST(root="./data/mnist", train=True,
                                   download=True, transform=transform)
            idxs = [i for i, (_, l) in enumerate(mnist) if l == digit][:2]
            images_by_class[digit] = [mnist[i][0] for i in idxs]
        except Exception:
            images_by_class[digit] = [
                torch.rand(1, 28, 28) * (0.3 + 0.05 * digit)
                for _ in range(2)
            ]

    # Run
    engine = RealtimeLoop(
        host=args.host,
        command_port=args.cmd_port,
        waveform_port=args.wave_port,
        n_rec_channels=args.n_rec,
    )

    try:
        engine.connect()

        enable_feedback = (args.mode == "closedloop")
        summary = engine.run_classification_loop(
            images_by_class=images_by_class,
            encoder_pipeline=pipeline,
            snn_model=snn_model,
            n_trials=args.n_trials,
            isi_s=args.isi,
            enable_feedback=enable_feedback,
        )

    except ConnectionRefusedError:
        print("[FAIL] Cannot connect. Is Intan RHX running with TCP enabled?")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        engine.disconnect()


if __name__ == "__main__":
    main()
