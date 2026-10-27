"""Unified open loop: simulate/collect/train/run/replay/demo. See docs/OPENLOOP.md."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.openloop_backend import RHXBackend, SyntheticBackend
from experiments.openloop_data import FeatureSpec, FrozenDecoder, train_from_sessions
from experiments.openloop_engine import ImageEncoder, run_session
from experiments.openloop_replay import replay_session


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_inputs(args):
    simulated = args.command == "simulate"
    if not simulated and not args.live_stim:
        raise ValueError("Physical collect/run requires explicit --live-stim")
    decoder = FrozenDecoder.load(args.decoder) if args.decoder else None
    actions = read_json(args.actions) if args.actions else None
    if args.command == "run" and decoder is None:
        raise ValueError("run requires a trained --decoder and --actions")
    if args.live_arm and (simulated or decoder is None):
        raise ValueError("Physical arm output requires real frozen-decoder inference")
    model_encoder = {} if decoder is None else decoder.contract["encoder"]
    prefix = args.stim_prefix or model_encoder.get("stim_prefix", "A")
    encoding = args.encoding or model_encoder.get("encoding", "delayed_phase")
    params = read_json(args.encoder_params) if args.encoder_params else model_encoder.get("encoder_params", {})
    seed = args.seed if args.seed is not None else model_encoder.get("seed", 42)
    encoder = ImageEncoder(args.cnn_checkpoint, encoding=encoding, seed=seed,
                           simulated=simulated, encoder_params=params,
                           pulse_profile=read_json(args.pulse_profile) if args.pulse_profile else model_encoder.get("pulse_profile", {}))
    trials = encoder.manifest(args.manifest)
    if args.shuffle:
        import numpy as np
        np.random.default_rng(seed).shuffle(trials)
    spec = FeatureSpec(**(read_json(args.features) if args.features else
                         {} if decoder is None else decoder.contract["features"]))
    channels = args.channels or ([f"A-{i:03d}" for i in range(8)] if decoder is None else decoder.contract["channel_names"])
    if simulated:
        backend = partial(SyntheticBackend, prefix=prefix)
    else:
        if not args.timing_calibration or not args.rhx_path:
            raise ValueError("Physical acquisition requires --timing-calibration and --rhx-path")
        backend = partial(RHXBackend, timing=read_json(args.timing_calibration), rhx_path=args.rhx_path,
                          prefix=prefix, host=args.host, command_port=args.cmd_port,
                          waveform_port=args.wave_port, spike_port=args.spike_port, ttl_bit=args.ttl_bit,
                          preparation_id=args.preparation_id)
    return run_session(trials, backend, args.out, channels, encoder.metadata, spec=spec, decoder=decoder,
                       action_map=actions, live_arm=args.live_arm, arm_port=args.arm_port, home=args.home,
                       min_confidence=args.min_confidence)


def demo(output):
    """Reproducible synthetic wiring exercise, including frozen inference and arm."""
    import numpy as np
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    rng = np.random.default_rng(42)
    for label in range(3):
        for repeat in range(6):
            image = rng.uniform(0, .15, (28, 28)).astype(np.float32)
            image[3 + label * 7:8 + label * 7, 3 + repeat:22 + repeat] = .8
            path = output / f"image_{label}_{repeat}.npy"
            np.save(path, image)
            records.append({"trial_id": f"image-{label}-{repeat}", "image": path.name,
                            "label": str(label), "group": "heldout" if repeat >= 4 else "train"})
    manifest = output / "inputs.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    encoder = ImageEncoder(simulated=True, encoder_params={"t_max_ms": 200})
    trials = encoder.manifest(manifest)
    channels = [f"A-{i:03d}" for i in range(8)]
    spec = FeatureSpec(duration_ms=500, n_bins=20)
    collected = run_session(trials, SyntheticBackend, output / "collection", channels, encoder.metadata, spec=spec)
    report = train_from_sessions([output / "collection"], output / "decoder.json", ["heldout"])
    decoder = FrozenDecoder.load(output / "decoder.json")
    actions = {"0": ["left"], "1": ["right"], "2": ["grab", "release"]}
    (output / "actions.json").write_text(json.dumps(actions, indent=2), encoding="utf-8")
    inferred = run_session([t for t in trials if t.group == "heldout"], SyntheticBackend, output / "inference",
                           channels, encoder.metadata, spec=spec, decoder=decoder, action_map=actions,
                           min_confidence=0)  # Explicit wiring demo; not an acceptance threshold.
    replayed = replay_session(output / "inference", decoder=decoder, action_map=actions, min_confidence=0)
    result = {"scope": "SYNTHETIC INTERFACE TEST ONLY", "collection": collected["status"],
              "training": report, "inference": inferred["status"], "replay": replayed["status"],
              "arm_task_count": len(inferred["arm_tasks"]), "output": str(output),
              "physical_delivery_verified": False, "physical_task_success": None}
    (output / "demo_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("simulate", "collect", "run"):
        p = commands.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", required=True, help="New local journal directory; never overwritten")
        p.add_argument("--cnn-checkpoint")
        p.add_argument("--encoding", choices=("delayed_phase", "lif", "rate"))
        p.add_argument("--encoder-params", help="JSON encoder keyword parameters")
        p.add_argument("--pulse-profile", help="JSON current/phase/gap/polarity/waveform overrides, before charge trimming")
        p.add_argument("--features", help="JSON FeatureSpec; defaults to decoder contract during inference")
        p.add_argument("--seed", type=int)
        p.add_argument("--shuffle", action="store_true", help="Reproducibly shuffle the input presentation order")
        p.add_argument("--channels", nargs="+", help="Ordered recording native names")
        p.add_argument("--stim-prefix", choices=list("ABCD"))
        p.add_argument("--decoder")
        p.add_argument("--actions")
        p.add_argument("--min-confidence", type=float, default=.8)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--cmd-port", type=int, default=5000)
        p.add_argument("--wave-port", type=int, default=5001)
        p.add_argument("--spike-port", type=int, default=5002)
        p.add_argument("--rhx-path", help="Save path on the RHX host, no whitespace")
        p.add_argument("--preparation-id", help="Stable culture/organoid identity, required for physical acquisition")
        p.add_argument("--timing-calibration", help="JSON timing bounds and operator validation flag")
        p.add_argument("--ttl-bit", type=int, help="Optional zero-based digital input bit; STIM anchors are always recorded")
        p.add_argument("--live-stim", action="store_true")
        p.add_argument("--live-arm", action="store_true")
        p.add_argument("--arm-port")
        p.add_argument("--home", action="store_true", help="Explicit physical arm initialization")
    p = commands.add_parser("train")
    p.add_argument("--sessions", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test-groups", nargs="+", required=True)
    p.add_argument("--kind", choices=("lr", "mlp"), default="lr")
    p = commands.add_parser("replay")
    p.add_argument("--session", required=True)
    p.add_argument("--decoder")
    p.add_argument("--actions")
    p.add_argument("--min-confidence", type=float, default=.8)
    p = commands.add_parser("demo")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo(args.out)
        elif args.command == "train":
            result = train_from_sessions(args.sessions, args.out, args.test_groups, kind=args.kind)
        elif args.command == "replay":
            result = replay_session(args.session, decoder=FrozenDecoder.load(args.decoder) if args.decoder else None,
                                    action_map=read_json(args.actions) if args.actions else None,
                                    min_confidence=args.min_confidence)
        else:
            result = run_inputs(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("status") not in ("FAILED", "NO_VALID_TRIALS") else 1
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
