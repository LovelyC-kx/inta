"""Number-key AM1 bench console; simulation by default, independent of RHX."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "01_basic_motion" / "01_basic_motion"))
from organoid_arm import ArmController


ACTIONS = {"1": "left", "2": "right", "3": "grab", "4": "release",
           "5": "up", "6": "down", "7": "forward", "8": "back",
           "9": "cw", "10": "ccw", "11": "center", "0": "stop"}


def main(argv=None, *, controller_factory=ArmController, read=input):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true", help="Compatibility flag; console is always interactive")
    parser.add_argument("--live", action="store_true", help="Open the physical arm serial port")
    parser.add_argument("--port")
    parser.add_argument("--home", action="store_true", help="Move to home before reading commands")
    parser.add_argument("--step-mm", type=float, default=5)
    parser.add_argument("--move-time-ms", type=int, default=800)
    args = parser.parse_args(argv)
    if args.live and (not args.port or not args.home):
        parser.error("Physical operation requires --live --port PORT --home")
    print("Mode: " + ("PHYSICAL ARM" if args.live else "SIMULATION"), flush=True)
    print("1=left 2=right 3=grab 4=release 5=up 6=down 7=forward 8=back "
          "9=cw 10=ccw 11=center 0=STOP h=home q=quit", flush=True)
    with controller_factory(port=args.port, dry_run=not args.live, step_mm=args.step_mm,
                            move_time_ms=args.move_time_ms) as arm:
        try:
            if args.home or not args.live:
                print(json.dumps(arm.home()), flush=True)
            while True:
                try:
                    key = read("Number: ").strip().lower()
                except EOFError:
                    break
                if key in ("q", "quit", "exit"):
                    break
                if key != "h" and key not in ACTIONS:
                    print("Unknown command; use the menu above.", flush=True)
                    continue
                try:
                    result = arm.home() if key == "h" else arm.execute(ACTIONS[key])
                    print(json.dumps(result, allow_nan=False), flush=True)
                except (ValueError, RuntimeError, OSError) as exc:
                    print(f"Action rejected: {exc}", flush=True)
        except KeyboardInterrupt:
            print("Interrupted.", flush=True)
        finally:
            arm.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
