"""Read one decoder event per JSON line on stdin."""

import argparse
import json
import sys

from .controller import ArmController, SignalAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--home", action="store_true", help="Move to home before accepting events")
    args = parser.parse_args()
    with ArmController(port=args.port, dry_run=not args.live) as arm:
        adapter = SignalAdapter(arm)
        try:
            if args.home:
                print(json.dumps(arm.home()), flush=True)
            for line in sys.stdin:
                try:
                    result = adapter.handle(json.loads(line))
                except (ValueError, RuntimeError, OSError) as exc:
                    result = {"status": "error", "message": str(exc)}
                print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
        finally:
            if not arm.dry_run:
                arm.stop()


if __name__ == "__main__":
    main()
