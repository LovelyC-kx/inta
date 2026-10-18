"""
Master experiment log — append-only JSONL.

Every experiment-running script (stress test, demo pipeline, pretraining,
2D-culture phases, realtime loop, offline analysis) appends one record at
the end of its run. This gives a single chronological audit trail that
answers "what was run, when, with what args, what was the verdict, where
is the detailed data" without needing to dig through `data/` subdirectories.

Log location: `experiment_log.jsonl` at the repo root.

Each record is one JSON object, one line. Schema:
    {
      "ts":      "ISO 8601 local time",
      "script":  "filename.py",
      "args":    {... vars(argparse.Namespace) ...},
      "verdict": "PASS" | "FAIL" | "STREAM STABLE" | etc.,
      "summary": {... script-specific scalar/short metrics ...},
      "data":    "data/<subdir>/" or "<file>"   (optional)
      ...                                       (any extra keys allowed)
    }

CLI quick-view:
    python utils/experiment_log.py --tail 10
    python utils/experiment_log.py --filter script=stream_stress_test.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Resolve repo root from this file's location (utils/ -> repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "experiment_log.jsonl"


def _jsonable(obj):
    """Best-effort coercion of common non-JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


def log_experiment(script, args=None, verdict=None, summary=None,
                   data=None, **extra):
    """
    Append one experiment record to the master log.

    Args:
        script:  Name of the script (typically `Path(__file__).name`).
        args:    Dict of arguments (use `vars(argparse_namespace)`).
        verdict: One-line outcome ("PASS", "STREAM STABLE", "FAIL: ...", etc.).
        summary: Dict of short scalar metrics relevant to this experiment.
        data:    Path to detailed-data file or directory (str or Path).
        **extra: Any additional keys merged into the record.

    Returns:
        The record dict that was written.
    """
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "script": script,
    }
    if args is not None:
        record["args"] = _jsonable(args)
    if verdict is not None:
        record["verdict"] = verdict
    if summary is not None:
        record["summary"] = _jsonable(summary)
    if data is not None:
        record["data"] = str(data)
    for k, v in extra.items():
        record[k] = _jsonable(v)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_log(n_tail=None, filters=None):
    """
    Read records from the master log.

    Args:
        n_tail:  If set, return only the last N records.
        filters: Dict of {key: expected_value} — return only records where
                 record[key] == expected_value (or record[key] contains it
                 if key is in record["args"]/record["summary"]).

    Returns:
        List of record dicts in chronological order.
    """
    if not LOG_PATH.exists():
        return []
    records = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if filters:
        def matches(r):
            for k, v in filters.items():
                if r.get(k) == v:
                    continue
                if isinstance(r.get("args"), dict) and r["args"].get(k) == v:
                    continue
                if isinstance(r.get("summary"), dict) and r["summary"].get(k) == v:
                    continue
                return False
            return True
        records = [r for r in records if matches(r)]

    if n_tail is not None:
        records = records[-n_tail:]
    return records


def _format_record_short(r):
    parts = [r.get("ts", "?"), r.get("script", "?")]
    if "verdict" in r:
        parts.append(f"verdict={r['verdict']}")
    if "summary" in r and isinstance(r["summary"], dict):
        # show first 2 summary keys
        kv = list(r["summary"].items())[:2]
        for k, v in kv:
            parts.append(f"{k}={v}")
    if "data" in r:
        parts.append(f"data={r['data']}")
    return " | ".join(str(p) for p in parts)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tail", type=int, default=None,
                        help="Show only the last N records")
    parser.add_argument("--filter", action="append", default=[],
                        help="Filter as key=value (repeatable)")
    parser.add_argument("--full", action="store_true",
                        help="Print full JSON of each record instead of one-line summary")
    parser.add_argument("--path", action="store_true",
                        help="Print log file path and exit")
    args = parser.parse_args()

    if args.path:
        print(LOG_PATH)
        return

    filters = {}
    for f in args.filter:
        if "=" not in f:
            print(f"[WARN] Skipping malformed filter: {f}", file=sys.stderr)
            continue
        k, v = f.split("=", 1)
        filters[k] = v

    records = read_log(n_tail=args.tail, filters=filters or None)
    if not records:
        print(f"(empty log at {LOG_PATH})")
        return

    for r in records:
        if args.full:
            print(json.dumps(r, ensure_ascii=False, indent=2))
        else:
            print(_format_record_short(r))


if __name__ == "__main__":
    main()
