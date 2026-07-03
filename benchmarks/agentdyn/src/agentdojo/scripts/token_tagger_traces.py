from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_tagger_traces(
    logdir: str | Path,
    *,
    defense: str,
    since_ns: int,
) -> dict[str, int]:
    summary = {
        "trace_files": 0,
        "events": 0,
        "failures": 0,
    }
    for path in Path(logdir).rglob("*.json"):
        if path.stat().st_mtime_ns < since_ns:
            continue
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(trace, dict):
            continue
        matching_events = [
            event
            for event in trace.get("tagger_defense_events", [])
            if isinstance(event, dict) and event.get("defense") == defense
        ]
        if not matching_events:
            continue
        summary["trace_files"] += 1
        summary["events"] += len(matching_events)
        summary["failures"] += sum(event.get("success") is False for event in matching_events)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check token tagger benchmark traces for inference failures.")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--defense", required=True)
    parser.add_argument("--since-ns", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary: dict[str, Any] = summarize_tagger_traces(
        args.logdir,
        defense=args.defense,
        since_ns=args.since_ns,
    )
    print(json.dumps(summary, sort_keys=True))
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
