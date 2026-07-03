from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline.pi_sanitizer_log_analysis import analyze_trace_paths, summarize_sanitizer_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PI sanitizer events stored in AgentDyn trace logs.")
    parser.add_argument("--logdir", type=Path, default=Path("runs"))
    parser.add_argument("--pipeline", type=str, default=None)
    parser.add_argument("--suites", nargs="*", default=None)
    parser.add_argument("--attack", type=str, default="important_instructions")
    parser.add_argument("--output", type=Path, default=Path("runs/pi_sanitizer_agentdyn_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_paths = discover_trace_paths(args.logdir, pipeline=args.pipeline, suites=args.suites, attack=args.attack)
    rows = analyze_trace_paths(trace_paths)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summarize_sanitizer_events(rows), indent=2))
    write_jsonl(args.output / "events.jsonl", rows)
    write_grouped_csv(args.output / "by_suite.csv", rows, "suite_name")
    write_grouped_csv(args.output / "by_tool.csv", rows, "tool_name")
    write_failures_markdown(args.output / "failures.md", rows)
    print(f"Wrote PI sanitizer analysis artifacts to {args.output}")


def discover_trace_paths(
    logdir: Path,
    *,
    pipeline: str | None,
    suites: list[str] | None,
    attack: str | None,
) -> list[Path]:
    paths = []
    for path in sorted(logdir.rglob("*.json")):
        parts = path.relative_to(logdir).parts
        if len(parts) != 5:
            continue
        path_pipeline, path_suite, _user_task, path_attack, _filename = parts
        if pipeline is not None and path_pipeline != pipeline:
            continue
        if suites is not None and path_suite not in set(suites):
            continue
        if attack is not None and path_attack != attack:
            continue
        paths.append(path)
    return paths


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_grouped_csv(path: Path, rows: list[dict[str, Any]], key: str) -> None:
    fieldnames = [
        key,
        "num_events",
        "num_injection_events",
        "cleaning_success_rate",
        "missed_injection_rate",
        "over_filter_proxy_rate",
        "sanitizer_detection_accuracy",
        "mean_sas",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, group_rows in sorted(grouped.items()):
            summary = summarize_sanitizer_events(group_rows)
            writer.writerow({key: group_name, **summary})


def write_failures_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    failures = [
        row
        for row in rows
        if row["missed_injection"] or row["over_filter_proxy"] or not row.get("api_ok") or not row.get("parse_ok")
    ]
    lines = ["# PI Sanitizer Failures", ""]
    if not failures:
        lines.append("No sanitizer failures found.")
    for row in failures:
        lines.extend(
            [
                f"## {row['suite_name']} | {row['user_task_id']} | {row['injection_task_id']} | {row['tool_name']}",
                "",
                f"- trace_path: `{row['trace_path']}`",
                f"- missed_injection: {str(row['missed_injection']).lower()}",
                f"- over_filter_proxy: {str(row['over_filter_proxy']).lower()}",
                f"- api_ok: {str(row.get('api_ok')).lower()}",
                f"- parse_ok: {str(row.get('parse_ok')).lower()}",
                f"- sas: {row.get('sas')}",
                "",
                "### Original",
                "",
                "```text",
                row["original_tool_output"],
                "```",
                "",
                "### Filtered",
                "",
                "```text",
                row["filtered_tool_output"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
