from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from difflib import unified_diff
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline.pi_sanitizer_metrics import removed_fragments
from agentdojo.agent_pipeline.pi_sanitizer_optimization import (
    SanitizerExample,
    SanitizerPrediction,
    score_sanitizer_prediction,
)


def build_comparison_row(
    example: SanitizerExample,
    seed_prediction: SanitizerPrediction,
    optimized_prediction: SanitizerPrediction,
) -> dict[str, Any]:
    seed_metric = score_sanitizer_prediction(example, seed_prediction)
    optimized_metric = score_sanitizer_prediction(example, optimized_prediction)
    sas_delta = optimized_metric.score - seed_metric.score
    return {
        "id": example.sample_id,
        "tool_name": example.tool_name,
        "user_instruction": example.user_instruction,
        "gold_is_injection": example.is_injection,
        "attack_type": example.attack_type,
        "decoy": example.decoy,
        "seed_metrics": seed_metric.components,
        "optimized_metrics": optimized_metric.components,
        "sas_delta": sas_delta,
        "winner": choose_winner(seed_metric.score, optimized_metric.score),
        "seed_feedback": seed_metric.feedback,
        "optimized_feedback": optimized_metric.feedback,
        "tool_output": example.tool_output,
        "gold_filtered_tool_output": example.gold_filtered_tool_output,
        "seed_filtered_tool_output": seed_prediction.filtered_tool_output,
        "optimized_filtered_tool_output": optimized_prediction.filtered_tool_output,
        "seed_removed_fragments": removed_fragments(example.tool_output, seed_prediction.filtered_tool_output),
        "optimized_removed_fragments": removed_fragments(
            example.tool_output, optimized_prediction.filtered_tool_output
        ),
    }


def choose_winner(seed_score: float, optimized_score: float, epsilon: float = 1e-12) -> str:
    if optimized_score - seed_score > epsilon:
        return "optimized"
    if seed_score - optimized_score > epsilon:
        return "seed"
    return "tie"


def summarize_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_examples": 0,
            "seed_mean_sas": 0.0,
            "optimized_mean_sas": 0.0,
            "delta_sas": 0.0,
            "seed_exact_match_rate": 0.0,
            "optimized_exact_match_rate": 0.0,
            "seed_detection_accuracy": 0.0,
            "optimized_detection_accuracy": 0.0,
            "seed_over_filter_rate": 0.0,
            "optimized_over_filter_rate": 0.0,
            "seed_under_filter_rate": 0.0,
            "optimized_under_filter_rate": 0.0,
            "seed_residual_injection_rate": 0.0,
            "optimized_residual_injection_rate": 0.0,
            "optimized_wins": 0,
            "seed_wins": 0,
            "ties": 0,
        }

    seed_mean_sas = _mean_metric(rows, "seed_metrics", "sas")
    optimized_mean_sas = _mean_metric(rows, "optimized_metrics", "sas")
    return {
        "num_examples": len(rows),
        "seed_mean_sas": seed_mean_sas,
        "optimized_mean_sas": optimized_mean_sas,
        "delta_sas": optimized_mean_sas - seed_mean_sas,
        "seed_exact_match_rate": _mean_metric(rows, "seed_metrics", "exact_match"),
        "optimized_exact_match_rate": _mean_metric(rows, "optimized_metrics", "exact_match"),
        "seed_detection_accuracy": _mean_metric(rows, "seed_metrics", "derived_detection_correct"),
        "optimized_detection_accuracy": _mean_metric(rows, "optimized_metrics", "derived_detection_correct"),
        "seed_over_filter_rate": _mean_metric(rows, "seed_metrics", "over_filter"),
        "optimized_over_filter_rate": _mean_metric(rows, "optimized_metrics", "over_filter"),
        "seed_under_filter_rate": _mean_metric(rows, "seed_metrics", "under_filter"),
        "optimized_under_filter_rate": _mean_metric(rows, "optimized_metrics", "under_filter"),
        "seed_residual_injection_rate": _mean_metric(rows, "seed_metrics", "residual_injection"),
        "optimized_residual_injection_rate": _mean_metric(rows, "optimized_metrics", "residual_injection"),
        "optimized_wins": sum(1 for row in rows if row["winner"] == "optimized"),
        "seed_wins": sum(1 for row in rows if row["winner"] == "seed"),
        "ties": sum(1 for row in rows if row["winner"] == "tie"),
    }


def write_comparison_artifacts(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_comparison_rows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output_dir / "predictions.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (output_dir / "diffs.md").write_text(render_diffs_markdown(rows))


def render_diffs_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    sections = ["# PI Sanitizer Prompt Comparison", ""]
    selected_rows = [row for row in rows if _should_render_row(row)]
    if not selected_rows:
        sections.append("No non-tie or diagnostic-risk examples to render.")
    for row in selected_rows:
        sections.extend(_render_row_markdown(row))
    return "\n".join(sections).rstrip() + "\n"


def _should_render_row(row: Mapping[str, Any]) -> bool:
    if row["winner"] != "tie":
        return True
    return any(
        float(row[metrics_key][metric_name]) >= 0.5
        for metrics_key in ("seed_metrics", "optimized_metrics")
        for metric_name in ("over_filter", "under_filter", "residual_injection")
    ) or any(float(row[metrics_key]["exact_match"]) < 0.5 for metrics_key in ("seed_metrics", "optimized_metrics"))


def _render_row_markdown(row: Mapping[str, Any]) -> list[str]:
    seed_metrics = row["seed_metrics"]
    optimized_metrics = row["optimized_metrics"]
    lines = [
        f"## {row['id']} | {row['winner']} | {row['tool_name']}",
        "",
        f"- gold_is_injection: {_bool_text(row['gold_is_injection'])}",
        f"- seed_sas: {float(seed_metrics['sas']):.4f}",
        f"- optimized_sas: {float(optimized_metrics['sas']):.4f}",
        f"- sas_delta: {float(row.get('sas_delta', 0.0)):+.4f}",
        f"- seed_exact_match: {_metric_bool_text(seed_metrics, 'exact_match')}",
        f"- optimized_exact_match: {_metric_bool_text(optimized_metrics, 'exact_match')}",
        f"- seed_detection_correct: {_metric_bool_text(seed_metrics, 'derived_detection_correct')}",
        f"- optimized_detection_correct: {_metric_bool_text(optimized_metrics, 'derived_detection_correct')}",
        f"- seed_over_filter: {_metric_bool_text(seed_metrics, 'over_filter')}",
        f"- optimized_over_filter: {_metric_bool_text(optimized_metrics, 'over_filter')}",
        f"- seed_under_filter: {_metric_bool_text(seed_metrics, 'under_filter')}",
        f"- optimized_under_filter: {_metric_bool_text(optimized_metrics, 'under_filter')}",
        f"- seed_residual_injection: {_metric_bool_text(seed_metrics, 'residual_injection')}",
        f"- optimized_residual_injection: {_metric_bool_text(optimized_metrics, 'residual_injection')}",
        "",
        "### Input vs Gold",
        "",
        _diff_block(row["tool_output"], row["gold_filtered_tool_output"], "tool_output", "gold"),
        "",
        "### Seed vs Gold",
        "",
        _diff_block(row["seed_filtered_tool_output"], row["gold_filtered_tool_output"], "seed", "gold"),
        "",
        "### Optimized vs Gold",
        "",
        _diff_block(row["optimized_filtered_tool_output"], row["gold_filtered_tool_output"], "optimized", "gold"),
        "",
        "### Seed Removed Fragments",
        "",
        _text_list_block(row.get("seed_removed_fragments", [])),
        "",
        "### Optimized Removed Fragments",
        "",
        _text_list_block(row.get("optimized_removed_fragments", [])),
        "",
    ]
    return lines


def _diff_block(left: str, right: str, fromfile: str, tofile: str) -> str:
    diff = "".join(
        unified_diff(
            left.splitlines(keepends=True),
            right.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )
    if not diff:
        diff = "(no diff)\n"
    return f"```diff\n{diff}```"


def _text_list_block(values: Sequence[str]) -> str:
    if not values:
        return "```text\n(no removed fragments)\n```"
    return "```text\n" + "\n---\n".join(values) + "\n```"


def _mean_metric(rows: Sequence[Mapping[str, Any]], metrics_key: str, metric_name: str) -> float:
    return sum(float(row[metrics_key][metric_name]) for row in rows) / len(rows)


def _metric_bool_text(metrics: Mapping[str, Any], key: str) -> str:
    return _bool_text(float(metrics[key]) >= 0.5)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
