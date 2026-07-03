"""Inference-time post-processing for tagger predictions, plus an offline
rescoring CLI that re-evaluates saved evaluate.py result JSONs.

Post-processing steps (both optional, applied in this order):
  1. threshold  -- rebuild drop spans from dumped per-token P(drop) values
                   (requires --dump-drop-probabilities at inference time)
  2. bridging   -- merge predicted drop spans separated by <= max_gap chars,
                   closing tokenization-artifact gaps (single spaces/newlines)
                   between two already-flagged spans

Bridging never deletes text that is not flanked by predicted spans on both
sides, so its false-positive risk is limited to clean records that already
carry two near-adjacent false-positive spans.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

from minspan.evaluate import (
    _build_prediction_details,
    compute_report,
)
from minspan.metrics import _merge_spans, reconstruct_clean_text


def bridge_gaps(
    spans: Sequence[dict[str, int]],
    max_gap: int,
) -> list[dict[str, int]]:
    """Merge spans whose gap to the previous span is at most max_gap chars."""
    bridged: list[dict[str, int]] = []
    for span in _merge_spans(spans):
        if bridged and span["start"] - bridged[-1]["end"] <= max_gap:
            bridged[-1]["end"] = span["end"]
        else:
            bridged.append(dict(span))
    return bridged


def token_drop_probabilities(
    probabilities: Sequence[float],
    offset_mapping: Sequence[tuple[int, int]],
    data_start: int,
    data_end: int,
) -> list[list[float]]:
    """Map per-token P(drop) onto data-relative char ranges: [start, end, p].

    Special tokens (empty offsets) and tokens outside the data range are
    dropped; tokens straddling the boundary are clipped to it.
    """
    entries: list[list[float]] = []
    for probability, (token_start, token_end) in zip(probabilities, offset_mapping):
        if token_start == token_end:
            continue
        overlap_start = max(token_start, data_start)
        overlap_end = min(token_end, data_end)
        if overlap_start < overlap_end:
            entries.append(
                [
                    overlap_start - data_start,
                    overlap_end - data_start,
                    round(float(probability), 4),
                ]
            )
    return entries


def spans_from_probabilities(
    token_probabilities: Sequence[Sequence[float]],
    threshold: float,
) -> list[dict[str, int]]:
    """Build drop spans from [start, end, p] entries with p >= threshold."""
    return _merge_spans(
        [
            {"start": int(start), "end": int(end)}
            for start, end, probability in token_probabilities
            if probability >= threshold
        ]
    )


def _record_from_detail(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": detail["id"],
        "attack_type": detail["attack_type"],
        "position": detail["position"],
        "cut_type": detail["cut_type"],
        "instruction": detail["instruction"],
        "attacked_data": detail["attacked_data"],
        "injection": detail["injection"],
        "drop_spans": detail["true_drop_spans"],
        "clean_data": detail["clean_data"],
    }


def rescore_payload(
    payload: dict[str, Any],
    max_gap: int,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Re-evaluate the tagger predictions of a saved results JSON after
    applying threshold/bridging post-processing. Returns a new payload."""
    details = payload["predictions"]["tagger"]
    records = [_record_from_detail(detail) for detail in details]

    source_suffix = ""
    if threshold is not None:
        source_suffix += f"+thr{threshold:g}"
    source_suffix += f"+bridge{max_gap}"

    predictions: list[str] = []
    predicted_drop_spans: list[list[dict[str, int]]] = []
    probabilities: list[Any] = []
    for detail in details:
        spans = detail["predicted_drop_spans"]
        if threshold is not None:
            if "token_drop_probabilities" not in detail:
                raise ValueError(
                    f"record {detail['id']!r} has no token_drop_probabilities; "
                    "re-run evaluate.py with --dump-drop-probabilities to use "
                    "--threshold"
                )
            spans = spans_from_probabilities(
                detail["token_drop_probabilities"], threshold
            )
        spans = bridge_gaps(spans, max_gap)
        predicted_drop_spans.append(spans)
        predictions.append(reconstruct_clean_text(detail["attacked_data"], spans))
        probabilities.append(detail.get("token_drop_probabilities"))

    new_details = _build_prediction_details(
        predictions,
        records,
        predicted_drop_spans=predicted_drop_spans,
        span_sources=[
            detail["predicted_drop_spans_source"] + source_suffix
            for detail in details
        ],
        input_tokens=[detail["input_tokens"] for detail in details],
        latency_seconds=[detail["latency_seconds"] for detail in details],
        token_drop_probabilities=(
            probabilities if all(p is not None for p in probabilities) else None
        ),
    )

    rescored = copy.deepcopy(payload)
    rescored["predictions"]["tagger"] = new_details
    rescored["tagger"] = compute_report(predictions, records)
    rescored["run_config"] = {
        **payload["run_config"],
        "postprocess": {"max_gap": max_gap, "threshold": threshold},
    }
    return rescored


def _format_overall(report: dict[str, Any]) -> str:
    overall = report["overall"]
    injection_recall = overall["injection_recall"]
    injection_text = f"{injection_recall:.4f}" if injection_recall is not None else "  n/a "
    return (
        f"exact={overall['exact_match']:.4f}  "
        f"inj_recall={injection_text}  "
        f"clean_recall={overall['clean_recall']:.4f}  "
        f"ned={overall['normalized_edit_distance']:.4f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="evaluate.py output JSON")
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="bridge gaps of at most this many chars between predicted spans",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="rebuild spans from dumped P(drop) at this threshold first",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="save rescored JSON here"
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rescored = rescore_payload(payload, max_gap=args.max_gap, threshold=args.threshold)

    print(f"{args.results.name}  (n={rescored['records_evaluated']})")
    print(f"  before  {_format_overall(payload['tagger'])}")
    print(f"  after   {_format_overall(rescored['tagger'])}")
    for attack_type, report in rescored["tagger"]["by_attack_type"].items():
        before = payload["tagger"]["by_attack_type"][attack_type]
        print(f"  [{attack_type}]")
        print(f"    before  {_format_overall({'overall': before})}")
        print(f"    after   {_format_overall({'overall': report})}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(rescored, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"rescored results saved to {args.output}")


if __name__ == "__main__":
    main()
