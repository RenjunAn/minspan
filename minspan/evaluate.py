"""Compare Token Tagger vs generative DataFilter on a JSONL test split."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from collections import defaultdict
from pathlib import Path
import time
from typing import Any


def _progress_bar(total: int, desc: str):
    """Return a tqdm progress bar, or None if tqdm is unavailable.

    Writes to stderr so it never corrupts JSON emitted on stdout.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, unit="rec", dynamic_ncols=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _text_metrics(predicted: str, ground_truth: str, record: dict) -> dict:
    exact = predicted == ground_truth
    attacked_data = record["attacked_data"]

    source_preserved = [False] * len(attacked_data)
    for match in SequenceMatcher(
        None,
        attacked_data,
        predicted,
        autojunk=False,
    ).get_matching_blocks():
        for index in range(match.a, match.a + match.size):
            source_preserved[index] = True

    injection_indices = [
        index
        for span in record["drop_spans"]
        for index in range(span["start"], span["end"])
    ]
    if injection_indices:
        retained_injection = sum(source_preserved[index] for index in injection_indices)
        injection_recall = 1.0 - retained_injection / len(injection_indices)
    else:
        injection_recall = None

    matching_clean_characters = sum(
        match.size
        for match in SequenceMatcher(
            None,
            ground_truth,
            predicted,
            autojunk=False,
        ).get_matching_blocks()
    )
    if ground_truth:
        clean_recall = matching_clean_characters / len(ground_truth)
    else:
        clean_recall = 1.0 if not predicted else 0.0

    # Indel edit distance (insertions + deletions) against the clean ground
    # truth, reusing the same alignment as clean_recall. Penalizes both lost
    # clean content and retained extra content; exact match iff distance == 0.
    edit_distance = len(predicted) + len(ground_truth) - 2 * matching_clean_characters
    normalized_edit_distance = edit_distance / max(len(predicted), len(ground_truth), 1)

    return {
        "exact": exact,
        "injection_recall": injection_recall,
        "clean_recall": clean_recall,
        "edit_distance": edit_distance,
        "normalized_edit_distance": normalized_edit_distance,
    }


def _aggregate(entries: list[dict]) -> dict:
    n = len(entries)
    if n == 0:
        return {
            "n": 0,
            "exact_match": 0.0,
            "injection_recall": None,
            "injection_records": 0,
            "clean_recall": 0.0,
            "normalized_edit_distance": None,
        }
    injection_recalls = [
        entry["injection_recall"]
        for entry in entries
        if entry["injection_recall"] is not None
    ]
    return {
        "n": n,
        "exact_match": sum(e["exact"] for e in entries) / n,
        "injection_recall": (
            sum(injection_recalls) / len(injection_recalls)
            if injection_recalls
            else None
        ),
        "injection_records": len(injection_recalls),
        "clean_recall": sum(e["clean_recall"] for e in entries) / n,
        "normalized_edit_distance": (
            sum(e["normalized_edit_distance"] for e in entries) / n
        ),
    }


def compute_report(predictions: list[str], records: list[dict]) -> dict:
    if len(predictions) != len(records):
        raise ValueError(
            f"prediction count {len(predictions)} != record count {len(records)}"
        )
    all_entries: list[dict] = []
    by_attack_type: dict[str, list] = defaultdict(list)
    by_position: dict[str, list] = defaultdict(list)
    by_cut_type: dict[str, list] = defaultdict(list)

    for pred, record in zip(predictions, records):
        entry = _text_metrics(pred, record["clean_data"], record)
        all_entries.append(entry)
        by_attack_type[record["attack_type"]].append(entry)
        by_position[record["position"]].append(entry)
        by_cut_type[record["cut_type"]].append(entry)

    return {
        "overall": _aggregate(all_entries),
        "by_attack_type": {k: _aggregate(v) for k, v in sorted(by_attack_type.items())},
        "by_position": {k: _aggregate(v) for k, v in sorted(by_position.items())},
        "by_cut_type": {k: _aggregate(v) for k, v in sorted(by_cut_type.items())},
    }


def _mask_to_spans(mask: list[bool]) -> list[dict[str, int]]:
    spans = []
    start = None
    for index, selected in enumerate(mask + [False]):
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            spans.append({"start": start, "end": index})
            start = None
    return spans


def _removed_source_spans(source: str, predicted: str) -> list[dict[str, int]]:
    """Estimate removed source ranges by aligning generated output to its input."""
    preserved = [False] * len(source)
    for match in SequenceMatcher(None, source, predicted, autojunk=False).get_matching_blocks():
        for index in range(match.a, match.a + match.size):
            preserved[index] = True
    return _mask_to_spans([not value for value in preserved])


def _merge_spans(spans: list[dict[str, int]]) -> list[dict[str, int]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (span["start"], span["end"]))
    merged = [dict(ordered[0])]
    for span in ordered[1:]:
        previous = merged[-1]
        if span["start"] <= previous["end"]:
            previous["end"] = max(previous["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged


def _subtract_spans(
    spans: list[dict[str, int]],
    subtract: list[dict[str, int]],
) -> list[dict[str, int]]:
    result = []
    blockers = _merge_spans(subtract)
    for span in _merge_spans(spans):
        cursor = span["start"]
        for blocker in blockers:
            if blocker["end"] <= cursor:
                continue
            if blocker["start"] >= span["end"]:
                break
            if blocker["start"] > cursor:
                result.append(
                    {"start": cursor, "end": min(blocker["start"], span["end"])}
                )
            cursor = max(cursor, blocker["end"])
            if cursor >= span["end"]:
                break
        if cursor < span["end"]:
            result.append({"start": cursor, "end": span["end"]})
    return result


def _build_prediction_details(
    predictions: list[str],
    records: list[dict],
    *,
    predicted_drop_spans: list[list[dict[str, int]]] | None = None,
    span_sources: list[str] | None = None,
    input_tokens: list[int] | None = None,
    latency_seconds: list[float] | None = None,
    token_drop_probabilities: list[Any] | None = None,
) -> list[dict[str, Any]]:
    if len(predictions) != len(records):
        raise ValueError(
            f"prediction count {len(predictions)} != record count {len(records)}"
        )
    count = len(records)
    if predicted_drop_spans is None:
        predicted_drop_spans = [
            _removed_source_spans(record["attacked_data"], predicted)
            for predicted, record in zip(predictions, records)
        ]
    if span_sources is None:
        span_sources = ["sequence_alignment_estimate"] * count
    if input_tokens is None:
        input_tokens = [0] * count
    if latency_seconds is None:
        latency_seconds = [0.0] * count
    for name, values in (
        ("predicted_drop_spans", predicted_drop_spans),
        ("span_sources", span_sources),
        ("input_tokens", input_tokens),
        ("latency_seconds", latency_seconds),
    ):
        if len(values) != count:
            raise ValueError(f"{name} count {len(values)} != record count {count}")
    if token_drop_probabilities is not None and len(token_drop_probabilities) != count:
        raise ValueError(
            f"token_drop_probabilities count {len(token_drop_probabilities)} "
            f"!= record count {count}"
        )

    details = []
    for index, (predicted, record, predicted_spans, span_source, token_count, latency) in enumerate(
        zip(
            predictions,
            records,
            predicted_drop_spans,
            span_sources,
            input_tokens,
            latency_seconds,
        )
    ):
        metrics = _text_metrics(predicted, record["clean_data"], record)
        true_spans = record["drop_spans"]
        false_positive_spans = _subtract_spans(predicted_spans, true_spans)
        missed_spans = _subtract_spans(true_spans, predicted_spans)
        details.append(
            {
                "id": record["id"],
                "attack_type": record["attack_type"],
                "position": record["position"],
                "cut_type": record["cut_type"],
                "instruction": record["instruction"],
                "attacked_data": record["attacked_data"],
                "injection": record["injection"],
                "true_drop_spans": true_spans,
                "predicted_drop_spans": predicted_spans,
                "predicted_drop_spans_source": span_source,
                "false_positive_drop_spans": false_positive_spans,
                "missed_drop_spans": missed_spans,
                "predicted_clean_data": predicted,
                "clean_data": record["clean_data"],
                "input_tokens": token_count,
                "latency_seconds": latency,
                **metrics,
            }
        )
        if token_drop_probabilities is not None:
            details[-1]["token_drop_probabilities"] = token_drop_probabilities[index]
    return details


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _build_timing(
    *,
    load_seconds: float,
    inference_seconds: float,
    records: int,
    input_characters: int,
    input_tokens: int,
    per_record_seconds: list[float],
    batch_size: int,
) -> dict[str, Any]:
    average_latency = (
        sum(per_record_seconds) / len(per_record_seconds)
        if per_record_seconds
        else 0.0
    )
    return {
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": load_seconds + inference_seconds,
        "records": records,
        "input_characters": input_characters,
        "records_per_second": records / inference_seconds if inference_seconds else 0.0,
        "input_characters_per_second": (
            input_characters / inference_seconds if inference_seconds else 0.0
        ),
        "input_tokens": input_tokens,
        "input_tokens_per_second": (
            input_tokens / inference_seconds if inference_seconds else 0.0
        ),
        "average_latency_seconds": average_latency,
        "p50_latency_seconds": _percentile(per_record_seconds, 0.50),
        "p95_latency_seconds": _percentile(per_record_seconds, 0.95),
        "batch_size": batch_size,
        "latency_measurement": (
            "wall_clock_per_record"
            if batch_size == 1
            else "batch_wall_clock_divided_by_batch_size"
        ),
    }


def _filter_records(
    records: list[dict],
    *,
    attack_type: str | None,
    position: str | None,
    limit: int | None,
) -> list[dict]:
    selected = [
        record
        for record in records
        if (attack_type is None or record["attack_type"] == attack_type)
        and (position is None or record["position"] == position)
    ]
    return selected[:limit] if limit is not None else selected


def _build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format_version": 2,
        "test_data": args.test_data,
        "mode": args.mode,
        "tagger_checkpoint": args.tagger_checkpoint,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "device": args.device,
        "max_model_len": args.max_model_len,
        "attack_type": args.attack_type,
        "position": args.position,
        "limit": args.limit,
        "summary_only": getattr(args, "summary_only", False),
        "dump_drop_probabilities": getattr(args, "dump_drop_probabilities", False),
    }


def _prepare_output_results(
    results: dict[str, Any],
    prediction_details: dict[str, list[dict[str, Any]]],
    *,
    summary_only: bool,
) -> dict[str, Any]:
    output = dict(results)
    if not summary_only:
        output["predictions"] = prediction_details
    return output


def _build_vllm_kwargs(model_name: str, max_model_len: int) -> dict[str, Any]:
    return {
        "model": model_name,
        "max_model_len": max_model_len,
    }


def _build_speed_comparison(
    tagger_timing: dict[str, Any],
    generative_timing: dict[str, Any],
) -> dict[str, float]:
    tagger_throughput = tagger_timing["records_per_second"]
    generative_throughput = generative_timing["records_per_second"]
    tagger_latency = tagger_timing["average_latency_seconds"]
    generative_latency = generative_timing["average_latency_seconds"]
    return {
        "tagger_speedup_factor": (
            tagger_throughput / generative_throughput
            if generative_throughput
            else 0.0
        ),
        "tagger_to_generative_latency_ratio": (
            tagger_latency / generative_latency if generative_latency else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Token Tagger inference
# ---------------------------------------------------------------------------

def _checkpoint_head_type(checkpoint_dir: str | Path) -> str:
    config_path = Path(checkpoint_dir) / "tagger_config.json"
    if not config_path.is_file():
        raise ValueError(f"checkpoint is missing tagger_config.json: {checkpoint_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("head_type", "linear")


def run_tagger(
    records: list[dict],
    checkpoint_dir: str,
    model_name: str,
    batch_size: int,
    device: str,
    dump_probabilities: bool = False,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from minspan.modeling import load_encoder_tagger, load_tagger_head
    from minspan.prompting import build_encoder_prompt, build_tagger_prompt
    from minspan.training_data import TaggerCollator
    from minspan.metrics import DROP_LABEL, prediction_to_data_spans, reconstruct_clean_text
    from minspan.postprocess import token_drop_probabilities

    head_type = _checkpoint_head_type(checkpoint_dir)
    load_started = time.perf_counter()
    backbone = None
    if head_type == "encoder":
        print(f"[tagger] loading encoder checkpoint {checkpoint_dir} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        model = load_encoder_tagger(checkpoint_dir)
        prompt_builder = build_encoder_prompt
    else:
        print(f"[tagger] loading backbone {model_name} ...", flush=True)
        backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        model = load_tagger_head(backbone, checkpoint_dir)
        prompt_builder = build_tagger_prompt
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval().to(device)
    model_config = model.architecture_config()
    print(
        f"[tagger] loaded head_type={model_config['head_type']} "
        f"trainable_parameters={model_config['trainable_parameters']}",
        flush=True,
    )
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    collator = TaggerCollator(tokenizer, prompt_builder=prompt_builder)
    predictions: list[str] = []
    predicted_drop_spans: list[list[dict[str, int]]] = []
    drop_probabilities: list[list[list[float]]] | None = (
        [] if dump_probabilities else None
    )
    input_token_counts: list[int] = []
    latency_seconds: list[float] = []
    total = len(records)
    inference_started = time.perf_counter()
    progress = _progress_bar(total, "[tagger]")

    for start in range(0, total, batch_size):
        batch_started = time.perf_counter()
        batch_records = records[start : start + batch_size]
        batch = collator(batch_records)
        input_token_counts.extend(batch["token_lengths"])

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids, attention_mask)

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        preds = out.logits.argmax(dim=-1)
        batch_drop_probs = (
            torch.softmax(out.logits.float(), dim=-1)[..., DROP_LABEL]
            if dump_probabilities
            else None
        )

        for i, record in enumerate(batch_records):
            token_preds = preds[i]
            offsets = batch["offset_mapping"][i].tolist()
            data_start, data_end = batch["data_ranges"][i]
            spans = _merge_spans(
                prediction_to_data_spans(
                    token_preds,
                    offsets,
                    data_start,
                    data_end,
                )
            )
            predicted_drop_spans.append(spans)
            predictions.append(reconstruct_clean_text(record["attacked_data"], spans))
            if batch_drop_probs is not None:
                drop_probabilities.append(
                    token_drop_probabilities(
                        batch_drop_probs[i].tolist(),
                        offsets,
                        data_start,
                        data_end,
                    )
                )

        per_record_latency = (time.perf_counter() - batch_started) / len(batch_records)
        latency_seconds.extend([per_record_latency] * len(batch_records))
        if progress is not None:
            progress.update(len(batch_records))
        else:
            done = min(start + batch_size, total)
            print(f"\r[tagger] {done}/{total}", end="", flush=True)

    if progress is not None:
        progress.close()
    else:
        print()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    del model
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, _build_timing(
        load_seconds=load_seconds,
        inference_seconds=inference_seconds,
        records=total,
        input_characters=sum(
            len(record["instruction"]) + len(record["attacked_data"])
            for record in records
        ),
        input_tokens=sum(input_token_counts),
        per_record_seconds=latency_seconds,
        batch_size=batch_size,
    ), {
        "predicted_drop_spans": predicted_drop_spans,
        "span_sources": ["token_tagger"] * total,
        "input_tokens": input_token_counts,
        "latency_seconds": latency_seconds,
        "token_drop_probabilities": drop_probabilities,
        "model_config": model_config,
    }


# ---------------------------------------------------------------------------
# Generative DataFilter inference
# ---------------------------------------------------------------------------

def run_generative(
    records: list[dict],
    model_name: str,
    batch_size: int,
    max_model_len: int,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    from inference_utils import apply_filter_in_batch, format_prompt
    from vllm import LLM

    load_started = time.perf_counter()
    print(f"[generative] loading {model_name} via vLLM ...", flush=True)
    llm = LLM(**_build_vllm_kwargs(model_name, max_model_len))
    tokenizer = llm.get_tokenizer()
    load_seconds = time.perf_counter() - load_started

    predictions: list[str] = []
    input_token_counts = [
        len(
            tokenizer.encode(
                format_prompt(
                    f"{record['instruction']} <|end_of_instruction|> "
                    f"{record['attacked_data']}"
                ),
                add_special_tokens=False,
            )
        )
        for record in records
    ]
    latency_seconds: list[float] = []
    total = len(records)
    inference_started = time.perf_counter()
    progress = _progress_bar(total, "[generative]")
    # When our own bar is active, silence vLLM's per-call "Processed prompts"
    # bar so the overall progress line stays readable.
    inner_tqdm = progress is None

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        instructions = [r["instruction"] for r in batch]
        datas = [r["attacked_data"] for r in batch]
        batch_started = time.perf_counter()
        batch_predictions = apply_filter_in_batch(
            llm, instructions, datas, use_tqdm=inner_tqdm
        )
        batch_elapsed = time.perf_counter() - batch_started
        predictions.extend(batch_predictions)
        per_record_latency = batch_elapsed / len(batch)
        latency_seconds.extend([per_record_latency] * len(batch))

        if progress is not None:
            progress.update(len(batch))
        else:
            done = min(start + batch_size, total)
            print(f"\r[generative] {done}/{total}", end="", flush=True)

    if progress is not None:
        progress.close()
    else:
        print()
    inference_seconds = time.perf_counter() - inference_started
    return predictions, _build_timing(
        load_seconds=load_seconds,
        inference_seconds=inference_seconds,
        records=total,
        input_characters=sum(
            len(record["instruction"]) + len(record["attacked_data"])
            for record in records
        ),
        input_tokens=sum(input_token_counts),
        per_record_seconds=latency_seconds,
        batch_size=batch_size,
    ), {
        "predicted_drop_spans": [
            _removed_source_spans(record["attacked_data"], predicted)
            for predicted, record in zip(predictions, records)
        ],
        "span_sources": ["sequence_alignment_estimate"] * total,
        "input_tokens": input_token_counts,
        "latency_seconds": latency_seconds,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Token Tagger and/or generative DataFilter on a JSONL test split"
    )
    parser.add_argument("--test-data", required=True, help="Path to test JSONL file")
    parser.add_argument(
        "--mode",
        choices=["tagger", "generative", "both"],
        default="both",
        help="Which model(s) to evaluate",
    )
    parser.add_argument(
        "--tagger-checkpoint",
        default="outputs/tagger-linear-full/best",
        help="Path to tagger checkpoint directory",
    )
    parser.add_argument(
        "--model-name",
        default="JoyYizhu/DataFilter",
        help="HuggingFace model name for both backbone and generative model",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM context limit for generative DataFilter (default: 32768)",
    )
    parser.add_argument(
        "--attack-type",
        choices=["Clean", "Naive", "Ignore", "Completion"],
        default=None,
        help="Evaluate only records with this attack type",
    )
    parser.add_argument(
        "--position",
        choices=["none", "prepend", "middle", "append"],
        default=None,
        help="Evaluate only records with this injection position",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N records")
    parser.add_argument(
        "--output", default=None, help="Save full results JSON to this path"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="When saving JSON, omit per-record predictions and keep only aggregate metrics",
    )
    parser.add_argument(
        "--dump-drop-probabilities",
        action="store_true",
        help="Store per-token P(drop) as [data_start, data_end, p] entries in each "
        "tagger prediction detail, enabling offline threshold sweeps",
    )
    return parser.parse_args(argv)


def _print_report(name: str, report: dict) -> None:
    def row(label: str, stats: dict) -> None:
        injection_recall = stats["injection_recall"]
        injection_text = (
            f"{injection_recall:.3f}" if injection_recall is not None else "n/a"
        )
        print(
            f"  {label:<30}  n={stats['n']:>5}  "
            f"exact={stats['exact_match']:.3f}  "
            f"inj_recall={injection_text:>5}  "
            f"clean_recall={stats['clean_recall']:.3f}"
        )

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    row("overall", report["overall"])
    print()
    for k, v in report["by_attack_type"].items():
        row(f"  attack_type={k}", v)
    print()
    for k, v in report["by_position"].items():
        row(f"  position={k}", v)
    print()
    for k, v in report["by_cut_type"].items():
        row(f"  cut_type={k}", v)


def _print_timing(name: str, timing: dict[str, Any]) -> None:
    print(f"\n  {name} timing")
    print(f"    load_seconds={timing['load_seconds']:.3f}")
    print(f"    inference_seconds={timing['inference_seconds']:.3f}")
    print(f"    records_per_second={timing['records_per_second']:.3f}")
    print(
        "    input_characters_per_second="
        f"{timing['input_characters_per_second']:.3f}"
    )
    print(f"    input_tokens={timing['input_tokens']}")
    print(f"    input_tokens_per_second={timing['input_tokens_per_second']:.3f}")
    print(f"    average_latency_seconds={timing['average_latency_seconds']:.6f}")
    print(f"    p50_latency_seconds={timing['p50_latency_seconds']:.6f}")
    print(f"    p95_latency_seconds={timing['p95_latency_seconds']:.6f}")
    print(f"    latency_measurement={timing['latency_measurement']}")


def main() -> None:
    args = parse_args()

    records: list[dict] = []
    with open(args.test_data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = _filter_records(
        records,
        attack_type=args.attack_type,
        position=args.position,
        limit=args.limit,
    )
    print(f"Loaded {len(records)} records from {args.test_data}", flush=True)

    results: dict[str, Any] = {
        "records_evaluated": len(records),
        "run_config": _build_run_config(args),
        "timing": {},
    }
    prediction_details: dict[str, list[dict[str, Any]]] = {}

    if args.mode in ("tagger", "both"):
        tagger_preds, tagger_timing, tagger_diagnostics = run_tagger(
            records,
            args.tagger_checkpoint,
            args.model_name,
            args.batch_size,
            args.device,
            dump_probabilities=args.dump_drop_probabilities,
        )
        tagger_report = compute_report(tagger_preds, records)
        results["tagger"] = tagger_report
        results["tagger_model"] = tagger_diagnostics["model_config"]
        results["timing"]["tagger"] = tagger_timing
        prediction_details["tagger"] = _build_prediction_details(
            tagger_preds,
            records,
            predicted_drop_spans=tagger_diagnostics["predicted_drop_spans"],
            span_sources=tagger_diagnostics["span_sources"],
            input_tokens=tagger_diagnostics["input_tokens"],
            latency_seconds=tagger_diagnostics["latency_seconds"],
            token_drop_probabilities=tagger_diagnostics["token_drop_probabilities"],
        )
        _print_report("Token Tagger", tagger_report)
        _print_timing("Token Tagger", tagger_timing)

    if args.mode in ("generative", "both"):
        gen_preds, gen_timing, gen_diagnostics = run_generative(
            records,
            args.model_name,
            args.batch_size,
            args.max_model_len,
        )
        gen_report = compute_report(gen_preds, records)
        results["generative"] = gen_report
        results["timing"]["generative"] = gen_timing
        prediction_details["generative"] = _build_prediction_details(
            gen_preds,
            records,
            **gen_diagnostics,
        )
        _print_report("Generative DataFilter", gen_report)
        _print_timing("Generative DataFilter", gen_timing)

    if args.mode == "both":
        print(f"\n{'='*70}")
        print("  Delta (Tagger - Generative)")
        print(f"{'='*70}")
        for key in ("exact_match", "injection_recall", "clean_recall"):
            tagger_value = results["tagger"]["overall"][key]
            generative_value = results["generative"]["overall"][key]
            if tagger_value is None or generative_value is None:
                print(f"  {key:<30}  n/a")
                continue
            delta = tagger_value - generative_value
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<30}  {sign}{delta:.3f}")

        speed_comparison = _build_speed_comparison(
            results["timing"]["tagger"],
            results["timing"]["generative"],
        )
        results["speed_comparison"] = speed_comparison
        print()
        print(
            "  tagger_speedup_factor"
            f"{speed_comparison['tagger_speedup_factor']:>28.3f}x"
        )
        print(
            "  tagger_to_generative_latency_ratio"
            f"{speed_comparison['tagger_to_generative_latency_ratio']:>14.3f}"
        )

    if args.output:
        output_results = _prepare_output_results(
            results,
            prediction_details,
            summary_only=args.summary_only,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
