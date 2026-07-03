"""Run the evaluation metric matrix across tagger checkpoints and data splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class EvaluationJob:
    model: str
    split: str
    mode: str
    data_path: str
    checkpoint: str | None
    output_path: Path


def build_jobs(
    *,
    train_data: str,
    validation_data: str,
    test_data: str,
    linear_checkpoint: str,
    bidir_checkpoint: str,
    output_dir: Path,
) -> list[EvaluationJob]:
    splits = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
    }
    jobs = []
    for model, checkpoint in (
        ("bidirectional", bidir_checkpoint),
        ("linear", linear_checkpoint),
    ):
        for split in ("test", "validation", "train"):
            data_path = splits[split]
            jobs.append(
                EvaluationJob(
                    model=model,
                    split=split,
                    mode="tagger",
                    data_path=data_path,
                    checkpoint=checkpoint,
                    output_path=output_dir / f"{model}-{split}.json",
                )
            )

    for split in ("test", "validation"):
        jobs.append(
            EvaluationJob(
                model="datafilter",
                split=split,
                mode="generative",
                data_path=splits[split],
                checkpoint=None,
                output_path=output_dir / f"datafilter-{split}.json",
            )
        )
    return jobs


def command_for_job(
    job: EvaluationJob,
    *,
    python_executable: str,
    model_name: str,
    batch_size: int,
    device: str,
    max_model_len: int,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "minspan.evaluate",
        "--test-data",
        job.data_path,
        "--mode",
        job.mode,
        "--model-name",
        model_name,
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--max-model-len",
        str(max_model_len),
        "--output",
        str(job.output_path),
    ]
    if job.split == "train":
        command.append("--summary-only")
    if job.mode == "tagger":
        if job.checkpoint is None:
            raise ValueError(f"tagger job {job.model}/{job.split} has no checkpoint")
        command.extend(["--tagger-checkpoint", job.checkpoint])
    return command


def _load_completed_result(job: EvaluationJob) -> dict[str, Any] | None:
    if not job.output_path.is_file():
        return None
    try:
        result = json.loads(job.output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    result_key = "tagger" if job.mode == "tagger" else "generative"
    if not isinstance(result.get(result_key, {}).get("overall"), dict):
        return None
    return result


def build_summary(jobs: list[EvaluationJob]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {
        "linear": {},
        "bidirectional": {},
        "datafilter": {
            "train": {
                "status": "skipped",
                "reason": "generative DataFilter training-set evaluation disabled",
            }
        },
    }
    for job in jobs:
        result = _load_completed_result(job)
        if result is None:
            models[job.model][job.split] = {
                "status": "pending",
                "output": str(job.output_path),
            }
            continue

        result_key = "tagger" if job.mode == "tagger" else "generative"
        overall = result[result_key]["overall"]
        timing = result.get("timing", {}).get(result_key, {})
        models[job.model][job.split] = {
            "status": "complete",
            "records": overall.get("n"),
            "exact_match": overall.get("exact_match"),
            "injection_recall": overall.get("injection_recall"),
            "clean_recall": overall.get("clean_recall"),
            "inference_seconds": timing.get("inference_seconds"),
            "records_per_second": timing.get("records_per_second"),
            "average_latency_seconds": timing.get("average_latency_seconds"),
            "output": str(job.output_path),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_names": {
            "exact_match": "Exact Match",
            "injection_recall": "注入删除率",
            "clean_recall": "正常内容保留率",
        },
        "models": models,
    }


def _write_summary(jobs: list[EvaluationJob], output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(build_summary(jobs), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate linear tagger and bidirectional tagger on train/validation/test, "
            "and generative DataFilter on validation/test"
        )
    )
    parser.add_argument("--train-data", default="data/train.jsonl")
    parser.add_argument("--validation-data", default="data/validation.jsonl")
    parser.add_argument("--test-data", default="data/sep_test.jsonl")
    parser.add_argument(
        "--linear-checkpoint",
        default="outputs/tagger-linear-full/best",
    )
    parser.add_argument(
        "--bidir-checkpoint",
        default="outputs/tagger-bidir-1l-512/best",
    )
    parser.add_argument("--model-name", default="JoyYizhu/DataFilter")
    parser.add_argument("--output-dir", default="outputs/evaluation-matrix")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun jobs even when a valid result JSON already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation commands without running them",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(
        train_data=args.train_data,
        validation_data=args.validation_data,
        test_data=args.test_data,
        linear_checkpoint=args.linear_checkpoint,
        bidir_checkpoint=args.bidir_checkpoint,
        output_dir=output_dir,
    )

    print("Evaluation matrix: 8 jobs")
    print("  bidirectional: test, validation, train")
    print("  linear:        test, validation, train")
    print("  DataFilter:    test, validation (train skipped)")

    for index, job in enumerate(jobs, start=1):
        command = command_for_job(
            job,
            python_executable=sys.executable,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            max_model_len=args.max_model_len,
        )
        label = f"[{index}/{len(jobs)}] {job.model}/{job.split}"
        if not args.overwrite and _load_completed_result(job) is not None:
            print(f"{label}: skip (completed)")
            continue
        print(f"{label}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        try:
            subprocess.run(command, check=True)
        finally:
            _write_summary(jobs, output_dir)

    summary_path = _write_summary(jobs, output_dir)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
