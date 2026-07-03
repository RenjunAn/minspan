#!/usr/bin/env python3
"""Evaluate the ablation checkpoints on the local held-out splits.

Variants (see scripts/run_ablations.sh):
  control            checkpoints/ablation-control/best
  no_task_cond       checkpoints/ablation-no-task-cond/best — evaluated with
                     the instruction blanked, its operating condition
  no_hard_negatives  checkpoints/ablation-no-hard-negatives/best

Each variant runs minspan.evaluate on the three P3 test splits; the summary
lands in results/ablations.csv.

    python scripts/eval_ablations.py [--device cuda]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("p3_direct_test", "p3_strategy_test", "p3_clean_hard_negative_test")
VARIANTS = {
    "control": {"checkpoint": "checkpoints/ablation-control/best", "blank_instruction": False},
    "no_task_cond": {"checkpoint": "checkpoints/ablation-no-task-cond/best", "blank_instruction": True},
    "no_hard_negatives": {"checkpoint": "checkpoints/ablation-no-hard-negatives/best", "blank_instruction": False},
}


def blank_instructions(src: Path, dst: Path) -> None:
    with open(src) as fh_in, open(dst, "w") as fh_out:
        for line in fh_in:
            record = json.loads(line)
            record["instruction"] = ""
            fh_out.write(json.dumps(record) + "\n")


def run_eval(checkpoint: Path, test_data: Path, output: Path, device: str) -> dict:
    subprocess.run(
        [
            sys.executable, "-m", "minspan.evaluate",
            "--test-data", str(test_data),
            "--mode", "tagger",
            "--tagger-checkpoint", str(checkpoint),
            "--device", device,
            "--output", str(output),
            "--summary-only",
        ],
        cwd=ROOT,
        check=True,
        env={"TORCHDYNAMO_DISABLE": "1", **os.environ},
    )
    return json.loads(output.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ablations.csv")
    args = parser.parse_args()

    rows = []
    out_dir = ROOT / "results" / "local-eval" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for variant, spec in VARIANTS.items():
            checkpoint = ROOT / spec["checkpoint"]
            if not checkpoint.is_dir():
                print(f"skip {variant}: checkpoint missing ({checkpoint})")
                continue
            for split in SPLITS:
                test_data = ROOT / "data" / f"{split}.jsonl"
                if spec["blank_instruction"]:
                    blanked = Path(tmp) / f"{split}_blank.jsonl"
                    if not blanked.exists():
                        blank_instructions(test_data, blanked)
                    test_data = blanked
                result = run_eval(checkpoint, test_data, out_dir / f"{variant}-{split}.json", args.device)
                overall = result["tagger"]["overall"]
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "records": overall.get("n"),
                        "exact_match": overall.get("exact_match"),
                        "injection_recall": overall.get("injection_recall"),
                        "clean_recall": overall.get("clean_recall"),
                        "normalized_edit_distance": overall.get("normalized_edit_distance"),
                    }
                )
                print(f"{variant} {split}: exact={overall.get('exact_match')} "
                      f"inj={overall.get('injection_recall')} clean={overall.get('clean_recall')}")

    if not rows:
        raise SystemExit("no ablation checkpoints found — run scripts/run_ablations.sh first")
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
