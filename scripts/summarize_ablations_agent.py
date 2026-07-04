#!/usr/bin/env python3
"""Summarize the PIArena agent-level ablation runs into results/ablations_agent.csv.

Reads the per-dataset Direct-attack evaluation JSONs written by
main.py --name ablation_{control,no_task_cond,no_hard_negatives} and reports,
for each variant, the 13-dataset macro-averaged utility and ASR — the
agent-level counterpart of the token-level results in results/ablations.csv.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "benchmarks" / "piarena" / "results" / "evaluation_results"

VARIANTS = ("ablation_control", "ablation_no_task_cond", "ablation_no_hard_negatives")
DATASETS = (
    "squad_v2", "dolly_closed_qa", "dolly_information_extraction", "dolly_summarization",
    "nq_rag", "msmarco_rag", "hotpotqa_rag", "hotpotqa_long", "qasper_long",
    "gov_report_long", "multi_news_long", "passage_retrieval_en_long", "lcc_long",
)
SUFFIX = "-Qwen-Qwen3-4B-Instruct-2507-direct-modernbert_tagger-42.json"


def dataset_metrics(path: Path) -> tuple[float, float]:
    records = json.loads(path.read_text())
    utility = statistics.fmean(float(r["utility"]) for r in records.values()) * 100
    asr = statistics.fmean(float(r["asr"]) for r in records.values()) * 100
    return utility, asr


def main() -> None:
    rows = []
    for variant in VARIANTS:
        per_dataset = []
        for dataset in DATASETS:
            path = EVAL_ROOT / variant / f"{dataset}{SUFFIX}"
            if not path.exists():
                print(f"missing: {variant}/{dataset}")
                continue
            per_dataset.append(dataset_metrics(path))
        if not per_dataset:
            continue
        utilities = [u for u, _ in per_dataset]
        asrs = [a for _, a in per_dataset]
        rows.append(
            {
                "variant": variant.replace("ablation_", ""),
                "datasets": len(per_dataset),
                "direct_utility": round(statistics.fmean(utilities), 2),
                "direct_asr": round(statistics.fmean(asrs), 2),
            }
        )
        print(f"{variant:<28} n={len(per_dataset):>2} utility={rows[-1]['direct_utility']:.2f} "
              f"asr={rows[-1]['direct_asr']:.2f}")

    if rows:
        out = ROOT / "results" / "ablations_agent.csv"
        with open(out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
