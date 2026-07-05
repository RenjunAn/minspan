#!/usr/bin/env python3
"""Summarize the adaptive-attack robustness runs into results/adaptive_attack.csv.

For each defense variant, macro-average utility and ASR over the datasets
evaluated under the adaptive attack, alongside the Direct-attack baseline
(from results/piarena_main.csv) for reference.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "benchmarks" / "piarena" / "results" / "evaluation_results"
ATTACK = "adaptive_task_camouflage"
VARIANTS = {"nodef": "No Defense", "minspan": "MinSpan", "no_task_cond": "MinSpan − task cond."}
SUFFIX = f"-Qwen-Qwen3-4B-Instruct-2507-{ATTACK}-{{defense}}-42.json"
DEFENSE_KEY = {"nodef": "none", "minspan": "modernbert_tagger", "no_task_cond": "modernbert_tagger"}


def metrics(path: Path) -> tuple[float, float]:
    recs = json.loads(path.read_text())
    return (
        statistics.fmean(float(r["utility"]) for r in recs.values()) * 100,
        statistics.fmean(float(r["asr"]) for r in recs.values()) * 100,
    )


def main() -> None:
    rows = []
    for variant, label in VARIANTS.items():
        d = EVAL / f"adaptive_{variant}"
        if not d.is_dir():
            print(f"missing dir: {d}")
            continue
        us, as_ = [], []
        datasets = []
        for path in sorted(d.glob(f"*-{ATTACK}-*.json")):
            u, a = metrics(path)
            us.append(u)
            as_.append(a)
            datasets.append(path.name.split("-Qwen")[0])
        if not us:
            continue
        rows.append({
            "variant": label,
            "datasets": len(us),
            "adaptive_utility": round(statistics.fmean(us), 2),
            "adaptive_asr": round(statistics.fmean(as_), 2),
        })
        print(f"{label:<24} n={len(us)} utility={rows[-1]['adaptive_utility']:.2f} "
              f"asr={rows[-1]['adaptive_asr']:.2f}  [{', '.join(datasets)}]")

    if rows:
        out = ROOT / "results" / "adaptive_attack.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
