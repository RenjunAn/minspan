#!/usr/bin/env python3
"""Summarize adaptive-attack robustness runs into results/adaptive_attack.csv.

Covers both threat models:
  adaptive_generic          attacker does NOT know the user task (standard IPI)
  adaptive_task_camouflage  attacker embeds the exact task verbatim (stronger,
                            outside the standard IPI threat model)

For each attack and defense variant, reports the macro-averaged utility and ASR
over the datasets evaluated.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "benchmarks" / "piarena" / "results" / "evaluation_results"
ATTACKS = ("adaptive_generic", "adaptive_task_camouflage")
VARIANTS = {"nodef": ("No Defense", "none"),
            "minspan": ("MinSpan", "modernbert_tagger"),
            "no_task_cond": ("MinSpan - task cond.", "modernbert_tagger")}


def metrics(path: Path) -> tuple[float, float]:
    recs = json.loads(path.read_text())
    return (statistics.fmean(float(r["utility"]) for r in recs.values()) * 100,
            statistics.fmean(float(r["asr"]) for r in recs.values()) * 100)


def main() -> None:
    rows = []
    for attack in ATTACKS:
        for variant, (label, defense) in VARIANTS.items():
            d = EVAL / f"adaptive_{variant}"
            paths = sorted(d.glob(f"*-{attack}-{defense}-42.json")) if d.is_dir() else []
            if not paths:
                continue
            us, as_ = zip(*(metrics(p) for p in paths))
            rows.append({
                "attack": attack,
                "variant": label,
                "datasets": len(paths),
                "utility": round(statistics.fmean(us), 2),
                "asr": round(statistics.fmean(as_), 2),
            })
            print(f"{attack:<26} {label:<22} n={len(paths)} "
                  f"utility={rows[-1]['utility']:.2f} asr={rows[-1]['asr']:.2f}")

    if rows:
        out = ROOT / "results" / "adaptive_attack.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
