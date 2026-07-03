#!/usr/bin/env python3
"""Build ablation training sets.

Currently one variant: data/ablations/train_no_hard_negatives.jsonl —
the paper training set minus the 15,000 clean-hard-negative records
(base_id prefix `p3-clean-hard-negative`). Validation stays unchanged so
model selection is comparable across runs.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "train.jsonl"
OUT_DIR = ROOT / "data" / "ablations"
PREFIX = "p3-clean-hard-negative"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "train_no_hard_negatives.jsonl"
    kept = dropped = 0
    with open(SRC) as src, open(out, "w") as dst:
        for line in src:
            if json.loads(line)["base_id"].startswith(PREFIX):
                dropped += 1
                continue
            dst.write(line)
            kept += 1
    print(f"{out}: kept {kept}, dropped {dropped}")
    assert dropped == 15000, f"expected 15000 clean-hard-negative records, got {dropped}"


if __name__ == "__main__":
    main()
