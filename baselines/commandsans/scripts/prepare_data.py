#!/usr/bin/env python3
"""Sample raw annotation inputs from BFCL v3 and OpenOrca.

Follows the CommandSans data recipe: 2,000 samples per source, seed 42,
records written as {"id", "complete_text"} JSONL ready for commandsans.annotate.

    python scripts/prepare_data.py --output-dir data/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def bfcl_texts(limit: int, seed: int) -> list[str]:
    """BFCL v3 multi-turn: concatenate the textual content of each entry's turns."""
    dataset = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train")
    texts = []
    for row in dataset:
        question = row.get("question")
        parts: list[str] = []
        if isinstance(question, list):
            for turn in question:
                if isinstance(turn, list):
                    parts.extend(str(m.get("content", "")) for m in turn if isinstance(m, dict))
                elif isinstance(turn, dict):
                    parts.append(str(turn.get("content", "")))
        elif question:
            parts.append(str(question))
        text = "\n".join(p for p in parts if p.strip())
        if text.strip():
            texts.append(text)
    random.Random(seed).shuffle(texts)
    return texts[:limit]


def orca_texts(limit: int, seed: int) -> list[str]:
    """OpenOrca: system prompt + question concatenated, sampled via streaming."""
    dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    pool: list[str] = []
    # reservoir-free deterministic take: read a fixed prefix, then sample
    prefix_size = limit * 20
    for i, row in enumerate(dataset):
        if i >= prefix_size:
            break
        text = "\n".join(s for s in (row.get("system_prompt"), row.get("question")) if s and s.strip())
        if text.strip():
            pool.append(text)
    random.Random(seed).shuffle(pool)
    return pool[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--per-source", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "raw.jsonl"
    with open(out, "w") as fh:
        for source, texts in (
            ("bfcl", bfcl_texts(args.per_source, args.seed)),
            ("orca", orca_texts(args.per_source, args.seed)),
        ):
            for index, text in enumerate(texts):
                fh.write(json.dumps({"id": f"{source}-{index:06d}", "complete_text": text}) + "\n")
            print(f"{source}: {len(texts)} records")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
