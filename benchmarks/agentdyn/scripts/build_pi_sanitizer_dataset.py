"""Build paired prompt-injection sanitizer examples from data/pi_detector.

The output adds `gold_filtered_tool_output` to each row. For injected rows, the
gold output is the paired benign tool output with the same tool and user
instruction. For benign rows, the gold output is the original tool output.
"""

from __future__ import annotations

from pathlib import Path

from agentdojo.agent_pipeline.pi_sanitizer_optimization import (
    dump_sanitizer_examples_jsonl,
    load_paired_sanitizer_examples,
)

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "pi_detector"
OUTPUT_DIR = ROOT / "data" / "pi_sanitizer"


def main() -> None:
    for split in ("train", "val", "test"):
        examples = load_paired_sanitizer_examples(INPUT_DIR / f"{split}.jsonl")
        dump_sanitizer_examples_jsonl(examples, OUTPUT_DIR / f"{split}.jsonl")
    print(f"Wrote sanitizer examples to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
