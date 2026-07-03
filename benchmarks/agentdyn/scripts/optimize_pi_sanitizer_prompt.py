"""Optimize the PI sanitizer prompt with DSPy GEPA."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentdojo.agent_pipeline.pi_sanitizer_dspy_optimizer import THINKING_MODES, optimize_pi_sanitizer_prompt

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "pi_detector" / "train.jsonl")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "pi_detector" / "val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "pi_sanitizer_gepa_sas")
    parser.add_argument("--task-model", default="openai/deepseek-v4-flash")
    parser.add_argument("--reflection-model", default="openai/deepseek-v4-pro")
    parser.add_argument("--task-api-base", default="https://api.deepseek.com")
    parser.add_argument("--reflection-api-base", default="https://api.deepseek.com")
    parser.add_argument("--task-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--reflection-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--task-thinking", default="disabled", choices=THINKING_MODES)
    parser.add_argument("--reflection-thinking", default="default", choices=THINKING_MODES)
    parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-metric-calls", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimize_pi_sanitizer_prompt(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output_dir,
        task_model=args.task_model,
        reflection_model=args.reflection_model,
        task_api_base=args.task_api_base,
        reflection_api_base=args.reflection_api_base,
        task_api_key_env=args.task_api_key_env,
        reflection_api_key_env=args.reflection_api_key_env,
        task_thinking=args.task_thinking,
        reflection_thinking=args.reflection_thinking,
        auto=args.auto,
        max_metric_calls=args.max_metric_calls,
        num_threads=args.num_threads,
    )
    print(f"Wrote optimized sanitizer artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
