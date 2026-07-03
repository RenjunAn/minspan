"""Compare seed and optimized PI sanitizer prompts on a fixed dataset."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline.pi_sanitizer_comparison import build_comparison_row, write_comparison_artifacts
from agentdojo.agent_pipeline.pi_sanitizer_direct_api import (
    build_run_config,
    load_optimized_sanitizer_prompt,
    make_openai_compatible_client,
    run_direct_comparison_pair,
    write_run_config,
)
from agentdojo.agent_pipeline.pi_sanitizer_dspy_optimizer import (
    THINKING_MODES,
    build_thinking_extra_body,
    make_lm,
    make_sanitizer_program,
    parse_sanitizer_prediction,
    require_dspy,
)
from agentdojo.agent_pipeline.pi_sanitizer_optimization import (
    SANITIZER_SYSTEM_PROMPT_V0,
    SanitizerExample,
    load_paired_sanitizer_examples,
)

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "pi_detector" / "val.jsonl")
    parser.add_argument(
        "--optimized-program",
        type=Path,
        default=ROOT / "runs" / "pi_sanitizer_gepa_smoke_60_sas" / "optimized_program.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "pi_sanitizer_compare_sas")
    parser.add_argument("--model", default="openai/deepseek-v4-flash")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--thinking", default="disabled", choices=THINKING_MODES)
    parser.add_argument("--runner", default="direct", choices=("direct", "dspy"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def run_program(program: Callable[..., object], example: SanitizerExample) -> object:
    return program(
        user_instruction=example.user_instruction,
        tool_name=example.tool_name,
        tool_output=example.tool_output,
    )


def main() -> None:
    args = parse_args()
    examples = load_paired_sanitizer_examples(args.data)
    if args.limit is not None:
        examples = examples[: args.limit]

    extra_body = build_thinking_extra_body(args.thinking)
    optimized_prompt = load_optimized_sanitizer_prompt(args.optimized_program)
    if args.runner == "direct":
        rows = run_direct_api_compare(args, examples, optimized_prompt, extra_body)
    else:
        rows = run_dspy_compare(args, examples, extra_body)

    write_comparison_artifacts(rows, args.output_dir)
    write_run_config(
        args.output_dir,
        build_run_config(
            args,
            num_examples=len(examples),
            extra_body=extra_body,
            seed_prompt=SANITIZER_SYSTEM_PROMPT_V0,
            optimized_prompt=optimized_prompt,
        ),
    )
    print(f"Wrote sanitizer comparison artifacts to {args.output_dir}")


def run_direct_api_compare(
    args: argparse.Namespace,
    examples: list[SanitizerExample],
    optimized_prompt: str,
    extra_body: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    client = make_openai_compatible_client(api_base=args.api_base, api_key_env=args.api_key_env)

    rows = []
    for index, example in enumerate(examples, start=1):
        row = run_direct_comparison_pair(
            example=example,
            client=client,
            model=args.model,
            seed_prompt=SANITIZER_SYSTEM_PROMPT_V0,
            optimized_prompt=optimized_prompt,
            extra_body=extra_body,
            temperature=args.temperature,
        )
        rows.append(row)
        print_progress(index, len(examples), example, row)
    return rows


def run_dspy_compare(
    args: argparse.Namespace,
    examples: list[SanitizerExample],
    extra_body: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    dspy_mod = require_dspy()
    lm = make_lm(
        args.model,
        dspy_module=dspy_mod,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        extra_body=extra_body,
    )
    seed_program = make_sanitizer_program(dspy_mod)
    optimized_program = make_sanitizer_program(dspy_mod)
    optimized_program.load(str(args.optimized_program))

    rows = []
    with dspy_mod.context(lm=lm):
        for index, example in enumerate(examples, start=1):
            seed_prediction = parse_sanitizer_prediction(run_program(seed_program, example))
            optimized_prediction = parse_sanitizer_prediction(run_program(optimized_program, example))
            row = build_comparison_row(example, seed_prediction, optimized_prediction)
            rows.append(row)
            print_progress(index, len(examples), example, row)
    return rows


def print_progress(
    index: int,
    total: int,
    example: SanitizerExample,
    row: dict[str, Any],
) -> None:
    seed_metrics = row["seed_metrics"]
    optimized_metrics = row["optimized_metrics"]
    print(
        f"[{index}/{total}] {example.sample_id}: "
        f"seed_sas={seed_metrics['sas']:.4f} "
        f"optimized_sas={optimized_metrics['sas']:.4f} "
        f"delta={row['sas_delta']:+.4f} winner={row['winner']}"
    )


if __name__ == "__main__":
    main()
