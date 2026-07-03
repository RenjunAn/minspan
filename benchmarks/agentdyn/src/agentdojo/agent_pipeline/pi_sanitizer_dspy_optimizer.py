from __future__ import annotations

import json
import os
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline.pi_sanitizer_optimization import (
    SANITIZER_SYSTEM_PROMPT_V0,
    SanitizerExample,
    SanitizerPrediction,
    load_paired_sanitizer_examples,
    score_sanitizer_prediction,
)

try:
    import dspy
except ImportError:
    dspy = None


DSPY_INPUT_FIELDS = ("user_instruction", "tool_name", "tool_output")
THINKING_MODES = ("default", "disabled", "enabled")


def require_dspy(dspy_module: Any | None = dspy) -> Any:
    if dspy_module is None:
        raise RuntimeError(
            "DSPy is required to run GEPA optimization. Install DSPy in this environment before running "
            "`scripts/optimize_pi_sanitizer_prompt.py`."
        )
    return dspy_module


def build_thinking_extra_body(mode: str) -> dict[str, dict[str, str]] | None:
    if mode == "default":
        return None
    if mode in {"disabled", "enabled"}:
        return {"thinking": {"type": mode}}
    raise ValueError(f"Unsupported thinking mode {mode!r}; expected one of {THINKING_MODES}.")


def convert_examples_for_dspy(
    examples: Iterable[SanitizerExample],
    dspy_module: Any | None = dspy,
) -> list[Any]:
    dspy_mod = require_dspy(dspy_module)
    return [
        dspy_mod.Example(
            sample_id=example.sample_id,
            tool_name=example.tool_name,
            user_instruction=example.user_instruction,
            tool_output=example.tool_output,
            is_injection=example.is_injection,
            attack_type=example.attack_type,
            injection_text=example.injection_text,
            decoy=example.decoy,
            gold_filtered_tool_output=example.gold_filtered_tool_output,
        ).with_inputs(*DSPY_INPUT_FIELDS)
        for example in examples
    ]


def parse_sanitizer_prediction(prediction: Any) -> SanitizerPrediction:
    raw_json = _get_field(prediction, "sanitizer_json", "")
    try:
        data = json.loads(_strip_json_fence(str(raw_json)))
    except json.JSONDecodeError:
        return SanitizerPrediction(filtered_tool_output="")

    return SanitizerPrediction(
        filtered_tool_output=str(data.get("filtered_tool_output", "")),
    )


def sanitizer_gepa_metric(
    example: Any,
    prediction: Any,
    trace: Any | None = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None,
    dspy_module: Any | None = dspy,
) -> Any:
    dspy_mod = require_dspy(dspy_module)
    sanitizer_example = _example_from_dspy(example)
    sanitizer_prediction = parse_sanitizer_prediction(prediction)
    metric_result = score_sanitizer_prediction(sanitizer_example, sanitizer_prediction)
    return dspy_mod.Prediction(score=metric_result.score, feedback=metric_result.feedback)


def make_sanitizer_program(dspy_module: Any | None = dspy) -> Any:
    dspy_mod = require_dspy(dspy_module)

    class SanitizeToolOutput(dspy_mod.Signature):
        __doc__ = SANITIZER_SYSTEM_PROMPT_V0

        user_instruction = dspy_mod.InputField(desc="Trusted user request defining the authorized task boundary.")
        tool_name = dspy_mod.InputField(desc="Name of the tool that returned TOOL_OUTPUT; use only as weak context.")
        tool_output = dspy_mod.InputField(desc="Untrusted tool output that may contain prompt injection.")
        sanitizer_json = dspy_mod.OutputField(
            desc="A valid JSON object with exactly one key: filtered_tool_output (string)."
        )

    class SanitizerProgram(dspy_mod.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sanitize = dspy_mod.Predict(SanitizeToolOutput)

        def forward(self, user_instruction: str, tool_name: str, tool_output: str) -> Any:
            return self.sanitize(
                user_instruction=user_instruction,
                tool_name=tool_name,
                tool_output=tool_output,
            )

    return SanitizerProgram()


def make_lm(
    model: str,
    *,
    dspy_module: Any | None = dspy,
    api_base: str | None = None,
    api_key_env: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> Any:
    dspy_mod = require_dspy(dspy_module)
    kwargs: dict[str, Any] = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} must be set to use model {model}.")
        kwargs["api_key"] = api_key
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    return dspy_mod.LM(model, **kwargs)


def optimize_pi_sanitizer_prompt(
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    task_model: str,
    reflection_model: str,
    task_api_base: str | None = None,
    reflection_api_base: str | None = None,
    task_api_key_env: str | None = None,
    reflection_api_key_env: str | None = None,
    task_thinking: str = "disabled",
    reflection_thinking: str = "default",
    auto: str = "light",
    max_metric_calls: int | None = None,
    num_threads: int = 1,
    dspy_module: Any | None = dspy,
) -> Any:
    dspy_mod = require_dspy(dspy_module)
    trainset = convert_examples_for_dspy(load_paired_sanitizer_examples(train_path), dspy_mod)
    valset = convert_examples_for_dspy(load_paired_sanitizer_examples(val_path), dspy_mod)

    task_lm = make_lm(
        task_model,
        dspy_module=dspy_mod,
        api_base=task_api_base,
        api_key_env=task_api_key_env,
        extra_body=build_thinking_extra_body(task_thinking),
    )
    reflection_lm = make_lm(
        reflection_model,
        dspy_module=dspy_mod,
        api_base=reflection_api_base,
        api_key_env=reflection_api_key_env,
        extra_body=build_thinking_extra_body(reflection_thinking),
    )
    dspy_mod.configure(lm=task_lm)

    gepa_kwargs: dict[str, Any] = {
        "metric": partial(sanitizer_gepa_metric, dspy_module=dspy_mod),
        "reflection_lm": reflection_lm,
        "track_stats": True,
        "num_threads": num_threads,
    }
    if max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = max_metric_calls
    else:
        gepa_kwargs["auto"] = auto

    optimizer = dspy_mod.GEPA(**gepa_kwargs)
    optimized_program = optimizer.compile(make_sanitizer_program(dspy_mod), trainset=trainset, valset=valset)
    save_optimizer_artifacts(optimized_program, output_dir)
    return optimized_program


def save_optimizer_artifacts(optimized_program: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "system_prompt_seed": SANITIZER_SYSTEM_PROMPT_V0,
        "output_schema": {"filtered_tool_output": "string"},
        "metric": "SAS = 1 - LevenshteinDistance(filtered_tool_output, gold_filtered_tool_output) / max_length",
        "predictor_instructions": extract_predictor_instructions(optimized_program),
    }
    detailed_results = getattr(optimized_program, "detailed_results", None)
    if detailed_results is not None:
        summary["detailed_results_repr"] = repr(detailed_results)

    (output_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2))
    if hasattr(optimized_program, "save"):
        optimized_program.save(str(output_dir / "optimized_program.json"))


def extract_predictor_instructions(program: Any) -> dict[str, str]:
    if not hasattr(program, "named_predictors"):
        return {}
    instructions = {}
    for name, predictor in program.named_predictors():
        signature = getattr(predictor, "signature", None)
        instruction = getattr(signature, "instructions", None) or getattr(signature, "__doc__", None)
        if instruction:
            instructions[str(name)] = str(instruction)
    return instructions


def _example_from_dspy(example: Any) -> SanitizerExample:
    return SanitizerExample(
        sample_id=str(_get_field(example, "sample_id", "")),
        tool_name=str(_get_field(example, "tool_name", "")),
        user_instruction=str(_get_field(example, "user_instruction", "")),
        tool_output=str(_get_field(example, "tool_output", "")),
        is_injection=bool(_get_field(example, "is_injection", False)),
        attack_type=str(_get_field(example, "attack_type", "")),
        injection_text=_get_field(example, "injection_text", None),
        decoy=bool(_get_field(example, "decoy", False)),
        gold_filtered_tool_output=str(_get_field(example, "gold_filtered_tool_output", "")),
    )


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```json"):
        stripped = stripped.removeprefix("```json").strip()
    elif stripped.startswith("```"):
        stripped = stripped.removeprefix("```").strip()
    if stripped.endswith("```"):
        stripped = stripped.removesuffix("```").strip()
    return stripped
