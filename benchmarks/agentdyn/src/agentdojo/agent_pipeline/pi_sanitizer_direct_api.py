from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from openai import OpenAI

from agentdojo.agent_pipeline.pi_sanitizer_comparison import build_comparison_row
from agentdojo.agent_pipeline.pi_sanitizer_optimization import (
    SanitizerExample,
    SanitizerPrediction,
    format_sanitizer_input,
)


@dataclass(frozen=True)
class DirectSanitizerCall:
    prediction: SanitizerPrediction
    raw_response: str
    finish_reason: str | None
    usage: dict[str, Any] | None


def load_optimized_sanitizer_prompt(path: Path) -> str:
    data = json.loads(path.read_text())
    prompt = _get_nested(data, ("sanitize", "signature", "instructions"))
    if prompt is None:
        prompt = _get_nested(data, ("predictor_instructions", "sanitize"))
    if prompt is None:
        prompt = data.get("prompt") or data.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Could not find an optimized sanitizer prompt in {path}")
    return prompt


def make_openai_compatible_client(*, api_base: str | None, api_key_env: str | None) -> OpenAI:
    api_key = os.getenv(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} must be set to call the sanitizer model.")
    return OpenAI(api_key=api_key, base_url=api_base)


def build_sanitizer_messages(system_prompt: str, example: SanitizerExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_sanitizer_input(example)},
    ]


def run_openai_compatible_sanitizer(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    example: SanitizerExample,
    extra_body: dict[str, Any] | None = None,
    temperature: float | None = 0.0,
) -> SanitizerPrediction:
    return run_openai_compatible_sanitizer_call(
        client=client,
        model=model,
        system_prompt=system_prompt,
        example=example,
        extra_body=extra_body,
        temperature=temperature,
    ).prediction


def run_openai_compatible_sanitizer_call(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    example: SanitizerExample,
    extra_body: dict[str, Any] | None = None,
    temperature: float | None = 0.0,
) -> DirectSanitizerCall:
    request: dict[str, Any] = {
        "model": normalize_openai_compatible_model(model),
        "messages": build_sanitizer_messages(system_prompt, example),
    }
    if extra_body is not None:
        request["extra_body"] = extra_body
    if temperature is not None:
        request["temperature"] = temperature

    response = client.chat.completions.create(**request)
    raw_response = _completion_content(response)
    return DirectSanitizerCall(
        prediction=parse_sanitizer_response_text(raw_response),
        raw_response=raw_response,
        finish_reason=_completion_finish_reason(response),
        usage=_completion_usage(response),
    )


def run_direct_comparison_pair(
    *,
    example: SanitizerExample,
    client: Any,
    model: str,
    seed_prompt: str,
    optimized_prompt: str,
    extra_body: dict[str, Any] | None = None,
    temperature: float | None = 0.0,
) -> dict[str, Any]:
    seed_call = run_openai_compatible_sanitizer_call(
        client=client,
        model=model,
        system_prompt=seed_prompt,
        example=example,
        extra_body=extra_body,
        temperature=temperature,
    )
    optimized_call = run_openai_compatible_sanitizer_call(
        client=client,
        model=model,
        system_prompt=optimized_prompt,
        example=example,
        extra_body=extra_body,
        temperature=temperature,
    )
    row = build_comparison_row(example, seed_call.prediction, optimized_call.prediction)
    row["seed_raw_response"] = seed_call.raw_response
    row["optimized_raw_response"] = optimized_call.raw_response
    row["seed_finish_reason"] = seed_call.finish_reason
    row["optimized_finish_reason"] = optimized_call.finish_reason
    row["seed_usage"] = seed_call.usage
    row["optimized_usage"] = optimized_call.usage
    return row


def build_direct_comparison_rows(
    *,
    examples: Sequence[SanitizerExample],
    client: Any,
    model: str,
    seed_prompt: str,
    optimized_prompt: str,
    extra_body: dict[str, Any] | None = None,
    temperature: float | None = 0.0,
) -> list[dict[str, Any]]:
    return [
        run_direct_comparison_pair(
            example=example,
            client=client,
            model=model,
            seed_prompt=seed_prompt,
            optimized_prompt=optimized_prompt,
            extra_body=extra_body,
            temperature=temperature,
        )
        for example in examples
    ]


def parse_sanitizer_response_text(raw_text: str) -> SanitizerPrediction:
    try:
        data = json.loads(_strip_json_fence(raw_text))
    except json.JSONDecodeError:
        return SanitizerPrediction(filtered_tool_output="")
    if not isinstance(data, dict):
        return SanitizerPrediction(filtered_tool_output="")
    return SanitizerPrediction(filtered_tool_output=str(data.get("filtered_tool_output", "")))


def normalize_openai_compatible_model(model: str) -> str:
    if model.startswith("openai/"):
        return model.removeprefix("openai/")
    return model


def build_run_config(
    args: Any,
    *,
    num_examples: int,
    extra_body: dict[str, Any] | None,
    seed_prompt: str,
    optimized_prompt: str,
) -> dict[str, Any]:
    return {
        "runner": args.runner,
        "data": str(args.data),
        "optimized_program": str(args.optimized_program),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "direct_api_model": normalize_openai_compatible_model(args.model),
        "api_base": args.api_base,
        "api_key_env": args.api_key_env,
        "thinking": args.thinking,
        "extra_body": extra_body,
        "temperature": args.temperature,
        "limit": args.limit,
        "num_examples": num_examples,
        "seed_prompt_sha256": _hash_text(seed_prompt),
        "optimized_prompt_sha256": _hash_text(optimized_prompt),
        "seed_prompt": seed_prompt,
        "optimized_prompt": optimized_prompt,
    }


def write_run_config(output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))


def _completion_content(response: Any) -> str:
    choices = _get_field(response, "choices", [])
    if not choices:
        return ""
    message = _get_field(choices[0], "message", {})
    content = _get_field(message, "content", "")
    if content is None:
        return ""
    return str(content)


def _completion_finish_reason(response: Any) -> str | None:
    choices = _get_field(response, "choices", [])
    if not choices:
        return None
    finish_reason = _get_field(choices[0], "finish_reason", None)
    if finish_reason is None:
        return None
    return str(finish_reason)


def _completion_usage(response: Any) -> dict[str, Any] | None:
    usage = _get_field(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else None
    if hasattr(usage, "__dict__"):
        return dict(usage.__dict__)
    return None


def _get_nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


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


def _hash_text(value: str) -> str:
    return sha256(value.encode()).hexdigest()
