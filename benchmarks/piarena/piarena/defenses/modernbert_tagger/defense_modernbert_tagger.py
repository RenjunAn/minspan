from __future__ import annotations

import gc
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..base import BaseDefense, register_defense

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ImportError:
    AutoModelForTokenClassification = None
    AutoTokenizer = None


INSTRUCTION_SEPARATOR = "<|end_of_instruction|>"
DROP_LABEL = 1
ENCODER_PROMPT_FORMAT_VERSION = 2
EXPECTED_LABEL2ID = {"KEEP": 0, "DROP": 1}


@dataclass(frozen=True)
class SerializedTaggerPrompt:
    text: str
    data_start: int
    data_end: int


def build_encoder_tagger_prompt(instruction: str, tool_output: str) -> SerializedTaggerPrompt:
    prefix = f"{instruction} {INSTRUCTION_SEPARATOR} "
    return SerializedTaggerPrompt(
        text=prefix + tool_output,
        data_start=len(prefix),
        data_end=len(prefix) + len(tool_output),
    )


@dataclass(frozen=True)
class TaggerPrediction:
    filtered_tool_output: str
    predicted_drop_spans: Sequence[Any]
    input_tokens: int
    latency_ms: int = 0
    error: str | None = None


class TaggerBatchError(RuntimeError):
    def __init__(self, message: str, *, input_tokens: Sequence[int]) -> None:
        super().__init__(message)
        self.input_tokens = list(input_tokens)


class TaggerBackend(Protocol):
    backend_type: str
    architecture: dict[str, Any]
    checkpoint_fingerprint: str

    def sanitize_batch(
        self,
        instructions: Sequence[str],
        tool_outputs: Sequence[str],
    ) -> Sequence[TaggerPrediction]: ...


def merge_spans(spans: Sequence[dict[str, int]]) -> list[dict[str, int]]:
    merged: list[dict[str, int]] = []
    for span in sorted(spans, key=lambda item: (item["start"], item["end"])):
        if merged and span["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged


def reconstruct_without_spans(text: str, spans: Sequence[dict[str, int]]) -> str:
    chunks: list[str] = []
    cursor = 0
    for span in merge_spans(spans):
        chunks.append(text[cursor : span["start"]])
        cursor = span["end"]
    chunks.append(text[cursor:])
    return "".join(chunks)


def predictions_to_data_spans(
    predictions: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    data_start: int,
    data_end: int,
) -> list[dict[str, int]]:
    spans = []
    for prediction, (token_start, token_end) in zip(predictions, offsets):
        if int(prediction) != DROP_LABEL or token_start == token_end:
            continue
        overlap_start = max(int(token_start), data_start)
        overlap_end = min(int(token_end), data_end)
        if overlap_start < overlap_end:
            spans.append(
                {
                    "start": overlap_start - data_start,
                    "end": overlap_end - data_start,
                }
            )
    return merge_spans(spans)


class TokenTaggerBackend:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt_builder: Callable[[str, str], SerializedTaggerPrompt],
        device: str,
        batch_size: int,
        backend_type: str,
        architecture: dict[str, Any] | None = None,
        checkpoint_fingerprint: str = "unverified",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("token tagger requires a fast tokenizer")
        if tokenizer.pad_token_id is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise ValueError("tokenizer must define a pad or EOS token")
            tokenizer.pad_token = eos_token
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder
        self.device = device
        self.batch_size = batch_size
        self.backend_type = backend_type
        self.checkpoint_fingerprint = checkpoint_fingerprint
        self.architecture = architecture or _model_architecture(model, backend_type)

    @classmethod
    def from_encoder_checkpoint(
        cls,
        *,
        checkpoint_path: str,
        device: str,
        batch_size: int,
    ) -> "TokenTaggerBackend":
        checkpoint = _checkpoint_path(checkpoint_path)
        config = _read_tagger_config(checkpoint)
        _validate_checkpoint_contract(
            config,
            expected_head_type="encoder",
            expected_prompt_format_version=ENCODER_PROMPT_FORMAT_VERSION,
        )
        if config.get("base_model_type") != "modernbert":
            raise ValueError("ModernBERT tagger checkpoint must use base_model_type 'modernbert'")
        checkpoint_fingerprint = _checkpoint_fingerprint(checkpoint)
        _require_loader(AutoTokenizer, "transformers")
        _require_loader(AutoModelForTokenClassification, "transformers")
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(str(checkpoint))
        _validate_encoder_model_config(model, config)
        return cls(
            model=model,
            tokenizer=tokenizer,
            prompt_builder=build_encoder_tagger_prompt,
            device=device,
            batch_size=batch_size,
            backend_type="modernbert",
            checkpoint_fingerprint=checkpoint_fingerprint,
            architecture={
                "head_type": "encoder",
                "model_type": getattr(model.config, "model_type", "unknown"),
                "hidden_size": getattr(model.config, "hidden_size", None),
            },
        )

    def sanitize_batch(
        self,
        instructions: Sequence[str],
        tool_outputs: Sequence[str],
    ) -> list[TaggerPrediction]:
        if len(instructions) != len(tool_outputs):
            raise ValueError(f"instruction count {len(instructions)} != tool output count {len(tool_outputs)}")
        predictions: list[TaggerPrediction] = []
        for start in range(0, len(tool_outputs), self.batch_size):
            batch_instructions = instructions[start : start + self.batch_size]
            batch_outputs = tool_outputs[start : start + self.batch_size]
            predictions.extend(self._sanitize_batch_with_isolation(batch_instructions, batch_outputs))
        return predictions

    def _sanitize_batch_with_isolation(
        self,
        instructions: Sequence[str],
        tool_outputs: Sequence[str],
    ) -> list[TaggerPrediction]:
        started_at = time.monotonic()
        try:
            return self._sanitize_model_batch(instructions, tool_outputs)
        except Exception as exc:
            _clear_cuda_cache(exc)
            if len(tool_outputs) > 1:
                midpoint = len(tool_outputs) // 2
                return [
                    *self._sanitize_batch_with_isolation(instructions[:midpoint], tool_outputs[:midpoint]),
                    *self._sanitize_batch_with_isolation(instructions[midpoint:], tool_outputs[midpoint:]),
                ]

            latency_ms = int((time.monotonic() - started_at) * 1000)
            input_tokens = getattr(exc, "input_tokens", None)
            if not isinstance(input_tokens, list) or len(input_tokens) != 1:
                input_tokens = [0]
            return [
                TaggerPrediction(
                    filtered_tool_output=tool_outputs[0],
                    predicted_drop_spans=[],
                    input_tokens=int(input_tokens[0]),
                    latency_ms=latency_ms,
                    error=str(exc),
                )
            ]

    def _sanitize_model_batch(
        self,
        instructions: Sequence[str],
        tool_outputs: Sequence[str],
    ) -> list[TaggerPrediction]:
        if torch is None:
            raise ImportError("Token tagger inference requires PyTorch.")
        prompts = [
            self.prompt_builder(instruction, tool_output)
            for instruction, tool_output in zip(instructions, tool_outputs)
        ]
        encoded = self.tokenizer(
            [prompt.text for prompt in prompts],
            add_special_tokens=False,
            truncation=False,
            padding=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        attention_mask = encoded["attention_mask"]
        token_lengths = [int(value) for value in attention_mask.sum(dim=1)]
        context_limit = self._context_limit()
        for token_length in token_lengths:
            if context_limit is not None and token_length > context_limit:
                raise ValueError(f"tagger input has {token_length} tokens, exceeding context limit {context_limit}")

        input_ids = encoded["input_ids"].to(self.device)
        device_attention_mask = attention_mask.to(self.device)
        started_at = time.perf_counter()
        try:
            with torch.no_grad():
                if input_ids.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = self.model(input_ids=input_ids, attention_mask=device_attention_mask)
                else:
                    output = self.model(input_ids=input_ids, attention_mask=device_attention_mask)
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            raise TaggerBatchError(
                f"CUDA OOM for token lengths {token_lengths}",
                input_tokens=token_lengths,
            ) from exc

        latency_ms = int(((time.perf_counter() - started_at) * 1000) / len(prompts))
        batch_predictions = output.logits.argmax(dim=-1).detach().cpu().tolist()
        offsets = encoded["offset_mapping"].tolist()

        results = []
        for index, (prompt, tool_output) in enumerate(zip(prompts, tool_outputs)):
            spans = predictions_to_data_spans(
                batch_predictions[index],
                [tuple(offset) for offset in offsets[index]],
                prompt.data_start,
                prompt.data_end,
            )
            results.append(
                TaggerPrediction(
                    filtered_tool_output=reconstruct_without_spans(tool_output, spans),
                    predicted_drop_spans=spans,
                    input_tokens=token_lengths[index],
                    latency_ms=latency_ms,
                )
            )
        return results

    def _context_limit(self) -> int | None:
        candidates = [
            getattr(self.tokenizer, "model_max_length", None),
            getattr(getattr(self.model, "config", None), "max_position_embeddings", None),
        ]
        valid = [int(value) for value in candidates if isinstance(value, int) and 0 < value < 1_000_000_000]
        return min(valid) if valid else None


@register_defense
class ModernBERTTagger(BaseDefense):
    name = "modernbert_tagger"
    DEFAULT_CONFIG = {
        "checkpoint_path": None,
        "device": "cuda",
        "batch_size": 8,
        # ablation switch: feed an empty instruction to the tagger, so a
        # no-task-conditioning checkpoint runs in its operating condition
        "blank_instruction": False,
    }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._backend = self.config.get("backend")

    def execute(self, target_inst: str, context: str) -> dict:
        return self.execute_batch([target_inst], [context])[0]

    def execute_batch(self, target_insts: list[str], contexts: list[str]) -> list[dict]:
        self._validate_batch_inputs(target_insts, contexts)
        if self.config.get("blank_instruction"):
            target_insts = ["" for _ in target_insts]
        predictions = list(self._get_backend().sanitize_batch(target_insts, contexts))
        if len(predictions) != len(contexts):
            raise ValueError(f"tagger returned {len(predictions)} predictions for {len(contexts)} contexts")
        return [
            self._prediction_to_result(context, prediction)
            for context, prediction in zip(contexts, predictions)
        ]

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        checkpoint_path = self.config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("modernbert_tagger requires checkpoint_path in defense_config.")
        self._backend = TokenTaggerBackend.from_encoder_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device=str(self.config["device"]),
            batch_size=int(self.config["batch_size"]),
        )
        return self._backend

    def _prediction_to_result(self, context: str, prediction: Any) -> dict:
        cleaned_context = str(getattr(prediction, "filtered_tool_output", context))
        return {
            "detect_flag": cleaned_context != context,
            "cleaned_context": cleaned_context,
            "predicted_drop_spans": getattr(prediction, "predicted_drop_spans", []),
            "input_tokens": getattr(prediction, "input_tokens", 0),
            "latency_ms": getattr(prediction, "latency_ms", 0),
            "error": getattr(prediction, "error", None),
            "backend_type": getattr(self._get_backend(), "backend_type", "modernbert"),
            "checkpoint_fingerprint": getattr(self._get_backend(), "checkpoint_fingerprint", "unverified"),
        }


def _checkpoint_path(checkpoint_path: str) -> Path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir():
        raise ValueError(f"tagger checkpoint directory not found: {checkpoint}")
    return checkpoint


def _model_architecture(model: Any, backend_type: str) -> dict[str, Any]:
    config = getattr(model, "config", None)
    return {
        "backend_type": backend_type,
        "model_type": getattr(config, "model_type", "unknown"),
        "hidden_size": getattr(config, "hidden_size", None),
    }


def _read_tagger_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint / "tagger_config.json"
    if not config_path.is_file():
        raise ValueError(f"checkpoint is missing tagger_config.json: {checkpoint}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"tagger_config.json must contain an object: {checkpoint}")
    return config


def _validate_checkpoint_contract(
    config: dict[str, Any],
    *,
    expected_head_type: str,
    expected_prompt_format_version: int,
) -> None:
    if config.get("head_type") != expected_head_type:
        raise ValueError(f"tagger checkpoint must use head_type {expected_head_type!r}")
    if config.get("label2id") != EXPECTED_LABEL2ID:
        raise ValueError("tagger checkpoint label2id must be {'KEEP': 0, 'DROP': 1}")
    if config.get("prompt_format_version") != expected_prompt_format_version:
        raise ValueError(f"tagger checkpoint prompt_format_version must be {expected_prompt_format_version}")


def _validate_encoder_model_config(model: Any, checkpoint_config: dict[str, Any]) -> None:
    model_config = getattr(model, "config", None)
    model_type = getattr(model_config, "model_type", None)
    if model_type != checkpoint_config["base_model_type"]:
        raise ValueError(
            "loaded encoder model_type does not match tagger checkpoint: "
            f"{model_type!r} != {checkpoint_config['base_model_type']!r}"
        )
    if getattr(model_config, "label2id", None) != EXPECTED_LABEL2ID:
        raise ValueError("loaded encoder model label2id must be {'KEEP': 0, 'DROP': 1}")


def _checkpoint_fingerprint(checkpoint: str | Path) -> str:
    checkpoint_path = Path(checkpoint)
    files = sorted(path for path in checkpoint_path.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"tagger checkpoint contains no files: {checkpoint_path}")
    digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(checkpoint_path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _require_loader(loader: Any, dependency: str) -> None:
    if loader is None:
        raise ImportError(f"Token tagger inference requires the optional '{dependency}' dependency.")


def _clear_cuda_cache(exc: Exception) -> None:
    if torch is None:
        return
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if not isinstance(exc, TaggerBatchError) and (oom_type is None or not isinstance(exc, oom_type)):
        return
    cause = exc.__cause__
    exc.__traceback__ = None
    exc.__cause__ = None
    if cause is not None:
        cause.__traceback__ = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
