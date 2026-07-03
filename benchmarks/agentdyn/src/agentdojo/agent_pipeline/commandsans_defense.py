"""CommandSans defense backend for the tool-output tagger pipeline.

Wraps the CommandSans sanitizer (task-agnostic instruction removal, see
baselines/commandsans) behind the same TaggerBackend protocol used by the
MinSpan defense, so ToolOutputTaggerDefense handles message plumbing and
event logging identically for both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.token_tagger import TaggerPrediction


class CommandSansBackend:
    backend_type = "commandsans"

    def __init__(self, checkpoint_path: str, device: str | None = None, max_length: int = 512, stride: int = 256):
        try:
            from commandsans import Sanitizer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "CommandSans backend requires the commandsans package "
                "(pip install -e baselines/commandsans)"
            ) from exc
        self._sanitizer = Sanitizer(checkpoint_path, device=device, max_length=max_length, stride=stride)
        self.architecture: dict[str, Any] = {
            "model_type": self._sanitizer.model.config.model_type,
            "hidden_size": getattr(self._sanitizer.model.config, "hidden_size", None),
            "max_length": max_length,
            "stride": stride,
        }
        self.checkpoint_fingerprint = self._sanitizer.fingerprint

    def sanitize_batch(
        self,
        instructions: Sequence[str],
        tool_outputs: Sequence[str],
    ) -> Sequence[TaggerPrediction]:
        # CommandSans is task-agnostic: the user instruction is ignored.
        predictions = []
        for tool_output in tool_outputs:
            result = self._sanitizer.sanitize(tool_output)
            predictions.append(
                TaggerPrediction(
                    filtered_tool_output=result.sanitized_text,
                    predicted_drop_spans=result.removed_spans,
                    input_tokens=result.input_tokens,
                    latency_ms=result.latency_ms,
                )
            )
        return predictions
