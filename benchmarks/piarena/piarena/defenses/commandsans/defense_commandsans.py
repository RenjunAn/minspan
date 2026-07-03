"""CommandSans defense for PIArena.

Task-agnostic instruction removal from the untrusted context: a token
classifier flags instruction words in the context and deletes them; the user
task is not part of the classifier input. Implementation lives in
baselines/commandsans of this repository (pip install -e baselines/commandsans).
"""

from __future__ import annotations

from ..base import BaseDefense, register_defense


@register_defense
class CommandSans(BaseDefense):
    name = "commandsans"
    DEFAULT_CONFIG = {
        "checkpoint_path": None,
        "device": "cuda",
        "max_length": 512,
        "stride": 256,
    }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._sanitizer = None

    def _get_sanitizer(self):
        if self._sanitizer is None:
            checkpoint_path = self.config.get("checkpoint_path")
            if not checkpoint_path:
                raise ValueError("commandsans requires checkpoint_path in defense_config.")
            from commandsans import Sanitizer

            self._sanitizer = Sanitizer(
                str(checkpoint_path),
                device=str(self.config["device"]),
                max_length=int(self.config["max_length"]),
                stride=int(self.config["stride"]),
            )
        return self._sanitizer

    def execute(self, target_inst: str, context: str) -> dict:
        sanitizer = self._get_sanitizer()
        result = sanitizer.sanitize(context)
        return {
            "detect_flag": result.sanitized_text != context,
            "cleaned_context": result.sanitized_text,
            "predicted_drop_spans": result.removed_spans,
            "input_tokens": result.input_tokens,
            "latency_ms": result.latency_ms,
            "error": None,
            "backend_type": "commandsans",
            "checkpoint_fingerprint": sanitizer.fingerprint,
        }
