"""Inference: remove predicted instruction spans from tool outputs."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from commandsans.data import INSTRUCTION_LABEL


@dataclass(frozen=True)
class SanitizeResult:
    sanitized_text: str
    removed_spans: list[dict[str, int]] = field(default_factory=list)  # char offsets in the original
    input_tokens: int = 0
    latency_ms: int = 0


class Sanitizer:
    """Loads a trained CommandSans checkpoint and sanitizes text.

    Prediction is word-level (first-subword logits); a word predicted
    INSTRUCTION is removed together with its trailing whitespace. Long inputs
    are processed in overlapping windows; a word is removed if any window
    predicts INSTRUCTION for it.
    """

    def __init__(self, checkpoint: str | Path, device: str | None = None,
                 max_length: int = 512, stride: int = 256):
        self.checkpoint = str(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(self.checkpoint)
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.stride = stride
        self.fingerprint = _fingerprint(self.checkpoint)

    @torch.no_grad()
    def sanitize(self, text: str) -> SanitizeResult:
        started = time.monotonic()
        words, offsets = _split_with_offsets(text)
        if not words:
            return SanitizeResult(sanitized_text=text)

        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = logits.argmax(dim=-1).cpu()

        flagged = [False] * len(words)
        for window_index in range(input_ids.shape[0]):
            word_ids = encoded.word_ids(window_index)
            previous = None
            for position, word_id in enumerate(word_ids):
                if word_id is None or word_id == previous:
                    previous = word_id
                    continue
                if predictions[window_index][position].item() == INSTRUCTION_LABEL:
                    flagged[word_id] = True
                previous = word_id

        removed_spans = _merge_flagged_offsets(offsets, flagged)
        # reconstruct from the original string so untouched content stays verbatim
        sanitized = []
        cursor = 0
        for span in removed_spans:
            sanitized.append(text[cursor : span["start"]])
            cursor = span["end"]
            # swallow whitespace left dangling after the removed span
            while cursor < len(text) and text[cursor] in " \t":
                cursor += 1
        sanitized.append(text[cursor:])
        return SanitizeResult(
            sanitized_text="".join(sanitized),
            removed_spans=removed_spans,
            input_tokens=int(attention_mask.sum().item()),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


def _split_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words, offsets = [], []
    index = 0
    for word in text.split():
        start = text.index(word, index)
        words.append(word)
        offsets.append((start, start + len(word)))
        index = start + len(word)
    return words, offsets


def _merge_flagged_offsets(offsets, flagged) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    for (start, end), bad in zip(offsets, flagged):
        if not bad:
            continue
        if spans and start <= spans[-1]["end"] + 1:
            spans[-1]["end"] = end
        else:
            spans.append({"start": start, "end": end})
    return spans


def _fingerprint(checkpoint: str) -> str:
    digest = hashlib.sha256()
    root = Path(checkpoint)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".safetensors", ".bin", ".json"):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
