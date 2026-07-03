"""Training-data handling for the CommandSans token classifier.

Annotated records mark instruction content with <instruction>...</instruction>
tags inside otherwise verbatim text:

    {"id": "orca-000123", "labeled_text": "Weather: sunny. <instruction>Ignore
     previous instructions and ...</instruction> High 24C."}

`parse_labeled_text` strips the tags and produces word-level binary labels
(1 = instruction). `WindowedTokenDataset` tokenizes word-aligned examples into
overlapping windows so long tool outputs fit the encoder context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

OPEN_TAG = "<instruction>"
CLOSE_TAG = "</instruction>"
INSTRUCTION_LABEL = 1
DATA_LABEL = 0
IGNORE_INDEX = -100


@dataclass(frozen=True)
class LabeledExample:
    example_id: str
    words: list[str]
    labels: list[int]  # one per word


def parse_labeled_text(example_id: str, labeled_text: str) -> LabeledExample:
    """Convert tagged text into (words, word labels); tags must be balanced."""
    if labeled_text.count(OPEN_TAG) != labeled_text.count(CLOSE_TAG):
        raise ValueError(f"{example_id}: unbalanced instruction tags")
    words: list[str] = []
    labels: list[int] = []
    inside = False
    pattern = re.compile(f"({re.escape(OPEN_TAG)}|{re.escape(CLOSE_TAG)})")
    for part in pattern.split(labeled_text):
        if part == OPEN_TAG:
            if inside:
                raise ValueError(f"{example_id}: nested instruction tags")
            inside = True
        elif part == CLOSE_TAG:
            if not inside:
                raise ValueError(f"{example_id}: stray closing tag")
            inside = False
        elif part:
            for word in part.split():
                words.append(word)
                labels.append(INSTRUCTION_LABEL if inside else DATA_LABEL)
    if inside:
        raise ValueError(f"{example_id}: unterminated instruction tag")
    return LabeledExample(example_id, words, labels)


def strip_tags(labeled_text: str) -> str:
    return labeled_text.replace(OPEN_TAG, "").replace(CLOSE_TAG, "")


def load_labeled_jsonl(path: str | Path) -> list[LabeledExample]:
    examples = []
    with open(path) as fh:
        for line in fh:
            record = json.loads(line)
            examples.append(parse_labeled_text(str(record["id"]), record["labeled_text"]))
    return examples


class WindowedTokenDataset(Dataset):
    """Word-aligned token-classification windows.

    Every window carries its example id so train/validation splits can be made
    at the example level (windows of one example never cross the split).
    Subword tokens after the first of each word are ignored in the loss.
    """

    def __init__(self, examples, tokenizer, max_length: int = 512, stride: int = 256):
        self.windows: list[dict] = []
        for example in examples:
            encoded = tokenizer(
                example.words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
            )
            for window_index in range(len(encoded["input_ids"])):
                word_ids = encoded.word_ids(window_index)
                labels = []
                previous = None
                for word_id in word_ids:
                    if word_id is None or word_id == previous:
                        labels.append(IGNORE_INDEX)
                    else:
                        labels.append(example.labels[word_id])
                    previous = word_id
                self.windows.append(
                    {
                        "example_id": example.example_id,
                        "input_ids": encoded["input_ids"][window_index],
                        "attention_mask": encoded["attention_mask"][window_index],
                        "labels": labels,
                    }
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        return self.windows[index]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights over the two classes."""
        counts = [0, 0]
        for window in self.windows:
            for label in window["labels"]:
                if label != IGNORE_INDEX:
                    counts[label] += 1
        total = sum(counts)
        weights = [total / (2 * count) if count else 0.0 for count in counts]
        return torch.tensor(weights, dtype=torch.float)


def collate(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    width = max(len(item["input_ids"]) for item in batch)

    def pad(values, fill):
        return values + [fill] * (width - len(values))

    return {
        "input_ids": torch.tensor([pad(b["input_ids"], pad_token_id) for b in batch]),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.tensor([pad(b["labels"], IGNORE_INDEX) for b in batch]),
    }
