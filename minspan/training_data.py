"""Lazy JSONL loading, dynamic padding, and preflight checks for tagger training."""

from __future__ import annotations

import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from minspan.data_schema import RecordValidationError, validate_record
from minspan.data_stats import percentile
from minspan.prompting import build_tagger_prompt
from minspan.token_labels import IGNORE_LABEL, encode_record


class ContextLengthError(ValueError):
    """Raised when a record exceeds the backbone context window."""


class JsonlTaggerDataset(Dataset):
    """Random-access JSONL dataset backed by byte offsets."""

    def __init__(self, path: str | Path, limit: int | None = None):
        self.path = Path(path)
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        offsets: list[tuple[int, int]] = []
        with self.path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if line.strip():
                    offsets.append((offset, line_number))
                    if limit is not None and len(offsets) >= limit:
                        break

        self._offsets = offsets
        self._handle = None
        self._handle_pid: int | None = None

    def __len__(self) -> int:
        return len(self._offsets)

    def _get_handle(self):
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            if self._handle is not None:
                self._handle.close()
            self._handle = self.path.open("rb")
            self._handle_pid = pid
        return self._handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        offset, line_number = self._offsets[index]
        handle = self._get_handle()
        handle.seek(offset)
        raw_line = handle.readline()
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{self.path} line {line_number}: invalid JSON") from error

        record_id = record.get("id", "<unknown>") if isinstance(record, dict) else "<unknown>"
        try:
            validate_record(record)
        except (RecordValidationError, TypeError) as error:
            raise ValueError(
                f"{self.path} line {line_number} record {record_id}: {error}"
            ) from error
        return record

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()


class TaggerCollator:
    """Tokenize raw records and dynamically pad to the longest sequence.

    instruction_dropout blanks the instruction with the given probability per
    record (training-time regularization; keep 0 for evaluation collators)."""

    def __init__(
        self,
        tokenizer: Any,
        prompt_builder: Any = build_tagger_prompt,
        instruction_dropout: float = 0.0,
        dropout_seed: int = 0,
    ):
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        if not 0.0 <= instruction_dropout <= 1.0:
            raise ValueError("instruction_dropout must be between 0 and 1")
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder
        self.instruction_dropout = instruction_dropout
        self._dropout_rng = random.Random(dropout_seed)

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("cannot collate an empty batch")

        if self.instruction_dropout:
            records = [
                {**record, "instruction": ""}
                if self._dropout_rng.random() < self.instruction_dropout
                else record
                for record in records
            ]

        encoded_records = [
            encode_record(self.tokenizer, record, prompt_builder=self.prompt_builder)
            for record in records
        ]
        token_lengths = [len(encoded["input_ids"]) for encoded in encoded_records]
        max_length = max(token_lengths)
        batch_size = len(records)

        input_ids = torch.full(
            (batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full(
            (batch_size, max_length),
            IGNORE_LABEL,
            dtype=torch.long,
        )
        offset_mapping = torch.zeros(
            (batch_size, max_length, 2),
            dtype=torch.long,
        )

        for row, encoded in enumerate(encoded_records):
            length = token_lengths[row]
            input_ids[row, :length] = torch.tensor(
                encoded["input_ids"],
                dtype=torch.long,
            )
            attention_mask[row, :length] = torch.tensor(
                encoded["attention_mask"],
                dtype=torch.long,
            )
            labels[row, :length] = torch.tensor(encoded["labels"], dtype=torch.long)
            offset_mapping[row, :length] = torch.tensor(
                encoded["offset_mapping"],
                dtype=torch.long,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "offset_mapping": offset_mapping,
            "data_ranges": [
                (encoded["data_start"], encoded["data_end"])
                for encoded in encoded_records
            ],
            "record_ids": [str(record["id"]) for record in records],
            "records": records,
            "token_lengths": token_lengths,
        }


def inspect_dataset(
    dataset: Dataset,
    tokenizer: Any,
    *,
    model_max_length: int,
    prompt_builder: Any = build_tagger_prompt,
) -> dict[str, int | float]:
    """Validate all selected records and report untruncated token lengths."""
    if model_max_length <= 0:
        raise ValueError("model_max_length must be positive")
    if len(dataset) == 0:
        raise ValueError("dataset must contain at least one record")

    lengths: list[int] = []
    for index in range(len(dataset)):
        record = dataset[index]
        encoded = encode_record(tokenizer, record, prompt_builder=prompt_builder)
        token_length = len(encoded["input_ids"])
        if token_length > model_max_length:
            raise ContextLengthError(
                f"record {record['id']} has {token_length} tokens, exceeding "
                f"model context limit {model_max_length}"
            )
        if not any(label != IGNORE_LABEL for label in encoded["labels"]):
            raise ValueError(f"record {record['id']} has no supervised data tokens")
        lengths.append(token_length)

    return {
        "count": len(lengths),
        "min": min(lengths),
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "p95": percentile(lengths, 0.95),
        "max": max(lengths),
    }
