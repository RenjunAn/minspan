"""Record schema and validation for character-level injection annotations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "base_id",
    "split",
    "instruction",
    "original_data",
    "clean_data",
    "attacked_data",
    "injection",
    "inserted_text",
    "drop_spans",
    "attack_type",
    "position",
    "cut_type",
}
VALID_SPLITS = {
    "train",
    "validation",
    "sep_test",
    "nemo_test",
    "format_test",
    "p3_direct_test",
    "p3_strategy_test",
    "p3_clean_hard_negative_test",
}
VALID_ATTACK_TYPES = {"Clean", "Naive", "Ignore", "Completion", "IndirectAgentic"}
VALID_POSITIONS = {"none", "prepend", "middle", "append"}
VALID_CUT_TYPES = {"none", "empty", "two_thirds", "half"}


class RecordValidationError(ValueError):
    """Raised when a generated tagger record violates the data contract."""


def remove_spans(text: str, spans: Sequence[dict[str, int]]) -> str:
    """Remove half-open character spans without invalidating later offsets."""
    result = text
    for span in reversed(spans):
        result = result[: span["start"]] + result[span["end"] :]
    return result


def _require_string(record: dict[str, Any], field: str) -> None:
    if not isinstance(record[field], str):
        raise RecordValidationError(f"{field} must be a string")


def _validate_spans(text: str, spans: Any) -> None:
    if not isinstance(spans, list):
        raise RecordValidationError("drop_spans must be a list")

    previous_end = 0
    for index, span in enumerate(spans):
        if not isinstance(span, dict) or set(span) != {"start", "end"}:
            raise RecordValidationError(f"drop_spans[{index}] must contain start/end")
        start = span["start"]
        end = span["end"]
        if not isinstance(start, int) or isinstance(start, bool):
            raise RecordValidationError(f"drop_spans[{index}].start must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise RecordValidationError(f"drop_spans[{index}].end must be an integer")
        if not 0 <= start < end <= len(text):
            raise RecordValidationError(f"drop_spans[{index}] is out of bounds")
        if index and start < previous_end:
            raise RecordValidationError("drop_spans must be sorted and non-overlapping")
        previous_end = end


def validate_record(record: dict[str, Any]) -> None:
    """Validate one generated JSONL record."""
    if not isinstance(record, dict):
        raise RecordValidationError("record must be an object")

    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise RecordValidationError(f"missing required fields: {sorted(missing)}")

    for field in REQUIRED_FIELDS - {"drop_spans"}:
        _require_string(record, field)

    if record["split"] not in VALID_SPLITS:
        raise RecordValidationError(f"invalid split: {record['split']}")
    if record["attack_type"] not in VALID_ATTACK_TYPES:
        raise RecordValidationError(f"invalid attack_type: {record['attack_type']}")
    if record["position"] not in VALID_POSITIONS:
        raise RecordValidationError(f"invalid position: {record['position']}")
    if record["cut_type"] not in VALID_CUT_TYPES:
        raise RecordValidationError(f"invalid cut_type: {record['cut_type']}")
    if not record["id"] or not record["base_id"]:
        raise RecordValidationError("id and base_id must not be empty")

    spans = record["drop_spans"]
    attacked_data = record["attacked_data"]
    _validate_spans(attacked_data, spans)

    if record["attack_type"] == "Clean":
        if spans:
            raise RecordValidationError("Clean records must not contain drop_spans")
        if record["injection"] or record["inserted_text"]:
            raise RecordValidationError("Clean records must not contain injected text")
        if record["position"] != "none" or record["cut_type"] != "none":
            raise RecordValidationError("Clean records must use none position/cut_type")
        if attacked_data != record["clean_data"]:
            raise RecordValidationError("Clean attacked_data must equal clean_data")
        return

    if not spans:
        raise RecordValidationError("attacked records must contain at least one drop_span")
    if not record["injection"] or not record["inserted_text"]:
        raise RecordValidationError("attacked records must contain injection/inserted_text")
    if record["position"] == "none":
        raise RecordValidationError("attacked records must have an insertion position")

    span = spans[0]
    actual_inserted = attacked_data[span["start"] : span["end"]]
    if actual_inserted != record["inserted_text"]:
        raise RecordValidationError("inserted_text does not match its first drop_span")
    if record["injection"] not in record["inserted_text"]:
        raise RecordValidationError("inserted_text must contain the semantic injection")
    if remove_spans(attacked_data, spans) != record["clean_data"]:
        raise RecordValidationError("removing drop_spans does not reconstruct clean_data")
