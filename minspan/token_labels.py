"""Map exact injection character spans to KEEP/DROP token labels."""

from __future__ import annotations

from typing import Any, Callable

from minspan.prompting import SerializedPrompt, build_tagger_prompt


KEEP_LABEL = 0
DROP_LABEL = 1
IGNORE_LABEL = -100


def _contains_non_whitespace(text: str, start: int, end: int) -> bool:
    return any(not character.isspace() for character in text[start:end])


def _token_label(
    prompt_text: str,
    token_start: int,
    token_end: int,
    data_start: int,
    data_end: int,
    shifted_drop_spans: list[tuple[int, int]],
) -> int:
    overlap_start = max(token_start, data_start)
    overlap_end = min(token_end, data_end)
    if overlap_start >= overlap_end:
        return IGNORE_LABEL

    for drop_start, drop_end in shifted_drop_spans:
        injected_start = max(overlap_start, drop_start)
        injected_end = min(overlap_end, drop_end)
        if injected_start >= injected_end:
            continue

        fully_inside_drop_span = (
            drop_start <= overlap_start and overlap_end <= drop_end
        )
        if fully_inside_drop_span or _contains_non_whitespace(
            prompt_text,
            injected_start,
            injected_end,
        ):
            return DROP_LABEL
    return KEEP_LABEL


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    prompt_builder: Callable[[str, str], SerializedPrompt] = build_tagger_prompt,
) -> dict[str, Any]:
    """Tokenize one record without truncation and align its character spans."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("token alignment requires a fast tokenizer")

    serialized = prompt_builder(
        str(record["instruction"]),
        str(record["attacked_data"]),
    )
    encoded = tokenizer(
        serialized.text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    offset_mapping = [tuple(offset) for offset in encoded["offset_mapping"]]
    if not (len(input_ids) == len(attention_mask) == len(offset_mapping)):
        raise ValueError("tokenizer returned fields with inconsistent lengths")

    shifted_drop_spans = [
        (
            serialized.data_start + int(span["start"]),
            serialized.data_start + int(span["end"]),
        )
        for span in record["drop_spans"]
    ]
    labels = [
        _token_label(
            serialized.text,
            token_start,
            token_end,
            serialized.data_start,
            serialized.data_end,
            shifted_drop_spans,
        )
        if token_start != token_end
        else IGNORE_LABEL
        for token_start, token_end in offset_mapping
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offset_mapping,
        "prompt_text": serialized.text,
        "data_start": serialized.data_start,
        "data_end": serialized.data_end,
    }
