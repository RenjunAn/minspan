"""Derive injection-free counterpart datasets from attacked tagger JSONL files.

Each attacked record becomes a Clean record whose ``attacked_data`` is the
ground-truth ``clean_data``, suitable for measuring false-positive rates on
distributions that ship without clean negatives (e.g. the nemo test sets).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_ATTACK_ID_SUFFIXES = ("-ipi", "-attacked")
# Injection prefixes shorter than this are too likely to collide with benign
# text (single common words), so they are excluded from the cross-record check.
_MIN_PREFIX_CHARS = 40


def _apply_drop_spans(attacked_data: str, drop_spans: list[dict[str, int]]) -> str:
    kept = []
    cursor = 0
    for span in sorted(drop_spans, key=lambda s: s["start"]):
        kept.append(attacked_data[cursor : span["start"]])
        cursor = span["end"]
    kept.append(attacked_data[cursor:])
    return "".join(kept)


def _clean_id(record_id: str) -> str:
    for suffix in _ATTACK_ID_SUFFIXES:
        if record_id.endswith(suffix):
            return record_id[: -len(suffix)] + "-clean"
    return record_id + "-clean"


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the Clean counterpart of an attacked tagger record."""
    reconstructed = _apply_drop_spans(record["attacked_data"], record["drop_spans"])
    if reconstructed != record["clean_data"]:
        raise ValueError(
            f"record {record['id']!r}: applying drop_spans to attacked_data "
            "does not reproduce clean_data; refusing to derive a clean record"
        )
    clean_data = record["clean_data"]
    return {
        **record,
        "id": _clean_id(record["id"]),
        "original_data": clean_data,
        "attacked_data": clean_data,
        "injection": "",
        "inserted_text": "",
        "drop_spans": [],
        "attack_type": "Clean",
        "position": "none",
        "cut_type": "none",
    }


def _is_leaky(record: dict[str, Any], injection_prefixes: set[str]) -> bool:
    clean_data = record["clean_data"]
    if record["injection"] and record["injection"] in clean_data:
        return True
    return any(prefix in clean_data for prefix in injection_prefixes)


def convert_file(
    input_path: Path,
    output_path: Path,
    drop_containing: tuple[str, ...] = (),
) -> tuple[int, int]:
    """Write clean counterparts; returns (written, skipped_leaky) counts.

    Records whose ground-truth ``clean_data`` still contains injection text
    are skipped: their "clean" form is not actually injection-free. This
    covers both the record's own injection (duplicate occurrences the
    drop_spans never covered) and injection templates shared across the file
    (matched via 40-char prefixes of every record's injection).
    """
    records = [
        json.loads(line) for line in input_path.open(encoding="utf-8")
    ]
    injection_prefixes = {
        record["injection"][:_MIN_PREFIX_CHARS]
        for record in records
        if len(record["injection"]) >= _MIN_PREFIX_CHARS
    }
    written = skipped = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for record in records:
            if _is_leaky(record, injection_prefixes) or any(
                marker in record["clean_data"] for marker in drop_containing
            ):
                skipped += 1
                continue
            sink.write(json.dumps(clean_record(record), ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Attacked tagger JSONL file")
    parser.add_argument("output", type=Path, help="Destination for clean JSONL")
    parser.add_argument(
        "--drop-containing",
        action="append",
        default=[],
        metavar="TEXT",
        help="Skip records whose clean_data contains TEXT (repeatable); use for "
        "dataset-specific injection markers the generic filters cannot infer",
    )
    args = parser.parse_args(argv)

    written, skipped = convert_file(
        args.input, args.output, drop_containing=tuple(args.drop_containing)
    )
    print(
        f"wrote {written} clean records to {args.output}"
        + (f" (skipped {skipped} with injection text leaked into clean_data)" if skipped else "")
    )


if __name__ == "__main__":
    main()
