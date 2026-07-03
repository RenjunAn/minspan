"""Convert the Nemotron agentic indirect-prompt-injection dataset into the
tagger character-level annotation schema (one JSONL record per attack).

Mapping
-------
Source record (agentic IPI): an ``injection.injection_text`` string is embedded
in one field of a tool-output object living inside ``environment``. We treat
that tool-output object (serialized as JSON) as the data blob the tagger must
scan, and mark the injected substring with a single ``drop_span``.

    instruction   <- the user message in responses_create_params.input
    attacked_data <- json.dumps(tool_output_object)        (contains injection)
    clean_data    <- json.dumps(tool_output_object - injection text)
    drop_spans    <- every occurrence of the injection inside attacked_data
    inserted_text <- attacked_data slice of the first span
    injection     <- the trimmed injection text (escaped to stay a substring)
    attack_type   <- "IndirectAgentic"   (position "middle", cut_type "none")

Some source records store only the payload in ``injection_text`` while the
environment embeds it inside a ``[PRIORITY MESSAGE]...[/PRIORITY MESSAGE]``
wrapper; the span is expanded to cover the whole wrapper so the template does
not survive into clean_data.

Every emitted record is checked against minspan.data_schema.validate_record.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from minspan.data_schema import RecordValidationError, remove_spans, validate_record


def _user_instruction(record: dict[str, Any]) -> str:
    """Concatenate the user-turn content from the responses API input."""
    messages = record["responses_create_params"]["input"]
    parts = [m["content"] for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
    return "\n".join(parts)


def _match_substring(value: str, injection_text: str) -> str | None:
    """Locate injection_text inside value and return the substring as it actually
    appears there. Falls back to whitespace-tolerant matching, because some
    source records render the injection's newlines as spaces in the field."""
    if injection_text in value:
        return injection_text
    tokens = re.split(r"\s+", injection_text.strip())
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, value)
    return match.group(0) if match else None


def _find_injection_container(node: Any, injection_text: str) -> tuple[dict, str, str] | None:
    """Return the innermost dict that directly holds injection_text in one of its
    string fields, that field's key, and the substring as it appears in the field.
    Depth-first; innermost match wins."""
    if isinstance(node, dict):
        for value in node.values():
            nested = _find_injection_container(value, injection_text)
            if nested is not None:
                return nested
        for key, value in node.items():
            if isinstance(value, str):
                actual = _match_substring(value, injection_text)
                if actual is not None:
                    return node, key, actual
    elif isinstance(node, list):
        for item in node:
            nested = _find_injection_container(item, injection_text)
            if nested is not None:
                return nested
    return None


def _escaped(text: str) -> str:
    """The substring as it appears inside a json.dumps(..., ensure_ascii=False) string."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


_WRAPPER_PATTERN = re.compile(
    r"\s*\[PRIORITY MESSAGE\].*?\[/PRIORITY MESSAGE\]", re.DOTALL
)


def _expand_to_wrapper(value: str, injected: str) -> str:
    """If the injected substring sits inside a [PRIORITY MESSAGE] wrapper in
    value, return the full wrapper (with leading whitespace); otherwise return
    the substring unchanged."""
    position = value.find(injected)
    for match in _WRAPPER_PATTERN.finditer(value):
        if match.start() <= position and position + len(injected) <= match.end():
            return match.group(0)
    return injected


def _strip_text(node: Any, text: str) -> Any:
    """Return a copy of node with every occurrence of text removed from all
    string fields."""
    if isinstance(node, dict):
        return {key: _strip_text(value, text) for key, value in node.items()}
    if isinstance(node, list):
        return [_strip_text(item, text) for item in node]
    if isinstance(node, str):
        return node.replace(text, "")
    return node


def convert_record(source: dict[str, Any], carrier: str = "container") -> dict[str, Any] | None:
    """Convert one Nemotron record; return None if the injection can't be located.

    ``carrier`` chooses the data blob the tagger scans:
        "container"   -> the innermost dict directly holding the injected field
        "environment" -> the full environment object the agent observes
    """
    injection_text = source.get("injection", {}).get("injection_text")
    if not injection_text:
        return None

    environment = source.get("environment", {})
    found = _find_injection_container(environment, injection_text)
    if found is None:
        return None
    container, field, actual_injection = found

    actual_injection = _expand_to_wrapper(container[field], actual_injection)

    blob = environment if carrier == "environment" else container

    attacked_data = json.dumps(blob, ensure_ascii=False)
    clean_data = json.dumps(_strip_text(blob, actual_injection), ensure_ascii=False)

    needle = _escaped(actual_injection)
    spans = []
    start = attacked_data.find(needle)
    while start >= 0:
        spans.append({"start": start, "end": start + len(needle)})
        start = attacked_data.find(needle, start + len(needle))
    if not spans:
        return None

    # The serializations must differ by exactly the injected spans.
    if remove_spans(attacked_data, spans) != clean_data:
        return None

    first, last = spans[0], spans[-1]
    inserted_text = attacked_data[first["start"] : first["end"]]
    injection = _escaped(actual_injection.strip()) or inserted_text
    if injection not in inserted_text:
        injection = inserted_text

    if first["start"] == 0:
        position = "prepend"
    elif last["end"] == len(attacked_data):
        position = "append"
    else:
        position = "middle"

    base_id = f"nemo-{source['id']:06d}"
    record = {
        "id": f"nemo_test-{base_id}-ipi",
        "base_id": base_id,
        "split": "nemo_test",
        "instruction": _user_instruction(source),
        "original_data": clean_data,
        "clean_data": clean_data,
        "attacked_data": attacked_data,
        "injection": injection,
        "inserted_text": inserted_text,
        "drop_spans": spans,
        "attack_type": "IndirectAgentic",
        "position": position,
        "cut_type": "none",
    }
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="data/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1.jsonl",
        help="Path to the Nemotron agentic IPI JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/nemo_test.jsonl",
        help="Destination tagger JSONL file",
    )
    parser.add_argument(
        "--carrier",
        choices=("container", "environment"),
        default="container",
        help="Data blob the tagger scans: innermost container (default) or full environment",
    )
    args = parser.parse_args(argv)

    total = converted = skipped = invalid = 0
    with open(args.source, encoding="utf-8") as src, open(args.output, "w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            source = json.loads(line)
            record = convert_record(source, carrier=args.carrier)
            if record is None:
                skipped += 1
                print(f"[skip] id={source.get('id')} (injection not locatable)")
                continue
            try:
                validate_record(record)
            except RecordValidationError as exc:
                invalid += 1
                print(f"[invalid] id={source.get('id')}: {exc}")
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"\nsource={total}  converted={converted}  skipped={skipped}  invalid={invalid}")
    print(f"wrote {converted} records -> {args.output}")


if __name__ == "__main__":
    main()
