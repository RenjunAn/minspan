"""Generate DataFilter tagger records with exact injection character spans."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from config import (
    DELIMITERS,
    IGNORE_ATTACK_SENTENCES,
    OTHER_DELM_FOR_TEST,
    OTHER_DELM_TOKENS,
)
from minspan.augment import (
    ALL_ENVELOPE_FORMATS,
    choose_envelope_format,
    escape_for_envelope,
    insert_hard_negative,
    wrap_in_envelope,
    wrap_injection,
)
from minspan.data_schema import validate_record


ATTACK_TYPES = ("Clean", "Naive", "Ignore", "Completion")
WRAPPED_ATTACK_TYPES = ("Naive", "Ignore")


@dataclasses.dataclass(frozen=True)
class AugmentationConfig:
    """Phase 2 augmentation probabilities; all-zero reproduces Phase 1 output
    byte-for-byte (no RNG draws are consumed for disabled augmentations)."""

    envelope_prob: float = 0.0
    wrapper_prob: float = 0.0
    hard_negative_prob: float = 0.0
    wrapper_pool: str = "train"
    envelope_format: str | None = None


PHASE2_AUGMENTATION = AugmentationConfig(
    envelope_prob=0.25,
    wrapper_prob=0.30,
    hard_negative_prob=0.30,
)


def derive_injected_prompt(source: dict[str, Any]) -> str:
    """Match the prompt construction used by the original data generator."""
    instruction = str(source.get("instruction", "")).strip()
    input_text = str(source.get("input", "")).strip()
    if not instruction:
        prompt = input_text
    elif instruction.endswith("?"):
        prompt = f"answer the following question. {instruction}"
        if input_text:
            prompt += f" {input_text}"
    else:
        prompt = instruction[0].lower() + instruction[1:]
        if input_text:
            prompt += f" {input_text}"
        if prompt and not prompt.rstrip().endswith((".", "!", "?")):
            prompt += "."
    return prompt


def choose_position(rng: random.Random) -> str:
    """Use the repository's 20% append, 20% prepend, 60% random split."""
    value = rng.random()
    if value < 0.2:
        return "append"
    if value < 0.4:
        return "prepend"
    return "middle"


def cut_benign(text: str, rng: random.Random) -> tuple[str, str]:
    """Apply the original cut_benign probabilities and character slicing."""
    value = rng.random()
    if value < 0.15:
        return "", "empty"
    if value < 0.25:
        return text[: int(len(text) * (2 / 3))].strip(), "two_thirds"
    if value < 0.35:
        return text[: int(len(text) * (1 / 2))].strip(), "half"
    return text, "none"


def _sample_delimiter_name(
    delimiter_name: str,
    rng: random.Random,
    test_templates: bool,
) -> str:
    role_name = "user" if delimiter_name in {"inst", "inpt"} else "asst"
    split_at = -OTHER_DELM_FOR_TEST
    if test_templates:
        roles = OTHER_DELM_TOKENS[role_name][split_at:]
        delimiters = OTHER_DELM_TOKENS[delimiter_name][split_at:]
    else:
        roles = OTHER_DELM_TOKENS[role_name][:split_at]
        delimiters = OTHER_DELM_TOKENS[delimiter_name][:split_at]

    value = rng.choice(roles) + rng.choice(delimiters)
    casing = rng.random()
    if casing < 1 / 3:
        return value.upper()
    if casing < 2 / 3:
        return value.lower()
    return value


def format_with_other_delimiters(
    text: str,
    rng: random.Random,
    test_templates: bool,
) -> str:
    """Randomize Completion delimiters using the original delimiter pools."""
    split_at = -OTHER_DELM_FOR_TEST
    marks = (
        OTHER_DELM_TOKENS["mark"][split_at:]
        if test_templates
        else OTHER_DELM_TOKENS["mark"][:split_at]
    )
    mark = rng.choice(marks) + ":"
    result = text
    for delimiter_group in DELIMITERS.values():
        replacements = (
            _sample_delimiter_name("inst", rng, test_templates),
            _sample_delimiter_name("inpt", rng, test_templates),
            _sample_delimiter_name("resp", rng, test_templates),
        )
        for original, replacement in zip(delimiter_group, replacements):
            result = result.replace(original, mark.format(s=replacement))
    return result


def _capitalize_first(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def build_attack_content(
    attack_type: str,
    target: dict[str, Any],
    injected_source: dict[str, Any],
    rng: random.Random,
    test_templates: bool,
    provided_injection: str | None = None,
) -> str:
    """Build the exact semantic attack block inserted into benign data."""
    injected_prompt = (
        provided_injection
        if provided_injection is not None
        else derive_injected_prompt(injected_source)
    )
    if not injected_prompt:
        raise ValueError("injected prompt must not be empty")

    if attack_type == "Naive":
        return (
            injected_prompt
            if provided_injection is not None
            else _capitalize_first(injected_prompt)
        )

    if attack_type == "Ignore":
        pool_name = "test" if test_templates else "train"
        template = rng.choice(IGNORE_ATTACK_SENTENCES[pool_name])
        return template.format(injected_prompt=injected_prompt)

    if attack_type == "Completion":
        response = str(target.get("output", ""))
        semantic_prompt = (
            injected_prompt
            if provided_injection is not None
            else injected_prompt.capitalize()
        )
        content = (
            "\n\n"
            + DELIMITERS["SpclSpclSpcl"][2]
            + "\n"
            + response
            + "\n\n"
            + DELIMITERS["SpclSpclSpcl"][0]
            + "\n"
            + semantic_prompt
        )
        injected_input = str(injected_source.get("input", ""))
        if provided_injection is None and injected_input:
            content += (
                "\n\n"
                + DELIMITERS["SpclSpclSpcl"][1]
                + "\n"
                + injected_input
            )
        return format_with_other_delimiters(content, rng, test_templates)

    raise ValueError(f"unsupported attack type: {attack_type}")


def _separator(left: str, right: str) -> str:
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return ""
    return " "


def _insertion_boundaries(text: str) -> list[int]:
    """Word-end boundaries, so the separator joining a middle insertion sits
    BEFORE the injection. Leading-separator spans align with the BPE
    tokenizer's leading-space tokens; trailing separators land inside the
    next clean token and become unlearnable."""
    ends = [match.end() for match in re.finditer(r"\S+", text)]
    return list(dict.fromkeys(ends)) or [len(text)]


def insert_attack(
    clean_data: str,
    injection: str,
    position: str,
    rng: random.Random,
) -> tuple[str, str, list[dict[str, int]]]:
    """Insert one contiguous attack and return its exact deletion span."""
    if not injection:
        raise ValueError("injection must not be empty")
    if position not in {"prepend", "middle", "append"}:
        raise ValueError(f"unsupported position: {position}")

    if not clean_data:
        return injection, injection, [{"start": 0, "end": len(injection)}]

    if position == "prepend":
        inserted_text = injection + _separator(injection, clean_data)
        attacked_data = inserted_text + clean_data
        start = 0
    elif position == "append":
        inserted_text = _separator(clean_data, injection) + injection
        attacked_data = clean_data + inserted_text
        start = len(clean_data)
    else:
        boundaries = _insertion_boundaries(clean_data)
        offset = boundaries[rng.randrange(len(boundaries))]
        left = clean_data[:offset]
        right = clean_data[offset:]
        inserted_text = (
            _separator(left, injection)
            + injection
            + _separator(injection, right)
        )
        attacked_data = left + inserted_text + right
        start = offset

    end = start + len(inserted_text)
    return attacked_data, inserted_text, [{"start": start, "end": end}]


def _drawn(rng: random.Random, probability: float) -> bool:
    """Probability gate that consumes no RNG draw when disabled, so an
    all-zero AugmentationConfig leaves the Phase 1 RNG stream untouched."""
    return probability > 0 and rng.random() < probability


def make_variant(
    *,
    base_id: str,
    split: str,
    target: dict[str, Any],
    injected_source: dict[str, Any],
    attack_type: str,
    rng: random.Random,
    test_templates: bool,
    cut_attacked: bool,
    provided_injection: str | None = None,
    augmentation: AugmentationConfig | None = None,
) -> dict[str, Any]:
    """Create and validate one Clean or attacked tagger record."""
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"unsupported attack type: {attack_type}")

    instruction = str(target.get("instruction_only") or target.get("instruction", ""))
    original_data = str(target.get("input", ""))
    record_id = f"{split}-{base_id}-{attack_type.lower()}"
    applied: dict[str, Any] = {}

    if attack_type == "Clean":
        clean_data = original_data
        if augmentation is not None:
            if _drawn(rng, augmentation.hard_negative_prob):
                clean_data = insert_hard_negative(clean_data, rng)
                applied["hard_negative"] = True
            if _drawn(rng, augmentation.envelope_prob):
                format_name = augmentation.envelope_format or choose_envelope_format(rng)
                clean_data, _, _ = wrap_in_envelope(
                    clean_data, clean_data, [], format_name, rng
                )
                applied["envelope_format"] = format_name
        record = {
            "id": record_id,
            "base_id": base_id,
            "split": split,
            "instruction": instruction,
            "original_data": original_data,
            "clean_data": clean_data,
            "attacked_data": clean_data,
            "injection": "",
            "inserted_text": "",
            "drop_spans": [],
            "attack_type": "Clean",
            "position": "none",
            "cut_type": "none",
        }
        if applied:
            record["augmentation"] = applied
        validate_record(record)
        return record

    clean_data, cut_type = (
        cut_benign(original_data, rng) if cut_attacked else (original_data, "none")
    )
    injection = build_attack_content(
        attack_type,
        target,
        injected_source,
        rng,
        test_templates,
        provided_injection=provided_injection,
    )
    if (
        augmentation is not None
        and attack_type in WRAPPED_ATTACK_TYPES
        and _drawn(rng, augmentation.wrapper_prob)
    ):
        injection = wrap_injection(injection, rng, pool=augmentation.wrapper_pool)
        applied["wrapper"] = True
    position = choose_position(rng)
    attacked_data, inserted_text, drop_spans = insert_attack(
        clean_data,
        injection,
        position,
        rng,
    )
    if augmentation is not None and _drawn(rng, augmentation.envelope_prob):
        format_name = augmentation.envelope_format or choose_envelope_format(rng)
        attacked_data, clean_data, drop_spans = wrap_in_envelope(
            attacked_data, clean_data, drop_spans, format_name, rng
        )
        first = drop_spans[0]
        inserted_text = attacked_data[first["start"] : first["end"]]
        injection = escape_for_envelope(injection, format_name)
        applied["envelope_format"] = format_name
    record = {
        "id": record_id,
        "base_id": base_id,
        "split": split,
        "instruction": instruction,
        "original_data": original_data,
        "clean_data": clean_data,
        "attacked_data": attacked_data,
        "injection": injection,
        "inserted_text": inserted_text,
        "drop_spans": drop_spans,
        "attack_type": attack_type,
        "position": position,
        "cut_type": cut_type,
    }
    if applied:
        record["augmentation"] = applied
    validate_record(record)
    return record


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """Load a source JSON file and require a top-level array of objects."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a JSON array of objects")
    return value


def select_nonempty_inputs(
    records: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    """Keep source indices for records with a non-empty input field."""
    return [
        (index, record)
        for index, record in enumerate(records)
        if str(record.get("input", "")) != ""
    ]


def split_alpaca(
    indexed_records: list[tuple[int, dict[str, Any]]],
    *,
    validation_size: int,
    seed: int,
) -> tuple[
    list[tuple[int, dict[str, Any]]],
    list[tuple[int, dict[str, Any]]],
]:
    """Create a deterministic, disjoint train/validation source split."""
    if validation_size < 0 or validation_size >= len(indexed_records):
        raise ValueError("validation_size must leave at least one training record")
    shuffled = list(indexed_records)
    random.Random(seed).shuffle(shuffled)
    validation = shuffled[:validation_size]
    train = shuffled[validation_size:]
    return train, validation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(path: Path, total_records: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "records": total_records,
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def _write_alpaca_split(
    path: Path,
    *,
    split: str,
    records: list[tuple[int, dict[str, Any]]],
    injection_sources: list[dict[str, Any]],
    rng: random.Random,
    seen_ids: set[str],
    augmentation: AugmentationConfig | None = None,
) -> int:
    if not injection_sources:
        raise ValueError("Alpaca injection source pool must not be empty")

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for source_index, target in records:
            base_id = f"alpaca-{source_index:06d}"
            for attack_type in ATTACK_TYPES:
                injected_source = (
                    target
                    if attack_type == "Clean"
                    else rng.choice(injection_sources)
                )
                record = make_variant(
                    base_id=base_id,
                    split=split,
                    target=target,
                    injected_source=injected_source,
                    attack_type=attack_type,
                    rng=rng,
                    test_templates=False,
                    cut_attacked=True,
                    augmentation=augmentation,
                )
                if record["id"] in seen_ids:
                    raise ValueError(f"duplicate record id: {record['id']}")
                seen_ids.add(record["id"])
                _write_jsonl_record(handle, record)
                count += 1
    return count


def _write_format_test_split(
    path: Path,
    *,
    records: list[tuple[int, dict[str, Any]]],
    injection_sources: list[dict[str, Any]],
    rng: random.Random,
    seen_ids: set[str],
) -> int:
    """One attacked + one Clean record per (source, envelope format), covering
    held-out formats; injections use the *test* wrapper pool."""
    count = 0
    attacked_types = tuple(t for t in ATTACK_TYPES if t != "Clean")
    with path.open("w", encoding="utf-8") as handle:
        for source_index, target in records:
            for format_name in ALL_ENVELOPE_FORMATS:
                base_id = f"alpaca-{source_index:06d}-{format_name}"
                attack_type = rng.choice(attacked_types)
                variants = (
                    (
                        attack_type,
                        AugmentationConfig(
                            envelope_prob=1.0,
                            wrapper_prob=0.30,
                            wrapper_pool="test",
                            envelope_format=format_name,
                        ),
                    ),
                    (
                        "Clean",
                        AugmentationConfig(
                            envelope_prob=1.0,
                            hard_negative_prob=0.30,
                            envelope_format=format_name,
                        ),
                    ),
                )
                for variant_type, config in variants:
                    record = make_variant(
                        base_id=base_id,
                        split="format_test",
                        target=target,
                        injected_source=(
                            target
                            if variant_type == "Clean"
                            else rng.choice(injection_sources)
                        ),
                        attack_type=variant_type,
                        rng=rng,
                        test_templates=True,
                        cut_attacked=False,
                        augmentation=config,
                    )
                    if record["id"] in seen_ids:
                        raise ValueError(f"duplicate record id: {record['id']}")
                    seen_ids.add(record["id"])
                    _write_jsonl_record(handle, record)
                    count += 1
    return count


def _write_sep_split(
    path: Path,
    *,
    records: list[dict[str, Any]],
    rng: random.Random,
    seen_ids: set[str],
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for source_index, target in enumerate(records):
            if str(target.get("input", "")) == "":
                raise ValueError(f"SEP record {source_index} has an empty input")
            provided_injection = str(target.get("injection", ""))
            if not provided_injection:
                raise ValueError(f"SEP record {source_index} has an empty injection")
            base_id = f"sep-{source_index:06d}"
            for attack_type in ATTACK_TYPES:
                record = make_variant(
                    base_id=base_id,
                    split="sep_test",
                    target=target,
                    injected_source=target,
                    attack_type=attack_type,
                    rng=rng,
                    test_templates=True,
                    cut_attacked=False,
                    provided_injection=(
                        None if attack_type == "Clean" else provided_injection
                    ),
                )
                if record["id"] in seen_ids:
                    raise ValueError(f"duplicate record id: {record['id']}")
                seen_ids.add(record["id"])
                _write_jsonl_record(handle, record)
                count += 1
    return count


def _publish_directory(temp_dir: Path, output_dir: Path) -> None:
    backup_dir = None
    if output_dir.exists():
        backup_dir = output_dir.with_name(
            f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        )
        os.replace(output_dir, backup_dir)
    try:
        os.replace(temp_dir, output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def generate_datasets(
    *,
    alpaca_path: Path,
    sep_path: Path,
    output_dir: Path,
    validation_size: int = 1000,
    seed: int = 42,
    skip_token_stats: bool = False,
    tokenizer_file: Path | None = None,
    augmentation: AugmentationConfig | None = None,
    format_test_size: int = 0,
) -> dict[str, Any]:
    """Generate all JSONL files and publish them after complete validation."""
    alpaca_path = Path(alpaca_path)
    sep_path = Path(sep_path)
    output_dir = Path(output_dir)

    alpaca_all = load_json_array(alpaca_path)
    alpaca_eligible = select_nonempty_inputs(alpaca_all)
    train, validation = split_alpaca(
        alpaca_eligible,
        validation_size=validation_size,
        seed=seed,
    )
    sep_records = load_json_array(sep_path)
    if format_test_size < 0 or format_test_size > len(validation):
        raise ValueError(
            "format_test_size must be between 0 and the validation source count"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        rng = random.Random(seed)
        seen_ids: set[str] = set()
        counts = {
            "train": _write_alpaca_split(
                temp_dir / "train.jsonl",
                split="train",
                records=train,
                injection_sources=alpaca_all,
                rng=rng,
                seen_ids=seen_ids,
                augmentation=augmentation,
            ),
            "validation": _write_alpaca_split(
                temp_dir / "validation.jsonl",
                split="validation",
                records=validation,
                injection_sources=alpaca_all,
                rng=rng,
                seen_ids=seen_ids,
                augmentation=augmentation,
            ),
            "sep_test": _write_sep_split(
                temp_dir / "sep_test.jsonl",
                records=sep_records,
                rng=rng,
                seen_ids=seen_ids,
            ),
        }
        if format_test_size:
            counts["format_test"] = _write_format_test_split(
                temp_dir / "format_test.jsonl",
                records=validation[:format_test_size],
                injection_sources=alpaca_all,
                rng=rng,
                seen_ids=seen_ids,
            )

        manifest = {
            "seed": seed,
            "validation_size": validation_size,
            "attack_types": list(ATTACK_TYPES),
            "position_distribution": {
                "append": 0.2,
                "prepend": 0.2,
                "middle": 0.6,
            },
            "cut_benign": {
                "empty": 0.15,
                "two_thirds": 0.10,
                "half": 0.10,
                "none": 0.65,
            },
            "sources": {
                "alpaca": _source_identity(alpaca_path, len(alpaca_all)),
                "sep": _source_identity(sep_path, len(sep_records)),
            },
            "augmentation": (
                dataclasses.asdict(augmentation) if augmentation is not None else None
            ),
            "format_test_size": format_test_size,
            "eligible_alpaca_records": len(alpaca_eligible),
            "alpaca_injection_source_pool": {
                "policy": "all_source_records",
                "records": len(alpaca_all),
            },
            "splits": {
                "train": [index for index, _ in train],
                "validation": [index for index, _ in validation],
                "sep_test": list(range(len(sep_records))),
            },
            "counts": counts,
        }
        _write_json(temp_dir / "split_manifest.json", manifest)

        if not skip_token_stats:
            from minspan.data_stats import write_dataset_stats

            write_dataset_stats(
                data_dir=temp_dir,
                output_path=temp_dir / "stats.json",
                tokenizer_file=tokenizer_file,
            )

        _publish_directory(temp_dir, output_dir)
        return {"counts": counts, "manifest": manifest}
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate untruncated DataFilter tagger datasets."
    )
    parser.add_argument(
        "--alpaca",
        type=Path,
        default=Path("data/alpaca_data_cleaned.json"),
    )
    parser.add_argument(
        "--sep",
        type=Path,
        default=Path("data/SEP_1000_keyword.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-file", type=Path)
    parser.add_argument(
        "--skip-token-stats",
        action="store_true",
        help="Skip tokenizer loading; intended for tests and debugging only.",
    )
    parser.add_argument(
        "--envelope-prob",
        type=float,
        default=0.0,
        help="Probability of wrapping a train/validation record in a "
        "structured envelope (Phase 2 default: 0.25)",
    )
    parser.add_argument(
        "--wrapper-prob",
        type=float,
        default=0.0,
        help="Probability of wrapping a Naive/Ignore injection in a marker "
        "template (Phase 2 default: 0.30)",
    )
    parser.add_argument(
        "--hard-negative-prob",
        type=float,
        default=0.0,
        help="Probability of inserting a benign business imperative into a "
        "Clean record (Phase 2 default: 0.30)",
    )
    parser.add_argument(
        "--format-test-size",
        type=int,
        default=0,
        help="Number of validation sources used to build the format_test "
        "split (0 disables it)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    augmentation = None
    if args.envelope_prob or args.wrapper_prob or args.hard_negative_prob:
        augmentation = AugmentationConfig(
            envelope_prob=args.envelope_prob,
            wrapper_prob=args.wrapper_prob,
            hard_negative_prob=args.hard_negative_prob,
        )
    result = generate_datasets(
        alpaca_path=args.alpaca,
        sep_path=args.sep,
        output_dir=args.output_dir,
        validation_size=args.validation_size,
        seed=args.seed,
        skip_token_stats=args.skip_token_stats,
        tokenizer_file=args.tokenizer_file,
        augmentation=augmentation,
        format_test_size=args.format_test_size,
    )
    total = sum(result["counts"].values())
    print(f"Generated {total} records in {args.output_dir}")


if __name__ == "__main__":
    main()
