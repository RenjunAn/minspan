import json
from collections import defaultdict
from pathlib import Path


def test_pi_detector_pairs_are_exact_counterfactuals():
    for split in ("train", "val", "test"):
        records = _read_jsonl(Path("data") / "pi_detector" / f"{split}.jsonl")
        groups = defaultdict(list)
        for record in records:
            groups[(record["tool_name"], record["user_instruction"])].append(record)

        for key, group in groups.items():
            benign = [record for record in group if not record["is_injection"]]
            injected = [record for record in group if record["is_injection"]]
            assert len(benign) == 1, f"{split} {key!r} should have exactly one benign sample"
            assert len(injected) == 1, f"{split} {key!r} should have exactly one injected sample"

            injection_text = injected[0]["injection_text"]
            assert injection_text
            assert injection_text in injected[0]["tool_output"], (
                f"{split} {key!r} injected sample {injected[0]['id']} must contain its exact injection_text"
            )
            assert injected[0]["tool_output"].replace(injection_text, "") == benign[0]["tool_output"], (
                f"{split} {key!r} injected sample {injected[0]['id']} must reduce to benign sample "
                f"{benign[0]['id']} when injection_text is removed"
            )


def test_pi_detector_splits_do_not_overlap():
    by_split = {
        split: _read_jsonl(Path("data") / "pi_detector" / f"{split}.jsonl") for split in ("train", "val", "test")
    }

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_instructions = {record["user_instruction"] for record in by_split[left]}
        right_instructions = {record["user_instruction"] for record in by_split[right]}
        assert not left_instructions & right_instructions

        left_outputs = {record["tool_output"] for record in by_split[left]}
        right_outputs = {record["tool_output"] for record in by_split[right]}
        assert not left_outputs & right_outputs


def test_pi_detector_test_split_is_balanced_and_agentdyn_sourced():
    records = _read_jsonl(Path("data") / "pi_detector" / "test.jsonl")

    assert len(records) == 96
    assert sum(1 for record in records if record["is_injection"]) == 48
    assert sum(1 for record in records if not record["is_injection"]) == 48
    assert {record["tool_name"] for record in records} == {
        "read_calendar",
        "read_document",
        "read_inbox",
        "read_issue_thread",
        "search_products",
        "search_web",
    }
    assert all(record["id"].startswith("test_") for record in records)
    assert all(record.get("source_suite") for record in records)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
