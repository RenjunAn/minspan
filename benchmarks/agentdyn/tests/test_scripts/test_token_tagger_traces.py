import json
import os

from agentdojo.scripts.token_tagger_traces import summarize_tagger_traces


def test_summarize_tagger_traces_filters_run_and_counts_failures(tmp_path):
    since_ns = 2_000_000_000
    old_trace = tmp_path / "old.json"
    current_trace = tmp_path / "current.json"
    other_defense_trace = tmp_path / "other.json"

    old_trace.write_text(
        json.dumps({"tagger_defense_events": [{"defense": "modernbert_tagger", "success": False}]}),
        encoding="utf-8",
    )
    current_trace.write_text(
        json.dumps(
            {
                "tagger_defense_events": [
                    {"defense": "modernbert_tagger", "success": True},
                    {"defense": "modernbert_tagger", "success": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    other_defense_trace.write_text(
        json.dumps({"tagger_defense_events": [{"defense": "datafilter_bidir_tagger", "success": False}]}),
        encoding="utf-8",
    )
    os.utime(old_trace, ns=(1_000_000_000, 1_000_000_000))
    os.utime(current_trace, ns=(3_000_000_000, 3_000_000_000))
    os.utime(
        other_defense_trace,
        ns=(3_000_000_000, 3_000_000_000),
    )

    summary = summarize_tagger_traces(
        tmp_path,
        defense="modernbert_tagger",
        since_ns=since_ns,
    )

    assert summary == {
        "trace_files": 1,
        "events": 2,
        "failures": 1,
    }


def test_summarize_tagger_traces_ignores_invalid_json(tmp_path):
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")

    summary = summarize_tagger_traces(
        tmp_path,
        defense="modernbert_tagger",
        since_ns=0,
    )

    assert summary == {
        "trace_files": 0,
        "events": 0,
        "failures": 0,
    }
