#!/usr/bin/env python3
"""Train/test contamination audit.

Checks the MinSpan training set against every evaluation source used in the
paper, on two axes:

  documents   does any test-side document (tool output / RAG context) share
              verbatim content with a training document?
              -> normalized exact match + hashed word-8-gram overlap
  injections  does any test-side injection payload appear in training?
              -> normalized exact match + 8-gram overlap

Test sources
  local       data/p3_*_test.jsonl, sep/format/nemo test splits (documents and
              injections; also base_id collisions with train)
  piarena     benchmarks/piarena/results/evaluation_results/... (context,
              injected_task per record)
  agentdyn    original tool outputs and injection payloads recorded in the
              MinSpan AgentDyn run

Output: results/contamination_audit.csv + a human-readable summary on stdout.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PIARENA_EVAL = ROOT / "benchmarks" / "piarena" / "results" / "evaluation_results" / "modernbert_tagger_p3_full"
AGENTDYN_RUN = ROOT / "benchmarks" / "agentdyn" / "runs" / "deepseek-v4-flash-modernbert_tagger-89810d4f66b9"

N = 8  # words per shingle

LOCAL_SPLITS = (
    "p3_direct_test",
    "p3_strategy_test",
    "p3_clean_hard_negative_test",
    "sep_test",
    "format_test",
    "nemo_test",
    "nemo_test_clean",
    "nemo_test_env",
    "nemo_test_env_clean",
)


def normalize(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def shingles(text: str) -> set[int]:
    words = normalize(text).split()
    return {hash(" ".join(words[i : i + N])) for i in range(len(words) - N + 1)}


def overlap_fraction(text: str, train_grams: set[int]) -> float | None:
    grams = shingles(text)
    if not grams:
        return None
    return sum(g in train_grams for g in grams) / len(grams)


def audit_texts(texts: list[str], train_exact: set[str], train_grams: set[int]) -> dict:
    exact = 0
    fractions = []
    for text in texts:
        if normalize(text) in train_exact:
            exact += 1
        frac = overlap_fraction(text, train_grams)
        if frac is not None:
            fractions.append(frac)
    return {
        "n": len(texts),
        "exact_matches": exact,
        "docs_with_any_overlap": sum(f > 0 for f in fractions),
        "docs_over_50pct": sum(f >= 0.5 for f in fractions),
        "mean_overlap_pct": statistics.fmean(fractions) * 100 if fractions else 0.0,
        "max_overlap_pct": max(fractions) * 100 if fractions else 0.0,
    }


def main() -> None:
    # ---- training side
    train_docs: set[str] = set()
    train_injections: set[str] = set()
    train_base_ids: set[str] = set()
    with open(DATA / "train.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            train_docs.add(normalize(rec.get("original_data")))
            train_base_ids.add(rec.get("base_id"))
            injection = normalize(rec.get("injection"))
            if injection:
                train_injections.add(injection)
    train_docs.discard("")
    print(f"train: {len(train_base_ids)} base documents, {len(train_docs)} unique doc texts, "
          f"{len(train_injections)} unique injections")

    doc_grams: set[int] = set()
    for doc in train_docs:
        doc_grams.update(shingles(doc))
    inj_grams: set[int] = set()
    for inj in train_injections:
        inj_grams.update(shingles(inj))
    print(f"train: {len(doc_grams)} doc {N}-grams, {len(inj_grams)} injection {N}-grams")

    rows = []

    def record(source: str, kind: str, stats: dict, note: str = "") -> None:
        rows.append({"source": source, "kind": kind, **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in stats.items()}, "note": note})
        print(f"  {source:<34} {kind:<10} n={stats['n']:<6} exact={stats['exact_matches']:<5} "
              f"any-overlap={stats['docs_with_any_overlap']:<5} >=50%={stats['docs_over_50pct']:<5} "
              f"mean={stats['mean_overlap_pct']:.2f}% max={stats['max_overlap_pct']:.2f}%")

    # ---- local held-out splits
    print("\nlocal splits (documents = original_data, injections = injection):")
    for split in LOCAL_SPLITS:
        path = DATA / f"{split}.jsonl"
        if not path.exists():
            continue
        docs, injections, id_collisions = [], [], 0
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                docs.append(rec.get("original_data"))
                if normalize(rec.get("injection")):
                    injections.append(rec.get("injection"))
                if rec.get("base_id") in train_base_ids:
                    id_collisions += 1
        record(split, "documents", audit_texts(docs, train_docs, doc_grams),
               note=f"base_id collisions with train: {id_collisions}")
        if injections:
            record(split, "injections", audit_texts(injections, train_injections, inj_grams))

    # ---- PIArena (context + injected task per record, deduplicated)
    print("\nPIArena evaluation inputs:")
    contexts: set[str] = set()
    injected: set[str] = set()
    for path in sorted(PIARENA_EVAL.glob("*.json")):
        for rec in json.loads(path.read_text()).values():
            contexts.add(str(rec.get("context") or ""))
            if str(rec.get("injected_task") or "").strip():
                injected.add(str(rec["injected_task"]))
    record("piarena", "documents", audit_texts(sorted(contexts), train_docs, doc_grams))
    record("piarena", "injections", audit_texts(sorted(injected), train_injections, inj_grams))

    # ---- AgentDyn (original tool outputs + injection payloads from the MinSpan run)
    print("\nAgentDyn evaluation inputs:")
    outputs: set[str] = set()
    payloads: set[str] = set()
    for path in sorted(AGENTDYN_RUN.rglob("*.json")):
        result = json.loads(path.read_text())
        for event in result.get("tagger_defense_events") or []:
            text = str(event.get("original_tool_output") or "")
            if len(text.split()) >= N:  # skip trivially short outputs
                outputs.add(text)
        payloads.update(str(v) for v in (result.get("injections") or {}).values())
    record("agentdyn", "documents", audit_texts(sorted(outputs), train_docs, doc_grams))
    record("agentdyn", "injections", audit_texts(sorted(payloads), train_injections, inj_grams))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "contamination_audit.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
