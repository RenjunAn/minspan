#!/usr/bin/env python3
"""Collect PIArena results from raw evaluation outputs into the paper's result tables.

Inputs
  benchmarks/piarena/results/evaluation_results/modernbert_tagger_p3_full/
      per-dataset per-attack record JSONs (utility per record, ASR flag per record)
  benchmarks/piarena/results/agent_evaluations/injecagent/modernbert_tagger_p3_full/
      InjecAgent evaluation (utility = valid_rate, ASR = asr_valid_total, base/enhanced)
  results/agentdyn_main.csv
      for the AgentDojo (static suites) and AgentDyn (dynamic suites) composite rows

Outputs
  results/piarena_main.csv           per-dataset per-attack utility/ASR (full precision)
  results/piarena_table.csv          macro-averaged leaderboard-convention table (paper tab:piarena)
  figures/src/piarena_per_dataset.csv  Direct-attack per-dataset bars for the paper figure

The PIArena leaderboard reports integer-rounded per-dataset numbers; the
published baselines exist only in that form.  results/piarena_table.csv
therefore rounds MinSpan's per-dataset values the same way before macro-
averaging, so its rows are directly comparable with the baselines.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "benchmarks" / "piarena" / "website" / "data" / "results.json"
EVAL_DIR = ROOT / "benchmarks" / "piarena" / "results" / "evaluation_results" / "modernbert_tagger_p3_full"
INJECAGENT = (
    ROOT / "benchmarks" / "piarena" / "results" / "agent_evaluations" / "injecagent"
    / "modernbert_tagger_p3_full" / "Qwen3-4B-Instruct-2507-modernbert_tagger-seed42.json"
)
AGENTDYN_MAIN = ROOT / "results" / "agentdyn_main.csv"
RESULTS = ROOT / "results"
FIGURE_SRC = ROOT / "figures" / "src"

LLM = "qwen3-4b"
DEFENSE = "minspan"

# evaluation file dataset id -> canonical short name
DATASETS = {
    "squad_v2": "squad_v2",
    "dolly_closed_qa": "dolly_qa",
    "dolly_information_extraction": "dolly_ie",
    "dolly_summarization": "dolly_summ",
    "nq_rag": "nq_rag",
    "msmarco_rag": "msmarco_rag",
    "hotpotqa_rag": "hotpotqa_rag",
    "hotpotqa_long": "hotpotqa_long",
    "qasper_long": "qasper",
    "gov_report_long": "govreport",
    "multi_news_long": "multinews",
    "passage_retrieval_en_long": "passage_retrieval",
    "lcc_long": "lcc",
    "nq_rag_knowledge_corruption": ("nq_rag", "knowledge_corruption"),
}
ATTACKS = {"none": "none", "direct": "direct", "combined": "combined", "strategy_search": "strategy"}

# paper figure order and display names (short-text/RAG group first, long-text group second)
FIGURE_ORDER = [
    ("squad_v2", "SQuAD v2"),
    ("dolly_qa", "Dolly QA"),
    ("dolly_ie", "Dolly IE"),
    ("dolly_summ", "Dolly Summ."),
    ("nq_rag", "NQ"),
    ("msmarco_rag", "MS-MARCO"),
    ("hotpotqa_rag", "HotpotQA RAG"),
    ("hotpotqa_long", "HotpotQA Long"),
    ("qasper", "Qasper"),
    ("govreport", "GovReport"),
    ("multinews", "MultiNews"),
    ("passage_retrieval", "Passage Retr."),
    ("lcc", "LCC"),
]


def read_eval_file(path: Path) -> tuple[float, float, int]:
    records = json.loads(path.read_text())
    utility = statistics.fmean(float(r["utility"]) for r in records.values()) * 100
    asr = statistics.fmean(float(r["asr"]) for r in records.values()) * 100
    return utility, asr, len(records)


def main() -> None:
    rows = []  # (dataset, attack, utility, asr, n)
    for path in sorted(EVAL_DIR.glob("*.json")):
        parts = path.stem.split("-Qwen-Qwen3-4B-Instruct-2507-")
        if len(parts) != 2:
            continue
        dataset_id = parts[0]
        attack_id = parts[1].rsplit("-modernbert_tagger-", 1)[0]
        if dataset_id not in DATASETS:
            raise SystemExit(f"unknown dataset in {path.name}")
        mapped = DATASETS[dataset_id]
        if isinstance(mapped, tuple):
            dataset, attack = mapped
        else:
            dataset, attack = mapped, ATTACKS.get(attack_id)
        if attack is None:
            raise SystemExit(f"unknown attack in {path.name}")
        utility, asr, n = read_eval_file(path)
        rows.append((dataset, attack, utility, asr, n))

    # InjecAgent: utility = valid_rate, ASR = mean of base/enhanced asr_valid_total
    inj = json.loads(INJECAGENT.read_text())
    settings = [inj[k]["metrics"] for k in ("base", "enhanced") if k in inj]
    rows.append(
        (
            "injecagent",
            "default",
            statistics.fmean(m["valid_rate"] for m in settings),
            statistics.fmean(m["asr_valid_total"] for m in settings),
            sum(m["total_cases"] for m in settings),
        )
    )

    # AgentDojo (static suites) / AgentDyn (dynamic suites) composite rows,
    # derived from the AgentDyn harness run (see collect_agentdyn.py)
    with open(AGENTDYN_MAIN) as fh:
        suite_rows = [r for r in csv.DictReader(fh) if r["scope"] == "suite"]
    static = [r for r in suite_rows if r["suite"] in ("banking", "slack", "travel", "workspace")]
    dynamic = [r for r in suite_rows if r["suite"] in ("shopping", "github", "dailylife")]
    for name, group in (("agentdojo", static), ("agentdyn", dynamic)):
        rows.append(
            (
                name,
                "default",
                statistics.fmean(float(r["attack_utility_pct"]) for r in group),
                statistics.fmean(float(r["asr_pct"]) for r in group),
                sum(int(r["attack_n"]) for r in group),
            )
        )

    rows.sort()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "piarena_main.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["defense", "dataset", "attack", "llm", "utility_pct", "asr_pct", "n"])
        for dataset, attack, utility, asr, n in rows:
            writer.writerow([DEFENSE, dataset, attack, LLM, f"{utility:.10g}", f"{asr:.10g}", n])
    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} rows)")

    direct = {(d): (u, a) for d, attack, u, a, _ in rows if attack == "direct"}
    fig = FIGURE_SRC / "piarena_per_dataset.csv"
    with open(fig, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "direct_utility", "direct_asr"])
        for key, display in FIGURE_ORDER:
            utility, asr = direct[key]
            writer.writerow([display, f"{utility:.10g}", f"{asr:.10g}"])
    print(f"wrote {fig.relative_to(ROOT)} ({len(FIGURE_ORDER)} rows)")

    mean_u = statistics.fmean(direct[k][0] for k, _ in FIGURE_ORDER)
    mean_a = statistics.fmean(direct[k][1] for k, _ in FIGURE_ORDER)
    print(f"13-dataset Direct mean (full precision): utility {mean_u:.2f}, ASR {mean_a:.2f}")

    # ---- leaderboard-convention macro table with the published baselines
    text_datasets = {k for k, _ in FIGURE_ORDER}
    cells: dict[tuple[str, str, str], tuple[float, float]] = {}
    for r in json.loads(BASELINES.read_text())["results"]:
        if r["llm"] != LLM or r["dataset"] not in text_datasets:
            continue
        if r["defense"].startswith("modernbert_tagger"):
            continue  # historical aliases of our own defense; the minspan row below is the source of truth
        if r.get("utility") is None or r.get("asr") is None:
            continue
        cells[(r["defense"], r["dataset"], r["attack"])] = (float(r["utility"]), float(r["asr"]))
    for dataset, attack, utility, asr, _ in rows:
        if dataset in text_datasets and attack in ("none", "direct", "combined"):
            cells[(DEFENSE, dataset, attack)] = (float(round(utility)), float(round(asr)))

    defenses = sorted({d for d, _, _ in cells}, key=lambda d: (d != "none", d != DEFENSE, d))
    table = RESULTS / "piarena_table.csv"
    with open(table, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["defense", "none_utility", "direct_utility", "direct_asr", "combined_utility", "combined_asr"]
        )
        for defense in defenses:
            out_row = [defense]
            for attack, want in (("none", ("u",)), ("direct", ("u", "a")), ("combined", ("u", "a"))):
                vals = [cells.get((defense, ds, attack)) for ds, _ in FIGURE_ORDER]
                if any(v is None for v in vals):
                    out_row.extend([""] * len(want))
                    continue
                if "u" in want:
                    out_row.append(f"{statistics.fmean(v[0] for v in vals):.2f}")
                if "a" in want:
                    out_row.append(f"{statistics.fmean(v[1] for v in vals):.2f}")
            writer.writerow(out_row)
    print(f"wrote {table.relative_to(ROOT)} ({len(defenses)} defenses)")


if __name__ == "__main__":
    main()
