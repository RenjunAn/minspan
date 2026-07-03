#!/usr/bin/env python3
"""Shared full leaderboard runner for one PIArena defense."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, NamedTuple


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DEFENSES = ("deepseek_pisanitizer", "modernbert_tagger")
DEFAULT_BACKEND_LLM = "Qwen/Qwen3-4B-Instruct-2507"


class LeaderboardBenchmark(NamedTuple):
    dataset: str
    runner_dataset: str
    attack: str
    runner_attack: str
    entrypoint: str = "main.py"


class AgentLeaderboardBenchmark(NamedTuple):
    dataset: str
    entrypoint: str
    attack: str = "default"
    suites: tuple[str, ...] = ()


DATASET_TO_RUNNER = {
    "squad_v2": "squad_v2",
    "dolly_qa": "dolly_closed_qa",
    "dolly_ie": "dolly_information_extraction",
    "dolly_summ": "dolly_summarization",
    "nq_rag": "nq_rag",
    "msmarco_rag": "msmarco_rag",
    "hotpotqa_rag": "hotpotqa_rag",
    "hotpotqa_long": "hotpotqa_long",
    "qasper": "qasper_long",
    "govreport": "gov_report_long",
    "multinews": "multi_news_long",
    "passage_retrieval": "passage_retrieval_en_long",
    "lcc": "lcc_long",
}

CORE_ATTACKS = ("none", "direct", "combined", "strategy")
CORE_DATASETS = tuple(DATASET_TO_RUNNER)
AVAILABLE_STANDARD_RUNNER_DATASETS = frozenset(
    {
        *DATASET_TO_RUNNER.values(),
        "nq_rag_knowledge_corruption",
        "msmarco_rag_knowledge_corruption",
        "hotpotqa_rag_knowledge_corruption",
    }
)
STANDARD_RESULT_COUNTS = {
    "dolly_closed_qa": 200,
    "dolly_information_extraction": 200,
    "dolly_summarization": 200,
    "gov_report_long": 100,
    "hotpotqa_long": 100,
    "hotpotqa_rag": 100,
    "hotpotqa_rag_knowledge_corruption": 100,
    "lcc_long": 100,
    "msmarco_rag": 100,
    "msmarco_rag_knowledge_corruption": 100,
    "multi_news_long": 100,
    "nq_rag": 100,
    "nq_rag_knowledge_corruption": 100,
    "passage_retrieval_en_long": 100,
    "qasper_long": 100,
    "squad_v2": 200,
}

LEADERBOARD_STANDARD_MATRIX = (
    *(
        LeaderboardBenchmark(
            dataset=dataset,
            runner_dataset=DATASET_TO_RUNNER[dataset],
            attack=attack,
            runner_attack="strategy_search" if attack == "strategy" else attack,
            entrypoint="main_search.py" if attack == "strategy" else "main.py",
        )
        for dataset in CORE_DATASETS
        for attack in CORE_ATTACKS
    ),
    LeaderboardBenchmark(
        dataset="nq_rag",
        runner_dataset="nq_rag_knowledge_corruption",
        attack="knowledge_corruption",
        runner_attack="none",
    ),
    LeaderboardBenchmark(
        dataset="multinews",
        runner_dataset="multi_news_long",
        attack="gcg",
        runner_attack="nanogcg",
        entrypoint="main_search.py",
    ),
    LeaderboardBenchmark(
        dataset="opi",
        runner_dataset="open_prompt_injection",
        attack="default",
        runner_attack="combined",
    ),
    LeaderboardBenchmark(
        dataset="sep",
        runner_dataset="sep",
        attack="default",
        runner_attack="combined",
    ),
)

LEADERBOARD_AGENT_MATRIX = (
    AgentLeaderboardBenchmark("injecagent", "main_injecagent.py"),
    AgentLeaderboardBenchmark("agentdojo", "main_agentdojo.py", suites=("workspace", "slack", "travel", "banking")),
    AgentLeaderboardBenchmark("agentdyn", "main_agentdojo.py", suites=("shopping", "github", "dailylife")),
)


def parse_args(
    argv: list[str] | None = None,
    *,
    defense: str,
    default_config: str,
    default_name: str,
    script_name: str,
) -> argparse.Namespace:
    if defense not in SUPPORTED_DEFENSES:
        raise ValueError(f"Unsupported defense: {defense}")

    parser = argparse.ArgumentParser(
        description=f"Run the non-WASP PIArena leaderboard matrix for {defense}."
    )
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--backend-llm", default=DEFAULT_BACKEND_LLM)
    parser.add_argument("--attacker-llm", default=DEFAULT_BACKEND_LLM)
    parser.add_argument("--name", default=default_name)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--only-dataset", action="append", help="Leaderboard dataset key to run.")
    parser.add_argument("--only-attack", action="append", help="Leaderboard attack key to run.")
    parser.add_argument("--skip-standard", action="store_true")
    parser.add_argument("--skip-agents", action="store_true")
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Only prepare commands whose result files are missing or incomplete.",
    )
    parser.add_argument(
        "--local-datasets-only",
        action="store_true",
        help="Skip standard commands whose runner datasets are not in the bundled PIArena split set.",
    )
    parser.add_argument(
        "--force-agentdojo",
        action="store_true",
        help="Pass --force-rerun to AgentDojo/AgentDyn commands and run them even under --pending-only.",
    )
    parser.add_argument("--no-export", action="store_true", help="Do not write leaderboard JSON after execution.")
    parser.add_argument(
        "--leaderboard-output",
        default=None,
        help="Output JSON for leaderboard-compatible entries.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args(argv)
    args.defense = defense
    args.script_name = script_name
    return args


def selected_entry(args: argparse.Namespace, dataset: str, attack: str) -> bool:
    if args.only_dataset and dataset not in set(args.only_dataset):
        return False
    if args.only_attack and attack not in set(args.only_attack):
        return False
    return True


def build_standard_command(args: argparse.Namespace, entry: LeaderboardBenchmark) -> list[str]:
    command = [
        sys.executable,
        entry.entrypoint,
        "--config",
        args.config,
        "--dataset",
        entry.runner_dataset,
        "--backend_llm",
        args.backend_llm,
        "--attack",
        entry.runner_attack,
        "--defense",
        args.defense,
        "--name",
        args.name,
        "--seed",
        str(args.seed),
    ]
    if entry.entrypoint == "main_search.py":
        command.extend(
            [
                "--attacker_llm",
                args.attacker_llm,
                "--batch_size",
                str(args.batch_size),
            ]
        )
    return command


def build_agent_command(args: argparse.Namespace, entry: AgentLeaderboardBenchmark) -> list[str]:
    command = [
        sys.executable,
        entry.entrypoint,
        "--config",
        args.config,
        "--model",
        args.backend_llm,
        "--defense",
        args.defense,
        "--name",
        args.name,
    ]
    if entry.entrypoint == "main_agentdojo.py":
        command.extend(["--attack", "important_instructions"])
        if entry.suites:
            command.append("--suite")
            command.extend(entry.suites)
        if args.force_agentdojo:
            command.append("--force-rerun")
    else:
        command.extend(["--seed", str(args.seed)])
    return command


def result_row_count(raw: Any) -> int | None:
    if isinstance(raw, dict):
        return len(raw)
    if isinstance(raw, list):
        return len(raw)
    return None


def load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def standard_result_complete(args: argparse.Namespace, entry: LeaderboardBenchmark) -> bool:
    raw = load_json_if_exists(result_file_for(args, entry, args.defense))
    if raw is None:
        return False

    row_count = result_row_count(raw)
    if row_count is None:
        return False

    expected_count = STANDARD_RESULT_COUNTS.get(entry.runner_dataset)
    if expected_count is None:
        return row_count > 0
    return row_count >= expected_count


def agent_result_complete(args: argparse.Namespace, entry: AgentLeaderboardBenchmark) -> bool:
    result, _ = load_agent_entry(args, entry, args.defense)
    return result is not None


def should_run_standard_entry(args: argparse.Namespace, entry: LeaderboardBenchmark) -> bool:
    if not selected_entry(args, entry.dataset, entry.attack):
        return False
    if args.local_datasets_only and entry.runner_dataset not in AVAILABLE_STANDARD_RUNNER_DATASETS:
        return False
    if args.pending_only and standard_result_complete(args, entry):
        return False
    return True


def should_run_agent_entry(args: argparse.Namespace, entry: AgentLeaderboardBenchmark) -> bool:
    if not selected_entry(args, entry.dataset, entry.attack):
        return False
    if args.force_agentdojo and entry.entrypoint == "main_agentdojo.py":
        return True
    if args.pending_only and agent_result_complete(args, entry):
        return False
    return True


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []

    if not args.skip_standard:
        for entry in LEADERBOARD_STANDARD_MATRIX:
            if should_run_standard_entry(args, entry):
                commands.append(build_standard_command(args, entry))

    if not args.skip_agents:
        for entry in LEADERBOARD_AGENT_MATRIX:
            if should_run_agent_entry(args, entry):
                commands.append(build_agent_command(args, entry))

    return commands


def run_commands(commands: list[list[str]], dry_run: bool) -> None:
    for command in commands:
        rendered = " ".join(shlex.quote(part) for part in command)
        print(rendered)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def model_slug(model: str) -> str:
    lowered = model.lower()
    if "qwen3-4b" in lowered:
        return "qwen3-4b"
    return model.split("/")[-1].replace("_", "-").lower()


def metric_to_percent(values: list[Any]) -> int | None:
    numeric = [float(value) for value in values if isinstance(value, (bool, int, float))]
    if not numeric:
        return None
    score = mean(numeric)
    if all(0 <= value <= 1 for value in numeric):
        score *= 100
    return round(score)


def result_file_for(args: argparse.Namespace, entry: LeaderboardBenchmark, defense: str) -> Path:
    llm_name = args.backend_llm.replace("/", "-")
    return (
        ROOT
        / "results"
        / "evaluation_results"
        / args.name
        / f"{entry.runner_dataset}-{llm_name}-{entry.runner_attack}-{defense}-{args.seed}.json"
    )


def load_standard_entry(
    args: argparse.Namespace,
    entry: LeaderboardBenchmark,
    defense: str,
) -> tuple[dict[str, Any] | None, str | None]:
    path = result_file_for(args, entry, defense)
    if not path.exists():
        return None, str(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = list(raw.values()) if isinstance(raw, dict) else raw
    result = {
        "dataset": entry.dataset,
        "attack": entry.attack,
        "defense": defense,
        "llm": model_slug(args.backend_llm),
        "utility": metric_to_percent([row.get("utility") for row in rows if isinstance(row, dict)]),
        "asr": metric_to_percent([row.get("asr") for row in rows if isinstance(row, dict)]),
    }
    return result, None


def load_agentdojo_entry(
    args: argparse.Namespace,
    entry: AgentLeaderboardBenchmark,
    defense: str,
) -> tuple[dict[str, Any] | None, str | None]:
    logdirs = agentdojo_logdirs(args)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for suite in entry.suites:
        suite_rows: list[dict[str, Any]] = []
        for logdir in logdirs:
            current_logdir_rows: list[dict[str, Any]] = []
            for path in logdir.glob(f"*piarena-{defense}/{suite}/*/important_instructions/*.json"):
                with path.open("r", encoding="utf-8") as f:
                    current_logdir_rows.append(json.load(f))
            if current_logdir_rows:
                suite_rows.extend(current_logdir_rows)
                break
        if not suite_rows:
            searched = [
                str(logdir / f"*piarena-{defense}" / suite / "*" / "important_instructions" / "*.json")
                for logdir in logdirs
            ]
            missing.append(" or ".join(searched))
        rows.extend(suite_rows)

    if missing:
        return None, "; ".join(missing)

    result = {
        "dataset": entry.dataset,
        "attack": entry.attack,
        "defense": defense,
        "llm": model_slug(args.backend_llm),
        "utility": metric_to_percent([row.get("utility") for row in rows]),
        "asr": metric_to_percent([row.get("security") for row in rows]),
    }
    return result, None


def agentdojo_logdirs(args: argparse.Namespace) -> list[Path]:
    primary = ROOT / "results" / "agent_evaluations" / "agentdojo" / args.name
    legacy = ROOT / args.name
    if legacy == primary:
        return [primary]
    return [primary, legacy]


def load_injecagent_entry(
    args: argparse.Namespace,
    entry: AgentLeaderboardBenchmark,
    defense: str,
) -> tuple[dict[str, Any] | None, str | None]:
    model_short = args.backend_llm.split("/")[-1]
    path = (
        ROOT
        / "results"
        / "agent_evaluations"
        / "injecagent"
        / args.name
        / f"{model_short}-{defense}-seed{args.seed}.json"
    )
    if not path.exists():
        return None, str(path)

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    metrics = [
        value.get("metrics", {})
        for value in raw.values()
        if isinstance(value, dict) and isinstance(value.get("metrics"), dict)
    ]
    result = {
        "dataset": entry.dataset,
        "attack": entry.attack,
        "defense": defense,
        "llm": model_slug(args.backend_llm),
        "utility": metric_to_percent([metric.get("valid_rate") for metric in metrics]),
        "asr": metric_to_percent([metric.get("asr_all_total") for metric in metrics]),
    }
    return result, None


def load_agent_entry(
    args: argparse.Namespace,
    entry: AgentLeaderboardBenchmark,
    defense: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if entry.entrypoint == "main_agentdojo.py":
        return load_agentdojo_entry(args, entry, defense)
    return load_injecagent_entry(args, entry, defense)


def default_leaderboard_output(args: argparse.Namespace) -> Path:
    return ROOT / "results" / "leaderboard_entries" / f"{args.name}.json"


def export_leaderboard_results(args: argparse.Namespace) -> Path:
    output_path = Path(args.leaderboard_output) if args.leaderboard_output else default_leaderboard_output(args)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    missing: list[str] = []
    if not args.skip_standard:
        for entry in LEADERBOARD_STANDARD_MATRIX:
            if not selected_entry(args, entry.dataset, entry.attack):
                continue
            result, missing_path = load_standard_entry(args, entry, args.defense)
            if result is None:
                if missing_path is not None:
                    missing.append(missing_path)
                continue
            results.append(result)

    if not args.skip_agents:
        for entry in LEADERBOARD_AGENT_MATRIX:
            if not selected_entry(args, entry.dataset, entry.attack):
                continue
            result, missing_path = load_agent_entry(args, entry, args.defense)
            if result is None:
                if missing_path is not None:
                    missing.append(missing_path)
                continue
            results.append(result)

    payload = {
        "_schema": {
            "description": f"Leaderboard-compatible entries generated by scripts/{args.script_name}.",
            "note": "InjecAgent utility is derived from valid_rate; AgentDojo and AgentDyn ASR is the average injection-success flag stored as security in AgentDojo logs.",
        },
        "results": results,
        "missing_result_files": missing,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote leaderboard entries to {output_path}")
    if missing:
        print(f"Missing {len(missing)} result files; rerun without --dry-run or with a narrower filter.")
    return output_path


def main(
    argv: list[str] | None = None,
    *,
    defense: str,
    default_config: str,
    default_name: str,
    script_name: str,
) -> int:
    args = parse_args(
        argv,
        defense=defense,
        default_config=default_config,
        default_name=default_name,
        script_name=script_name,
    )
    commands = build_commands(args)
    run_commands(commands, args.dry_run)
    print(f"Prepared {len(commands)} leaderboard benchmark commands for {args.defense}.")
    if not args.no_export and not args.dry_run:
        export_leaderboard_results(args)
    elif args.dry_run:
        print("Dry run only; no leaderboard JSON exported.")
    return 0
