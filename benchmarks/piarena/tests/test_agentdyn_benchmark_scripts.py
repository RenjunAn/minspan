import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_matrix_covers_non_wasp_leaderboard_benchmarks():
    smoke = load_script("scripts/run_modernbert_tagger_smoke.py")

    assert smoke.DEFENSE == "modernbert_tagger"
    dataset_keys = {entry.leaderboard_dataset for entry in smoke.STANDARD_SMOKE_MATRIX}
    assert "wasp" not in dataset_keys
    assert {"opi", "sep", "squad_v2", "nq_rag", "qasper"}.issubset(dataset_keys)

    agent_keys = {entry.leaderboard_dataset for entry in smoke.AGENT_SMOKE_MATRIX}
    assert agent_keys == {"injecagent", "agentdojo", "agentdyn"}


def test_smoke_commands_use_defense_configs_and_agent_attack_override(tmp_path):
    smoke = load_script("scripts/run_deepseek_pisanitizer_smoke.py")
    args = smoke.parse_args(
        [
            "--config", "configs/experiments/deepseek_pisanitizer.yaml",
            "--work-dir", str(tmp_path),
            "--dry-run",
        ]
    )

    commands = smoke.build_commands(args)
    rendered = [" ".join(command) for command in commands]
    assert any("--config configs/experiments/deepseek_pisanitizer.yaml" in command for command in rendered)
    assert all("--defense deepseek_pisanitizer" in command for command in rendered)
    assert any("main_agentdojo.py" in command and "--attack important_instructions" in command for command in rendered)
    assert all("wasp" not in command for command in rendered)


def test_full_matrix_matches_leaderboard_scope_without_wasp():
    full = load_script("scripts/run_deepseek_pisanitizer_full.py")

    assert full.DEFENSE == "deepseek_pisanitizer"
    leaderboard_keys = {(entry.dataset, entry.attack) for entry in full.LEADERBOARD_STANDARD_MATRIX}
    assert ("wasp", "default") not in leaderboard_keys
    assert ("opi", "default") in leaderboard_keys
    assert ("sep", "default") in leaderboard_keys
    assert ("multinews", "gcg") in leaderboard_keys
    assert ("nq_rag", "knowledge_corruption") in leaderboard_keys

    agent_keys = {entry.dataset for entry in full.LEADERBOARD_AGENT_MATRIX}
    assert agent_keys == {"injecagent", "agentdojo", "agentdyn"}


def test_full_script_can_build_dry_run_commands():
    full = load_script("scripts/run_modernbert_tagger_full.py")
    args = full.parse_args(
        [
            "--config", "configs/experiments/modernbert_tagger.yaml",
            "--dry-run",
        ]
    )

    commands = full.build_commands(args)
    rendered = [" ".join(command) for command in commands]
    assert any("main.py" in command and "--dataset squad_v2" in command for command in rendered)
    assert any("main_search.py" in command and "--attack strategy_search" in command for command in rendered)
    assert any("main_agentdojo.py" in command and "--suite shopping" in command for command in rendered)
    assert all("--defense modernbert_tagger" in command for command in rendered)
    assert all("wasp" not in command for command in rendered)


def test_full_script_pending_only_skips_complete_standard_results(tmp_path, monkeypatch):
    full = load_script("scripts/run_modernbert_tagger_full.py")
    monkeypatch.setitem(full.build_commands.__globals__, "ROOT", tmp_path)
    args = full.parse_args(
        [
            "--only-dataset", "squad_v2",
            "--only-attack", "combined",
            "--skip-agents",
            "--pending-only",
            "--dry-run",
        ]
    )

    result_dir = tmp_path / "results" / "evaluation_results" / "modernbert_tagger_full"
    result_dir.mkdir(parents=True)
    result_file = result_dir / "squad_v2-Qwen-Qwen3-4B-Instruct-2507-combined-modernbert_tagger-42.json"
    result_file.write_text(
        json.dumps({str(idx): {"utility": 1, "asr": 0} for idx in range(200)}),
        encoding="utf-8",
    )

    assert full.build_commands(args) == []


def test_full_script_local_datasets_only_skips_unavailable_standard_splits():
    full = load_script("scripts/run_modernbert_tagger_full.py")
    args = full.parse_args(
        [
            "--only-dataset", "opi",
            "--only-attack", "default",
            "--skip-agents",
            "--local-datasets-only",
            "--dry-run",
        ]
    )
    assert full.build_commands(args) == []

    args = full.parse_args(
        [
            "--only-dataset", "squad_v2",
            "--only-attack", "combined",
            "--skip-agents",
            "--local-datasets-only",
            "--dry-run",
        ]
    )
    rendered = [" ".join(command) for command in full.build_commands(args)]
    assert rendered == [
        f"{full.build_commands.__globals__['sys'].executable} main.py --config configs/experiments/modernbert_tagger.yaml "
        "--dataset squad_v2 --backend_llm Qwen/Qwen3-4B-Instruct-2507 --attack combined "
        "--defense modernbert_tagger --name modernbert_tagger_full --seed 42"
    ]


def write_agentdojo_row(root: Path, name: str, defense: str, suite: str, utility: bool, security: bool) -> None:
    path = (
        root
        / "results"
        / "agent_evaluations"
        / "agentdojo"
        / name
        / f"local-piarena-{defense}"
        / suite
        / "user_task_0"
        / "important_instructions"
        / "injection_task_0.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"utility": utility, "security": security}), encoding="utf-8")


def test_agentdojo_export_uses_security_field_as_asr(tmp_path, monkeypatch):
    full = load_script("scripts/_pi_defense_full.py")
    monkeypatch.setattr(full, "ROOT", tmp_path)
    args = full.parse_args(
        ["--dry-run"],
        defense="modernbert_tagger",
        default_config="configs/experiments/modernbert_tagger.yaml",
        default_name="modernbert_tagger_full",
        script_name="run_modernbert_tagger_full.py",
    )

    write_agentdojo_row(tmp_path, args.name, "modernbert_tagger", "banking", utility=True, security=False)
    write_agentdojo_row(tmp_path, args.name, "modernbert_tagger", "workspace", utility=False, security=False)
    entry = full.AgentLeaderboardBenchmark(
        "agentdojo",
        "main_agentdojo.py",
        suites=("banking", "workspace"),
    )

    result, missing = full.load_agentdojo_entry(args, entry, "modernbert_tagger")

    assert missing is None
    assert result["utility"] == 50
    assert result["asr"] == 0


def test_agentdojo_entry_requires_all_requested_suites(tmp_path, monkeypatch):
    full = load_script("scripts/_pi_defense_full.py")
    monkeypatch.setattr(full, "ROOT", tmp_path)
    args = full.parse_args(
        ["--pending-only", "--dry-run"],
        defense="modernbert_tagger",
        default_config="configs/experiments/modernbert_tagger.yaml",
        default_name="modernbert_tagger_full",
        script_name="run_modernbert_tagger_full.py",
    )

    write_agentdojo_row(tmp_path, args.name, "modernbert_tagger", "banking", utility=True, security=False)
    entry = full.AgentLeaderboardBenchmark(
        "agentdojo",
        "main_agentdojo.py",
        suites=("banking", "workspace"),
    )

    result, missing = full.load_agentdojo_entry(args, entry, "modernbert_tagger")

    assert result is None
    assert "workspace" in missing
    assert not full.agent_result_complete(args, entry)


def test_force_agentdojo_runs_even_when_pending_result_is_complete(tmp_path, monkeypatch):
    full = load_script("scripts/run_modernbert_tagger_full.py")
    monkeypatch.setitem(full.build_commands.__globals__, "ROOT", tmp_path)
    args = full.parse_args(
        [
            "--skip-standard",
            "--only-dataset",
            "agentdojo",
            "--pending-only",
            "--force-agentdojo",
            "--dry-run",
        ]
    )
    for suite in ("workspace", "slack", "travel", "banking"):
        write_agentdojo_row(tmp_path, args.name, "modernbert_tagger", suite, utility=True, security=False)

    commands = full.build_commands(args)
    rendered = [" ".join(command) for command in commands]

    assert len(commands) == 1
    assert "--force-rerun" in rendered[0]
    assert "main_agentdojo.py" in rendered[0]


def test_agentdojo_export_reads_legacy_root_logdir(tmp_path, monkeypatch):
    full = load_script("scripts/_pi_defense_full.py")
    monkeypatch.setattr(full, "ROOT", tmp_path)
    args = full.parse_args(
        ["--dry-run"],
        defense="modernbert_tagger",
        default_config="configs/experiments/modernbert_tagger.yaml",
        default_name="modernbert_tagger_full",
        script_name="run_modernbert_tagger_full.py",
    )
    for suite in ("shopping", "github", "dailylife"):
        path = (
            tmp_path
            / args.name
            / "local-piarena-modernbert_tagger"
            / suite
            / "user_task_0"
            / "important_instructions"
            / "injection_task_0.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"utility": True, "security": False}), encoding="utf-8")
    entry = full.AgentLeaderboardBenchmark(
        "agentdyn",
        "main_agentdojo.py",
        suites=("shopping", "github", "dailylife"),
    )

    result, missing = full.load_agentdojo_entry(args, entry, "modernbert_tagger")

    assert missing is None
    assert result["utility"] == 100
    assert result["asr"] == 0
