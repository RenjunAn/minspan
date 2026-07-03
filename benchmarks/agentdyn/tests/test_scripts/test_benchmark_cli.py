import os
import subprocess
from pathlib import Path

from click.testing import CliRunner

from agentdojo.scripts import benchmark as benchmark_script


class FakeSuite:
    name = "shopping"


def test_benchmark_cli_forwards_tool_output_format(monkeypatch):
    seen = {}

    def fake_benchmark_suite(
        suite,
        model,
        logdir,
        force_rerun,
        benchmark_version,
        user_tasks=(),
        injection_tasks=(),
        model_id=None,
        attack=None,
        defense=None,
        tool_delimiter="tool",
        system_message_name=None,
        system_message=None,
        live=None,
        tool_output_format=None,
    ):
        seen["tool_output_format"] = tool_output_format
        return {
            "utility_results": {("user_task_0", "injection_task_0"): True},
            "security_results": {},
            "injection_tasks_utility_results": {},
        }

    monkeypatch.setattr(benchmark_script, "get_suite", lambda benchmark_version, suite_name: FakeSuite())
    monkeypatch.setattr(benchmark_script, "benchmark_suite", fake_benchmark_suite)
    monkeypatch.setattr(benchmark_script, "show_results", lambda suite_name, results, show_security_results: None)

    result = CliRunner().invoke(
        benchmark_script.main,
        [
            "--suite",
            "shopping",
            "--model",
            "deepseek-v4-flash",
            "--tool-output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert seen["tool_output_format"] == "json"


def test_benchmark_cli_lists_token_tagger_defenses():
    result = CliRunner().invoke(benchmark_script.main, ["--help"])

    assert result.exit_code == 0
    assert "datafilter_bidir_tagger" in result.output
    assert "modernbert_tagger" in result.output


def test_token_tagger_runner_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "DEFENSES": "modernbert_tagger datafilter_bidir_tagger",
        "AGENTS": "deepseek-v4-flash",
        "SUITES": "shopping",
        "ATTACKS": "important_instructions",
        "MODERNBERT_TAGGER_CHECKPOINT": str(tmp_path / "mb"),
        "DATAFILTER_TAGGER_CHECKPOINT": str(tmp_path / "bidir"),
        "DATAFILTER_BACKBONE_MODEL": str(tmp_path / "df"),
        "RUN_LOG": str(tmp_path / "token_taggers.log"),
    }

    result = subprocess.run(
        ["bash", "scripts/run_defense_token_taggers.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--defense modernbert_tagger" in result.stdout
    assert "--defense datafilter_bidir_tagger" in result.stdout
    assert " -f" in result.stdout
