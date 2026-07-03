import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "analyze_pi_sanitizer_logs.py"
SPEC = importlib.util.spec_from_file_location("analyze_pi_sanitizer_logs", SCRIPT_PATH)
assert SPEC is not None
analyze_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyze_module)


def test_discover_trace_paths_ignores_nested_smoke_runs(tmp_path):
    standard = (
        tmp_path
        / "deepseek-v4-flash-deepseek_flash_pi_sanitizer"
        / "shopping"
        / "user_task_0"
        / "important_instructions"
        / "injection_task_0.json"
    )
    nested_smoke = (
        tmp_path
        / "pi_sanitizer_agentdyn_smoke_v3"
        / "deepseek-v4-flash-deepseek_flash_pi_sanitizer"
        / "shopping"
        / "user_task_0"
        / "important_instructions"
        / "injection_task_0.json"
    )
    standard.parent.mkdir(parents=True)
    standard.write_text("{}")
    nested_smoke.parent.mkdir(parents=True)
    nested_smoke.write_text("{}")

    paths = analyze_module.discover_trace_paths(
        tmp_path,
        pipeline="deepseek-v4-flash-deepseek_flash_pi_sanitizer",
        suites=["shopping"],
        attack="important_instructions",
    )

    assert paths == [standard]
