from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def load_analysis_module():
    module_path = Path(__file__).resolve().parents[2] / "notebooks" / "over-defense-analysis.py"
    spec = importlib.util.spec_from_file_location("over_defense_analysis", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_entrypoint_uses_over_defense_analysis_filename():
    module = load_analysis_module()

    assert module.DEFAULT_OUTPUT_DIRNAME == "over-defense-analysis"


def test_tradeoff_figures_keep_baseline_explanation_off_canvas():
    module = load_analysis_module()
    metrics = pd.DataFrame(
        [
            {
                "model": "gpt-4o-mini",
                "defense": "none",
                "metric": "benign_utility",
                "suite": "overall",
                "value_pct": 50.0,
                "source": "appendix_g",
            },
            {
                "model": "gpt-4o-mini",
                "defense": "repeat_user_prompt",
                "metric": "benign_utility",
                "suite": "overall",
                "value_pct": 45.0,
                "source": "appendix_g",
            },
        ]
    )

    figure = module.plot_tradeoff_figure(metrics, "benign_utility", model_order=["gpt-4o-mini"])

    assert module.TRADEOFF_FIGURE_TEXT["benign_utility"]["baseline_label"] == "Baseline"
    assert "note" not in module.TRADEOFF_FIGURE_TEXT["benign_utility"]
    assert not any("no attack" in text.get_text() or "defense curves" in text.get_text() for text in figure.texts)


def test_notebook_displays_freshly_generated_png_bytes():
    notebook_path = Path(__file__).resolve().parents[2] / "notebooks" / "over-defense-analysis.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "def _display_png(path: Path)" in source
    assert "Image(data=path.read_bytes())" in source
    assert source.count("_display_png(paths[") == 4
    assert 'metrics, paths = write_outputs(ROOT)' in source
    assert "the baseline is" not in source
    assert "defense curves" not in source


def test_filter_detail_summary_reports_benign_output_edits(monkeypatch):
    module = load_analysis_module()
    metrics = pd.DataFrame(
        [
            {
                "model": "deepseek-v4-flash",
                "defense": "data_filter",
                "metric": "benign_utility",
                "suite": "overall",
                "value_pct": 50.0,
                "source": "deepseek",
            },
            {
                "model": "deepseek-v4-flash",
                "defense": "data_filter",
                "metric": "benign_utility",
                "suite": "dailylife",
                "value_pct": 50.0,
                "source": "deepseek",
            },
        ]
    )
    op_rows = pd.DataFrame(
        [
            {
                "defense": "data_filter",
                "suite": "dailylife",
                "case": "benign",
                "changed": True,
                "api_ok": True,
                "parse_ok": True,
                "latency_ms": 1000,
            },
            {
                "defense": "data_filter",
                "suite": "dailylife",
                "case": "benign",
                "changed": False,
                "api_ok": True,
                "parse_ok": True,
                "latency_ms": 1000,
            },
            {
                "defense": "data_filter",
                "suite": "dailylife",
                "case": "attack",
                "changed": True,
                "api_ok": True,
                "parse_ok": True,
                "latency_ms": 1000,
            },
        ]
    )
    run_rows = pd.DataFrame(
        [{"defense": "data_filter", "suite": "dailylife", "status": "complete"}]
    )
    event_rows = pd.DataFrame(
        [{"defense": "data_filter", "suite": "dailylife", "status": "complete"}]
    )
    monkeypatch.setattr(module, "build_combined_metrics", lambda _root: metrics)
    monkeypatch.setattr(module, "load_filter_detail_rows", lambda _root: (run_rows, event_rows, op_rows))

    summary = module.build_filter_detail_summary(Path.cwd())

    benign_edits = summary["benign_edit_totals"]
    row = benign_edits[benign_edits["defense"] == "data_filter"].iloc[0]
    assert row["changed_count"] == 1
    assert row["total"] == 2
    assert row["changed_percent"] == 50.0
