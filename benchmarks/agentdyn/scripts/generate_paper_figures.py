from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "notebooks" / "outputs" / "over-defense-analysis"
OUTPUT_DIR = ROOT / "docs" / "paper_figures"

METRIC_LABELS = {
    "benign_utility": "Benign utility",
    "utility_under_attack": "Utility under attack",
    "asr": "ASR",
}

METRIC_SHORT = {
    "benign_utility": "BU",
    "utility_under_attack": "UAA",
    "asr": "ASR",
}

FAMILY_LABELS = {
    "none": "No defense",
    "prompting_based": "Prompting",
    "filtering_based": "Filtering",
    "system_level": "System-level",
}

FAMILY_ORDER = ["none", "prompting_based", "filtering_based", "system_level"]

DEFENSE_ORDER = [
    "none",
    "repeat_user_prompt",
    "spotlighting_with_delimiting",
    "prompt_guard_2_detector",
    "piguard_detector",
    "transformers_pi_detector",
    "tool_filter",
    "progent",
    "drift",
    "data_filter",
    "pi_sanitizer",
]

DEFENSE_COLORS = {
    "none": "#8F8F8F",
    "repeat_user_prompt": "#4C78A8",
    "spotlighting_with_delimiting": "#72B7B2",
    "prompt_guard_2_detector": "#F58518",
    "piguard_detector": "#ECA82C",
    "transformers_pi_detector": "#B279A2",
    "tool_filter": "#7F3C8D",
    "progent": "#9D755D",
    "drift": "#6B6ECF",
    "data_filter": "#2A9D8F",
    "pi_sanitizer": "#E76F51",
    "modernbert_tagger": "#4C9F70",
}

FAMILY_COLORS = {
    "none": "#8F8F8F",
    "prompting_based": "#4C78A8",
    "filtering_based": "#F58518",
    "system_level": "#7F3C8D",
}

STATUS_COLORS = {
    "complete": "#2A9D8F",
    "partial": "#E9C46A",
    "full": "#E76F51",
    "not_seen": "#B0BEC5",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def metric_pivot(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = metrics[(metrics["suite"] == "overall") & metrics["metric"].isin(METRIC_LABELS)].copy()
    row_meta = overall[
        ["model", "model_label", "defense", "defense_label", "family"]
    ].drop_duplicates(subset=["model", "defense"])
    pivot = overall.pivot_table(
        index=["model", "defense"],
        columns="metric",
        values="value_pct",
        aggfunc="first",
    ).reset_index()
    return pivot.merge(row_meta, on=["model", "defense"], how="left")


def format_float(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    (OUTPUT_DIR / f"{name}.md").write_text(to_markdown_table(df), encoding="utf-8")
    (OUTPUT_DIR / f"{name}.tex").write_text(
        df.to_latex(index=False, float_format="%.2f", escape=True),
        encoding="utf-8",
    )


def plot_family_tradeoff(metrics: pd.DataFrame) -> None:
    data = metrics[(metrics["suite"] == "overall") & metrics["metric"].isin(METRIC_LABELS)].copy()
    data["family_group"] = data["family"].fillna("none")
    data = data[data["family_group"].isin(FAMILY_ORDER)]

    records = []
    for family in FAMILY_ORDER:
        for metric in METRIC_LABELS:
            subset = data[(data["family_group"] == family) & (data["metric"] == metric)]["value_pct"].dropna()
            if subset.empty:
                continue
            sem = float(subset.std(ddof=1) / np.sqrt(len(subset))) if len(subset) > 1 else 0.0
            records.append(
                {
                    "family": family,
                    "metric": metric,
                    "mean": float(subset.mean()),
                    "sem": sem,
                    "n": int(len(subset)),
                }
            )
    summary = pd.DataFrame.from_records(records)
    save_table(
        summary.assign(
            family=summary["family"].map(FAMILY_LABELS),
            metric=summary["metric"].map(METRIC_LABELS),
        ).rename(columns={"family": "Defense group", "metric": "Metric", "mean": "Mean (%)", "sem": "SEM", "n": "N"}),
        "table_defense_family_summary",
    )

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.35), sharex=True)
    x = np.arange(len(FAMILY_ORDER))
    for ax, metric in zip(axes, METRIC_LABELS):
        subset = summary[summary["metric"] == metric].set_index("family").reindex(FAMILY_ORDER)
        means = subset["mean"].to_numpy()
        sems = subset["sem"].to_numpy()
        colors = [FAMILY_COLORS[family] for family in FAMILY_ORDER]
        bars = ax.bar(x, means, color=colors, width=0.68, edgecolor="white", linewidth=0.7)
        ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, zorder=3)
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.4,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0, max(80, float(np.nanmax(means + sems)) + 8))
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS[family] for family in FAMILY_ORDER], rotation=24, ha="right")
        if metric == "asr":
            ax.set_ylabel("% (lower is better)")
        else:
            ax.set_ylabel("% (higher is better)")
    fig.tight_layout()
    save_figure(fig, "fig1_defense_family_tradeoff")


def plot_deepseek_pareto(metrics: pd.DataFrame) -> None:
    pivot = metric_pivot(metrics)
    pivot = pivot[pivot["model"].isin(["deepseek-v4-flash", "deepseek-v4-pro"])].copy()
    pivot = pivot[pivot["defense"].isin(DEFENSE_ORDER)]
    pivot["family_group"] = pivot["family"].fillna("none")
    pivot.loc[pivot["defense"].isin(["data_filter", "pi_sanitizer"]), "family_group"] = "sanitizer"

    table = pivot[["model_label", "defense_label", "benign_utility", "utility_under_attack", "asr"]].copy()
    table = table.rename(
        columns={
            "model_label": "Model",
            "defense_label": "Defense",
            "benign_utility": "BU (%)",
            "utility_under_attack": "UAA (%)",
            "asr": "ASR (%)",
        }
    )
    table["Defense"] = pd.Categorical(table["Defense"], [metrics[metrics["defense"] == d]["defense_label"].iloc[0] for d in DEFENSE_ORDER if not metrics[metrics["defense"] == d].empty], ordered=True)
    table = table.sort_values(["Model", "Defense"]).reset_index(drop=True)
    save_table(table, "table_deepseek_defense_tradeoff")

    label_offsets = {
        ("deepseek-v4-flash", "none"): (4, 8),
        ("deepseek-v4-flash", "prompt_guard_2_detector"): (4, -10),
        ("deepseek-v4-flash", "tool_filter"): (4, -5),
        ("deepseek-v4-flash", "progent"): (4, -10),
        ("deepseek-v4-flash", "drift"): (4, 5),
        ("deepseek-v4-flash", "data_filter"): (4, -9),
        ("deepseek-v4-flash", "pi_sanitizer"): (4, 8),
        ("deepseek-v4-pro", "none"): (4, 5),
        ("deepseek-v4-pro", "spotlighting_with_delimiting"): (4, 5),
        ("deepseek-v4-pro", "prompt_guard_2_detector"): (4, -10),
        ("deepseek-v4-pro", "piguard_detector"): (4, 3),
        ("deepseek-v4-pro", "tool_filter"): (4, -5),
        ("deepseek-v4-pro", "progent"): (4, -10),
        ("deepseek-v4-pro", "drift"): (4, 5),
    }
    labeled = set(label_offsets)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0), sharey=True)
    for ax, model in zip(axes, ["deepseek-v4-flash", "deepseek-v4-pro"]):
        sub = pivot[pivot["model"] == model].sort_values("asr", ascending=False)
        for _, row in sub.iterrows():
            defense = row["defense"]
            label = row["defense_label"]
            color = DEFENSE_COLORS.get(defense, "#777777")
            edge = "#111111" if defense in {"data_filter", "pi_sanitizer"} else "white"
            marker = "*" if defense in {"data_filter", "pi_sanitizer"} else "o"
            size = 38 + float(row["benign_utility"] or 0) * (2.0 if defense in {"data_filter", "pi_sanitizer"} else 1.4)
            ax.scatter(row["asr"], row["utility_under_attack"], s=size, color=color, edgecolor=edge, linewidth=0.7, marker=marker, alpha=0.9, zorder=3)
            if (model, defense) in labeled:
                dx, dy = label_offsets[(model, defense)]
                ax.annotate(label, (row["asr"], row["utility_under_attack"]), textcoords="offset points", xytext=(dx, dy), fontsize=6.5)
        ax.set_title(sub["model_label"].iloc[0])
        ax.set_xlabel("ASR (%) ↓")
        ax.set_ylim(-3, 88)
        ax.axhline(70, color="#CCCCCC", linestyle="--", linewidth=0.8, zorder=1)
        ax.axvline(5, color="#CCCCCC", linestyle="--", linewidth=0.8, zorder=1)
        ax.text(1.2, 82, "Target\nregion", fontsize=7, color="#555555", ha="left", va="top")
        if model == "deepseek-v4-flash":
            ax.set_xlim(-0.6, 6.5)
        else:
            ax.set_xlim(-1, 45)
    axes[0].set_ylabel("Utility under attack (%) ↑")
    fig.tight_layout()
    save_figure(fig, "fig2_deepseek_pareto")


def plot_sanitizer_metrics(metrics: pd.DataFrame) -> None:
    defenses = ["none", "data_filter", "pi_sanitizer"]
    data = metrics[
        (metrics["source"] == "deepseek")
        & (metrics["model"] == "deepseek-v4-flash")
        & (metrics["suite"] == "overall")
        & (metrics["defense"].isin(defenses))
        & (metrics["metric"].isin(METRIC_LABELS))
    ].copy()
    pivot = data.pivot_table(index=["defense", "defense_label"], columns="metric", values="value_pct", aggfunc="first").reset_index()
    pivot["defense"] = pd.Categorical(pivot["defense"], defenses, ordered=True)
    pivot = pivot.sort_values("defense")
    save_table(
        pivot[["defense_label", "benign_utility", "utility_under_attack", "asr"]].rename(
            columns={
                "defense_label": "Defense",
                "benign_utility": "BU (%)",
                "utility_under_attack": "UAA (%)",
                "asr": "ASR (%)",
            }
        ),
        "table_flash_sanitizer_macro",
    )

    fig, ax = plt.subplots(figsize=(4.75, 2.55))
    metrics_order = ["benign_utility", "utility_under_attack", "asr"]
    x = np.arange(len(metrics_order))
    width = 0.24
    for i, (_, row) in enumerate(pivot.iterrows()):
        defense = row["defense"]
        offset = (i - 1) * width
        values = [row[metric] for metric in metrics_order]
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.9,
            label=row["defense_label"],
            color=DEFENSE_COLORS.get(defense, "#777777"),
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_SHORT[metric] for metric in metrics_order])
    ax.set_ylabel("%")
    ax.set_ylim(0, 85)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    save_figure(fig, "fig3_flash_sanitizer_macro")


def plot_sanitizer_cleaning_ops() -> None:
    run_status = pd.read_csv(INPUT_DIR / "fig4_run_status.csv")
    benign_edits = pd.read_csv(INPUT_DIR / "fig4_benign_edit_totals.csv")
    ops = pd.read_csv(INPUT_DIR / "fig4_ops.csv")
    defense_labels = {
        "data_filter": "DataFilter",
        "pi_sanitizer": "PI sanitizer",
        "modernbert_tagger": "ModernBERT tagger",
    }

    details = pd.read_csv(INPUT_DIR / "fig4_macro.csv").pivot_table(
        index=["defense", "defense_label"],
        columns="metric",
        values="value_pct",
        aggfunc="first",
    ).reset_index()
    status_pivot = run_status.pivot_table(index="defense", columns="status", values="percent", aggfunc="first").reset_index()
    details = details.merge(status_pivot, on="defense", how="left")
    details = details.merge(benign_edits[["defense", "changed_percent"]], on="defense", how="left")
    details = details.merge(ops[["defense", "parse_ok_pct", "avg_latency_sec"]], on="defense", how="left")
    details = details.rename(
        columns={
            "defense_label": "Defense",
            "benign_utility": "BU (%)",
            "utility_under_attack": "UAA (%)",
            "asr": "ASR (%)",
            "complete": "Full removal (%)",
            "partial": "Partial removal (%)",
            "full": "Full miss (%)",
            "not_seen": "Not seen (%)",
            "changed_percent": "Benign edit (%)",
            "parse_ok_pct": "Parse OK (%)",
            "avg_latency_sec": "Latency (s)",
        }
    )
    save_table(
        details[
            [
                "Defense",
                "BU (%)",
                "UAA (%)",
                "ASR (%)",
                "Full removal (%)",
                "Partial removal (%)",
                "Full miss (%)",
                "Not seen (%)",
                "Benign edit (%)",
                "Parse OK (%)",
                "Latency (s)",
            ]
        ],
        "table_sanitizer_detailed",
    )

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.45), gridspec_kw={"width_ratios": [1.7, 1.0, 1.0]})
    defenses = ["data_filter", "pi_sanitizer", "modernbert_tagger"]
    y = np.arange(len(defenses))
    left = np.zeros(len(defenses))
    for status in ["complete", "partial", "full", "not_seen"]:
        values = (
            run_status[run_status["status"] == status]
            .set_index("defense")
            .reindex(defenses)["percent"]
            .fillna(0)
            .to_numpy()
        )
        axes[0].barh(y, values, left=left, color=STATUS_COLORS[status], label=status.replace("_", " ").title(), edgecolor="white", linewidth=0.6)
        left += values
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([defense_labels[d] for d in defenses])
    axes[0].set_xlabel("Attack cases (%)")
    axes[0].set_title("Injection removal")
    axes[0].invert_yaxis()

    edit_values = benign_edits.set_index("defense").reindex(defenses)["changed_percent"].to_numpy()
    bars = axes[1].bar(y, edit_values, color=[DEFENSE_COLORS[d] for d in defenses], edgecolor="white", linewidth=0.7)
    axes[1].set_xticks(y)
    axes[1].set_xticklabels([defense_labels[d] for d in defenses], rotation=20, ha="right")
    axes[1].set_ylim(0, 60)
    axes[1].set_title("Benign edits")
    axes[1].set_ylabel("%")
    for bar, value in zip(bars, edit_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    latency_values = ops.set_index("defense").reindex(defenses)["avg_latency_sec"].to_numpy()
    bars = axes[2].bar(y, latency_values, color=[DEFENSE_COLORS[d] for d in defenses], edgecolor="white", linewidth=0.7)
    axes[2].set_xticks(y)
    axes[2].set_xticklabels([defense_labels[d] for d in defenses], rotation=20, ha="right")
    axes[2].set_ylim(0, max(latency_values) + 0.8)
    axes[2].set_title("Latency")
    axes[2].set_ylabel("seconds")
    for bar, value in zip(bars, latency_values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, "fig4_sanitizer_cleaning_ops")


def write_figure_plan() -> None:
    text = """# Paper Figure Plan

This directory contains first-pass paper figures generated from `notebooks/outputs/over-defense-analysis/`.

## Main Figures

1. `fig1_defense_family_tradeoff.pdf`
   - Purpose: establish the broad security-utility tradeoff across existing defense families.
   - Takeaway: prompting-based defenses preserve utility but leave ASR; filtering/system-level defenses reduce ASR with large utility loss.

2. `fig2_deepseek_pareto.pdf`
   - Purpose: show the DeepSeek-V4 Flash/Pro results as a Pareto view.
   - Takeaway: system-level defenses can reach low ASR by collapsing utility, while selective sanitization is closer to the target region.

3. `fig3_flash_sanitizer_macro.pdf`
   - Purpose: compare No defense, DataFilter, and PI sanitizer on DeepSeek-V4 Flash.
   - Takeaway: PI sanitizer reaches ASR=0 with utility close to the undefended agent; DataFilter supports the same direction but is weaker.

4. `fig4_sanitizer_cleaning_ops.pdf`
   - Purpose: show sanitizer-level removal quality and deployment signals.
   - Takeaway: PI sanitizer has stronger removal and lower benign edit rate, but higher latency than DataFilter; this motivates a lightweight sanitizer.

## Tables

- `table_defense_family_summary.*`: means and SEMs used in Figure 1.
- `table_deepseek_defense_tradeoff.*`: DeepSeek-V4 Flash/Pro defense metrics.
- `table_flash_sanitizer_macro.*`: macro task metrics for No defense/DataFilter/PI sanitizer.
- `table_sanitizer_detailed.*`: removal, benign edit, parse, and latency metrics.

## Caveats Before Paper Submission

- Re-run PI sanitizer on all suites with one fixed runtime configuration.
- Add confidence intervals or raw numerator/denominator tables for final results.
- Replace DataFilter/PI sanitizer placeholders with the lightweight sanitizer once its results are available.
- Keep PDF outputs for LaTeX; PNG files are for quick review.
"""
    (OUTPUT_DIR / "figure_plan.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    metrics = pd.read_csv(INPUT_DIR / "combined_metrics.csv")
    plot_family_tradeoff(metrics)
    plot_deepseek_pareto(metrics)
    plot_sanitizer_metrics(metrics)
    plot_sanitizer_cleaning_ops()
    write_figure_plan()
    print(f"Wrote paper figures and tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
