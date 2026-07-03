"""Figure 3: cumulative defense latency vs. number of protected tool outputs.

Generative baselines: per-task mean and mean calls/task measured on AgentDyn
(DeepSeek-V4 Flash deployment, 620 cases; from results/defense_ops.csv, the
per-call mean is per-task / calls-per-task). MinSpan per-call latency is the
local Direct-test median (results/local-eval/p3_direct_test.json).
Squares mark measured per-task means at the observed mean call count.
"""
import csv
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import paperstyle as ps

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ps.apply()

ops = {row["defense"]: row for row in csv.DictReader(open(ROOT / "results" / "defense_ops.csv"))}


def per_call_and_task(defense: str) -> tuple[float, float]:
    task_s = float(ops[defense]["mean_task_latency_ms"]) / 1000
    return task_s / float(ops[defense]["mean_calls_per_task"]), task_s


minspan_local = json.load(open(ROOT / "results" / "local-eval" / "p3_direct_test.json"))
minspan_per_call = float(minspan_local["timing"]["tagger"]["p50_latency_seconds"])

SERIES = [
    # name, per-call mean (s), measured per-task mean (s), color
    (*(("DS Sanitizer",) + per_call_and_task("deepseek_flash_pi_sanitizer")), ps.VERMILLION),
    (*(("DataFilter",) + per_call_and_task("data_filter")), ps.PINK),
    ("MinSpan (ours)", minspan_per_call, None, ps.BLUE),
]
MARKERS = {"DS Sanitizer": "s", "DataFilter": "D", "MinSpan (ours)": "o"}

n = np.arange(1, 31)
fig, ax = plt.subplots(figsize=(ps.TEXTWIDTH, 2.5))

for name, per_call, per_task, color in SERIES:
    ax.plot(n, per_call * n, color=color, lw=1.4, zorder=3,
            marker=MARKERS[name], markevery=[4, 9, 14, 19, 24, 29],
            ms=3.2, markerfacecolor="white", markeredgewidth=0.9)
    ax.annotate(name, (n[-1], per_call * n[-1]), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=7.5,
                color=color, fontweight="bold")

# measured per-task means at the observed mean call count (~11.4 calls/task)
for name, per_call, per_task, color in SERIES:
    if per_task is None:
        continue
    calls = per_task / per_call
    ax.scatter([calls], [per_task], s=22, marker=MARKERS[name], color=color,
               zorder=4, edgecolors=ps.INK, linewidths=0.5)
_ds_call, _ds_task = per_call_and_task("deepseek_flash_pi_sanitizer")
ax.annotate("measured per-task mean", (_ds_task / _ds_call, _ds_task),
            xytext=(8, 10), textcoords="offset points", fontsize=7,
            color=ps.MUTED,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.MUTED,
                            shrinkA=0, shrinkB=3))

ax.axhline(1.0, color=ps.GRID, lw=0.7, ls=(0, (4, 3)), zorder=1)
ax.annotate("1 s", (36.8, 1.0), fontsize=7, color=ps.MUTED,
            va="center", ha="right")

ax.set_yscale("log")
ax.set_ylim(1e-3, 2e2)
ax.set_xlim(0, 38)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])

def fmt(v, _):
    if v >= 1:
        return f"{v:g} s"
    return f"{v*1000:g} ms"

ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))
ax.set_xlabel("Protected tool outputs per task")
ax.set_ylabel("Cumulative defense latency")
ax.grid(axis="y")
ax.grid(axis="x", visible=False)

fig.savefig(HERE.parent / "cumulative_latency.pdf")
print("wrote cumulative_latency.pdf")
