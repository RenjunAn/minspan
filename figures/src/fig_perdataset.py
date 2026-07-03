"""Figure 4: MinSpan per-dataset PIArena results under Direct attacks."""
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import paperstyle as ps

HERE = pathlib.Path(__file__).resolve().parent
ps.apply()

rows = list(csv.DictReader(open(HERE / "piarena_per_dataset.csv")))
names = [r["dataset"] for r in rows]
util = np.array([float(r["direct_utility"]) for r in rows])
asr = np.array([float(r["direct_asr"]) for r in rows])
y = np.arange(len(names))[::-1]  # first dataset on top

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(ps.TEXTWIDTH, 2.6), sharey=True,
    gridspec_kw={"wspace": 0.06},
)

ax1.barh(y, util, height=0.62, color=ps.BLUE, zorder=3)
ax1.set_xlabel("Direct utility (%) ↑")
ax1.set_xlim(0, 104)
ax1.set_yticks(y, names)

ax2.barh(y, asr, height=0.62, color=ps.VERMILLION, zorder=3)
ax2.set_xlabel("Direct ASR (%) ↓")
ax2.set_xlim(0, 20)

# selective labels: only the long-form summarization residue called out in text
for i, (name, a) in enumerate(zip(names, asr)):
    if a >= 10:
        ax2.annotate(f"{a:g}", (a, y[i]), xytext=(3, 0),
                     textcoords="offset points", va="center",
                     fontsize=7, color=ps.MUTED)

# separator between short-text/RAG group and long-text group
for ax in (ax1, ax2):
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.axhline(y[6] - 0.5, color=ps.MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.tick_params(axis="y", length=0)
ax2.annotate("short-text / RAG", (19.6, y[6] - 0.05), fontsize=6.5,
             color=ps.MUTED, ha="right", va="bottom", style="italic")
ax2.annotate("long-text", (19.6, y[7] + 0.1), fontsize=6.5,
             color=ps.MUTED, ha="right", va="top", style="italic")

fig.savefig(HERE.parent / "piarena_per_dataset.pdf")
print("wrote piarena_per_dataset.pdf")
