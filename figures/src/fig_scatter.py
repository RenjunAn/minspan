"""Figure 2: utility under attack vs. attack success rate on AgentDyn."""
import csv
import pathlib

import matplotlib.pyplot as plt

import paperstyle as ps

HERE = pathlib.Path(__file__).resolve().parent
ps.apply()

pts = {}
with open(HERE / "agentdyn_scatter.csv") as fh:
    for row in csv.DictReader(fh):
        pts.setdefault(row["family"], []).append(
            (float(row["asr"]), float(row["ua"]))
        )

fig, ax = plt.subplots(figsize=(ps.TEXTWIDTH, 2.9))

STYLE = {
    "System-level": dict(color=ps.BLUE),
    "Detection": dict(color=ps.VERMILLION),
    "Prevention": dict(color=ps.GREEN),
    "Filtering": dict(color=ps.PINK),
}
for fam, st in STYLE.items():
    xs, ys = zip(*pts[fam])
    ax.scatter(xs, ys, s=16, color=st["color"], label=fam,
               linewidths=0.6, edgecolors="white", alpha=0.9, zorder=3)

xs, ys = zip(*pts["No defense"])
ax.scatter(xs, ys, s=26, marker="x", color=ps.INK, label="No defense",
           linewidths=1.0, zorder=4)

(mx, my), = pts["MinSpan"]
ax.scatter([mx], [my], s=110, marker="*", color=ps.INK,
           edgecolors="white", linewidths=0.6, label="MinSpan (ours)", zorder=5)
ax.annotate("MinSpan", (mx, my), xytext=(7, -10), textcoords="offset points",
            fontsize=8, fontweight="bold", color=ps.INK)

# direct labels required for the low-contrast filtering hue
fx = sorted(pts["Filtering"], key=lambda p: p[0])
lead = dict(arrowstyle="-", lw=0.5, color=ps.MUTED, shrinkA=1, shrinkB=2)
ax.annotate("DS Sanitizer", fx[0], xytext=(5.5, 71.5), textcoords="data",
            fontsize=7, color=ps.MUTED, va="center", arrowprops=lead)
ax.annotate("DataFilter", fx[1], xytext=(5.5, 59.5), textcoords="data",
            fontsize=7, color=ps.MUTED, va="center", arrowprops=lead)

# the mutual-destruction cluster at the origin
ax.annotate("CaMeL / Tool Filter: ASR 0, utility 0", (1.3, 0.9),
            xytext=(26, 2.5), textcoords="data", fontsize=7,
            color=ps.MUTED, va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.MUTED,
                            shrinkA=1, shrinkB=2))

# ideal operating corner
ax.annotate("ideal $\\nwarrow$", (1.2, 79.5), fontsize=7.5, style="italic",
            color=ps.MUTED)

ax.set_xlabel("Attack success rate (%)")
ax.set_ylabel("Utility under attack (%)")
ax.set_xlim(-1.5, 55)
ax.set_ylim(-3, 84)
ax.grid(axis="y")
ax.grid(axis="x", visible=False)

handles, labels = ax.get_legend_handles_labels()
order = ["No defense", "System-level", "Detection", "Prevention",
         "Filtering", "MinSpan (ours)"]
hl = dict(zip(labels, handles))
ax.legend([hl[k] for k in order], order, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), ncols=6, fontsize=7,
          handletextpad=0.2, columnspacing=0.7, borderpad=0.1)

fig.savefig(HERE.parent / "security_utility_scatter.pdf")
print("wrote security_utility_scatter.pdf")
