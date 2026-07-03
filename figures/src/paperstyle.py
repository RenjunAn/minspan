"""Shared matplotlib style for the paper figures (Okabe-Ito palette)."""
import matplotlib as mpl

# Okabe-Ito, colorblind-safe (validated: lightness band, chroma, CVD dE ok)
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
ORANGE = "#E69F00"
INK = "#1a1a1a"
MUTED = "#666666"
GRID = "#d9d9d9"

# ICLR text block is 5.5in wide; \small caption text is 9pt.
TEXTWIDTH = 5.5

def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
