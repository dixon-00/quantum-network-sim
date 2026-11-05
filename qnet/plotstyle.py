"""Shared plot style: one colour per series in a fixed order, thin lines, quiet grid."""
import matplotlib as mpl
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
THRESH = "#898781"   # reference lines (thresholds) are grey, never a series colour


def apply():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5,
    })


def threshold(ax, y, label, horizontal=True):
    """Grey dashed reference line with a small label."""
    if horizontal:
        ax.axhline(y, color=THRESH, lw=1, ls="--", zorder=1)
        ax.annotate(label, (1, y), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK2)
    else:
        ax.axvline(y, color=THRESH, lw=1, ls="--", zorder=1)
        ax.annotate(label, (y, 1), xycoords=("data", "axes fraction"), xytext=(3, -4),
                    textcoords="offset points", ha="left", va="top", fontsize=8, color=INK2)


def save(fig, name):
    fig.savefig(f"figures/{name}.png")
