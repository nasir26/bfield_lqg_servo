"""Shared figure style, so every figure in the repo looks like the same document.

Vector output with embedded Type-42 fonts, colourblind-safe palette, and a
mandatory [SIMULATED] stamp on every figure. The stamp is applied here rather
than left to each caller precisely so it cannot be forgotten: section 9 of the
brief requires it without exception, and a marker that depends on remembering it
is a marker that will eventually be missing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "figures"

MM = 1 / 25.4
SINGLE = 88 * MM
DOUBLE = 180 * MM

# Okabe-Ito
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def setup_style() -> None:
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })


def save_fig(fig, name: str, simulated: bool = True) -> Path:
    """Save to figures/<name>.pdf (and .png), stamping [SIMULATED].

    `simulated=False` is reserved for figures built from measured hardware data.
    Nothing in this repository may pass it until such data exists.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    if simulated:
        fig.text(0.995, 0.005, "[SIMULATED]", ha="right", va="bottom",
                 fontsize=6, color="0.45", alpha=0.9)
    out = FIGDIR / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / f"{name}.png", dpi=200)
    plt.close(fig)
    return out
