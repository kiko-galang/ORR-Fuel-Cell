"""
Book house style for chapter figures, per Figure_Preparation_Guide.pdf.

Importing this module applies the style globally via rcParams.update() at
import time, so import it before creating any figure.

Scope (see Figure_Preparation_Guide.pdf, Sections 2-11):
    - Fixed mm-based figure canvases (FIGSIZE_*), not cropped by savefig.
    - Arial typography, with a documented fallback chain if unavailable.
    - A 4-color discrete palette (black/blue/green/red) plus viridis for
      genuinely continuous quantities; more than 4 discrete series combine
      the 4 colors with marker shape / linestyle / hatch (Section 6).
    - Panel labels "(a)", "(b)", ... at 9pt bold, upper-left of each panel.

600 dpi PNG is used as the raster deliverable throughout (the guide's
explicitly sanctioned fallback when vector PDF isn't produced); the mm
canvas sizes and typography below apply equally to either.
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
# Universal book figure dimensions
# ============================================================
MM_TO_INCH = 1 / 25.4

FIGSIZE_SMALL        = (80 * MM_TO_INCH, 60 * MM_TO_INCH)    # single-panel / compact
FIGSIZE_SMALL_SQUARE = (80 * MM_TO_INCH, 80 * MM_TO_INCH)    # maps, spectra, square
FIGSIZE_LARGE        = (180 * MM_TO_INCH, 108 * MM_TO_INCH)  # wide / multi-panel
FIGSIZE_LARGE_SQUARE = (180 * MM_TO_INCH, 180 * MM_TO_INCH)  # complex square composites

# ============================================================
# Colors
# ============================================================
BOOK_COLORS = {
    "black": "#000000",   # primary data, controls, outlines, text
    "blue":  "#06739C",   # C96 M26 Y0 K39 -- dataset/condition 1
    "green": "#00991A",   # C100 M0 Y83 K40 -- dataset/condition 2
    "red":   "#B30000",   # C0 M100 Y100 K30 -- highlight/benchmark/critical
}
COLOR_CYCLE = [BOOK_COLORS["black"], BOOK_COLORS["blue"],
               BOOK_COLORS["green"], BOOK_COLORS["red"]]

MARKERS    = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":"]
HATCHES    = ["", "///", "...", "xx"]   # for a >4th series in a filled/stacked plot

# ============================================================
# rcParams (verbatim from the Figure Preparation Guide, Section 11)
# ============================================================
BOOK_RC = {
    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,

    # Mathematical text
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.default": "regular",

    # Default figure and display
    "figure.figsize": FIGSIZE_SMALL,
    "figure.dpi": 150,
    "figure.facecolor": "white",

    # Export
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.02,

    # Axes
    "axes.linewidth": 0.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.titlepad": 4,
    "axes.labelpad": 3,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.unicode_minus": True,

    # Axis spines
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True,

    # Continuous quantitative data
    "image.cmap": "viridis",

    # Tick direction and placement
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,

    # Tick labels
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.color": "black",
    "ytick.color": "black",

    # Major ticks
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,

    # Minor ticks
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,

    # Lines and markers
    "lines.linewidth": 0.9,
    "lines.linestyle": "-",
    "lines.markersize": 4.0,
    "lines.markeredgewidth": 0.5,
    "lines.markeredgecolor": "black",

    # Scatter plots
    "scatter.marker": "o",
    "scatter.edgecolors": "black",

    # Legends
    "legend.frameon": False,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "legend.handlelength": 1.6,
    "legend.handleheight": 0.7,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 0.8,
    "legend.borderaxespad": 0.4,

    # Error bars, patches, and hatching
    "errorbar.capsize": 2.0,
    "patch.linewidth": 0.5,
    "patch.edgecolor": "black",
    "hatch.linewidth": 0.5,

    # Layout
    "figure.constrained_layout.use": False,
}
mpl.rcParams.update(BOOK_RC)

# Secondary line width (Section 7): use for reference/annotation lines that
# should read as subordinate to the main data series (0.9 pt).
SECONDARY_LINEWIDTH = 0.7


# ============================================================
# Helpers
# ============================================================

def panel_label(ax, letter: str, **kw) -> None:
    """
    (a)/(b)/... panel label, upper-left of each panel (Section 8).

    Placed just above the axes rather than the guide's literal in-axes
    (0.02, 0.98) so it can never collide with an in-panel legend -- a
    real collision the guide's own snippet doesn't guard against, since
    legends are commonly placed "upper left" precisely because that
    corner is often the emptiest part of a data plot.
    """
    ax.text(
        0.0, 1.02, f"({letter})",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9, fontweight="bold",
        **kw,
    )


def add_panel_labels(axes) -> None:
    """Label every Axes in `axes` (any shape) (a), (b), (c), ... in order."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    for ax, letter in zip(np.ravel(np.asarray(axes, dtype=object)), letters):
        panel_label(ax, letter)


def series_style(i: int) -> dict:
    """
    Style for discrete series index i (0-based): cycle the 4 approved colors
    first: once past 4 series, repeat colors but vary linestyle (Section 6 --
    "combine the four approved colors with marker shape, line style").
    """
    color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
    ls    = LINESTYLES[i // len(COLOR_CYCLE) % len(LINESTYLES)]
    marker = MARKERS[i % len(MARKERS)]
    return {"color": color, "linestyle": ls, "marker": marker}


def style_axes_default(ax) -> None:
    """
    Re-assert the tick/spine rules on an Axes created outside the normal
    rcParams path (e.g. a twin axis, or a 3D/projection axis that ignores
    some 2D rcParams). No-op for anything already covered by BOOK_RC.
    """
    ax.tick_params(axis="both", which="both", direction="in")
    ax.tick_params(top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(BOOK_RC["axes.linewidth"])


def savefig_book(fig, path: str) -> None:
    """
    Save at the figure's own fixed canvas size -- NOT bbox_inches="tight",
    which would crop the export away from the mandated mm dimensions
    (Section 10: "Remove unnecessary external white space WITHOUT changing
    the required figure canvas dimensions"). Call fig.tight_layout() before
    this to fit labels inside the fixed canvas instead.

    Also saves a sibling PDF (same basename) -- Section 10's preferred
    delivery format for plots/diagrams/line art, vector so it is resolution
    independent regardless of the 600 dpi PNG raster preview.
    """
    fig.savefig(path)
    pdf_path = str(path).rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path)
    print(f"  Saved: {path}  +  {pdf_path}")
