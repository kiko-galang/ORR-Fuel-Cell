# customplot.py — Usage Guide

## Overview

`customplot.py` is a project-level plotting utility module that wraps matplotlib/seaborn with publication-ready defaults. It exports pre-built color palettes and a single primary function, `gengrid`, for generating consistently styled figure grids.

**Import:**
```python
from customplot import gengrid
# Color palettes are also available as module-level names, e.g.:
# from customplot import tableau20, rainbow_1, Pantone, etc.
```

> **Note:** On import, `customplot.py` reads `Color Swatches from UC Berkeley.xlsx` from the **current working directory**. Make sure this file is present before importing; otherwise color palette construction will fail.

---

## `gengrid` — Figure Grid Generator

### Signature

```python
fig, axs, sec_ax_list = gengrid(
    n_cols=1,
    n_rows=1,
    dpi_fig=600,
    fig_size=(8, 6),
    genlabels=True,
    size_inches=(3.25, 2.5),
    ticklabel_size=8,
    label_pos=-0.15,
    bold=False,
    minor=True,
    secondary_yaxis=None
)
```

### What It Does

1. Creates a matplotlib `fig, axs` via `plt.subplots(ncols=n_cols, nrows=n_rows)`.
2. Overrides the figure size with `size_inches` (the `fig_size` parameter is effectively superseded by this).
3. Applies consistent tick styling to every axis:
   - Ticks point **inward** on all four sides.
   - Minor ticks on by default (`AutoMinorLocator(2)` — one minor tick between each major).
   - Spine linewidth set to 1.25.
   - Tick label font size controlled by `ticklabel_size`.
4. Optionally adds **subplot labels** (a), b), c), … placed at `(label_pos, 1.1)` in axes-relative coordinates (upper-left, outside the frame).
5. Optionally attaches **secondary y-axes** (right-side twinx) to specific panels.

### Return Values

| Variable | Type | Description |
|---|---|---|
| `fig` | `matplotlib.figure.Figure` | The figure object |
| `axs` | `ndarray` or single `Axes` | Axes grid. Shape depends on grid size (see below). |
| `sec_ax_list` | list / `Axes` / `None` | Secondary y-axes. Shape mirrors `axs`. `None` at positions where no secondary axis was requested. If single panel with secondary axis, returns a single `Axes` object directly. |

### `axs` Shape by Grid Configuration

| `n_rows` × `n_cols` | `axs` type | Access pattern |
|---|---|---|
| 1 × 1 (single panel) | single `Axes` | `axs` directly |
| N × 1 or 1 × N (1D grid) | 1D `ndarray` | `axs[i]` |
| N × M (2D grid) | 2D `ndarray` | `axs[row][col]` |

The third return value is almost always discarded as `_` unless secondary axes are needed.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_cols` | `1` | Number of columns in the subplot grid |
| `n_rows` | `1` | Number of rows in the subplot grid |
| `dpi_fig` | `600` | DPI for the figure (high-res, suitable for publication) |
| `fig_size` | `(8, 6)` | Initial figsize passed to `plt.subplots` — **overridden** by `size_inches` |
| `genlabels` | `True` | Whether to auto-label panels a), b), c), … |
| `size_inches` | `(3.25, 2.5)` | Final figure size set via `fig.set_size_inches(w, h)`. This is the effective size. |
| `ticklabel_size` | `8` | Font size for tick labels on all axes |
| `label_pos` | `-0.15` | Horizontal position of panel labels in axes coordinates (negative = left of y-axis) |
| `bold` | `False` | If `True`, tick labels and panel labels use bold font weight |
| `minor` | `True` | If `True`, enables minor ticks with `AutoMinorLocator(2)` |
| `secondary_yaxis` | `None` | List of indices specifying which panels get a secondary right y-axis (see below) |

---

## `secondary_yaxis` Argument

- **Single panel:** pass any truthy value (e.g., `True`). Returns the secondary `Axes` directly as `sec_ax_list`.
- **1D grid:** pass a list of integer indices, e.g., `[0, 2]`. Returns a flat list where indexed positions hold the secondary `Axes`.
- **2D grid:** pass a list of `(row, col)` tuples, e.g., `[(0, 1), (1, 0)]`. Returns a 2D list where indexed positions hold the secondary `Axes`.

---

## Conventions Used in This Project

All notebooks call `gengrid` with positional `n_cols, n_rows` and keyword `size_inches` (and optionally `ticklabel_size`). The third return value is almost always `_`:

```python
# Single panel (most common — concentration or pH profile)
fig, ax, _ = gengrid(1, 1, size_inches=(3.25, 2.5))

# 2×2 grid (comparison plots)
fig, axes, _ = gengrid(2, 2, size_inches=(6.5, 6.25), ticklabel_size=7)

# 1×2 grid (side-by-side panels)
fig, axes, _ = gengrid(1, 2, size_inches=(6.5, 3.0), ticklabel_size=7)
```

### Recommended `size_inches` by layout

| Layout | `size_inches` | Notes |
|---|---|---|
| 1×1 | `(3.25, 2.5)` | Single column figure width |
| 2×2 | `(6.5, 6.25)` | Double column, square-ish |
| 1×2 | `(6.5, 3.0)` | Double column, short |

---

## Available Color Palettes

These are constructed at import time and available as module-level variables:

| Name | Description |
|---|---|
| `tableau20` | 20-color Tableau palette |
| `rainbow_1` / `j_i_pal` | 12-color custom warm/cool palette |
| `rainbow_2` / `barplot` | 8-color high-contrast palette |
| `rainbow_4` | 10-color LEGO-themed palette (requires Excel file) |
| `PrismaColor` | PrismaColor pencil-inspired palette (requires Excel file) |
| `Pantone` | Full Pantone palette (requires Excel file) |
| `Pantone_Blue` | 7-step blue Pantone ramp |
| `Pantone_Red` | 7-step red Pantone ramp |
| `Pantone_Red_Blue` | 14-step red-to-blue diverging Pantone ramp |
| `cool_sequential` | 9-step blue-green sequential |
| `warm_sequential` | 9-step yellow-red sequential |

All palettes are seaborn `color_palette` objects and can be used directly in `sns` or `matplotlib` calls.

---

## Minimal Working Example

```python
import matplotlib.pyplot as plt
from customplot import gengrid, tableau20

# Single panel
fig, ax, _ = gengrid(1, 1, size_inches=(3.25, 2.5))
ax.plot([0, 1, 2], [0, 1, 4], color=tableau20[0])
ax.set_xlabel("x", fontsize=8)
ax.set_ylabel("y", fontsize=8)
fig.tight_layout()
fig.savefig("output.svg")

# 2×2 grid
fig, axes, _ = gengrid(2, 2, size_inches=(6.5, 6.25), ticklabel_size=7)
for i in range(2):
    for j in range(2):
        axes[i][j].plot([0, 1], [0, 1], color=tableau20[i*2+j])
fig.tight_layout()
fig.savefig("grid_output.svg")
```

---

## Notes for Claude

- `customplot.py` lives at the root of the `co2_reduction/` project directory.
- It **must** be imported from that directory (or with its path on `sys.path`) because it opens `Color Swatches from UC Berkeley.xlsx` using a relative path.
- The module sets `mpl.use('svg')` on import — figures will be SVG-backend by default.
- `gengrid` is the only function you need for creating figures in this project. Do not use bare `plt.subplots` — always go through `gengrid` to maintain consistent styling.
- Panel labels (a), b), …) are generated automatically when `genlabels=True` (the default). Do not add them manually.
