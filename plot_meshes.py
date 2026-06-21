"""
Schematic mesh graphics for the chapter / slides.

Generates three publication-style figures (matched to the project's customplot
house style, 300 dpi). Each panel is drawn by a reusable `_draw_*` helper so the
standalone and combined figures stay in sync.

    mesh_1d.png       a) generic 1D cell-centred finite-volume mesh
                      b) the model's GDL + CL layout (CL finer)

    mesh_2d_tri.png   a) generic 2D triangular finite-volume mesh
                      b) the model's GDL + CL layout (discrete colours, CL finer,
                         boundary-layer refinement)

    mesh_combined.png 2x2: 1D (top row) and 2D (bottom row); generic mesh on the
                      left, GDL/CL model layout on the right.

These are illustrative — the solver itself is 1D finite volume; the 2D meshes
show what a 2D extension would look like.

Run:
    python plot_meshes.py
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.rcParams["savefig.dpi"] = 300

# customplot sets the SVG backend, Lato font and palettes on import; switch to
# a headless Agg backend afterwards so this runs cleanly anywhere.
from customplot import gengrid, cool_sequential, warm_sequential
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap, to_rgba

from params import Params

# ── Palette picks ─────────────────────────────────────────────────────────────
BLUE      = cool_sequential[5]
BLUE_DK   = cool_sequential[8]
ORANGE    = warm_sequential[4]
ORANGE_DK = warm_sequential[7]
RED       = warm_sequential[8]
GREY      = "0.35"


# ── Small helpers ─────────────────────────────────────────────────────────────

def _save(fig, png_path):
    """Save both a PNG (preview) and an SVG (vector, editable) of the figure."""
    svg_path = png_path.rsplit(".", 1)[0] + ".svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"  Saved: {png_path}  +  {svg_path}")


def _tint(color, f):
    """Light tint of a base colour: blend f*colour + (1-f)*white."""
    r, g, b = to_rgba(color)[:3]
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def _region_label(ax, xc, text, color, yc=0.90):
    """GDL/CL label placed inside the domain (top), with a soft white halo."""
    ax.text(xc, yc, text, ha="center", va="center", fontsize=9, color=color,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))


def _annot(ax, text, xy, xytext, color):
    """Leader-line annotation with a soft white halo (for the generic mesh)."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=7, color=color,
                ha="center", va="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.85))


def _rect_triangulation(xs, ys):
    """Delaunay triangulation of the tensor grid xs × ys."""
    X, Y = np.meshgrid(xs, ys)
    x, y = X.ravel(), Y.ravel()
    return mtri.Triangulation(x, y), x, y


def _cluster_both_ends(a, b, n):
    """n nodes in [a, b] clustered toward BOTH ends (cosine / boundary-layer)."""
    t = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n)))
    return a + (b - a) * t


def _gdl_cl_mesh(x_if=1.4, x_end=2.0, ny=9):
    """
    Two-region GDL+CL triangulation with boundary-layer refinement.

    Both regions cluster cells toward their boundaries (outer face + the
    GDL/CL interface); the CL carries more nodes over a shorter width, so it
    is the finer mesh.  Returns the triangulation, a per-triangle region flag
    (0 = GDL, 1 = CL), the node coordinates and the interface location.
    """
    xg = _cluster_both_ends(0.0,  x_if,   8)     # GDL: coarse
    xc = _cluster_both_ends(x_if, x_end, 14)     # CL : finer
    xs = np.unique(np.concatenate([xg, xc]))
    ys = _cluster_both_ends(0.0, 1.0, ny)        # boundary layers at the walls
    tri, x, y = _rect_triangulation(xs, ys)
    xc_tri  = x[tri.triangles].mean(axis=1)
    region  = (xc_tri >= x_if).astype(float)     # 0 GDL, 1 CL
    return tri, region, x, y, x_if


# ── Panel drawers (each draws onto a supplied Axes) ───────────────────────────

def _draw_1d_generic(ax):
    """1D cell-centred finite-volume mesh: centres, faces, CV, Delta x."""
    N      = 5
    faces  = np.arange(N + 1, dtype=float)
    cents  = faces[:-1] + 0.5

    ax.plot([0, N], [0, 0], color=GREY, lw=1.6, zorder=1, solid_capstyle="round")
    for f in faces:
        ax.plot([f, f], [-0.22, 0.22], color="0.55", lw=1.3, zorder=2)

    i = 2
    ax.fill_between([faces[i], faces[i + 1]], -0.22, 0.22,
                    color=BLUE, alpha=0.16, lw=0, zorder=1)
    ax.plot(cents, np.zeros_like(cents), "o", ms=10, color=BLUE,
            mec="white", mew=1.1, zorder=4)

    for off, name in [(-1, "i-1"), (0, "i"), (1, "i+1")]:
        ax.text(cents[i + off], -0.36, fr"$x_{{{name}}}$",
                ha="center", va="top", fontsize=8)

    ax.annotate("", xy=(cents[i + 1], 0.55), xytext=(cents[i], 0.55),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3))
    ax.text(0.5 * (cents[i] + cents[i + 1]), 0.62, r"$\Delta x$",
            ha="center", va="bottom", fontsize=8, color=RED)

    ax.annotate("", xy=(0.5, -0.62), xytext=(0.0, -0.62),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.0))
    ax.text(0.25, -0.84, r"$\Delta x/2$", ha="center", va="top",
            fontsize=7, color="0.3")

    ax.annotate("faces", xy=(faces[i], 0.22), xytext=(1.0, 1.02),
                ha="center", fontsize=7, color="0.4",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8))
    ax.text(cents[i], 0.30, "control\nvolume $i$", ha="center", va="bottom",
            fontsize=7, color=BLUE_DK)

    ax.set_xlim(-0.7, N + 0.7)
    ax.set_ylim(-1.05, 1.25)
    ax.set_title("Cell-centred finite-volume mesh (1D)", fontsize=9)
    ax.axis("off")


def _draw_1d_layout(ax, p):
    """1D model layout: coarse GDL band + finer CL band with boundary labels."""
    xg0, xg1 = 0.0, 7.0
    xc0, xc1 = 7.0, 11.0

    ax.fill_between([xg0, xg1], -0.3, 0.3, color=BLUE,   alpha=0.13, lw=0)
    ax.fill_between([xc0, xc1], -0.3, 0.3, color=ORANGE, alpha=0.16, lw=0)
    ax.plot([xg0, xc1], [0, 0], color=GREY, lw=1.6, solid_capstyle="round")

    gx = np.linspace(xg0 + 0.45, xg1 - 0.45, 8)
    cx = np.linspace(xc0 + 0.22, xc1 - 0.22, 13)
    ax.plot(gx, 0 * gx, "o", ms=6.5, color=BLUE,   mec="white", mew=0.8, zorder=3)
    ax.plot(cx, 0 * cx, "o", ms=5.0, color=ORANGE, mec="white", mew=0.7, zorder=3)

    ax.plot([xg1, xg1], [-0.5, 0.55], ls="--", color="0.4", lw=1.3)

    ax.text((xg0 + xg1) / 2, 0.50, "GDL", ha="center", fontsize=9, color=BLUE_DK)
    ax.text((xg0 + xg1) / 2, 0.36,
            fr"$N_{{GDL}}={p.N_GDL}$,  {p.L_GDL*1e6:.0f} $\mu$m",
            ha="center", fontsize=7, color=BLUE_DK)
    ax.text((xc0 + xc1) / 2, 0.50, "CL", ha="center", fontsize=9, color=ORANGE_DK)
    ax.text((xc0 + xc1) / 2, 0.36,
            fr"$N_{{CL}}={p.N_CL}$,  {p.L_CL*1e6:.0f} $\mu$m",
            ha="center", fontsize=7, color=ORANGE_DK)

    ax.text(xg0, -0.55, "gas channel\n$x=0$", ha="center", va="top", fontsize=7)
    ax.text(xg1, -0.62, "GDL/CL", ha="center", va="top", fontsize=7, color="0.3")
    ax.text(xc1, -0.55, "membrane\n$x=L$", ha="center", va="top", fontsize=7)

    ax.set_xlim(-1.2, 12.2)
    ax.set_ylim(-1.05, 0.9)
    ax.set_title("Model layout: gas-diffusion + catalyst layers", fontsize=9)
    ax.axis("off")


def _draw_2d_generic(ax):
    """Generic 2D triangular FV mesh: node / element (CV) / edge / size h."""
    tri_u, xu, yu = _rect_triangulation(np.linspace(0, 2, 7), np.linspace(0, 1, 4))
    ax.triplot(tri_u, color="0.45", lw=0.9, zorder=2)
    ax.plot(xu, yu, "o", ms=5, color=BLUE, mec="white", mew=0.6, zorder=3)

    cxt = xu[tri_u.triangles].mean(axis=1)
    cyt = yu[tri_u.triangles].mean(axis=1)
    k   = int(np.argmin((cxt - 0.95) ** 2 + (cyt - 0.62) ** 2))
    vt  = tri_u.triangles[k]
    ax.fill(xu[vt], yu[vt], color=BLUE, alpha=0.22, lw=0, zorder=1.5)

    lo = vt[np.argsort(yu[vt])][:2]
    ax.annotate("", xy=(xu[lo[1]], yu[lo[1]]), xytext=(xu[lo[0]], yu[lo[0]]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2), zorder=5)
    ax.text(xu[lo].mean(), yu[lo].mean() - 0.085, "$h$", color=RED,
            fontsize=8, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    _annot(ax, "control\nvolume", (cxt[k], cyt[k]), (cxt[k] - 0.02, 0.90), BLUE_DK)
    ni = int(np.argmin((xu - 1.0) ** 2 + (yu - 1.0 / 3.0) ** 2))
    _annot(ax, "node", (xu[ni], yu[ni]), (xu[ni] + 0.34, 0.13), "0.25")
    em = (0.5 * (xu[vt[0]] + xu[vt[1]]), 0.5 * (yu[vt[0]] + yu[vt[1]]))
    _annot(ax, "edge\n(face)", em, (em[0] - 0.46, 0.15), "0.25")

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Triangular finite-volume mesh (2D)", fontsize=9, pad=8)


def _draw_2d_layout(ax):
    """2D model layout: GDL + CL two-region mesh, CL finer, BL-refined."""
    region_cmap = ListedColormap([_tint(BLUE, 0.28), _tint(ORANGE, 0.38)])
    tri_g, region, xg, yg, x_if = _gdl_cl_mesh()

    ax.tripcolor(tri_g, facecolors=region, cmap=region_cmap,
                 vmin=0, vmax=1, edgecolors="none", zorder=1)
    ax.triplot(tri_g, color="0.4", lw=0.7, zorder=2)
    ax.plot([x_if, x_if], [0, 1], ls="--", color="0.3", lw=1.4, zorder=4)

    _region_label(ax, 0.5 * x_if, "GDL", BLUE_DK)
    _region_label(ax, 0.5 * (x_if + 2.0), "CL", ORANGE_DK)
    ax.text(x_if, -0.10, "GDL/CL interface", ha="center", va="top",
            fontsize=7, color="0.3")

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Model layout: gas-diffusion + catalyst layers",
                 fontsize=9, pad=8)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_mesh_1d(p, save_path="mesh_1d.png"):
    fig, axes, _ = gengrid(1, 2, size_inches=(6.5, 4.3), ticklabel_size=8)
    _draw_1d_generic(axes[0])
    _draw_1d_layout(axes[1], p)
    fig.tight_layout(h_pad=1.6)
    _save(fig, save_path)
    plt.close(fig)


def plot_mesh_2d(save_path="mesh_2d_tri.png"):
    fig, axes, _ = gengrid(1, 2, size_inches=(5.8, 5.6), ticklabel_size=8)
    _draw_2d_generic(axes[0])
    _draw_2d_layout(axes[1])
    fig.tight_layout(h_pad=2.0)
    _save(fig, save_path)
    plt.close(fig)


def plot_mesh_combined(p, save_path="mesh_combined.png"):
    # 2x2: rows = dimensionality (1D top, 2D bottom);
    #      cols = generic mesh (left) vs GDL/CL model layout (right).
    fig, axes, _ = gengrid(2, 2, size_inches=(8.5, 6.2), ticklabel_size=8)
    _draw_1d_generic(axes[0][0])
    _draw_1d_layout(axes[0][1], p)
    _draw_2d_generic(axes[1][0])
    _draw_2d_layout(axes[1][1])
    fig.tight_layout(h_pad=2.2, w_pad=2.4)
    _save(fig, save_path)
    plt.close(fig)


def main():
    p = Params()
    print("  Generating mesh schematics ...")
    plot_mesh_1d(p)
    plot_mesh_2d()
    plot_mesh_combined(p)
    print("  Done.")


if __name__ == "__main__":
    main()
