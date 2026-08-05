"""
Schematic mesh graphics for the chapter / slides.

Generates four publication-style figures (book house style per
Figure_Preparation_Guide.pdf, 600 dpi -- fonts and panel labels only; these
are illustrative 3D/isometric schematics, exempted from the guide's anti-3D
and discrete-dataset color rules, see the import comment below). Each panel
is drawn by a reusable `_draw_*` helper so the standalone and combined
figures stay in sync.

    mesh_1d.png       a) generic 1D mesh (centres, faces, cell, dx)
                      b) the model's GDL + CL layout (CL finer)

    mesh_2d_tri.png   a) generic 2D triangular mesh (node / cell / edge / size h)
                      b) model layout: flow channel + GDL + CL (CL finest,
                         boundary-layer refinement)

    mesh_3d.png       a) generic 3D tetrahedral mesh (node / face / cell)
                      b) model layout: serpentine flow channel over GDL/CL

    mesh_combined.png 3x2: 1D / 2D / 3D rows; generic mesh (left) vs model
                      layout (right).

Tool-neutral element types (triangular in 2D, tetrahedral in 3D) mirror what a
general-purpose mesher such as COMSOL produces by default.  These figures are
illustrative — the solver itself is 1D finite volume.

Run:
    python plot_meshes.py
"""
from __future__ import annotations
from itertools import combinations
import numpy as np

# customplot sets the SVG backend and its own font/palette on import; only the
# two illustrative sequential palettes are still used below (region colors,
# not "discrete dataset" colors, so the book palette doesn't apply to them --
# see Figure_Preparation_Guide.pdf Section 5.1 vs these schematic diagrams).
# book_style, imported after, overrides the font/rcParams customplot set.
from customplot import cool_sequential, warm_sequential
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import book_style  # noqa: F401  (import for its rcParams.update() side effect)
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Polygon, Ellipse, FancyArrowPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import Delaunay

from params import Params

# ── Palette picks ─────────────────────────────────────────────────────────────
BLUE      = cool_sequential[5]
BLUE_DK   = cool_sequential[8]
ORANGE    = warm_sequential[4]
ORANGE_DK = warm_sequential[7]
RED       = warm_sequential[8]
GREY      = "0.35"
CHANNEL   = (0.86, 0.89, 0.90)   # neutral grey-blue for the open flow channel
CHANNEL_DK = "0.35"


# ── Small helpers ─────────────────────────────────────────────────────────────

def _save(fig, png_path):
    """Save a PNG (preview), an SVG (vector, editable), and a PDF (Section 10's
    preferred delivery format) of the figure."""
    svg_path = png_path.rsplit(".", 1)[0] + ".svg"
    pdf_path = png_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {png_path}  +  {svg_path}  +  {pdf_path}")


def _tint(color, f):
    """Light tint of a base colour: blend f*colour + (1-f)*white."""
    r, g, b = to_rgba(color)[:3]
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def _region_label(ax, xc, text, color, yc=0.90, fs=9):
    """Region label placed inside the domain (top), with a soft white halo."""
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color,
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


def _channel_gdl_cl_mesh(x_ch=0.55, x_if=1.55, x_end=2.0, ny=9):
    """
    Three-region channel + GDL + CL triangulation with boundary-layer
    refinement.  Region flag per triangle: 0 = flow channel, 1 = GDL, 2 = CL.
    The CL carries the most nodes over the shortest width (finest mesh).
    """
    xch = _cluster_both_ends(0.0,  x_ch,   5)     # open flow channel (coarse)
    xg  = _cluster_both_ends(x_ch, x_if,   7)     # GDL
    xc  = _cluster_both_ends(x_if, x_end, 12)     # CL (finest)
    xs  = np.unique(np.concatenate([xch, xg, xc]))
    ys  = _cluster_both_ends(0.0, 1.0, ny)
    tri, x, y = _rect_triangulation(xs, ys)
    xc_tri = x[tri.triangles].mean(axis=1)
    region = np.zeros(len(xc_tri))
    region[xc_tri >= x_ch] = 1
    region[xc_tri >= x_if] = 2
    return tri, region, x, y, x_ch, x_if


# ── Panel drawers (each draws onto a supplied Axes) ───────────────────────────

def _draw_1d_generic(ax):
    """1D mesh: cell centres, faces, one highlighted cell, Delta x."""
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

    ax.annotate("", xy=(0.5, -0.46), xytext=(0.0, -0.46),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.0))
    ax.text(0.25, -0.60, r"$\Delta x/2$", ha="center", va="top",
            fontsize=7, color="0.3")

    ax.annotate("faces", xy=(faces[i], 0.22), xytext=(1.0, 1.02),
                ha="center", fontsize=7, color="0.4",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8))
    ax.text(cents[i], 0.30, "cell $i$", ha="center", va="bottom",
            fontsize=7, color=BLUE_DK)

    ax.set_xlim(-0.7, N + 0.7)
    ax.set_ylim(-0.72, 1.25)
    ax.set_title("Mesh (1D)", fontsize=8)
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
    ax.set_title("Model layout: gas-diffusion + catalyst layers", fontsize=8)
    ax.axis("off")


def _draw_2d_generic(ax):
    """Generic 2D triangular mesh: node / cell / edge / element size h."""
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

    _annot(ax, "cell", (cxt[k], cyt[k]), (cxt[k] - 0.02, 0.90), BLUE_DK)
    ni = int(np.argmin((xu - 1.0) ** 2 + (yu - 1.0 / 3.0) ** 2))
    _annot(ax, "node", (xu[ni], yu[ni]), (xu[ni] + 0.34, 0.13), "0.25")
    em = (0.5 * (xu[vt[0]] + xu[vt[1]]), 0.5 * (yu[vt[0]] + yu[vt[1]]))
    _annot(ax, "edge", em, (em[0] - 0.42, 0.15), "0.25")

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Triangular mesh (2D)", fontsize=8, pad=8)


def _draw_2d_layout(ax):
    """2D model layout: flow channel + GDL + CL (three meshed rectangles)."""
    tri, region, x, y, x_ch, x_if = _channel_gdl_cl_mesh()
    region_cmap = ListedColormap([CHANNEL, _tint(BLUE, 0.28), _tint(ORANGE, 0.38)])

    ax.tripcolor(tri, facecolors=region, cmap=region_cmap,
                 vmin=0, vmax=2, edgecolors="none", zorder=1)
    ax.triplot(tri, color="0.4", lw=0.6, zorder=2)
    for xb in (x_ch, x_if):
        ax.plot([xb, xb], [0, 1], ls="--", color="0.3", lw=1.2, zorder=4)

    # flow direction arrow along the open channel
    ax.annotate("", xy=(0.5 * x_ch, 0.72), xytext=(0.5 * x_ch, 0.16),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.6), zorder=5)

    _region_label(ax, 0.5 * x_ch, "flow\nchannel", CHANNEL_DK, fs=7)
    _region_label(ax, 0.5 * (x_ch + x_if), "GDL", BLUE_DK)
    _region_label(ax, 0.5 * (x_if + 2.0), "CL", ORANGE_DK)

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Model layout: flow channel + GDL/CL", fontsize=8, pad=8)


# ── 3D helpers ────────────────────────────────────────────────────────────────

def _box_faces(x0, x1, y0, y1, z0, z1):
    """The six quad faces of an axis-aligned box (for Poly3DCollection)."""
    c = lambda x, y, z: (x, y, z)
    return [
        [c(x0, y0, z0), c(x1, y0, z0), c(x1, y1, z0), c(x0, y1, z0)],  # z = z0
        [c(x0, y0, z1), c(x1, y0, z1), c(x1, y1, z1), c(x0, y1, z1)],  # z = z1
        [c(x0, y0, z0), c(x1, y0, z0), c(x1, y0, z1), c(x0, y0, z1)],  # y = y0
        [c(x0, y1, z0), c(x1, y1, z0), c(x1, y1, z1), c(x0, y1, z1)],  # y = y1
        [c(x0, y0, z0), c(x0, y1, z0), c(x0, y1, z1), c(x0, y0, z1)],  # x = x0
        [c(x1, y0, z0), c(x1, y1, z0), c(x1, y1, z1), c(x1, y0, z1)],  # x = x1
    ]


def _setup_3d(ax, xmax=2.0, ymax=1.0, zmax=1.0, elev=20, azim=-58):
    """Common 3D axes styling: fixed view, equal-ish box, no panes/ticks."""
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((xmax, ymax, zmax))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _t3(ax, x, y, z, text, color, fs=7, ha="center"):
    """3D text label with a soft white halo so it reads over the mesh."""
    ax.text(x, y, z, text, color=color, fontsize=fs, ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


class _Arrow3D(FancyArrowPatch):
    """A FancyArrowPatch that lives in 3D data space (for leader arrows)."""

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxyz = (dx, dy, dz)

    def _project(self):
        x, y, z = self._xyz
        dx, dy, dz = self._dxyz
        xs, ys, _ = proj_transform((x, x + dx), (y, y + dy), (z, z + dz), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return xs, ys

    def do_3d_projection(self, renderer=None):
        self._project()
        # Return a large depth so leader arrows always draw on top of the mesh.
        return 1e4

    def draw(self, renderer):
        self._project()
        super().draw(renderer)


def _leader3d(ax, txy, target, text, color, fs=7, style="-|>"):
    """3D leader-line label: arrow from the text anchor to a feature point."""
    tx, ty, tz = txy
    gx, gy, gz = target
    ax.add_artist(_Arrow3D(tx, ty, tz, gx - tx, gy - ty, gz - tz,
                           arrowstyle=style, color=color, lw=0.9,
                           mutation_scale=8, zorder=10))
    ax.text(tx, ty, tz, text, color=color, fontsize=fs, ha="center", va="center",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))


def _tri_surface_z(ax, x0, x1, y0, y1, nx, ny, z, color="0.5", lw=0.4):
    """Draw a triangulated (tetra-style) surface patch at constant z."""
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    tri = mtri.Triangulation(P[:, 0], P[:, 1])
    segs = [[(P[a, 0], P[a, 1], z), (P[b, 0], P[b, 1], z)] for a, b in tri.edges]
    ax.add_collection3d(Line3DCollection(segs, colors=color, lw=lw))


def _tri_surface_y(ax, x0, x1, z0, z1, y, nx, nz, color="0.5", lw=0.4):
    """Triangulated patch on a constant-y (x-z) face — a meshed cross-section."""
    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(z0, z1, nz)
    X, Z = np.meshgrid(xs, zs)
    P = np.column_stack([X.ravel(), Z.ravel()])
    tri = mtri.Triangulation(P[:, 0], P[:, 1])
    segs = [[(P[a, 0], y, P[a, 1]), (P[b, 0], y, P[b, 1])] for a, b in tri.edges]
    ax.add_collection3d(Line3DCollection(segs, colors=color, lw=lw))


def _draw_3d_generic(ax):
    """Generic 3D tetrahedral mesh: node / face / cell (one tet highlighted)."""
    # node cloud: a structured box shell + two interior points -> irregular tets
    xs = np.linspace(0, 2, 4)
    ys = np.linspace(0, 1, 2)
    zs = np.linspace(0, 1, 2)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts = np.vstack([pts, [[0.70, 0.5, 0.5], [1.35, 0.5, 0.55]]])

    tets = Delaunay(pts).simplices

    # unique tetra edges
    eset = set()
    for t in tets:
        for a, b in combinations(t, 2):
            eset.add((min(a, b), max(a, b)))
    segs = [[tuple(pts[a]), tuple(pts[b])] for a, b in eset]
    ax.add_collection3d(Line3DCollection(segs, colors="0.55", lw=0.6))

    # highlight one near-central tetrahedron
    cents = pts[tets].mean(axis=1)
    k  = int(np.argmin(((cents - [1.0, 0.5, 0.5]) ** 2).sum(axis=1)))
    tv = tets[k]
    tet_faces = [[tuple(pts[tv[i]]) for i in f]
                 for f in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]]
    ax.add_collection3d(Poly3DCollection(tet_faces, facecolor=_tint(BLUE, 0.45),
                                         edgecolor=BLUE_DK, lw=0.8, alpha=0.9))

    cc = cents[k]
    pv = pts[tv]                                  # the 4 tet vertices

    # cell (element): leader from clear space above to the tet centroid
    _leader3d(ax, (cc[0], cc[1], 1.30), (cc[0], cc[1], cc[2] + 0.06),
              "cell\n(element)", BLUE_DK)

    # node: leader to a visible box corner (a mesh vertex)
    ni = int(np.argmin(((pts - [0, 0, 1]) ** 2).sum(axis=1)))
    nd = pts[ni]
    ax.scatter([nd[0]], [nd[1]], [nd[2]], color=BLUE, s=22, depthshade=False)
    _leader3d(ax, (nd[0] + 0.12, nd[1], nd[2] + 0.36), tuple(nd), "node", "0.25")

    # edge: leader to the midpoint of a tet edge
    em = 0.5 * (pv[0] + pv[1])
    _leader3d(ax, (em[0] - 0.5, em[1] - 0.1, em[2] - 0.35), tuple(em), "edge", "0.25")

    # h: element-size double arrow along another tet edge (mirrors the 2D panel)
    ha_, hb_ = pv[2], pv[3]
    ax.add_artist(_Arrow3D(ha_[0], ha_[1], ha_[2],
                           hb_[0] - ha_[0], hb_[1] - ha_[1], hb_[2] - ha_[2],
                           arrowstyle="<|-|>", color=RED, lw=1.1,
                           mutation_scale=7, zorder=10))
    hm = 0.5 * (ha_ + hb_)
    ax.text(hm[0] + 0.12, hm[1], hm[2], "$h$", color=RED, fontsize=8, zorder=11,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

    _setup_3d(ax)
    ax.set_title("Tetrahedral mesh (3D)", fontsize=8, pad=2)


def _serpentine_boxes(x0, x1, y0, y1, n_pass=7, inset=0.22):
    """
    Axis-aligned (xa,xb,ya,yb) boxes tracing ONE continuous serpentine channel:
    n_pass straight passes along x, stacked in y, joined end-to-end by short
    connectors at alternating ends (left/right) so it forms a single snake.
    """
    ys = np.linspace(y0 + inset, y1 - inset, n_pass)
    pitch = ys[1] - ys[0]
    w  = 0.5 * pitch                                  # channel width (≈ half-pitch)
    xa, xb = x0 + inset, x1 - inset
    boxes = [(xa, xb, yj - w / 2, yj + w / 2) for yj in ys]      # straight passes
    for j in range(n_pass - 1):                                  # U-turn connectors
        if j % 2 == 0:                       # turn at the right edge (flush)
            cx0, cx1 = xb - w, xb
        else:                                # turn at the left edge (flush)
            cx0, cx1 = xa, xa + w
        boxes.append((cx0, cx1, ys[j] - w / 2, ys[j + 1] + w / 2))
    return boxes, ys, w, xa, xb


# ── Isometric MEA layout (panel b) ────────────────────────────────────────────
# Hand-drawn 2D isometric illustration (full painter's-order control) of a
# serpentine flow field on the membrane-electrode assembly, mirroring the
# canonical fuel-cell schematic.

_ICA, _ISA = np.cos(np.radians(30)), np.sin(np.radians(30))

# MEA layer colours
_C_BLUE   = (0.16, 0.67, 0.95)   # flow field
_C_WHITE  = (0.95, 0.95, 0.95)   # GDL / FF plate
_C_ORANGE = (0.96, 0.52, 0.10)   # catalyst layers
_C_TEAL   = (0.12, 0.72, 0.64)   # membrane


def _iso(x, y, z):
    """Isometric projection -> (X, Y) screen coordinates."""
    return ((x - y) * _ICA, (x + y) * _ISA + z)


def _shade(c, f):
    r, g, b = to_rgba(c)[:3]
    return (min(1, r * f), min(1, g * f), min(1, b * f))


def _poly_iso(ax, pts, fc, ec="0.35", lw=0.5, z=1, off=(0.0, 0.0)):
    xy = [(_iso(*p)[0] + off[0], _iso(*p)[1] + off[1]) for p in pts]
    ax.add_patch(Polygon(xy, closed=True, facecolor=fc, edgecolor=ec,
                         lw=lw, zorder=z, joinstyle="round"))


def _box_iso(ax, x0, x1, y0, y1, z0, z1, top, ec="0.35", lw=0.5, z=1, off=(0, 0)):
    """Three visible faces (left x0, front y0, top) of an isometric box."""
    _poly_iso(ax, [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
              _shade(top, 0.88), ec, lw, z, off)                       # left
    _poly_iso(ax, [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
              _shade(top, 0.76), ec, lw, z, off)                       # front
    _poly_iso(ax, [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
              top, ec, lw, z + 0.05, off)                              # top


def _quad_mesh_segs(A, B, C, D, off=(0, 0), step=0.2):
    """Triangulated-surface line segments (in iso screen coords) for a quad."""
    A, B, C, D = (np.asarray(P, float) for P in (A, B, C, D))
    nu = max(1, int(round(np.linalg.norm(B - A) / step)))
    nv = max(1, int(round(np.linalg.norm(D - A) / step)))

    def S(P):
        X, Y = _iso(*P)
        return (X + off[0], Y + off[1])

    g = [[S((1 - u / nu) * ((1 - v / nv) * A + (v / nv) * D)
            + (u / nu) * ((1 - v / nv) * B + (v / nv) * C))
          for v in range(nv + 1)] for u in range(nu + 1)]
    segs = []
    for u in range(nu + 1):
        for v in range(nv):
            segs.append([g[u][v], g[u][v + 1]])
    for v in range(nv + 1):
        for u in range(nu):
            segs.append([g[u][v], g[u + 1][v]])
    for u in range(nu):                                  # diagonals -> triangles
        for v in range(nv):
            segs.append([g[u][v], g[u + 1][v + 1]])
    return segs


def _offset_polyline(cl, d, side):
    """Offset a rectilinear centreline by distance d (side = +1 left / -1 right)."""
    cl = [np.asarray(p, float) for p in cl]
    nrm = []
    for i in range(len(cl) - 1):
        dv = cl[i + 1] - cl[i]
        dv = dv / np.linalg.norm(dv)
        nrm.append(np.array([-dv[1], dv[0]]) * side)        # 90deg normal
    pts = [cl[0] + d * nrm[0]]
    for i in range(1, len(cl) - 1):                          # miter at 90deg turns
        pts.append(cl[i] + d * (nrm[i - 1] + nrm[i]))
    pts.append(cl[-1] + d * nrm[-1])
    return pts


def _box_mesh(ax, x0, x1, y0, y1, z0, z1, top, ec, mcolor, z, off=(0, 0), step=0.2):
    """Isometric box with each visible face carrying a triangulated mesh."""
    faces = [
        ([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], _shade(top, 0.88)),
        ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], _shade(top, 0.76)),
        ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], top),
    ]
    segs = []
    for corners, fc in faces:
        _poly_iso(ax, corners, fc, ec, 0.5, z, off)
        segs += _quad_mesh_segs(*corners, off=off, step=step)
    ax.add_collection(LineCollection(segs, colors=[mcolor], linewidths=0.3,
                                     zorder=z + 0.1))


def _draw_mea_layout(ax, callout=True):
    """Isometric serpentine flow field on the layered MEA (panel b)."""
    Lx = Ly = 3.0

    # layer stack, bottom -> top: (label, thickness, colour)
    layers = [("A-FF", 0.16, _C_WHITE), ("A-GDL", 0.06, _C_WHITE),
              ("A-CL", 0.045, _C_ORANGE), ("MEM", 0.05, _C_TEAL),
              ("C-CL", 0.045, _C_ORANGE), ("C-GDL", 0.06, _C_WHITE)]
    z = 0.0
    lv = []
    for name, t, c in layers:
        lv.append((name, z, z + t, c)); z += t
    z_top = z

    ec = "0.4"
    # stacked layer side faces (front y=0 and left x=0) -> coloured edge bands,
    # each carrying a triangulated tet-style surface mesh through the thickness
    side_segs = []
    for name, z0, z1, c in lv:
        f_front = [(0, 0, z0), (Lx, 0, z0), (Lx, 0, z1), (0, 0, z1)]
        f_left  = [(0, 0, z0), (0, Ly, z0), (0, Ly, z1), (0, 0, z1)]
        _poly_iso(ax, f_front, _shade(c, 0.76), ec, 0.5, z=2)
        _poly_iso(ax, f_left,  _shade(c, 0.88), ec, 0.5, z=2)
        side_segs += _quad_mesh_segs(*f_front, step=0.24)
        side_segs += _quad_mesh_segs(*f_left, step=0.24)
    ax.add_collection(LineCollection(side_segs, colors=["0.45"], linewidths=0.25,
                                     zorder=2.3))
    # platform top (C-GDL upper surface) — triangulated tet-style mesh
    plat = [(0, 0, z_top), (Lx, 0, z_top), (Lx, Ly, z_top), (0, Ly, z_top)]
    _poly_iso(ax, plat, _C_WHITE, ec, 0.5, z=3)
    ax.add_collection(LineCollection(_quad_mesh_segs(*plat, step=0.24),
                                     colors=["0.62"], linewidths=0.3, zorder=3.1))

    # ONE continuous serpentine flow field, built as a single extruded ribbon
    # (no internal box edges) so it reads as one connected channel.
    boxes, ys, w, xa, xb = _serpentine_boxes(0, Lx, 0, Ly, n_pass=7, inset=0.32)
    h  = 0.12
    ce = (0.04, 0.28, 0.52)          # channel edge
    mc = (0.10, 0.38, 0.64)          # channel surface-mesh colour

    # serpentine centreline (turns inset by w/2 so the ribbon spans [xa, xb])
    xl, xr = xa + w / 2, xb - w / 2
    cl = []
    for j, yj in enumerate(ys):
        cl += [(xl, yj), (xr, yj)] if j % 2 == 0 else [(xr, yj), (xl, yj)]
    outline = _offset_polyline(cl, w / 2, +1) + _offset_polyline(cl, w / 2, -1)[::-1]

    # extruded side walls (drawn first; the top face covers the hidden ones)
    wsegs = []
    for k in range(len(outline)):
        A, B = outline[k], outline[(k + 1) % len(outline)]
        wall = [(A[0], A[1], z_top + h), (B[0], B[1], z_top + h),
                (B[0], B[1], z_top), (A[0], A[1], z_top)]
        _poly_iso(ax, wall, _shade(_C_BLUE, 0.74), ec=ce, lw=0.3, z=5)
        wsegs += _quad_mesh_segs(*wall, step=0.2)
    ax.add_collection(LineCollection(wsegs, colors=[mc], linewidths=0.25, zorder=5.2))

    # continuous top face as one polygon; mesh is CLIPPED to it so it lines up
    # exactly with the ribbon ends (no overhanging tiles).
    top_xy = [_iso(P[0], P[1], z_top + h) for P in outline]
    top_patch = Polygon(top_xy, closed=True, facecolor=_C_BLUE, edgecolor=ce,
                        lw=0.7, zorder=6, joinstyle="round")
    ax.add_patch(top_patch)
    tsegs = []
    for (bxa, bxb, bya, byb) in boxes:
        tsegs += _quad_mesh_segs((bxa, bya, z_top + h), (bxb, bya, z_top + h),
                                 (bxb, byb, z_top + h), (bxa, byb, z_top + h), step=0.2)
    tlc = LineCollection(tsegs, colors=[mc], linewidths=0.3, zorder=6.2)
    tlc.set_clip_path(top_patch)
    ax.add_collection(tlc)

    # inlet flow arrow on the first pass
    p0 = _iso(xa, ys[0], z_top + h)
    p1 = _iso(xa + 0.7, ys[0], z_top + h)
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.6), zorder=9)
    ax.text(p0[0] - 0.12, p0[1] + 0.18, "flow", fontsize=7, color="0.15", ha="right")

    if callout:
        _draw_mea_callout(ax)
        ax.set_xlim(-3.0, 7.4)
    else:
        ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.5, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Model layout: serpentine flow field on the MEA",
                 fontsize=9, pad=4)


def _draw_mea_callout(ax):
    """Exploded, labelled layer stack to the right (C-FF … A-FF)."""
    tiles = [("C-FF", _C_BLUE), ("C-GDL", _C_WHITE), ("C-CL", _C_ORANGE),
             ("MEM", _C_TEAL), ("A-CL", _C_ORANGE), ("A-GDL", _C_WHITE),
             ("A-FF", _C_WHITE)]
    OX, OY0, gap = 4.0, 2.85, 0.40
    wf, hf = 1.0, 0.13
    x_lab = OX + 1.95

    for i, (name, c) in enumerate(tiles):
        oy = OY0 - i * gap
        _box_iso(ax, 0, wf, 0, wf, 0, hf, c, ec="0.35", lw=0.5,
                 z=8 + i * 0.01, off=(OX, oy))
        # leader line + label
        tip = (_iso(wf, 0, hf)[0] + OX + 0.05, _iso(wf, 0, 0.5 * hf)[1] + oy)
        ax.plot([tip[0], x_lab - 0.08], [tip[1], oy + 0.30], color="0.45", lw=0.7,
                zorder=9)
        ax.text(x_lab, oy + 0.30, name, fontsize=7.5, va="center", ha="left",
                color="0.15")

    # zoom indicator: ring on the plate + two lines to the callout
    ring = _iso(2.3, 0.35, 0.18)
    ax.add_patch(Ellipse(ring, 0.9, 0.5, angle=24, fill=False,
                         edgecolor="0.3", lw=0.9, zorder=9))
    top_tile = (_iso(0, wf, hf)[0] + OX, _iso(0, wf, hf)[1] + OY0)
    bot_tile = (_iso(0, wf, 0)[0] + OX, _iso(0, wf, 0)[1] + OY0 - 6 * gap)
    for tgt in (top_tile, bot_tile):
        ax.plot([ring[0] + 0.42, tgt[0]], [ring[1] + 0.05, tgt[1]],
                color="0.45", lw=0.7, zorder=8)


def _panel_label(ax, lab, is3d=False):
    """(a)/(b)/... label per Figure_Preparation_Guide.pdf Section 8.

    Placed just above the axes (matches book_style.panel_label) rather than
    inside it, so it can't collide with the region/leader labels these
    schematic panels draw near their own top edge. Uses identical
    axes-relative coordinates for 2D and 3D panels so the labels line up
    vertically by column (a/c/e and b/d/f). 3D axes need text2D (plain
    .text() on an Axes3D plots in 3D data space, not screen space).
    """
    fn = ax.text2D if is3d else ax.text
    fn(0.0, 1.02, f"({lab})", transform=ax.transAxes,
       ha="left", va="bottom", fontsize=9, fontweight="bold")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_mesh_1d(p, save_path="mesh_1d.png"):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.3), dpi=600)
    _draw_1d_generic(axes[0])
    _draw_1d_layout(axes[1], p)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    fig.tight_layout(h_pad=1.6)
    _save(fig, save_path)
    plt.close(fig)


def plot_mesh_2d(save_path="mesh_2d_tri.png"):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 5.6), dpi=600)
    _draw_2d_generic(axes[0])
    _draw_2d_layout(axes[1])
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    fig.tight_layout(h_pad=2.0)
    _save(fig, save_path)
    plt.close(fig)


def plot_mesh_3d(p, save_path="mesh_3d.png"):
    fig = plt.figure(figsize=(9.4, 3.9))
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig.add_subplot(1, 2, 2)
    _draw_3d_generic(ax_a)
    _draw_mea_layout(ax_b, callout=True)
    _panel_label(ax_a, "a", is3d=True)
    _panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02, wspace=0.02)
    _save(fig, save_path)
    plt.close(fig)


def plot_mesh_combined(p, save_path="mesh_combined.png"):
    # 3x2: rows = dimensionality (1D / 2D / 3D);
    #      cols = generic mesh (left) vs model layout (right).
    fig = plt.figure(figsize=(9.0, 7.6))
    gs  = fig.add_gridspec(3, 2)

    ax_a = fig.add_subplot(gs[0, 0]); _draw_1d_generic(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); _draw_1d_layout(ax_b, p)
    ax_c = fig.add_subplot(gs[1, 0]); _draw_2d_generic(ax_c)
    ax_d = fig.add_subplot(gs[1, 1]); _draw_2d_layout(ax_d)
    ax_e = fig.add_subplot(gs[2, 0], projection="3d"); _draw_3d_generic(ax_e)
    ax_f = fig.add_subplot(gs[2, 1]); _draw_mea_layout(ax_f, callout=False)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.02,
                        hspace=0.08, wspace=0.18)

    # Place panel labels at the GRIDSPEC cell corners so they line up by column
    # (a/c/e and b/d/f) regardless of 2D vs 3D axes insets.
    for (r, c, lab) in [(0, 0, "a"), (0, 1, "b"), (1, 0, "c"),
                        (1, 1, "d"), (2, 0, "e"), (2, 1, "f")]:
        cell = gs[r, c].get_position(fig)
        fig.text(cell.x0, cell.y1 + 0.006, f"({lab})",
                 fontsize=9, fontweight="bold", va="bottom", ha="left")
    _save(fig, save_path)
    plt.close(fig)


def main():
    p = Params()
    print("  Generating mesh schematics ...")
    plot_mesh_1d(p)
    plot_mesh_2d()
    plot_mesh_3d(p)
    plot_mesh_combined(p)
    print("  Done.")


if __name__ == "__main__":
    main()
