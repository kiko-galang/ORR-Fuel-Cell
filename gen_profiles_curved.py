"""
Generate a curved-text version of the Stage 1 spatial profiles figure.

Outputs:
  stage1_profiles_curved.png  — voltage labels ride along each curve
  (stage1_profiles.png is the original — left untouched)
"""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from curved_text import curved_text

from params import Params
from mesh import make_mesh
from assembly_stage1 import unpack
from cache import load_cache
from customplot import gengrid, warm_sequential

CACHE_PATH = Path(__file__).parent / "stage1_cache.npz"
_LABELSIZE = 8


def main():
    p    = Params()
    mesh = make_mesh(p)
    voltages, solutions, _ = load_cache(CACHE_PATH)

    n = len(voltages)
    V_sample = [voltages[0], voltages[n // 3], voltages[2 * n // 3], voltages[-1]]

    V_arr  = np.asarray(voltages)
    xc_um  = mesh.xc * 1e6

    cidx   = np.linspace(2, len(warm_sequential) - 1, len(V_sample)).round().astype(int)
    colors = [warm_sequential[i] for i in cidx]
    N = mesh.N

    fig, axes, _ = gengrid(2, 2, size_inches=(6.5, 6.25), ticklabel_size=7)
    ax_c, ax_phiL, ax_phiS, ax_eta = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    # label positions staggered per voltage so they don't stack on top of each other
    pos_map = [0.20, 0.38, 0.56, 0.74]

    for (V_t, col, pos) in zip(V_sample, colors, pos_map):
        idx = int(np.argmin(np.abs(V_arr - V_t)))
        u   = solutions[idx]
        lbl = f"$V$ = {voltages[idx]:.3f} V"

        ln_cO2, phi_L, phi_s = unpack(u, N)
        c_O2  = np.exp(ln_cO2)
        U_eq  = p.U_ORR_eq(c_O2)
        eta   = (phi_s - phi_L) - U_eq

        box = {"color": "white", "alpha": 0.75, "pad": 1.0}

        ax_c.plot(xc_um, c_O2, color=col)
        curved_text(ax_c, xc_um, c_O2, lbl,
                    pos=pos, anchor="center", offset=6,
                    color=col, fontsize=5.5, box=box)

        ax_phiL.plot(xc_um, phi_L * 1e3, color=col)
        curved_text(ax_phiL, xc_um, phi_L * 1e3, lbl,
                    pos=pos, anchor="center", offset=6,
                    color=col, fontsize=5.5, box=box)

        ax_phiS.plot(xc_um, phi_s, color=col)
        curved_text(ax_phiS, xc_um, phi_s, lbl,
                    pos=pos, anchor="center", offset=6,
                    color=col, fontsize=5.5, box=box)

        ax_eta.plot(xc_um, eta * 1e3, color=col)
        curved_text(ax_eta, xc_um, eta * 1e3, lbl,
                    pos=pos, anchor="center", offset=6,
                    color=col, fontsize=5.5, box=box)

    # Extra headroom on phi_s and eta so V=1.000 label isn't cramped at the top
    for ax in (ax_phiS, ax_eta):
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.18 * (y1 - y0))

    ax_c.set_ylabel("$c_{O_2}$  (mol m$^{-3}$)", fontsize=_LABELSIZE)
    ax_phiL.set_ylabel("$\\phi_L$  (mV)", fontsize=_LABELSIZE)
    ax_phiS.set_ylabel("$\\phi_s$  (V vs SHE)", fontsize=_LABELSIZE)
    ax_eta.set_ylabel("$\\eta = (\\phi_s - \\phi_L) - U_{eq}$  (mV)", fontsize=_LABELSIZE)

    for ax in axes.flat:
        ax.set_xlabel("$x$  ($\\mu$m)", fontsize=_LABELSIZE)

    fig.tight_layout()
    out = "stage1_profiles_curved.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
