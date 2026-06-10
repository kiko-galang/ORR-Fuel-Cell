"""
Stage 2 driver: Sherwood-Reynolds Neumann BC for O2 at the CL/membrane face.

Physics change from Stage 1:
    J_O2[N] = -k_MT * c_O2[N-1]    (was: 0, no-flux)

Expected outcome: plots visually identical to Stage 1.  Any discrepancy
larger than a few percent at the membrane-facing cells indicates a bug.

Run:
    python run_stage2.py

Re-run from scratch:
    Delete stage2_cache.npz then re-run.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from params import Params
from mesh import make_mesh
from assembly_stage1 import compute_current, unpack
from assembly_stage2 import residual_stage2, K_MT_DEFAULT
from solver import voltage_sweep
from cache import save_cache, load_cache, cache_exists, nearest_solution

CACHE_S1 = Path(__file__).parent / "stage1_cache.npz"
CACHE_S2 = Path(__file__).parent / "stage2_cache.npz"

V_START = 1.00
V_END   = 0.35
TOL     = 1e-8


def main():
    p    = Params()
    mesh = make_mesh(p)
    N    = mesh.N

    print("=" * 55)
    print("  Stage 2: Sherwood-Reynolds Neumann BC")
    print("=" * 55)
    print(f"  k_MT = {K_MT_DEFAULT:.3e} m/s")
    print(f"  (J_O2 at membrane = -k_MT * c_O2[N-1])")

    # ── Warm-start from Stage 1 ───────────────────────────────────────────────
    if not cache_exists(CACHE_S1):
        raise FileNotFoundError(
            "Stage 1 cache not found. Run run_stage1.py first."
        )
    vs1, sols1, _ = load_cache(CACHE_S1)
    u_warm, V_warm = nearest_solution(vs1, sols1, V_START)
    print(f"\n  Warm-start from Stage 1 at V = {V_warm:.4f} V\n")

    # ── Load Stage 2 cache or run sweep ──────────────────────────────────────
    if cache_exists(CACHE_S2):
        print(f"  Cache found: {CACHE_S2}")
        print("  WARNING: Verify k_MT matches before reusing!\n")
        voltages, solutions, _ = load_cache(CACHE_S2)
    else:
        print("  Running Stage 2 sweep ...\n")

        def res_fn(u, mesh, p, V):
            return residual_stage2(u, mesh, p, V, k_MT=K_MT_DEFAULT)

        voltages, solutions = voltage_sweep(
            mesh, p,
            residual_fn=res_fn,
            V_start=V_START, V_end=V_END,
            dV_init=0.02, tol=TOL,
            verbose=True,
        )
        save_cache(CACHE_S2, voltages, solutions, p, stage=2)

    # ── Comparison with Stage 1 ───────────────────────────────────────────────
    print("\n  Stage 1 vs Stage 2 comparison:")
    print(f"  {'V':>8}  {'J_s1 (mA/cm2)':>15}  {'J_s2 (mA/cm2)':>15}  {'diff%':>8}")
    print("  " + "-" * 55)

    V_arr = np.asarray(voltages)
    for u2, V in zip(solutions, voltages):
        J2 = compute_current(u2, mesh, p) * 1e-4 * 1e3   # mA/cm2
        u1, _ = nearest_solution(vs1, sols1, V)
        J1 = compute_current(u1, mesh, p) * 1e-4 * 1e3
        diff = abs(J2 - J1) / (abs(J1) + 1e-15) * 100
        print(f"  {V:>8.4f}  {J1:>15.5f}  {J2:>15.5f}  {diff:>8.3f}%")

    # ── Physicality checks ────────────────────────────────────────────────────
    print("\n  Physicality checks (highest-current solution):")
    u = solutions[-1]
    ln_cO2, phi_L, phi_s = unpack(u, N)
    c_O2 = np.exp(ln_cO2)

    check1 = bool(np.all(np.diff(phi_s) <= 0.0))
    check2 = bool(np.all(phi_L >= -1e-6))
    check3 = bool(c_O2[0] >= c_O2[-1])
    print(f"    phi_s monotone decreasing : {'PASS' if check1 else 'FAIL'}")
    print(f"    phi_L >= 0 throughout     : {'PASS' if check2 else 'FAIL'}")
    print(f"    c_O2 depletes inlet->mem  : {'PASS' if check3 else 'FAIL'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating Stage 2 plots ...")
    _plot_comparison(vs1, sols1, voltages, solutions, mesh, p)

    print("\n  Stage 2 complete.")
    print("  >> Plots should be visually identical to Stage 1.")
    print("  >> Review before proceeding to Stage 3.")


def _plot_comparison(vs1, sols1, vs2, sols2, mesh, p):
    """Overlay Stage 1 and Stage 2 polarization curves."""
    import matplotlib.pyplot as plt
    from customplot import gengrid, rainbow_2

    J1 = np.array([compute_current(u, mesh, p) * 1e-4 * 1e3 for u in sols1])
    J2 = np.array([compute_current(u, mesh, p) * 1e-4 * 1e3 for u in sols2])

    fig, axes, _ = gengrid(1, 2, size_inches=(6.5, 3.0), ticklabel_size=7)

    # Polarization overlay
    axes[0].plot(J1, vs1, marker="o", ms=4, lw=1.5, color=rainbow_2[1],
                 label="Stage 1 (no-flux)")
    axes[0].plot(J2, vs2, marker="s", ms=4, lw=1.5, ls="--", color=rainbow_2[4],
                 label="Stage 2 (Sherwood)")
    axes[0].set_xlabel("Current density  (mA cm$^{-2}$)", fontsize=8)
    axes[0].set_ylabel("$V_{cathode}$  (V vs SHE)", fontsize=8)
    axes[0].set_title("Polarization: Stage 1 vs Stage 2", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)

    # c_O2 profile at highest-current point
    N = mesh.N
    xc = mesh.xc * 1e6
    u1_hc = sols1[-1]
    u2_hc = sols2[-1]
    ln1, _, _ = unpack(u1_hc, N)
    ln2, _, _ = unpack(u2_hc, N)
    axes[1].plot(xc, np.exp(ln1), lw=1.5, color=rainbow_2[1],
                 label=f"Stage 1  V={vs1[-1]:.3f}")
    axes[1].plot(xc, np.exp(ln2), lw=1.5, ls="--", color=rainbow_2[4],
                 label=f"Stage 2  V={vs2[-1]:.3f}")
    axes[1].set_xlabel("x  (um)", fontsize=8)
    axes[1].set_ylabel("$c_{O_2}$  (mol m$^{-3}$)", fontsize=8)
    axes[1].set_title("O2 profile at highest-current point", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig("stage2_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Saved: stage2_comparison.png")


if __name__ == "__main__":
    main()
