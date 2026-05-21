"""
Stage 1 driver: ORR PEMFC cathode CL bulk model.

Physics: dissolved O2 diffusion + Tafel ORR kinetics + phi_L/phi_s charge conservation.
Gas phase: constant Dirichlet BCs (no Stefan-Maxwell yet — that is Stage 3).

Workflow:
    1. Build Params and CLMesh.
    2. Print parameter summary.
    3. If cache exists, load it; otherwise run voltage sweep and save.
    4. Generate all Stage 1 diagnostic plots.
    5. STOP and wait for human review before proceeding to Stage 2.

Run:
    python run_stage1.py

Re-run from scratch:
    Delete stage1_cache.npz then re-run.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# ── Make sure the module directory is on the path ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from params import Params
from mesh import make_mesh
from assembly_stage1 import residual_stage1, compute_current
from solver import voltage_sweep
from cache import save_cache, load_cache, cache_exists
from plot_results import (
    plot_polarization,
    plot_profiles,
    plot_ir_breakdown,
    plot_consistency_check,
)

# ── Configuration ─────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).parent / "stage1_cache.npz"

V_START  = 1.00   # V vs SHE  (close to OCV; U_ORR_eq ~ 1.17 V at 80 °C)
V_END    = 0.35   # V vs SHE  (high overpotential, mass-transport limited)
DV_INIT  = 0.02   # V  initial voltage step
TOL      = 1e-8   # Newton convergence tolerance


def main():
    # ── Parameters ────────────────────────────────────────────────────────────
    p    = Params()
    mesh = make_mesh(p)

    p.print_summary()
    print(f"\n  U_ORR_0(T)  = {p.U_ORR_0:.4f} V vs SHE  (T = {p.T:.2f} K)")
    print(f"  U_ORR_eq    = {p.U_ORR_eq(p.c_O2_bc):.4f} V vs SHE  "
          f"(at inlet c_O2 = {p.c_O2_bc:.4f} mol/m3)")

    # ── Warm-start voltage estimate for first PAC point ───────────────────────
    U_ocv = float(p.U_ORR_eq(p.c_O2_bc))
    print(f"\n  Sweeping from V = {V_START:.3f} V  ->  V = {V_END:.3f} V")
    print(f"  (OCV ~ {U_ocv:.3f} V -> starting slightly below)\n")

    # ── Load cache or run sweep ───────────────────────────────────────────────
    if cache_exists(CACHE_PATH):
        print(f"  Cache found: {CACHE_PATH}")
        print("  WARNING: Verify kinetics match before reusing cached results!\n")
        voltages, solutions, p_stored = load_cache(CACHE_PATH)
        print(f"  Stored i0   = {p_stored.get('i0',  '?'):.2e}  "
              f"(current: {p.i0:.2e})")
        print(f"  Stored alpha= {p_stored.get('alpha','?')}  "
              f"(current: {p.alpha})")
    else:
        print("  No cache found — running voltage sweep ...\n")
        voltages, solutions = voltage_sweep(
            mesh, p,
            residual_fn=residual_stage1,
            V_start=V_START,
            V_end=V_END,
            dV_init=DV_INIT,
            tol=TOL,
            verbose=True,
        )
        save_cache(CACHE_PATH, voltages, solutions, p, stage=1)

    # ── Summary ───────────────────────────────────────────────────────────────
    V_arr = np.asarray(voltages)
    J_arr = np.array([compute_current(u, mesh, p) * 1e-4 for u in solutions])  # A/cm2
    print(f"\n  Sweep summary:")
    print(f"    V range   : {V_arr[-1]:.4f} – {V_arr[0]:.4f} V vs SHE")
    print(f"    J range   : {J_arr[0]:.4f} – {J_arr[-1]:.4f} A/cm2")
    print(f"    n_points  : {len(voltages)}")
    max_J = float(J_arr.max())
    print(f"    J_max     : {max_J:.4f} A/cm2")

    # ── Physicality quick-check ───────────────────────────────────────────────
    _quick_checks(voltages, solutions, mesh, p)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_polarization(voltages, solutions, mesh, p,
                      save_path="stage1_polarization.png")
    plot_profiles(voltages, solutions, mesh, p,
                  save_path="stage1_profiles.png")
    plot_ir_breakdown(voltages, solutions, mesh, p,
                      save_path="stage1_ir_breakdown.png")
    plot_consistency_check(voltages, solutions, mesh, p,
                           save_path="stage1_consistency.png")

    print("\n  Stage 1 complete.")
    print("  >> Review all plots before proceeding to Stage 2.")


def _quick_checks(voltages, solutions, mesh, p):
    """Print pass/fail for the Stage 1 physicality checks (§11)."""
    from assembly_stage1 import unpack, current_from_flux

    print("\n  Physicality checks:")
    N = mesh.N

    # Pick the highest-current solution for a stringent check
    idx_hc = -1 if voltages[-1] < voltages[0] else 0

    u      = solutions[idx_hc]
    V_cath = voltages[idx_hc]
    ln_cO2, phi_L, phi_s = unpack(u, N)

    # 1. phi_s monotonically decreasing from GDL to membrane
    d_phis = np.diff(phi_s)
    check1 = bool(np.all(d_phis <= 0.0))
    print(f"    phi_s monotone decreasing  : {'PASS' if check1 else 'FAIL'}")

    # 2. phi_L >= 0 throughout (positive ionic potential relative to membrane)
    check2 = bool(np.all(phi_L >= -1e-6))
    print(f"    phi_L >= 0 throughout      : {'PASS' if check2 else 'FAIL'}")

    # 3. c_O2 depletes from inlet toward membrane
    c_O2 = np.exp(ln_cO2)
    check3 = bool(c_O2[0] >= c_O2[-1])
    print(f"    c_O2 depletes inlet->mem    : {'PASS' if check3 else 'FAIL'}")

    # 4. Three-way current consistency
    d   = current_from_flux(u, mesh, p, V_cath)
    ref = abs(d["integral"]) + 1e-10
    e1  = abs(d["solid_flux"] - d["integral"]) / ref * 100
    e2  = abs(d["ionic_flux"] - d["integral"]) / ref * 100
    check4 = e1 < 1.0 and e2 < 1.0
    print(f"    3-way i_total agreement    : {'PASS' if check4 else 'FAIL'}"
          f"  (err_solid={e1:.3f}%, err_ionic={e2:.3f}%)")

    # 5. OCV sanity (at V_start, i_total should be very small)
    u0   = solutions[0]
    J0   = compute_current(u0, mesh, p) * 1e-4   # A/cm2
    check5 = abs(J0) < 1e-3
    print(f"    Near-OCV current < 1 mA/cm2: {'PASS' if check5 else 'FAIL'}"
          f"  (J={J0*1e3:.4f} mA/cm2)")


if __name__ == "__main__":
    main()
