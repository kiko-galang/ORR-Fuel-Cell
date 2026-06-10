"""
Stage 3 driver: gas-phase O2 transport in GDL + CL pores.

Physics change from Stage 1/2:
    - Gas-phase O2 DOFs added for GDL (N_GDL=30) and CL (N_CL=50)
    - Ionomer O2 diagnosed via local Henry's law equilibrium:
          c_O2_ion = K_eq * c_O2_gas   (K_eq ~ 0.017 at 80 degC)
    - DOF layout: [ln_c_gas_GDL | ln_c_gas_CL | phi_L | phi_s]
      Total: N_GDL + 3*N_CL = 180 DOFs

Expected outcome:
    - At OCV (V~1.0), currents should match Stage 1 closely (same O2 at inlet)
    - At high overpotential, Stage 3 currents are >> Stage 1 because gas
      transport (D_O2_N2_bulk*eps_G^1.5 ~ 6.9e-6 m2/s) is ~54000x faster
      than ionomer transport (D_O2_eff ~ 1.3e-10 m2/s)
    - Polarization curve transitions from kinetic region (~1.0-0.9 V) into
      mass-transport limit at lower voltages

Run:
    python run_stage3.py

Re-run from scratch:
    Delete stage3_cache.npz then re-run.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from params          import Params
from mesh            import make_mesh, make_gdl_mesh
from assembly_stage1 import compute_current as compute_current_s1, unpack
from assembly_stage2 import residual_stage2
from assembly_stage3 import (
    residual_stage3, compute_current_s3,
    unpack_s3, initial_guess_s3, clamp_step_s3, diagnostics_s3,
)
from solver import voltage_sweep
from cache  import save_cache, load_cache, cache_exists, nearest_solution

CACHE_S1 = Path(__file__).parent / "stage1_cache.npz"
CACHE_S3 = Path(__file__).parent / "stage3_cache.npz"

V_START = 1.00
V_END   = 0.35
TOL     = 1e-8


def main():
    p        = Params()
    mesh_cl  = make_mesh(p)
    mesh_gdl = make_gdl_mesh(p)
    NG       = mesh_gdl.N
    NC       = mesh_cl.N

    print("=" * 60)
    print("  Stage 3: Gas-phase O2 transport (GDL + CL)")
    print("=" * 60)
    print(f"  GDL: L={p.L_GDL*1e6:.0f} um, N={NG}, "
          f"D_eff={p.D_O2_gdl_eff:.3e} m2/s")
    print(f"  CL gas: D_eff={p.D_O2_cl_gas_eff:.3e} m2/s  "
          f"(ionomer: {p.D_O2_eff:.3e} m2/s, ratio={p.D_O2_cl_gas_eff/p.D_O2_eff:.0f}x)")
    print(f"  K_eq   = {p.K_eq_gas_ion:.4f}  (c_ion/c_gas, Henry's law)")
    print(f"  c_O2_gas_inlet = {p.c_O2_gas_inlet:.4f} mol/m3")
    print(f"  c_O2_ion(inlet)= {p.K_eq_gas_ion*p.c_O2_gas_inlet:.4f} mol/m3 "
          f"(Stage 1 c_O2_bc = {p.c_O2_bc:.4f} mol/m3)")
    print()

    # Custom clamp and current functions for Stage 3 DOF layout
    def clamp_fn(du):
        return clamp_step_s3(du, NG, NC)

    def ig_fn(V):
        return initial_guess_s3(mesh_gdl, mesh_cl, p, V)

    def curr_fn(u):
        return compute_current_s3(u, mesh_gdl, mesh_cl, p)

    # Residual wrapper (voltage_sweep expects res_fn(u, mesh, p, V))
    # We pass mesh_cl as 'mesh'; the wrapper ignores it and uses closures.
    def res_fn(u, _mesh, _p, V):
        return residual_stage3(u, mesh_gdl, mesh_cl, p, V)

    # ── Warm-start from Stage 1 (phi_L, phi_s) ───────────────────────────────
    if not cache_exists(CACHE_S1):
        raise FileNotFoundError(
            "Stage 1 cache not found. Run run_stage1.py first."
        )
    vs1, sols1, _ = load_cache(CACHE_S1)
    u1_start, V1_warm = nearest_solution(vs1, sols1, V_START)
    ln1, phi_L1, phi_s1 = unpack(u1_start, NC)

    # Build Stage 3 warm start: gas O2 at inlet value, phi from Stage 1
    ln_c_in  = np.log(p.c_O2_gas_inlet)
    u_warm   = np.concatenate([
        np.full(NG, ln_c_in),   # GDL gas O2
        np.full(NC, ln_c_in),   # CL gas O2 (equilibrium inlet)
        phi_L1,                 # ionic potential from Stage 1
        phi_s1,                 # solid potential from Stage 1
    ])
    print(f"  Warm-start phi from Stage 1 at V = {V1_warm:.4f} V\n")

    # Override ig_fn to use the Stage 1 warm start for the first point
    _first_call = [True]
    def ig_fn_warmstart(V):
        if _first_call[0]:
            _first_call[0] = False
            return u_warm.copy()
        return ig_fn(V)

    # ── Load Stage 3 cache or run sweep ──────────────────────────────────────
    if cache_exists(CACHE_S3):
        print(f"  Cache found: {CACHE_S3}")
        print("  WARNING: Verify params match before reusing!\n")
        voltages, solutions, _ = load_cache(CACHE_S3)
    else:
        print("  Running Stage 3 sweep ...\n")
        voltages, solutions = voltage_sweep(
            mesh_cl, p,
            residual_fn=res_fn,
            V_start=V_START, V_end=V_END,
            dV_init=0.02, tol=TOL,
            verbose=True,
            clamp_fn=clamp_fn,
            initial_guess_fn=ig_fn_warmstart,
            current_fn=curr_fn,
        )
        save_cache(CACHE_S3, voltages, solutions, p, stage=3)

    # ── Comparison with Stage 1 ───────────────────────────────────────────────
    print("\n  Stage 1 vs Stage 3 comparison:")
    print(f"  {'V':>8}  {'J_s1 (mA/cm2)':>15}  {'J_s3 (mA/cm2)':>15}  {'ratio':>8}")
    print("  " + "-" * 55)
    for u3, V in zip(solutions, voltages):
        J3 = compute_current_s3(u3, mesh_gdl, mesh_cl, p) * 1e-4 * 1e3
        u1, _ = nearest_solution(vs1, sols1, V)
        J1 = compute_current_s1(u1, mesh_cl, p) * 1e-4 * 1e3
        ratio = J3 / (abs(J1) + 1e-15)
        print(f"  {V:>8.4f}  {J1:>15.5f}  {J3:>15.5f}  {ratio:>8.1f}x")

    # ── Physicality checks ────────────────────────────────────────────────────
    print("\n  Physicality checks (highest-current solution):")
    u_hc = solutions[-1]
    V_hc = voltages[-1]
    d    = diagnostics_s3(u_hc, mesh_gdl, mesh_cl, p, V_hc)

    c_gdl = d["c_O2_gdl"]
    c_cl  = d["c_O2_cl"]
    p_O2  = d["p_O2_cl"]
    phi_L = d["phi_L"]
    phi_s = d["phi_s"]

    check1 = bool(np.all(np.diff(phi_s) <= 0.0))
    check2 = bool(np.all(phi_L >= -1e-6))
    check3 = bool(c_gdl[0] >= c_gdl[-1] and c_cl[0] >= c_cl[-1])
    check4 = bool(np.all(p_O2 >= -1e-6) and np.all(p_O2 <= p.p_O2_inlet + 1e-4))
    err_cv = abs(d["i_s_left"] - d["i_total"]) / (abs(d["i_total"]) + 1e-10) * 100
    check5 = err_cv < 1.0

    print(f"    phi_s monotone decreasing   : {'PASS' if check1 else 'FAIL'}")
    print(f"    phi_L >= 0 throughout       : {'PASS' if check2 else 'FAIL'}")
    print(f"    c_O2_gas depletes GDL->mem  : {'PASS' if check3 else 'FAIL'}")
    print(f"    p_O2 in [0, 0.21] atm range : {'PASS' if check4 else 'FAIL'}")
    print(f"    i_s(x=0) vs integral < 1%   : {'PASS' if check5 else 'FAIL'}"
          f"  (err={err_cv:.3f}%)")
    print(f"\n    J_max = {d['i_total']*1e-4*1e3:.2f} mA/cm2  at V = {V_hc:.3f} V")
    print(f"    p_O2 at CL/mem face: {p_O2[-1]*1e3:.3f} matm"
          f"  (inlet: {p.p_O2_inlet*1e3:.0f} matm = 210 matm)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating Stage 3 plots ...")
    _plot_stage3(vs1, sols1, voltages, solutions, mesh_gdl, mesh_cl, p)

    print("\n  Stage 3 complete.")
    print("  >> Gas transport raises current density by the ratio shown above.")
    print("  >> Review plots before proceeding to Stage 4.")


def _plot_stage3(vs1, sols1, vs3, sols3, mesh_gdl, mesh_cl, p):
    """Stage 3 diagnostic plots: polarization overlay + ionomer O2 profiles."""
    import matplotlib.pyplot as plt
    from customplot import gengrid, rainbow_2, warm_sequential
    from assembly_stage1 import compute_current as cc1, unpack

    NC = mesh_cl.N
    NG = mesh_gdl.N

    J1 = np.array([cc1(u, mesh_cl, p) * 1e-4 * 1e3 for u in sols1])
    J3 = np.array([compute_current_s3(u, mesh_gdl, mesh_cl, p) * 1e-4 * 1e3
                   for u in sols3])

    # Two panels at the same width used by every other figure in the chapter
    fig, axes, _ = gengrid(1, 2, size_inches=(6.5, 2.5), ticklabel_size=7)

    # ── Panel 1: polarization overlay ────────────────────────────────────────
    axes[0].plot(J1, vs1, marker="o", ms=3, lw=1.5, color=rainbow_2[1],
                 label="Stage 1 (ionomer only)")
    axes[0].plot(J3, vs3, marker="s", ms=3, lw=1.5, ls="--", color=rainbow_2[4],
                 label="Stage 3 (gas transport)")
    axes[0].set_xlabel("Current density  (mA cm$^{-2}$)", fontsize=8)
    axes[0].set_ylabel("$V_{\\mathrm{cathode}}$  (V vs SHE)", fontsize=8)
    axes[0].set_title("Polarization: Stage 1 vs Stage 3", fontsize=9)
    axes[0].legend(fontsize=6, frameon=False)

    # ── Panel 2: ionomer O2 profiles at several voltages ─────────────────────
    xc_um  = mesh_cl.xc * 1e6
    n3 = len(vs3)
    idx_samples = [0, n3 // 4, n3 // 2, 3 * n3 // 4, n3 - 1]
    cidx   = np.linspace(2, len(warm_sequential) - 1, len(idx_samples)).round().astype(int)
    colors = [warm_sequential[i] for i in cidx]
    for idx, col in zip(idx_samples, colors):
        u3  = sols3[idx]
        V3  = vs3[idx]
        _, ln_c_cl_i, _, _ = unpack_s3(u3, NG, NC)
        c_ion = p.K_eq_gas_ion * np.exp(ln_c_cl_i)
        axes[1].plot(xc_um, c_ion, color=col, lw=1.5, label=f"V={V3:.3f}")
    # Stage 1 at highest current for comparison
    u1_hc = sols1[-1]
    ln1, _, _ = unpack(u1_hc, NC)
    axes[1].plot(xc_um, np.exp(ln1), lw=1.2, ls="--", color=rainbow_2[1],
                 label=f"S1 V={vs1[-1]:.3f}")
    axes[1].set_xlabel("$x$  (μm)", fontsize=8)
    axes[1].set_ylabel("$c_{O_2}$ ionomer  (mol m$^{-3}$)", fontsize=8)
    axes[1].set_title("Ionomer O$_2$ profiles (Stage 3 vs Stage 1)", fontsize=9)
    axes[1].legend(fontsize=6, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(left=0.12)
    fig.savefig("stage3_results.png", bbox_inches="tight")
    plt.close()
    print("  Saved: stage3_results.png")


if __name__ == "__main__":
    main()
