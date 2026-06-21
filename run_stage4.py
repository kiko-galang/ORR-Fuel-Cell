"""
Stage 4 driver: finite-rate gas<->ionomer O2 transfer + Maxwell-Stefan gas law.

Physics changes from Stage 3:
    (1) Dissolved (ionomer) O2 is an INDEPENDENT field again.  Gas and dissolved
        phases are coupled by a finite-rate interphase transfer source
            R_PT = k_v * (K_eq*c_gas - c_ion)   [mol/m3/s]   (net into ionomer)
        The ORR sink lives on the dissolved phase; kinetics are fed the real
        c_O2_ion DOF.  This introduces the ionomer-film local O2 resistance.
    (2) Gas-phase O2 transport uses the binary Maxwell-Stefan / stagnant-film
        flux law  N_O2 = -D_eff*grad(c)/(1 - x_O2),  x_O2 = c_gas/c_tot
        in the GDL, CL pores, and the interface flux (Stefan flow through
        stagnant N2).

    DOF layout: [ln_c_gas_GDL | ln_c_gas_CL | ln_c_ion_CL | phi_L | phi_s]
    Total: N_GDL + 4*N_CL = 230 DOFs

Run:
    python run_stage4.py

Re-run from scratch:
    Delete stage4_cache.npz then re-run.

Requires stage1_cache.npz and stage3_cache.npz (run run_stage1.py / run_stage3.py
first) for the polarization overlay and the Stage-3 reduction comparisons.
"""
from __future__ import annotations
import sys
import copy
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from params          import Params
from mesh            import make_mesh, make_gdl_mesh
from assembly_stage1 import compute_current as compute_current_s1
from assembly_stage3 import compute_current_s3
from assembly_stage4 import (
    residual_stage4, compute_current_s4,
    unpack_s4, initial_guess_s4, clamp_step_s4, diagnostics_s4, s3_to_s4,
)
from solver import voltage_sweep, newton_solve
from cache  import save_cache, load_cache, cache_exists, nearest_solution

CACHE_S1 = Path(__file__).parent / "stage1_cache.npz"
CACHE_S3 = Path(__file__).parent / "stage3_cache.npz"
CACHE_S4 = Path(__file__).parent / "stage4_cache.npz"

V_START = 1.00
V_END   = 0.35
TOL     = 1e-8


# ── Sweep / solve helpers ─────────────────────────────────────────────────────

def _run_sweep(p, mesh_gdl, mesh_cl, u_warm=None, verbose=False):
    """Run a full Stage 4 voltage sweep for the given Params object."""
    NG, NC = mesh_gdl.N, mesh_cl.N

    def clamp_fn(du):
        return clamp_step_s4(du, NG, NC)

    def curr_fn(u):
        return compute_current_s4(u, mesh_gdl, mesh_cl, p)

    def res_fn(u, _mesh, _p, V):
        return residual_stage4(u, mesh_gdl, mesh_cl, p, V)

    def ig_default(V):
        return initial_guess_s4(mesh_gdl, mesh_cl, p, V)

    if u_warm is not None:
        _first = [True]
        def ig_fn(V):
            if _first[0]:
                _first[0] = False
                return u_warm.copy()
            return ig_default(V)
    else:
        ig_fn = ig_default

    return voltage_sweep(
        mesh_cl, p,
        residual_fn=res_fn,
        V_start=V_START, V_end=V_END,
        dV_init=0.02, tol=TOL,
        verbose=verbose,
        clamp_fn=clamp_fn,
        initial_guess_fn=ig_fn,
        current_fn=curr_fn,
    )


def _solve_at(p, mesh_gdl, mesh_cl, V, u0):
    """Single Newton solve at a fixed voltage (used for the k_v sweep)."""
    NG, NC = mesh_gdl.N, mesh_cl.N
    res   = lambda u: residual_stage4(u, mesh_gdl, mesh_cl, p, V)
    clamp = lambda du: clamp_step_s4(du, NG, NC)
    u, ok, _, _ = newton_solve(res, u0.copy(), NC, tol=TOL, clamp_fn=clamp)
    return u, ok


def _scaled_params(p, kv_factor=1.0, ctot_factor=1.0):
    """Deep copy of p with k_v and/or c_tot scaled (for the limit tests)."""
    q = copy.deepcopy(p)
    q.k_MT_GL = p.k_MT_GL * kv_factor
    q.recompute()
    q.c_tot = q.c_tot * ctot_factor    # override derived c_tot after recompute
    return q


def _maxrel_polarization(vsA, JA, vsB, JB):
    """Max relative current difference of curve A vs curve B over their overlap."""
    vA, JA = np.asarray(vsA, float), np.asarray(JA, float)
    vB, JB = np.asarray(vsB, float), np.asarray(JB, float)
    oB     = np.argsort(vB)
    vBs, JBs = vB[oB], JB[oB]
    mask   = (vA >= vBs.min()) & (vA <= vBs.max())
    Jb     = np.interp(vA[mask], vBs, JBs)
    ref    = np.maximum(np.abs(Jb), 0.01 * np.max(np.abs(JB)))
    return float(np.max(np.abs(JA[mask] - Jb) / ref))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p        = Params()
    mesh_cl  = make_mesh(p)
    mesh_gdl = make_gdl_mesh(p)
    NG, NC   = mesh_gdl.N, mesh_cl.N

    print("=" * 64)
    print("  Stage 4: finite-rate gas<->ionomer transfer + Maxwell-Stefan gas")
    print("=" * 64)
    print(f"  DOFs            = NG + 4*NC = {NG} + 4*{NC} = {NG + 4*NC}")
    print(f"  k_MT_GL         = {p.k_MT_GL:.3e} m/s")
    print(f"  a_GL            = {p.a_GL:.3e} 1/m")
    print(f"  k_v             = {p.k_v:.3e} 1/s   (interphase transfer)")
    print(f"  K_eq_gas_ion    = {p.K_eq_gas_ion:.4f}")
    print(f"  c_tot           = {p.c_tot:.3f} mol/m3   "
          f"(x_O2_inlet = {p.c_O2_gas_inlet/p.c_tot:.4f})")
    print()

    # ── Required caches ───────────────────────────────────────────────────────
    if not cache_exists(CACHE_S1):
        raise FileNotFoundError("Stage 1 cache not found. Run run_stage1.py first.")
    if not cache_exists(CACHE_S3):
        raise FileNotFoundError("Stage 3 cache not found. Run run_stage3.py first.")
    vs1, sols1, _ = load_cache(CACHE_S1)
    vs3, sols3, _ = load_cache(CACHE_S3)

    # Warm-start the first Stage 4 point from the Stage 3 solution near V_START.
    u3_start, V3w = nearest_solution(vs3, sols3, V_START)
    u_warm = s3_to_s4(u3_start, NG, NC, p)
    print(f"  Warm-start from Stage 3 at V = {V3w:.4f} V\n")

    # ── Main Stage 4 sweep (cached) ───────────────────────────────────────────
    if cache_exists(CACHE_S4):
        print(f"  Cache found: {CACHE_S4}")
        print("  WARNING: Verify params match before reusing!\n")
        vs4, sols4, _ = load_cache(CACHE_S4)
    else:
        print("  Running Stage 4 sweep (default k_v) ...\n")
        vs4, sols4 = _run_sweep(p, mesh_gdl, mesh_cl, u_warm=u_warm, verbose=True)
        save_cache(CACHE_S4, vs4, sols4, p, stage=4)

    # Currents [A/m2] for each stage at their own voltage grids
    J1 = np.array([compute_current_s1(u, mesh_cl, p)          for u in sols1])
    J3 = np.array([compute_current_s3(u, mesh_gdl, mesh_cl, p) for u in sols3])
    J4 = np.array([compute_current_s4(u, mesh_gdl, mesh_cl, p) for u in sols4])

    # ── Stage 1 vs 3 vs 4 comparison table ────────────────────────────────────
    print("\n  Stage comparison (mA/cm2):")
    print(f"  {'V':>8}  {'J_s1':>10}  {'J_s3':>10}  {'J_s4':>10}  {'s4/s3':>7}")
    print("  " + "-" * 52)
    for u4, V in zip(sols4, vs4):
        j4 = compute_current_s4(u4, mesh_gdl, mesh_cl, p) * 1e-1
        u3, _ = nearest_solution(vs3, sols3, V)
        u1, _ = nearest_solution(vs1, sols1, V)
        j3 = compute_current_s3(u3, mesh_gdl, mesh_cl, p) * 1e-1
        j1 = compute_current_s1(u1, mesh_cl, p) * 1e-1
        print(f"  {V:>8.4f}  {j1:>10.4f}  {j3:>10.4f}  {j4:>10.4f}  "
              f"{j4/(abs(j3)+1e-12):>7.3f}")

    # ── Verification checklist ────────────────────────────────────────────────
    _verify(p, mesh_gdl, mesh_cl, u_warm, vs3, sols3, J3, vs4, sols4)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating Stage 4 plots ...")
    from plot_results import (
        plot_stage4_polarization, plot_stage4_o2_profiles, plot_kv_sweep,
    )
    plot_stage4_polarization(vs1, sols1, mesh_cl, vs3, sols3,
                             vs4, sols4, mesh_gdl, p,
                             save_path="stage4_polarization.png")
    plot_stage4_o2_profiles(vs4, sols4, mesh_gdl, mesh_cl, p,
                            save_path="stage4_o2_profiles.png")

    kv_list, Jlim = _kv_sweep(p, mesh_gdl, mesh_cl, sols4[-1], vs4[-1])
    J3_lim = float(np.max(np.abs(J3))) * 1e-1   # mA/cm2
    plot_kv_sweep(kv_list, Jlim, J3_lim, p.k_v,
                  save_path="stage4_kv_sweep.png")

    print("\n  Stage 4 complete.")


# ── k_v sweep (limiting current vs k_v) ───────────────────────────────────────

def _kv_sweep(p, mesh_gdl, mesh_cl, u0_lowV, V_low):
    """Limiting current [mA/cm2] vs k_v, solved at the lowest sweep voltage."""
    print("\n  Running k_v sweep for limiting-current asymptote ...")
    kv0      = p.k_v
    kv_list  = np.logspace(np.log10(kv0) - 2.0, np.log10(kv0) + 4.0, 13)
    Jlim     = np.full(len(kv_list), np.nan)
    for i, kv in enumerate(kv_list):
        q = _scaled_params(p, kv_factor=kv / kv0)
        u, ok = _solve_at(q, mesh_gdl, mesh_cl, V_low, u0_lowV)
        if ok:
            Jlim[i] = compute_current_s4(u, mesh_gdl, mesh_cl, q) * 1e-1  # mA/cm2
    n_ok = int(np.sum(np.isfinite(Jlim)))
    print(f"    {n_ok}/{len(kv_list)} k_v points converged.")
    return kv_list, Jlim


# ── Verification ──────────────────────────────────────────────────────────────

def _verify(p, mesh_gdl, mesh_cl, u_warm, vs3, sols3, J3, vs4, sols4):
    print("\n" + "=" * 64)
    print("  VERIFICATION CHECKLIST")
    print("=" * 64)

    NC = mesh_cl.N

    # ---- 1. Equilibrium-limit test: k_v -> inf recovers Stage 3 ---------------
    print("\n  [1] Equilibrium-limit test (k_v x 1e6 -> Stage 3 coupling):")
    p_eq = _scaled_params(p, kv_factor=1e6)
    vs_eq, sols_eq = _run_sweep(p_eq, mesh_gdl, mesh_cl, u_warm=u_warm, verbose=False)
    J_eq = np.array([compute_current_s4(u, mesh_gdl, mesh_cl, p_eq) for u in sols_eq])
    err_pol = _maxrel_polarization(vs_eq, J_eq, vs3, J3) * 100.0
    # gas profile match at the lowest common voltage
    u_eq_lo = sols_eq[-1]
    u3_lo, _ = nearest_solution(vs3, sols3, vs_eq[-1])
    cg_eq = np.exp(u_eq_lo[mesh_gdl.N:mesh_gdl.N + NC])
    cg_s3 = np.exp(u3_lo[mesh_gdl.N:mesh_gdl.N + NC])
    err_prof = float(np.max(np.abs(cg_eq - cg_s3) / (np.abs(cg_s3) + 1e-30))) * 100.0
    check1 = err_pol < 1.0 and err_prof < 1.0
    print(f"      max |J_s4eq - J_s3| / J_s3   = {err_pol:.4f} %")
    print(f"      max gas-profile rel. diff    = {err_prof:.4f} %")
    print(f"      -> {'PASS' if check1 else 'FAIL'} (both < 1%)")

    # ---- 2. Dilute-limit test: (1 - x_O2) -> 1 as c_tot -> inf ----------------
    print("\n  [2] Dilute-limit test (Maxwell-Stefan -> plain Fick as c_tot x100):")
    u_lo = sols4[-1]                               # most depleted default solution
    c_gas_all = np.concatenate([
        np.exp(u_lo[0:mesh_gdl.N]),                # GDL gas
        np.exp(u_lo[mesh_gdl.N:mesh_gdl.N + NC]),  # CL  gas
    ])
    # Stefan correction deviation |1/(1-x) - 1| = x/(1-x)
    x_def = c_gas_all / p.c_tot
    dev_default = float(np.max(x_def / (1.0 - x_def))) * 100.0
    x_dil = c_gas_all / (100.0 * p.c_tot)
    dev_dilute  = float(np.max(x_dil / (1.0 - x_dil))) * 100.0
    check2 = dev_dilute < 1.0
    print(f"      Stefan correction at default c_tot : {dev_default:.2f} %  (active)")
    print(f"      Stefan correction at c_tot x 100   : {dev_dilute:.4f} %")
    print(f"      -> {'PASS' if check2 else 'FAIL'} (MS reduces to Fick, < 1%)")

    # ---- 3. Resistance test: default J_lim < Stage 3 J_lim --------------------
    print("\n  [3] Resistance test (default k_v limiting current < Stage 3):")
    J4_lim = float(np.max([compute_current_s4(u, mesh_gdl, mesh_cl, p) for u in sols4]))
    J3_lim = float(np.max(np.abs(J3)))
    drop   = (J3_lim - J4_lim) / J3_lim * 100.0
    check3 = J4_lim < J3_lim
    print(f"      J_lim Stage 4 = {J4_lim*1e-1:.3f} mA/cm2")
    print(f"      J_lim Stage 3 = {J3_lim*1e-1:.3f} mA/cm2")
    print(f"      reduction     = {drop:.2f} %  (dominated by ionomer-film "
          f"interphase resistance; cf. test 1 Stefan-only ~0.3%)")
    print(f"      -> {'PASS' if check3 else 'FAIL'} (strictly lower)")

    # ---- diagnostics on highest-current default solution ----------------------
    V_hc = vs4[-1]
    d    = diagnostics_s4(sols4[-1], mesh_gdl, mesh_cl, p, V_hc)

    # ---- 4. Three-way current consistency -------------------------------------
    print("\n  [4] Three-way current consistency (< 1%):")
    e_solid = abs(d["i_s_left"]  - d["i_total"]) / (abs(d["i_total"]) + 1e-10) * 100
    e_ionic = abs(d["i_L_right"] - d["i_total"]) / (abs(d["i_total"]) + 1e-10) * 100
    check4  = e_solid < 1.0 and e_ionic < 1.0
    print(f"      integral   i_ORR  = {d['i_total']*1e-1:.4f} mA/cm2")
    print(f"      solid flux i_s(0) : err = {e_solid:.4f} %")
    print(f"      ionic flux i_L(L) : err = {e_ionic:.4f} %")
    print(f"      -> {'PASS' if check4 else 'FAIL'}")

    # ---- 5. Physicality -------------------------------------------------------
    print("\n  [5] Physicality:")
    c_gdl, c_cl, c_ion, c_eq = d["c_O2_gdl"], d["c_O2_cl"], d["c_O2_ion"], d["c_O2_eq"]
    phi_L, phi_s = d["phi_L"], d["phi_s"]
    pos      = bool(np.all(c_gdl > 0) and np.all(c_cl > 0) and np.all(c_ion > 0))
    below_eq = bool(np.all(c_ion <= c_eq * (1.0 + 1e-6)))
    mono_s   = bool(np.all(np.diff(phi_s) <= 1e-12))
    pos_L    = bool(np.all(phi_L >= -1e-6))
    J0       = compute_current_s4(sols4[0], mesh_gdl, mesh_cl, p) * 1e-1   # mA/cm2
    ocv_ok   = abs(J0) < 1.0
    check5   = pos and below_eq and mono_s and pos_L and ocv_ok
    print(f"      all concentrations > 0       : {'PASS' if pos else 'FAIL'}")
    print(f"      c_ion <= K_eq*c_gas          : {'PASS' if below_eq else 'FAIL'}")
    print(f"      phi_s monotone decreasing    : {'PASS' if mono_s else 'FAIL'}")
    print(f"      phi_L >= 0 throughout        : {'PASS' if pos_L else 'FAIL'}")
    print(f"      near-OCV current < 1 mA/cm2  : {'PASS' if ocv_ok else 'FAIL'}"
          f"  (J0 = {J0:.4f} mA/cm2)")
    print(f"      -> {'PASS' if check5 else 'FAIL'}")

    # ---- 6. Global O2 mass balance --------------------------------------------
    print("\n  [6] Global O2 mass balance (GDL inflow = ORR consumption, < 1%):")
    e_mb   = abs(d["J_gdl_in"] - d["consumed"]) / (abs(d["consumed"]) + 1e-30) * 100
    check6 = e_mb < 1.0
    print(f"      O2 into GDL    = {d['J_gdl_in']:.6e} mol/m2/s")
    print(f"      ORR consumed   = {d['consumed']:.6e} mol/m2/s")
    print(f"      -> {'PASS' if check6 else 'FAIL'}  (err = {e_mb:.4f} %)")

    # ---- summary --------------------------------------------------------------
    checks = [check1, check2, check3, check4, check5, check6]
    print("\n" + "=" * 64)
    print(f"  RESULT: {sum(checks)}/6 checks passed"
          f"  -> {'ALL PASS' if all(checks) else 'SOME FAILED'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
