"""
Newton solver with FD Jacobian + adaptive voltage-stepping.

Step-clamping strategy:
    ln(c_O2) DOFs  : |Δu| <= 5.0  (prevents exp() overflow in early iterations)
    phi_L, phi_s   : |Δu| <= 0.2  (200 mV per Newton step)

Voltage sweep strategy:
    Simple natural continuation: solve at V, use converged solution as warm
    start for V − dV.  Adaptive dV: grow on easy convergence, halve on failure.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.linalg import solve, LinAlgError
from assembly_stage1 import unpack, pack


# ── Step-clamp limits ─────────────────────────────────────────────────────────
CLAMP_LN_CO2 = 5.0    # log units
CLAMP_PHI    = 0.2    # V


def _clamp_step(du: np.ndarray, N: int) -> np.ndarray:
    """Apply per-DOF-type clamping to a Newton step."""
    du = du.copy()
    du[0*N:1*N] = np.clip(du[0*N:1*N], -CLAMP_LN_CO2, CLAMP_LN_CO2)
    du[1*N:2*N] = np.clip(du[1*N:2*N], -CLAMP_PHI,    CLAMP_PHI)
    du[2*N:3*N] = np.clip(du[2*N:3*N], -CLAMP_PHI,    CLAMP_PHI)
    return du


# ── Finite-difference Jacobian ────────────────────────────────────────────────

def fd_jacobian(
    F_fn: Callable[[np.ndarray], np.ndarray],
    u:    np.ndarray,
    F0:   np.ndarray | None = None,
    eps:  float = 1e-7,
) -> np.ndarray:
    """
    Forward-difference Jacobian  J[i,j] ~ (F(u + h·eⱼ) - F(u)) / h.

    h = eps · max(|u[j]|, 1)  (relative perturbation, avoids tiny h near zero)

    Parameters
    ----------
    F_fn : callable, F_fn(u) -> residual vector of length M
    u    : current DOF vector, length N
    F0   : F_fn(u) already evaluated (avoids one redundant call); computed here if None
    eps  : relative perturbation size

    Returns
    -------
    J : (M, N) Jacobian matrix
    """
    if F0 is None:
        F0 = F_fn(u)
    N = len(u)
    M = len(F0)
    J = np.empty((M, N), dtype=float)
    for j in range(N):
        h     = eps * max(abs(u[j]), 1.0)
        u_p   = u.copy()
        u_p[j] += h
        J[:, j] = (F_fn(u_p) - F0) / h
    return J


# ── Newton solver ─────────────────────────────────────────────────────────────

def newton_solve(
    F_fn:     Callable[[np.ndarray], np.ndarray],
    u0:       np.ndarray,
    N_cl:     int,
    tol:      float = 1e-8,
    max_iter: int   = 60,
    verbose:  bool  = False,
    clamp_fn: Callable | None = None,
) -> tuple[np.ndarray, bool, int, float]:
    """
    Damped Newton solver.

    Parameters
    ----------
    F_fn     : residual function
    u0       : initial guess
    N_cl     : number of CL cells (needed for per-DOF step clamping)
    tol      : convergence criterion on ||F||_inf
    max_iter : maximum Newton iterations
    verbose  : print iteration history
    clamp_fn : optional callable(du) -> du for per-DOF step clamping.
               If None, uses default _clamp_step(du, N_cl).

    Returns
    -------
    u         : solution vector
    converged : bool
    n_iter    : number of iterations taken
    norm_F    : final ||F||_inf
    """
    if clamp_fn is None:
        clamp_fn = lambda du: _clamp_step(du, N_cl)

    u = u0.copy()

    prev_norm = np.inf
    for it in range(max_iter):
        F0     = F_fn(u)
        norm_F = float(np.max(np.abs(F0)))

        if verbose:
            print(f"    iter {it:3d}: ||F||_inf = {norm_F:.3e}")

        if norm_F < tol:
            return u, True, it, norm_F

        # Soft convergence: if residual is stagnating very close to tol, accept it
        if norm_F < 5.0 * tol and abs(norm_F - prev_norm) < 0.1 * tol:
            return u, True, it, norm_F

        prev_norm = norm_F

        J = fd_jacobian(F_fn, u, F0=F0)

        # Row scaling: normalise each equation by its infinity-norm so that
        # the O2 equations (scale ~1e-4) and potential equations (scale ~1e8)
        # are treated equally by the linear solver.
        scale = np.maximum(np.abs(J).max(axis=1), 1e-30)
        J_sc  = J  / scale[:, None]
        F_sc  = F0 / scale

        try:
            du = solve(J_sc, -F_sc, assume_a="gen")
        except LinAlgError:
            # Near-singular Jacobian — use least-norm step
            du, *_ = np.linalg.lstsq(J_sc, -F_sc, rcond=None)

        du  = clamp_fn(du)
        u   = u + du

    F0     = F_fn(u)
    norm_F = float(np.max(np.abs(F0)))
    return u, norm_F < tol, max_iter, norm_F


# ── Initial guess ─────────────────────────────────────────────────────────────

def initial_guess(mesh, p, V_cathode: float) -> np.ndarray:
    """
    Trivial zero-current initial guess (exact solution at OCV).

    At V_cathode = U_ORR_eq(c_O2_bc):
      - c_O2 is uniform at c_O2_bc  (no reaction, no gradient)
      - phi_L = 0 everywhere          (satisfies Dirichlet at membrane and no-flux at GDL)
      - phi_s = V_cathode everywhere  (satisfies Dirichlet at GDL and no-flux at membrane)
    """
    N      = mesh.N
    ln_cO2 = np.full(N, np.log(p.c_O2_bc))
    phi_L  = np.zeros(N)
    phi_s  = np.full(N, V_cathode)
    return pack(ln_cO2, phi_L, phi_s)


# ── Voltage sweep (adaptive natural continuation) ─────────────────────────────

def voltage_sweep(
    mesh,
    p,
    residual_fn:      Callable,
    V_start:          float = 1.0,
    V_end:            float = 0.35,
    dV_init:          float = 0.02,
    dV_min:           float = 1e-4,
    dV_max:           float = 0.05,
    tol:              float = 1e-8,
    max_iter:         int   = 60,
    verbose:          bool  = True,
    clamp_fn:         Callable | None = None,
    initial_guess_fn: Callable | None = None,
    current_fn:       Callable | None = None,
) -> tuple[list[float], list[np.ndarray]]:
    """
    Sweep V_cathode from V_start to V_end using adaptive step size.

    Adaptive rules:
      converged in <= 5  iters : dV <- min(dV * 1.5, dV_max)
      converged in 6-20 iters  : dV unchanged
      failed or > 20 iters     : dV <- max(dV * 0.5, dV_min), retry same V
      dV < dV_min              : abort

    Optional overrides (for Stage 3+):
      clamp_fn         : callable(du) -> du  (custom Newton step clamping)
      initial_guess_fn : callable(V) -> u0  (custom initial guess)
      current_fn       : callable(u) -> J [A/m2]  (for verbose printing)

    Returns
    -------
    voltages  : list of converged V_cathode values (descending)
    solutions : list of corresponding DOF vectors
    """
    from assembly_stage1 import compute_current

    if initial_guess_fn is None:
        initial_guess_fn = lambda V: initial_guess(mesh, p, V)
    if current_fn is None:
        current_fn = lambda u: compute_current(u, mesh, p)

    voltages  = []
    solutions = []
    N         = mesh.N

    V  = V_start
    dV = dV_init
    u  = initial_guess_fn(V_start)

    # Solve at the starting point to make sure we have a good warm start
    F_fn = lambda u_: residual_fn(u_, mesh, p, V)
    u, ok, nit, nrm = newton_solve(
        F_fn, u, N, tol=tol, max_iter=max_iter, clamp_fn=clamp_fn
    )
    if not ok:
        raise RuntimeError(
            f"Newton failed at starting voltage V={V:.4f} (||F||={nrm:.2e}). "
            "Check initial guess or tighten dV_init."
        )
    voltages.append(V)
    solutions.append(u.copy())
    if verbose:
        J_cell = current_fn(u)
        print(f"  V = {V:.4f} V  |  J = {J_cell*1e-4:.4f} A/cm2  |  iter = {nit:3d}")

    while V - dV >= V_end - 1e-10:
        V_try  = max(V - dV, V_end)
        F_fn   = lambda u_: residual_fn(u_, mesh, p, V_try)
        u_try, ok, nit, nrm = newton_solve(
            F_fn, u, N, tol=tol, max_iter=max_iter, clamp_fn=clamp_fn
        )

        if ok:
            V  = V_try
            u  = u_try
            voltages.append(V)
            solutions.append(u.copy())

            if verbose:
                J_cell = current_fn(u)
                print(f"  V = {V:.4f} V  |  J = {J_cell*1e-4:.4f} A/cm2"
                      f"  |  iter = {nit:3d}  |  ||F|| = {nrm:.2e}")

            # Adapt step size
            if nit <= 5:
                dV = min(dV * 1.5, dV_max)
            elif nit > 20:
                dV = max(dV * 0.5, dV_min)

            if abs(V - V_end) < 1e-12:
                break
        else:
            dV *= 0.5
            if verbose:
                print(f"  V = {V_try:.4f} FAILED (||F||={nrm:.2e}), "
                      f"halving dV -> {dV:.5f}")
            if dV < dV_min:
                print(f"  Step size below dV_min={dV_min:.2e} at V={V:.4f}. Stopping.")
                break

    return voltages, solutions
