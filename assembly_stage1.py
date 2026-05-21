"""
Stage 1 residual assembly.

DOF layout (non-interleaved):
    u[0·N : 1·N]  =  ln(c_O2)   (log-stored dissolved O2)
    u[1·N : 2·N]  =  phi_L       (ionic potential [V])
    u[2·N : 3·N]  =  phi_s       (solid potential [V])

Total DOFs: 3·N_CL

FV residual form  F[cell] = J_left − J_right + S·dx = 0

Equations per cell:
    R_O2  :  J_O2[f] − J_O2[f+1] + (−i_ORR / 4F)·dx  = 0   [mol/m²/s]
    R_phiL:  i_L[f]  − i_L[f+1]  + (+i_ORR)·dx        = 0   [A/m²]
    R_phiS:  i_s[f]  − i_s[f+1]  + (−i_ORR)·dx        = 0   [A/m²]

Sign conventions (all consistent with the Implementation Guide §3):
    i_ORR > 0  for cathodic ORR (O2 consumed)
    phi_L ≥ 0 inside CL; = 0 at membrane (Dirichlet)
    phi_s ≈ V_cathode at GDL (Dirichlet); slightly lower at membrane (IR drop)
    i_L > 0 (flows from GDL toward membrane, grows from 0 to i_total)
    i_s > 0 (flows from GDL into CL, decreases from i_total to 0)
"""
from __future__ import annotations
import numpy as np
from kinetics import R_ORR, N_ELEC
from transport import diffusion_face_fluxes, ohmic_face_fluxes


def unpack(u: np.ndarray, N: int):
    """Split flat DOF vector into (ln_cO2, phi_L, phi_s), each shape (N,)."""
    return u[0*N:1*N], u[1*N:2*N], u[2*N:3*N]


def pack(ln_cO2, phi_L, phi_s) -> np.ndarray:
    """Concatenate three arrays into the flat DOF vector."""
    return np.concatenate([ln_cO2, phi_L, phi_s])


def residual_stage1(
    u:        np.ndarray,
    mesh,               # CLMesh
    p,                  # Params
    V_cathode: float,   # Dirichlet phi_s at x = 0  [V vs SHE]
) -> np.ndarray:
    """
    Compute the Stage 1 residual vector F(u) of shape (3·N,).

    Parameters
    ----------
    u         : DOF vector, length 3·N
    mesh      : CLMesh
    p         : Params
    V_cathode : applied cathode potential [V vs SHE]

    Returns
    -------
    F : residual vector, length 3·N
    """
    N  = mesh.N
    dx = mesh.dx

    ln_cO2, phi_L, phi_s = unpack(u, N)

    # ── Kinetics ──────────────────────────────────────────────────────────────
    i_ORR = R_ORR(ln_cO2, phi_s, phi_L, p)    # (N,)  [A/m³]

    # ── O2 diffusion fluxes ───────────────────────────────────────────────────
    #   BC left  (x = 0)   : Dirichlet c_O2 = c_O2_bc
    #   BC right (x = L_CL): no-flux
    J_O2 = diffusion_face_fluxes(
        ln_cO2, dx, p.D_O2_eff,
        bc_left=p.c_O2_bc, bc_right=None
    )   # (N+1,)  [mol/m²/s]

    # ── Ionic current (phi_L) fluxes ──────────────────────────────────────────
    #   BC left  (x = 0)   : no-flux  (ionic current does not cross GDL)
    #   BC right (x = L_CL): Dirichlet phi_L = 0  (membrane reference)
    i_L = ohmic_face_fluxes(
        phi_L, dx, p.kappa_L_eff,
        bc_left=None, bc_right=p.phi_L_mem
    )   # (N+1,)  [A/m²]

    # ── Solid current (phi_s) fluxes ──────────────────────────────────────────
    #   BC left  (x = 0)   : Dirichlet phi_s = V_cathode
    #   BC right (x = L_CL): no-flux  (no electronic current into membrane)
    i_s = ohmic_face_fluxes(
        phi_s, dx, p.sigma_s_eff,
        bc_left=V_cathode, bc_right=None
    )   # (N+1,)  [A/m²]

    # ── FV residuals  (F = J_left − J_right + S·dx) ──────────────────────────
    R_O2   = J_O2[:-1] - J_O2[1:]  +  (-i_ORR / (N_ELEC * p.F)) * dx
    R_phiL = i_L[:-1]  - i_L[1:]   +  (+i_ORR) * dx
    R_phiS = i_s[:-1]  - i_s[1:]   +  (-i_ORR) * dx

    return pack(R_O2, R_phiL, R_phiS)


def compute_current(u: np.ndarray, mesh, p) -> float:
    """
    Total current density [A/m²_geo] by integrating volumetric ORR rate.

    Equivalent to  i_s(x=0) = −σ_eff · dφ_s/dx|_{x=0}
    and          i_L(x=L_CL) = −κ_eff · dφ_L/dx|_{x=L_CL}.
    All three should agree to < 1 % for a converged solution.
    """
    N = mesh.N
    ln_cO2, phi_s, phi_L = u[0*N:1*N], u[2*N:3*N], u[1*N:2*N]
    i_ORR = R_ORR(ln_cO2, phi_s, phi_L, p)
    return float(np.sum(i_ORR) * mesh.dx)


def current_from_flux(u: np.ndarray, mesh, p, V_cathode: float) -> dict:
    """
    Compute i_total three independent ways for the consistency check (§11).

    Returns dict with keys 'integral', 'solid_flux', 'ionic_flux'.
    """
    N  = mesh.N
    dx = mesh.dx
    ln_cO2, phi_L, phi_s = unpack(u, N)

    # 1. Integral of volumetric rate
    i_ORR    = R_ORR(ln_cO2, phi_s, phi_L, p)
    i_integ  = float(np.sum(i_ORR) * dx)

    # 2. Solid-current flux at the left face (GDL/CL)
    i_solid_left = -p.sigma_s_eff * (phi_s[0] - V_cathode) / (0.5 * dx)

    # 3. Ionic-current flux at the right face (CL/membrane)
    i_ionic_right = -p.kappa_L_eff * (p.phi_L_mem - phi_L[N - 1]) / (0.5 * dx)

    return {
        "integral":    i_integ,
        "solid_flux":  float(i_solid_left),
        "ionic_flux":  float(i_ionic_right),
    }
