"""
Stage 3 residual assembly: gas-phase O2 transport in GDL + CL pores.

Physics changes from Stage 1/2:
    - GDL gas O2 DOFs added (Fickian diffusion, D_O2_N2_bulk * Bruggeman)
    - CL gas O2 DOFs replace ionomer O2 DOFs
    - Ionomer O2 diagnosed via local Henry's law equilibrium:
          c_O2_ion = K_eq * c_O2_gas,  K_eq = H_O2 * R * T / 101325
    - ORR kinetics unchanged; fed c_O2_ion derived from gas
    - phi_L and phi_s equations identical to Stage 1

With D_O2_cl_gas_eff ~ 6.9e-6 m2/s vs D_O2_eff ~ 1.3e-10 m2/s (ionomer),
Stage 3 has ~54,000x higher O2 conductance in the CL.  Currents shift from
the mA/cm2 range (Stage 1) to potentially hundreds of mA/cm2 (Stage 3),
limited by Tafel kinetics rather than O2 transport.

DOF layout (non-interleaved):
    u[0       : NG]        = ln_c_O2_gdl   GDL gas O2, log [mol/m3]
    u[NG      : NG+NC]     = ln_c_O2_cl    CL  gas O2, log [mol/m3]
    u[NG+NC   : NG+2*NC]   = phi_L         ionic potential  [V]
    u[NG+2*NC : NG+3*NC]   = phi_s         solid potential  [V]

    NG = mesh_gdl.N,  NC = mesh_cl.N
    Total DOFs: NG + 3 * NC

Newton step clamping:
    ln_c_O2_gdl, ln_c_O2_cl : +/- 5  (log units, same as Stage 1)
    phi_L, phi_s             : +/- 0.2 V
"""
from __future__ import annotations
import numpy as np
from kinetics   import R_ORR, N_ELEC
from transport  import ohmic_face_fluxes
from gas_transport import gdl_face_fluxes, cl_gas_face_fluxes, interface_flux


# ── DOF helpers ───────────────────────────────────────────────────────────────

def unpack_s3(u: np.ndarray, NG: int, NC: int):
    """
    Split Stage 3 DOF vector into components.

    Returns
    -------
    ln_c_gdl : (NG,)  log gas O2 in GDL
    ln_c_cl  : (NC,)  log gas O2 in CL pores
    phi_L    : (NC,)  ionic potential
    phi_s    : (NC,)  solid potential
    """
    return (
        u[0        : NG],
        u[NG       : NG + NC],
        u[NG + NC  : NG + 2*NC],
        u[NG + 2*NC: NG + 3*NC],
    )


def pack_s3(ln_c_gdl, ln_c_cl, phi_L, phi_s) -> np.ndarray:
    """Concatenate Stage 3 arrays into a flat DOF vector."""
    return np.concatenate([ln_c_gdl, ln_c_cl, phi_L, phi_s])


# ── Newton step clamping ──────────────────────────────────────────────────────

CLAMP_LN = 5.0    # log units (gas and ionomer O2)
CLAMP_PHI = 0.2   # V

def clamp_step_s3(du: np.ndarray, NG: int, NC: int) -> np.ndarray:
    """Per-DOF-type Newton step clamping for Stage 3."""
    du = du.copy()
    du[0        : NG + NC] = np.clip(du[0        : NG + NC], -CLAMP_LN,  CLAMP_LN)
    du[NG + NC  : NG + 2*NC] = np.clip(du[NG + NC  : NG + 2*NC], -CLAMP_PHI, CLAMP_PHI)
    du[NG + 2*NC: NG + 3*NC] = np.clip(du[NG + 2*NC: NG + 3*NC], -CLAMP_PHI, CLAMP_PHI)
    return du


# ── Initial guess ─────────────────────────────────────────────────────────────

def initial_guess_s3(mesh_gdl, mesh_cl, p, V_cathode: float) -> np.ndarray:
    """
    OCV initial guess for Stage 3.

    At zero current:
      - Gas O2 uniform at inlet concentration throughout GDL and CL
      - phi_L = 0 everywhere
      - phi_s = V_cathode everywhere
    """
    NG = mesh_gdl.N
    NC = mesh_cl.N
    ln_c_in  = np.log(p.c_O2_gas_inlet)
    ln_c_gdl = np.full(NG, ln_c_in)
    ln_c_cl  = np.full(NC, ln_c_in)
    phi_L    = np.zeros(NC)
    phi_s    = np.full(NC, V_cathode)
    return pack_s3(ln_c_gdl, ln_c_cl, phi_L, phi_s)


# ── Stage 3 residual ─────────────────────────────────────────────────────────

def residual_stage3(
    u:          np.ndarray,
    mesh_gdl,              # CLMesh for GDL domain
    mesh_cl,               # CLMesh for CL domain
    p,                     # Params
    V_cathode:  float,
) -> np.ndarray:
    """
    Stage 3 residual vector F(u) of shape (NG + 3*NC,).

    GDL equations (NG):
        R_gdl[i] = J_gdl[i] - J_gdl[i+1]          (no source in GDL)

    CL gas equations (NC):
        R_cgas[j] = J_cgas[j] - J_cgas[j+1]
                    + (-i_ORR[j] / (N_ELEC * F)) * dx_cl

    CL phi_L equations (NC):  same as Stage 1
    CL phi_s equations (NC):  same as Stage 1
    """
    NG  = mesh_gdl.N
    NC  = mesh_cl.N
    dxG = mesh_gdl.dx
    dxC = mesh_cl.dx

    ln_c_gdl, ln_c_cl, phi_L, phi_s = unpack_s3(u, NG, NC)

    # ── Ionomer O2 via equilibrium coupling ────────────────────────────────────
    c_O2_ion   = p.K_eq_gas_ion * np.exp(ln_c_cl)       # (NC,) [mol/m3]
    ln_cO2_ion = np.log(np.maximum(c_O2_ion, 1e-30))    # (NC,) for R_ORR

    # ── Kinetics (same form as Stage 1, using ionomer c_O2) ───────────────────
    i_ORR = R_ORR(ln_cO2_ion, phi_s, phi_L, p)          # (NC,) [A/m3]

    # ── GDL/CL interface flux (harmonic mean diffusivity) ─────────────────────
    J_if = interface_flux(
        ln_c_gdl, ln_c_cl, dxG, dxC,
        p.D_O2_gdl_eff, p.D_O2_cl_gas_eff,
    )

    # ── GDL gas O2 face fluxes ────────────────────────────────────────────────
    J_gdl = gdl_face_fluxes(
        ln_c_gdl, dxG, p.D_O2_gdl_eff,
        c_in_gas=p.c_O2_gas_inlet, J_if=J_if,
    )

    # ── CL gas O2 face fluxes ─────────────────────────────────────────────────
    J_cl = cl_gas_face_fluxes(
        ln_c_cl, dxC, p.D_O2_cl_gas_eff, J_if=J_if,
    )

    # ── Ionic current fluxes (phi_L) ──────────────────────────────────────────
    i_L = ohmic_face_fluxes(
        phi_L, dxC, p.kappa_L_eff,
        bc_left=None, bc_right=p.phi_L_mem,
    )

    # ── Solid current fluxes (phi_s) ──────────────────────────────────────────
    i_s = ohmic_face_fluxes(
        phi_s, dxC, p.sigma_s_eff,
        bc_left=V_cathode, bc_right=None,
    )

    # ── FV residuals  F = J_left - J_right + S*dx ─────────────────────────────
    R_gdl  = J_gdl[:-1] - J_gdl[1:]                                   # (NG,)
    R_cgas = J_cl[:-1]  - J_cl[1:]  + (-i_ORR / (N_ELEC * p.F)) * dxC  # (NC,)
    R_phiL = i_L[:-1]   - i_L[1:]   + (+i_ORR) * dxC                  # (NC,)
    R_phiS = i_s[:-1]   - i_s[1:]   + (-i_ORR) * dxC                  # (NC,)

    return pack_s3(R_gdl, R_cgas, R_phiL, R_phiS)


# ── Current and diagnostics ───────────────────────────────────────────────────

def compute_current_s3(u: np.ndarray, mesh_gdl, mesh_cl, p) -> float:
    """
    Total cathodic current density [A/m2_geo] by integrating volumetric ORR.

    Uses ionomer O2 diagnosed from gas equilibrium: c_O2_ion = K_eq * c_O2_gas.
    """
    NG = mesh_gdl.N
    NC = mesh_cl.N
    _, ln_c_cl, phi_L, phi_s = unpack_s3(u, NG, NC)
    c_O2_ion   = p.K_eq_gas_ion * np.exp(ln_c_cl)
    ln_cO2_ion = np.log(np.maximum(c_O2_ion, 1e-30))
    i_ORR = R_ORR(ln_cO2_ion, phi_s, phi_L, p)
    return float(np.sum(i_ORR) * mesh_cl.dx)


def diagnostics_s3(u: np.ndarray, mesh_gdl, mesh_cl, p, V_cathode: float) -> dict:
    """
    Physicality diagnostics for Stage 3.

    Returns dict with arrays and scalar checks.
    """
    NG = mesh_gdl.N
    NC = mesh_cl.N
    ln_c_gdl, ln_c_cl, phi_L, phi_s = unpack_s3(u, NG, NC)

    c_O2_gdl  = np.exp(ln_c_gdl)
    c_O2_cl   = np.exp(ln_c_cl)
    c_O2_ion  = p.K_eq_gas_ion * c_O2_cl

    # Pressure check: p_O2 = c_O2_gas * R*T / 101325 (should be 0..0.21 atm)
    p_O2_gdl = c_O2_gdl * p.R * p.T / 101325.0
    p_O2_cl  = c_O2_cl  * p.R * p.T / 101325.0

    i_ORR    = R_ORR(np.log(np.maximum(c_O2_ion, 1e-30)), phi_s, phi_L, p)
    i_total  = float(np.sum(i_ORR) * mesh_cl.dx)

    # Solid-current at GDL/CL interface (left face of CL)
    i_s_left  = -p.sigma_s_eff * (phi_s[0] - V_cathode) / (0.5 * mesh_cl.dx)
    # Ionic-current at CL/membrane interface (right face of CL)
    i_L_right = -p.kappa_L_eff * (p.phi_L_mem - phi_L[NC-1]) / (0.5 * mesh_cl.dx)

    return {
        "c_O2_gdl":   c_O2_gdl,
        "c_O2_cl":    c_O2_cl,
        "c_O2_ion":   c_O2_ion,
        "p_O2_gdl":   p_O2_gdl,
        "p_O2_cl":    p_O2_cl,
        "phi_L":      phi_L,
        "phi_s":      phi_s,
        "i_ORR":      i_ORR,
        "i_total":    i_total,
        "i_s_left":   float(i_s_left),
        "i_L_right":  float(i_L_right),
    }
