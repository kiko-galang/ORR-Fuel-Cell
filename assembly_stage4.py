"""
Stage 4 residual assembly: finite-rate gas<->ionomer O2 mass transfer with a
binary Maxwell-Stefan (stagnant-film) gas-phase flux law.

Two physics changes from Stage 3:

(1) Interphase coupling.
    Stage 3 sets c_O2_ion = K_eq * c_O2_gas algebraically (instantaneous local
    equilibrium) and applies the ORR sink to the GAS balance.  Stage 4 instead
    treats the dissolved (ionomer) O2 as an INDEPENDENT field again, coupled to
    the pore gas by a finite-rate interphase mass-transfer source:

        R_PT = k_v * (K_eq * c_O2_gas - c_O2_ion)   [mol/m3/s]   (net INTO ionomer)
        k_v  = k_MT_GL * a_GL                        [1/s]

    The ORR sink is moved to the DISSOLVED phase, and kinetics are fed the real
    c_O2_ion DOF.  Physically this introduces the ionomer-film local O2
    transport resistance (the dominant loss in low-Pt cathodes).

    As k_v -> infinity, R_PT can stay finite only if c_O2_ion -> K_eq*c_O2_gas,
    recovering Stage 3's coupling exactly (the reduction acceptance test).

(2) Gas-phase flux law.
    Stage 3 uses plain Fick.  Stage 4 uses the binary Maxwell-Stefan /
    stagnant-film law for O2 through inert stagnant N2 (N_N2 = 0):

        N_O2 = - D_O2N2_eff * grad(c_O2_gas) / (1 - x_O2),  x_O2 = c_O2_gas/c_tot

    applied in the GDL, the CL pore gas, and the harmonic-mean interface flux.
    N2 is closed algebraically (x_N2 = 1 - x_O2); no N2 DOF is added.  As
    x_O2 -> 0 (dilute limit, c_tot -> inf) the (1 - x_O2) factor -> 1 and the
    gas law reduces to Stage 3's plain Fick.

DOF layout (non-interleaved, log-stored concentrations):
    u[0        : NG]        = ln_c_O2_gdl    GDL gas O2,   log [mol/m3]
    u[NG       : NG+NC]     = ln_c_O2_clgas  CL  pore gas O2, log [mol/m3]
    u[NG+NC    : NG+2*NC]   = ln_c_O2_ion    CL  dissolved (ionomer) O2, log  <-- NEW
    u[NG+2*NC  : NG+3*NC]   = phi_L          ionic potential  [V]
    u[NG+3*NC  : NG+4*NC]   = phi_s          solid potential  [V]

    NG = mesh_gdl.N,  NC = mesh_cl.N
    Total DOFs: NG + 4 * NC

Governing equations (FV residual  F = J_left - J_right + S*dx):
    GDL gas : Maxwell-Stefan diffusion, no source
    CL gas  : Maxwell-Stefan diffusion, source  S = -R_PT
    CL ion  : Fickian diffusion (D_O2_eff), no-flux at BOTH faces,
              source  S = +R_PT - i_ORR/(N_ELEC*F)
    phi_L   : ohmic, source +i_ORR        (unchanged from Stage 3)
    phi_s   : ohmic, source -i_ORR        (unchanged from Stage 3)

Newton step clamping:
    ln_c (gdl gas, cl gas, ion) : +/- 5  (log units)
    phi_L, phi_s                : +/- 0.2 V
"""
from __future__ import annotations
import numpy as np
from kinetics   import R_ORR, N_ELEC
from transport  import ohmic_face_fluxes
from gas_transport import gdl_face_fluxes_ms, cl_gas_face_fluxes_ms, interface_flux_ms


# ── DOF helpers ───────────────────────────────────────────────────────────────

def unpack_s4(u: np.ndarray, NG: int, NC: int):
    """
    Split Stage 4 DOF vector into components.

    Returns
    -------
    ln_c_gdl   : (NG,)  log gas O2 in GDL
    ln_c_clgas : (NC,)  log gas O2 in CL pores
    ln_c_ion   : (NC,)  log dissolved (ionomer) O2 in CL
    phi_L      : (NC,)  ionic potential
    phi_s      : (NC,)  solid potential
    """
    return (
        u[0          : NG],
        u[NG         : NG + NC],
        u[NG + NC    : NG + 2*NC],
        u[NG + 2*NC  : NG + 3*NC],
        u[NG + 3*NC  : NG + 4*NC],
    )


def pack_s4(ln_c_gdl, ln_c_clgas, ln_c_ion, phi_L, phi_s) -> np.ndarray:
    """Concatenate Stage 4 arrays into a flat DOF vector."""
    return np.concatenate([ln_c_gdl, ln_c_clgas, ln_c_ion, phi_L, phi_s])


# ── Newton step clamping ──────────────────────────────────────────────────────

CLAMP_LN  = 5.0    # log units (gas and ionomer O2)
CLAMP_PHI = 0.2    # V

def clamp_step_s4(du: np.ndarray, NG: int, NC: int) -> np.ndarray:
    """Per-DOF-type Newton step clamping for Stage 4."""
    du = du.copy()
    # Three concentration blocks (gdl gas, cl gas, cl ion) span [0, NG+2*NC)
    du[0          : NG + 2*NC] = np.clip(du[0          : NG + 2*NC], -CLAMP_LN,  CLAMP_LN)
    du[NG + 2*NC  : NG + 3*NC] = np.clip(du[NG + 2*NC  : NG + 3*NC], -CLAMP_PHI, CLAMP_PHI)
    du[NG + 3*NC  : NG + 4*NC] = np.clip(du[NG + 3*NC  : NG + 4*NC], -CLAMP_PHI, CLAMP_PHI)
    return du


# ── Dissolved-O2 face fluxes (no-flux at BOTH CL faces) ───────────────────────

def ion_face_fluxes(ln_c_ion: np.ndarray, dx: float, D_eff: float) -> np.ndarray:
    """
    Fickian face fluxes for dissolved O2 in the ionomer, shape (NC+1,).

    The ionomer's only O2 supply is the volumetric phase-transfer rate R_PT,
    not a boundary flux, so BOTH faces are no-flux:
        Face 0    : x=0     (GDL/CL)      no-flux
        Face NC   : x=L_CL  (CL/membrane) no-flux
    """
    c = np.exp(ln_c_ion)
    N = len(c)
    J = np.empty(N + 1)
    J[0]    = 0.0                              # no-flux at GDL/CL face
    J[1:N]  = -D_eff * np.diff(c) / dx         # interior Fick fluxes
    J[N]    = 0.0                              # no-flux at CL/membrane face
    return J


# ── Initial guess ─────────────────────────────────────────────────────────────

def initial_guess_s4(mesh_gdl, mesh_cl, p, V_cathode: float) -> np.ndarray:
    """
    OCV initial guess for Stage 4 (starts from the Stage-3 equilibrium state).

    At zero current:
      - Gas O2 uniform at inlet concentration throughout GDL and CL
      - Dissolved O2 at local equilibrium  c_ion = K_eq * c_gas
      - phi_L = 0 everywhere
      - phi_s = V_cathode everywhere
    """
    NG = mesh_gdl.N
    NC = mesh_cl.N
    ln_c_in   = np.log(p.c_O2_gas_inlet)
    ln_c_ion  = np.log(p.K_eq_gas_ion * p.c_O2_gas_inlet)
    ln_c_gdl  = np.full(NG, ln_c_in)
    ln_c_clg  = np.full(NC, ln_c_in)
    ln_c_io   = np.full(NC, ln_c_ion)
    phi_L     = np.zeros(NC)
    phi_s     = np.full(NC, V_cathode)
    return pack_s4(ln_c_gdl, ln_c_clg, ln_c_io, phi_L, phi_s)


def s3_to_s4(u3: np.ndarray, NG: int, NC: int, p) -> np.ndarray:
    """
    Build a Stage 4 warm-start vector from a converged Stage 3 solution.

    Stage 3 layout [gdl | clgas | phi_L | phi_s] is expanded by inserting the
    dissolved-O2 block at equilibrium  c_ion = K_eq * c_gas.
    """
    ln_c_gdl = u3[0        : NG]
    ln_c_clg = u3[NG       : NG + NC]
    phi_L    = u3[NG + NC  : NG + 2*NC]
    phi_s    = u3[NG + 2*NC: NG + 3*NC]
    ln_c_ion = np.log(p.K_eq_gas_ion) + ln_c_clg   # ln(K_eq * c_gas)
    return pack_s4(ln_c_gdl, ln_c_clg, ln_c_ion, phi_L, phi_s)


# ── Stage 4 residual ─────────────────────────────────────────────────────────

def residual_stage4(
    u:          np.ndarray,
    mesh_gdl,              # CLMesh for GDL domain
    mesh_cl,               # CLMesh for CL domain
    p,                     # Params
    V_cathode:  float,
) -> np.ndarray:
    """
    Stage 4 residual vector F(u) of shape (NG + 4*NC,).

    GDL gas  (NG):  R = J_gdl[i] - J_gdl[i+1]                       (no source)
    CL gas   (NC):  R = J_cgas[j] - J_cgas[j+1] + (-R_PT)*dx
    CL ion   (NC):  R = J_ion[j]  - J_ion[j+1]  + (R_PT - i_ORR/(N_ELEC*F))*dx
    phi_L    (NC):  R = i_L[j]    - i_L[j+1]    + (+i_ORR)*dx
    phi_s    (NC):  R = i_s[j]    - i_s[j+1]    + (-i_ORR)*dx
    """
    NG  = mesh_gdl.N
    NC  = mesh_cl.N
    dxG = mesh_gdl.dx
    dxC = mesh_cl.dx

    ln_c_gdl, ln_c_clgas, ln_c_ion, phi_L, phi_s = unpack_s4(u, NG, NC)

    c_gas = np.exp(ln_c_clgas)
    c_ion = np.exp(ln_c_ion)

    # ── Finite-rate interphase phase transfer (net rate INTO ionomer) ─────────
    R_PT = p.k_v * (p.K_eq_gas_ion * c_gas - c_ion)     # (NC,) [mol/m3/s]

    # ── Kinetics (fed the REAL dissolved O2 DOF) ──────────────────────────────
    i_ORR = R_ORR(ln_c_ion, phi_s, phi_L, p)            # (NC,) [A/m3]

    # ── GDL/CL interface flux (harmonic mean D + Stefan correction) ───────────
    J_if = interface_flux_ms(
        ln_c_gdl, ln_c_clgas, dxG, dxC,
        p.D_O2_gdl_eff, p.D_O2_cl_gas_eff, p.c_tot,
    )

    # ── Gas-phase face fluxes (Maxwell-Stefan / stagnant-film) ────────────────
    J_gdl  = gdl_face_fluxes_ms(
        ln_c_gdl, dxG, p.D_O2_gdl_eff,
        c_in_gas=p.c_O2_gas_inlet, J_if=J_if, c_tot=p.c_tot,
    )
    J_cgas = cl_gas_face_fluxes_ms(
        ln_c_clgas, dxC, p.D_O2_cl_gas_eff, J_if=J_if, c_tot=p.c_tot,
    )

    # ── Dissolved-O2 diffusion (no-flux both faces) ───────────────────────────
    J_ion = ion_face_fluxes(ln_c_ion, dxC, p.D_O2_eff)

    # ── Ionic / solid current fluxes ──────────────────────────────────────────
    i_L = ohmic_face_fluxes(
        phi_L, dxC, p.kappa_L_eff,
        bc_left=None, bc_right=p.phi_L_mem,
    )
    i_s = ohmic_face_fluxes(
        phi_s, dxC, p.sigma_s_eff,
        bc_left=V_cathode, bc_right=None,
    )

    # ── FV residuals  F = J_left - J_right + S*dx ─────────────────────────────
    R_gdl  = J_gdl[:-1]  - J_gdl[1:]                                          # (NG,)
    R_cgas = J_cgas[:-1] - J_cgas[1:] + (-R_PT) * dxC                         # (NC,)
    R_cion = J_ion[:-1]  - J_ion[1:]  + (R_PT - i_ORR / (N_ELEC * p.F)) * dxC  # (NC,)
    R_phiL = i_L[:-1]    - i_L[1:]    + (+i_ORR) * dxC                        # (NC,)
    R_phiS = i_s[:-1]    - i_s[1:]    + (-i_ORR) * dxC                        # (NC,)

    return pack_s4(R_gdl, R_cgas, R_cion, R_phiL, R_phiS)


# ── Current and diagnostics ───────────────────────────────────────────────────

def compute_current_s4(u: np.ndarray, mesh_gdl, mesh_cl, p) -> float:
    """
    Total cathodic current density [A/m2_geo] by integrating volumetric ORR.

    Uses the REAL dissolved-O2 DOF (not gas equilibrium).
    """
    NG = mesh_gdl.N
    NC = mesh_cl.N
    _, _, ln_c_ion, phi_L, phi_s = unpack_s4(u, NG, NC)
    i_ORR = R_ORR(ln_c_ion, phi_s, phi_L, p)
    return float(np.sum(i_ORR) * mesh_cl.dx)


def diagnostics_s4(u: np.ndarray, mesh_gdl, mesh_cl, p, V_cathode: float) -> dict:
    """
    Physicality + consistency diagnostics for Stage 4.

    Returns dict with arrays and scalar checks.
    """
    NG = mesh_gdl.N
    NC = mesh_cl.N
    ln_c_gdl, ln_c_clgas, ln_c_ion, phi_L, phi_s = unpack_s4(u, NG, NC)

    c_O2_gdl = np.exp(ln_c_gdl)
    c_O2_cl  = np.exp(ln_c_clgas)
    c_O2_ion = np.exp(ln_c_ion)
    c_O2_eq  = p.K_eq_gas_ion * c_O2_cl        # local equilibrium curve

    # Pressure check (gas): p_O2 = c_O2_gas * R*T / 101325  (0 .. 0.21 atm)
    p_O2_gdl = c_O2_gdl * p.R * p.T / 101325.0
    p_O2_cl  = c_O2_cl  * p.R * p.T / 101325.0

    i_ORR    = R_ORR(ln_c_ion, phi_s, phi_L, p)
    i_total  = float(np.sum(i_ORR) * mesh_cl.dx)

    # Solid-current at GDL/CL interface (left face of CL)
    i_s_left  = -p.sigma_s_eff * (phi_s[0] - V_cathode) / (0.5 * mesh_cl.dx)
    # Ionic-current at CL/membrane interface (right face of CL)
    i_L_right = -p.kappa_L_eff * (p.phi_L_mem - phi_L[NC-1]) / (0.5 * mesh_cl.dx)

    # Global O2 mass balance: gas O2 entering the GDL at x=0 (gas channel face).
    # Must use the SAME Maxwell-Stefan (1 - x_O2) correction as the residual,
    # otherwise it disagrees with consumption by the Stefan-flow factor.
    #   J_gdl_in [mol/m2/s] should equal total ORR consumption i_total/(N_ELEC*F)
    c_face0    = 0.5 * (c_O2_gdl[0] + p.c_O2_gas_inlet)
    stefan0    = max(1.0 - c_face0 / p.c_tot, 1e-3)
    J_gdl_in   = -p.D_O2_gdl_eff * (c_O2_gdl[0] - p.c_O2_gas_inlet) / (0.5 * mesh_gdl.dx) / stefan0
    consumed   = i_total / (N_ELEC * p.F)

    return {
        "c_O2_gdl":   c_O2_gdl,
        "c_O2_cl":    c_O2_cl,
        "c_O2_ion":   c_O2_ion,
        "c_O2_eq":    c_O2_eq,
        "p_O2_gdl":   p_O2_gdl,
        "p_O2_cl":    p_O2_cl,
        "phi_L":      phi_L,
        "phi_s":      phi_s,
        "i_ORR":      i_ORR,
        "i_total":    i_total,
        "i_s_left":   float(i_s_left),
        "i_L_right":  float(i_L_right),
        "J_gdl_in":   float(J_gdl_in),
        "consumed":   float(consumed),
    }
