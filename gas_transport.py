"""
Gas-phase O2 transport functions for Stage 3.

Binary Fickian diffusion (O2 / N2), isobaric, isothermal.
Bruggeman correction applied in each domain:
    GDL: D_gdl_eff = eps_G_gdl^brugg * D_O2_N2_bulk
    CL:  D_cl_eff  = eps_G_CL^brugg  * D_O2_N2_bulk

DOF convention:
    Gas O2 is stored as ln(c_O2_gas) [log mol/m3], matching the ionomer
    convention.  This keeps all concentration DOFs well-scaled for Newton.

Equilibrium phase coupling (Henry's law at local T):
    c_O2_ion = K_eq * c_O2_gas
    K_eq = H_O2 * R * T / 101325   (dimensionless)

At the GDL/CL interface the diffusivity changes.  The interface flux uses a
harmonic-mean effective diffusivity weighted by the half-cell distances on
each side, which is the standard FV treatment for heterogeneous media.
"""
from __future__ import annotations
import numpy as np


def gdl_face_fluxes(
    ln_c_gdl:    np.ndarray,   # (N_GDL,) log gas O2 in GDL
    dx_gdl:      float,
    D_gdl_eff:   float,
    c_in_gas:    float,        # Dirichlet BC at x=0 (gas channel) [mol/m3]
    J_if:        float,        # interface flux at x=L_GDL [mol/m2/s]
) -> np.ndarray:
    """
    Face fluxes for GDL gas O2, shape (N_GDL+1,).

    Face 0      : left  Dirichlet (gas channel)
    Faces 1..N-1: interior
    Face N_GDL  : GDL/CL interface (passed in as J_if)
    """
    c = np.exp(ln_c_gdl)
    N = len(c)
    J = np.empty(N + 1)

    # Left boundary: Dirichlet c = c_in_gas
    J[0] = -D_gdl_eff * (c[0] - c_in_gas) / (0.5 * dx_gdl)

    # Interior faces
    J[1:N] = -D_gdl_eff * np.diff(c) / dx_gdl

    # Right boundary: interface flux (set by caller)
    J[N] = J_if

    return J


def cl_gas_face_fluxes(
    ln_c_cl:    np.ndarray,   # (N_CL,) log gas O2 in CL pores
    dx_cl:      float,
    D_cl_eff:   float,
    J_if:       float,        # interface flux at x=L_GDL (left face of CL)
) -> np.ndarray:
    """
    Face fluxes for CL gas O2, shape (N_CL+1,).

    Face 0    : GDL/CL interface (continuity, passed in as J_if)
    Faces 1..N-1: interior
    Face N_CL : right Neumann no-flux (membrane impermeable to O2)
    """
    c = np.exp(ln_c_cl)
    N = len(c)
    J = np.empty(N + 1)

    # Left boundary: continuity with GDL
    J[0] = J_if

    # Interior faces
    J[1:N] = -D_cl_eff * np.diff(c) / dx_cl

    # Right boundary: no-flux
    J[N] = 0.0

    return J


def interface_flux(
    ln_c_gdl:   np.ndarray,   # GDL gas concentrations (log)
    ln_c_cl:    np.ndarray,   # CL  gas concentrations (log)
    dx_gdl:     float,
    dx_cl:      float,
    D_gdl_eff:  float,
    D_cl_eff:   float,
) -> float:
    """
    O2 flux at the GDL/CL interface [mol/m2/s].

    Uses a harmonic-mean diffusivity with half-cell distances on each side:
        1/D_if = (dx_gdl/2) / (D_gdl * (dx_gdl/2 + dx_cl/2))
               + (dx_cl/2)  / (D_cl  * (dx_gdl/2 + dx_cl/2))

    Simplified to:
        J_if = -(c_cl[0] - c_gdl[-1]) / (0.5*dx_gdl/D_gdl + 0.5*dx_cl/D_cl)
    """
    c_gdl_last  = np.exp(ln_c_gdl[-1])
    c_cl_first  = np.exp(ln_c_cl[0])
    resistance  = 0.5 * dx_gdl / D_gdl_eff + 0.5 * dx_cl / D_cl_eff
    return -(c_cl_first - c_gdl_last) / resistance
