"""
Stage 2 residual assembly.

Identical to Stage 1 except the right-face O2 boundary condition at
x = L_CL (CL/membrane interface) changes from:

    Stage 1:  J_O2[N] = 0                         (no-flux)
    Stage 2:  J_O2[N] = -k_MT * c_O2[N-1]         (Sherwood-Reynolds Neumann)

Physically: the membrane is opaque to O2 (c_O2_bulk_mem = 0), so
the convective flux drives c_O2 toward zero at the membrane face.
For typical k_MT values this is nearly indistinguishable from no-flux
because the ionomer-phase diffusion resistance dominates.  Stage 2 is
included for code-path completeness (§7.2 of the guide).

All other equations (phi_L, phi_s, kinetics) are unchanged from Stage 1.
"""
from __future__ import annotations
import numpy as np
from kinetics import R_ORR, N_ELEC
from transport import diffusion_face_fluxes, ohmic_face_fluxes
from assembly_stage1 import unpack, pack


# Default mass-transfer coefficient [m/s]
# The PEM membrane is essentially impermeable to O2, so k_MT << D_O2_eff/L_CL.
# Using a near-zero value makes Stage 2 a numerical no-op (confirms membrane opacity).
# If you want to explore water back-diffusion or partial O2 permeation, increase this.
K_MT_DEFAULT = 1e-10   # m/s  (essentially zero — impermeable membrane)


def diffusion_face_fluxes_stage2(
    ln_cO2:  np.ndarray,
    dx:      float,
    D_eff:   float,
    bc_left: float,
    k_MT:    float,
) -> np.ndarray:
    """
    O2 face fluxes with Sherwood-Reynolds Neumann BC at the right boundary.

    Right face:  J[N] = -k_MT * c_O2[N-1]
                        (convective loss toward membrane-side bulk c = 0)
    """
    c = np.exp(ln_cO2)
    N = len(c)
    J = np.empty(N + 1)

    # Left boundary: Dirichlet c_O2 = bc_left
    J[0] = -D_eff * (c[0] - bc_left) / (0.5 * dx)

    # Interior faces
    J[1:N] = -D_eff * np.diff(c) / dx

    # Right boundary: Sherwood-Reynolds Neumann (membrane-side bulk = 0)
    J[N] = -k_MT * c[N - 1]

    return J


def residual_stage2(
    u:         np.ndarray,
    mesh,                   # CLMesh
    p,                      # Params
    V_cathode: float,
    k_MT:      float = K_MT_DEFAULT,
) -> np.ndarray:
    """
    Stage 2 residual — same as Stage 1 with Sherwood-Reynolds O2 BC.

    Parameters
    ----------
    u         : DOF vector, length 3·N
    mesh      : CLMesh
    p         : Params
    V_cathode : applied cathode potential [V vs SHE]
    k_MT      : mass-transfer coefficient at membrane face [m/s]
    """
    N  = mesh.N
    dx = mesh.dx

    ln_cO2, phi_L, phi_s = unpack(u, N)

    # Kinetics
    i_ORR = R_ORR(ln_cO2, phi_s, phi_L, p)

    # O2 fluxes — Stage 2 BC at right face
    J_O2 = diffusion_face_fluxes_stage2(
        ln_cO2, dx, p.D_O2_eff, bc_left=p.c_O2_bc, k_MT=k_MT
    )

    # Ionic current — unchanged from Stage 1
    i_L = ohmic_face_fluxes(
        phi_L, dx, p.kappa_L_eff,
        bc_left=None, bc_right=p.phi_L_mem
    )

    # Solid current — unchanged from Stage 1
    i_s = ohmic_face_fluxes(
        phi_s, dx, p.sigma_s_eff,
        bc_left=V_cathode, bc_right=None
    )

    # FV residuals
    R_O2   = J_O2[:-1] - J_O2[1:]  +  (-i_ORR / (N_ELEC * p.F)) * dx
    R_phiL = i_L[:-1]  - i_L[1:]   +  (+i_ORR) * dx
    R_phiS = i_s[:-1]  - i_s[1:]   +  (-i_ORR) * dx

    return pack(R_O2, R_phiL, R_phiS)
