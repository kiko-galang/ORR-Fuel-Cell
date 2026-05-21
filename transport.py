"""
Face-flux helpers for the 1D cell-centred FV scheme.

Sign convention throughout:
  Positive flux = flow in the +x direction (GDL → membrane).

Boundary treatment — half-cell ghost cells:
  Left face (f=0, x=0):   distance between BC and cell centre 0 is dx/2.
  Right face (f=N, x=L):  distance between cell centre N-1 and BC is dx/2.
  Interior faces (f=1…N-1): distance between adjacent cell centres is dx.
"""
from __future__ import annotations
import numpy as np


def diffusion_face_fluxes(
    ln_cO2: np.ndarray,
    dx:     float,
    D_eff:  float,
    bc_left:  float,           # Dirichlet c_O2 at x = 0  [mol/m³]
    bc_right: float | None,    # Dirichlet c_O2 at x = L, or None (no-flux)
) -> np.ndarray:
    """
    Fick's-law face fluxes for dissolved O2.

    Returns J of shape (N+1,) [mol/(m²·s)].
    J[f] = −D_eff · (c_O2[f] − c_O2[f−1]) / Δx_centers
    """
    c = np.exp(ln_cO2)      # (N,)
    N = len(c)
    J = np.empty(N + 1)

    # Left boundary (Dirichlet)
    J[0] = -D_eff * (c[0] - bc_left) / (0.5 * dx)

    # Interior faces
    J[1:N] = -D_eff * np.diff(c) / dx

    # Right boundary
    if bc_right is None:
        J[N] = 0.0                                          # no-flux
    else:
        J[N] = -D_eff * (bc_right - c[N - 1]) / (0.5 * dx)

    return J


def ohmic_face_fluxes(
    phi:      np.ndarray,
    dx:       float,
    sigma:    float,
    bc_left:  float | None,    # Dirichlet phi at x = 0, or None (no-flux)
    bc_right: float | None,    # Dirichlet phi at x = L, or None (no-flux)
) -> np.ndarray:
    """
    Ohm's-law face fluxes for a potential field (ionic or solid).

    Returns i_face of shape (N+1,) [A/m²].
    i_face[f] = −σ · (φ[f] − φ[f−1]) / Δx_centers
    """
    N = len(phi)
    i = np.empty(N + 1)

    # Left boundary
    if bc_left is None:
        i[0] = 0.0                                          # no-flux
    else:
        i[0] = -sigma * (phi[0] - bc_left) / (0.5 * dx)

    # Interior faces
    i[1:N] = -sigma * np.diff(phi) / dx

    # Right boundary
    if bc_right is None:
        i[N] = 0.0                                          # no-flux
    else:
        i[N] = -sigma * (bc_right - phi[N - 1]) / (0.5 * dx)

    return i
