"""
ORR Tafel / Butler-Volmer kinetics.

Single cathodic reaction:
    O2 + 4 H⁺ + 4 e⁻  →  2 H2O      (n_e = 4)

Volumetric current density:
    i_ORR  = a_v · i0 · (c_O2 / c_O2_ref)^γ · exp(−α_c · F · η / RT)

    where  η = (φ_s − φ_L) − U_ORR_eq(c_O2)   [kinetic overpotential, V]

i_ORR > 0  ⟺  cathodic reaction is proceeding (O2 consumed, electrons flowing in).
"""
from __future__ import annotations
import numpy as np


N_ELEC = 4   # electrons transferred per O2 molecule


def R_ORR(
    ln_cO2: np.ndarray,
    phi_s:  np.ndarray,
    phi_L:  np.ndarray,
    p,                    # Params object
) -> np.ndarray:
    """
    Volumetric ORR reaction rate [A/m³_CL].

    Parameters
    ----------
    ln_cO2 : (N,)  natural log of dissolved O2 concentration [mol/m³]
    phi_s  : (N,)  solid-phase potential [V]
    phi_L  : (N,)  ionic-phase potential [V]
    p      : Params

    Returns
    -------
    i_vol  : (N,)  volumetric current density [A/m³_CL], positive = cathodic
    """
    c_O2 = np.exp(ln_cO2)                           # always positive

    # Nernst-corrected equilibrium potential
    U_eq = p.U_ORR_eq(c_O2)                         # (N,) [V vs SHE]

    # Kinetic overpotential: negative for cathodic reaction (phi_s < U_eq)
    eta = (phi_s - phi_L) - U_eq                    # (N,) [V]

    # Concentration factor  (c_O2 / c_O2_ref)^gamma
    conc = np.maximum(c_O2 / p.c_O2_ref, 0.0) ** p.gamma

    # Exponential argument — clamp to prevent IEEE overflow
    arg = np.clip(-p.alpha * p.F * eta / (p.R * p.T), -700.0, 700.0)

    # Per-Pt-surface current density [A/m²_Pt]
    i_Pt = p.i0 * conc * np.exp(arg)

    # Volumetric rate [A/m³_CL]
    return p.a_v * i_Pt
