"""
Physical parameters for the ORR PEMFC cathode CL bulk model (Stage 1).

All quantities in SI units unless noted. Temperature fixed at 80 °C (353.15 K).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Params:
    """
    Container for all model parameters.  Derived quantities are recomputed
    automatically via recompute(); call it any time a base parameter is changed.
    """

    # ── Physical constants ────────────────────────────────────────────────────
    F: float = 96485.332      # C/mol  (Faraday)
    R: float = 8.314463       # J/mol/K
    T: float = 353.15         # K  (80 °C, typical PEMFC operating temperature)

    # ── Geometry ──────────────────────────────────────────────────────────────
    L_CL: float = 10e-6       # m  (CL thickness)
    N_CL: int   = 50          # number of FV cells (uniform mesh)

    # ── Volume fractions (sum = 1) ────────────────────────────────────────────
    eps_S: float = 0.23       # solid (Pt/C + carbon support)
    eps_G: float = 0.34       # gas-filled macro-pores
    eps_I: float = 0.43       # ionomer (Nafion)
    brugg: float = 1.5        # Bruggeman exponent

    # ── O2 diffusivity (ionomer phase) ────────────────────────────────────────
    D_O2_water:     float = 4.5e-9   # m2/s  bulk water at 80 °C
    ionomer_factor: float = 0.1      # D_ionomer = D_water * factor (Nafion suppression)

    # ── Ionic conductivity (Nafion, via collapsed Nernst-Planck) ─────────────
    D_H_water: float = 1.5e-8        # m2/s  H+ in water at 80 °C
    c_H_fixed: float = 1200.0        # mol/m3  fixed sulfonate charge concentration

    # ── Electronic conductivity ───────────────────────────────────────────────
    sigma_s_bulk: float = 100.0      # S/m  (Vulcan XC-72 / carbon support)

    # ── Henry's law (O2 in ionomer, van't Hoff fit) ───────────────────────────
    H_O2_298:    float = 1.3         # mol/(m3·atm)  at 25 °C
    H_O2_dHsol: float = 1500.0      # K  (van't Hoff sensitivity)
    p_O2_inlet:  float = 0.21        # atm  (dry-air O2 partial pressure)

    # ── ORR equilibrium potential ─────────────────────────────────────────────
    U_ORR_std: float = 1.229         # V vs SHE at 25 °C
    dU_dT:     float = -8.46e-4      # V/K  (temperature coefficient)

    # ── ORR kinetics (mutable for fitting) ───────────────────────────────────
    i0:    float = 1.0e-4            # A/m2_Pt  exchange current density
    alpha: float = 0.5               # cathodic transfer coefficient
    gamma: float = 1.0               # O2 reaction order
    a_v:   float = 2.0e7             # m^-1  volumetric Pt surface area
                                     # = (0.4 mg/cm2 × 50 m2/g) / 10 um

    # ── Boundary conditions ───────────────────────────────────────────────────
    phi_L_mem: float = 0.0           # V  Dirichlet phi_L at CL/membrane (x = L_CL)

    # ── GDL geometry (Stage 3+) ───────────────────────────────────────────────
    L_GDL: float = 200e-6      # m  (GDL thickness, Toray TGP-H-060)
    N_GDL: int   = 30          # GDL cells

    # ── Gas-phase binary O2/N2 diffusivity (Stage 3+) ────────────────────────
    D_O2_N2_bulk: float = 3.5e-5  # m2/s  (O2/N2, 80 degC, 1 atm, Chapman-Enskog)
    eps_G_gdl:    float = 0.78     # GDL macro-pore fraction (Toray carbon paper)

    # ── Derived (auto-computed; do not set manually) ──────────────────────────
    D_O2_eff:       float = field(init=False, repr=False)
    kappa_L_eff:    float = field(init=False, repr=False)
    sigma_s_eff:    float = field(init=False, repr=False)
    c_O2_bc:        float = field(init=False, repr=False)
    c_O2_ref:       float = field(init=False, repr=False)
    U_ORR_0:        float = field(init=False, repr=False)
    _H_O2:          float = field(init=False, repr=False)
    # Stage 3 derived
    D_O2_gdl_eff:    float = field(init=False, repr=False)
    D_O2_cl_gas_eff: float = field(init=False, repr=False)
    K_eq_gas_ion:    float = field(init=False, repr=False)
    c_O2_gas_inlet:  float = field(init=False, repr=False)

    def __post_init__(self):
        self.recompute()

    def recompute(self) -> None:
        """Recompute all derived quantities. Call after modifying any base parameter."""
        brg_I = self.eps_I ** self.brugg
        brg_S = self.eps_S ** self.brugg

        # O2 effective diffusivity in ionomer
        self.D_O2_eff = brg_I * self.D_O2_water * self.ionomer_factor

        # Ionic conductivity (Nernst-Planck collapsed under fixed c_H+)
        D_H_ion       = self.D_H_water * self.ionomer_factor
        kappa_bulk     = (self.F**2 / (self.R * self.T)) * self.c_H_fixed * D_H_ion
        self.kappa_L_eff = brg_I * kappa_bulk

        # Electronic conductivity
        self.sigma_s_eff = brg_S * self.sigma_s_bulk

        # Henry's law at operating temperature
        self._H_O2    = self.H_O2_298 * np.exp(
            self.H_O2_dHsol * (1.0 / self.T - 1.0 / 298.15)
        )
        self.c_O2_bc  = self._H_O2 * self.p_O2_inlet   # mol/m3  (GDL Dirichlet BC)
        self.c_O2_ref = self._H_O2 * 1.0               # mol/m3  (1 atm reference)

        # ORR equilibrium potential (temperature-corrected)
        self.U_ORR_0 = self.U_ORR_std + self.dU_dT * (self.T - 298.15)

        # Stage 3: gas-phase transport in GDL and CL pores
        self.D_O2_gdl_eff    = self.eps_G_gdl ** self.brugg * self.D_O2_N2_bulk
        self.D_O2_cl_gas_eff = self.eps_G      ** self.brugg * self.D_O2_N2_bulk
        # Equilibrium: c_O2_ion [mol/m3] = K_eq * c_O2_gas [mol/m3]
        # K_eq = H_O2 [mol/(m3*atm)] * R*T [J/mol] / 101325 [Pa/atm]
        self.K_eq_gas_ion    = self._H_O2 * self.R * self.T / 101325.0
        # Gas-channel inlet O2 concentration (ideal gas, p_O2_inlet in atm)
        self.c_O2_gas_inlet  = self.p_O2_inlet * 101325.0 / (self.R * self.T)

    def H_O2(self) -> float:
        """Henry's constant for O2 in ionomer at operating T [mol/(m3·atm)]."""
        return self._H_O2

    def U_ORR_eq(self, c_O2_local: np.ndarray) -> np.ndarray:
        """
        Nernst-corrected local ORR equilibrium potential [V vs SHE].

        U_eq = U_ORR_0 + (RT/4F) * ln(p_O2 / 1 atm)
        where p_O2 = c_O2_local / H_O2(T)
        """
        p_O2 = np.maximum(c_O2_local / self._H_O2, 1e-30)   # atm
        return self.U_ORR_0 + (self.R * self.T / (4.0 * self.F)) * np.log(p_O2)

    def print_summary(self) -> None:
        print("=" * 55)
        print("  ORR PEMFC CL Model — Parameters")
        print("=" * 55)
        print(f"  T                = {self.T:.2f} K ({self.T - 273.15:.1f} °C)")
        print(f"  L_CL             = {self.L_CL * 1e6:.1f} um,  N_CL = {self.N_CL}")
        print(f"  eps [S,G,I]      = [{self.eps_S}, {self.eps_G}, {self.eps_I}]")
        print(f"  D_O2_eff         = {self.D_O2_eff:.3e} m2/s")
        print(f"  kappa_L_eff      = {self.kappa_L_eff:.3f} S/m")
        print(f"  sigma_s_eff      = {self.sigma_s_eff:.3f} S/m")
        print(f"  c_O2_bc          = {self.c_O2_bc:.4f} mol/m3")
        print(f"  c_O2_ref         = {self.c_O2_ref:.4f} mol/m3")
        print(f"  U_ORR_0(T)       = {self.U_ORR_0:.4f} V vs SHE")
        print(f"  i0               = {self.i0:.2e} A/m2_Pt")
        print(f"  alpha_c          = {self.alpha}")
        print(f"  gamma_O2         = {self.gamma}")
        print(f"  a_v_Pt           = {self.a_v:.2e} m^-1")
        print(f"  --- Stage 3 gas-phase ---")
        print(f"  D_O2_gdl_eff     = {self.D_O2_gdl_eff:.3e} m2/s")
        print(f"  D_O2_cl_gas_eff  = {self.D_O2_cl_gas_eff:.3e} m2/s")
        print(f"  K_eq_gas_ion     = {self.K_eq_gas_ion:.4f}  (c_ion/c_gas)")
        print(f"  c_O2_gas_inlet   = {self.c_O2_gas_inlet:.4f} mol/m3")
        print("=" * 55)
