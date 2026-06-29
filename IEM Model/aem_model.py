"""
aem_model.py
============
1D continuum model for a Homogeneous Anion-Exchange Membrane (AEM)
Non-Ideal Thermodynamics + Onsager-Stefan-Maxwell Concentrated Solution Theory

Domain layout (x-axis, SI units):
  [0, L_BL]                   -- catholyte boundary layer   (comp2, domains 1)
  [L_BL, L_BL+L_AEM]         -- AEM                        (comp2, domain  2)
  [L_BL+L_AEM, 2*L_BL+L_AEM] -- anolyte boundary layer     (comp2, domain  3)

Two 0D Donnan equilibrium solvers sit at the two ES|AEM interfaces.

Species (all concentrations in mol/m³):
  ES  domains : Na⁺, OH⁻, Cl⁻, H₂O  (electroneutrality: cNa = cOH + cCl)
  AEM domain  : Na⁺, OH⁻, Cl⁻, H₂O, fixed-charge groups M⁺ (via IEC)

Solver strategy:
  Each sub-domain is a 1st-order BVP solved with scipy.integrate.solve_bvp.
  The three BVPs are coupled via Donnan equilibrium at the two interfaces.
  An outer fixed-point iteration drives interface concentrations/potential
  to self-consistency.

References:
  Newman & Thomas-Alyea, Electrochemical Systems (3rd ed.)
  Chapman, PhD Thesis (CST diffusion coefficients)
  Manning, J. Chem. Phys. (condensation theory)
"""

import numpy as np
from scipy import linalg
from scipy.optimize import fsolve
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# =============================================================================
# 1. PHYSICAL CONSTANTS
# =============================================================================

F     = 96485.0       # C/mol      Faraday constant
R     = 8.314         # J/mol/K    gas constant
k_B   = 1.3806e-23    # J/K        Boltzmann constant
e_c   = 1.6022e-19    # C          elementary charge
N_A   = 6.0221e23     # 1/mol      Avogadro
eps0  = 8.8542e-12    # F/m        vacuum permittivity
pi    = np.pi

# =============================================================================
# 2. MODEL PARAMETERS
# =============================================================================

# --- Thermodynamic conditions ---
T   = 298.15    # K
p   = 101325.0  # Pa

# --- Geometry ---
L_AEM = 50e-6   # m
L_BL  = 10e-6   # m

# --- Exchange solution (same catholyte and anolyte in symmetric case) ---
molal_total = 0.03      # mol/kg  total salt molality
rho0        = 997.0     # kg/m³   water density
M0          = 0.018015  # kg/mol  water molar mass
cH2O0       = 55000.0   # mol/m³  ~55 M
V0          = 1.0 / cH2O0  # m³/mol  molar volume of water

frac_Vx   = 0.4         # fraction of total that is NaCl (rest NaOH)
molar_tot = molal_total * rho0 / (1.0 + molal_total * M0)   # mol/m³
cES_Na    = molar_tot                           # mol/m³
cES_OH    = molar_tot * (1.0 - frac_Vx)        # mol/m³
cES_Cl    = molar_tot * frac_Vx                 # mol/m³

# --- Ion valences ---
z = {"Na": +1, "OH": -1, "Cl": -1, "M": +1}

# --- Free-solution diffusion coefficients (m²/s) ---
D_w = {"Na": 1.334e-9, "Cl": 2.032e-9, "OH": 5.260e-9, "H2O": 3.0e-9}

# --- Non-ideal parameters ---
A_dh_sq  = 1.1779**2 * 1e-3    # m³/mol  (COMSOL value 1.1779² liter/mol → ×1e-3)
B_dh_sq  = 3.291**2  * 1e15   # m/mol   (COMSOL value 3.291² liter/mol/nm² → ×1e15)
a_ion    = 0.3e-9             # m  mean ion diameter
beta_NaCl = 0.0069            # kg/mol  Pitzer
beta_NaOH = -0.0351           # kg/mol

# Solvation: k = binding constant, n = max coordination number
solv = {
    "Na":  {"k": 0.8306, "n": 5},
    "OH":  {"k": 3.2471, "n": 4},
    "Cl":  {"k": 0.14,   "n": 4},
    "M":   {"k": 0.3,    "n": 5},
}

# --- AEM membrane properties ---
IEC     = 2.35          # mol/kg  ion exchange capacity
rho_AEM = 1730.0        # kg/m³   dry membrane density
EW_AEM  = 1.0 / IEC    # kg/mol  equivalent weight
Vp_AEM  = EW_AEM / rho_AEM  # m³/mol  molar volume of dry membrane

Y0_AEM  = 250e6 * 0.5   # Pa  scaled Young's modulus
G0_AEM  = Y0_AEM / 3.0  # Pa  approx shear modulus

b_spc = 1.422e-9        # m  mean spacing between fixed charges
r_p   = 0.56e-9         # m  approx polymer chain radius (steric)

# Membrane Pitzer coefficients (ion–fixed-charge interactions)
beta_ClM = 0.11         # kg/mol
beta_OHM = 0.37         # kg/mol

# Dielectric permittivities
eps_polymer = 2.5
eps_water   = 78.5

# --- CST binary diffusion references ---
Diff_NaCl_ref = 2.631e-10   # m²/s  at I_NaCl_ref
I_NaCl_ref    = 3000.0      # mol/m³
Diff_NaOH_ref = 2.019e-10   # m²/s  at I_NaOH_ref
I_NaOH_ref    = 1500.0      # mol/m³
Diff_ClOH_ref = 0.1         # m²/s  artificially large (like-charged pair)

eta0 = 8.9e-4   # Pa·s  pure water viscosity

# Ion hydrodynamic diameters from Stokes-Einstein (m)
Di = {sp: 2.0 * k_B * T / (6.0 * pi * eta0 * D_w[sp])
      for sp in ("Na", "Cl", "OH")}

# Molar masses (kg/mol)
MW = {"Na": 0.02299, "Cl": 0.03545, "OH": 0.017008, "H2O": 0.018015}

# Total reference concentration for CST friction coefficients
c_T = 55200.0   # mol/m³ (≈ solvent + ions)

# Activation parameters for AEM diffusivities
dEa_Cl = 5000.0     # J/mol
dEa_OH = 1000.0     # J/mol
ddS_Cl = 23.1       # J/mol/K
ddS_OH = 10.0       # J/mol/K

# --- Robin coupling coefficients (artificial interface penalty) ---
K_CT = 1e4    # S/m²  charge-transfer
K_MT = 0.01   # m/s   mass-transfer

# --- Solver / numerical ---
phi_M_safe  = 1e-6   # lower bound on phi_M to avoid div/zero
lambda_safe = 1e-9   # lower bound on lambda

# --- Initial guesses for AEM interior ---
phi_mem_guess = 0.075447    # V
cH2O_mem_guess = 26140.0   # mol/m³
cCl_mem_guess  = 1950.0    # mol/m³
cOH_mem_guess  = 179.64    # mol/m³
cNa_mem_guess  = 0.1585    # mol/m³

# =============================================================================
# 3. THERMODYNAMICS — FREE SOLUTION (ES)
# =============================================================================

def sigma_dh(x):
    """
    Debye-Hückel σ-function.
    σ(x) = (3/x³) * [(1+x) - 2·ln(1+x) - 1/(1+x)]
    Returns a scalar when given a scalar, array when given an array.
    """
    scalar = np.ndim(x) == 0
    xa = np.asarray(x, dtype=float).ravel()
    out = np.where(
        xa < 1e-8,
        1.0,
        (3.0 / xa**3) * ((1.0 + xa) - 2.0 * np.log(1.0 + xa) - 1.0 / (1.0 + xa))
    )
    return float(out[0]) if scalar else out


def ionic_strength_sol(cNa, cOH, cCl):
    """I = (cNa + cOH + cCl) / 2   [mol/m³]"""
    return 0.5 * (cNa + cOH + cCl)


def dh_water_sol(cNa, cOH, cCl, cH2O):
    """
    Extended Debye-Hückel electrostatic excess chemical potential of
    water in free solution (J/mol).  Assumes only 1:1 electrolytes.
    """
    I = ionic_strength_sol(cNa, cOH, cCl)
    # Convert DH parameters to SI (m-based)
    A = np.sqrt(A_dh_sq) * 1e-3        # (m³/mol)^0.5  [COMSOL stores A²]
    B = np.sqrt(B_dh_sq) * 1e9 / 1e3   # m/mol ... TODO: check unit chain
    Ba = B * a_ion * np.sqrt(I)
    # Water contribution proportional to σ(Ba)
    mu = (np.sqrt(A_dh_sq * I**3) * R * T
          * (2.0 / 3.0) * (M0 / rho0) * sigma_dh(Ba))
    return mu


def dh_ion_sol(cNa, cOH, cCl):
    """
    Extended Debye-Hückel excess chemical potential of an ion in free
    solution (J/mol).  Same value for all 1:1 ions.
    """
    I = ionic_strength_sol(cNa, cOH, cCl)
    A = np.sqrt(A_dh_sq)
    B_a_sqI = np.sqrt(B_dh_sq * a_ion**2 * I)   # dimensionless-ish
    return -np.sqrt(A_dh_sq * I) * R * T / (1.0 + B_a_sqI)


def pitzer_water_sol(mNa, mCl, mOH):
    """
    Pitzer virial excess chemical potential of water (J/mol).
    """
    return -2.0 * M0 * R * T * (beta_NaCl * mNa * mCl
                                 + beta_NaOH * mNa * mOH)


def pitzer_ion_sol(species, mNa, mCl, mOH):
    """
    Pitzer virial excess chemical potential of ion `species` (J/mol).
    species: 'Na', 'OH', 'Cl'
    """
    if species == "Na":
        return 2.0 * R * T * (beta_NaCl * mCl + beta_NaOH * mOH)
    elif species == "Cl":
        return 2.0 * R * T * (beta_NaCl * mNa)
    elif species == "OH":
        return 2.0 * R * T * (beta_NaOH * mNa)
    else:
        raise ValueError(f"Unknown species: {species}")


def solvation_coefficients_sol(cNa, cOH, cCl, cH2O, f0):
    """
    Stokes-Robinson solvation activity coefficients for free solution.
    f0 is the self-consistent water solvation correction (scalar, solved
    via solve_solvation_sol).

    Returns dict: {"H2O": f_w, "Na": f_Na, "OH": f_OH, "Cl": f_Cl}
    """
    c_tot = cH2O + cNa + cOH + cCl
    x = {"H2O": cH2O / c_tot,
         "Na":  cNa  / c_tot,
         "OH":  cOH  / c_tot,
         "Cl":  cCl  / c_tot}

    # Average solvation numbers  h̄_i = n_i * k_i * f0 * x_H2O / (k_i * f0 * x_H2O + 1)
    h_bar = {}
    for sp in ("Na", "OH", "Cl"):
        k = solv[sp]["k"]; n = solv[sp]["n"]
        h_bar[sp] = n * k * f0 * x["H2O"] / (k * f0 * x["H2O"] + 1.0)

    # Water solvation activity
    num = 1.0 - sum(h_bar[sp] * c / cH2O
                    for sp, c in [("Na", cNa), ("OH", cOH), ("Cl", cCl)])
    den = (x["H2O"]
           + x["Na"]  * (1.0 - h_bar["Na"])
           + x["OH"]  * (1.0 - h_bar["OH"])
           + x["Cl"]  * (1.0 - h_bar["Cl"]))
    f_w = num / den

    # Ion solvation activities
    sum_help = (1.0
                + (cNa / cH2O) * (1.0 - h_bar["Na"])
                + (cOH / cH2O) * (1.0 - h_bar["OH"])
                + (cCl / cH2O) * (1.0 - h_bar["Cl"]))
    f_ion = {}
    for sp in ("Na", "OH", "Cl"):
        k = solv[sp]["k"]; n = solv[sp]["n"]
        f_ion[sp] = ((1.0 + k)**n
                     / (x["H2O"] * (1.0 + k * f_w * x["H2O"])**n * sum_help))

    return {"H2O": f_w, "Na": f_ion["Na"], "OH": f_ion["OH"], "Cl": f_ion["Cl"]}


def solve_solvation_sol(cNa, cOH, cCl, cH2O):
    """
    Self-consistently solve for the water solvation activity f_solv_0
    in free solution (COMSOL ODE3/ODE4 approach).
    Finds f0 such that  f0 = f_solv_0(cNa, cOH, cCl, cH2O, f0).
    """
    def residual(f0):
        coeffs = solvation_coefficients_sol(cNa, cOH, cCl, cH2O, f0[0])
        return [f0[0] - coeffs["H2O"]]

    f0_sol, _, converged, _ = fsolve(residual, x0=[1.0], full_output=True)
    if not converged:
        raise RuntimeError("Solvation self-consistency (ES) did not converge.")
    return float(f0_sol[0])


def mu_water_sol(cNa, cOH, cCl, cH2O, f0):
    """Total chemical potential of water in free solution (J/mol)."""
    c_tot = cH2O + cNa + cOH + cCl
    mu_id = R * T * np.log(cH2O / c_tot)
    mu_dh = dh_water_sol(cNa, cOH, cCl, cH2O)
    mNa = cNa / (M0 * cH2O); mCl = cCl / (M0 * cH2O); mOH = cOH / (M0 * cH2O)
    mu_phy = pitzer_water_sol(mNa, mCl, mOH)
    coeffs = solvation_coefficients_sol(cNa, cOH, cCl, cH2O, f0)
    mu_solv = R * T * np.log(coeffs["H2O"])
    return mu_id + mu_dh + mu_phy + mu_solv


def mu_ion_sol(species, cNa, cOH, cCl, cH2O, Phi, f0):
    """
    Total electrochemical potential of ion in free solution (J/mol).
    Reference convention: OH⁻ excess term is subtracted (see COMSOL mu_Na_sol).
    species: 'Na', 'OH', 'Cl'
    """
    c_tot = cH2O + cNa + cOH + cCl
    conc  = {"Na": cNa, "OH": cOH, "Cl": cCl}
    x_i   = conc[species] / c_tot

    mu_id  = R * T * np.log(x_i)

    mNa = cNa / (M0 * cH2O); mCl = cCl / (M0 * cH2O); mOH = cOH / (M0 * cH2O)
    mu_dh  = dh_ion_sol(cNa, cOH, cCl)
    mu_phy = pitzer_ion_sol(species, mNa, mCl, mOH)

    coeffs  = solvation_coefficients_sol(cNa, cOH, cCl, cH2O, f0)
    mu_solv = R * T * np.log(coeffs[species])

    mu_ex = mu_dh + mu_phy + mu_solv
    # subtract z_i/z_OH * mu_OH_ex  (reference elimination)
    mu_OH_ex = (dh_ion_sol(cNa, cOH, cCl)
                + pitzer_ion_sol("OH", mNa, mCl, mOH)
                + R * T * np.log(coeffs["OH"]))
    z_i = z[species]
    mu_ex_ref = mu_ex - (z_i / z["OH"]) * mu_OH_ex

    return mu_id + mu_ex_ref + z_i * F * Phi


# =============================================================================
# 4. THERMODYNAMICS — AEM MEMBRANE
# =============================================================================

def eps_eff(phi_W):
    """Effective dielectric permittivity of hydrated membrane (F/m)."""
    return eps_polymer * (1.0 - phi_W) + eps_water * phi_W


def dh_A_eff(phi_W):
    """Effective DH A constant squared (m³/mol) for membrane permittivity."""
    eps = eps_eff(phi_W)
    return (F**4 * e_c**2 * 2.0) / ((8.0 * pi)**2 * (eps * eps0 * R * T)**3)


def dh_B_eff(phi_W):
    """Effective DH B constant (m/mol) for membrane permittivity."""
    eps = eps_eff(phi_W)
    return F**2 / (eps * eps0 * R * T / 2.0)


def manning_condensation(cOH_mem, cCl_mem, cH2O_mem, phi_W):
    """
    Manning condensation model.

    Computes the condensed fractions θ_OH, θ_Cl and returns effective
    (mobile) concentrations.

    Returns
    -------
    theta_fix : float   total condensed fraction of fixed charges
    cOH_eff   : float   mol/m³  mobile OH concentration
    cCl_eff   : float   mol/m³  mobile Cl concentration
    cM_W      : float   mol/m³  total fixed-charge molar concentration
    cM_W_eff  : float   mol/m³  effective (un-condensed) fixed-charge conc
    """
    phi_M = max(1.0 - phi_W, phi_M_safe)
    lam = max(Vp_AEM / phi_M / V0 - Vp_AEM / V0, lambda_safe)
    cM_W = cH2O_mem / lam

    eps_m = eps_eff(phi_W)
    lambda_B = e_c**2 / (4.0 * pi * eps_m * eps0 * k_B * T)
    xi = lambda_B / b_spc
    xi_crit = 1.0 / abs(z["M"])
    theta_fix = max(0.0, 1.0 - xi_crit / xi)

    Q_target = theta_fix * cM_W
    alpha = solv["OH"]["k"] / solv["Cl"]["k"]   # selectivity OH vs Cl
    D_sel = cOH_mem + alpha * cCl_mem
    cOH_cond = Q_target * cOH_mem / (D_sel + 1e-30)
    cCl_cond = Q_target * alpha * cCl_mem / (D_sel + 1e-30)

    theta_OH = cOH_cond / (cOH_mem + 1e-20)
    theta_Cl = cCl_cond / (cCl_mem + 1e-20)

    cOH_eff = cOH_mem * (1.0 - theta_OH)
    cCl_eff = cCl_mem * (1.0 - theta_Cl)
    cM_W_eff = cM_W * (1.0 - theta_fix)

    return theta_fix, cOH_eff, cCl_eff, cM_W, cM_W_eff


def swelling_potential(phi_M):
    """Elastic swelling chemical potential contribution to water (J/mol)."""
    return G0_AEM * V0 * (phi_M**(1.0 / 3.0) - 0.5 * phi_M)


def solvation_coefficients_mem(cNa, cOH_eff, cCl_eff, cH2O, cM_W_eff, f0):
    """
    Stokes-Robinson solvation activity coefficients for AEM membrane.
    Extends the ES version by including the fixed-charge group M.

    f0 : float
        Self-consistent water solvation activity (from solve_solvation_mem).

    Returns dict: {"H2O": f_w, "Na": f_Na, "OH": f_OH, "Cl": f_Cl}
    """
    c_tot = cH2O + cNa + cOH_eff + cCl_eff + cM_W_eff
    x_H2O = max(cH2O, 1.0) / c_tot
    x = {
        "Na": max(cNa,     1e-6) / c_tot,
        "OH": max(cOH_eff, 1e-6) / c_tot,
        "Cl": max(cCl_eff, 1e-6) / c_tot,
        "M":  cM_W_eff           / c_tot,
    }

    # Average solvation numbers  h̄_i = n_i k_i f0 x_w / (k_i f0 x_w + 1)
    h_bar = {}
    for sp in ("Na", "OH", "Cl", "M"):
        k = solv[sp]["k"]; n = solv[sp]["n"]
        h_bar[sp] = n * k * f0 * x_H2O / (k * f0 * x_H2O + 1.0)

    # Water solvation activity coefficient
    conc = {"Na": cNa, "OH": cOH_eff, "Cl": cCl_eff, "M": cM_W_eff}
    num = 1.0 - sum(h_bar[sp] * conc[sp] / cH2O for sp in conc)
    den = (x_H2O
           + x["Na"] * (1.0 - h_bar["Na"])
           + x["OH"] * (1.0 - h_bar["OH"])
           + x["Cl"] * (1.0 - h_bar["Cl"])
           + x["M"]  * (1.0 - h_bar["M"]))
    f_w = num / den

    # sum_help: denominator for ion activity coefficients
    sum_help = (1.0
                + (cNa     / cH2O) * (1.0 - h_bar["Na"])
                + (cOH_eff / cH2O) * (1.0 - h_bar["OH"])
                + (cCl_eff / cH2O) * (1.0 - h_bar["Cl"])
                + (cM_W_eff / cH2O) * (1.0 - h_bar["M"]))

    # Ion solvation activity coefficients
    f_ion = {}
    for sp in ("Na", "OH", "Cl"):
        k = solv[sp]["k"]; n = solv[sp]["n"]
        f_ion[sp] = ((1.0 + k)**n
                     / (x_H2O * (1.0 + k * f_w * x_H2O)**n * sum_help))

    return {"H2O": f_w, "Na": f_ion["Na"], "OH": f_ion["OH"], "Cl": f_ion["Cl"]}


def solve_solvation_mem(cNa, cOH_eff, cCl_eff, cH2O, cM_W_eff):
    """
    Self-consistently solve for the water solvation activity f_solv_0
    in the AEM membrane (COMSOL ODE4 analogue).
    Finds f0 such that  f0 = f_w(f0)  from solvation_coefficients_mem.
    """
    def residual(f0):
        coeffs = solvation_coefficients_mem(cNa, cOH_eff, cCl_eff,
                                            cH2O, cM_W_eff, f0[0])
        return [f0[0] - coeffs["H2O"]]

    f0_sol, _, converged, _ = fsolve(residual, x0=[1.0], full_output=True)
    if not converged:
        raise RuntimeError("Solvation self-consistency (AEM) did not converge.")
    return float(f0_sol[0])


def mu_water_mem(cNa, cOH_eff, cCl_eff, cH2O, cM_W_eff, phi_W, f0_mem):
    """
    Total chemical potential of water in AEM (J/mol).
    Includes DH (membrane permittivity), Pitzer (ion-ion + ion-M),
    solvation, and elastic swelling.
    """
    phi_M = max(1.0 - phi_W, phi_M_safe)

    # Molalities (mol / kg_water)
    mNa    = cNa     / (M0 * cH2O)
    mCl    = cCl_eff / (M0 * cH2O)
    mOH    = cOH_eff / (M0 * cH2O)
    mM_eff = cM_W_eff / (M0 * cH2O)

    # Extended DH with membrane effective permittivity
    A_eff = dh_A_eff(phi_W)
    B_eff = dh_B_eff(phi_W)
    I_mem = max((cOH_eff + cNa + cCl_eff) / 2.0 / phi_W, 1e-6)
    Ba    = np.sqrt(B_eff * a_ion**2 * I_mem)

    mu_dh_t1 = (np.sqrt(A_eff * I_mem**3) * R * T
                * (2.0 / 3.0) * (M0 / rho0) * sigma_dh(Ba))
    mu_dh_t2 = np.sqrt(A_eff / B_eff) * R * T * M0 * mM_eff / b_spc
    mu_dh    = mu_dh_t1 + mu_dh_t2

    # Pitzer virial (ion-ion and ion-fixed-charge interactions)
    mu_phy = -2.0 * M0 * R * T * (
        beta_NaCl * mNa * mCl
        + beta_ClM * mCl * mM_eff
        + beta_NaOH * mOH * mNa
        + beta_OHM * mOH * mM_eff
    )

    # Solvation activity of water
    coeffs  = solvation_coefficients_mem(cNa, cOH_eff, cCl_eff,
                                         cH2O, cM_W_eff, f0_mem)
    mu_solv = R * T * np.log(coeffs["H2O"])

    # Elastic swelling
    mu_swell = swelling_potential(phi_M)

    # Ideal mixing (mole fraction basis)
    c_tot = cH2O + cNa + cOH_eff + cCl_eff + cM_W_eff
    mu_id = R * T * np.log(cH2O / c_tot)

    return mu_id + mu_dh + mu_phy + mu_solv + mu_swell


def theta_fix_from_cM(cM_W_eff, cM_W):
    """
    Reverse-compute Manning condensed fraction from effective and total
    fixed-charge concentrations.  Utility for diagnostics.
    """
    return max(0.0, 1.0 - cM_W_eff / (cM_W + 1e-30))


def mu_ion_mem(species, cNa, cOH_eff, cCl_eff, cH2O, cM_W_eff,
               phi_W, Phi_mem, f0_mem):
    """
    Total electrochemical potential of ion in AEM (J/mol).

    Contributions
    -------------
    mu_id    : ideal mixing (mole fraction, includes M in total)
    mu_dh    : extended Debye-Hückel with membrane effective permittivity
    mu_phy   : Pitzer virial (ion-ion + ion–fixed-charge group)
    mu_solv  : Stokes-Robinson solvation activity
    mu_stc   : steric exclusion by polymer matrix
    electric : z_i * F * Phi_mem

    Reference convention (same as ES phase): OH⁻ excess subtracted so
    that mu_OH carries no excess term and single-ion standard states cancel.

    species: 'Na', 'OH', 'Cl'
    """
    phi_M = max(1.0 - phi_W, phi_M_safe)

    # Molalities
    mNa    = cNa     / (M0 * cH2O)
    mOH    = cOH_eff / (M0 * cH2O)
    mCl    = cCl_eff / (M0 * cH2O)
    mM_eff = cM_W_eff / (M0 * cH2O)

    # Mole fraction of target species (M included in total)
    c_tot = cH2O + cNa + cOH_eff + cCl_eff + cM_W_eff
    x_i   = max({"Na": cNa, "OH": cOH_eff, "Cl": cCl_eff}[species], 1e-6) / c_tot
    mu_id = R * T * np.log(x_i)

    # Extended DH — same value for all 1:1 ions at this ionic strength
    A_eff = dh_A_eff(phi_W)
    B_eff = dh_B_eff(phi_W)
    I_mem = max((cOH_eff + cNa + cCl_eff) / 2.0 / phi_W, 1e-6)
    Ba    = np.sqrt(B_eff * a_ion**2 * I_mem)

    mu_dh_t1 = -np.sqrt(A_eff * I_mem) * R * T / (1.0 + Ba)
    mu_dh_t2 = (-np.sqrt(A_eff / B_eff) * R * T
                * mM_eff * rho0 / (2.0 * b_spc * I_mem))
    mu_dh_i  = mu_dh_t1 + mu_dh_t2

    # Pitzer (species-specific)
    phy = {
        "Na": 2.0 * R * T * (beta_NaCl * mCl  + beta_NaOH * mOH),
        "Cl": 2.0 * R * T * (beta_NaCl * mNa  + beta_ClM  * mM_eff),
        "OH": 2.0 * R * T * (beta_NaOH * mNa  + beta_OHM  * mM_eff),
    }
    mu_phy_i = phy[species]

    # Solvation
    coeffs    = solvation_coefficients_mem(cNa, cOH_eff, cCl_eff,
                                           cH2O, cM_W_eff, f0_mem)
    mu_solv_i = R * T * np.log(coeffs[species])

    # Steric exclusion: r_Na = r_Cl = r_OH = a_ion/2  (COMSOL convention)
    r_i     = a_ion / 2.0
    mu_stc_i = R * T * phi_M * (1.0 + r_i / r_p)**3

    # Total excess for this species
    mu_ex_i = mu_dh_i + mu_phy_i + mu_solv_i + mu_stc_i

    # OH excess — reference species for standard-state elimination
    # (all ions share the same DH and steric terms at this I and phi_M)
    mu_OH_ex = (mu_dh_i
                + phy["OH"]
                + R * T * np.log(coeffs["OH"])
                + R * T * phi_M * (1.0 + r_i / r_p)**3)

    # Subtract z_i/z_OH * mu_OH_ex  →  mu_OH_ex_ref = 0 by construction
    z_i       = z[species]
    mu_ex_ref = mu_ex_i - (z_i / z["OH"]) * mu_OH_ex

    return mu_id + mu_ex_ref + z_i * F * Phi_mem


# =============================================================================
# 5. DONNAN EQUILIBRIUM SOLVER (0D)
# =============================================================================

def solve_donnan(cNa_sol, cOH_sol, cCl_sol, cH2O_sol, Phi_sol,
                 x0=None):
    """
    Solve 0D Donnan equilibrium at one ES|AEM interface.

    Solves the five-equation system (mirrors COMSOL comp1 / comp3 ODEs):
      μ_OH(sol)  = μ_OH(mem)           [electrochemical potential, OH]
      μ_Na(sol)  = μ_Na(mem)           [electrochemical potential, Na]
      μ_Cl(sol)  = μ_Cl(mem)           [electrochemical potential, Cl]
      μ_H2O(sol) = μ_H2O(mem)          [chemical potential, water]
      ρ_charge_mem = 0                  [electroneutrality incl. fixed charge]

    Implementation notes
    --------------------
    Concentrations are solved in **log-space** (v = ln c) to guarantee
    positivity throughout the iteration.  Residuals are scaled to O(1)
    by dividing by RT (potentials) and a representative concentration
    (electroneutrality).  The solvation activity f0_mem is solved
    self-consistently inside each residual evaluation.

    Parameters
    ----------
    cNa_sol, cOH_sol, cCl_sol, cH2O_sol : float (mol/m³)
        Solution-side concentrations at the interface.
    Phi_sol : float (V)
        Solution-side electric potential at the interface.
    x0 : array-like (5,), optional
        Initial guess in the internal parameterisation:
        [ln(cOH_mem), ln(cNa_mem), ln(cCl_mem), ln(cH2O_mem), Phi_mem].
        Defaults to the COMSOL parameter-file values.

    Returns
    -------
    dict with keys: cOH_mem, cNa_mem, cCl_mem, cH2O_mem, Phi_mem (all SI)
    """
    # --- Default initial guess from COMSOL parameter table (section 1.1.5) ---
    if x0 is None:
        x0 = [np.log(cOH_mem_guess),    # ln(cOH_mem)  = ln(179.64 mol/m³)
              np.log(cNa_mem_guess),    # ln(cNa_mem)  = ln(0.1585 mol/m³)
              np.log(cCl_mem_guess),    # ln(cCl_mem)  = ln(1950   mol/m³)
              np.log(cH2O_mem_guess),   # ln(cH2O_mem) = ln(26140  mol/m³)
              phi_mem_guess]            # Phi_mem  =  0.075447 V

    # --- Precompute solution-side quantities (fixed during the solve) ---
    f0_sol        = solve_solvation_sol(cNa_sol, cOH_sol, cCl_sol, cH2O_sol)
    mu_OH_ref     = mu_ion_sol("OH",  cNa_sol, cOH_sol, cCl_sol, cH2O_sol, Phi_sol, f0_sol)
    mu_Na_ref     = mu_ion_sol("Na",  cNa_sol, cOH_sol, cCl_sol, cH2O_sol, Phi_sol, f0_sol)
    mu_Cl_ref     = mu_ion_sol("Cl",  cNa_sol, cOH_sol, cCl_sol, cH2O_sol, Phi_sol, f0_sol)
    mu_H2O_ref    = mu_water_sol(cNa_sol, cOH_sol, cCl_sol, cH2O_sol, f0_sol)

    # Scale for electroneutrality residual: use solution ionic strength as reference
    c_scale = max(cOH_sol + cCl_sol, 1.0)   # mol/m³, O(cES)

    # --- Residual function (log-space parameterisation) ---
    def residuals(v):
        lOH, lNa, lCl, lH2O, Phi_m = v

        # Exponentiate to recover physical concentrations (always positive)
        cOH_m  = np.exp(lOH)
        cNa_m  = np.exp(lNa)
        cCl_m  = np.exp(lCl)
        cH2O_m = np.exp(lH2O)

        phi_W_m = cH2O_m / cH2O0

        # Manning condensation → effective mobile concentrations
        _, cOH_eff, cCl_eff, cM_W, cM_W_eff = manning_condensation(
            cOH_m, cCl_m, cH2O_m, phi_W_m)

        # Self-consistent solvation activity at this membrane state
        f0_mem = solve_solvation_mem(cNa_m, cOH_eff, cCl_eff, cH2O_m, cM_W_eff)

        # Electrochemical potential residuals, scaled by RT → dimensionless
        r_OH  = (mu_OH_ref  - mu_ion_mem("OH", cNa_m, cOH_eff, cCl_eff,
                                          cH2O_m, cM_W_eff, phi_W_m, Phi_m, f0_mem)) / (R * T)
        r_Na  = (mu_Na_ref  - mu_ion_mem("Na", cNa_m, cOH_eff, cCl_eff,
                                          cH2O_m, cM_W_eff, phi_W_m, Phi_m, f0_mem)) / (R * T)
        r_Cl  = (mu_Cl_ref  - mu_ion_mem("Cl", cNa_m, cOH_eff, cCl_eff,
                                          cH2O_m, cM_W_eff, phi_W_m, Phi_m, f0_mem)) / (R * T)
        r_H2O = (mu_H2O_ref - mu_water_mem(cNa_m, cOH_eff, cCl_eff,
                                            cH2O_m, cM_W_eff, phi_W_m, f0_mem)) / (R * T)

        # Electroneutrality: (−cOH_eff + cNa − cCl_eff + cM_W_eff) / c_scale
        rho_b = (z["OH"] * cOH_eff + z["Na"] * cNa_m
                 + z["Cl"] * cCl_eff + z["M"] * cM_W_eff) / c_scale

        return [r_OH, r_Na, r_Cl, r_H2O, rho_b]

    # --- Solve ---
    v_sol, info, ier, msg = fsolve(residuals, x0, full_output=True)

    if ier != 1:
        raise RuntimeError(
            f"Donnan solve did not converge (ier={ier}): {msg}\n"
            f"  Final residual norm: {np.linalg.norm(info['fvec']):.3e}"
        )

    lOH, lNa, lCl, lH2O, Phi_m = v_sol
    return {
        "cOH_mem":  np.exp(lOH),
        "cNa_mem":  np.exp(lNa),
        "cCl_mem":  np.exp(lCl),
        "cH2O_mem": np.exp(lH2O),
        "Phi_mem":  Phi_m,
    }


# =============================================================================
# 6. AEM TRANSPORT — FRICTION MATRIX AND ONSAGER COEFFICIENTS
# =============================================================================

def mackie_meares(phi_W):
    """Mackie-Meares tortuosity/geometric correction factor."""
    phi_M = 1.0 - phi_W
    return phi_W * ((1.0 - phi_M) / (1.0 + phi_M))**2


def einstein_viscosity(cNa_p, cCl_p, cOH_p):
    """
    Einstein viscosity correction for concentrated pore fluid (Pa·s).
    Uses Krieger-Dougherty-like formula with hydrodynamic volumes.
    """
    V_vis = {sp: N_A * (pi / 6.0) * Di[sp]**3 for sp in ("Na", "Cl", "OH")}
    sumCV = cNa_p * V_vis["Na"] + cCl_p * V_vis["Cl"] + cOH_p * V_vis["OH"]
    sumCV_safe = min(sumCV, 0.95)
    return eta0 * (1.0 + 0.5 * sumCV_safe) / (1.0 - sumCV_safe)**2


def binary_diffusivities(cNa_p, cCl_p, cOH_p, cM_p, phi_W, cH2O_p):
    """
    All pairwise binary diffusivities (m²/s) in AEM pore space.

    Returns dict with keys:
      NaCl, NaOH, ClOH, Na0, Cl0, OH0, NaM, ClM, OHM, 0M
    (0 denotes water)
    """
    scal = mackie_meares(phi_W)
    I_mol = 0.5 * (cNa_p * z["Na"]**2 + cCl_p * z["Cl"]**2
                   + cOH_p * z["OH"]**2)
    eta_E = einstein_viscosity(cNa_p, cCl_p, cOH_p)

    D = {}
    D["NaCl"] = scal * Diff_NaCl_ref * np.sqrt(I_mol / I_NaCl_ref)
    D["NaOH"] = scal * Diff_NaOH_ref * np.sqrt(I_mol / I_NaOH_ref)
    D["ClOH"] = Diff_ClOH_ref                        # like-charge: artificially high

    D["Na0"]  = scal * (eta0 / eta_E) * D_w["Na"]
    D["Cl0"]  = scal * (eta0 / eta_E) * D_w["Cl"]
    D["OH0"]  = scal * (eta0 / eta_E) * D_w["OH"]

    D["NaM"]  = scal * D_w["Na"]
    D["ClM"]  = scal * D_w["Cl"]  * np.exp(-dEa_Cl / (R * T)) * np.exp(ddS_Cl / R)
    D["OHM"]  = scal * D_w["OH"]  * np.exp(-dEa_OH / (R * T)) * np.exp(ddS_OH / R)

    # Water-membrane: reduced by fraction of non-solvated water
    # theta_0M = 1 - (cM_W_eff / cH2O) * h_bar_M  (TODO: hook up solvation)
    theta_0M = 1.0   # placeholder
    D["0M"]   = scal * theta_0M * D_w["H2O"]

    return D


def friction_K(ci_p, cj_p, D_ij):
    """K_ij = RT * ci * cj / (D_ij * c_T)  [kg/m³/s]"""
    return R * T * ci_p * cj_p / (D_ij * c_T)


def build_friction_matrix(cNa_p, cCl_p, cOH_p, c0_p, cM_p, D):
    """
    Build 4×4 Stefan-Maxwell friction matrix M (kg/m³/s).
    Species order: [H2O(0), Na(1), Cl(2), OH(3)]

    Off-diagonal: M_ij = K_ij
    Diagonal:     M_ii = -Σ_{j≠i} K_ij
    """
    c = [c0_p, cNa_p, cCl_p, cOH_p]
    pairs = {
        (0, 1): D["Na0"],  (0, 2): D["Cl0"],  (0, 3): D["OH0"],
        (1, 2): D["NaCl"], (1, 3): D["NaOH"], (2, 3): D["ClOH"],
    }
    # Membrane coupling K_iM (off-diagonal entries in extended matrix;
    # here folded into effective diagonal)
    K_iM = {
        0: friction_K(c0_p,   cM_p, D["0M"]),
        1: friction_K(cNa_p,  cM_p, D["NaM"]),
        2: friction_K(cCl_p,  cM_p, D["ClM"]),
        3: friction_K(cOH_p,  cM_p, D["OHM"]),
    }

    M = np.zeros((4, 4))
    for (i, j), D_ij in pairs.items():
        K = friction_K(c[i], c[j], D_ij)
        M[i, j] = K
        M[j, i] = K

    for i in range(4):
        M[i, i] = -(np.sum(M[i, :]) + K_iM[i])

    return M


def onsager_L(M):
    """
    Onsager transport matrix L (m³·s/kg).
    L = -M⁻¹
    Species order: [H2O, Na, Cl, OH]
    """
    return -linalg.inv(M)


def cst_fluxes(L, c_pore, grad_mu):
    """
    Molar fluxes from Onsager-Stefan-Maxwell theory (mol/m²/s).

    N_i = -Σ_j  L_ij * c_i * c_j * grad_mu_j

    Parameters
    ----------
    L       : (4,4) array  Onsager coefficients
    c_pore  : (4,) array   [c0, cNa, cCl, cOH]  interstitial mol/m³
    grad_mu : (4,) array   [∂μ₀/∂x, ∂μ_Na/∂x, ∂μ_Cl/∂x, ∂μ_OH/∂x]  J/mol/m

    Returns
    -------
    N : (4,) array  [N_H2O, N_Na, N_Cl, N_OH]  mol/m²/s
    """
    N = np.zeros(4)
    for i in range(4):
        for j in range(4):
            N[i] -= L[i, j] * c_pore[i] * c_pore[j] * grad_mu[j]
    return N


# =============================================================================
# 7. BVP — ODE RIGHT-HAND SIDES
# =============================================================================
#
# State vector conventions
# ------------------------
#
#   ES sub-domains (catholyte, anolyte):
#     y[0] = cOH_sol   (mol/m³)
#     y[1] = cCl_sol   (mol/m³)
#     y[2] = cH2O_sol  (mol/m³)
#     y[3] = Phi_sol   (V)
#     y[4] = log_f_solv (dimensionless — solvation auxiliary)
#     cNa_sol = y[0] + y[1]  (electroneutrality)
#
#   AEM sub-domain:
#     y[0] = cOH_mem   (mol/m³)   [raw, before Manning reduction]
#     y[1] = cCl_mem   (mol/m³)
#     y[2] = cH2O_mem  (mol/m³)
#     y[3] = Phi_mem   (V)
#     y[4] = log_f_solv_mem
#     y[5] = cNa_mem   (mol/m³)
#
# Each conservation law ∇·N_i = 0 in steady-state means N_i = const.
# We treat fluxes as parameters passed between sub-domains at interfaces.
# Within each domain, N_i = const yields a 1st-order algebraic+ODE system;
# the ODE is for the potential Phi (current conservation) plus the
# concentrations driven by the flux constitutive relations.

def ode_es(x, y, fluxes):
    """
    ODE RHS for one ES sub-domain.

    Parameters
    ----------
    x      : float or (n,) array  spatial coordinate (m)
    y      : (5, n) array          state vector columns
    fluxes : dict with keys N_OH, N_Na, N_Cl, N_H2O (mol/m²/s, constants)

    Returns
    -------
    dydx : (5, n) array
    """
    # TODO: invert non-ideal flux constitutive relations to get
    #       dc/dx and dPhi/dx consistent with constant fluxes.
    raise NotImplementedError("ode_es not yet implemented")


def ode_aem(x, y):
    """
    ODE RHS for AEM sub-domain.

    Parameters
    ----------
    x : float or (n,) array
    y : (6, n) array

    Returns
    -------
    dydx : (6, n) array

    Notes
    -----
    1. Compute pore concentrations from superficial (/ phi_W).
    2. Apply Manning condensation to get effective concentrations.
    3. Compute chemical potential gradients from y-derivatives (implicit).
    4. Build friction matrix M, invert to L, compute CST fluxes.
    5. Enforce ∇·N_i = 0 → residual on flux divergence (or integrate flux const).
    6. Electroneutrality is enforced as a pointwise constraint on cNa_mem.
    """
    raise NotImplementedError("ode_aem not yet implemented")


# =============================================================================
# 8. BVP — BOUNDARY CONDITIONS
# =============================================================================

def bc_catholyte(ya, yb, Vapp, donnan_left):
    """
    BCs for catholyte BL.

    ya (x=0) : Dirichlet — bulk feed concentrations + Phi = 0
    yb (x=L_BL) : Robin flux continuity with left AEM interface
    """
    # --- Left (bulk feed) ---
    r_OH_a  = ya[0] - cES_OH
    r_Cl_a  = ya[1] - cES_Cl
    r_H2O_a = ya[2] - cH2O0
    r_Phi_a = ya[3] - 0.0        # reference potential

    # --- Right (interface with AEM, Robin penalty) ---
    # K_MT*(c_interface_donnan - c_AEM_side) drives concentrations to match
    # K_CT*(Phi_interface - Phi_AEM_side) drives potential
    r_f = ya[4] - ya[4]          # solvation: zero flux (natural)

    return np.array([r_OH_a, r_Cl_a, r_H2O_a, r_Phi_a, r_f])


def bc_anolyte(ya, yb, Vapp, donnan_right):
    """
    BCs for anolyte BL.

    ya (x=L_BL+L_AEM) : Robin flux continuity with right AEM interface
    yb (x=2*L_BL+L_AEM): Dirichlet — bulk + Phi = Vapp
    """
    r_OH_b  = yb[0] - cES_OH
    r_Cl_b  = yb[1] - cES_Cl
    r_H2O_b = yb[2] - cH2O0
    r_Phi_b = yb[3] - Vapp
    r_f     = yb[4] - yb[4]

    return np.array([r_OH_b, r_Cl_b, r_H2O_b, r_Phi_b, r_f])


def bc_aem(ya, yb, donnan_left, donnan_right):
    """
    BCs for AEM sub-domain.

    ya : membrane concentrations/potential matched to left Donnan solution
    yb : matched to right Donnan solution
    """
    r = np.zeros(6)
    r[0] = ya[0] - donnan_left["cOH_mem"]
    r[1] = ya[1] - donnan_left["cCl_mem"]
    r[2] = ya[2] - donnan_left["cH2O_mem"]
    r[3] = ya[3] - donnan_left["Phi_mem"]
    r[4] = yb[3] - donnan_right["Phi_mem"]   # potential at right boundary
    r[5] = ya[5] - donnan_left["cNa_mem"]
    return r


# =============================================================================
# 9. COUPLED SOLVER (single Vapp)
# =============================================================================

def build_initial_mesh(n_es=50, n_aem=100):
    """Return x-arrays for each sub-domain."""
    x_cat = np.linspace(0.0, L_BL, n_es)
    x_aem = np.linspace(L_BL, L_BL + L_AEM, n_aem)
    x_ano = np.linspace(L_BL + L_AEM, 2.0 * L_BL + L_AEM, n_es)
    return x_cat, x_aem, x_ano


def initial_guess_es(x, Phi_left, Phi_right):
    """Linear interpolation of bulk conditions for ES domain initial guess."""
    n = len(x)
    Phi = np.linspace(Phi_left, Phi_right, n)
    y = np.zeros((5, n))
    y[0] = cES_OH
    y[1] = cES_Cl
    y[2] = cH2O0
    y[3] = Phi
    y[4] = 0.0   # log_f_solv
    return y


def initial_guess_aem(x):
    """Constant AEM initial guess from user-supplied values."""
    n = len(x)
    y = np.zeros((6, n))
    y[0] = cOH_mem_guess
    y[1] = cCl_mem_guess
    y[2] = cH2O_mem_guess
    y[3] = phi_mem_guess
    y[4] = 0.0
    y[5] = cNa_mem_guess
    return y


def solve_one_voltage(Vapp, prev_solution=None, verbose=False,
                      max_iter=20, tol=1e-6):
    """
    Full coupled solve for a single applied voltage Vapp (V).

    Algorithm
    ---------
    1. Build x-meshes for all three sub-domains.
    2. Set initial guesses (or use prev_solution for continuation).
    3. Solve Donnan equilibrium at both interfaces.
    4. Solve catholyte BVP  → extract interface fluxes.
    5. Solve anolyte  BVP  → extract interface fluxes.
    6. Solve AEM BVP        → check flux balance.
    7. Check convergence; repeat 3–6 until converged.

    Returns
    -------
    dict with fields:
        x_cat, y_cat   catholyte solution
        x_aem, y_aem   AEM solution
        x_ano, y_ano   anolyte solution
        donnan_left    left interface Donnan dict
        donnan_right   right interface Donnan dict
        i_total        current density at centre of AEM (A/m²)
        converged      bool
    """
    x_cat, x_aem, x_ano = build_initial_mesh()

    if prev_solution is None:
        y_cat = initial_guess_es(x_cat, 0.0, 0.0)
        y_aem = initial_guess_aem(x_aem)
        y_ano = initial_guess_es(x_ano, 0.0, Vapp)
    else:
        y_cat = prev_solution["y_cat"]
        y_aem = prev_solution["y_aem"]
        y_ano = prev_solution["y_ano"]

    converged = False
    for iteration in range(max_iter):
        # --- Interface concentrations for Donnan solve ---
        cOH_left  = y_cat[0, -1]; cCl_left  = y_cat[1, -1]
        cH2O_left = y_cat[2, -1]; Phi_left  = y_cat[3, -1]
        cNa_left  = cOH_left + cCl_left

        cOH_right  = y_ano[0, 0]; cCl_right  = y_ano[1, 0]
        cH2O_right = y_ano[2, 0]; Phi_right  = y_ano[3, 0]
        cNa_right  = cOH_right + cCl_right

        try:
            donnan_L = solve_donnan(cNa_left, cOH_left, cCl_left,
                                    cH2O_left, Phi_left)
            donnan_R = solve_donnan(cNa_right, cOH_right, cCl_right,
                                    cH2O_right, Phi_right)
        except (RuntimeError, NotImplementedError):
            if verbose:
                print(f"  iter {iteration}: Donnan solve failed, using guesses")
            donnan_L = {"cOH_mem": cOH_mem_guess, "cNa_mem": cNa_mem_guess,
                        "cCl_mem": cCl_mem_guess, "cH2O_mem": cH2O_mem_guess,
                        "Phi_mem": phi_mem_guess}
            donnan_R = donnan_L.copy()

        # --- Catholyte BVP ---
        try:
            sol_cat = solve_bvp(
                fun=lambda x, y: ode_es(x, y, fluxes=None),   # fluxes TBD
                bc=lambda ya, yb: bc_catholyte(ya, yb, Vapp, donnan_L),
                x=x_cat, y=y_cat, tol=1e-4, verbose=0
            )
            y_cat = sol_cat.sol(x_cat)
        except (NotImplementedError, Exception) as err:
            if verbose:
                print(f"  iter {iteration}: catholyte BVP: {err}")

        # --- Anolyte BVP ---
        try:
            sol_ano = solve_bvp(
                fun=lambda x, y: ode_es(x, y, fluxes=None),
                bc=lambda ya, yb: bc_anolyte(ya, yb, Vapp, donnan_R),
                x=x_ano, y=y_ano, tol=1e-4, verbose=0
            )
            y_ano = sol_ano.sol(x_ano)
        except (NotImplementedError, Exception) as err:
            if verbose:
                print(f"  iter {iteration}: anolyte BVP: {err}")

        # --- AEM BVP ---
        try:
            sol_aem = solve_bvp(
                fun=lambda x, y: ode_aem(x, y),
                bc=lambda ya, yb: bc_aem(ya, yb, donnan_L, donnan_R),
                x=x_aem, y=y_aem, tol=1e-4, verbose=0
            )
            y_aem = sol_aem.sol(x_aem)
            converged = sol_aem.success
        except (NotImplementedError, Exception) as err:
            if verbose:
                print(f"  iter {iteration}: AEM BVP: {err}")
            break

        if verbose:
            print(f"  iter {iteration}: done")
        break   # remove once ODE stubs are implemented

    return {
        "x_cat": x_cat, "y_cat": y_cat,
        "x_aem": x_aem, "y_aem": y_aem,
        "x_ano": x_ano, "y_ano": y_ano,
        "donnan_left":  donnan_L,
        "donnan_right": donnan_R,
        "i_total": None,   # extracted from AEM flux once implemented
        "converged": converged,
        "Vapp": Vapp,
    }


# =============================================================================
# 10. PARAMETRIC SWEEP
# =============================================================================

def polarization_curve(Vapp_list=None):
    """
    Sweep Vapp and collect current density at each voltage.
    Uses solution from previous step as initial guess (continuation).

    Returns list of result dicts (one per voltage).
    """
    if Vapp_list is None:
        Vapp_list = np.arange(0.0, 0.151, 0.001)

    results = []
    prev = None
    for V in Vapp_list:
        sol = solve_one_voltage(V, prev_solution=prev, verbose=True)
        results.append(sol)
        if sol["converged"]:
            prev = sol
    return results


# =============================================================================
# 11. POST-PROCESSING
# =============================================================================

def concat_domains(result):
    """Stack catholyte + AEM + anolyte into single arrays for plotting."""
    x  = np.concatenate([result["x_cat"],
                         result["x_aem"],
                         result["x_ano"]])
    # ES y has 5 rows, AEM has 6; we plot species common to both.
    # Return raw per-domain arrays and let caller decide.
    return x, result["y_cat"], result["y_aem"], result["y_ano"]


def plot_profiles(result):
    """Concentration and potential profiles across all three sub-domains."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Vapp = {result['Vapp'] * 1000:.1f} mV")

    x_cat = result["x_cat"] * 1e6   # µm
    x_aem = result["x_aem"] * 1e6
    x_ano = result["x_ano"] * 1e6

    y_cat = result["y_cat"]
    y_aem = result["y_aem"]
    y_ano = result["y_ano"]

    ax = axes[0, 0]
    ax.plot(x_cat, y_cat[0] / 1e3, label="catholyte")
    ax.plot(x_aem, y_aem[0] / 1e3, label="AEM")
    ax.plot(x_ano, y_ano[0] / 1e3, label="anolyte")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("c_OH (mM)"); ax.legend()

    ax = axes[0, 1]
    ax.plot(x_cat, y_cat[1] / 1e3)
    ax.plot(x_aem, y_aem[1] / 1e3)
    ax.plot(x_ano, y_ano[1] / 1e3)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("c_Cl (mM)")

    ax = axes[1, 0]
    ax.plot(x_cat, y_cat[3] * 1e3)
    ax.plot(x_aem, y_aem[3] * 1e3)
    ax.plot(x_ano, y_ano[3] * 1e3)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("Φ (mV)")

    ax = axes[1, 1]
    ax.plot(x_cat, y_cat[2] / 1e3)
    ax.plot(x_aem, y_aem[2] / 1e3)
    ax.plot(x_ano, y_ano[2] / 1e3)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("c_H2O (mM)")

    plt.tight_layout()
    return fig


def plot_polarization(results):
    """Plot I-V curve from a list of result dicts."""
    V_list = [r["Vapp"] * 1e3 for r in results]
    i_list = [r["i_total"] for r in results if r["i_total"] is not None]
    if not i_list:
        print("No current data yet — implement ode_aem first.")
        return
    fig, ax = plt.subplots()
    ax.plot(V_list[:len(i_list)], i_list)
    ax.set_xlabel("Vapp (mV)"); ax.set_ylabel("i (A/m²)")
    ax.set_title("Polarization curve")
    return fig


# =============================================================================
# 12. MAIN
# =============================================================================

if __name__ == "__main__":
    # Quick single-voltage test (will raise NotImplementedError until
    # ode_es and ode_aem are filled in)
    print("Running single-voltage solve at Vapp = 1 mV ...")
    try:
        sol = solve_one_voltage(Vapp=0.001, verbose=True)
        print("Donnan left :", sol["donnan_left"])
        print("Donnan right:", sol["donnan_right"])
        plot_profiles(sol)
        plt.show()
    except NotImplementedError as e:
        print(f"Stub not yet implemented: {e}")

    # Full sweep (uncomment once ODE stubs are filled):
    # results = polarization_curve()
    # plot_polarization(results)
    # plt.show()
