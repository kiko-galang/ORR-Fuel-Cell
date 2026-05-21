"""
Plotting routines for Stage 1 ORR PEMFC CL model results.

Follows the PEMFC convention (§15.5):
    y-axis : V_cathode [V vs SHE], increasing upward
    x-axis : current density [A/cm2], increasing rightward
"""
from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from assembly_stage1 import unpack, compute_current, current_from_flux
from kinetics import R_ORR

matplotlib.rcParams.update({
    "font.size":    11,
    "axes.labelsize": 12,
    "figure.dpi":   120,
})


def _currents_mA_cm2(voltages, solutions, mesh, p) -> np.ndarray:
    """Compute current density [mA/cm2] for each voltage point."""
    return np.array([
        compute_current(u, mesh, p) * 1e-4 * 1e3   # A/m2 -> mA/cm2
        for u in solutions
    ])


# ── 1. Polarization curve ─────────────────────────────────────────────────────

def plot_polarization(
    voltages:  list[float],
    solutions: list[np.ndarray],
    mesh,
    p,
    ax=None,
    label: str = "Stage 1 model",
    save_path: str | None = "polarization.png",
) -> plt.Axes:
    """Plot V vs J (PEMFC convention: V on y-axis, J on x-axis)."""
    J = _currents_mA_cm2(voltages, solutions, mesh, p)
    V = np.asarray(voltages)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(J, V, "b-o", ms=3, lw=1.5, label=label)
    ax.set_xlabel("Current density  (mA cm$^{-2}$)")
    ax.set_ylabel("$V_{\\mathrm{cathode}}$  (V vs SHE)")
    ax.set_title("Polarization curve — Stage 1 CL model")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    return ax


# ── 2. Spatial profiles ───────────────────────────────────────────────────────

def plot_profiles(
    voltages:  list[float],
    solutions: list[np.ndarray],
    mesh,
    p,
    V_sample:  list[float] | None = None,
    save_path: str | None = "profiles.png",
) -> None:
    """
    4-panel figure: c_O2, phi_L, phi_s, and eta = (phi_s − phi_L) − U_eq
    as functions of position x for a set of sampled voltages.
    """
    from assembly_stage1 import unpack

    if V_sample is None:
        n  = len(voltages)
        V_sample = [voltages[0],
                    voltages[n // 3],
                    voltages[2 * n // 3],
                    voltages[-1]]

    V_arr = np.asarray(voltages)
    xc_um = mesh.xc * 1e6   # m -> um

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(V_sample)))
    N = mesh.N

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax_c, ax_phiL, ax_phiS, ax_eta = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    for V_t, col in zip(V_sample, colors):
        idx = int(np.argmin(np.abs(V_arr - V_t)))
        u   = solutions[idx]
        lbl = f"V = {voltages[idx]:.3f} V"

        ln_cO2, phi_L, phi_s = unpack(u, N)
        c_O2  = np.exp(ln_cO2)
        U_eq  = p.U_ORR_eq(c_O2)
        eta   = (phi_s - phi_L) - U_eq

        ax_c.plot   (xc_um, c_O2,  color=col, label=lbl)
        ax_phiL.plot(xc_um, phi_L * 1e3, color=col, label=lbl)   # V -> mV
        ax_phiS.plot(xc_um, phi_s,  color=col, label=lbl)
        ax_eta.plot (xc_um, eta * 1e3, color=col, label=lbl)      # V -> mV

    ax_c.set_ylabel("$c_{O_2}$  (mol m$^{-3}$)")
    ax_phiL.set_ylabel("$\\phi_L$  (mV)")
    ax_phiS.set_ylabel("$\\phi_s$  (V vs SHE)")
    ax_eta.set_ylabel("$\\eta = (\\phi_s - \\phi_L) - U_{eq}$  (mV)")

    for ax in axes.flat:
        ax.set_xlabel("$x$  (um)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Spatial profiles — Stage 1 CL model", y=1.01)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()


# ── 3. IR breakdown ───────────────────────────────────────────────────────────

def plot_ir_breakdown(
    voltages:  list[float],
    solutions: list[np.ndarray],
    mesh,
    p,
    save_path: str | None = "ir_breakdown.png",
) -> None:
    """
    Stack plot showing V_cell decomposition into eta_kin, IR_solid, IR_ionic.

    V_cell = U_ORR_eq_avg − |eta_kin| − IR_solid − IR_ionic
    """
    N   = mesh.N
    V   = np.asarray(voltages)
    J   = _currents_mA_cm2(voltages, solutions, mesh, p)

    eta_kin_vals  = []
    ir_solid_vals = []
    ir_ionic_vals = []
    U_eq_avg_vals = []

    for u, V_cath in zip(solutions, voltages):
        ln_cO2, phi_L, phi_s = unpack(u, N)
        c_O2   = np.exp(ln_cO2)
        U_eq   = p.U_ORR_eq(c_O2)
        eta    = (phi_s - phi_L) - U_eq
        i_ORR  = R_ORR(ln_cO2, phi_s, phi_L, p)
        w      = i_ORR / (np.sum(i_ORR) + 1e-30)       # reaction-rate weights

        eta_kin_vals.append(float(np.dot(w, eta)))         # reaction-weighted avg eta
        U_eq_avg_vals.append(float(np.dot(w, U_eq)))

        # IR drops (estimate from uniform-reaction approximation)
        i_total = float(np.sum(i_ORR) * mesh.dx)
        ir_solid_vals.append(i_total / p.sigma_s_eff * p.L_CL)
        ir_ionic_vals.append(i_total / p.kappa_L_eff * p.L_CL)

    eta_kin  = np.abs(np.array(eta_kin_vals))
    ir_solid = np.array(ir_solid_vals)
    ir_ionic = np.array(ir_ionic_vals)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.stackplot(J, eta_kin * 1e3, ir_solid * 1e3, ir_ionic * 1e3,
                 labels=["Kinetic |eta|", "IR solid", "IR ionic"],
                 alpha=0.7)
    ax.set_xlabel("Current density  (mA cm$^{-2}$)")
    ax.set_ylabel("Overpotential  (mV)")
    ax.set_title("IR breakdown — Stage 1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    plt.show()


# ── 4. Consistency check plot ─────────────────────────────────────────────────

def plot_consistency_check(
    voltages:  list[float],
    solutions: list[np.ndarray],
    mesh,
    p,
    save_path: str | None = "consistency.png",
) -> None:
    """
    Plot i_total computed three independent ways and their relative errors.
    Should agree to < 0.1 % for a well-converged solution.
    """
    V    = np.asarray(voltages)
    integ  = []
    solid  = []
    ionic  = []

    for u, V_cath in zip(solutions, voltages):
        d = current_from_flux(u, mesh, p, V_cath)
        integ.append(d["integral"])
        solid.append(d["solid_flux"])
        ionic.append(d["ionic_flux"])

    integ = np.array(integ) * 1e-4 * 1e3   # mA/cm2
    solid = np.array(solid) * 1e-4 * 1e3
    ionic = np.array(ionic) * 1e-4 * 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(V, integ, "b-",  label="∫ i_ORR dx")
    ax1.plot(V, solid, "r--", label="i_s(x=0)")
    ax1.plot(V, ionic, "g:",  label="i_L(x=L_CL)")
    ax1.set_xlabel("$V_{\\mathrm{cathode}}$ (V vs SHE)")
    ax1.set_ylabel("Current density (mA cm$^{-2}$)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Three-way current consistency")

    err_solid = 100.0 * np.abs((solid - integ) / (np.abs(integ) + 1e-10))
    err_ionic = 100.0 * np.abs((ionic - integ) / (np.abs(integ) + 1e-10))
    ax2.semilogy(V, err_solid, "r--", label="|i_s − ∫| / |∫|")
    ax2.semilogy(V, err_ionic, "g:",  label="|i_L − ∫| / |∫|")
    ax2.axhline(0.1, color="k", lw=0.8, ls=":", label="0.1 % threshold")
    ax2.set_xlabel("$V_{\\mathrm{cathode}}$ (V vs SHE)")
    ax2.set_ylabel("Relative error (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Relative error between three methods")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    plt.show()
