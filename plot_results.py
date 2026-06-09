"""
Plotting routines for Stage 1 ORR PEMFC CL model results.

Follows the PEMFC convention (§15.5):
    y-axis : V_cathode [V vs SHE], increasing upward
    x-axis : current density [A/cm2], increasing rightward
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from assembly_stage1 import unpack, compute_current, current_from_flux
from kinetics import R_ORR

# Project-standard plotting: gengrid styling + UC Berkeley color palettes.
# (customplot sets the SVG backend, Lato font, and consistent tick styling.)
from customplot import gengrid, rainbow_2, warm_sequential

# Axis-label font size (gengrid controls tick-label size separately).
_LABELSIZE = 8


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
        fig, ax, _ = gengrid(1, 1, size_inches=(3.25, 2.5))
    else:
        fig = ax.figure

    ax.plot(J, V, marker="o", ms=3, lw=1.5, color=rainbow_2[1], label=label)
    ax.set_xlabel("Current density  (mA cm$^{-2}$)", fontsize=_LABELSIZE)
    ax.set_ylabel("$V_{\\mathrm{cathode}}$  (V vs SHE)", fontsize=_LABELSIZE)
    ax.set_title("Polarization curve — Stage 1 CL model", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
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

    # Sequential warm ramp: pale (low overpotential) -> dark (high overpotential)
    cidx   = np.linspace(2, len(warm_sequential) - 1, len(V_sample)).round().astype(int)
    colors = [warm_sequential[i] for i in cidx]
    N = mesh.N

    fig, axes, _ = gengrid(2, 2, size_inches=(6.5, 6.25), ticklabel_size=7)
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

    ax_c.set_ylabel("$c_{O_2}$  (mol m$^{-3}$)", fontsize=_LABELSIZE)
    ax_phiL.set_ylabel("$\\phi_L$  (mV)", fontsize=_LABELSIZE)
    ax_phiS.set_ylabel("$\\phi_s$  (V vs SHE)", fontsize=_LABELSIZE)
    ax_eta.set_ylabel("$\\eta = (\\phi_s - \\phi_L) - U_{eq}$  (mV)", fontsize=_LABELSIZE)

    for ax in axes.flat:
        ax.set_xlabel("$x$  (um)", fontsize=_LABELSIZE)
        ax.legend(fontsize=6, frameon=False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)


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

    fig, ax, _ = gengrid(1, 1, size_inches=(3.25, 2.5))
    ax.stackplot(J, eta_kin * 1e3, ir_solid * 1e3, ir_ionic * 1e3,
                 labels=["Kinetic |eta|", "IR solid", "IR ionic"],
                 colors=[rainbow_2[1], rainbow_2[0], rainbow_2[5]],
                 alpha=0.85)
    ax.set_xlabel("Current density  (mA cm$^{-2}$)", fontsize=_LABELSIZE)
    ax.set_ylabel("Overpotential  (mV)", fontsize=_LABELSIZE)
    ax.set_title("IR breakdown — Stage 1", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=False)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)


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

    fig, axes, _ = gengrid(1, 2, size_inches=(6.5, 3.0), ticklabel_size=7)
    ax1, ax2 = axes[0], axes[1]

    ax1.plot(V, integ, color=rainbow_2[1], ls="-",  label="∫ i_ORR dx")
    ax1.plot(V, solid, color=rainbow_2[4], ls="--", label="i_s(x=0)")
    ax1.plot(V, ionic, color=rainbow_2[0], ls=":",  label="i_L(x=L_CL)")
    ax1.set_xlabel("$V_{\\mathrm{cathode}}$ (V vs SHE)", fontsize=_LABELSIZE)
    ax1.set_ylabel("Current density (mA cm$^{-2}$)", fontsize=_LABELSIZE)
    ax1.legend(fontsize=7, frameon=False)
    ax1.set_title("Three-way current consistency", fontsize=9)

    err_solid = 100.0 * np.abs((solid - integ) / (np.abs(integ) + 1e-10))
    err_ionic = 100.0 * np.abs((ionic - integ) / (np.abs(integ) + 1e-10))
    ax2.semilogy(V, err_solid, color=rainbow_2[4], ls="--", label="|i_s − ∫| / |∫|")
    ax2.semilogy(V, err_ionic, color=rainbow_2[0], ls=":",  label="|i_L − ∫| / |∫|")
    ax2.axhline(0.1, color="k", lw=0.8, ls=":", label="0.1 % threshold")
    ax2.set_xlabel("$V_{\\mathrm{cathode}}$ (V vs SHE)", fontsize=_LABELSIZE)
    ax2.set_ylabel("Relative error (%)", fontsize=_LABELSIZE)
    ax2.legend(fontsize=7, frameon=False)
    ax2.set_title("Relative error between three methods", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)
