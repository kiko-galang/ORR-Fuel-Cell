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
    Two-panel voltage loss breakdown.

    Left : stacked-area decomposition of total loss (OCV − V_cell) vs J.
           Kinetic overpotential dominates; ohmic bands are labeled.
    Right: ohmic-only losses (IR_ionic, IR_solid) on an expanded mV scale,
           making the small contributions visible and labeled.

    IR drops are computed from the actual φ profiles, not a uniform
    approximation:
        IR_ionic = φ_L[0]          (φ_L drops from GDL face to membrane ref=0)
        IR_solid = φ_s[0] − φ_s[-1]  (φ_s drop across the CL)
    """
    N = mesh.N
    J = _currents_mA_cm2(voltages, solutions, mesh, p)

    eta_kin_vals  = []
    ir_solid_vals = []
    ir_ionic_vals = []
    U_eq_avg_vals = []

    for u, V_cath in zip(solutions, voltages):
        ln_cO2, phi_L, phi_s = unpack(u, N)
        c_O2  = np.exp(ln_cO2)
        U_eq  = p.U_ORR_eq(c_O2)
        eta   = (phi_s - phi_L) - U_eq
        i_ORR = R_ORR(ln_cO2, phi_s, phi_L, p)
        w     = i_ORR / (np.sum(i_ORR) + 1e-30)

        eta_kin_vals.append(float(np.dot(w, eta)))
        U_eq_avg_vals.append(float(np.dot(w, U_eq)))

        # IR drops from actual potential profiles (exact, no approximation)
        ir_ionic_vals.append(float(phi_L[0]))                   # V  (φ_L[0] - 0)
        ir_solid_vals.append(float(phi_s[0] - phi_s[-1]))       # V

    eta_kin  = np.abs(np.array(eta_kin_vals))
    ir_solid = np.array(ir_solid_vals)
    ir_ionic = np.array(ir_ionic_vals)
    U_eq_avg = np.array(U_eq_avg_vals)
    V_arr    = np.asarray(voltages)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left panel: full stacked breakdown ────────────────────────────────────
    colors = ["#4878CF", "#D65F5F", "#6ACC65"]   # blue, red, green
    ax1.stackplot(J,
                  eta_kin  * 1e3,
                  ir_ionic * 1e3,
                  ir_solid * 1e3,
                  labels=["Kinetic $|\\eta|$", "IR ionic", "IR solid"],
                  colors=colors, alpha=0.80)
    ax1.set_xlabel("Current density  (mA cm$^{-2}$)")
    ax1.set_ylabel("Voltage loss  (mV)")
    ax1.set_title("Voltage loss breakdown — Stage 1")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Annotate max-J percentage split
    idx_hc  = int(np.argmax(np.abs(J)))
    tot_mv  = (eta_kin[idx_hc] + ir_ionic[idx_hc] + ir_solid[idx_hc]) * 1e3
    pct_kin = 100.0 * eta_kin[idx_hc]  * 1e3 / tot_mv
    pct_ion = 100.0 * ir_ionic[idx_hc] * 1e3 / tot_mv
    pct_sol = 100.0 * ir_solid[idx_hc] * 1e3 / tot_mv
    ax1.text(0.97, 0.05,
             f"At J$_{{max}}$:\n"
             f"  Kinetic : {pct_kin:.1f}%\n"
             f"  IR ionic: {pct_ion:.2f}%\n"
             f"  IR solid: {pct_sol:.2f}%",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Right panel: ohmic losses only, expanded scale ────────────────────────
    ax2.plot(J, ir_ionic * 1e3, color=colors[1], lw=2, label="IR ionic  ($\\kappa_L$)")
    ax2.plot(J, ir_solid * 1e3, color=colors[2], lw=2, label="IR solid  ($\\sigma_s$)")
    ax2.fill_between(J, ir_ionic * 1e3, alpha=0.25, color=colors[1])
    ax2.fill_between(J, ir_solid * 1e3, alpha=0.25, color=colors[2])

    ax2.set_xlabel("Current density  (mA cm$^{-2}$)")
    ax2.set_ylabel("Ohmic loss  (mV)  — expanded")
    ax2.set_title("Ohmic contributions (zoomed)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Label the ionic/solid values at max J
    for val, col, lbl in [
        (ir_ionic[idx_hc] * 1e3, colors[1], "ionic"),
        (ir_solid[idx_hc] * 1e3, colors[2], "solid"),
    ]:
        ax2.annotate(f"{val:.3f} mV ({lbl})",
                     xy=(J[idx_hc], val),
                     xytext=(-10, 8), textcoords="offset points",
                     fontsize=8, color=col,
                     arrowprops=dict(arrowstyle="-", color=col, lw=0.8))

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
