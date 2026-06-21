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
from transport import diffusion_face_fluxes, ohmic_face_fluxes

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
    Two-panel voltage loss breakdown.

    Left : stacked-area decomposition of total loss vs J. Kinetic
           overpotential dominates; ohmic bands are labeled with %.
    Right: ohmic-only losses (IR_ionic, IR_solid) on an expanded mV
           scale, making the small contributions visible.

    IR drops are read from the actual phi profiles (exact):
        IR_ionic = phi_L[0]           (phi_L drops GDL face -> membrane ref=0)
        IR_solid = phi_s[0] - phi_s[-1]
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

        # IR drops from actual potential profiles (no approximation)
        ir_ionic_vals.append(float(phi_L[0]))
        ir_solid_vals.append(float(phi_s[0] - phi_s[-1]))

    eta_kin  = np.abs(np.array(eta_kin_vals))
    ir_solid = np.array(ir_solid_vals)
    ir_ionic = np.array(ir_ionic_vals)

    fig, axes, _ = gengrid(2, 1, size_inches=(6.5, 2.5), ticklabel_size=7)
    ax1, ax2 = axes[0], axes[1]

    # Left: full stacked breakdown
    ax1.stackplot(J,
                  eta_kin  * 1e3,
                  ir_ionic * 1e3,
                  ir_solid * 1e3,
                  labels=["Kinetic $|\\eta|$", "IR ionic", "IR solid"],
                  colors=[rainbow_2[1], rainbow_2[0], rainbow_2[2]],
                  alpha=0.85)
    ax1.set_xlabel("Current density  (mA cm$^{-2}$)", fontsize=_LABELSIZE)
    ax1.set_ylabel("Voltage loss  (mV)", fontsize=_LABELSIZE)
    ax1.set_title("Voltage loss breakdown — Stage 1", fontsize=9)
    ax1.legend(loc="upper left", fontsize=6, frameon=False)

    # Percentage annotation at max J
    idx_hc = int(np.argmax(np.abs(J)))
    tot    = (eta_kin[idx_hc] + ir_ionic[idx_hc] + ir_solid[idx_hc]) * 1e3
    ax1.text(0.97, 0.05,
             f"At $J_{{max}}$:  kinetic {100*eta_kin[idx_hc]*1e3/tot:.1f}%  "
             f"ionic {100*ir_ionic[idx_hc]*1e3/tot:.2f}%  "
             f"solid {100*ir_solid[idx_hc]*1e3/tot:.2f}%",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=6, family="monospace",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))

    # Right: ohmic only, expanded scale
    ax2.plot(J, ir_ionic * 1e3, color=rainbow_2[0], lw=1.5,
             label="IR ionic  ($\\kappa_L$)")
    ax2.plot(J, ir_solid * 1e3, color=rainbow_2[2], lw=1.5,
             label="IR solid  ($\\sigma_s$)")
    ax2.fill_between(J, ir_ionic * 1e3, alpha=0.25, color=rainbow_2[0])
    ax2.fill_between(J, ir_solid * 1e3, alpha=0.25, color=rainbow_2[2])
    ax2.set_xlabel("Current density  (mA cm$^{-2}$)", fontsize=_LABELSIZE)
    ax2.set_ylabel("Ohmic loss  (mV) — expanded", fontsize=_LABELSIZE)
    ax2.set_title("Ohmic contributions (zoomed)", fontsize=9)
    ax2.legend(fontsize=6, frameon=False)

    for val, col, lbl in [
        (ir_ionic[idx_hc] * 1e3, rainbow_2[0], "ionic"),
        (ir_solid[idx_hc] * 1e3, rainbow_2[2], "solid"),
    ]:
        ax2.annotate(f"{val:.3f} mV ({lbl})",
                     xy=(J[idx_hc], val),
                     xytext=(-8, 6), textcoords="offset points",
                     fontsize=6, color=col,
                     arrowprops=dict(arrowstyle="-", color=col, lw=0.8))

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


# ── 5. Flux profiles ──────────────────────────────────────────────────────────

def plot_flux_profiles(
    voltages:  list[float],
    solutions: list[np.ndarray],
    mesh,
    p,
    V_sample:  list[float] | None = None,
    save_path: str | None = "flux_profiles.png",
) -> None:
    """
    Two-panel figure of face-centred fluxes along the CL at sampled voltages.

<<<<<<< Updated upstream
    Left  : O2 diffusion flux J_O2(x) [μmol m-2 s-1] — enters at the GDL face
=======
    Left  : O2 diffusion flux J_O2(x) [mol m-2 s-1] — enters at the GDL face
>>>>>>> Stashed changes
            (x=0) and reaches zero at the no-flux membrane face (x=L_CL).
    Right : Solid i_s(x) and ionic i_L(x) current densities [mA cm-2].
            i_s falls from i_total at x=0 to zero at x=L_CL; i_L rises
            from zero to i_total.  Their sum equals i_total everywhere
            (charge conservation).
    """
    if V_sample is None:
        n = len(voltages)
        V_sample = [voltages[0],
                    voltages[n // 3],
                    voltages[2 * n // 3],
                    voltages[-1]]

    V_arr  = np.asarray(voltages)
    xf_um  = mesh.xf * 1e6          # face positions [μm]
    conv   = 1e-4 * 1e3             # A/m² → mA/cm²

    cidx   = np.linspace(2, len(warm_sequential) - 1, len(V_sample)).round().astype(int)
    colors = [warm_sequential[i] for i in cidx]

    fig, axes, _ = gengrid(2, 1, size_inches=(6.5, 3.5), ticklabel_size=8)
    ax_J, ax_i = axes[0], axes[1]

    for V_t, col in zip(V_sample, colors):
        idx = int(np.argmin(np.abs(V_arr - V_t)))
        u   = solutions[idx]
        V_c = float(voltages[idx])
        lbl = f"V = {V_c:.3f} V"

        N = mesh.N
        ln_cO2, phi_L, phi_s = unpack(u, N)

        J_O2 = diffusion_face_fluxes(
            ln_cO2, mesh.dx, p.D_O2_eff,
            bc_left=p.c_O2_bc, bc_right=None,
        )
        i_L = ohmic_face_fluxes(
            phi_L, mesh.dx, p.kappa_L_eff,
            bc_left=None, bc_right=p.phi_L_mem,
        )
        i_s = ohmic_face_fluxes(
            phi_s, mesh.dx, p.sigma_s_eff,
            bc_left=V_c, bc_right=None,
        )

        ax_J.plot(xf_um, J_O2 * 1e6, color=col, label=lbl)     # → μmol m-2 s-1
        ax_i.plot(xf_um, i_s  * conv, color=col, ls="-")
        ax_i.plot(xf_um, i_L  * conv, color=col, ls="--")

    ax_J.set_xlabel("$x$  (μm)", fontsize=_LABELSIZE)
    ax_J.set_ylabel("$J_{O_2}$  (μmol m$^{-2}$ s$^{-1}$)", fontsize=_LABELSIZE)
    ax_J.set_title("O$_2$ diffusion flux", fontsize=9)
    ax_J.legend(fontsize=6, frameon=False)

    ax_i.set_xlabel("$x$  (μm)", fontsize=_LABELSIZE)
    ax_i.set_ylabel("Current density  (mA cm$^{-2}$)", fontsize=_LABELSIZE)
    ax_i.set_title("Solid (—) and ionic (– –) current sharing", fontsize=9)
    from matplotlib.lines import Line2D
    ax_i.legend(
        handles=[
            Line2D([0], [0], color="gray", ls="-",  lw=1.2, label="solid  $i_s$"),
            Line2D([0], [0], color="gray", ls="--", lw=1.2, label="ionic  $i_L$"),
        ],
        fontsize=6, frameon=False,
    )

    fig.tight_layout()
    fig.subplots_adjust(left=0.14)   # prevent y-axis label clipping
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)


# Micron label that renders in the Lato/mathtext font (raw U+03BC has no glyph)
_UM = "$x$  ($\\mu$m)"


# ── 6. Stage 4: polarization overlay (Stage 1 vs 3 vs 4) ──────────────────────

def plot_stage4_polarization(
    vs1, sols1, mesh_cl,
    vs3, sols3,
    vs4, sols4, mesh_gdl,
    p,
    save_path: str | None = "stage4_polarization.png",
) -> None:
    """
    Polarization curves for Stage 1, Stage 3 and Stage 4 on one axis, with the
    Stage 3 -> Stage 4 limiting-current gap (the ionomer-film interphase loss)
    annotated.
    """
    from assembly_stage1 import compute_current as cc1
    from assembly_stage3 import compute_current_s3
    from assembly_stage4 import compute_current_s4

    J1 = np.array([cc1(u, mesh_cl, p) * 1e-1 for u in sols1])            # mA/cm2
    J3 = np.array([compute_current_s3(u, mesh_gdl, mesh_cl, p) * 1e-1 for u in sols3])
    J4 = np.array([compute_current_s4(u, mesh_gdl, mesh_cl, p) * 1e-1 for u in sols4])

    fig, ax, _ = gengrid(1, 1, size_inches=(3.5, 2.9), ticklabel_size=8,
                         genlabels=False)
    ax.plot(J1, vs1, marker="o", ms=3, lw=1.5, color=rainbow_2[0],
            label="Stage 1  (ionomer only)")
    ax.plot(J3, vs3, marker="s", ms=3, lw=1.5, ls="--", color=rainbow_2[4],
            label="Stage 3  (gas, local equil.)")
    ax.plot(J4, vs4, marker="^", ms=3.5, lw=1.6, ls="-", color=rainbow_2[2],
            label="Stage 4  (finite-rate + M-S)")

    # Annotate the Stage 3 -> Stage 4 limiting-current gap at the lowest voltage.
    V_lo = float(min(np.min(vs3), np.min(vs4)))
    j3_lo = float(J3[int(np.argmin(vs3))])
    j4_lo = float(J4[int(np.argmin(vs4))])
    ax.annotate("", xy=(j3_lo, V_lo), xytext=(j4_lo, V_lo),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))
    ax.text(0.5 * (j3_lo + j4_lo), V_lo + 0.012,
            f"film loss\n{(j3_lo - j4_lo) / j3_lo * 100:.0f}%",
            ha="center", va="bottom", fontsize=6, color="0.25")

    ax.set_xlabel("Current density  (mA cm$^{-2}$)", fontsize=_LABELSIZE)
    ax.set_ylabel("$V_{\\mathrm{cathode}}$  (V vs SHE)", fontsize=_LABELSIZE)
    ax.set_title("Polarization: Stage 1 vs 3 vs 4", fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ── 7. Stage 4: gas / dissolved / equilibrium O2 profiles ─────────────────────

def plot_stage4_o2_profiles(
    vs4, sols4, mesh_gdl, mesh_cl, p,
    V_sample: list[float] | None = None,
    save_path: str | None = "stage4_o2_profiles.png",
) -> None:
    """
    Two-panel Stage 4 O2 profiles at sampled voltages.

    a) Full-domain pore-gas O2 from the gas channel (x=0) through the GDL and CL
       (dashed line marks the GDL/CL interface).  The gas depletes only mildly,
       showing gas transport is not the bottleneck here.
    b) CL dissolved O2 c_ion(x) vs the local equilibrium K_eq*c_gas(x).  The
       shaded gap at the highest current is the finite-rate interphase
       (ionomer-film) transport resistance — the dominant loss.
    """
    from assembly_stage4 import unpack_s4

    NG, NC = mesh_gdl.N, mesh_cl.N
    if V_sample is None:
        n = len(vs4)
        V_sample = [vs4[0], vs4[n // 3], vs4[2 * n // 3], vs4[-1]]

    V_arr   = np.asarray(vs4)
    x_gdl   = mesh_gdl.xc * 1e6                       # 0 .. L_GDL  [um]
    x_cl    = (p.L_GDL + mesh_cl.xc) * 1e6            # L_GDL .. L_GDL+L_CL [um]
    xc_cl   = mesh_cl.xc * 1e6                        # CL-local coordinate [um]
    x_if    = p.L_GDL * 1e6                           # GDL/CL interface [um]
    cidx    = np.linspace(2, len(warm_sequential) - 1, len(V_sample)).round().astype(int)
    colors  = [warm_sequential[i] for i in cidx]

    fig, axes, _ = gengrid(2, 1, size_inches=(6.5, 3.0), ticklabel_size=8)
    ax_gas, ax_ion = axes[0], axes[1]

    lowest = None
    for V_t, col in zip(V_sample, colors):
        idx = int(np.argmin(np.abs(V_arr - V_t)))
        u   = sols4[idx]
        lbl = f"$V$ = {vs4[idx]:.3f} V"
        ln_c_gdl, ln_c_gas, ln_c_ion, _, _ = unpack_s4(u, NG, NC)
        c_gdl = np.exp(ln_c_gdl)
        c_gas = np.exp(ln_c_gas)
        c_ion = np.exp(ln_c_ion)
        c_eq  = p.K_eq_gas_ion * c_gas

        # Panel a: continuous gas profile across GDL + CL
        ax_gas.plot(np.concatenate([x_gdl, x_cl]),
                    np.concatenate([c_gdl, c_gas]),
                    color=col, lw=1.5, label=lbl)

        # Panel b: dissolved (solid) vs equilibrium (dotted)
        ax_ion.plot(xc_cl, c_ion, color=col, lw=1.6, ls="-")
        ax_ion.plot(xc_cl, c_eq,  color=col, lw=1.0, ls=":")
        lowest = (xc_cl, c_ion, c_eq, col)

    # Shade the interphase gap at the highest-current (lowest-V) case
    if lowest is not None:
        xc_cl, c_ion, c_eq, col = lowest
        ax_ion.fill_between(xc_cl, c_ion, c_eq, color=col, alpha=0.18,
                            lw=0, label="interphase gap")

    ax_gas.axvline(x_if, color="0.5", lw=0.9, ls="--")
    ax_gas.axhline(p.c_O2_gas_inlet, color="0.7", lw=0.8, ls=":")
    ax_gas.text(x_if - 4, ax_gas.get_ylim()[0], " GDL", ha="right", va="bottom",
                fontsize=6, color="0.4")
    ax_gas.text(x_if + 4, ax_gas.get_ylim()[0], "CL ", ha="left", va="bottom",
                fontsize=6, color="0.4")
    ax_gas.set_xlabel(_UM.replace("$x$", "$x$ from gas channel"), fontsize=_LABELSIZE)
    ax_gas.set_ylabel("$c_{O_2}$ gas  (mol m$^{-3}$)", fontsize=_LABELSIZE)
    ax_gas.set_title("Pore-gas O$_2$  (GDL + CL)", fontsize=9)
    ax_gas.legend(fontsize=5.5, frameon=False, loc="lower left")

    ax_ion.set_xlabel(_UM.replace("$x$", "$x$ in CL"), fontsize=_LABELSIZE)
    ax_ion.set_ylabel("$c_{O_2}$ ionomer  (mol m$^{-3}$)", fontsize=_LABELSIZE)
    ax_ion.set_title("Dissolved vs equilibrium $K_{eq}c_{gas}$", fontsize=9)
    from matplotlib.lines import Line2D
    ax_ion.legend(
        handles=[
            Line2D([0], [0], color="gray", ls="-", lw=1.6, label="dissolved $c_{ion}$"),
            Line2D([0], [0], color="gray", ls=":", lw=1.2, label="equil. $K_{eq}c_{gas}$"),
        ],
        fontsize=5.5, frameon=False, loc="center right",
    )

    fig.tight_layout()
    fig.subplots_adjust(left=0.10, wspace=0.32)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ── 8. Stage 4: limiting current vs k_v ───────────────────────────────────────

def plot_kv_sweep(
    kv_list, Jlim, J3_lim, kv_default,
    save_path: str | None = "stage4_kv_sweep.png",
) -> None:
    """
    Limiting current vs interphase coefficient k_v, with the Stage 3
    (k_v -> inf, local-equilibrium) value drawn as the horizontal asymptote.
    """
    kv = np.asarray(kv_list, float)
    J  = np.asarray(Jlim, float)
    ok = np.isfinite(J)

    fig, ax, _ = gengrid(1, 1, size_inches=(3.5, 2.9), ticklabel_size=8,
                         genlabels=False)

    # Shade the transport-limited (low-k_v) vs equilibrium (high-k_v) regimes
    ax.axhline(J3_lim, color=rainbow_2[4], ls="--", lw=1.2,
               label="Stage 3  ($k_v\\!\\to\\!\\infty$)")
    ax.semilogx(kv[ok], J[ok], marker="o", ms=4, lw=1.6, color=rainbow_2[2],
                label="Stage 4  $J_{\\lim}(k_v)$")
    ax.axvline(kv_default, color="0.5", ls=":", lw=1.0)

    # Mark the default operating point
    i_def = int(np.argmin(np.abs(kv - kv_default)))
    if np.isfinite(J[i_def]):
        ax.plot(kv[i_def], J[i_def], marker="*", ms=11, color="0.15", zorder=5)
        ax.annotate(f"default $k_v$\n{kv_default:.0e} s$^{{-1}}$",
                    xy=(kv[i_def], J[i_def]),
                    xytext=(8, -22), textcoords="offset points",
                    fontsize=6, color="0.2", ha="left")

    ax.set_xlabel("$k_v = k_{MT}\\,a_{GL}$  (s$^{-1}$)", fontsize=_LABELSIZE)
    ax.set_ylabel("Limiting current  (mA cm$^{-2}$)", fontsize=_LABELSIZE)
    ax.set_title("Limiting current vs interphase transfer rate", fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="center right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
