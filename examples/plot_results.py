"""
Plotting routines for Stage 1 ORR PEMFC CL model results.

Follows the PEMFC convention (§15.5):
    y-axis : V_cathode [V vs SHE], increasing upward
    x-axis : current density [A/cm2], increasing rightward

Styled to the book Figure_Preparation_Guide.pdf: fixed mm figure canvases,
Arial typography, and the 4-color discrete palette (black/blue/green/red)
combined with marker/linestyle when a panel needs more than 4 series.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from assembly_stage1 import unpack, compute_current, current_from_flux
from kinetics import R_ORR
from transport import diffusion_face_fluxes, ohmic_face_fluxes

from book_style import (
    FIGSIZE_SMALL, FIGSIZE_LARGE, FIGSIZE_LARGE_SQUARE,
    BOOK_COLORS, COLOR_CYCLE, LINESTYLES,
    add_panel_labels, savefig_book,
)

BLACK, BLUE, GREEN, RED = (BOOK_COLORS[k] for k in ("black", "blue", "green", "red"))

# Shared axis-label text ("Quantity / Unit", per Section 4).
_CURRENT_LBL  = "Current density / mA cm$^{-2}$"
_POSITION_UM  = "$x$ / $\\mu$m"


def _currents_mA_cm2(voltages, solutions, mesh, p) -> np.ndarray:
    """Compute current density [mA/cm2] for each voltage point."""
    return np.array([
        compute_current(u, mesh, p) * 1e-4 * 1e3   # A/m2 -> mA/cm2
        for u in solutions
    ])


def _voltage_samples(voltages, V_sample=None, n=4):
    """Pick n representative voltages and return (indices, colors, labels).

    n must be <= 4: each sampled voltage is a discrete, individually-labeled
    condition (not a dense continuous sweep), so it maps onto the four
    approved colors (Section 5.1) rather than a viridis continuum. The
    mapping is fixed chapter-wide: index 0 (near-OCV) -> black, ... ,
    index -1 (highest overpotential) -> red.
    """
    assert n <= 4, "only 4 approved discrete colors are available"
    if V_sample is None:
        m = len(voltages)
        idx_frac = np.linspace(0, m - 1, n).round().astype(int)
        V_sample = [voltages[i] for i in idx_frac]
    V_arr = np.asarray(voltages)
    idx = [int(np.argmin(np.abs(V_arr - V_t))) for V_t in V_sample]
    colors = COLOR_CYCLE[:n]
    labels = [f"$V$ = {voltages[i]:.3f} V" for i in idx]
    return idx, colors, labels


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
        fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    else:
        fig = ax.figure

    ax.plot(J, V, marker="o", color=BLACK, label=label)
    ax.set_xlabel(_CURRENT_LBL)
    ax.set_ylabel("$V_{\\mathrm{cathode}}$ / V vs. SHE")
    ax.set_title("Stage 1 polarization curve")
    ax.legend()

    if save_path:
        fig.tight_layout()
        savefig_book(fig, save_path)

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
    idx_samples, colors, labels = _voltage_samples(voltages, V_sample, n=4)
    xc_um = mesh.xc * 1e6   # m -> um
    N = mesh.N

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE_SQUARE)
    ax_c, ax_phiL, ax_phiS, ax_eta = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    for idx, col, lbl in zip(idx_samples, colors, labels):
        u = solutions[idx]
        ln_cO2, phi_L, phi_s = unpack(u, N)
        c_O2  = np.exp(ln_cO2)
        U_eq  = p.U_ORR_eq(c_O2)
        eta   = (phi_s - phi_L) - U_eq

        ax_c.plot   (xc_um, c_O2,  color=col, label=lbl)
        ax_phiL.plot(xc_um, phi_L * 1e3, color=col, label=lbl)   # V -> mV
        ax_phiS.plot(xc_um, phi_s,  color=col, label=lbl)
        ax_eta.plot (xc_um, eta * 1e3, color=col, label=lbl)      # V -> mV

    ax_c.set_ylabel("$c_{O_2}$ / mol m$^{-3}$")
    ax_phiL.set_ylabel("$\\phi_L$ / mV")
    ax_phiS.set_ylabel("$\\phi_s$ / V vs. SHE")
    ax_eta.set_ylabel("$\\eta = (\\phi_s - \\phi_L) - U_{eq}$ / mV")

    for ax in axes.flat:
        ax.set_xlabel(_POSITION_UM)
        ax.legend()

    # c_O2 curves decay from the left, leaving an empty band on the centre-right;
    # use 2 columns so the legend is short (2 rows) and stays clear of the curves
    ax_c.legend(loc="center right", ncol=2,
                columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(0.99, 0.72))

    add_panel_labels(axes)
    fig.tight_layout()

    if save_path:
        savefig_book(fig, save_path)

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
    Stage 1 applied-voltage breakdown by the power-loss post-processing method
    of Gerhardt et al., J. Electrochem. Soc. 168, 074503 (2021) — the SAME
    method as the Stage 4 breakdown, so the two are consistent.

      ohmic, ionic : integral(i_L^2 / kappa_L_eff) dx / I_cell      (Joule, Eq. 18)
      ohmic, solid : integral(i_s^2 / sigma_s_eff) dx / I_cell      (Joule, Eq. 18)
      mass-transp. : integral i_ORR (RT/alpha_c F) ln(c_bc/c_O2) dx / I_cell (Eq. 23)
      kinetic      : (U_OCV - V) - (ohmic + mass-transport)          (activation)

    The four contributions sum EXACTLY to U_OCV - V.  Colors follow a
    chapter-wide convention: kinetic=black, mass-transport=red (the loss
    this stage's model exists to expose), ohmic-ionic=blue, ohmic-solid=green.
    """
    from transport import ohmic_face_fluxes

    N    = mesh.N
    dx   = mesh.dx
    RTaF = p.R * p.T / (p.alpha * p.F)            # Eq. 23 prefactor RT/(alpha_c F)
    U_ocv = float(p.U_ORR_eq(p.c_O2_bc))
    J = _currents_mA_cm2(voltages, solutions, mesh, p)

    kin, mt, ohm_i, ohm_s, tot = ([] for _ in range(5))
    for u, V in zip(solutions, voltages):
        ln_cO2, phi_L, phi_s = unpack(u, N)
        c_O2  = np.exp(ln_cO2)
        i_ORR = R_ORR(ln_cO2, phi_s, phi_L, p)
        Icell = float(np.sum(i_ORR) * dx)
        w     = i_ORR / (np.sum(i_ORR) + 1e-30)

        iL = ohmic_face_fluxes(phi_L, dx, p.kappa_L_eff, bc_left=None, bc_right=p.phi_L_mem)
        iS = ohmic_face_fluxes(phi_s, dx, p.sigma_s_eff, bc_left=V, bc_right=None)
        iLc = 0.5 * (iL[:-1] + iL[1:])
        iSc = 0.5 * (iS[:-1] + iS[1:])
        oi = float(np.sum(iLc ** 2 / p.kappa_L_eff) * dx / Icell)
        os = float(np.sum(iSc ** 2 / p.sigma_s_eff) * dx / Icell)
        m_ = float(RTaF * p.gamma * np.dot(w, np.log(np.maximum(p.c_O2_bc / c_O2, 1e-30))))
        t_ = U_ocv - float(V)
        ohm_i.append(oi); ohm_s.append(os); mt.append(m_); tot.append(t_)
        kin.append(t_ - oi - os - m_)            # activation = remainder

    stacks = [np.clip(np.asarray(v), 0, None) * 1e3
              for v in (kin, mt, ohm_i, ohm_s)]
    labels = ["Kinetic (activation)", "Mass transp. (ionomer O$_2$)",
              "Ohmic (ionic)", "Ohmic (solid)"]
    colors = [BLACK, RED, BLUE, GREEN]
    tot = np.asarray(tot) * 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # ── Left: full stacked breakdown (sums exactly to U_OCV - V) ─────────────
    ax1.stackplot(J, *stacks, labels=labels, colors=colors, alpha=0.9)
    ax1.plot(J, tot, color=BLACK, lw=1.0, ls="--", label="$U_{OCV}-V$")
    ax1.set_xlabel(_CURRENT_LBL)
    ax1.set_ylabel("Voltage loss / mV")
    ax1.set_title("Stage 1 applied-voltage breakdown")
    ax1.set_ylim(0, float(tot.max()) * 1.5)       # headroom for the legend
    ax1.legend(loc="upper left", ncol=2)

    idx = int(np.argmax(J))
    parts = [s[idx] for s in stacks]
    tt = sum(parts) + 1e-30
    ax1.text(0.97, 0.05,
             f"At $J_{{max}}$: kin {100*parts[0]/tt:.0f}%  "
             f"mt {100*parts[1]/tt:.0f}%  ohm {100*(parts[2]+parts[3])/tt:.2f}%",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=6,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

    # ── Right: each non-kinetic loss INDIVIDUALLY (not stacked), expanded ─────
    for s, c, lbl in [
        (stacks[1], RED,  "Mass transp. (ionomer O$_2$)"),
        (stacks[2], BLUE, "Ohmic (ionic)"),
        (stacks[3], GREEN, "Ohmic (solid)"),
    ]:
        ax2.plot(J, s, color=c, label=lbl)
    ax2.set_xlabel(_CURRENT_LBL)
    ax2.set_ylabel("Loss / mV")
    ax2.set_title("Non-kinetic losses")
    ax2.legend(loc="upper left")

    add_panel_labels((ax1, ax2))
    fig.tight_layout()
    if save_path:
        savefig_book(fig, save_path)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    ax1.plot(V, integ, color=BLACK, ls="-",  label="$\\int i_{ORR}\\,dx$")
    ax1.plot(V, solid, color=BLUE,  ls="--", label="$i_s(x=0)$")
    ax1.plot(V, ionic, color=GREEN, ls=":",  label="$i_L(x=L_{CL})$")
    ax1.set_xlabel("$V_{\\mathrm{cathode}}$ / V vs. SHE")
    ax1.set_ylabel(_CURRENT_LBL)
    ax1.legend()
    ax1.set_title("Three-way current consistency")

    err_solid = 100.0 * np.abs((solid - integ) / (np.abs(integ) + 1e-10))
    err_ionic = 100.0 * np.abs((ionic - integ) / (np.abs(integ) + 1e-10))
    ax2.semilogy(V, err_solid, color=BLUE,  ls="--", label="$|i_s - \\int| / |\\int|$")
    ax2.semilogy(V, err_ionic, color=GREEN, ls=":",  label="$|i_L - \\int| / |\\int|$")
    ax2.axhline(0.1, color=RED, lw=0.7, ls=":", label="0.1% threshold")
    ax2.set_xlabel("$V_{\\mathrm{cathode}}$ / V vs. SHE")
    ax2.set_ylabel("Relative error / %")
    ax2.legend()
    ax2.set_title("Relative error between methods")

    add_panel_labels((ax1, ax2))
    fig.tight_layout()
    if save_path:
        savefig_book(fig, save_path)

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

    Left  : O2 diffusion flux J_O2(x) [umol m-2 s-1] — enters at the GDL face
            (x=0) and reaches zero at the no-flux membrane face (x=L_CL).
    Right : Solid i_s(x) and ionic i_L(x) current densities [mA cm-2].
            i_s falls from i_total at x=0 to zero at x=L_CL; i_L rises
            from zero to i_total.  Their sum equals i_total everywhere
            (charge conservation).
    """
    idx_samples, colors, _ = _voltage_samples(voltages, V_sample, n=4)
    xf_um = mesh.xf * 1e6          # face positions [um]
    conv  = 1e-4 * 1e3             # A/m2 -> mA/cm2

    fig, (ax_J, ax_i) = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    for idx, col in zip(idx_samples, colors):
        u   = solutions[idx]
        V_c = float(voltages[idx])
        lbl = f"$V$ = {V_c:.3f} V"

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

        ax_J.plot(xf_um, J_O2 * 1e6, color=col, label=lbl)     # -> umol m-2 s-1
        ax_i.plot(xf_um, i_s  * conv, color=col, ls="-")
        ax_i.plot(xf_um, i_L  * conv, color=col, ls="--")

    ax_J.set_xlabel(_POSITION_UM)
    ax_J.set_ylabel("$J_{O_2}$ / $\\mu$mol m$^{-2}$ s$^{-1}$")
    ax_J.set_title("O$_2$ diffusion flux")
    ax_J.legend()

    ax_i.set_xlabel(_POSITION_UM)
    ax_i.set_ylabel(_CURRENT_LBL)
    ax_i.set_title("Solid (—) and ionic (– –) current sharing")
    ax_i.legend(
        handles=[
            Line2D([0], [0], color="0.35", ls="-",  label="solid  $i_s$"),
            Line2D([0], [0], color="0.35", ls="--", label="ionic  $i_L$"),
        ],
    )

    add_panel_labels((ax_J, ax_i))
    fig.tight_layout()
    if save_path:
        savefig_book(fig, save_path)

    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    ax.plot(J1, vs1, marker="o", color=BLACK, label="Stage 1 (ionomer only)")
    ax.plot(J3, vs3, marker="s", ls="--", color=BLUE, label="Stage 3 (gas, local equil.)")
    ax.plot(J4, vs4, marker="^", ls="-", color=GREEN, label="Stage 4 (finite-rate + M-S)")

    # Annotate the Stage 3 -> Stage 4 limiting-current gap at the lowest voltage.
    V_lo = float(min(np.min(vs3), np.min(vs4)))
    j3_lo = float(J3[int(np.argmin(vs3))])
    j4_lo = float(J4[int(np.argmin(vs4))])
    ax.annotate("", xy=(j3_lo, V_lo), xytext=(j4_lo, V_lo),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=0.9))
    ax.text(0.5 * (j3_lo + j4_lo), V_lo + 0.012,
            f"film loss\n{(j3_lo - j4_lo) / j3_lo * 100:.0f}%",
            ha="center", va="bottom", fontsize=6, color=RED)

    ax.set_xlabel(_CURRENT_LBL)
    ax.set_ylabel("$V_{\\mathrm{cathode}}$ / V vs. SHE")
    ax.set_title("Polarization: Stage 1 vs 3 vs 4")
    ax.legend(loc="upper right")

    fig.tight_layout()
    if save_path:
        savefig_book(fig, save_path)
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
    idx_samples, colors, _ = _voltage_samples(vs4, V_sample, n=4)

    x_gdl = mesh_gdl.xc * 1e6                       # 0 .. L_GDL  [um]
    x_cl  = (p.L_GDL + mesh_cl.xc) * 1e6            # L_GDL .. L_GDL+L_CL [um]
    xc_cl = mesh_cl.xc * 1e6                        # CL-local coordinate [um]
    x_if  = p.L_GDL * 1e6                           # GDL/CL interface [um]

    fig, (ax_gas, ax_ion) = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    lowest = None
    for idx, col in zip(idx_samples, colors):
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
                    color=col, label=lbl)

        # Panel b: dissolved (solid) vs equilibrium (dotted)
        ax_ion.plot(xc_cl, c_ion, color=col, ls="-")
        ax_ion.plot(xc_cl, c_eq,  color=col, ls=":")
        lowest = (xc_cl, c_ion, c_eq, col)

    # Shade the interphase gap at the highest-current (lowest-V) case
    if lowest is not None:
        xc_cl, c_ion, c_eq, col = lowest
        ax_ion.fill_between(xc_cl, c_ion, c_eq, color=col, alpha=0.18,
                            lw=0, label="interphase gap")

    ax_gas.axvline(x_if, color="0.5", lw=0.7, ls="--")
    ax_gas.axhline(p.c_O2_gas_inlet, color="0.7", lw=0.6, ls=":")
    ax_gas.text(x_if - 4, ax_gas.get_ylim()[0], " GDL", ha="right", va="bottom",
                fontsize=6, color="0.4")
    ax_gas.text(x_if + 4, ax_gas.get_ylim()[0], "CL ", ha="left", va="bottom",
                fontsize=6, color="0.4")
    ax_gas.set_xlabel("$x$ from gas channel / $\\mu$m")
    ax_gas.set_ylabel("$c_{O_2}$ gas / mol m$^{-3}$")
    ax_gas.set_title("Pore-gas O$_2$ (GDL + CL)")
    ax_gas.legend(loc="lower left")

    ax_ion.set_xlabel("$x$ in CL / $\\mu$m")
    ax_ion.set_ylabel("$c_{O_2}$ ionomer / mol m$^{-3}$")
    ax_ion.set_title("Dissolved vs equilibrium $K_{eq}c_{gas}$")
    ax_ion.legend(
        handles=[
            Line2D([0], [0], color="0.35", ls="-", label="dissolved $c_{ion}$"),
            Line2D([0], [0], color="0.35", ls=":", label="equil. $K_{eq}c_{gas}$"),
        ],
        loc="center right", bbox_to_anchor=(0.88, 0.5),
    )

    add_panel_labels((ax_gas, ax_ion))
    fig.tight_layout()
    if save_path:
        savefig_book(fig, save_path)
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

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)

    # Shade the transport-limited (low-k_v) vs equilibrium (high-k_v) regimes
    ax.axhline(J3_lim, color=BLACK, ls="--", lw=1.0,
               label="Stage 3 ($k_v\\!\\to\\!\\infty$)")
    ax.semilogx(kv[ok], J[ok], marker="o", color=BLUE,
                label="Stage 4 $J_{\\lim}(k_v)$")
    ax.axvline(kv_default, color="0.5", ls=":", lw=0.8)

    # Mark the default operating point
    i_def = int(np.argmin(np.abs(kv - kv_default)))
    if np.isfinite(J[i_def]):
        ax.plot(kv[i_def], J[i_def], marker="*", ms=9, color=RED, zorder=5)
        ax.annotate(f"default $k_v$\n{kv_default:.0e} s$^{{-1}}$",
                    xy=(kv[i_def], J[i_def]),
                    xytext=(8, -22), textcoords="offset points",
                    fontsize=6, color=RED, ha="left")

    ax.set_xlabel("$k_v = k_{MT}\\,a_{GL}$ / s$^{-1}$")
    ax.set_ylabel("Limiting current / mA cm$^{-2}$")
    ax.set_title("Limiting current vs interphase rate")
    ax.legend(loc="center right")

    fig.tight_layout()
    if save_path:
        savefig_book(fig, save_path)
    plt.close(fig)


# ── 9. Stage 4: applied-voltage loss breakdown ────────────────────────────────

def plot_voltage_breakdown_s4(
    voltages, solutions, mesh_gdl, mesh_cl, p,
    save_path: str | None = "stage4_voltage_breakdown.png",
) -> None:
    """
    Stage 4 applied-voltage breakdown (AVB) by the power-loss post-processing
    method of Gerhardt et al., J. Electrochem. Soc. 168, 074503 (2021).

    Each loss is the volumetric power dissipated by a mechanism, divided by the
    cell current density, so the contributions sum EXACTLY to U_OCV - V:

      ohmic, ionic : integral(i_L^2 / kappa_eff) dx / I_cell           (Joule, Eq. 18)
      ohmic, solid : integral(i_s^2 / sigma_eff) dx / I_cell           (Joule, Eq. 18)
      mass-transp. : integral i_ORR (RT/alpha_c F) ln((c_ref/c)^gamma) dx / I_cell (Eq. 23)
                       gas  : reference c = pore-gas O2 c_gas
                       film : reference c = dissolved O2 c_ion (vs K_eq c_gas)
      kinetic      : (U_OCV - V) - (ohmic + mass-transport)     (activation)

    Colors keep the Stage 1 convention (kinetic=black, ohmic-ionic=blue,
    ohmic-solid=green) and split the "mass-transport" family (red) into a
    solid fill for the dominant ionomer-film term and a hatched fill of the
    same color for the minor gas-phase term (Section 6: >4 series combine
    the approved colors with a second visual channel, here a hatch pattern).
    """
    from assembly_stage4 import unpack_s4, compute_current_s4
    from transport import ohmic_face_fluxes

    NG, NC = mesh_gdl.N, mesh_cl.N
    dx     = mesh_cl.dx
    RTaF   = p.R * p.T / (p.alpha * p.F)             # Eq. 23 prefactor RT/(alpha_c F)
    U_ocv  = float(p.U_ORR_eq(p.c_O2_bc))            # inlet-equilibrium OCV

    J, kin, film, gas, ohm_i, ohm_s, tot = ([] for _ in range(7))
    for u, V in zip(solutions, voltages):
        _, ln_cg, ln_ci, phi_L, phi_s = unpack_s4(u, NG, NC)
        c_gas = np.exp(ln_cg)
        c_ion = np.exp(ln_ci)
        c_eq  = p.K_eq_gas_ion * c_gas
        i_ORR = R_ORR(ln_ci, phi_s, phi_L, p)
        Icell = float(np.sum(i_ORR) * dx)
        w     = i_ORR / (np.sum(i_ORR) + 1e-30)

        # ohmic Joule integrals (cell-centred currents from the face fluxes)
        iL = ohmic_face_fluxes(phi_L, dx, p.kappa_L_eff, None, p.phi_L_mem)
        iS = ohmic_face_fluxes(phi_s, dx, p.sigma_s_eff, V, None)
        iLc = 0.5 * (iL[:-1] + iL[1:])
        iSc = 0.5 * (iS[:-1] + iS[1:])
        oi = float(np.sum(iLc ** 2 / p.kappa_L_eff) * dx / Icell)
        os = float(np.sum(iSc ** 2 / p.sigma_s_eff) * dx / Icell)

        # mass-transport (Eq. 23): RT/(alpha_c F) * gamma * <ln(c_ref / c)>_w
        mtg = float(RTaF * p.gamma * np.dot(
            w, np.log(np.maximum(p.c_O2_gas_inlet / c_gas, 1e-30))))
        mtf = float(RTaF * p.gamma * np.dot(
            w, np.log(np.maximum(c_eq / c_ion, 1e-30))))

        t_ = U_ocv - float(V)
        gas.append(mtg); film.append(mtf); ohm_i.append(oi); ohm_s.append(os)
        tot.append(t_)
        kin.append(t_ - oi - os - mtg - mtf)         # activation = remainder
        J.append(compute_current_s4(u, mesh_gdl, mesh_cl, p) * 1e-1)   # mA/cm2

    J = np.asarray(J)
    stacks = [np.clip(np.asarray(v), 0, None) * 1e3
              for v in (kin, film, gas, ohm_i, ohm_s)]
    labels = ["Kinetic (activation)", "Mass transp. (ionomer film)",
              "Mass transp. (gas-phase)", "Ohmic (ionic)", "Ohmic (solid)"]
    colors = [BLACK, RED, RED, BLUE, GREEN]
    hatches = ["", "", "///", "", ""]
    tot = np.asarray(tot) * 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # ── Left: full stacked breakdown (sums exactly to U_OCV - V) ─────────────
    polys = ax1.stackplot(J, *stacks, labels=labels, colors=colors, alpha=0.9)
    for poly, hatch in zip(polys, hatches):
        if hatch:
            poly.set_hatch(hatch)
    ax1.plot(J, tot, color=BLACK, lw=1.0, ls="--", label="$U_{OCV}-V$")
    ax1.set_xlabel(_CURRENT_LBL)
    ax1.set_ylabel("Voltage loss / mV")
    ax1.set_title("Stage 4 applied-voltage breakdown")
    # headroom so the legend sits above the filled stack, not over it
    ax1.set_ylim(0, float(tot.max()) * 1.5)
    ax1.legend(loc="upper left", ncol=2)

    idx = int(np.argmax(J))
    parts = [s[idx] for s in stacks]
    tt = sum(parts)
    ax1.text(0.97, 0.05,
             f"At $J_{{max}}$: kin {100*parts[0]/tt:.0f}%  "
             f"film {100*parts[1]/tt:.1f}%  gas {100*parts[2]/tt:.1f}%  "
             f"ohm {100*(parts[3]+parts[4])/tt:.1f}%",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=6,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

    # ── Right: each non-kinetic loss INDIVIDUALLY (not stacked), expanded ─────
    for s, c, ls, lbl in [
        (stacks[1], RED,   "-",  "Mass transp. (film)"),
        (stacks[3], BLUE,  "-",  "Ohmic (ionic)"),
        (stacks[2], RED,   "--", "Mass transp. (gas)"),
        (stacks[4], GREEN, "-",  "Ohmic (solid)"),
    ]:
        ax2.plot(J, s, color=c, ls=ls, label=lbl)
    ax2.set_xlabel(_CURRENT_LBL)
    ax2.set_ylabel("Loss / mV")
    ax2.set_title("Non-kinetic losses")
    ax2.legend(loc="upper left")

    add_panel_labels((ax1, ax2))
    fig.tight_layout()
    if save_path:
        savefig_book(fig, save_path)
    plt.close(fig)
