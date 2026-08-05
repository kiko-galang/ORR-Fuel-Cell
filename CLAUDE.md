# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the model

Each stage is self-contained and run directly:

```powershell
python run_stage1.py   # CL bulk model: ionomer-phase O2 + phi_L + phi_s
python run_stage2.py   # Stage 1 + Sherwood-Reynolds Neumann BC at membrane
python run_stage3.py   # Full model: gas-phase O2 in GDL + CL pores
python run_stage4.py   # Stage 3 + finite-rate gas<->ionomer transfer + Maxwell-Stefan gas
```

`run_stage4.py` requires `stage1_cache.npz` and `stage3_cache.npz` (run Stage 1
and Stage 3 first); it warm-starts from the Stage 3 solution and prints a 6-item
verification checklist.

To force a re-run from scratch, delete the corresponding cache file before running:

```powershell
Remove-Item stage1_cache.npz   # or stage2_cache.npz, stage3_cache.npz, stage4_cache.npz
```

The AEM model is independent:

```powershell
python "IEM Model\aem_model.py"
```

## Reproducing the chapter figures

`make_all_figures.py` is the single entry point. It runs every figure-producing
script in dependency order, verifies each expected file appeared, and fails if
matplotlib reports a missing glyph:

```powershell
python make_all_figures.py            # use existing caches (fast)
python make_all_figures.py --fresh    # delete caches and re-solve from scratch
```

Figure ownership — every tracked image must have a producer here, or it silently
rots when the plotting code changes:

| Script | Figures |
|--------|---------|
| `run_stage1.py` | `stage1_polarization`, `stage1_profiles`, `stage1_ir_breakdown`, `stage1_consistency`, `stage1_flux_profiles` |
| `run_stage2.py` | `stage2_comparison` |
| `run_stage3.py` | `stage3_results` |
| `run_stage4.py` | `stage4_polarization`, `stage4_o2_profiles`, `stage4_voltage_breakdown`, `stage4_kv_sweep` |
| `plot_meshes.py` | `mesh_1d`, `mesh_2d_tri`, `mesh_3d`, `mesh_combined` (`.png` + `.svg` each) |
| `gen_profiles_curved.py` | `stage1_profiles_curved` |

Two tracked images are deliberately **not** reproducible — they are manual
artifacts, so don't try to regenerate them: `stage4_o2_profiles.svg` (vector
export) and `stage4_o2_profiles_edited.png` (hand-edited from it).

### Book house style (`book_style.py`)

All chapter figures follow `Figure_Preparation_Guide.pdf` (the Wiley book house
style), implemented in `book_style.py` and imported by every figure-producing
script: Arial typography (`font.sans-serif` falls back to Liberation Sans /
DejaVu Sans if Arial is unavailable), fixed mm figure canvases (`FIGSIZE_SMALL`
80x60mm for single-panel plots, `FIGSIZE_LARGE` 180x108mm for 2-panel figures,
`FIGSIZE_LARGE_SQUARE` 180x180mm for the 2x2 profile grids), a 4-color discrete
palette (`COLOR_CYCLE` = black/blue/green/red — `#000000`/`#06739C`/`#00991A`/
`#B30000`) plus `viridis` for genuinely continuous data, and `(a)`/`(b)` panel
labels via `add_panel_labels()`. 600 dpi PNG is used throughout as the guide's
explicitly sanctioned raster fallback to vector PDF.

**Chapter-wide color convention** — kept consistent across every figure so a
color means the same thing everywhere: for a family of ≤4 sampled voltages
(profiles, flux profiles, O2 profiles), index 0 (near-OCV) → black through
index 3 (highest overpotential) → red, via `_voltage_samples()` /
`COLOR_CYCLE[:n]` in `plot_results.py`. In voltage-loss breakdowns: kinetic =
black, ohmic-ionic = blue, ohmic-solid = green, mass-transport = red (the loss
each stage's model exists to expose). A >4-series stack (Stage 4's breakdown
splits mass-transport into film + gas) reuses a color with a hatch pattern
(Section 6: combine the 4 colors with marker/linestyle/hatch) rather than
adding a 5th color.

**`plot_meshes.py` is a deliberate partial exception.** Its 3D isometric mesh
and MEA/serpentine-flow-field illustrations are schematic diagrams, not
"discrete datasets" (Figure_Preparation_Guide.pdf Section 5.1) — literally
applying Section 2's anti-3D/anti-gradient rule to them would mean discarding
the 3D views these figures exist to show. Only fonts, panel-label typography,
and title sizing were updated there; the illustrative sequential palettes
(`cool_sequential`/`warm_sequential` from `customplot.py`) and the isometric
shading/tinting are unchanged by design — don't "fix" this without re-checking
that tradeoff.

### Figure gotchas

- **Write the micron as mathtext `$\mu$`, never a literal `μ`.** Arial has no
  U+03BC glyph, so a bare `μ` renders as a hollow box (or, on some matplotlib
  versions, logs "does not have a glyph ... dummy symbol" instead of "missing
  from font" — `make_all_figures.py` matches both phrasings). Matplotlib only
  warns on stderr, so a broken label is easy to commit unnoticed.
- Every figure gets its canvas size from `book_style.FIGSIZE_*`; don't pass an
  explicit `dpi=` or `figsize=` elsewhere, or that figure will be inconsistent
  with the rest.
- `savefig_book()` deliberately does **not** pass `bbox_inches="tight"` — the
  guide requires the mm canvas size be preserved exactly (Section 10: trim
  whitespace "WITHOUT changing the required figure canvas dimensions"), and
  `bbox_inches="tight"` crops the export away from that fixed size. Call
  `fig.tight_layout()` before saving to fit labels inside the fixed canvas
  instead of relying on autocrop.
- Panel labels are placed just above each panel's axes (not literally inside
  at the guide's example coordinates), specifically so they can never collide
  with an in-panel legend anchored to the same "upper left" corner — a real
  collision hit during development of the stacked voltage-breakdown figures.
- Pin matplotlib (see `requirements.txt`). SVG output embeds a timestamp,
  randomised clip-path ids and the matplotlib version, and minor releases shift
  tight-layout by a pixel or two, so a version change dirties committed figures
  without changing any curve.

## Architecture

### Staged model progression

The codebase implements a 1D finite-volume PEMFC cathode model in four stages, each adding physics on top of the previous:

| Stage | Physics | Key files |
|-------|---------|-----------|
| 1 | Ionomer-phase O2 diffusion + ORR Tafel kinetics + ohmic charge conservation in CL only | `assembly_stage1.py`, `run_stage1.py` |
| 2 | Stage 1 + convective O2 loss BC at CL/membrane interface | `assembly_stage2.py`, `run_stage2.py` |
| 3 | Gas-phase O2 in GDL and CL pores via Fickian diffusion; ionomer O2 from local Henry's law equilibrium | `assembly_stage3.py`, `gas_transport.py`, `run_stage3.py` |
| 4 | Stage 3 + dissolved (ionomer) O2 as an INDEPENDENT field, coupled to pore gas by a finite-rate interphase transfer source `R_PT = k_v·(K_eq·c_gas − c_ion)` (ionomer-film resistance; ORR sink moves to the dissolved phase); gas-phase O2 via binary Maxwell-Stefan / stagnant-film law `N = −D·∇c/(1 − x_O2)`. As `k_v → ∞` recovers Stage 3's equilibrium coupling; as `c_tot → ∞` the MS law reduces to Fick | `assembly_stage4.py`, `gas_transport.py`, `run_stage4.py` |

The **IEM Model** subdirectory is a separate, self-contained model: a 1D continuum AEM (anion-exchange membrane) using Non-Ideal Thermodynamics + Onsager-Stefan-Maxwell Concentrated Solution Theory with three coupled BVPs and Donnan equilibrium at interfaces.

### Shared infrastructure

- **`params.py`** — `Params` dataclass: all physical constants and geometry. Derived quantities (effective conductivities, Henry's law, Nernst-corrected OCV, and the Stage 4 `k_v = k_MT_GL·a_GL` and `c_tot = P/RT`) are recomputed in `recompute()`. Always call `p.recompute()` after changing a base parameter.
- **`mesh.py`** — `CLMesh` dataclass: uniform cell-centred FV mesh. `make_mesh(p)` for CL, `make_gdl_mesh(p)` for GDL.
- **`kinetics.py`** — `R_ORR()`: volumetric ORR rate [A/m³_CL]. Takes `ln(c_O2)` (log-stored to keep Newton well-scaled), `phi_s`, `phi_L`.
- **`transport.py`** — `diffusion_face_fluxes()` and `ohmic_face_fluxes()`: face-centred Fick/Ohm fluxes for the FV scheme.
- **`gas_transport.py`** — GDL and CL gas-phase fluxes. Stage 3 uses plain-Fick helpers (`gdl_face_fluxes`, `cl_gas_face_fluxes`, `interface_flux`); Stage 4 uses the `*_ms` variants that add the `(1 − x_O2)` Maxwell-Stefan/stagnant-film correction. Interface flux uses harmonic-mean diffusivity.
- **`solver.py`** — Newton solver with FD Jacobian, per-DOF step clamping, and adaptive voltage sweep via natural continuation.
- **`cache.py`** — `.npz` cache for voltage sweep results. Includes stored `Params` JSON so stale-cache warnings can be printed on reload.
- **`plot_results.py`** — diagnostic plots: polarization curve, spatial profiles, IR breakdown, three-way current consistency check. Stage 4 adds `plot_stage4_polarization` (Stage 1/3/4 overlay), `plot_stage4_o2_profiles` (gas, dissolved, and equilibrium `K_eq·c_gas`), and `plot_kv_sweep` (limiting current vs `k_v` with the Stage 3 value as the `k_v → ∞` asymptote).

### DOF layout conventions

**Stage 1/2** (3·N_CL DOFs, non-interleaved):
```
u[0·N : 1·N]  = ln(c_O2)   [mol/m³, log-stored]
u[1·N : 2·N]  = phi_L       [V]
u[2·N : 3·N]  = phi_s       [V]
```

**Stage 3** (N_GDL + 3·N_CL DOFs):
```
u[0       : NG]        = ln_c_O2_gdl   GDL gas O2
u[NG      : NG+NC]     = ln_c_O2_cl    CL  gas O2
u[NG+NC   : NG+2*NC]   = phi_L
u[NG+2*NC : NG+3*NC]   = phi_s
```

**Stage 4** (N_GDL + 4·N_CL DOFs — adds an independent dissolved-O2 block):
```
u[0        : NG]        = ln_c_O2_gdl     GDL gas O2
u[NG       : NG+NC]     = ln_c_O2_clgas   CL  pore gas O2
u[NG+NC    : NG+2*NC]   = ln_c_O2_ion     CL  dissolved (ionomer) O2  (NEW)
u[NG+2*NC  : NG+3*NC]   = phi_L
u[NG+3*NC  : NG+4*NC]   = phi_s
```
CL ionomer O2 has no-flux at BOTH faces (its only O2 supply is the volumetric
`R_PT`, not a boundary); the ORR sink lives on the dissolved equation, fed the
real `c_O2_ion` DOF. Step clamps: `|Δ ln(c)| ≤ 5` on all three concentration
blocks, `|Δ phi| ≤ 0.2 V`.

### Coordinate system

`x = 0` is the GDL/CL interface (gas inlet, Dirichlet `c_O2` and `phi_s`). `x = L_CL` is the CL/membrane interface (Dirichlet `phi_L = 0`, no-flux `phi_s`, no-flux O2 in Stage 1). Positive flux is in the +x direction.

### Newton solver details

- FD Jacobian with row scaling to balance O2 (~1e-4) and potential (~1e8) equation scales.
- Step clamping: `|Δ ln(c_O2)| ≤ 5`, `|Δ phi| ≤ 0.2 V`.
- Voltage sweep uses natural continuation from high V to low V with adaptive step size.
- Initial guess: uniform c_O2, phi_L = 0, phi_s = V_cathode (exact OCV solution).

## Dependencies

NumPy, SciPy, Matplotlib. Python 3.11 (per `__pycache__` artifacts).
