# A staged 1D PEMFC cathode model

Example code accompanying the chapter. It builds a 1D finite-volume model of a
PEM fuel cell cathode in **four stages**, each adding one physical mechanism on
top of the previous one, so the effect of each mechanism can be seen in
isolation rather than all at once in a finished model.

Only NumPy, SciPy and Matplotlib are required.

```bash
pip install -r requirements.txt

python run_stage1.py   # ionomer-phase O2 + ORR kinetics + ohmic transport in the CL
python run_stage2.py   # + convective O2 loss at the CL/membrane boundary
python run_stage3.py   # + gas-phase O2 in the GDL and CL pores
python run_stage4.py   # + finite-rate gas->ionomer transfer, Maxwell-Stefan gas
```

Run them in order. Each script solves a full voltage sweep, prints a set of
physicality checks, writes its figures as PNG + PDF, and caches its solution to
`stage<N>_cache.npz`. Stage 4 warm-starts from the Stage 1 and Stage 3 caches,
so those two must have been run first. Delete a cache to force a re-solve.

Stage 1 takes a few seconds; Stage 4 with its `k_v` sweep takes a few minutes.

## The four stages

| Stage | Physics added | Files |
|-------|---------------|-------|
| 1 | O2 diffusion in the ionomer, ORR Tafel kinetics, ohmic charge conservation — catalyst layer only | `assembly_stage1.py`, `run_stage1.py` |
| 2 | A convective O2 loss (Sherwood–Reynolds) boundary condition at the CL/membrane interface | `assembly_stage2.py`, `run_stage2.py` |
| 3 | Gas-phase O2 in the GDL and CL pores by Fickian diffusion; ionomer O2 from local Henry's-law equilibrium | `assembly_stage3.py`, `run_stage3.py` |
| 4 | Dissolved O2 as an *independent* field, coupled to the pore gas by a finite-rate interphase source `R_PT = k_v·(K_eq·c_gas − c_ion)`; gas-phase O2 by the binary Maxwell–Stefan / stagnant-film law `N = −D·∇c/(1 − x_O2)` | `assembly_stage4.py`, `run_stage4.py` |

Stage 4 contains both earlier models as limits, which is worth checking as a
sanity test on any implementation: as `k_v → ∞` the interphase resistance
vanishes and Stage 3's equilibrium coupling is recovered, and as `c_tot → ∞`
the Maxwell–Stefan flux reduces to Fick's law. `run_stage4.py` sweeps `k_v` and
plots the limiting current against the Stage 3 value as its asymptote.

## Shared modules

| File | Contents |
|------|----------|
| `params.py` | `Params` dataclass — all physical constants and geometry. Derived quantities (effective conductivities, Henry's law, Nernst-corrected OCV, `k_v`, `c_tot`) are recomputed in `recompute()`; **call it after changing any base parameter**. |
| `mesh.py` | `CLMesh` — a uniform cell-centred finite-volume mesh. `make_mesh(p)` for the catalyst layer, `make_gdl_mesh(p)` for the GDL. |
| `kinetics.py` | `R_ORR()` — volumetric ORR rate [A/m³]. Takes `ln(c_O2)` rather than `c_O2`. |
| `transport.py` | Face-centred Fick and Ohm fluxes for the FV scheme. |
| `gas_transport.py` | GDL and CL pore-gas fluxes. Plain-Fick helpers for Stage 3; the `*_ms` variants add the `(1 − x_O2)` Maxwell–Stefan correction for Stage 4. Interface flux uses a harmonic-mean diffusivity. |
| `solver.py` | Newton solver with a finite-difference Jacobian, per-DOF step clamping, and an adaptive voltage sweep by natural continuation. |
| `cache.py` | `.npz` cache of sweep results, storing the `Params` used so a stale cache can be flagged on reload. |
| `plot_results.py` | Polarization curves, spatial profiles, IR breakdown, a three-way current-consistency check, and the Stage 4 figures. |
| `book_style.py` | Figure house style — fixed mm canvas sizes, typography, a 4-colour palette, panel labels. Applied on import. |

## Conventions

**Coordinates.** `x = 0` is the GDL/CL interface (gas inlet; Dirichlet in `c_O2`
and `phi_s`). `x = L_CL` is the CL/membrane interface (Dirichlet `phi_L = 0`,
no-flux in `phi_s`). Positive flux is in `+x`.

**Concentrations are stored as `ln(c)`**, not `c`. O2 concentration varies over
orders of magnitude across a voltage sweep and must stay positive; solving for
its logarithm enforces positivity automatically and keeps the Newton step
well scaled. This is why `R_ORR()` takes `ln(c_O2)` as its argument.

**Scaling.** The O2 equations carry residuals of order 1e-4 and the potential
equations order 1e8, so the Jacobian is row-scaled before each solve. Newton
steps are clamped to `|Δ ln c| ≤ 5` and `|Δ phi| ≤ 0.2 V`.

**Degrees of freedom.** Blocks are contiguous, not interleaved. With `NG` GDL
cells and `NC` catalyst-layer cells:

```
Stage 1/2 (3·NC)             Stage 3 (NG + 3·NC)          Stage 4 (NG + 4·NC)
  ln(c_O2)                     ln_c_O2_gdl                  ln_c_O2_gdl
  phi_L                        ln_c_O2_cl                   ln_c_O2_clgas
  phi_s                        phi_L                        ln_c_O2_ion   <- added
                               phi_s                        phi_L
                                                            phi_s
```

In Stage 4 the dissolved-O2 field has **no-flux at both faces** — its only
supply is the volumetric transfer term `R_PT`, not a boundary — and the ORR
sink moves onto the dissolved equation, where it is fed the real `c_O2_ion`
rather than an equilibrium estimate.

## Verifying a run

Each script prints its own checks, and it is worth watching them rather than
going straight to the figures. Stage 1 confirms that `phi_s` decreases
monotonically, that `phi_L ≥ 0`, that O2 depletes from inlet to membrane, and
that the total current agrees when computed three independent ways (from the
solid-phase flux, the ionic flux, and the volumetric reaction integral). Stage
4 adds a global O2 mass balance — GDL inflow against total ORR consumption —
and reports a 6-item checklist.

A model that produces a plausible-looking polarization curve while failing the
three-way current check is not converged; it is the check most likely to catch
a sign error or a mis-indexed DOF block.
