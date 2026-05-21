# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the model

Each stage is self-contained and run directly:

```powershell
python run_stage1.py   # CL bulk model: ionomer-phase O2 + phi_L + phi_s
python run_stage2.py   # Stage 1 + Sherwood-Reynolds Neumann BC at membrane
python run_stage3.py   # Full model: gas-phase O2 in GDL + CL pores
```

To force a re-run from scratch, delete the corresponding cache file before running:

```powershell
Remove-Item stage1_cache.npz   # or stage2_cache.npz, stage3_cache.npz
```

The AEM model is independent:

```powershell
python "IEM Model\aem_model.py"
```

## Architecture

### Staged model progression

The codebase implements a 1D finite-volume PEMFC cathode model in three stages, each adding physics on top of the previous:

| Stage | Physics | Key files |
|-------|---------|-----------|
| 1 | Ionomer-phase O2 diffusion + ORR Tafel kinetics + ohmic charge conservation in CL only | `assembly_stage1.py`, `run_stage1.py` |
| 2 | Stage 1 + convective O2 loss BC at CL/membrane interface | `assembly_stage2.py`, `run_stage2.py` |
| 3 | Gas-phase O2 in GDL and CL pores via Fickian diffusion; ionomer O2 from local Henry's law equilibrium | `assembly_stage3.py`, `gas_transport.py`, `run_stage3.py` |

The **IEM Model** subdirectory is a separate, self-contained model: a 1D continuum AEM (anion-exchange membrane) using Non-Ideal Thermodynamics + Onsager-Stefan-Maxwell Concentrated Solution Theory with three coupled BVPs and Donnan equilibrium at interfaces.

### Shared infrastructure

- **`params.py`** — `Params` dataclass: all physical constants and geometry. Derived quantities (effective conductivities, Henry's law, Nernst-corrected OCV) are recomputed in `recompute()`. Always call `p.recompute()` after changing a base parameter.
- **`mesh.py`** — `CLMesh` dataclass: uniform cell-centred FV mesh. `make_mesh(p)` for CL, `make_gdl_mesh(p)` for GDL.
- **`kinetics.py`** — `R_ORR()`: volumetric ORR rate [A/m³_CL]. Takes `ln(c_O2)` (log-stored to keep Newton well-scaled), `phi_s`, `phi_L`.
- **`transport.py`** — `diffusion_face_fluxes()` and `ohmic_face_fluxes()`: face-centred Fick/Ohm fluxes for the FV scheme.
- **`gas_transport.py`** — GDL and CL gas-phase fluxes for Stage 3. Interface flux uses harmonic-mean diffusivity.
- **`solver.py`** — Newton solver with FD Jacobian, per-DOF step clamping, and adaptive voltage sweep via natural continuation.
- **`cache.py`** — `.npz` cache for voltage sweep results. Includes stored `Params` JSON so stale-cache warnings can be printed on reload.
- **`plot_results.py`** — diagnostic plots: polarization curve, spatial profiles, IR breakdown, three-way current consistency check.

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

### Coordinate system

`x = 0` is the GDL/CL interface (gas inlet, Dirichlet `c_O2` and `phi_s`). `x = L_CL` is the CL/membrane interface (Dirichlet `phi_L = 0`, no-flux `phi_s`, no-flux O2 in Stage 1). Positive flux is in the +x direction.

### Newton solver details

- FD Jacobian with row scaling to balance O2 (~1e-4) and potential (~1e8) equation scales.
- Step clamping: `|Δ ln(c_O2)| ≤ 5`, `|Δ phi| ≤ 0.2 V`.
- Voltage sweep uses natural continuation from high V to low V with adaptive step size.
- Initial guess: uniform c_O2, phi_L = 0, phi_s = V_cathode (exact OCV solution).

## Dependencies

NumPy, SciPy, Matplotlib. Python 3.11 (per `__pycache__` artifacts).
