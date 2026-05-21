"""
1D cell-centred finite-volume mesh for the cathode catalyst layer.

Domain: x ∈ [0, L_CL]
  x = 0      : GDL / CL interface  (gas inlet, phi_s Dirichlet)
  x = L_CL   : CL / membrane       (phi_L Dirichlet = 0)

Indexing convention (uniform mesh):
  Cells  : 0 … N-1   (cell centres at xc[i] = (i + 0.5) * dx)
  Faces  : 0 … N     (face positions at xf[f] = f * dx)
           face 0   = left  boundary (GDL/CL)
           face N   = right boundary (CL/membrane)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CLMesh:
    L: float   # domain length [m]
    N: int     # number of cells

    def __post_init__(self):
        self.dx  = self.L / self.N
        self.xc  = (np.arange(self.N) + 0.5) * self.dx   # cell centres
        self.xf  = np.arange(self.N + 1) * self.dx        # face positions


def make_mesh(p) -> CLMesh:
    """Convenience constructor from a Params object."""
    return CLMesh(L=p.L_CL, N=p.N_CL)


def make_gdl_mesh(p) -> CLMesh:
    """GDL mesh: x=0 (gas channel) to x=L_GDL (GDL/CL interface)."""
    return CLMesh(L=p.L_GDL, N=p.N_GDL)
