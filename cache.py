"""
Binary cache for voltage-sweep solutions.

Format: numpy .npz archive containing
    voltages  : 1-D float64 array
    solutions : 2-D float64 array, shape (n_V, n_DOF)
    params    : JSON-encoded string of the Params dataclass

Stale-cache detection: the stored params JSON is printed on load so the user
can compare against the current Params object and decide whether to re-run.
"""
from __future__ import annotations
from pathlib import Path
import json
import dataclasses
import numpy as np


def save_cache(
    path:      str | Path,
    voltages:  list[float],
    solutions: list[np.ndarray],
    p,            # Params object
    stage:     int = 1,
) -> None:
    """Save a complete voltage sweep to an .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    V_arr = np.asarray(voltages, dtype=np.float64)
    S_arr = np.stack(solutions, axis=0).astype(np.float64)

    # Serialise Params (skip private / derived fields)
    p_dict = {
        k: v for k, v in dataclasses.asdict(p).items()
        if not k.startswith("_")
    }
    p_json = json.dumps(p_dict, indent=2)

    np.savez_compressed(
        path,
        voltages=V_arr,
        solutions=S_arr,
        params_json=np.array(p_json),
        stage=np.array(stage),
    )
    print(f"  [CACHED] {path}  ({len(voltages)} voltages, {S_arr.shape[1]} DOFs)")


def load_cache(path: str | Path) -> tuple[list[float], list[np.ndarray], dict]:
    """
    Load a cached voltage sweep.

    Returns
    -------
    voltages  : list of floats
    solutions : list of 1-D arrays
    p_dict    : dict of stored parameters (for stale-cache check)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")

    data = np.load(path, allow_pickle=True)
    voltages  = list(data["voltages"])
    solutions = [data["solutions"][i] for i in range(data["solutions"].shape[0])]
    p_dict    = json.loads(str(data["params_json"]))

    print(f"  [LOADED] {path}  ({len(voltages)} voltages)")
    return voltages, solutions, p_dict


def cache_exists(path: str | Path) -> bool:
    return Path(path).exists()


def nearest_solution(
    voltages:  list[float],
    solutions: list[np.ndarray],
    V_target:  float,
) -> tuple[np.ndarray, float]:
    """Return the cached solution closest to V_target."""
    idx = int(np.argmin(np.abs(np.array(voltages) - V_target)))
    return solutions[idx], voltages[idx]
