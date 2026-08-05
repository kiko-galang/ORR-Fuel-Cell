"""
Regenerate every figure in the repository, in dependency order.

    python make_all_figures.py            # use existing caches (fast)
    python make_all_figures.py --fresh    # delete caches and re-solve from scratch

This is the single entry point for reproducing the chapter figures.  Each
stage is run as a subprocess so a failure in one stage is reported without
aborting the rest of the report.

Stage ordering matters: run_stage4.py warm-starts from stage1_cache.npz and
stage3_cache.npz, so Stages 1 and 3 must run first.  --fresh removes all
caches so the solver re-converges from the initial guess, which is the real
test that the committed figures are reproducible from source alone.

Two tracked images are NOT produced here because they are manual artifacts:

    stage4_o2_profiles.svg          vector export, hand-edited downstream
    stage4_o2_profiles_edited.png   result of that hand editing

Everything else is script-generated and should come back byte-stable on a
fixed matplotlib version (see requirements.txt).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Both of matplotlib's missing-glyph phrasings (plain-text and mathtext paths).
_GLYPH_MARKERS = re.compile(
    r"missing from font|does not have a glyph|substituting with a dummy symbol"
)

# (script, figures it writes) — in required run order.
TARGETS: list[tuple[str, list[str]]] = [
    ("run_stage1.py", [
        "stage1_polarization.png",
        "stage1_profiles.png",
        "stage1_ir_breakdown.png",
        "stage1_consistency.png",
        "stage1_flux_profiles.png",
    ]),
    ("run_stage2.py", ["stage2_comparison.png"]),
    ("run_stage3.py", ["stage3_results.png"]),
    ("run_stage4.py", [
        "stage4_polarization.png",
        "stage4_o2_profiles.png",
        "stage4_voltage_breakdown.png",
        "stage4_kv_sweep.png",
    ]),
    ("plot_meshes.py", [
        "mesh_1d.png", "mesh_1d.svg",
        "mesh_2d_tri.png", "mesh_2d_tri.svg",
        "mesh_3d.png", "mesh_3d.svg",
        "mesh_combined.png", "mesh_combined.svg",
    ]),
    ("gen_profiles_curved.py", ["stage1_profiles_curved.png"]),
]

CACHES = ["stage1_cache.npz", "stage2_cache.npz",
          "stage3_cache.npz", "stage4_cache.npz"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", action="store_true",
                    help="delete solver caches first and re-solve from scratch")
    args = ap.parse_args()

    if args.fresh:
        for c in CACHES:
            path = ROOT / c
            if path.exists():
                path.unlink()
                print(f"  removed {c}")
        print()

    failed: list[str] = []
    missing: list[str] = []

    for script, figures in TARGETS:
        print(f"  {script:<24} ", end="", flush=True)
        proc = subprocess.run([sys.executable, script], cwd=ROOT,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print("FAILED")
            # Surface the reason rather than the whole log.
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
            for line in tail:
                print(f"      {line}")
            failed.append(script)
            continue

        absent = [f for f in figures if not (ROOT / f).exists()]
        missing.extend(absent)
        note = f"  [MISSING: {', '.join(absent)}]" if absent else ""
        print(f"OK   {len(figures) - len(absent)}/{len(figures)} figures{note}")

        # A missing glyph silently renders as a hollow box in the saved figure,
        # so treat matplotlib's font complaints as a hard problem.  The two
        # render paths word it differently -- plain text says "missing from
        # font(s)", mathtext says "does not have a glyph ... dummy symbol" --
        # so match both or the check quietly passes on a broken figure.
        for line in proc.stderr.splitlines():
            if _GLYPH_MARKERS.search(line):
                print(f"      GLYPH: {line.strip()}")
                failed.append(f"{script} (missing glyph)")
                break

    print()
    if failed or missing:
        if failed:
            print(f"  FAILED: {', '.join(failed)}")
        if missing:
            print(f"  NOT PRODUCED: {', '.join(missing)}")
        return 1

    total = sum(len(f) for _, f in TARGETS)
    print(f"  All {total} figures regenerated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
