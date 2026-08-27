"""
Repo-root-relative filesystem paths.

Replaces the hard-coded absolute cluster path in the old ``config_gcmc.py``.
The repo root is the parent of this ``base/`` package, so everything resolves
correctly whether the code runs on a laptop or a cluster node, as long as the
tree layout is preserved.

All result directories live under ``<repo>/data/`` (git-ignored); figures under
``<repo>/plots/``.
"""

import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
GCMC_RUNS_DIR = DATA_DIR / "gcmc_runs"          # run_gcmc.py full runs
MU_SCAN_DIR = DATA_DIR / "mu_scan"              # scan_mu.py SLURM-array NPZs
YIELD_DIR = DATA_DIR / "yield"                  # canonical yield outputs
DESIGN_DIR = DATA_DIR / "design"                # optimize_design.py outputs
IC_DIR = DATA_DIR / "initial_conditions"        # cached initial conditions
ARCHIVE_DIR = DATA_DIR / "_archive"             # pre-refactor result files

PLOTS_DIR = REPO_ROOT / "plots"
ANALYSIS_DIR = REPO_ROOT / "analysis"
LOGS_DIR = REPO_ROOT / "Logs"

CONDA_ENV = "selfAssemblyIG"


def ensure_dir(path) -> Path:
    """Create ``path`` (and parents) if missing; return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_gcmc_run_dir(
    opening_angle: float,
    E_AB: float,
    mu: float,
    kT: float,
    realization: int = 0,
    base_dir=None,
) -> str:
    """
    Per-run subdirectory path:
        <base>/oa<deg>deg_EAB<E_AB>_mu<mu>_kT<kT>_r<rrr>/
    e.g.  data/gcmc_runs/oa90.00deg_EAB5.000_mu-4.0000_kT1.0_r000/
    """
    base_dir = str(GCMC_RUNS_DIR) if base_dir is None else str(base_dir)
    oa_deg = float(np.degrees(opening_angle))
    name = (
        f"oa{oa_deg:.2f}deg"
        f"_EAB{E_AB:.3f}"
        f"_mu{mu:.4f}"
        f"_kT{kT}"
        f"_r{realization:03d}"
    )
    return os.path.join(base_dir, name)
