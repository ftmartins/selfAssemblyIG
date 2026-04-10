"""
config_gcmc.py — shared configuration for the hybrid GCMC workflow.

Import this in mu_scan_gcmc.py, run_gcmc_hybrid.py, and aggregate_mu_scan.py
to keep paths, defaults, and physics constants in one place.

All paths are relative to the project root (the directory containing this file).
"""

import os
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Output directories ────────────────────────────────────────────────────────
# Results from the mu-scan SLURM array jobs (one NPZ per task)
MU_SCAN_DIR      = os.path.join(PROJECT_ROOT, 'results', 'mu_scan')

# Results from single full runs (run_gcmc_hybrid.py)
GCMC_RUNS_DIR    = os.path.join(PROJECT_ROOT, 'results', 'gcmc_runs')

# Plots from aggregate_mu_scan.py
PLOTS_DIR        = os.path.join(PROJECT_ROOT, 'results', 'plots')

# SLURM log files
LOGS_DIR         = os.path.join(PROJECT_ROOT, 'Logs')

# ── Box / density defaults ────────────────────────────────────────────────────
DEFAULT_BOX_AREA    = 1600.0    # L^2 = 40^2
DEFAULT_TARGET_PHI  = 0.2      # target area fraction for mu* search
DEFAULT_R_CENTER    = 1.0      # particle core radius (also = r_patch)

# Derived: N_target for phi=0.2
DEFAULT_N_TARGET = DEFAULT_TARGET_PHI * DEFAULT_BOX_AREA / (np.pi * DEFAULT_R_CENTER**2)

# ── Interaction energy defaults ───────────────────────────────────────────────
DEFAULT_E_AA  = 1.0        # A-A like-patch Morse depth
DEFAULT_E_BB  = 1.0        # B-B like-patch Morse depth
DEFAULT_E_AB  = 5.0        # A-B cross-patch Morse depth
DEFAULT_REP_A = 10_000.0   # soft-sphere repulsion prefactor

# ── Thermodynamic defaults ────────────────────────────────────────────────────
DEFAULT_KT    = 1.0        # thermal energy (kT)

# ── MD / GCMC defaults ────────────────────────────────────────────────────────
DEFAULT_MD_STEPS  = 5_000   # Langevin steps per displacement move
DEFAULT_MD_DT     = 1e-2    # Langevin time step
DEFAULT_GAMMA     = 1.0     # Langevin friction coefficient
DEFAULT_F_DISP    = 0.5     # fraction of MC moves that are MD displacement

# ── Simulation length defaults ────────────────────────────────────────────────
DEFAULT_N_EQUIL        = 10_000   # equilibration sweeps (mu scan)
DEFAULT_N_PROD         = 5_000    # production sweeps (mu scan)
DEFAULT_N_EQUIL_FULL   = 20_000   # equilibration sweeps (full run_gcmc_hybrid)
DEFAULT_N_PROD_FULL    = 10_000   # production sweeps (full run_gcmc_hybrid)
DEFAULT_SAVE_INTERVAL  = 100      # snapshot save interval (full run)
DEFAULT_N_INIT         = 16       # starting particle count for RSA IC
DEFAULT_SEED           = 42       # default random seed

# ── Equilibration check ───────────────────────────────────────────────────────
DEFAULT_ALPHA_EQUIL = 0.05  # Mann-Whitney p-value threshold

# ── mu scan grid ──────────────────────────────────────────────────────────────
# Used by submit_mu_scan.sh and aggregate_mu_scan.py for reference.
# 20 values from -6.0 to -1.25.
MU_SCAN_VALUES = list(np.arange(-6.0, -1.0, 0.25))   # [-6.0, -5.75, ..., -1.25]
MU_SCAN_N      = len(MU_SCAN_VALUES)                  # 20

# ── Conda environment name ────────────────────────────────────────────────────
CONDA_ENV = 'selfAssemblyIG'


# ── Run-directory helper ───────────────────────────────────────────────────────

def make_run_dir(
    opening_angle: float,
    E_AB: float,
    mu: float,
    kT: float,
    realization: int = 0,
    base_dir: str = None,
) -> str:
    """
    Return the path of the per-run subdirectory under GCMC_RUNS_DIR (or
    *base_dir* if supplied).

    Pattern:
        <base>/oa<deg>deg_EAB<E_AB>_mu<mu>_kT<kT>_r<rrr>/

    Example:
        results/gcmc_runs/oa90.00deg_EAB5.000_mu-4.0000_kT1.0_r000/
    """
    if base_dir is None:
        base_dir = GCMC_RUNS_DIR
    oa_deg = float(np.degrees(opening_angle))
    name = (
        f'oa{oa_deg:.2f}deg'
        f'_EAB{E_AB:.3f}'
        f'_mu{mu:.4f}'
        f'_kT{kT}'
        f'_r{realization:03d}'
    )
    return os.path.join(base_dir, name)
