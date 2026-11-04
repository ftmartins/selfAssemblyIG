"""
Parameter grids and SLURM job-index <-> parameter mappings.

Shared by the canonical yield sweep (``run_yield_cluster.py`` /
``preview_grid.py``) and the GCMC mu-scan / phase-diagram runs.
"""

import numpy as np

# ── Canonical-ensemble yield sweep (alpha, zeta) ────────────────────────────
YIELD_ALPHAS_DEG = np.linspace(70, 120, 12)                     # opening angles, deg
YIELD_ZETAS = np.logspace(np.log10(1), np.log10(50), 6)         # selectivity, log-spaced
NUM_REALIZATIONS = 5                                            # runs per (alpha, zeta)
NUM_YIELD_JOBS = len(YIELD_ALPHAS_DEG) * len(YIELD_ZETAS)

# ── GCMC phase-diagram grid (matches simulation/runners/slurm/*.sh) ─────────
GCMC_ANGLES_DEG = np.array([70., 75., 80., 85., 90., 95., 100., 105., 110., 115.])
GCMC_EAB_VALUES = np.array([1., 2., 5., 10., 20., 50.])

# ── GCMC mu scan ───────────────────────────────────────────────────────────
MU_SCAN_VALUES = list(np.arange(-6.0, -1.0, 0.5))
MU_SCAN_N = len(MU_SCAN_VALUES)

# ── Species partition-function scan (analysis.partition) ───────────────────
PARTITION_ANGLES_DEG = np.arange(60.0, 140.1, 5.0)
PARTITION_EAB_VALUES = np.array([5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0])
NUM_PARTITION_JOBS = len(PARTITION_ANGLES_DEG) * len(PARTITION_EAB_VALUES)

# ── Random-seed conventions ────────────────────────────────────────────────
KEY_PARAM_OPT = 0
KEY_PARAM_YIELD = 1273


def yield_job_to_params(job_id, n_zeta=None):
    """``job_id -> (alpha_deg, zeta)`` for the canonical yield sweep."""
    if n_zeta is None:
        n_zeta = len(YIELD_ZETAS)
    alpha_idx = job_id // n_zeta
    zeta_idx = job_id % n_zeta
    return float(YIELD_ALPHAS_DEG[alpha_idx]), float(YIELD_ZETAS[zeta_idx])


def mu_scan_job_to_params(job_id, mu_values=None, eab_values=None):
    """``job_id -> (mu, E_AB)`` for the GCMC mu-scan 2-D array (E_AB fastest)."""
    mu_values = MU_SCAN_VALUES if mu_values is None else mu_values
    eab_values = GCMC_EAB_VALUES if eab_values is None else eab_values
    n_eab = len(eab_values)
    return float(mu_values[job_id // n_eab]), float(eab_values[job_id % n_eab])


def partition_job_to_params(job_id, angles_deg=None, eab_values=None):
    """``job_id -> (angle_deg, E_AB)`` for the partition scan (E_AB fastest)."""
    angles_deg = PARTITION_ANGLES_DEG if angles_deg is None else angles_deg
    eab_values = PARTITION_EAB_VALUES if eab_values is None else eab_values
    n_eab = len(eab_values)
    return float(angles_deg[job_id // n_eab]), float(eab_values[job_id % n_eab])
