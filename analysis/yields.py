"""
Aggregate canonical-ensemble yield results into phase-diagram grids.

Consumes the ``job_*_yields.npz`` files written by
``simulation/runners/run_yield_cluster.py`` (one per (alpha, zeta) job, already
realization-averaged) and reshapes them onto the (opening angle, selectivity)
grid for :func:`analysis.plotting.plot_phase_diagram`.
"""

from pathlib import Path

import numpy as np

from base.grids import YIELD_ALPHAS_DEG, YIELD_ZETAS
from base.structure_detection import STRUCTURES
from base.paths import YIELD_DIR


def load_yield_grid(input_dir=None, alphas_deg=None, zetas=None):
    """
    Returns ``(alphas_deg, zetas, grids)`` where ``grids[struct]`` is a
    ``(len(zetas), len(alphas_deg))`` array of realization-averaged yields
    (NaN where a job is missing).
    """
    input_dir = Path(input_dir) if input_dir is not None else YIELD_DIR
    alphas_deg = np.asarray(YIELD_ALPHAS_DEG if alphas_deg is None else alphas_deg)
    zetas = np.asarray(YIELD_ZETAS if zetas is None else zetas)

    grids = {s: np.full((len(zetas), len(alphas_deg)), np.nan) for s in STRUCTURES}
    for f in Path(input_dir).glob('job_*_yields.npz'):
        d = np.load(f)
        a = float(d['alpha_deg'])
        z = float(d['selectivity'])
        if not (np.isfinite(a) and np.isfinite(z)):
            continue                    # argmin over NaN would silently pick index 0
        ai = int(np.argmin(np.abs(alphas_deg - a)))
        zi = int(np.argmin(np.abs(zetas - z)))
        for s in STRUCTURES:
            key = f'avg_Y_{s}'
            if key in d.files:
                grids[s][zi, ai] = float(d[key])
    return alphas_deg, zetas, grids


def relative_yield(grids, a='triangle', b='square'):
    """``(Y_a - Y_b) / (Y_a + Y_b)`` in ``[-1, 1]`` (NaN where the sum is 0)."""
    Ya, Yb = grids[a], grids[b]
    tot = Ya + Yb
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(tot > 0, (Ya - Yb) / tot, np.nan)
