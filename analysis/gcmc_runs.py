"""
Discover and load full GCMC runs produced by ``simulation/runners/run_gcmc.py``.

Run directories are named
``oa<deg>deg_EAB<E_AB>_mu<mu>_kT<kT>_r<realization>`` and contain ``run.npz``.
"""

import re
from pathlib import Path

import numpy as np

from base.paths import GCMC_RUNS_DIR

_DIR_RE = re.compile(
    r'oa(?P<oa_deg>[-\d.]+)deg_EAB(?P<E_AB>[-\d.]+)_mu(?P<mu>[-\d.]+)'
    r'_kT(?P<kT>[-\d.]+)_r(?P<realization>\d+)'
)


def parse_run_dirname(name):
    """dict of floats/ints from a run-directory name, or ``None`` if it does not match."""
    m = _DIR_RE.fullmatch(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        'oa_deg': float(d['oa_deg']),
        'E_AB': float(d['E_AB']),
        'mu': float(d['mu']),
        'kT': float(d['kT']),
        'realization': int(d['realization']),
    }


def discover_runs(root=None):
    """List ``(params_dict, run_npz_path)`` for every completed run under ``root``
    (default :data:`base.paths.GCMC_RUNS_DIR`)."""
    root = Path(root) if root is not None else GCMC_RUNS_DIR
    out = []
    for d in sorted(root.glob('oa*deg_EAB*')):
        meta = parse_run_dirname(d.name)
        npz = d / 'run.npz'
        if meta is not None and npz.exists():
            out.append((meta, npz))
    return out


def load_run(path):
    """Load one ``run.npz`` into a plain dict; unpacks the NaN-padded
    ``snapshot_q`` into a list of ``(N_i, 3)`` arrays under ``'frames'``."""
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if 'snapshot_q' in out and 'snapshot_N' in out:
        q, Ns = out['snapshot_q'], out['snapshot_N']
        out['frames'] = [np.asarray(q[i, :int(Ns[i]), :]) for i in range(len(Ns))]
    return out


def stack_N_trajectories(runs=None):
    """``(meta_list, N_traj_2d)`` for a set of runs sharing a production length,
    stacking their ``N_traj`` row-wise (shorter runs are dropped)."""
    runs = discover_runs() if runs is None else runs
    metas, rows = [], []
    length = None
    for meta, npz in runs:
        N = np.load(npz)['N_traj']
        if length is None:
            length = len(N)
        if len(N) == length:
            metas.append(meta)
            rows.append(N)
    return metas, (np.vstack(rows) if rows else np.zeros((0, 0)))
