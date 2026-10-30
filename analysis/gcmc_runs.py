"""
Discover and load full GCMC runs produced by ``simulation/runners/run_gcmc.py``.

Run directories are named
``oa<deg>deg_EAB<E_AB>_mu<mu>_kT<kT>_r<realization>`` and contain ``run.h5`` (current HDF5 store) or the legacy ``run.npz``.
"""

import re
from pathlib import Path

import numpy as np

from base.paths import GCMC_RUNS_DIR
from base.trajectory import open_trajectory

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
        run = next((d / n for n in ('run.h5', 'run.npz') if (d / n).exists()), None)
        if meta is not None and run is not None:
            out.append((meta, run))
    return out


def _load_run_h5(path):
    """HDF5 run -> dict with ``frames`` (list of ``(N_i, 3)`` [x, y, theta] arrays),
    ``N_traj`` / ``E_traj`` (production frames after the initial one) and ``attrs``."""
    with open_trajectory(path) as r:
        frames = r.frames('production')
        attrs = dict(r.attrs)
    return {
        'frames': [np.column_stack([f.positions, f.orientations]) for f in frames],
        'N_traj': np.array([f.N for f in frames[1:]], np.int32),
        'E_traj': np.array([f.potential_energy for f in frames[1:]], np.float32),
        'attrs': attrs,
    }


def _N_traj(path):
    if str(path).endswith(('.h5', '.hdf5')):
        return _load_run_h5(path)['N_traj']
    return np.load(path)['N_traj']


def load_run(path):
    """Load one ``run.h5`` / ``run.npz`` into a plain dict; for ``.npz`` unpacks the
    NaN-padded ``snapshot_q`` into a list of ``(N_i, 3)`` arrays under ``'frames'``."""
    if str(path).endswith(('.h5', '.hdf5')):
        return _load_run_h5(path)
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
        N = _N_traj(npz)
        if length is None:
            length = len(N)
        if len(N) == length:
            metas.append(meta)
            rows.append(N)
    return metas, (np.vstack(rows) if rows else np.zeros((0, 0)))
