"""
GCMC mu-scan aggregation — compute half of the old ``aggregate_mu_scan.py``.

Given the per-task NPZ files written by ``simulation/runners/scan_mu.py``:

1. flag non-equilibrated runs with a Mann-Whitney U test (first 25 % vs last 50 %
   of the N trajectory),
2. pool equilibrated runs per mu and compute <N>, <phi> with spread,
3. locate mu* = argmin |<phi> - target_phi| with linear interpolation.

Plotting lives in :mod:`analysis.plotting`.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu


def is_equilibrated(N_traj, alpha=0.05):
    """``(equilibrated: bool, p_value: float)`` from a Mann-Whitney U test
    comparing the first quarter and last half of ``N_traj``."""
    n = len(N_traj)
    if n < 8:
        return True, 1.0
    _, p = mannwhitneyu(N_traj[:n // 4], N_traj[n // 2:], alternative='two-sided')
    return bool(p >= alpha), float(p)


def N_to_phi(N, box_area, r_center):
    return N * np.pi * r_center ** 2 / box_area


def aggregate_by_mu(input_dir, E_AB=None, opening_angle=None, tol=1e-3,
                    alpha_equil=0.05, box_area=None, r_center=1.0):
    """
    Pool equilibrated runs by mu.

    Returns a dict with sorted arrays ``mu``, ``mean_N``, ``std_N``, ``sem_N``,
    ``mean_phi``, ``std_phi``, ``sem_phi``, ``n_sweeps`` plus ``skipped`` and
    ``n_filtered`` counts.  ``box_area`` (if given) is used for phi; otherwise it
    is read from each NPZ.
    """
    files = sorted(Path(input_dir).glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'no NPZ files in {input_dir}')

    groups = defaultdict(list)
    skipped, n_filtered = [], 0
    for f in files:
        try:
            data = np.load(f)
        except Exception:
            continue
        if E_AB is not None:
            v = float(data['E_AB'][0]) if 'E_AB' in data.files else None
            if v is None or abs(v - E_AB) > tol:
                n_filtered += 1
                continue
        if opening_angle is not None:
            v = float(data['opening_angle'][0]) if 'opening_angle' in data.files else None
            if v is None or abs(v - opening_angle) > tol:
                n_filtered += 1
                continue
        mu_val = float(data['mu'][0])
        N_traj = data['N_traj']
        ba = box_area if box_area is not None else (
            float(data['box_area'][0]) if 'box_area' in data.files else None)
        eq, p = is_equilibrated(N_traj, alpha=alpha_equil)
        if not eq:
            skipped.append((f.name, mu_val, p))
            continue
        groups[mu_val].append((N_traj, ba))

    if not groups:
        raise RuntimeError('no equilibrated runs found')

    mu_sorted = sorted(groups)
    out = {k: [] for k in ('mean_N', 'std_N', 'sem_N', 'mean_phi', 'std_phi',
                           'sem_phi', 'n_sweeps')}
    for mu_val in mu_sorted:
        all_N = np.concatenate([g[0] for g in groups[mu_val]])
        ba = groups[mu_val][0][1]
        n = len(all_N)
        mn, sd = float(np.mean(all_N)), float(np.std(all_N))
        se = sd / np.sqrt(n)
        scale = np.pi * r_center ** 2 / ba if ba else np.nan
        out['mean_N'].append(mn)
        out['std_N'].append(sd)
        out['sem_N'].append(se)
        out['mean_phi'].append(mn * scale)
        out['std_phi'].append(sd * scale)
        out['sem_phi'].append(se * scale)
        out['n_sweeps'].append(n)

    result = {'mu': np.array(mu_sorted)}
    result.update({k: np.array(v) for k, v in out.items()})
    result['skipped'] = skipped
    result['n_filtered'] = n_filtered
    return result


def find_mu_star(mu, mean_phi, target_phi):
    """Grid and linear-interpolated mu* for ``<phi> = target_phi``.
    Returns ``(mu_star_grid, mu_star_interp, idx)``."""
    idx = int(np.argmin(np.abs(mean_phi - target_phi)))
    mu_star = float(mu[idx])
    mu_interp = mu_star
    if 0 < idx < len(mu) - 1:
        lo = idx - 1 if mean_phi[idx] > target_phi else idx
        hi = lo + 1
        if hi < len(mu):
            dphi = mean_phi[hi] - mean_phi[lo]
            if abs(dphi) > 1e-9:
                t = (target_phi - mean_phi[lo]) / dphi
                mu_interp = float(mu[lo] + t * (mu[hi] - mu[lo]))
    return mu_star, mu_interp, idx
