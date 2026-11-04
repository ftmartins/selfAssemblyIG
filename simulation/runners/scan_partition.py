"""
scan_partition.py — species partition functions over (theta, E_AB).

One task computes, at one (opening angle, E_AB), ln q_s with errors, ESS and
scores for the monomer, open chains 2..n_chain_max and rings 3..n_ring_max
(``analysis.partition``), plus the ring mode data of ``analysis.ring_modes``
(E_0, J~, eigenvalues, filters, ln I_soft, b_n).  Results go to one NPZ per
point under ``data/partition/``; an existing file with the same settings is
kept, so reruns only fill gaps.  q_s do not depend on density, so density
sweeps afterwards need only ``analysis.mass_action``.

Usage
-----
# one point
python simulation/runners/scan_partition.py --theta_deg 90 --E_AB 10
# one SLURM array task over base.grids.PARTITION_* (E_AB fastest)
python simulation/runners/scan_partition.py --job_id $SLURM_ARRAY_TASK_ID
# the whole grid locally, several processes
python simulation/runners/scan_partition.py --all --workers 8
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from base.grids import PARTITION_ANGLES_DEG, PARTITION_EAB_VALUES, partition_job_to_params
from base.paths import PARTITION_DIR, ensure_dir

SCAN_VERSION = 1
DEFAULT_SETTINGS = dict(n_chain_max=16, n_ring_max=12, n_walkers=16000, n_trials=16,
                        n_close=32, n_replicas=4, E_AA=1.0, E_BB=1.0, kT=1.0, seed=0,
                        modes=True, soft_samples=100_000)


def point_path(theta_deg, E_AB, out_dir=None):
    out_dir = Path(PARTITION_DIR if out_dir is None else out_dir)
    return out_dir / f'oa{theta_deg:.2f}_EAB{E_AB:.3f}.npz'


def _settings_key(settings):
    return json.dumps(dict(settings, version=SCAN_VERSION), sort_keys=True)


def is_cached(path, settings):
    if not Path(path).exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            return str(d['settings']) == _settings_key(settings)
    except Exception:
        return False


def run_point(theta_deg, E_AB, out_dir=None, overwrite=False, verbose=True, **overrides):
    """Compute and save one design point; returns the NPZ path."""
    from base.jax_setup import configure_jax
    configure_jax()
    from analysis.partition import Model, estimate_species, species_arrays
    from analysis import ring_modes as rm

    settings = dict(DEFAULT_SETTINGS, **overrides)
    path = point_path(theta_deg, E_AB, out_dir)
    if not overwrite and is_cached(path, settings):
        if verbose:
            print(f'cached: {path.name}')
        return path
    t0 = time.time()
    model = Model(np.deg2rad(theta_deg), float(E_AB), E_AA=settings['E_AA'],
                  E_BB=settings['E_BB'], kT=settings['kT'])
    est = estimate_species(model, settings['n_chain_max'], settings['n_ring_max'],
                           settings['n_walkers'], settings['n_trials'], settings['n_close'],
                           settings['n_replicas'], settings['seed'])
    out = species_arrays(est)
    out.update(theta_deg=float(theta_deg), E_AB=float(E_AB), E_AA=settings['E_AA'],
               E_BB=settings['E_BB'], kT=settings['kT'], settings=_settings_key(settings))

    if settings['modes']:
        ring_n = np.arange(3, settings['n_ring_max'] + 1)
        n_soft_max = settings['n_ring_max'] - 3
        cols = {k: [] for k in ('E0', 'J_tilde', 'valid', 'ln_q_harmonic', 'ln_q_stiff',
                                'ln_I_soft', 'ln_I_soft_err', 'ln_soft_gaussian', 'b_n',
                                'lambda_stiff_min', 'reasons')}
        soft_ev = np.full((ring_n.size, max(n_soft_max, 1)), np.nan)
        lnq_ring = {int(n): l for k, n, l in zip(out['kind'], out['n'], out['lnq']) if k == 'ring'}
        for i, n in enumerate(ring_n):
            md = rm.analyze_ring(model, int(n))
            ls, lse, _ = rm.soft_integral(md, n_samples=settings['soft_samples'])
            bd = rm.decompose(md, lnq_ring[int(n)], ls)
            cols['E0'].append(md.E0)
            cols['J_tilde'].append(md.J_tilde)
            cols['valid'].append(md.valid)
            cols['ln_q_harmonic'].append(md.ln_q_harmonic())
            cols['ln_q_stiff'].append(md.ln_q_stiff())
            cols['ln_I_soft'].append(ls)
            cols['ln_I_soft_err'].append(lse)
            cols['ln_soft_gaussian'].append(bd.ln_soft_gaussian)
            cols['b_n'].append(bd.b_n)
            cols['lambda_stiff_min'].append(md.stiff_evals.min())
            cols['reasons'].append('; '.join(md.reasons))
            soft_ev[i, :md.n_soft] = md.soft_evals
        out['mode_n'] = ring_n
        out['mode_soft_evals'] = soft_ev
        for k, v in cols.items():
            out[f'mode_{k}'] = np.array(v)

    ensure_dir(path.parent)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **out)
    os.replace(tmp, path)
    if verbose:
        print(f'saved {path.name}  ({time.time() - t0:.0f} s)')
    return path


def _run_star(args):
    theta, E, out_dir, overwrite, overrides = args
    return str(run_point(theta, E, out_dir, overwrite, **overrides))


def run_grid(angles_deg=None, eab_values=None, out_dir=None, workers=1, overwrite=False, **overrides):
    """Every (theta, E_AB) of the grid; cached points are skipped."""
    angles_deg = PARTITION_ANGLES_DEG if angles_deg is None else angles_deg
    eab_values = PARTITION_EAB_VALUES if eab_values is None else eab_values
    jobs = [(float(a), float(e), out_dir, overwrite, overrides) for a in angles_deg for e in eab_values]
    if workers <= 1:
        return [_run_star(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(workers) as pool:
        return pool.map(_run_star, jobs, chunksize=1)


def parse_args():
    p = argparse.ArgumentParser(description='Species partition functions at (theta, E_AB)')
    p.add_argument('--theta_deg', type=float)
    p.add_argument('--E_AB', type=float)
    p.add_argument('--job_id', type=int, help='index into base.grids.PARTITION_* (E_AB fastest)')
    p.add_argument('--all', action='store_true', help='the whole default grid')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--overwrite', action='store_true')
    for k, v in DEFAULT_SETTINGS.items():
        if isinstance(v, bool):
            p.add_argument(f'--no_{k}', dest=k, action='store_false')
        else:
            p.add_argument(f'--{k}', type=type(v), default=v)
    return p.parse_args()


def main():
    a = parse_args()
    overrides = {k: getattr(a, k) for k in DEFAULT_SETTINGS}
    if a.all:
        run_grid(out_dir=a.output_dir, workers=a.workers, overwrite=a.overwrite, **overrides)
        return
    if a.job_id is not None:
        theta, E = partition_job_to_params(a.job_id)
    else:
        if a.theta_deg is None or a.E_AB is None:
            raise SystemExit('give --theta_deg and --E_AB, --job_id, or --all')
        theta, E = a.theta_deg, a.E_AB
    run_point(theta, E, a.output_dir, a.overwrite, **overrides)


if __name__ == '__main__':
    main()
