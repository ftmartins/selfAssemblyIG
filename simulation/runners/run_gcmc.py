"""
run_gcmc.py — grand-canonical (muVT) run for a single parameter set.

Thin wrapper over the unified entry point (``simulation/runners/run.py
--ensemble grand_canonical``); writes an HDF5 trajectory file.  For a specific
parameter point after mu* has been found via the mu-scan workflow
(scan_mu.py + analysis/aggregate_mu_scan.py).

Usage
-----
python simulation/runners/run_gcmc.py \\
    --opening_angle 1.5708 --kT 1.0 --E_AB 5.0 --mu -4.0 \\
    --n_equil 20000 --n_prod 10000 \\
    --output data/gcmc_runs/run.h5
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from base.jax_setup import configure_jax
from base.paths import make_gcmc_run_dir
from simulation.src.gcmc import DEFAULTS


def parse_args():
    p = argparse.ArgumentParser(description='Full hybrid GCMC simulation (single parameter set)')
    p.add_argument('--opening_angle', type=float, required=True, help='Patch opening angle (rad)')
    p.add_argument('--kT', type=float, default=DEFAULTS.kT)
    p.add_argument('--E_AB', type=float, required=True)
    p.add_argument('--mu', type=float, required=True, help='Chemical potential (kT units)')
    p.add_argument('--box_area', type=float, default=DEFAULTS.box_area, help='Box area A = L^2')
    p.add_argument('--E_AA', type=float, default=DEFAULTS.E_AA)
    p.add_argument('--E_BB', type=float, default=DEFAULTS.E_BB)
    p.add_argument('--rep_A', type=float, default=DEFAULTS.rep_A)
    p.add_argument('--n_equil', type=int, default=DEFAULTS.n_equil_full)
    p.add_argument('--n_prod', type=int, default=DEFAULTS.n_prod_full)
    p.add_argument('--md_steps', type=int, default=DEFAULTS.md_steps)
    p.add_argument('--md_dt', type=float, default=DEFAULTS.md_dt)
    p.add_argument('--gamma', type=float, default=DEFAULTS.gamma)
    p.add_argument('--f_disp', type=float, default=DEFAULTS.f_disp)
    p.add_argument('--N_init', type=float, default=DEFAULTS.n_init)
    p.add_argument('--seed', type=int, default=DEFAULTS.seed)
    p.add_argument('--output', type=str, default=None,
                   help='Output HDF5.  Defaults to <run_dir>/run.h5')
    p.add_argument('--snapshot_interval', type=int, default=DEFAULTS.save_interval)
    p.add_argument('--realization', type=int, default=0)
    p.add_argument('--no_checkpoint', action='store_true',
                   help='start afresh instead of resuming an existing output file')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


def main():
    configure_jax()
    from simulation.src.run import run_simulation
    args = parse_args()

    run_dir = make_gcmc_run_dir(
        opening_angle=args.opening_angle, E_AB=args.E_AB,
        mu=args.mu, kT=args.kT, realization=args.realization,
    )
    os.makedirs(run_dir, exist_ok=True)
    output_path = args.output or os.path.join(run_dir, 'run.h5')

    print(f'run_gcmc.py -> {output_path}')
    results = run_simulation(
        'grand_canonical', output_path, opening_angle=args.opening_angle,
        E_AB=args.E_AB, box_area=args.box_area, n_equil=args.n_equil,
        n_prod=args.n_prod, sample_interval=args.snapshot_interval, seed=args.seed,
        kT=args.kT, E_AA=args.E_AA, E_BB=args.E_BB, rep_A=args.rep_A, mu=args.mu,
        N_init=int(args.N_init), md_steps=args.md_steps, md_dt=args.md_dt,
        gamma=args.gamma, f_disp=args.f_disp, resume=not args.no_checkpoint,
        verbose=not args.quiet)

    N_traj = results['N_traj']
    print(f'  <N>_prod = {np.mean(N_traj):.2f} +/- {np.std(N_traj):.2f}')
    print(f'  acc_md={results["acc_md"]:.4f}  acc_ins={results["acc_ins"]:.4f}  '
          f'acc_del={results["acc_del"]:.4f}')


if __name__ == '__main__':
    main()
