"""
scan_mu.py — single-mu hybrid GCMC run for SLURM array parallelism.

Each SLURM task calls this with a different --mu; results are NPZ files under
data/mu_scan/ aggregated afterwards by analysis/aggregate_mu_scan.py.

Usage
-----
python simulation/runners/scan_mu.py \\
    --mu -5.0 --box_area 1600 --opening_angle 1.5708 --kT 1.0 --E_AB 5.0 \\
    --seed 42 --n_equil 10000 --n_prod 5000
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from base.jax_setup import configure_jax
from base.paths import MU_SCAN_DIR
from simulation.src.gcmc import DEFAULTS, make_gcmc_params, generate_ic, run_gcmc


def parse_args():
    p = argparse.ArgumentParser(description='Single-mu hybrid GCMC run (SLURM array)')
    p.add_argument('--mu', type=float, required=True)
    p.add_argument('--box_area', type=float, default=DEFAULTS.box_area)
    p.add_argument('--opening_angle', type=float, required=True)
    p.add_argument('--kT', type=float, default=DEFAULTS.kT)
    p.add_argument('--E_AB', type=float, default=DEFAULTS.E_AB)
    p.add_argument('--E_AA', type=float, default=DEFAULTS.E_AA)
    p.add_argument('--E_BB', type=float, default=DEFAULTS.E_BB)
    p.add_argument('--rep_A', type=float, default=DEFAULTS.rep_A)
    p.add_argument('--seed', type=int, default=DEFAULTS.seed)
    p.add_argument('--output_dir', type=str, default=str(MU_SCAN_DIR))
    p.add_argument('--n_equil', type=int, default=DEFAULTS.n_equil)
    p.add_argument('--n_prod', type=int, default=DEFAULTS.n_prod)
    p.add_argument('--N_init', type=float, default=DEFAULTS.n_init)
    p.add_argument('--f_disp', type=float, default=DEFAULTS.f_disp)
    p.add_argument('--md_steps', type=int, default=DEFAULTS.md_steps)
    p.add_argument('--md_dt', type=float, default=DEFAULTS.md_dt)
    p.add_argument('--gamma', type=float, default=DEFAULTS.gamma)
    return p.parse_args()


def main():
    configure_jax()
    args = parse_args()

    L = float(np.sqrt(args.box_area))
    params = make_gcmc_params(
        opening_angle=args.opening_angle, E_AB=args.E_AB,
        E_AA=args.E_AA, E_BB=args.E_BB, rep_A=args.rep_A,
    )
    rng = np.random.default_rng(args.seed)
    args.N_init = int(args.N_init)
    mu_ideal = args.kT * np.log(args.N_init / args.box_area)

    print('scan_mu.py')
    print(f'  mu = {args.mu:.4f}  (mu_ideal = {mu_ideal:.3f})')
    print(f'  box: L={L:.3f}  A={args.box_area:.0f}')
    print(f'  opening_angle = {np.degrees(args.opening_angle):.2f} deg  E_AB={args.E_AB}  kT={args.kT}')
    print(f'  n_equil={args.n_equil}  n_prod={args.n_prod}  seed={args.seed}')

    print(f'\nGenerating IC: N={args.N_init} via RSA...')
    state = generate_ic(args.N_init, L, params, kT=args.kT, seed=args.seed)
    print(f'  IC ready.  N={state.N}')

    t0 = time.time()
    results = run_gcmc(
        state, L, args.kT, args.mu, params, rng,
        n_equil=args.n_equil, n_prod=args.n_prod, f_disp=args.f_disp,
        MD_STEPS=args.md_steps, MD_DT=args.md_dt, GAMMA=args.gamma,
        snapshot_interval=100, verbose=True,
    )
    elapsed = time.time() - t0

    N_traj = results['N_traj']
    E_traj = results['E_traj']
    print(f'\nDone in {elapsed:.1f}s.')
    print(f'  <N>_prod = {np.mean(N_traj):.2f} +/- {np.std(N_traj):.2f}')
    print(f'  acc_md={results["acc_md"]:.3f}  acc_ins={results["acc_ins"]:.4f}  '
          f'acc_del={results["acc_del"]:.4f}')

    os.makedirs(args.output_dir, exist_ok=True)
    oa_deg = np.degrees(args.opening_angle)
    out_path = os.path.join(
        args.output_dir,
        f'mu{args.mu:.4f}_EAB{args.E_AB:.3f}_oa{oa_deg:.2f}deg_seed{args.seed:d}.npz',
    )
    np.savez(
        out_path,
        mu=np.array([args.mu]), N_traj=N_traj, E_traj=E_traj,
        acc_md=np.array([results['acc_md']]),
        acc_ins=np.array([results['acc_ins']]),
        acc_del=np.array([results['acc_del']]),
        elapsed_s=np.array([elapsed]),
        box_area=np.array([args.box_area]),
        opening_angle=np.array([args.opening_angle]),
        kT=np.array([args.kT]), E_AB=np.array([args.E_AB]),
        E_AA=np.array([args.E_AA]), E_BB=np.array([args.E_BB]),
        rep_A=np.array([args.rep_A]), seed=np.array([args.seed]),
        n_equil=np.array([args.n_equil]), n_prod=np.array([args.n_prod]),
        md_steps=np.array([args.md_steps]), md_dt=np.array([args.md_dt]),
        gamma=np.array([args.gamma]),
    )
    print(f'Saved: {out_path}')

    log_path = out_path.replace('.npz', '_params.txt')
    with open(log_path, 'w') as flog:
        flog.write('scan_mu.py  —  parameter log\n')
        flog.write(f'Generated: {datetime.datetime.now().isoformat()}\n')
        flog.write(f'Output:    {out_path}\n')
        flog.write('-' * 50 + '\n')
        for k in ('mu', 'opening_angle', 'E_AB', 'E_AA', 'E_BB', 'rep_A', 'kT',
                  'box_area', 'seed', 'n_equil', 'n_prod', 'N_init', 'f_disp',
                  'md_steps', 'md_dt', 'gamma'):
            flog.write(f'{k:16s} = {getattr(args, k)}\n')
        flog.write('-' * 50 + '\n')
        flog.write(f'elapsed_s     = {elapsed:.1f}\n')
        flog.write(f'mean_N        = {np.mean(N_traj):.3f}\n')
        flog.write(f'std_N         = {np.std(N_traj):.3f}\n')
    print(f'Params log:  {log_path}')


if __name__ == '__main__':
    main()
