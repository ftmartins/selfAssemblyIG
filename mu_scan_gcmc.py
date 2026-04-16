"""
mu_scan_gcmc.py — single-mu hybrid GCMC run for SLURM array parallelism.

Each SLURM task calls this script with a different --mu value.
Results are saved as NPZ files to be aggregated by aggregate_mu_scan.py.

Usage
-----
python mu_scan_gcmc.py \\
    --mu -5.0 \\
    --box_area 1600 \\
    --opening_angle 1.5708 \\
    --kT 1.0 \\
    --E_AB 5.0 \\
    --seed 42 \\
    --output_dir results/mu_scan \\
    --n_equil 10000 \\
    --n_prod 5000
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_gcmc import (
    MU_SCAN_DIR,
    DEFAULT_BOX_AREA,
    DEFAULT_E_AA,
    DEFAULT_E_BB,
    DEFAULT_E_AB,
    DEFAULT_REP_A,
    DEFAULT_KT,
    DEFAULT_MD_STEPS,
    DEFAULT_MD_DT,
    DEFAULT_GAMMA,
    DEFAULT_F_DISP,
    DEFAULT_N_EQUIL,
    DEFAULT_N_PROD,
    DEFAULT_N_INIT,
    DEFAULT_SEED,
)
from modules.gcmc_hybrid import (
    make_gcmc_params,
    generate_ic,
    run_gcmc,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Single-mu hybrid GCMC run (designed for SLURM array jobs)'
    )
    p.add_argument('--mu',            type=float, required=True,
                   help='Chemical potential (kT units)')
    p.add_argument('--box_area',      type=float, default=DEFAULT_BOX_AREA,
                   help=f'Box area A = L^2  (default: {DEFAULT_BOX_AREA})')
    p.add_argument('--opening_angle', type=float, required=True,
                   help='Patch opening angle in radians')
    p.add_argument('--kT',            type=float, default=DEFAULT_KT,
                   help=f'Thermal energy kT  (default: {DEFAULT_KT})')
    p.add_argument('--E_AB',          type=float, default=DEFAULT_E_AB,
                   help=f'A-B cross-patch Morse depth  (default: {DEFAULT_E_AB})')
    p.add_argument('--E_AA',          type=float, default=DEFAULT_E_AA,
                   help=f'A-A like-patch Morse depth   (default: {DEFAULT_E_AA})')
    p.add_argument('--E_BB',          type=float, default=DEFAULT_E_BB,
                   help=f'B-B like-patch Morse depth   (default: {DEFAULT_E_BB})')
    p.add_argument('--rep_A',         type=float, default=DEFAULT_REP_A,
                   help=f'Soft-sphere repulsion prefactor  (default: {DEFAULT_REP_A})')
    p.add_argument('--seed',          type=int,   default=DEFAULT_SEED,
                   help=f'Random seed  (default: {DEFAULT_SEED})')
    p.add_argument('--output_dir',    type=str,   default=MU_SCAN_DIR,
                   help=f'Directory for NPZ output  (default: {MU_SCAN_DIR})')
    p.add_argument('--n_equil',       type=int,   default=DEFAULT_N_EQUIL,
                   help=f'Equilibration sweeps  (default: {DEFAULT_N_EQUIL})')
    p.add_argument('--n_prod',        type=int,   default=DEFAULT_N_PROD,
                   help=f'Production sweeps  (default: {DEFAULT_N_PROD})')
    p.add_argument('--N_init',        type=float,   default=DEFAULT_N_INIT,
                   help=f'Initial particle count for RSA IC  (default: {DEFAULT_N_INIT})')
    p.add_argument('--f_disp',        type=float, default=DEFAULT_F_DISP,
                   help=f'Fraction of moves that are MD displacement  (default: {DEFAULT_F_DISP})')
    p.add_argument('--md_steps',      type=int,   default=DEFAULT_MD_STEPS,
                   help=f'Langevin steps per displacement move  (default: {DEFAULT_MD_STEPS})')
    p.add_argument('--md_dt',         type=float, default=DEFAULT_MD_DT,
                   help=f'Langevin time step  (default: {DEFAULT_MD_DT})')
    p.add_argument('--gamma',         type=float, default=DEFAULT_GAMMA,
                   help=f'Langevin friction coefficient  (default: {DEFAULT_GAMMA})')
    return p.parse_args()


def main():
    args = parse_args()

    L      = float(np.sqrt(args.box_area))
    params = make_gcmc_params(
        opening_angle = args.opening_angle,
        E_AB          = args.E_AB,
        E_AA          = args.E_AA,
        E_BB          = args.E_BB,
        rep_A         = args.rep_A,
    )
    rng = np.random.default_rng(args.seed)
 

    args.N_init = int(args.N_init)

    rho_N    = args.N_init / args.box_area
    mu_ideal = args.kT * np.log(rho_N)

    print(f'mu_scan_gcmc.py')
    print(f'  mu = {args.mu:.4f}  (mu_ideal = {mu_ideal:.3f})')
    print(f'  box: L={L:.3f}  A={args.box_area:.0f}')
    print(f'  opening_angle = {np.degrees(args.opening_angle):.2f} deg  '
          f'E_AB={args.E_AB}  kT={args.kT}')
    print(f'  n_equil={args.n_equil}  n_prod={args.n_prod}  seed={args.seed}')
    print(f'  md_steps={args.md_steps}  md_dt={args.md_dt}  gamma={args.gamma}')

    # Initial condition
    print(f'\nGenerating IC: N={args.N_init} particles via RSA...')
    state = generate_ic(args.N_init, L, params, kT=args.kT, seed=args.seed)
    print(f'  IC ready.  N={state.N}')

    # Run
    t0 = time.time()
    results = run_gcmc(
        state,
        L,
        args.kT,
        args.mu,
        params,
        rng,
        n_equil    = args.n_equil,
        n_prod     = args.n_prod,
        f_disp     = args.f_disp,
        MD_STEPS   = args.md_steps,
        MD_DT      = args.md_dt,
        GAMMA      = args.gamma,
        snapshot_interval = 100,
        verbose    = True,
    )
    elapsed = time.time() - t0

    N_traj = results['N_traj']
    E_traj = results['E_traj']
    print(f'\nDone in {elapsed:.1f}s.')
    print(f'  <N>_prod = {np.mean(N_traj):.2f} +/- {np.std(N_traj):.2f}')
    print(f'  acc_md={results["acc_md"]:.3f}  '
          f'acc_ins={results["acc_ins"]:.4f}  '
          f'acc_del={results["acc_del"]:.4f}')

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    oa_deg = np.degrees(args.opening_angle)
    out_path = os.path.join(
        args.output_dir,
        f'mu{args.mu:.4f}_EAB{args.E_AB:.3f}_oa{oa_deg:.2f}deg_seed{args.seed:d}.npz',
    )
    np.savez(
        out_path,
        # primary results
        mu            = np.array([args.mu]),
        N_traj        = N_traj,
        E_traj        = E_traj,
        acc_md        = np.array([results['acc_md']]),
        acc_ins       = np.array([results['acc_ins']]),
        acc_del       = np.array([results['acc_del']]),
        elapsed_s     = np.array([elapsed]),
        # all simulation params for reproducibility
        box_area      = np.array([args.box_area]),
        opening_angle = np.array([args.opening_angle]),
        kT            = np.array([args.kT]),
        E_AB          = np.array([args.E_AB]),
        E_AA          = np.array([args.E_AA]),
        E_BB          = np.array([args.E_BB]),
        rep_A         = np.array([args.rep_A]),
        seed          = np.array([args.seed]),
        n_equil       = np.array([args.n_equil]),
        n_prod        = np.array([args.n_prod]),
        md_steps      = np.array([args.md_steps]),
        md_dt         = np.array([args.md_dt]),
        gamma         = np.array([args.gamma]),
    )
    print(f'Saved: {out_path}')

    # ── Parameter log (human-readable txt alongside the NPZ) ─────────────────
    import datetime
    log_path = out_path.replace('.npz', '_params.txt')
    with open(log_path, 'w') as flog:
        flog.write(f'mu_scan_gcmc.py  —  parameter log\n')
        flog.write(f'Generated: {datetime.datetime.now().isoformat()}\n')
        flog.write(f'Output:    {out_path}\n')
        flog.write('─' * 50 + '\n')
        flog.write(f'mu            = {args.mu}\n')
        flog.write(f'opening_angle = {args.opening_angle}  ({np.degrees(args.opening_angle):.4f} deg)\n')
        flog.write(f'E_AB          = {args.E_AB}\n')
        flog.write(f'E_AA          = {args.E_AA}\n')
        flog.write(f'E_BB          = {args.E_BB}\n')
        flog.write(f'rep_A         = {args.rep_A}\n')
        flog.write(f'kT            = {args.kT}\n')
        flog.write(f'box_area      = {args.box_area}\n')
        flog.write(f'seed          = {args.seed}\n')
        flog.write(f'n_equil       = {args.n_equil}\n')
        flog.write(f'n_prod        = {args.n_prod}\n')
        flog.write(f'N_init        = {args.N_init}\n')
        flog.write(f'f_disp        = {args.f_disp}\n')
        flog.write(f'md_steps      = {args.md_steps}\n')
        flog.write(f'md_dt         = {args.md_dt}\n')
        flog.write(f'gamma         = {args.gamma}\n')
        flog.write('─' * 50 + '\n')
        flog.write(f'elapsed_s     = {elapsed:.1f}\n')
        flog.write(f'mean_N        = {np.mean(N_traj):.3f}\n')
        flog.write(f'std_N         = {np.std(N_traj):.3f}\n')
        flog.write(f'acc_ins       = {results["acc_ins"]:.4f}\n')
        flog.write(f'acc_del       = {results["acc_del"]:.4f}\n')
    print(f'Params log:  {log_path}')


if __name__ == '__main__':
    main()
