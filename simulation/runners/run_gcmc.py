"""
run_gcmc.py — full hybrid GCMC simulation for a single parameter set.

Runs a complete grand-canonical (muVT) simulation given an opening angle,
temperature, E_AB and chemical potential mu.  For interactive use or for a
specific parameter point after mu* has been found via the mu-scan workflow
(scan_mu.py + analysis/aggregate_mu_scan.py).

Usage
-----
python simulation/runners/run_gcmc.py \\
    --opening_angle 1.5708 --kT 1.0 --E_AB 5.0 --mu -4.0 \\
    --n_equil 20000 --n_prod 10000 \\
    --output data/gcmc_runs/run.npz

Add --dump_md_energies to also record energy along each MD segment.
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
from base.paths import make_gcmc_run_dir
from simulation.src.gcmc import (
    DEFAULTS,
    make_gcmc_params,
    generate_ic,
    run_gcmc,
    total_energy_pbc,
)


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
                   help='Output NPZ.  Defaults to <run_dir>/run.npz')
    p.add_argument('--snapshot_interval', type=int, default=DEFAULTS.save_interval)
    p.add_argument('--realization', type=int, default=0)
    p.add_argument('--no_checkpoint', action='store_true')
    p.add_argument('--dump_md_energies', action='store_true')
    p.add_argument('--md_energy_chunks', type=int, default=5000)
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


def _pack_snapshots(snapshots, snapshot_N):
    n = len(snapshots)
    if n == 0:
        return np.zeros((0, 0, 3), dtype=np.float32)
    max_N = int(snapshot_N.max())
    out = np.full((n, max_N, 3), np.nan, dtype=np.float32)
    for i, q in enumerate(snapshots):
        out[i, :len(q), :] = q.astype(np.float32)
    return out


def main():
    configure_jax()
    args = parse_args()

    run_dir = make_gcmc_run_dir(
        opening_angle=args.opening_angle, E_AB=args.E_AB,
        mu=args.mu, kT=args.kT, realization=args.realization,
    )
    os.makedirs(run_dir, exist_ok=True)

    output_path = args.output if args.output else os.path.join(run_dir, 'run.npz')
    checkpoint_path = None if args.no_checkpoint else os.path.join(run_dir, 'checkpoint.npy')
    log_path = os.path.join(run_dir, 'params.txt')

    L = float(np.sqrt(args.box_area))
    params = make_gcmc_params(
        opening_angle=args.opening_angle, E_AB=args.E_AB,
        E_AA=args.E_AA, E_BB=args.E_BB, rep_A=args.rep_A,
    )
    rng = np.random.default_rng(args.seed)
    args.N_init = int(args.N_init)
    mu_ideal = args.kT * np.log(args.N_init / args.box_area)

    print('run_gcmc.py')
    print(f'  opening_angle = {np.degrees(args.opening_angle):.2f} deg  E_AB={args.E_AB}  kT={args.kT}')
    print(f'  mu = {args.mu:.4f}  (mu_ideal = {mu_ideal:.3f})')
    print(f'  box: L={L:.3f}  A={args.box_area:.0f}')
    print(f'  n_equil={args.n_equil}  n_prod={args.n_prod}  seed={args.seed}')
    print(f'  run_dir:    {run_dir}')
    print(f'  output:     {output_path}')
    print(f'  checkpoint: {checkpoint_path if checkpoint_path else "disabled"}')

    print(f'\nGenerating IC: N={args.N_init} via RSA...')
    state = generate_ic(args.N_init, L, params, kT=args.kT, seed=args.seed)
    E0 = total_energy_pbc(state.q, L, params)
    print(f'  IC ready.  N={state.N}  E0={E0:.2f}  E0/N={E0 / max(state.N, 1):.3f}')

    t0 = time.time()
    results = run_gcmc(
        state, L, args.kT, args.mu, params, rng,
        n_equil=args.n_equil, n_prod=args.n_prod, f_disp=args.f_disp,
        MD_STEPS=args.md_steps, MD_DT=args.md_dt, GAMMA=args.gamma,
        snapshot_interval=args.snapshot_interval,
        dump_md_energies=args.dump_md_energies,
        md_energy_chunks=args.md_energy_chunks,
        verbose=not args.quiet,
        checkpoint_file=checkpoint_path,
        checkpoint_interval=args.snapshot_interval,
    )
    elapsed = time.time() - t0

    N_traj = results['N_traj']
    E_traj = results['E_traj']
    q_final = results['q_final']
    snapshots = results['snapshots']
    snap_sweeps = results['snapshot_sweeps']

    print(f'\nDone in {elapsed:.1f}s.')
    print(f'  <N>_prod = {np.mean(N_traj):.2f} +/- {np.std(N_traj):.2f}  '
          f'(min={N_traj.min()}, max={N_traj.max()})')
    print(f'  acc_md={results["acc_md"]:.4f}  acc_ins={results["acc_ins"]:.4f}  '
          f'acc_del={results["acc_del"]:.4f}')

    snapshot_N = np.array([len(q) for q in snapshots], dtype=np.int32)
    snapshot_q = _pack_snapshots(snapshots, snapshot_N) if snapshots else \
        np.zeros((0, 0, 3), dtype=np.float32)

    save_dict = dict(
        N_traj=N_traj,
        E_traj=E_traj,
        snapshot_sweeps=snap_sweeps,
        snapshots=np.array(snapshots, dtype=object),
        snapshot_q=snapshot_q,
        snapshot_N=snapshot_N,
        q_final=q_final.astype(np.float64),
        acc_md=np.array([results['acc_md']], dtype=np.float32),
        acc_ins=np.array([results['acc_ins']], dtype=np.float32),
        acc_del=np.array([results['acc_del']], dtype=np.float32),
        elapsed_s=np.array([elapsed], dtype=np.float32),
        opening_angle=np.array([args.opening_angle]),
        kT=np.array([args.kT]),
        E_AB=np.array([args.E_AB]),
        E_AA=np.array([args.E_AA]),
        E_BB=np.array([args.E_BB]),
        rep_A=np.array([args.rep_A]),
        mu=np.array([args.mu]),
        box_area=np.array([args.box_area]),
        seed=np.array([args.seed]),
        n_equil=np.array([args.n_equil]),
        n_prod=np.array([args.n_prod]),
        md_steps=np.array([args.md_steps]),
        md_dt=np.array([args.md_dt]),
        gamma=np.array([args.gamma]),
        f_disp=np.array([args.f_disp]),
        snapshot_interval=np.array([args.snapshot_interval]),
        realization=np.array([args.realization]),
    )
    if args.dump_md_energies and 'md_E_traj' in results:
        save_dict['md_E_traj'] = results['md_E_traj']
        save_dict['md_energy_chunks'] = np.array([args.md_energy_chunks])

    np.savez_compressed(output_path, **save_dict)
    print(f'Saved (compressed): {output_path}')

    if args.dump_md_energies and 'md_q_traj' in results and len(results['md_q_traj']) > 0:
        md_traj_path = os.path.join(run_dir, 'md_trajectories.npy')
        np.save(md_traj_path,
                {'MD_trajectories': {i: q for i, q in enumerate(results['md_q_traj'])}},
                allow_pickle=True)
        print(f'MD trajectories: {md_traj_path}')

    with open(log_path, 'w') as flog:
        flog.write('run_gcmc.py  —  parameter log\n')
        flog.write(f'Generated:    {datetime.datetime.now().isoformat()}\n')
        flog.write(f'run_dir:      {run_dir}\n')
        flog.write(f'Output:       {output_path}\n')
        flog.write('-' * 55 + '\n')
        for k in ('opening_angle', 'E_AB', 'E_AA', 'E_BB', 'rep_A', 'mu', 'kT',
                  'box_area', 'realization', 'seed', 'n_equil', 'n_prod', 'N_init',
                  'f_disp', 'md_steps', 'md_dt', 'gamma', 'snapshot_interval',
                  'dump_md_energies'):
            flog.write(f'{k:16s} = {getattr(args, k)}\n')
        flog.write('-' * 55 + '\n')
        flog.write(f'elapsed_s     = {elapsed:.1f}\n')
        flog.write(f'mean_N        = {np.mean(N_traj):.3f}\n')
        flog.write(f'std_N         = {np.std(N_traj):.3f}\n')
        flog.write(f'acc_ins       = {results["acc_ins"]:.4f}\n')
        flog.write(f'acc_del       = {results["acc_del"]:.4f}\n')
    print(f'Params log:  {log_path}')


if __name__ == '__main__':
    main()
