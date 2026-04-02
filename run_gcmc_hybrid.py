"""
run_gcmc_hybrid.py — full hybrid GCMC simulation for a single parameter set.

Runs a complete GCMC simulation given an opening angle, temperature, E_AB,
and chemical potential mu.  Designed for interactive/exploratory use or for
running a specific parameter point after mu* has been identified via the
mu-scan workflow (mu_scan_gcmc.py + aggregate_mu_scan.py).

Output (compressed binary NPZ)
-------------------------------
N_traj          : (n_prod,) int32      — particle count every production sweep
E_traj          : (n_snapshots,) f32   — total energy at snapshot times
snapshot_sweeps : (n_snapshots,) int32 — production sweep of each snapshot
snapshot_q      : (n_snapshots, max_N, 3) f32 — positions+orientations, NaN-padded
snapshot_N      : (n_snapshots,) int32 — particle count at each snapshot
q_final         : (N_final, 3) f64    — final positions+orientations
md_E_traj       : (n_disp, n_chunks) f32  — energy along each MD run
                  only present when --dump_md_energies is set

Usage
-----
python run_gcmc_hybrid.py \\
    --opening_angle 1.5708 \\
    --kT 1.0 \\
    --E_AB 5.0 \\
    --mu -4.0 \\
    --n_equil 20000 \\
    --n_prod 10000 \\
    --output results/gcmc_runs/run.npz \\
    --verbose

Add --dump_md_energies to also record energy along each MD segment.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_gcmc import (
    GCMC_RUNS_DIR,
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
    DEFAULT_N_EQUIL_FULL,
    DEFAULT_N_PROD_FULL,
    DEFAULT_SAVE_INTERVAL,
    DEFAULT_N_INIT,
    DEFAULT_SEED,
)
from modules.gcmc_hybrid import (
    make_gcmc_params,
    generate_ic,
    run_gcmc,
    total_energy_pbc,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Full hybrid GCMC simulation (single parameter set)'
    )
    # Required physics parameters
    p.add_argument('--opening_angle',   type=float, required=True,
                   help='Patch opening angle in radians')
    p.add_argument('--kT',              type=float, default=DEFAULT_KT,
                   help=f'Thermal energy kT  (default: {DEFAULT_KT})')
    p.add_argument('--E_AB',            type=float, required=True,
                   help='A-B cross-patch Morse depth')
    p.add_argument('--mu',              type=float, required=True,
                   help='Chemical potential (kT units)')
    # Box
    p.add_argument('--box_area',        type=float, default=DEFAULT_BOX_AREA,
                   help=f'Box area A = L^2  (default: {DEFAULT_BOX_AREA})')
    # Like-patch energies
    p.add_argument('--E_AA',            type=float, default=DEFAULT_E_AA,
                   help=f'A-A like-patch Morse depth  (default: {DEFAULT_E_AA})')
    p.add_argument('--E_BB',            type=float, default=DEFAULT_E_BB,
                   help=f'B-B like-patch Morse depth  (default: {DEFAULT_E_BB})')
    p.add_argument('--rep_A',           type=float, default=DEFAULT_REP_A,
                   help=f'Soft-sphere repulsion prefactor  (default: {DEFAULT_REP_A})')
    # Simulation length
    p.add_argument('--n_equil',         type=int,   default=DEFAULT_N_EQUIL_FULL,
                   help=f'Equilibration sweeps  (default: {DEFAULT_N_EQUIL_FULL})')
    p.add_argument('--n_prod',          type=int,   default=DEFAULT_N_PROD_FULL,
                   help=f'Production sweeps  (default: {DEFAULT_N_PROD_FULL})')
    # MD parameters
    p.add_argument('--md_steps',        type=int,   default=DEFAULT_MD_STEPS,
                   help=f'Langevin steps per displacement move  (default: {DEFAULT_MD_STEPS})')
    p.add_argument('--md_dt',           type=float, default=DEFAULT_MD_DT,
                   help=f'Langevin time step  (default: {DEFAULT_MD_DT})')
    p.add_argument('--gamma',           type=float, default=DEFAULT_GAMMA,
                   help=f'Langevin friction coefficient  (default: {DEFAULT_GAMMA})')
    p.add_argument('--f_disp',          type=float, default=DEFAULT_F_DISP,
                   help=f'Fraction of moves that are MD displacement  (default: {DEFAULT_F_DISP})')
    # IC
    p.add_argument('--N_init',          type=int,   default=DEFAULT_N_INIT,
                   help=f'Starting particle count for RSA IC  (default: {DEFAULT_N_INIT})')
    p.add_argument('--seed',            type=int,   default=DEFAULT_SEED,
                   help=f'Random seed  (default: {DEFAULT_SEED})')
    # Output
    p.add_argument('--output',          type=str,
                   default=os.path.join(GCMC_RUNS_DIR, 'gcmc_run.npz'),
                   help=f'Output NPZ file  (default: {GCMC_RUNS_DIR}/gcmc_run.npz)')
    p.add_argument('--snapshot_interval', type=int, default=DEFAULT_SAVE_INTERVAL,
                   help=f'Save snapshot + energy every N production sweeps  '
                        f'(default: {DEFAULT_SAVE_INTERVAL})')
    p.add_argument('--realization',      type=int, default=0,
                   help='Realization index — appended to the output filename so '
                        'multiple runs with the same parameters do not overwrite '
                        'each other  (default: 0)')
    # MD energy diagnostic
    p.add_argument('--dump_md_energies', action='store_true',
                   help='Record energy at checkpoints along each MD run.  '
                        'Use to verify MD_STEPS is long enough for equilibration.')
    p.add_argument('--md_energy_chunks', type=int, default=10,
                   help='Number of energy checkpoints per MD run  '
                        '(only used with --dump_md_energies, default: 10)')
    p.add_argument('--verbose',         action='store_true',
                   help='Print progress during simulation')
    return p.parse_args()


def _pack_snapshots(snapshots: list, snapshot_N: np.ndarray) -> np.ndarray:
    """
    Pack variable-length position arrays into a (n_snapshots, max_N, 3) float32
    array, NaN-padded.  max_N is the maximum particle count across all snapshots.

    Unpack with:
        q_i = snapshot_q[i, :snapshot_N[i], :]
    """
    n   = len(snapshots)
    if n == 0:
        return np.zeros((0, 0, 3), dtype=np.float32)
    max_N = int(snapshot_N.max())
    out   = np.full((n, max_N, 3), np.nan, dtype=np.float32)
    for i, q in enumerate(snapshots):
        out[i, :len(q), :] = q.astype(np.float32)
    return out


def main():
    args = parse_args()

    # Auto-build output path if user left the default, embedding all key params
    if args.output == os.path.join(GCMC_RUNS_DIR, 'gcmc_run.npz'):
        oa_deg = np.degrees(args.opening_angle)
        args.output = os.path.join(
            GCMC_RUNS_DIR,
            f'run_oa{oa_deg:.2f}deg_EAB{args.E_AB:.3f}_mu{args.mu:.4f}'
            f'_kT{args.kT}_r{args.realization:03d}.npz',
        )

    L      = float(np.sqrt(args.box_area))
    params = make_gcmc_params(
        opening_angle = args.opening_angle,
        E_AB          = args.E_AB,
        E_AA          = args.E_AA,
        E_BB          = args.E_BB,
        rep_A         = args.rep_A,
    )
    rng = np.random.default_rng(args.seed)

    rho_N    = args.N_init / args.box_area
    mu_ideal = args.kT * np.log(rho_N)

    print('run_gcmc_hybrid.py')
    print(f'  opening_angle = {np.degrees(args.opening_angle):.2f} deg  '
          f'E_AB={args.E_AB}  kT={args.kT}')
    print(f'  mu = {args.mu:.4f}  (mu_ideal = {mu_ideal:.3f})')
    print(f'  box: L={L:.3f}  A={args.box_area:.0f}')
    print(f'  n_equil={args.n_equil}  n_prod={args.n_prod}  seed={args.seed}')
    print(f'  MD: {args.md_steps} steps  dt={args.md_dt}  gamma={args.gamma}')
    print(f'  snapshot every {args.snapshot_interval} sweeps')
    if args.dump_md_energies:
        print(f'  dump_md_energies ON  ({args.md_energy_chunks} chunks/run)')
    print(f'  output: {args.output}')

    # Initial condition
    print(f'\nGenerating IC: N={args.N_init} via RSA...')
    state = generate_ic(args.N_init, L, params, kT=args.kT, seed=args.seed)
    E0    = total_energy_pbc(state.q, L, params)
    print(f'  IC ready.  N={state.N}  E0={E0:.2f}  E0/N={E0/max(state.N,1):.3f}')

    # Run
    t0 = time.time()
    results = run_gcmc(
        state,
        L,
        args.kT,
        args.mu,
        params,
        rng,
        n_equil           = args.n_equil,
        n_prod            = args.n_prod,
        f_disp            = args.f_disp,
        MD_STEPS          = args.md_steps,
        MD_DT             = args.md_dt,
        GAMMA             = args.gamma,
        snapshot_interval = args.snapshot_interval,
        dump_md_energies  = args.dump_md_energies,
        md_energy_chunks  = args.md_energy_chunks,
        verbose           = args.verbose,
    )
    elapsed = time.time() - t0

    N_traj   = results['N_traj']
    E_traj   = results['E_traj']
    q_final  = results['q_final']
    snapshots = results['snapshots']
    snap_sweeps = results['snapshot_sweeps']

    print(f'\nDone in {elapsed:.1f}s.')
    print(f'  <N>_prod = {np.mean(N_traj):.2f} +/- {np.std(N_traj):.2f}  '
          f'(min={N_traj.min()}, max={N_traj.max()})')
    if len(E_traj) > 0:
        mean_N_snap = np.array([len(q) for q in snapshots], dtype=np.float32)
        mean_N_snap[mean_N_snap == 0] = 1
        print(f'  <E/N>    = {np.mean(E_traj / mean_N_snap):.3f}')
    print(f'  acc_md   = {results["acc_md"]:.4f}')
    print(f'  acc_ins  = {results["acc_ins"]:.4f}')
    print(f'  acc_del  = {results["acc_del"]:.4f}')
    if args.dump_md_energies and 'md_E_traj' in results:
        md_E = results['md_E_traj']
        print(f'  md_E_traj shape: {md_E.shape}  '
              f'(first chunk mean={md_E[:,0].mean():.2f}  '
              f'last chunk mean={md_E[:,-1].mean():.2f})')

    # Pack snapshots
    snapshot_N = np.array([len(q) for q in snapshots], dtype=np.int32)
    snapshot_q = _pack_snapshots(snapshots, snapshot_N) if snapshots else \
                 np.zeros((0, 0, 3), dtype=np.float32)

    # Save
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    save_dict = dict(
        # Trajectories (every sweep)
        N_traj          = N_traj,
        # Snapshots (every snapshot_interval sweeps)
        E_traj          = E_traj,
        snapshot_sweeps = snap_sweeps,
        snapshot_q      = snapshot_q,   # (n_snap, max_N, 3) float32, NaN-padded
        snapshot_N      = snapshot_N,   # (n_snap,) int32 — real particle count per frame
        # Final state
        q_final         = q_final.astype(np.float64),
        # Acceptance rates
        acc_md          = np.array([results['acc_md']],  dtype=np.float32),
        acc_ins         = np.array([results['acc_ins']], dtype=np.float32),
        acc_del         = np.array([results['acc_del']], dtype=np.float32),
        elapsed_s       = np.array([elapsed],            dtype=np.float32),
        # All simulation parameters for reproducibility
        opening_angle   = np.array([args.opening_angle]),
        kT              = np.array([args.kT]),
        E_AB            = np.array([args.E_AB]),
        E_AA            = np.array([args.E_AA]),
        E_BB            = np.array([args.E_BB]),
        rep_A           = np.array([args.rep_A]),
        mu              = np.array([args.mu]),
        box_area        = np.array([args.box_area]),
        seed            = np.array([args.seed]),
        n_equil         = np.array([args.n_equil]),
        n_prod          = np.array([args.n_prod]),
        md_steps        = np.array([args.md_steps]),
        md_dt           = np.array([args.md_dt]),
        gamma           = np.array([args.gamma]),
        f_disp          = np.array([args.f_disp]),
        snapshot_interval = np.array([args.snapshot_interval]),
        realization       = np.array([args.realization]),
    )
    if args.dump_md_energies and 'md_E_traj' in results:
        save_dict['md_E_traj'] = results['md_E_traj']
        save_dict['md_energy_chunks'] = np.array([args.md_energy_chunks])

    np.savez_compressed(args.output, **save_dict)
    print(f'Saved (compressed): {args.output}')

    # ── Parameter log (human-readable txt alongside the NPZ) ─────────────────
    import datetime
    log_path = args.output.replace('.npz', '_params.txt')
    with open(log_path, 'w') as flog:
        flog.write(f'run_gcmc_hybrid.py  —  parameter log\n')
        flog.write(f'Generated:    {datetime.datetime.now().isoformat()}\n')
        flog.write(f'Output:       {args.output}\n')
        flog.write('─' * 55 + '\n')
        flog.write(f'opening_angle = {args.opening_angle}  ({np.degrees(args.opening_angle):.4f} deg)\n')
        flog.write(f'E_AB          = {args.E_AB}\n')
        flog.write(f'E_AA          = {args.E_AA}\n')
        flog.write(f'E_BB          = {args.E_BB}\n')
        flog.write(f'rep_A         = {args.rep_A}\n')
        flog.write(f'mu            = {args.mu}\n')
        flog.write(f'kT            = {args.kT}\n')
        flog.write(f'box_area      = {args.box_area}\n')
        flog.write(f'realization   = {args.realization}\n')
        flog.write(f'seed          = {args.seed}\n')
        flog.write(f'n_equil       = {args.n_equil}\n')
        flog.write(f'n_prod        = {args.n_prod}\n')
        flog.write(f'N_init        = {args.N_init}\n')
        flog.write(f'f_disp        = {args.f_disp}\n')
        flog.write(f'md_steps      = {args.md_steps}\n')
        flog.write(f'md_dt         = {args.md_dt}\n')
        flog.write(f'gamma         = {args.gamma}\n')
        flog.write(f'snapshot_interval = {args.snapshot_interval}\n')
        flog.write(f'dump_md_energies  = {args.dump_md_energies}\n')
        flog.write('─' * 55 + '\n')
        flog.write(f'elapsed_s     = {elapsed:.1f}\n')
        flog.write(f'mean_N        = {np.mean(N_traj):.3f}\n')
        flog.write(f'std_N         = {np.std(N_traj):.3f}\n')
        flog.write(f'acc_ins       = {results["acc_ins"]:.4f}\n')
        flog.write(f'acc_del       = {results["acc_del"]:.4f}\n')
        if len(E_traj) > 0:
            flog.write(f'mean_E_per_N  = {float(np.mean(E_traj / np.maximum(snapshot_N, 1))):.4f}\n')
    print(f'Params log:  {log_path}')


if __name__ == '__main__':
    main()
