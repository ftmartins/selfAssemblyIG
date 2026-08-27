#!/usr/bin/env python
"""
run_yield_cluster.py — SLURM-array canonical-ensemble (NVT) yield simulation.

Each array task runs one (alpha, zeta) parameter set with several independent
realizations.  The task id indexes the grid in ``base/grids.py``:

    job_id -> (YIELD_ALPHAS_DEG[alpha_idx], YIELD_ZETAS[zeta_idx])

Usage:
    python simulation/runners/run_yield_cluster.py <job_id> [--output-dir DIR]
    #SBATCH --array=0-71
    python simulation/runners/run_yield_cluster.py $SLURM_ARRAY_TASK_ID
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from base.jax_setup import configure_jax
from base.config import DT, CENTER_RADIUS, kT, get_box_size
from base.grids import NUM_YIELD_JOBS, NUM_REALIZATIONS, yield_job_to_params
from base.params import generate_parameter_grid
from base.initial_conditions import rsa_2d
from base.structure_detection import STRUCTURES, BOND_THRESHOLD, compute_yields_from_frame
from base.paths import YIELD_DIR
from simulation.src.canonical_md import DEFAULTS as YD, run_many_particle_md

# Box is sized for YD.num_particles (3000) regardless of the actual particle
# count below — this reproduces the low effective density of the original
# cluster script.
BOX_SIZE_YIELD = get_box_size(0.1, YD.num_particles, CENTER_RADIUS)


def parse_args():
    p = argparse.ArgumentParser(description='NVT yield simulation for one (alpha, zeta) set')
    p.add_argument('job_id', type=int)
    p.add_argument('--output-dir', type=str, default=str(YIELD_DIR))
    p.add_argument('--num-particles', type=int, default=1000)
    p.add_argument('--equilibration-steps', type=int, default=YD.equilibration_steps)
    p.add_argument('--num-steps', type=int, default=YD.num_steps)
    p.add_argument('--sample-interval', type=int, default=YD.sample_interval)
    p.add_argument('--checkpoint-interval', type=int, default=YD.checkpoint_interval)
    p.add_argument('--realizations', type=int, default=NUM_REALIZATIONS)
    return p.parse_args()


def compute_realization_yields(pos_traj, ori_traj, params):
    valid = np.any(pos_traj != 0, axis=(1, 2))
    n_valid = int(np.sum(valid))
    start = max(1, n_valid // 2)
    frame_yields, frame_counts = [], []
    for f in range(start, n_valid):
        y, c, _, _ = compute_yields_from_frame(
            pos_traj[f], ori_traj[f], params.r_patch, params.alpha, BOND_THRESHOLD
        )
        frame_yields.append(y)
        frame_counts.append(c)
    if frame_yields:
        avg_y = {s: float(np.mean([fy[s] for fy in frame_yields])) for s in STRUCTURES}
        avg_c = {s: float(np.mean([fc[s] for fc in frame_counts])) for s in STRUCTURES}
    else:
        avg_y = {s: 0.0 for s in STRUCTURES}
        avg_c = {s: 0.0 for s in STRUCTURES}
    return avg_y, avg_c, len(frame_yields)


def main():
    configure_jax()
    args = parse_args()
    job_id = args.job_id
    if not 0 <= job_id < NUM_YIELD_JOBS:
        print(f"Error: job_id {job_id} out of range [0, {NUM_YIELD_JOBS - 1}]")
        sys.exit(1)

    alpha_deg, zeta = yield_job_to_params(job_id)
    params = generate_parameter_grid(np.deg2rad([alpha_deg]), np.array([zeta]), kT=kT)[0]

    print("=" * 60)
    print(f"JOB {job_id}   alpha={alpha_deg:.1f} deg   zeta={zeta:.3f}")
    print(f"  E_RR={params.E_RR:.3f}  E_BB={params.E_BB:.3f}  E_RB={params.E_RB:.3f}")
    print(f"  particles={args.num_particles}  box={BOX_SIZE_YIELD:.2f}")
    print(f"  equil={args.equilibration_steps:,}  prod={args.num_steps:,}  "
          f"realizations={args.realizations}")
    print("=" * 60)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base = out_dir / 'checkpoints' / f'N{args.num_particles}_alpha{alpha_deg:.1f}_zeta{zeta:.3f}'

    all_pos, all_ori, all_e = [], [], []
    all_yields, all_counts, all_nframes = [], [], []
    t_start = datetime.now()

    for r in range(args.realizations):
        print(f"\n--- Realization {r + 1}/{args.realizations} ---")
        ckpt_dir = ckpt_base / f'r{r}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        import jax
        key = jax.random.PRNGKey(job_id * 1000 + r)
        positions, orientations = rsa_2d(
            args.num_particles, BOX_SIZE_YIELD, seed=job_id * 1000 + r, centered=False,
        )
        final_state, pos_traj, ori_traj, e_traj = run_many_particle_md(
            args.num_particles, BOX_SIZE_YIELD, params, args.num_steps, kT, key, DT,
            positions, orientations,
            sample_interval=args.sample_interval,
            equilibration_steps=args.equilibration_steps,
            checkpoint_dir=ckpt_dir, checkpoint_prefix='ckpt',
            checkpoint_interval=args.checkpoint_interval,
        )
        pos_traj, ori_traj, e_traj = np.array(pos_traj), np.array(ori_traj), np.array(e_traj)
        all_pos.append(pos_traj)
        all_ori.append(ori_traj)
        all_e.append(e_traj)

        avg_y, avg_c, n_frames = compute_realization_yields(pos_traj, ori_traj, params)
        all_yields.append(avg_y)
        all_counts.append(avg_c)
        all_nframes.append(n_frames)
        print(f"  Y_tri={avg_y['triangle']:.4f}  Y_sq={avg_y['square']:.4f}")

        ckpt_file = ckpt_dir / 'ckpt.npz'
        if ckpt_file.exists():
            ckpt_file.unlink()

    total_elapsed = (datetime.now() - t_start).total_seconds()
    print(f"\nAll {args.realizations} realizations done in {total_elapsed:.1f}s")

    traj_file = out_dir / f'job_{job_id:04d}_alpha{alpha_deg:.0f}_zeta{zeta:.2f}_trajectory.npz'
    traj_data = {
        'alpha_deg': alpha_deg, 'selectivity': zeta, 'box_size': BOX_SIZE_YIELD,
        'num_realizations': args.realizations, 'sample_interval': args.sample_interval,
        'equilibration_steps': args.equilibration_steps,
        'num_production_steps': args.num_steps,
    }
    for r in range(args.realizations):
        traj_data[f'r{r}_positions'] = all_pos[r]
        traj_data[f'r{r}_orientations'] = all_ori[r]
        traj_data[f'r{r}_energy'] = all_e[r]
    np.savez_compressed(traj_file, **traj_data)
    print(f"Saved trajectory: {traj_file}")

    yields_file = out_dir / f'job_{job_id:04d}_alpha{alpha_deg:.0f}_zeta{zeta:.2f}_yields.npz'
    yd = {
        'job_id': job_id, 'alpha_deg': alpha_deg, 'selectivity': zeta,
        'num_realizations': args.realizations,
        'E_RR': params.E_RR, 'E_BB': params.E_BB, 'E_RB': params.E_RB,
    }
    for r in range(args.realizations):
        for s in STRUCTURES:
            yd[f'r{r}_Y_{s}'] = all_yields[r][s]
            yd[f'r{r}_count_{s}'] = all_counts[r][s]
        yd[f'r{r}_n_frames'] = all_nframes[r]
    for s in STRUCTURES:
        yd[f'avg_Y_{s}'] = np.mean([all_yields[r][s] for r in range(args.realizations)])
        yd[f'avg_count_{s}'] = np.mean([all_counts[r][s] for r in range(args.realizations)])
    np.savez(yields_file, **yd)
    print(f"Saved yields: {yields_file}")
    print(f"  Y_tri={yd['avg_Y_triangle']:.4f}  Y_sq={yd['avg_Y_square']:.4f}  "
          f"Y_diff={yd['avg_Y_triangle'] - yd['avg_Y_square']:.4f}")


if __name__ == '__main__':
    main()
