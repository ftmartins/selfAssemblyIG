"""
preview_grid.py — print the canonical yield-sweep parameter grid.

Shows the job_id -> (alpha_deg, zeta) mapping so you can set the SLURM --array
directive before submitting run_yield_cluster.py.

Usage:
    python simulation/runners/preview_grid.py
    python simulation/runners/preview_grid.py --save grid.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from base.config import DENSITY, CENTER_RADIUS, kT, get_box_size
from base.grids import (
    YIELD_ALPHAS_DEG, YIELD_ZETAS, NUM_YIELD_JOBS, NUM_REALIZATIONS,
    yield_job_to_params,
)
from simulation.src.canonical_md import DEFAULTS as YD


def main():
    parser = argparse.ArgumentParser(description='Preview yield simulation parameter grid')
    parser.add_argument('--save', type=str, default=None, help='Save grid to NPZ (optional)')
    args = parser.parse_args()

    n_alpha, n_zeta = len(YIELD_ALPHAS_DEG), len(YIELD_ZETAS)
    box_size_yield = get_box_size(DENSITY, YD.num_particles, CENTER_RADIUS)

    print("=" * 70)
    print("YIELD SIMULATION PARAMETER GRID")
    print("=" * 70)
    print(f"  Alphas:          {n_alpha} values, {YIELD_ALPHAS_DEG[0]:.1f} to {YIELD_ALPHAS_DEG[-1]:.1f} deg")
    print(f"  Zetas:           {n_zeta} values, {YIELD_ZETAS[0]:.3f} to {YIELD_ZETAS[-1]:.3f} (log-spaced)")
    print(f"  Total jobs:      {NUM_YIELD_JOBS}")
    print(f"  Realizations:    {NUM_REALIZATIONS} per job")
    print(f"  Particles:       {YD.num_particles}")
    print(f"  Box size:        {box_size_yield:.2f}")
    print(f"  Equilibration:   {YD.equilibration_steps:,} steps")
    print(f"  Production:       {YD.num_steps:,} steps")
    print(f"  Sample interval: {YD.sample_interval}")
    print(f"  Temperature:     {kT}")
    print("=" * 70)
    print(f"\n  SLURM directive:\n    #SBATCH --array=0-{NUM_YIELD_JOBS - 1}\n")

    print(f"{'job_id':>8s}  {'alpha_deg':>10s}  {'selectivity':>12s}")
    print("-" * 34)
    for job_id in range(NUM_YIELD_JOBS):
        a, z = yield_job_to_params(job_id)
        print(f"{job_id:8d}  {a:10.1f}  {z:12.4f}")

    if args.save:
        rows = [yield_job_to_params(j) for j in range(NUM_YIELD_JOBS)]
        np.savez(args.save,
                 job_ids=np.arange(NUM_YIELD_JOBS),
                 alpha_deg=np.array([r[0] for r in rows]),
                 selectivity=np.array([r[1] for r in rows]),
                 YIELD_ALPHAS_DEG=YIELD_ALPHAS_DEG,
                 YIELD_ZETAS=YIELD_ZETAS)
        print(f"\nSaved grid to: {args.save}")


if __name__ == '__main__':
    main()
