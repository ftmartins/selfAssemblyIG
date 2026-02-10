#!/usr/bin/env python
"""
Preview the parameter grid for cluster yield simulations.

Reads YIELD_ALPHAS_DEG and YIELD_ZETAS from config_patchy_particle.py and
prints the mapping from job_id to (alpha_deg, zeta).  Useful for setting
the SLURM --array directive before submission.

Usage:
    python generate_simulation_tasks.py
    python generate_simulation_tasks.py --save grid.npz
"""

import argparse
import numpy as np

from config_patchy_particle import (
    YIELD_ALPHAS_DEG, YIELD_ZETAS, NUM_YIELD_JOBS, NUM_REALIZATIONS,
    NUM_PARTICLES_YIELD, BOX_SIZE_YIELD, NUM_STEPS_YIELD, EQUILIBRATION_STEPS,
    SAMPLE_INTERVAL_YIELD, kT,
)


def main():
    parser = argparse.ArgumentParser(description='Preview yield simulation parameter grid')
    parser.add_argument('--save', type=str, default=None,
                        help='Save grid to NPZ file (optional)')
    args = parser.parse_args()

    n_alpha = len(YIELD_ALPHAS_DEG)
    n_zeta = len(YIELD_ZETAS)

    print("=" * 70)
    print("YIELD SIMULATION PARAMETER GRID")
    print("=" * 70)
    print(f"  Alphas:          {n_alpha} values, {YIELD_ALPHAS_DEG[0]:.1f} to {YIELD_ALPHAS_DEG[-1]:.1f} deg")
    print(f"  Zetas:           {n_zeta} values, {YIELD_ZETAS[0]:.3f} to {YIELD_ZETAS[-1]:.3f} (log-spaced)")
    print(f"  Total jobs:      {NUM_YIELD_JOBS}")
    print(f"  Realizations:    {NUM_REALIZATIONS} per job")
    print(f"  Total sims:      {NUM_YIELD_JOBS * NUM_REALIZATIONS}")
    print(f"  Particles:       {NUM_PARTICLES_YIELD}")
    print(f"  Box size:        {BOX_SIZE_YIELD:.2f}")
    print(f"  Equilibration:   {EQUILIBRATION_STEPS:,} steps")
    print(f"  Production:      {NUM_STEPS_YIELD:,} steps")
    print(f"  Sample interval: {SAMPLE_INTERVAL_YIELD}")
    print(f"  Temperature:     {kT}")
    print("=" * 70)
    print(f"\n  SLURM directive:")
    print(f"    #SBATCH --array=0-{NUM_YIELD_JOBS - 1}")
    print()

    # Print mapping table
    print(f"{'job_id':>8s}  {'alpha_deg':>10s}  {'selectivity':>12s}")
    print("-" * 34)
    for job_id in range(NUM_YIELD_JOBS):
        alpha_idx = job_id // n_zeta
        zeta_idx = job_id % n_zeta
        alpha_deg = YIELD_ALPHAS_DEG[alpha_idx]
        zeta = YIELD_ZETAS[zeta_idx]
        print(f"{job_id:8d}  {alpha_deg:10.1f}  {zeta:12.4f}")

    # Optionally save to NPZ
    if args.save:
        all_alphas = []
        all_zetas = []
        for job_id in range(NUM_YIELD_JOBS):
            alpha_idx = job_id // n_zeta
            zeta_idx = job_id % n_zeta
            all_alphas.append(YIELD_ALPHAS_DEG[alpha_idx])
            all_zetas.append(YIELD_ZETAS[zeta_idx])

        np.savez(args.save,
                 job_ids=np.arange(NUM_YIELD_JOBS),
                 alpha_deg=np.array(all_alphas),
                 selectivity=np.array(all_zetas),
                 YIELD_ALPHAS_DEG=YIELD_ALPHAS_DEG,
                 YIELD_ZETAS=YIELD_ZETAS)
        print(f"\nSaved grid to: {args.save}")


if __name__ == '__main__':
    main()
