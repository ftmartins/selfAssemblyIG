"""
make_initial_condition.py — generate and cache a non-overlapping initial
condition for canonical-ensemble yield simulations.

Usage:
    python simulation/runners/make_initial_condition.py --num_particles 3000 --seed 42
    python simulation/runners/make_initial_condition.py --num_particles 500 --seed 1 --density 0.15
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from base.config import CENTER_RADIUS, ALPHA, PATCH_SIZE, get_box_size
from base.initial_conditions import rsa_2d, default_min_distance
from base.paths import IC_DIR


def parse_arguments():
    p = argparse.ArgumentParser(description='Generate + cache an RSA initial condition')
    p.add_argument('--num_particles', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--density', type=float, default=0.1)
    p.add_argument('--output_dir', type=str, default=str(IC_DIR))
    p.add_argument('--min_distance', type=float, default=None)
    p.add_argument('--max_attempts', type=int, default=20_000)
    p.add_argument('--force', action='store_true', help='Overwrite an existing IC file')
    return p.parse_args()


def validate_ic(positions, box_size, min_distance):
    n = len(positions)
    if np.any(positions < -1e-9) or np.any(positions >= box_size + 1e-9):
        raise ValueError("Particles outside box bounds")
    min_found = np.inf
    violations = 0
    for i in range(n):
        for j in range(i + 1, n):
            dr = positions[i] - positions[j]
            dr = dr - box_size * np.round(dr / box_size)
            d = np.sqrt(np.sum(dr ** 2))
            min_found = min(min_found, d)
            if d < min_distance - 1e-6:
                violations += 1
    print(f"  min distance found: {min_found:.6f}  (required {min_distance:.6f})  "
          f"violations: {violations}")
    if violations:
        raise ValueError(f"IC validation failed: {violations} overlapping pairs")


def main():
    args = parse_arguments()
    box_size = get_box_size(args.density, args.num_particles, CENTER_RADIUS)
    min_distance = args.min_distance if args.min_distance is not None else default_min_distance()
    os.makedirs(args.output_dir, exist_ok=True)

    out_name = f"ic_N{args.num_particles}_rho{args.density:.4f}_seed{args.seed}.npz"
    out_path = os.path.join(args.output_dir, out_name)
    if os.path.exists(out_path) and not args.force:
        print(f"ERROR: IC file already exists: {out_path}\nUse --force to overwrite")
        sys.exit(1)

    print(f"Generating IC: N={args.num_particles}  rho={args.density}  "
          f"box={box_size:.4f}  min_distance={min_distance:.4f}")
    t0 = datetime.now()
    positions, orientations = rsa_2d(
        args.num_particles, box_size, min_distance=min_distance,
        seed=args.seed, centered=False, max_attempts=args.max_attempts, verbose=True,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  done in {elapsed:.2f}s")

    validate_ic(positions, box_size, min_distance)

    np.savez(
        out_path,
        positions=positions, orientations=orientations,
        seed=args.seed, num_particles=args.num_particles, density=args.density,
        box_size=box_size, min_distance=min_distance,
        center_radius=CENTER_RADIUS, patch_size=PATCH_SIZE, alpha=ALPHA,
        timestamp=datetime.now().isoformat(),
        generation_time_seconds=elapsed, max_attempts=args.max_attempts,
    )
    print(f"IC saved: {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
