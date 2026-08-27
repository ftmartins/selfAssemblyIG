#!/usr/bin/env python3
"""
run_yield.py — canonical-ensemble yield simulation for one optimized parameter
set (the design output), with freud polygon counting.

Usage:
    python simulation/runners/run_yield.py --params data/design/square_params_YYYYMMDD_HHMMSS.npz
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from base.jax_setup import configure_jax
from base.config import CENTER_RADIUS, DENSITY, kT, get_box_size
from base.grids import KEY_PARAM_YIELD
from base.io import save_pickle_npy, load_pickle_npy
from base.paths import YIELD_DIR
from base.structure_detection import count_polygons, STRUCTURES
from simulation.src.canonical_md import DEFAULTS as YD, my_sim, random_IC
from simulation.src.design import make_params

CHECKPOINT_INTERVAL = 10_000


def parse_arguments():
    p = argparse.ArgumentParser(description='Run yield simulation from optimized parameters')
    p.add_argument('--params', type=str, required=True, help='Optimized-parameter NPZ file')
    p.add_argument('--seed', type=int, default=KEY_PARAM_YIELD)
    p.add_argument('--num_particles', type=int, default=YD.num_particles)
    p.add_argument('--num_steps', type=int, default=YD.num_steps)
    p.add_argument('--equilibration_steps', type=int, default=YD.equilibration_steps)
    p.add_argument('--output_dir', type=str, default=str(YIELD_DIR))
    return p.parse_args()


def load_parameters(param_file):
    if not os.path.exists(param_file):
        raise FileNotFoundError(param_file)
    data = np.load(param_file, allow_pickle=True)
    final_params = data['final_params']
    print(f"Loaded {param_file}  shape={data['shape']}  "
          f"opening_angle={np.rad2deg(final_params[0]):.2f} deg")
    return {
        'shape': str(data['shape']),
        'params': final_params,
        'param_file': param_file,
        'param_id': os.path.splitext(os.path.basename(param_file))[0],
    }


def _run_chunked(yield_params, state, box_size, n_steps, seed, ckpt_path, label):
    done = 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = load_pickle_npy(ckpt_path)
        state, done = ck['state'], int(ck['done'])
        print(f"  [{label}] resumed at {done:,}")
    traj_pos, traj_ori = [], []
    while done < n_steps:
        chunk = min(CHECKPOINT_INTERVAL, n_steps - done)
        key = random.PRNGKey(seed + done)
        state, pos_c, ori_c = my_sim(yield_params, state, chunk, CENTER_RADIUS, box_size,
                                     key, kT=kT)
        if label == 'prod':
            traj_pos.append(np.array(pos_c))
            traj_ori.append(np.array(ori_c))
        done += chunk
        print(f"  [{label}] {done:,}/{n_steps:,}", flush=True)
        if ckpt_path:
            save_pickle_npy(ckpt_path, {'state': state, 'done': done})
        # my_sim returns a full RigidBody state; feed .position back in
        state = state.position if hasattr(state, 'position') else state
    return state, traj_pos, traj_ori


def main():
    configure_jax()
    args = parse_arguments()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    pd = load_parameters(args.params)
    box_size = get_box_size(DENSITY, args.num_particles, CENTER_RADIUS)
    yield_params = make_params(pd['params'])

    run_dir = os.path.join(args.output_dir, pd['param_id'], f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    eq_ckpt = os.path.join(run_dir, 'checkpoint_eq.npy')
    prod_ckpt = os.path.join(run_dir, 'checkpoint_prod.npy')

    print(f"box_size={box_size:.2f}  N={args.num_particles}  "
          f"equil={args.equilibration_steps:,}  prod={args.num_steps:,}")

    init_body = random_IC(yield_params, random.PRNGKey(args.seed), args.num_particles, box_size)

    state, _, _ = _run_chunked(yield_params, init_body, box_size,
                               args.equilibration_steps, args.seed, eq_ckpt, 'equil')
    final_state, traj_pos, traj_ori = _run_chunked(yield_params, state, box_size,
                                                   args.num_steps, args.seed + 1,
                                                   prod_ckpt, 'prod')

    centers = np.array(final_state.center)
    orientations = np.array(final_state.orientation)
    print("\nCounting polygons...")
    counts = count_polygons(centers, orientations, float(yield_params[1]), box_size)

    total_structures = sum(counts[k] for k in STRUCTURES)
    monomers = counts.get('monomers', 0)
    yields_particles = {}
    for name in STRUCTURES:
        yields_particles[name] = counts[name] * {'triangle': 3, 'square': 4, 'pentagon': 5,
                                                 'hexagon': 6, 'heptagon': 7, 'octagon': 8}[name] / args.num_particles
    yields_structure = {name: (counts[name] / total_structures if total_structures else 0.0)
                        for name in STRUCTURES}
    assembly_efficiency = (args.num_particles - monomers) / args.num_particles

    print("Polygon counts:", {k: int(v) for k, v in counts.items()})
    print("Yields (fraction of structures):", {k: round(v, 4) for k, v in yields_structure.items()})
    print(f"Assembly efficiency: {assembly_efficiency:.4f}   monomers: {monomers}")

    out_file = os.path.join(run_dir, f"yields_{ts}.npz")
    np.savez(
        out_file,
        shape=pd['shape'],
        polygon_counts=counts,
        yields_particles=yields_particles,
        yields_structure=yields_structure,
        assembly_efficiency=assembly_efficiency,
        num_particles=args.num_particles, num_steps=args.num_steps,
        box_size=box_size, optimized_params=pd['params'],
        param_source_file=pd['param_file'], timestamp=ts,
        final_centers=centers, final_orientations=orientations,
    )
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
