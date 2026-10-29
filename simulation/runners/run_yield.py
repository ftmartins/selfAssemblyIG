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
import jax
import jax.numpy as jnp
from jax import random
from jax_md import rigid_body

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from base.jax_setup import configure_jax
from base.config import CENTER_RADIUS, DENSITY, DT, kT, get_box_size
from base.grids import KEY_PARAM_YIELD
from base.trajectory import TrajectoryWriter
from base.paths import YIELD_DIR
from base.structure_detection import count_polygons, STRUCTURES
from simulation.src.canonical_md import DEFAULTS as YD, my_sim, random_IC, energy_fn_from_design_vector
from simulation.src.design import make_params

CHECKPOINT_INTERVAL = 10_000


def parse_arguments():
    p = argparse.ArgumentParser(description='Run yield simulation from optimized parameters')
    p.add_argument('--params', type=str, required=True, help='Optimized-parameter NPZ file')
    p.add_argument('--seed', type=int, default=KEY_PARAM_YIELD)
    p.add_argument('--num_particles', type=int, default=YD.num_particles)
    p.add_argument('--num_steps', type=int, default=YD.num_steps)
    p.add_argument('--equilibration_steps', type=int, default=YD.equilibration_steps)
    p.add_argument('--checkpoint_interval', type=int, default=CHECKPOINT_INTERVAL)
    p.add_argument('--sample_interval', type=int, default=100,
                   help='store every n-th outer-step frame')
    p.add_argument('--equilibration_sample_interval', type=int, default=0,
                   help='0 = store no equilibration frames')
    p.add_argument('--precision', choices=['float32', 'float64'], default='float32')
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


def run_nvt(yield_params, init_body, box_size, equilibration_steps, num_steps, seed,
            trajectory_path, resume=False, chunk_size=CHECKPOINT_INTERVAL,
            frame_stride=1, equilibration_frame_stride=0, precision='float64'):
    """
    Canonical (NVT) run of the nested-scan driver, streaming frames to the HDF5
    store.  ``my_sim`` advances ``chunk**2`` MD steps per call and returns
    ``chunk`` outer-step frames; every ``frame_stride``-th one is stored
    (``equilibration_frame_stride`` = 0 stores none during equilibration),
    stamped with its cumulative MD step.  A restart (position only: ``my_sim``
    re-initialises the thermostat each call) is written after every chunk, so
    ``resume=True`` continues without duplicating frames.  Returns the final
    RigidBody.
    """
    v = np.asarray(yield_params)
    E_AA, E_AB, E_BB = float(v[2]), float(v[3]), float(v[4])
    attrs = dict(
        ensemble='canonical', engine='nested_scan_nose_hoover',
        thermostat='nose_hoover_chain', seed=int(seed), N=int(init_body.center.shape[0]),
        box_size=float(box_size), kT=float(kT), dt=float(DT),
        opening_angle=float(v[1]), E_AA=E_AA, E_BB=E_BB, E_AB=E_AB,
        bond_specificity=E_AB / (E_AA + E_BB),
        num_equilibration_steps=int(equilibration_steps),
        num_production_steps=int(num_steps), sample_interval=int(frame_stride),
        equilibration_sample_interval=int(equilibration_frame_stride),
        step_unit='md_steps_cumulative_chunk_squared',
    )
    energy_fn = energy_fn_from_design_vector(yield_params, box_size, CENTER_RADIUS)[0]
    pe_fn = jax.jit(jax.vmap(lambda c, o: energy_fn(rigid_body.RigidBody(c, o))))

    writer = TrajectoryWriter(trajectory_path, attrs, resume=resume, precision=precision)
    body = init_body
    phase, md_done, done = 'equilibration', 0, 0
    if writer.restart is not None:
        phase, md_done, leaves, extra = writer.restart
        done = int(extra.get('done', 0))
        body = rigid_body.RigidBody(jnp.asarray(leaves[0]), jnp.asarray(leaves[1]))
        print(f"  resumed: phase={phase} md_step={md_done:,}", flush=True)

    def checkpoint(ph, md, d):
        writer.write_restart(ph, md, [np.asarray(body.center), np.asarray(body.orientation)],
                             done=d)

    def record(ph, centers, oris, steps):
        pe = np.asarray(pe_fn(jnp.asarray(centers), jnp.asarray(oris)))
        for c, o, e, st in zip(centers, oris, pe, steps):
            writer.append_frame(np.asarray(c), np.asarray(o), float(e), int(st), phase=ph)

    try:
        for name, total, stride, run_seed in [
                ('equilibration', equilibration_steps, equilibration_frame_stride, seed),
                ('production', num_steps, frame_stride, seed + 1)]:
            if phase == 'done' or (phase == 'production' and name == 'equilibration'):
                continue
            if phase == 'equilibration' and name == 'production':
                phase, md_done, done = 'production', 0, 0
            if phase != name:
                md_done, done = 0, 0
            if stride > 0 and md_done == 0 and writer.n_frames(name) == 0:
                record(name, np.asarray(body.center)[None], np.asarray(body.orientation)[None], [0])
            while done < total:
                chunk = min(chunk_size, total - done)
                key = random.PRNGKey(run_seed + done)
                state, pos_c, ori_c = my_sim(yield_params, body, chunk, CENTER_RADIUS,
                                             box_size, key, kT=kT)
                body = state.position
                if stride > 0:
                    idx = np.arange(stride - 1, chunk, stride)
                    record(name, np.asarray(pos_c)[idx], np.asarray(ori_c)[idx],
                           md_done + (idx + 1) * chunk)
                md_done += chunk * chunk
                done += chunk
                print(f"  [{name}] {done:,}/{total:,}", flush=True)
                checkpoint(name, md_done, done)
            phase = name
            if total == 0:
                checkpoint(name, 0, 0)
        checkpoint('done', md_done, done)
    finally:
        writer.close()
    return body


def main():
    configure_jax()
    args = parse_arguments()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    pd = load_parameters(args.params)
    box_size = get_box_size(DENSITY, args.num_particles, CENTER_RADIUS)
    yield_params = make_params(pd['params'])

    run_dir = os.path.join(args.output_dir, pd['param_id'], f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    traj_path = os.path.join(run_dir, 'trajectory.h5')

    print(f"box_size={box_size:.2f}  N={args.num_particles}  "
          f"equil={args.equilibration_steps:,}  prod={args.num_steps:,}")

    init_body = random_IC(yield_params, random.PRNGKey(args.seed), args.num_particles, box_size)
    final_state = run_nvt(
        yield_params, init_body, box_size, args.equilibration_steps, args.num_steps,
        args.seed, traj_path, resume=True, chunk_size=args.checkpoint_interval,
        frame_stride=args.sample_interval,
        equilibration_frame_stride=args.equilibration_sample_interval,
        precision=args.precision,
    )

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
