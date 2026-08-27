#!/usr/bin/env python3
"""
timestep_comparison.py — run short NVT simulations at dt, dt/2, dt/10 and save
their trajectories, to check MD time-step convergence.

    python benchmarks/timestep_comparison.py --params data/design/square_params_*.npz
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax
from jax_md import simulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from base.jax_setup import configure_jax
from base.config import DT, CENTER_RADIUS, DENSITY, kT, get_box_size
from simulation.src.canonical_md import random_IC, energy_fn_from_design_vector
from simulation.src.design import make_params


def parse_arguments():
    p = argparse.ArgumentParser(description='Compare MD time steps')
    p.add_argument('--params', type=str, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_particles', type=int, default=100)
    p.add_argument('--num_frames', type=int, default=100)
    p.add_argument('--output_dir', type=str, default='benchmarks/timestep_output')
    return p.parse_args()


def run_sim_custom_dt(thetas_and_energy, x0, num_steps, box_size, key, dt, kT=kT):
    """Flat-scan NVT run at a custom time step (no nested scan / remat)."""
    energy_fn, shift, shape, _ = energy_fn_from_design_vector(thetas_and_energy, box_size)
    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)
    step_fn = jit(step_fn)
    state = init_fn(key, x0, mass=shape.mass())
    do_step = jit(lambda st, _: (step_fn(st), (st.position.center, st.position.orientation)))
    state, (pos, ori) = lax.scan(do_step, state, jnp.arange(num_steps))
    return state, pos, ori


def main():
    configure_jax()
    args = parse_arguments()
    data = np.load(args.params, allow_pickle=True)
    yield_params = make_params(data['final_params'])
    box_size = get_box_size(DENSITY, args.num_particles, CENTER_RADIUS)

    os.makedirs(args.output_dir, exist_ok=True)
    init_body = random_IC(yield_params, random.PRNGKey(args.seed),
                          args.num_particles, box_size)

    for dt_value, label in [(DT, 'DT'), (DT / 2.0, 'DT_half'), (DT / 10.0, 'DT_tenth')]:
        print(f"\n=== {label}  dt={dt_value:.6f} ===")
        _, pos, ori = run_sim_custom_dt(
            yield_params, init_body, args.num_frames, box_size,
            random.PRNGKey(args.seed + 1000), dt_value,
        )
        out = os.path.join(args.output_dir, f"trajectory_{label}.npz")
        np.savez(out, dt_value=dt_value, dt_label=label,
                 num_particles=args.num_particles, num_frames=args.num_frames,
                 box_size=box_size, positions=np.array(pos), orientations=np.array(ori),
                 params=data['final_params'])
        print(f"  saved {out}   ts={datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
