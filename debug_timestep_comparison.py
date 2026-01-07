#!/usr/bin/env python3
"""
Debug script to compare timestep effects on simulations.

Runs 3 short simulations with different timesteps:
- DT (current timestep from config)
- DT/2 (half timestep)
- DT/10 (tenth timestep)

Each simulation:
- 100 particles
- 100 frames saved
- Separate output files for each timestep

Usage:
    python debug_timestep_comparison.py --params path/to/params.npz [--seed SEED]
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax
from jax_md import space, rigid_body, simulate, energy

# Import configuration
from config_patchy_particle import *

# Import core modules
from modules.utility_functions import make_params, random_IC

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Debug script comparing different timesteps'
    )
    parser.add_argument(
        '--params',
        type=str,
        required=True,
        help='Path to optimized parameter NPZ file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for simulation (default: 42)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='debug_timestep_output',
        help='Output directory (default: debug_timestep_output)'
    )
    return parser.parse_args()

def load_parameters(param_file):
    """Load optimized parameters from NPZ file."""
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    data = np.load(param_file, allow_pickle=True)
    shape = str(data['shape'])
    final_params = data['final_params']

    print(f"\nLoaded parameters from: {param_file}")
    print(f"  Shape: {shape}")
    print(f"  Parameters: {final_params}")
    print(f"  Opening angle: {np.rad2deg(final_params[0]):.2f}°")

    return {
        'shape': shape,
        'params': final_params,
        'param_file': param_file
    }

def energy_matrix(eng):
    """Create energy matrix from parameters."""
    eng_mat = jnp.zeros((NUM_PATCHES, NUM_PATCHES))
    eng_mat = eng_mat.at[0, 1].set(eng[0])
    eng_mat = eng_mat.at[1, 0].set(eng[0])
    return eng_mat

def thetas_to_shape(thetas, radius=CENTER_RADIUS):
    """Convert patch angles to rigid body shape."""
    positions = []
    positions += [jnp.array([0.0, 0.0])]
    for theta in thetas:
        positions += [jnp.array([jnp.cos(theta), jnp.sin(theta)]) * radius]
    positions = jnp.array(positions)

    masses = []
    masses += [CENTER_MASS]
    for _ in thetas:
        masses += [PATCH_MASS]
    masses = jnp.array(masses)

    return rigid_body.RigidBody(positions, masses)

def run_sim_custom_dt(thetas_and_energy,
                      x0,
                      num_steps,
                      CENTER_RADIUS,
                      key,
                      dt,
                      kT=1.0):
    """
    Run simulation with custom timestep.

    Parameters
    ----------
    thetas_and_energy : array
        Particle parameters
    x0 : RigidBody
        Initial state
    num_steps : int
        Number of simulation steps
    CENTER_RADIUS : float
        Particle center radius
    key : PRNGKey
        Random key
    dt : float
        Custom timestep
    kT : float
        Temperature

    Returns
    -------
    tuple : (final_state, positions, orientations)
    """
    # Access BOX_SIZE at runtime
    import config_patchy_particle
    box_size = config_patchy_particle.BOX_SIZE

    thetas_and_energy = thetas_and_energy.at[0].set(0.0)
    thetas = thetas_and_energy[:NUM_PATCHES]
    eng = thetas_and_energy[NUM_PATCHES:]
    eng_mat = energy_matrix(eng)
    shape = thetas_to_shape(thetas, radius=CENTER_RADIUS)
    displacement, shift = space.periodic(box_size)

    morse_eps = jnp.pad(eng_mat, pad_width=(1, 0))
    soft_sphere_eps = jnp.zeros((len(thetas) + 1, len(thetas) + 1))
    soft_sphere_eps = soft_sphere_eps.at[0, 0].set(1.0)
    soft_sphere_eps = 10000 * soft_sphere_eps

    pair_energy_soft = energy.soft_sphere_pair(
        displacement,
        species=1+len(thetas),
        sigma=CENTER_RADIUS*2,
        epsilon=soft_sphere_eps
    )
    pair_energy_morse = energy.morse_pair(
        displacement,
        species=1+len(thetas),
        sigma=0.0,
        epsilon=morse_eps,
        alpha=ALPHA,
        r_cutoff=R_CUTOFF
    )
    pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) + pair_energy_morse(R, **kwargs)
    energy_fn = rigid_body.point_energy(pair_energy_fn, shape)

    # Use custom dt instead of global DT
    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)
    step_fn = jit(step_fn)
    state = init_fn(key, x0, mass=shape.mass())

    do_step = lambda state, t: (step_fn(state), (state.position.center, state.position.orientation))
    do_step = jit(do_step)

    # Run simulation and collect trajectory
    inner_steps = jnp.arange(num_steps)
    state, (positions, orientations) = lax.scan(do_step, state, inner_steps)

    return state, positions, orientations

def run_debug_simulation(params_dict, args, dt_value, dt_label, num_frames=100):
    """
    Run a single debug simulation with specified timestep.

    Parameters
    ----------
    params_dict : dict
        Loaded parameter dictionary
    args : Namespace
        Command-line arguments
    dt_value : float
        Timestep value to use
    dt_label : str
        Label for this timestep (e.g., "DT", "DT_half", "DT_tenth")
    num_frames : int
        Number of frames to save (default: 100)

    Returns
    -------
    dict : Results including trajectory data
    """
    # Configuration
    num_particles = 100

    # Update global NUM_PARTICLES and BOX_SIZE
    import config_patchy_particle
    config_patchy_particle.NUM_PARTICLES = num_particles

    # Calculate box size based on density
    box_size = np.sqrt(num_particles * np.pi * CENTER_RADIUS**2 / DENSITY)
    config_patchy_particle.BOX_SIZE = box_size

    print(f"\n{'='*80}")
    print(f"Debug Simulation: {dt_label}")
    print(f"{'='*80}")
    print(f"Timestep (dt): {dt_value:.6f}")
    print(f"Number of particles: {num_particles}")
    print(f"Number of frames: {num_frames}")
    print(f"Box size: {box_size:.2f}")
    print(f"Density: {DENSITY:.4f}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")

    # Prepare parameters
    yield_params = make_params(params_dict['params'])

    # Initialize
    yield_key = random.PRNGKey(args.seed)
    initial_body = random_IC(yield_params, yield_key)

    # Run simulation
    print(f"Running {num_frames} steps...")
    sim_key = random.PRNGKey(args.seed + 1000)
    final_state, positions, orientations = run_sim_custom_dt(
        yield_params,
        initial_body,
        num_frames,
        CENTER_RADIUS,
        sim_key,
        dt=dt_value,
        kT=kT
    )

    print(f"Simulation complete!")
    print(f"Trajectory shape: positions={positions.shape}, orientations={orientations.shape}")

    return {
        'dt_value': dt_value,
        'dt_label': dt_label,
        'num_particles': num_particles,
        'num_frames': num_frames,
        'box_size': box_size,
        'positions': np.array(positions),
        'orientations': np.array(orientations),
        'final_state': final_state,
        'params': params_dict['params']
    }

def save_trajectory(results, output_dir):
    """Save trajectory to NPZ file."""
    os.makedirs(output_dir, exist_ok=True)

    dt_label = results['dt_label']
    filename = os.path.join(output_dir, f"trajectory_{dt_label}.npz")

    np.savez(
        filename,
        dt_value=results['dt_value'],
        dt_label=results['dt_label'],
        num_particles=results['num_particles'],
        num_frames=results['num_frames'],
        box_size=results['box_size'],
        positions=results['positions'],
        orientations=results['orientations'],
        params=results['params']
    )

    print(f"Saved trajectory to: {filename}")
    return filename

def main():
    """Main execution function."""
    args = parse_arguments()

    # Load parameters
    params_dict = load_parameters(args.params)

    # Get base timestep from config
    dt_base = DT

    print(f"\n{'='*80}")
    print(f"TIMESTEP COMPARISON DEBUG SCRIPT")
    print(f"{'='*80}")
    print(f"Base timestep (DT): {dt_base:.6f}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")

    # Define timesteps to test
    timesteps = [
        (dt_base, "DT"),
        (dt_base / 2.0, "DT_half"),
        (dt_base / 10.0, "DT_tenth")
    ]

    results_all = []

    # Run simulations for each timestep
    for dt_value, dt_label in timesteps:
        results = run_debug_simulation(
            params_dict,
            args,
            dt_value,
            dt_label,
            num_frames=100
        )
        results_all.append(results)

        # Save trajectory
        save_trajectory(results, args.output_dir)

    # Create summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    for results in results_all:
        print(f"{results['dt_label']:12s} dt={results['dt_value']:.6f}  "
              f"frames={results['num_frames']}  "
              f"particles={results['num_particles']}")
    print(f"\nAll trajectories saved to: {args.output_dir}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
