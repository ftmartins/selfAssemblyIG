#!/usr/bin/env python3
"""
Run yield simulation and count polygon formation.

Usage:
    python run_yield_simulation.py --params path/to/params.npz

This script:
    1. Loads optimized parameters from NPZ file
    2. Runs large-scale simulation (default: 100 particles, 40k steps)
    3. Counts polygons using freud clustering
    4. Calculates yields as fraction of particles
    5. Saves results to NPZ file
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import jax.numpy as jnp
from jax import random
from scipy.spatial.distance import pdist, squareform
import freud

# Import configuration
from config_patchy_particle import *

# Import core modules
from modules.utility_functions import my_sim, make_params, random_IC

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run yield simulation from optimized parameters'
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
        default=KEY_PARAM_YIELD,
        help=f'Random seed for simulation (default: {KEY_PARAM_YIELD})'
    )
    parser.add_argument(
        '--num_particles',
        type=int,
        default=NUM_PARTICLES_YIELD,
        help=f'Number of particles (default: {NUM_PARTICLES_YIELD})'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=NUM_STEPS_YIELD,
        help=f'Number of simulation steps (default: {NUM_STEPS_YIELD})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=OUTPUT_DIR_YIELD,
        help=f'Output directory (default: {OUTPUT_DIR_YIELD})'
    )
    return parser.parse_args()

def load_parameters(param_file):
    """
    Load optimized parameters from NPZ file.

    Returns
    -------
    dict : Parameter data including shape, final_params, etc.
    """
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    data = np.load(param_file, allow_pickle=True)

    # Extract key information
    shape = str(data['shape'])
    final_params = data['final_params']

    print(f"\nLoaded parameters from: {param_file}")
    print(f"  Shape: {shape}")
    print(f"  Parameters: {final_params}")
    print(f"  Opening angle: {np.rad2deg(final_params[0]):.2f}°")

    # Extract param identifier from filename
    # e.g., 'optimal_params/triangle_params_20251208_121534.npz' -> 'triangle_params_20251208_121534'
    param_basename = os.path.basename(param_file)
    param_id = os.path.splitext(param_basename)[0]  # Remove .npz extension

    return {
        'shape': shape,
        'params': final_params,
        'param_file': param_file,
        'param_id': param_id
    }

def regular_ngon_reference(n, R=1.0, opening_angle_deg=100.0):
    """
    Construct a reference configuration for an n-gon of 2D patchy particles.
    (Extracted from notebook Cell 10)
    """
    if n < 3:
        raise ValueError("Need at least a triangle (n >= 3).")

    alpha = np.deg2rad(opening_angle_deg)

    # Particle positions on a regular n-gon
    vertex_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.stack(
        [R * np.cos(vertex_angles), R * np.sin(vertex_angles)],
        axis=1
    )

    orientations = np.zeros(n)
    patch_angles = np.zeros((n, 2))

    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n

        v_prev = positions[i_prev] - positions[i]
        v_next = positions[i_next] - positions[i]

        beta_prev = np.arctan2(v_prev[1], v_prev[0])
        beta_next = np.arctan2(v_next[1], v_next[0])

        avg_vec = np.array([
            np.cos(beta_prev) + np.cos(beta_next),
            np.sin(beta_prev) + np.sin(beta_next),
        ])
        avg_beta = np.arctan2(avg_vec[1], avg_vec[0])

        theta_i = avg_beta
        orientations[i] = theta_i

        patch_to_prev = theta_i - alpha / 2.0
        patch_to_next = theta_i + alpha / 2.0

        patch_to_prev = (patch_to_prev + np.pi) % (2.0 * np.pi) - np.pi
        patch_to_next = (patch_to_next + np.pi) % (2.0 * np.pi) - np.pi

        patch_angles[i, 0] = patch_to_prev
        patch_angles[i, 1] = patch_to_next

    return positions, orientations, patch_angles

def load_checkpoint(output_dir, seed, param_id=None):
    """
    Load full MD checkpoint if it exists for the given seed.

    Parameters
    ----------
    output_dir : str
        Output directory path
    seed : int
        Random seed (KEY_PARAM_YIELD)
    param_id : str, optional
        Parameter file identifier to match specific checkpoint

    Returns
    -------
    dict or None : Checkpoint data with full MD state if exists, None otherwise
    """
    import glob

    # Try new directory structure first: output_dir/{param_id}/seed{seed}/checkpoint_step*.npz
    if param_id:
        checkpoint_dir = os.path.join(output_dir, param_id, f"seed{seed}")
        checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_step*.npz")
        checkpoint_files = glob.glob(checkpoint_pattern)

        # Fall back to old flat naming if no match
        if not checkpoint_files:
            checkpoint_pattern = os.path.join(output_dir, f"checkpoint_{param_id}_seed{seed}_*.npz")
            checkpoint_files = glob.glob(checkpoint_pattern)

        # Fall back to even older naming
        if not checkpoint_files:
            checkpoint_pattern = os.path.join(output_dir, f"checkpoint_seed{seed}_*.npz")
            checkpoint_files = glob.glob(checkpoint_pattern)
    else:
        checkpoint_pattern = os.path.join(output_dir, f"checkpoint_seed{seed}_*.npz")
        checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        return None

    # Get the most recent checkpoint
    checkpoint_file = max(checkpoint_files, key=os.path.getmtime)

    print(f"Found checkpoint: {checkpoint_file}")
    checkpoint = np.load(checkpoint_file, allow_pickle=True)

    return {
        'file': checkpoint_file,
        'seed': int(checkpoint['seed']),
        'steps_completed': int(checkpoint['steps_completed']),
        'full_md_state': checkpoint['full_md_state'].item(),  # Extract pickled state
        'params': checkpoint['params'],
        'num_particles': int(checkpoint['num_particles'])
    }

def save_checkpoint(output_dir, seed, steps_completed, full_md_state, params, num_particles, param_id=None):
    """
    Save full MD simulation checkpoint including velocities and thermostat state.

    Parameters
    ----------
    output_dir : str
        Output directory path
    seed : int
        Random seed
    steps_completed : int
        Number of steps completed
    full_md_state : JAX-MD NVT state
        Complete simulation state (positions, velocities, thermostat variables)
    params : array
        Simulation parameters
    num_particles : int
        Number of particles
    param_id : str, optional
        Parameter file identifier (e.g., 'triangle_params_20251208_121534')
    """
    # Create directory structure: output_dir/{param_id}/seed{seed}/
    if param_id:
        checkpoint_dir = os.path.join(output_dir, param_id, f"seed{seed}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(
            checkpoint_dir,
            f"checkpoint_step{steps_completed}.npz"
        )
    else:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_file = os.path.join(
            output_dir,
            f"checkpoint_seed{seed}_step{steps_completed}.npz"
        )

    # Save full state using pickle
    np.savez(
        checkpoint_file,
        seed=seed,
        steps_completed=steps_completed,
        full_md_state=full_md_state,  # JAX pytree - requires allow_pickle=True
        params=params,
        num_particles=num_particles,
        allow_pickle=True
    )

    print(f"Checkpoint saved: {checkpoint_file} ({steps_completed} steps)")
    return checkpoint_file

def save_trajectory(output_dir, seed, positions, orientations, step_indices, params, append=False, param_id=None):
    """
    Save or append trajectory data.

    Parameters
    ----------
    output_dir : str
        Output directory path
    seed : int
        Random seed
    positions : array
        Position history (num_frames, num_particles, 2)
    orientations : array
        Orientation history (num_frames, num_particles)
    step_indices : array
        Step indices for each frame
    params : array
        Simulation parameters
    append : bool
        If True, append to existing trajectory file
    param_id : str, optional
        Parameter file identifier (e.g., 'triangle_params_20251208_121534')
    """
    # Create directory structure: output_dir/{param_id}/seed{seed}/
    if param_id:
        trajectory_dir = os.path.join(output_dir, param_id, f"seed{seed}")
        os.makedirs(trajectory_dir, exist_ok=True)
        trajectory_file = os.path.join(trajectory_dir, f"trajectory.npz")
    else:
        os.makedirs(output_dir, exist_ok=True)
        trajectory_file = os.path.join(output_dir, f"trajectory_seed{seed}.npz")

    if append and os.path.exists(trajectory_file):
        # Load existing trajectory and append
        existing = np.load(trajectory_file, allow_pickle=True)
        positions = np.concatenate([existing['positions'], positions], axis=0)
        orientations = np.concatenate([existing['orientations'], orientations], axis=0)
        step_indices = np.concatenate([existing['step_indices'], step_indices], axis=0)

    np.savez(
        trajectory_file,
        seed=seed,
        positions=positions,
        orientations=orientations,
        step_indices=step_indices,
        params=params
    )

    print(f"Trajectory saved: {trajectory_file} ({len(step_indices)} frames)")
    return trajectory_file

def count_polygons(state, params, box_size, patch_allowance=None, cluster_check=0.5):
    """
    Count polygons in the system using freud clustering.
    (Extracted and adapted from notebook Cell 11)

    Parameters
    ----------
    state : RigidBody.position
        Current state of the system
    params : array
        Full parameter array [theta0, theta1, energies...]
    box_size : float
        Simulation box size
    patch_allowance : float, optional
        Tolerance for patch distance matching
    cluster_check : float
        Loss threshold for cluster validation

    Returns
    -------
    dict : Counts for each polygon type
    """
    if patch_allowance is None:
        patch_allowance = PATCH_SIZE * 2 / 3.

    # Helper function to get reference shape
    def get_shape_local(shape_id):
        """Get reference shape for polygon validation."""
        shape_id = shape_id.lower()

        if shape_id == 'triangle':
            radius = 1.0 / np.sin(np.pi/3)
            return regular_ngon_reference(3, R=radius, opening_angle_deg=params[1])[0]
        elif shape_id == 'square':
            radius = 1.0 / np.sin(np.pi/4)
            return regular_ngon_reference(4, R=radius, opening_angle_deg=params[1])[0]
        elif shape_id == 'pentagon':
            radius = 1.0 / np.sin(np.pi/5)
            return regular_ngon_reference(5, R=radius, opening_angle_deg=params[1])[0]
        elif shape_id == 'hexagon':
            radius = 1.0 / np.sin(np.pi/6)
            return regular_ngon_reference(6, R=radius, opening_angle_deg=params[1])[0]
        elif shape_id == 'heptagon':
            radius = 1.0 / np.sin(np.pi/7)
            return regular_ngon_reference(7, R=radius, opening_angle_deg=params[1])[0]
        elif shape_id == 'octagon':
            radius = 1.0 / np.sin(np.pi/8)
            return regular_ngon_reference(8, R=radius, opening_angle_deg=params[1])[0]        
        else:
            raise ValueError("Invalid shape_id")
    # Get particle centers and orientations
    center_pos = np.array(state.center)
    orientations = np.array(state.orientation)

    # Compute patch positions
    opening_angle = params[1]
    patch_angles_local = np.array([-opening_angle/2, opening_angle/2])

    patches_pos = [[], []]
    for i in range(len(center_pos)):
        particle_orientation = orientations[i]
        for k, local_angle in enumerate(patch_angles_local):
            world_angle = particle_orientation + local_angle
            patch_x = center_pos[i, 0] + CENTER_RADIUS * np.cos(world_angle)
            patch_y = center_pos[i, 1] + CENTER_RADIUS * np.sin(world_angle)
            patches_pos[k].append([patch_x, patch_y])

    patches_pos = [np.array(p) for p in patches_pos]

    # Calculate patch distance
    def patch_dist_func(angle_rad, radius=CENTER_RADIUS):
        return 2 * radius * np.sin(angle_rad / 2.)

    patch_dists = abs(patch_dist_func(params[1]))

    # Box setup
    box = freud.box.Box(box_size, box_size, is2D=True)

    # Define polygon types
    polygon_types = [
        (3, 'triangle'),
        (4, 'square'),
        (5, 'pentagon'),
        (6, 'hexagon'),
        (7, 'heptagon'),
        (8, 'octagon')
    ]

    polygon_counts = {
        'triangle': 0,
        'square': 0,
        'pentagon': 0,
        'hexagon': 0,
        'heptagon': 0,
        'octagon': 0,
        'monomers': 0,
        'other': 0
    }

    all_found_clusters = set()

    # Search for each polygon type
    for ref_cl_size, polygon_name in polygon_types:
        max_dist = patch_dists + patch_allowance
        min_dist = 0.001

        # Get cluster information for each patch type
        patch_clusters = []
        for k in range(NUM_PATCHES):
            points_2d = np.hstack((patches_pos[k], np.zeros((len(patches_pos[k]), 1))))
            cl = freud.cluster.Cluster()
            cl.compute((box, points_2d), neighbors={'r_max': max_dist, 'r_min': min_dist})

            cl_props = freud.cluster.ClusterProperties()
            cl_props.compute((box, points_2d), cl.cluster_idx)

            cluster_keys = np.array(cl.cluster_keys, dtype=object)
            ref_clusters = tuple(c for c in cluster_keys if len(c) == ref_cl_size)
            patch_clusters.append(ref_clusters)

        # Find clusters in both patch types
        patch_A, patch_B = patch_clusters[0], patch_clusters[1]
        set_a = set(map(tuple, patch_A))
        set_b = set(map(tuple, patch_B))

        matches = set_a.intersection(set_b)
        only_in_a = set_a.difference(set_b)
        only_in_b = set_b.difference(set_a)

        # Validate clusters
        def validate_cluster(cluster, polygon_name):
            cluster_indices = np.array(list(cluster), dtype=int)
            if len(cluster_indices) != ref_cl_size:
                return False

            try:
                cluster_centers = center_pos[cluster_indices]
                ref_shape = np.array(get_shape_local(polygon_name))

                if len(cluster_centers) != len(ref_shape):
                    return False

                ref_dists_pairwise = squareform(pdist(ref_shape))
                ref_dists_sorted = np.sort(ref_dists_pairwise, axis=1)

                cluster_dists_pairwise = squareform(pdist(cluster_centers))
                cluster_dists_sorted = np.sort(cluster_dists_pairwise, axis=1)

                diff = np.abs(cluster_dists_sorted[:, 1:] - ref_dists_sorted[:, 1:])
                loss = np.mean(diff)

                is_valid = not np.isnan(loss) and not np.isinf(loss) and loss < cluster_check
                return is_valid

            except Exception as e:
                return False

        # Collect all potential clusters
        all_potential_clusters = set()
        all_potential_clusters.update(matches)
        all_potential_clusters.update(only_in_a)
        all_potential_clusters.update(only_in_b)

        # Validate all
        for cluster in all_potential_clusters:
            cluster_tuple = tuple(sorted(cluster))
            if cluster_tuple not in all_found_clusters:
                if validate_cluster(cluster, polygon_name):
                    polygon_counts[polygon_name] += 1
                    all_found_clusters.add(cluster_tuple)

    # Count monomers
    all_clustered_particles = set()
    for cluster in all_found_clusters:
        all_clustered_particles.update(cluster)

    total_particles = len(center_pos)
    polygon_counts['monomers'] = total_particles - len(all_clustered_particles)

    return polygon_counts

def run_yield_simulation(params_dict, args):
    """
    Run the yield simulation with checkpointing every 100 steps.

    Parameters
    ----------
    params_dict : dict
        Loaded parameter dictionary
    args : Namespace
        Command-line arguments (includes seed)

    Returns
    -------
    tuple : (final_state, polygon_counts, yields)
    """
    # Constants
    CHECKPOINT_INTERVAL = 1000  # Save checkpoint every 1000 steps

    # Update global NUM_PARTICLES and BOX_SIZE for yield simulation
    import config_patchy_particle
    config_patchy_particle.NUM_PARTICLES = args.num_particles
    config_patchy_particle.BOX_SIZE = get_BOX_SIZE(DENSITY, args.num_particles, CENTER_RADIUS)

    box_size_yield = config_patchy_particle.BOX_SIZE

    print(f"\n{'='*80}")
    print(f"Yield Simulation with Checkpointing")
    print(f"{'='*80}")
    print(f"Random seed: {args.seed}")
    print(f"Number of particles: {args.num_particles}")
    print(f"Simulation steps: {args.num_steps}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} steps")
    print(f"Box size: {box_size_yield:.2f}")
    print(f"{'='*80}\n")

    # Prepare parameters
    yield_params = make_params(params_dict['params'])

    # Check for existing checkpoint
    checkpoint = load_checkpoint(args.output_dir, args.seed, param_id=params_dict.get('param_id'))

    # Determine starting point
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint['steps_completed']} steps completed")
        full_md_state_checkpoint = checkpoint['full_md_state']
        current_state = full_md_state_checkpoint.position  # Extract position for next my_sim call
        start_step = checkpoint['steps_completed']
    else:
        print("Starting new simulation...")
        yield_key = random.PRNGKey(args.seed)
        initial_body = random_IC(yield_params, yield_key)
        current_state = initial_body
        start_step = 0

    # Check if already complete
    if start_step >= args.num_steps:
        print(f"Simulation already complete ({start_step}/{args.num_steps} steps)")
        print("Loading existing trajectory...")

        # Try new directory structure first, fall back to old
        param_id = params_dict.get('param_id')
        if param_id:
            # New structure: output_dir/{param_id}/seed{seed}/trajectory.npz
            trajectory_file = os.path.join(args.output_dir, param_id, f"seed{args.seed}", "trajectory.npz")
            if not os.path.exists(trajectory_file):
                # Old flat naming
                trajectory_file = os.path.join(args.output_dir, f"trajectory_{param_id}_seed{args.seed}.npz")
            if not os.path.exists(trajectory_file):
                trajectory_file = os.path.join(args.output_dir, f"trajectory_seed{args.seed}.npz")
        else:
            trajectory_file = os.path.join(args.output_dir, f"trajectory_seed{args.seed}.npz")

        if os.path.exists(trajectory_file):
            traj_data = np.load(trajectory_file, allow_pickle=True)
            # Use checkpoint state for final_state
            final_state = full_md_state_checkpoint
        else:
            raise FileNotFoundError("Checkpoint exists but trajectory file not found")
    else:
        # Run simulation in chunks with checkpointing
        print(f"Running simulation from step {start_step} to {args.num_steps}...")
        print("This may take several minutes...", flush=True)

        positions_chunks = []
        orientations_chunks = []

        for chunk_start in range(start_step, args.num_steps, CHECKPOINT_INTERVAL):
            chunk_steps = min(CHECKPOINT_INTERVAL, args.num_steps - chunk_start)

            print(f"  Running steps {chunk_start} to {chunk_start + chunk_steps}...", flush=True)

            # Generate key for this chunk
            chunk_key = random.PRNGKey(args.seed + chunk_start)

            # Run this chunk
            full_md_state, pos_chunk, ori_chunk = my_sim(
                yield_params,
                current_state,
                chunk_steps,
                CENTER_RADIUS,
                chunk_key,
                kT=kT,
            )

            # Store positions and orientations for later subsampling
            positions_chunks.append(np.array(pos_chunk))
            orientations_chunks.append(np.array(ori_chunk))

            # Save checkpoint with full MD state
            save_checkpoint(
                args.output_dir,
                args.seed,
                chunk_start + chunk_steps,
                full_md_state,
                yield_params,
                args.num_particles,
                param_id=params_dict.get('param_id')
            )

            # Update current_state with the position for next iteration
            current_state = full_md_state.position

        print("Simulation complete!")

        # Store final state (the last full MD state from the loop)
        final_state = full_md_state

        # Concatenate all chunks
        positions_full = np.concatenate(positions_chunks, axis=0)
        orientations_full = np.concatenate(orientations_chunks, axis=0)

        # Subsample to every 10 steps (POST-SIMULATION)
        step_indices = np.arange(0, len(positions_full), 10)
        positions_history = positions_full[::10]
        orientations_history = orientations_full[::10]

        print(f"Trajectory shape: positions={positions_history.shape}, orientations={orientations_history.shape}")

        # Save trajectory (AFTER simulation completes)
        save_trajectory(
            args.output_dir,
            args.seed,
            positions_history,
            orientations_history,
            step_indices,
            yield_params,
            append=False,
            param_id=params_dict.get('param_id')
        )

    # Count polygons
    print("\nCounting polygons...")
    polygon_counts = count_polygons(final_state.position, yield_params, box_size_yield)

    # Calculate yields
    yields = {}
    total_particles = args.num_particles

    vertices_per_type = {
        'triangle': 3,
        'square': 4,
        'pentagon': 5,
        'hexagon': 6,
        'heptagon': 7,
        'octagon': 8,
        'monomers': 1,
        'other': 0
    }

    for poly_type, count in polygon_counts.items():
        n_vertices = vertices_per_type.get(poly_type, 0)
        if poly_type == 'monomers':
            particles_in_type = count
        else:
            particles_in_type = count * n_vertices

        yields[poly_type] = particles_in_type / total_particles

    # Print results
    print(f"\n{'='*80}")
    print(f"Polygon Yield Results")
    print(f"{'='*80}")
    print(f"\nPolygon counts:")
    for poly_type, count in polygon_counts.items():
        print(f"  {poly_type.capitalize():10s}: {count}")

    print(f"\nYields (as fraction of particles):")
    for poly_type, yield_val in yields.items():
        print(f"  {poly_type.capitalize():10s}: {yield_val:.4f} ({yield_val*100:.2f}%)")
    print(f"{'='*80}\n")

    return final_state, polygon_counts, yields

def save_yield_results(params_dict, args, final_state, polygon_counts,
                       yields, timestamp):
    """Save yield results to NPZ file and text summary."""
    os.makedirs(args.output_dir, exist_ok=True)

    shape_name = params_dict['shape']
    param_id = params_dict.get('param_id', '')

    # Create directory structure: output_dir/{param_id}/
    if param_id:
        results_dir = os.path.join(args.output_dir, param_id)
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(
            results_dir,
            f"yields_{timestamp}_seed{args.seed}.npz"
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(
            args.output_dir,
            f"{shape_name}_yields_{timestamp}_{args.seed}.npz"
        )

    box_size_yield = get_BOX_SIZE(DENSITY, args.num_particles, CENTER_RADIUS)

    np.savez(
        output_file,
        shape=shape_name,
        yields=yields,
        polygon_counts=polygon_counts,
        num_particles=args.num_particles,
        num_steps=args.num_steps,
        box_size=box_size_yield,
        optimized_params=params_dict['params'],
        param_source_file=params_dict['param_file'],
        timestamp=timestamp,
        final_centers=np.array(final_state.position.center),
        final_orientations=np.array(final_state.position.orientation),
    )

    # Text summary
    if param_id:
        summary_file = os.path.join(
            results_dir,
            f"yield_summary_{timestamp}_seed{args.seed}.txt"
        )
    else:
        summary_file = os.path.join(
            args.output_dir,
            f"{shape_name}_yield_summary_{timestamp}_{args.seed}.txt"
        )

    with open(summary_file, 'w') as f:
        f.write(f"Patchy Particle Yield Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Shape: {shape_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Parameter source: {params_dict['param_file']}\n\n")
        f.write(f"Simulation Details:\n")
        f.write(f"  Number of particles: {args.num_particles}\n")
        f.write(f"  Simulation steps: {args.num_steps}\n")
        f.write(f"  Box size: {box_size_yield:.2f}\n")
        f.write(f"  Density: {DENSITY}\n\n")
        f.write(f"Optimized Parameters:\n")
        f.write(f"  {params_dict['params']}\n")
        f.write(f"  Opening angle: {np.rad2deg(params_dict['params'][0]):.2f}°\n\n")
        f.write(f"Polygon Counts:\n")
        for poly_type, count in polygon_counts.items():
            f.write(f"  {poly_type.capitalize():10s}: {count}\n")
        f.write(f"\nYields (% of particles):\n")
        for poly_type, yield_val in yields.items():
            f.write(f"  {poly_type.capitalize():10s}: {yield_val*100:.2f}%\n")

    print(f"Results saved:")
    print(f"  NPZ file: {output_file}")
    print(f"  Summary: {summary_file}")

    return output_file

def main():
    """Main yield simulation workflow."""
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*80}")
    print(f"Patchy Particle Yield Simulation")
    print(f"{'='*80}\n")

    # Load parameters
    params_dict = load_parameters(args.params)

    print(args.num_particles, 'num particles args')

    # Run simulation
    final_state, polygon_counts, yields = run_yield_simulation(params_dict, args)
    
    # Save results
    output_file = save_yield_results(
        params_dict, args, final_state, polygon_counts, yields, timestamp
    )

    print(f"\n{'='*80}")
    print(f"Yield simulation complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
