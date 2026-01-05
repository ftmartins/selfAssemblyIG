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

    return {
        'shape': shape,
        'params': final_params,
        'param_file': param_file
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

def load_checkpoint(output_dir, seed):
    """
    Load checkpoint if it exists for the given seed.

    Parameters
    ----------
    output_dir : str
        Output directory path
    seed : int
        Random seed (KEY_PARAM_YIELD)

    Returns
    -------
    dict or None : Checkpoint data if exists, None otherwise
    """
    checkpoint_pattern = os.path.join(output_dir, f"checkpoint_seed{seed}_*.npz")
    import glob
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
        'final_state_center': checkpoint['final_state_center'],
        'final_state_orientation': checkpoint['final_state_orientation'],
        'params': checkpoint['params'],
        'num_particles': int(checkpoint['num_particles'])
    }

def save_checkpoint(output_dir, seed, steps_completed, final_state, params, num_particles):
    """
    Save simulation checkpoint.

    Parameters
    ----------
    output_dir : str
        Output directory path
    seed : int
        Random seed
    steps_completed : int
        Number of steps completed
    final_state : RigidBody state
        Current simulation state
    params : array
        Simulation parameters
    num_particles : int
        Number of particles
    """
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_file = os.path.join(
        output_dir,
        f"checkpoint_seed{seed}_step{steps_completed}.npz"
    )

    np.savez(
        checkpoint_file,
        seed=seed,
        steps_completed=steps_completed,
        final_state_center=np.array(final_state.position.center),
        final_state_orientation=np.array(final_state.position.orientation),
        params=params,
        num_particles=num_particles
    )

    print(f"Checkpoint saved: {checkpoint_file}")
    return checkpoint_file

def save_trajectory(output_dir, seed, positions, orientations, step_indices, params, append=False):
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
    """
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
    Run the yield simulation.

    Parameters
    ----------
    params_dict : dict
        Loaded parameter dictionary
    args : Namespace
        Command-line arguments

    Returns
    -------
    tuple : (final_state, polygon_counts, yields)
    """
    # Update global NUM_PARTICLES and BOX_SIZE for yield simulation
    import config_patchy_particle
    config_patchy_particle.NUM_PARTICLES = args.num_particles
    config_patchy_particle.BOX_SIZE = get_BOX_SIZE(DENSITY, args.num_particles, CENTER_RADIUS)

    box_size_yield = config_patchy_particle.BOX_SIZE

    print(f"\n{'='*80}")
    print(f"Yield Simulation")
    print(f"{'='*80}")
    print(f"Number of particles: {args.num_particles}")
    print(f"Simulation steps: {args.num_steps}")
    print(f"Box size: {box_size_yield:.2f}")
    print(f"{'='*80}\n")

    # Prepare parameters
    yield_params = make_params(params_dict['params'])

    # Check for existing checkpoint
    checkpoint = load_checkpoint(args.output_dir, KEY_PARAM_YIELD)

    if checkpoint:
        steps_completed = checkpoint['steps_completed']
        remaining_steps = args.num_steps - steps_completed

        if remaining_steps <= 0:
            print(f"Simulation already complete ({steps_completed} steps)")
            print("Loading existing trajectory...")
            trajectory_file = os.path.join(args.output_dir, f"trajectory_seed{KEY_PARAM_YIELD}.npz")
            if os.path.exists(trajectory_file):
                traj_data = np.load(trajectory_file, allow_pickle=True)
                positions_history = traj_data['positions']
                orientations_history = traj_data['orientations']

                # Reconstruct final state from checkpoint
                from jax_md import rigid_body
                final_positions = rigid_body.RigidPointUnion(
                    jnp.array(checkpoint['final_state_center']),
                    jnp.array(checkpoint['final_state_orientation'])
                )
                # Create a minimal state object
                class FinalState:
                    def __init__(self, position):
                        self.position = position
                final_state = FinalState(final_positions)
            else:
                raise FileNotFoundError("Checkpoint exists but trajectory file not found")
        else:
            print(f"Resuming from checkpoint: {steps_completed} steps completed")
            print(f"Running remaining {remaining_steps} steps...")

            # Reconstruct initial state from checkpoint
            from jax_md import rigid_body
            initial_body = rigid_body.RigidBody(
                jnp.array(checkpoint['final_state_center']),
                jnp.array(checkpoint['final_state_orientation'])
            )

            yield_key = random.PRNGKey(KEY_PARAM_YIELD + steps_completed)

            # Run remaining simulation
            final_state, positions_new, orientations_new = my_sim(
                yield_params,
                initial_body,
                remaining_steps,
                CENTER_RADIUS,
                yield_key,
                kT=kT,
            )

            # Subsample to every 10 steps and save trajectory (append mode)
            step_indices = np.arange(steps_completed, args.num_steps, 10)
            sample_indices = (step_indices - steps_completed) // 10
            positions_sampled = np.array(positions_new[::10])
            orientations_sampled = np.array(orientations_new[::10])

            save_trajectory(
                args.output_dir,
                KEY_PARAM_YIELD,
                positions_sampled,
                orientations_sampled,
                step_indices,
                yield_params,
                append=True
            )

            # Update checkpoint
            save_checkpoint(
                args.output_dir,
                KEY_PARAM_YIELD,
                args.num_steps,
                final_state,
                yield_params,
                args.num_particles
            )

            # Load full trajectory for downstream analysis
            trajectory_file = os.path.join(args.output_dir, f"trajectory_seed{KEY_PARAM_YIELD}.npz")
            traj_data = np.load(trajectory_file, allow_pickle=True)
            positions_history = traj_data['positions']
            orientations_history = traj_data['orientations']

            print("Simulation complete!")
    else:
        # No checkpoint - run full simulation
        print("Generating initial condition...", flush=True)
        yield_key = random.PRNGKey(KEY_PARAM_YIELD)
        initial_body = random_IC(yield_params, yield_key)

        # Run simulation
        print("Running simulation...")
        print("This may take several minutes...", flush=True)

        final_state, positions_history_full, orientations_history_full = my_sim(
            yield_params,
            initial_body,
            args.num_steps,
            CENTER_RADIUS,
            yield_key,
            kT=kT,
        )

        print("Simulation complete!")

        # Subsample to every 10 steps
        step_indices = np.arange(0, args.num_steps, 10)
        positions_history = np.array(positions_history_full[::10])
        orientations_history = np.array(orientations_history_full[::10])

        print(f"Trajectory shape: positions={positions_history.shape}, orientations={orientations_history.shape}")

        # Save trajectory
        save_trajectory(
            args.output_dir,
            KEY_PARAM_YIELD,
            positions_history,
            orientations_history,
            step_indices,
            yield_params,
            append=False
        )

        # Save checkpoint
        save_checkpoint(
            args.output_dir,
            KEY_PARAM_YIELD,
            args.num_steps,
            final_state,
            yield_params,
            args.num_particles
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
    output_file = os.path.join(
        args.output_dir,
        f"{shape_name}_yields_{timestamp}_{KEY_PARAM_YIELD}.npz"
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
    summary_file = os.path.join(
        args.output_dir,
        f"{shape_name}_yield_summary_{timestamp}_{KEY_PARAM_YIELD}.txt"
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
