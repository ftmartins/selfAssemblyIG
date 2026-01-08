#!/usr/bin/env python3
"""
Generate and cache initial conditions for patchy particle simulations.

Usage:
    python generate_initial_condition.py --num_particles 1000 --seed 42 [--density 0.1]

This script:
    1. Computes box_size and min_distance from particle parameters
    2. Generates non-overlapping initial configuration using RSA
    3. Saves to NPZ file with complete metadata for validation
    4. Outputs: initial_conditions/ic_N{num_particles}_rho{density}_seed{seed}.npz
"""

import argparse
import os
import sys
import numpy as np
import jax.numpy as jnp
from jax import random
from datetime import datetime

# Import configuration
from config_patchy_particle import (
    CENTER_RADIUS,
    ALPHA,
    PATCH_SIZE,
    NUM_PATCHES,
    get_BOX_SIZE
)

# Import IC generation function
from modules.utility_functions import random_IC_nonoverlap


def parse_arguments():
    """Parse command-line arguments for IC generation."""
    parser = argparse.ArgumentParser(
        description='Generate and cache initial conditions for patchy particle simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate IC for 1000 particles with seed 42 at default density
  python generate_initial_condition.py --num_particles 1000 --seed 42

  # Generate IC with custom density
  python generate_initial_condition.py --num_particles 500 --seed 123 --density 0.15

  # Specify custom output directory
  python generate_initial_condition.py --num_particles 100 --seed 1 --output_dir my_ics
        """
    )

    parser.add_argument(
        '--num_particles',
        type=int,
        required=True,
        help='Number of particles in the system'
    )

    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--density',
        type=float,
        default=0.1,
        help='Area fraction / density (default: 0.1)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='initial_conditions',
        help='Output directory for IC files (default: initial_conditions)'
    )

    parser.add_argument(
        '--min_distance',
        type=float,
        default=None,
        help='Minimum center-to-center distance (default: 2*CENTER_RADIUS + 2*PATCH_SIZE)'
    )

    parser.add_argument(
        '--max_attempts',
        type=int,
        default=10000,
        help='Maximum RSA attempts per particle (default: 10000)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing IC file if it exists'
    )

    return parser.parse_args()


def compute_ic_parameters(num_particles, density):
    """
    Compute derived parameters for IC generation.

    Parameters
    ----------
    num_particles : int
        Number of particles
    density : float
        Area fraction / density

    Returns
    -------
    dict : Dictionary containing all computed parameters
    """
    # Compute box size using the same formula as main code
    box_size = get_BOX_SIZE(density, num_particles, CENTER_RADIUS)

    # Compute default minimum distance
    min_distance = 2 * CENTER_RADIUS + 2 * PATCH_SIZE

    # Additional derived quantities for validation
    actual_density = (np.pi * CENTER_RADIUS**2 * num_particles) / (box_size**2)

    return {
        'num_particles': num_particles,
        'density': density,
        'box_size': box_size,
        'min_distance': min_distance,
        'center_radius': CENTER_RADIUS,
        'patch_size': PATCH_SIZE,
        'alpha': ALPHA,
        'actual_density': actual_density,
    }


def validate_ic(positions, box_size, min_distance):
    """
    Validate generated IC for overlaps and box bounds.

    Parameters
    ----------
    positions : array
        Particle positions (N, 2)
    box_size : float
        Simulation box size
    min_distance : float
        Minimum allowed distance

    Raises
    ------
    ValueError : If validation fails
    """
    num_particles = len(positions)

    # Check box bounds
    if np.any(positions < 0) or np.any(positions >= box_size):
        raise ValueError("Particles outside box bounds")

    # Check minimum distances with periodic boundary conditions
    violations = 0
    min_found = float('inf')

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            # Minimum image convention for periodic boundaries
            dr = positions[i] - positions[j]
            dr = dr - box_size * np.round(dr / box_size)
            dist = np.sqrt(np.sum(dr**2))

            min_found = min(min_found, dist)

            if dist < min_distance - 1e-6:  # Small tolerance for numerical errors
                violations += 1

    print(f"  Validation results:")
    print(f"    Minimum distance found: {min_found:.6f}")
    print(f"    Required min distance:  {min_distance:.6f}")
    print(f"    Overlap violations:     {violations}")

    if violations > 0:
        raise ValueError(f"IC validation failed: {violations} overlapping particle pairs")

    print(f"    Status: PASSED")


def generate_ic(args):
    """
    Generate initial condition and save to file.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments

    Returns
    -------
    str : Path to saved IC file
    """
    # Compute parameters
    params = compute_ic_parameters(args.num_particles, args.density)

    # Override min_distance if provided
    if args.min_distance is not None:
        params['min_distance'] = args.min_distance

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename
    output_filename = f"ic_N{args.num_particles}_rho{args.density:.4f}_seed{args.seed}.npz"
    output_path = os.path.join(args.output_dir, output_filename)

    # Check if file exists
    if os.path.exists(output_path) and not args.force:
        print(f"ERROR: IC file already exists: {output_path}")
        print("Use --force to overwrite")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Generating Initial Condition")
    print(f"{'='*80}")
    print(f"Parameters:")
    print(f"  Number of particles: {params['num_particles']}")
    print(f"  Density (target):    {params['density']:.4f}")
    print(f"  Density (actual):    {params['actual_density']:.4f}")
    print(f"  Box size:            {params['box_size']:.4f}")
    print(f"  Min distance:        {params['min_distance']:.4f}")
    print(f"  Center radius:       {params['center_radius']:.4f}")
    print(f"  Patch size:          {params['patch_size']:.4f}")
    print(f"  Random seed:         {args.seed}")
    print(f"{'='*80}\n")

    # Create dummy thetas_and_energy array (only used for shape, not for IC generation)
    # IC generation doesn't actually use the patch angles or energies
    thetas_and_energy = jnp.zeros(NUM_PATCHES + NUM_PATCHES * (NUM_PATCHES - 1) // 2)

    # Generate PRNG key
    rng_key = random.PRNGKey(args.seed)

    # Update global NUM_PARTICLES and BOX_SIZE for IC generation
    import config_patchy_particle
    original_num_particles = config_patchy_particle.NUM_PARTICLES
    original_box_size = config_patchy_particle.BOX_SIZE

    try:
        config_patchy_particle.NUM_PARTICLES = args.num_particles
        config_patchy_particle.BOX_SIZE = params['box_size']

        # Generate IC
        print("Generating non-overlapping initial condition...")
        start_time = datetime.now()

        body = random_IC_nonoverlap(
            thetas_and_energy,
            rng_key,
            min_distance=params['min_distance'],
            max_attempts=args.max_attempts
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nGeneration completed in {elapsed:.2f} seconds")

    finally:
        # Restore original values
        config_patchy_particle.NUM_PARTICLES = original_num_particles
        config_patchy_particle.BOX_SIZE = original_box_size

    # Extract positions and orientations
    positions = np.array(body.center)
    orientations = np.array(body.orientation)

    # Validate IC
    print("\nValidating IC...")
    validate_ic(positions, params['box_size'], params['min_distance'])

    # Save to NPZ file
    timestamp = datetime.now().isoformat()

    np.savez(
        output_path,
        # IC data
        positions=positions,
        orientations=orientations,
        # Parameters for validation
        seed=args.seed,
        num_particles=args.num_particles,
        density=args.density,
        box_size=params['box_size'],
        min_distance=params['min_distance'],
        # Constants for validation
        center_radius=params['center_radius'],
        patch_size=params['patch_size'],
        alpha=params['alpha'],
        # Metadata
        timestamp=timestamp,
        generation_time_seconds=elapsed,
        max_attempts=args.max_attempts,
    )

    print(f"\n{'='*80}")
    print(f"IC saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"{'='*80}\n")

    return output_path


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        output_path = generate_ic(args)
        print(f"Success! IC ready for use in yield simulations.")
        return 0
    except Exception as e:
        print(f"\nERROR: IC generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
