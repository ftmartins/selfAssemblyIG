#!/usr/bin/env python3
"""
Optimize patchy particle parameters for target shape assembly.

Usage:
    python optimize_patchy_particles.py --shape [square|triangle]

This script performs three-stage optimization:
    Stage 1: Coarse search (LR=0.5, ~17 steps)
    Stage 2: Medium refinement (LR=0.1, ~3 steps)
    Stage 3: Fine refinement (LR=0.05, ~3 steps)

IMPORTANT: The opening angle between patches is OPTIMIZED (not fixed).
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import jax.numpy as jnp
from jax import random

# Import configuration
from config_patchy_particle import *

# Import core modules
from modules.optimizer import optimize, random_search, generate_random_params
from modules.utility_functions import rename_file

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimize patchy particle parameters for target shape assembly'
    )
    parser.add_argument(
        '--shape',
        type=str,
        required=True,
        choices=['square', 'triangle', 'Square', 'Triangle'],
        help='Target shape to optimize for'
    )
    parser.add_argument(
        '--max_energy',
        type=float,
        default=MAX_ENERGY,
        help=f'Maximum bond energy (default: {MAX_ENERGY})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for optimization (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=OUTPUT_DIR_OPT,
        help=f'Output directory (default: {OUTPUT_DIR_OPT})'
    )
    parser.add_argument(
        '--opt_steps',
        type=int,
        default=OPT_STEPS,
        help=f'Optimization steps per stage (default: {OPT_STEPS})'
    )
    return parser.parse_args()

def initialize_parameters(args, key):
    """
    Initialize optimization parameters via random search.

    CRITICAL: The opening angle is a PARAMETER TO OPTIMIZE, not fixed!
    """
    print("Performing random parameter search...")
    _, start_params = random_search(
        key,
        RAND_SEARCH_ITERATIONS,
        NUM_PATCHES,
        args.max_energy,
        args.batch_size * 3
    )

    print(f"Random search complete. Initial parameters: {start_params}")
    return start_params

def run_optimization_stage(stage_num, stage_name, params, key, args,
                          learning_rate, steps, tracker, cmd='w',
                          myoptsteps=1, shape_name=None):
    """
    Run a single optimization stage.

    Parameters
    ----------
    stage_num : int
        Stage number (1, 2, or 3)
    stage_name : str
        Description of stage (e.g., "Coarse")
    params : array
        Input parameters
    key : PRNGKey
        Random key
    args : Namespace
        Command-line arguments
    learning_rate : float
        Learning rate for this stage
    steps : int
        Number of optimization steps
    tracker : str
        Identifier for logging files
    cmd : str
        File mode ('w' or 'a')
    myoptsteps : int
        Running step counter

    Returns
    -------
    tuple : (min_loss, min_params, cluster_loss, cluster_params,
             final_params, myoptsteps)
    """
    print(f"\n{'='*80}")
    print(f"Stage {stage_num}: {stage_name} Optimization")
    print(f"{'='*80}")
    print(f"Learning rate: {learning_rate}")
    print(f"Steps: {steps}")
    print(f"Input parameters: {params}")

    min_loss, min_params, cluster_loss, cluster_params, final_params, myoptsteps = optimize(
        input_params=params,
        key=key,
        opt_steps=steps,
        batch_size=args.batch_size,
        loop_batch=3,
        save_every=WRITE_EVERY,
        tracker=tracker,
        learning_rate=learning_rate,
        cmd=cmd,
        myoptsteps=myoptsteps,
        optimizer=OPTIMIZER,
        cl_type=shape_name,
    )

    print(f"\nStage {stage_num} Results:")
    print(f"  Min loss: {min_loss:.6f}")
    print(f"  Min params: {min_params}")
    print(f"  Cluster loss: {cluster_loss:.6f}")
    print(f"  Cluster params: {cluster_params}")

    return min_loss, min_params, cluster_loss, cluster_params, final_params, myoptsteps

def save_optimization_results(args, shape_name, optimized_params,
                              stage_results, timestamp):
    """
    Save optimization results to NPZ file and text summary.

    File format: {shape}_params_YYYYMMDD_HHMMSS.npz
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract stage results
    min_params1, cluster_params1 = stage_results[0][1], stage_results[0][3]
    min_params2, cluster_params2 = stage_results[1][1], stage_results[1][3]
    min_params3, cluster_params3 = stage_results[2][1], stage_results[2][3]

    # Save to NPZ
    output_file = os.path.join(
        args.output_dir,
        f"{shape_name}_params_{timestamp}.npz"
    )

    np.savez(
        output_file,
        shape=shape_name,
        final_params=optimized_params,
        timestamp=timestamp,
        # Stage 1
        min_params_stage1=min_params1,
        cluster_params_stage1=cluster_params1,
        # Stage 2
        min_params_stage2=min_params2,
        cluster_params_stage2=cluster_params2,
        # Stage 3
        min_params_stage3=min_params3,
        cluster_params_stage3=cluster_params3,
        # Configuration
        num_patches=NUM_PATCHES,
        max_energy=args.max_energy,
        batch_size=args.batch_size,
        learning_rates=LEARNING_RATES,
        optimizer=OPTIMIZER,
    )

    # Also save human-readable text summary
    summary_file = os.path.join(
        args.output_dir,
        f"{shape_name}_summary_{timestamp}.txt"
    )

    with open(summary_file, 'w') as f:
        f.write(f"Patchy Particle Optimization Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Shape: {shape_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Optimizer: {OPTIMIZER}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Max energy: {args.max_energy}\n\n")
        f.write(f"Final Optimized Parameters:\n")
        f.write(f"  {optimized_params}\n\n")
        f.write(f"Parameter Breakdown:\n")
        f.write(f"  Patch 0 angle (fixed): 0.0 rad\n")
        f.write(f"  Patch 1 angle (optimized): {optimized_params[0]:.6f} rad ({np.rad2deg(optimized_params[0]):.2f}°)\n")
        f.write(f"  Bond energies (optimized): {optimized_params[1:]}\n\n")
        f.write(f"CRITICAL NOTE: The opening angle was OPTIMIZED, not fixed!\n")
        f.write(f"  Opening angle = {np.rad2deg(optimized_params[0]):.2f}° (optimized via gradient descent)\n")

    print(f"\n{'='*80}")
    print(f"Results saved:")
    print(f"  NPZ file: {output_file}")
    print(f"  Summary: {summary_file}")
    print(f"{'='*80}")

    return output_file

def main():
    """Main optimization workflow."""
    args = parse_arguments()

    # Setup
    shape_name = args.shape.capitalize()
    shape_config = SHAPE_CONFIGS[shape_name.lower()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Update global shape_ID for this run
    import config_patchy_particle
    config_patchy_particle.shape_ID = shape_name

    print(f"\n{'='*80}")
    print(f"Patchy Particle Optimization")
    print(f"{'='*80}")
    print(f"Target shape: {shape_name} ({shape_config['description']})")
    print(f"Opening angle: TO BE OPTIMIZED (not fixed!)")
    print(f"Number of patches: {NUM_PATCHES}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max energy: {args.max_energy}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"{'='*80}\n")

    # Create job directory for intermediate files
    global JOBID
    JOBID = rename_file(f"{shape_name}-Assembly_Optimization")
    os.makedirs(JOBID, exist_ok=True)
    print(f"Job directory: {JOBID}\n")

    # Update JOBID in config
    config_patchy_particle.JOBID = JOBID

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize random key
    key = random.PRNGKey(KEY_PARAM_OPT)
    keys = random.split(key, 4)  # 1 for init, 3 for stages

    # Initialize parameters
    start_params = initialize_parameters(args, keys[0])

    # Three-stage optimization
    stage_results = []
    opt_params = start_params
    myoptsteps = 1

    # Stage 1: Coarse
    results1 = run_optimization_stage(
        1, "Coarse", opt_params, keys[1], args,
        learning_rate=LEARNING_RATES[0],
        steps=OPT_STAGE_STEPS[0],
        tracker="stage1",
        cmd='w',
        myoptsteps=myoptsteps,
        shape_name=shape_name
    )
    stage_results.append(results1)
    opt_params = results1[3]  # cluster_params
    myoptsteps = results1[5]

    # Stage 2: Medium
    results2 = run_optimization_stage(
        2, "Medium", opt_params, keys[2], args,
        learning_rate=LEARNING_RATES[1],
        steps=OPT_STAGE_STEPS[1],
        tracker="stage2",
        cmd='a',
        myoptsteps=myoptsteps,
        shape_name=shape_name
    )
    stage_results.append(results2)
    opt_params = results2[3]  # cluster_params
    myoptsteps = results2[5]

    # Stage 3: Fine
    results3 = run_optimization_stage(
        3, "Fine", opt_params, keys[3], args,
        learning_rate=LEARNING_RATES[2],
        steps=OPT_STAGE_STEPS[2],
        tracker="stage3",
        cmd='a',
        myoptsteps=myoptsteps,
        shape_name=shape_name
    )
    stage_results.append(results3)
    optimized_params = results3[3]  # cluster_params (final)

    # Final results
    print(f"\n{'='*80}")
    print(f"FINAL OPTIMAL PARAMETERS")
    print(f"{'='*80}")
    print(f"Parameters: {optimized_params}")
    print(f"Opening angle: {np.rad2deg(optimized_params[0]):.2f}° (OPTIMIZED)")
    print(f"{'='*80}\n")

    # Save results
    output_file = save_optimization_results(
        args, shape_name.lower(), optimized_params, stage_results, timestamp
    )

    print(f"\nOptimization complete! Use this file for yield simulation:")
    print(f"  python run_yield_simulation.py --params {output_file}")

if __name__ == "__main__":
    main()
