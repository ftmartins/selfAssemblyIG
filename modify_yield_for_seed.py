#!/usr/bin/env python3
"""
Modify run_yield_simulation.py to accept seed as command-line argument.
"""

# Read the file
with open('run_yield_simulation.py', 'r') as f:
    content = f.read()

# Replace parse_arguments function
old_parse = '''def parse_arguments():
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
    return parser.parse_args()'''

new_parse = '''def parse_arguments():
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
    return parser.parse_args()'''

content = content.replace(old_parse, new_parse)

# Replace all occurrences of KEY_PARAM_YIELD with args.seed
replacements = [
    ('checkpoint = load_checkpoint(args.output_dir, KEY_PARAM_YIELD)',
     'checkpoint = load_checkpoint(args.output_dir, args.seed)'),
    ('yield_key = random.PRNGKey(KEY_PARAM_YIELD)',
     'yield_key = random.PRNGKey(args.seed)'),
    ('trajectory_file = os.path.join(args.output_dir, f"trajectory_seed{KEY_PARAM_YIELD}.npz")',
     'trajectory_file = os.path.join(args.output_dir, f"trajectory_seed{args.seed}.npz")'),
    ('chunk_key = random.PRNGKey(KEY_PARAM_YIELD + chunk_start)',
     'chunk_key = random.PRNGKey(args.seed + chunk_start)'),
    ('save_checkpoint(\n                args.output_dir,\n                KEY_PARAM_YIELD,',
     'save_checkpoint(\n                args.output_dir,\n                args.seed,'),
    ('save_trajectory(\n            args.output_dir,\n            KEY_PARAM_YIELD,',
     'save_trajectory(\n            args.output_dir,\n            args.seed,'),
    ('f"{shape_name}_yields_{timestamp}_{KEY_PARAM_YIELD}.npz"',
     'f"{shape_name}_yields_{timestamp}_{args.seed}.npz"'),
    ('f"{shape_name}_yield_summary_{timestamp}_{KEY_PARAM_YIELD}.txt"',
     'f"{shape_name}_yield_summary_{timestamp}_{args.seed}.txt"'),
]

for old, new in replacements:
    content = content.replace(old, new)

# Add seed information to the run_yield_simulation function
old_run_start = '''def run_yield_simulation(params_dict, args):
    """
    Run the yield simulation with checkpointing every 100 steps.

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
    # Constants
    CHECKPOINT_INTERVAL = 100  # Save checkpoint every 100 steps'''

new_run_start = '''def run_yield_simulation(params_dict, args):
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
    CHECKPOINT_INTERVAL = 100  # Save checkpoint every 100 steps'''

content = content.replace(old_run_start, new_run_start)

# Add seed printing
old_print_header = '''    print(f"\\n{'='*80}")
    print(f"Yield Simulation with Checkpointing")
    print(f"{'='*80}")
    print(f"Number of particles: {args.num_particles}")
    print(f"Simulation steps: {args.num_steps}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} steps")
    print(f"Box size: {box_size_yield:.2f}")
    print(f"{'='*80}\\n")'''

new_print_header = '''    print(f"\\n{'='*80}")
    print(f"Yield Simulation with Checkpointing")
    print(f"{'='*80}")
    print(f"Random seed: {args.seed}")
    print(f"Number of particles: {args.num_particles}")
    print(f"Simulation steps: {args.num_steps}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} steps")
    print(f"Box size: {box_size_yield:.2f}")
    print(f"{'='*80}\\n")'''

content = content.replace(old_print_header, new_print_header)

# Write back
with open('run_yield_simulation.py', 'w') as f:
    f.write(content)

print("✓ Modified run_yield_simulation.py to accept --seed argument")
print("  All KEY_PARAM_YIELD references replaced with args.seed")
