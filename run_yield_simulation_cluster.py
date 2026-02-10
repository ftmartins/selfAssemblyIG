#!/usr/bin/env python
"""
Cluster script for running many-particle MD yield simulations.

Each SLURM array task runs one (alpha, zeta) parameter set with multiple
independent realizations.  The task ID indexes into the grid defined in
config_patchy_particle.py:

    job_id -> (YIELD_ALPHAS_DEG[alpha_idx], YIELD_ZETAS[zeta_idx])

Usage:
    python run_yield_simulation_cluster.py <job_id> [--output-dir DIR]

    # With SLURM job array
    #SBATCH --array=0-99
    python run_yield_simulation_cluster.py $SLURM_ARRAY_TASK_ID
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# JAX imports
import jax
import jax.numpy as jnp
from jax import random, jit, lax
from jax_md import space, energy, simulate, rigid_body

# Local imports
from config_patchy_particle import (
    DT, CENTER_RADIUS, CENTER_MASS, ALPHA, PATCH_SIZE, R_CUTOFF, NUM_PATCHES, PATCH_MASS,
    EQUILIBRATION_STEPS, NUM_PARTICLES_YIELD, BOX_SIZE_YIELD, NUM_STEPS_YIELD, kT,
    YIELD_ALPHAS_DEG, YIELD_ZETAS, NUM_YIELD_JOBS, NUM_REALIZATIONS,
    SAMPLE_INTERVAL_YIELD, CHECKPOINT_INTERVAL,
)
# Import utility_functions directly to avoid freud dependency in modules/__init__.py
import importlib.util
from pathlib import Path as _Path
_spec = importlib.util.spec_from_file_location(
    "utility_functions",
    _Path(__file__).parent / "modules" / "utility_functions.py"
)
_utility_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utility_module)
thetas_to_shape = _utility_module.thetas_to_shape
energy_matrix = _utility_module.energy_matrix

_spec2 = importlib.util.spec_from_file_location(
    "params",
    _Path(__file__).parent / "modules" / "params.py"
)
_params_module = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_params_module)
SimpleParams = _params_module.SimpleParams
generate_parameter_grid = _params_module.generate_parameter_grid

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

NUM_PARTICLES = 1000
BOX_SIZE = BOX_SIZE_YIELD
NUM_MD_STEPS = NUM_STEPS_YIELD
TEMPERATURE = kT
DT_SIM = DT
SAMPLE_INTERVAL = SAMPLE_INTERVAL_YIELD

# Structure detection
BOND_THRESHOLD = 0.5
STRUCTURES = ['triangle', 'square', 'pentagon', 'hexagon', 'heptagon', 'octagon']

DEFAULT_OUTPUT_DIR = Path('yield_results_cluster')

# ==============================================================================
# CHECKPOINT FUNCTIONS
# ==============================================================================

def save_sim_checkpoint(ckpt_path, state, phase, phase_step, **extra):
    """
    Save NVT simulation state to disk for crash recovery.

    Uses np.savez with allow_pickle=True to serialize the JAX-MD state pytree.
    """
    np.savez(
        ckpt_path,
        full_md_state=state,
        phase=phase,
        phase_step=phase_step,
        **extra,
    )
    print(f"    Checkpoint saved: {ckpt_path.name} (phase={phase}, step={phase_step:,})", flush=True)


def load_sim_checkpoint(ckpt_path):
    """
    Load simulation checkpoint from disk.

    Returns dict with 'state', 'phase', 'phase_step' and any extra arrays,
    or None if checkpoint file does not exist.
    """
    if not ckpt_path.exists():
        return None
    data = np.load(ckpt_path, allow_pickle=True)
    result = {
        'state': data['full_md_state'].item(),
        'phase': str(data['phase']),
        'phase_step': int(data['phase_step']),
    }
    # Load optional production trajectory arrays
    for key in ['pos_traj', 'ori_traj', 'e_traj', 'sample_idx']:
        if key in data:
            result[key] = data[key] if key != 'sample_idx' else int(data[key])
    return result


# ==============================================================================
# STRUCTURE DETECTION FUNCTIONS
# ==============================================================================

def detect_clusters(positions, orientations, r_patch, opening_angle,
                    bond_threshold=0.5, box_size=None):
    """
    Detect clusters based on patch-patch proximity.

    Two particles are bonded if their patches are within bond_threshold distance.
    """
    n = len(positions)
    positions = np.asarray(positions)
    orientations = np.asarray(orientations)

    # Compute patch positions
    angle_A = orientations - opening_angle / 2
    angle_B = orientations + opening_angle / 2

    patches_A = positions + r_patch * np.column_stack([np.cos(angle_A), np.sin(angle_A)])
    patches_B = positions + r_patch * np.column_stack([np.cos(angle_B), np.sin(angle_B)])

    # Find bonds
    bonds = []
    adjacency = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            d_AA = np.linalg.norm(patches_A[i] - patches_A[j])
            d_BB = np.linalg.norm(patches_B[i] - patches_B[j])
            d_AB = np.linalg.norm(patches_A[i] - patches_B[j])
            d_BA = np.linalg.norm(patches_B[i] - patches_A[j])

            if d_AA < bond_threshold:
                bonds.append((i, j, 'A', 'A'))
                adjacency[i].add(j)
                adjacency[j].add(i)
            if d_BB < bond_threshold:
                bonds.append((i, j, 'B', 'B'))
                adjacency[i].add(j)
                adjacency[j].add(i)
            if d_AB < bond_threshold:
                bonds.append((i, j, 'A', 'B'))
                adjacency[i].add(j)
                adjacency[j].add(i)
            if d_BA < bond_threshold:
                bonds.append((i, j, 'B', 'A'))
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Find connected components (clusters)
    visited = set()
    clusters = []

    for start in range(n):
        if start in visited:
            continue

        cluster = []
        queue = [start]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            queue.extend(adjacency[node] - visited)

        clusters.append(cluster)

    return clusters, bonds


def classify_cluster(cluster_size):
    """Classify cluster by size."""
    size_to_name = {
        1: 'monomer', 2: 'dimer', 3: 'triangle', 4: 'square',
        5: 'pentagon', 6: 'hexagon', 7: 'heptagon', 8: 'octagon',
    }
    return size_to_name.get(cluster_size, f'cluster_{cluster_size}')


def compute_yields_from_frame(positions, orientations, r_patch, opening_angle,
                               bond_threshold=0.5):
    """
    Count number of structures and compute yields for a single frame.
    Y[structure] = num_structures_of_type / total_num_structures
    """
    clusters, bonds = detect_clusters(positions, orientations, r_patch,
                                       opening_angle, bond_threshold)

    structure_counts = {s: 0 for s in STRUCTURES}
    monomer_count = 0

    for cluster in clusters:
        size = len(cluster)
        cluster_type = classify_cluster(size)
        if cluster_type in structure_counts:
            structure_counts[cluster_type] += 1
        elif cluster_type == 'monomer':
            monomer_count += 1

    total_structures = sum(structure_counts.values())
    yields = {s: (structure_counts[s] / total_structures if total_structures > 0 else 0)
              for s in STRUCTURES}

    return yields, structure_counts, monomer_count, total_structures


# ==============================================================================
# INITIAL CONDITION GENERATION
# ==============================================================================

def generate_nonoverlap_ic(num_particles, box_size, key, min_distance=None):
    """Generate non-overlapping random positions with random orientations."""
    if min_distance is None:
        min_distance = 2 * CENTER_RADIUS + 2 * PATCH_SIZE

    displacement_fn, _ = space.periodic(box_size)
    key, pos_key, angle_key = random.split(key, 3)

    positions = []
    for i in range(num_particles):
        placed = False
        for attempt in range(10000):
            pos_key, subkey = random.split(pos_key)
            candidate = box_size * random.uniform(subkey, (2,))

            overlap = False
            for existing in positions:
                dr = displacement_fn(candidate, jnp.array(existing))
                if jnp.sqrt(jnp.sum(dr**2)) < min_distance:
                    overlap = True
                    break

            if not overlap:
                positions.append(candidate)
                placed = True
                break

        if not placed:
            raise RuntimeError(f"Failed to place particle {i} after 10000 attempts")

    positions = jnp.array(positions)
    orientations = random.uniform(angle_key, (num_particles,)) * 2 * jnp.pi

    return positions, orientations


# ==============================================================================
# MD SIMULATION
# ==============================================================================

def run_many_particle_md(num_particles, box_size, params, num_steps, kT, key, dt,
                         sample_interval=100, equilibration_steps=0,
                         checkpoint_dir=None, checkpoint_prefix='ckpt',
                         checkpoint_interval=200_000):
    """
    Many-particle NVT MD simulation with checkpointing, progress printing,
    optional equilibration, and trajectory tracking.

    Both equilibration and production phases are broken into chunks of
    `checkpoint_interval` steps. Between chunks, progress is printed and
    a checkpoint is saved to disk so that crashed jobs can resume.

    Parameters
    ----------
    num_particles : int
        Number of particles.
    box_size : float
        Simulation box size.
    params : SimpleParams
        Energy parameters (must have .alpha, .E_RR, .E_RB, .E_BB, .rep_A, .r_patch).
    num_steps : int
        Production MD steps (trajectory is recorded during these).
    kT : float
        Temperature.
    key : PRNGKey
        JAX random key.
    dt : float
        Time step.
    sample_interval : int
        Save trajectory every N steps during production.
    equilibration_steps : int
        Number of MD steps to run before production (no trajectory stored).
    checkpoint_dir : Path or None
        Directory for checkpoint files. None disables checkpointing.
    checkpoint_prefix : str
        Filename prefix for checkpoint (without .npz).
    checkpoint_interval : int
        Steps between checkpoints / progress prints.

    Returns
    -------
    final_state : NVT state
    pos_traj : array (n_samples, num_particles, 2)
    ori_traj : array (n_samples, num_particles)
    e_traj : array (n_samples,)
    """
    # Setup parameters
    alpha = params.alpha
    thetas = jnp.array([0.0, alpha])
    eng = jnp.array([params.E_RR, params.E_RB, params.E_BB])
    eng_mat = energy_matrix(eng)
    shape = thetas_to_shape(thetas, radius=CENTER_RADIUS)

    # Periodic boundary conditions
    displacement, shift = space.periodic(box_size)

    # Energy function
    morse_eps = jnp.pad(eng_mat, pad_width=(1, 0))
    soft_sphere_eps = jnp.zeros((3, 3))
    soft_sphere_eps = soft_sphere_eps.at[0, 0].set(params.rep_A)

    pair_energy_soft = energy.soft_sphere_pair(
        displacement, species=3,
        sigma=CENTER_RADIUS * 2, epsilon=soft_sphere_eps
    )
    pair_energy_morse = energy.morse_pair(
        displacement, species=3,
        sigma=0.0, epsilon=morse_eps,
        alpha=ALPHA, r_cutoff=R_CUTOFF
    )
    pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) + pair_energy_morse(R, **kwargs)
    energy_fn = rigid_body.point_energy(pair_energy_fn, shape)

    # Initialize NVT Nose-Hoover integrator (always needed, even on resume)
    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)
    energy_fn_jit = jit(energy_fn)

    # Equilibration step function (defined once, reused across chunks)
    def _eq_step(carry, _):
        return step_fn(carry), None

    # ---- Check for existing checkpoint ----
    ckpt_path = (checkpoint_dir / f'{checkpoint_prefix}.npz') if checkpoint_dir else None
    ckpt = load_sim_checkpoint(ckpt_path) if ckpt_path else None

    if ckpt is not None:
        state = ckpt['state']
        phase = ckpt['phase']
        phase_step = ckpt['phase_step']
        print(f"  Resumed from checkpoint: phase={phase}, step={phase_step:,}", flush=True)
    else:
        # Fresh start: generate IC
        key, ic_key = random.split(key)
        print(f"  Generating initial conditions...", flush=True)
        positions, orientations = generate_nonoverlap_ic(num_particles, box_size, ic_key)
        x0 = rigid_body.RigidBody(positions, orientations)
        print(f"  Initial conditions generated", flush=True)
        state = init_fn(key, x0, mass=shape.mass())
        phase = 'equilibration' if equilibration_steps > 0 else 'production'
        phase_step = 0

    # ==================================================================
    # EQUILIBRATION (chunked, no trajectory)
    # ==================================================================
    if phase == 'equilibration':
        eq_done = phase_step
        print(f"  Running equilibration ({equilibration_steps:,} steps, "
              f"starting from {eq_done:,})...", flush=True)
        while eq_done < equilibration_steps:
            chunk = min(checkpoint_interval, equilibration_steps - eq_done)
            state, _ = lax.scan(_eq_step, state, jnp.arange(chunk))
            eq_done += chunk
            pct = 100 * eq_done / equilibration_steps
            print(f"  Equilibration: {eq_done:,}/{equilibration_steps:,} "
                  f"({pct:.0f}%)", flush=True)
            if ckpt_path:
                save_sim_checkpoint(ckpt_path, state, 'equilibration', eq_done)
        print(f"  Equilibration complete", flush=True)
        phase = 'production'
        phase_step = 0

    # ==================================================================
    # PRODUCTION (chunked, with trajectory recording)
    # ==================================================================
    max_samples = num_steps // sample_interval + 2

    if phase == 'production' and phase_step > 0 and ckpt is not None:
        # Restore trajectory arrays from checkpoint
        positions_traj = jnp.array(ckpt['pos_traj'])
        orientations_traj = jnp.array(ckpt['ori_traj'])
        energy_traj = jnp.array(ckpt['e_traj'])
        sample_idx = jnp.array(ckpt['sample_idx'])
    else:
        # Fresh production: allocate arrays and store initial frame
        positions_traj = jnp.zeros((max_samples, num_particles, 2))
        orientations_traj = jnp.zeros((max_samples, num_particles))
        energy_traj = jnp.zeros(max_samples)
        positions_traj = positions_traj.at[0].set(state.position.center)
        orientations_traj = orientations_traj.at[0].set(state.position.orientation)
        energy_traj = energy_traj.at[0].set(energy_fn_jit(state.position))
        sample_idx = jnp.array(0)

    # Production step function
    def step_func(carry, step_idx):
        current_state, s_idx, pos_traj, ori_traj, e_traj = carry
        new_state = step_fn(current_state)

        should_sample = (step_idx + 1) % sample_interval == 0
        next_idx = jnp.where(should_sample & (s_idx < max_samples - 1),
                             s_idx + 1, s_idx)

        new_pos_traj = jnp.where(
            should_sample,
            pos_traj.at[next_idx].set(new_state.position.center),
            pos_traj
        )
        new_ori_traj = jnp.where(
            should_sample,
            ori_traj.at[next_idx].set(new_state.position.orientation),
            ori_traj
        )
        new_e_traj = jnp.where(
            should_sample,
            e_traj.at[next_idx].set(energy_fn_jit(new_state.position)),
            e_traj
        )

        return (new_state, next_idx, new_pos_traj, new_ori_traj, new_e_traj), None

    prod_done = phase_step
    carry = (state, sample_idx, positions_traj, orientations_traj, energy_traj)
    print(f"  Running production MD ({num_steps:,} steps, "
          f"starting from {prod_done:,})...", flush=True)

    while prod_done < num_steps:
        chunk = min(checkpoint_interval, num_steps - prod_done)
        step_indices = jnp.arange(prod_done, prod_done + chunk)
        carry, _ = lax.scan(step_func, carry, step_indices)
        prod_done += chunk
        pct = 100 * prod_done / num_steps
        print(f"  Production: {prod_done:,}/{num_steps:,} ({pct:.0f}%)", flush=True)
        if ckpt_path:
            state_now, sidx, pt, ot, et = carry
            save_sim_checkpoint(ckpt_path, state_now, 'production', prod_done,
                                pos_traj=np.array(pt), ori_traj=np.array(ot),
                                e_traj=np.array(et), sample_idx=int(sidx))

    print(f"  Production MD complete", flush=True)

    final_state, final_idx, pos_traj, ori_traj, e_traj = carry

    return final_state, pos_traj, ori_traj, e_traj


# ==============================================================================
# YIELD COMPUTATION FOR ONE REALIZATION
# ==============================================================================

def compute_realization_yields(pos_traj_np, ori_traj_np, params):
    """Compute yields from the post-equilibration trajectory of one realization."""
    # Find valid (non-zero) frames
    valid_frames = np.any(pos_traj_np != 0, axis=(1, 2))
    n_valid = np.sum(valid_frames)
    start_frame = max(1, n_valid // 2)

    frame_yields = []
    frame_counts = []

    for frame_idx in range(start_frame, n_valid):
        pos = pos_traj_np[frame_idx]
        ori = ori_traj_np[frame_idx]
        yields, counts, monomers, total = compute_yields_from_frame(
            pos, ori, params.r_patch, params.alpha, BOND_THRESHOLD
        )
        frame_yields.append(yields)
        frame_counts.append(counts)

    if len(frame_yields) > 0:
        avg_yields = {s: np.mean([fy[s] for fy in frame_yields]) for s in STRUCTURES}
        avg_counts = {s: np.mean([fc[s] for fc in frame_counts]) for s in STRUCTURES}
    else:
        avg_yields = {s: 0.0 for s in STRUCTURES}
        avg_counts = {s: 0.0 for s in STRUCTURES}

    return avg_yields, avg_counts, len(frame_yields)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run MD yield simulation for a single parameter set (multiple realizations)',
    )
    parser.add_argument('job_id', type=int, help='Job ID (index into parameter grid)')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Output directory for results')
    args = parser.parse_args()

    job_id = args.job_id

    # Validate job_id
    if job_id < 0 or job_id >= NUM_YIELD_JOBS:
        print(f"Error: job_id {job_id} out of range [0, {NUM_YIELD_JOBS - 1}]")
        sys.exit(1)

    # Map job_id to (alpha, zeta)
    alpha_idx = job_id // len(YIELD_ZETAS)
    zeta_idx = job_id % len(YIELD_ZETAS)
    alpha_deg = YIELD_ALPHAS_DEG[alpha_idx]
    zeta = YIELD_ZETAS[zeta_idx]

    # Build params for this (alpha, zeta) pair
    param_grid = generate_parameter_grid(
        np.deg2rad([alpha_deg]), np.array([zeta]), kT=TEMPERATURE
    )
    params = param_grid[0]

    print("=" * 60)
    print(f"JOB {job_id}")
    print("=" * 60)
    print(f"  Opening angle:  {alpha_deg:.1f} deg")
    print(f"  Selectivity:    {zeta:.3f}")
    print(f"  E_RR={params.E_RR:.3f}, E_BB={params.E_BB:.3f}, E_RB={params.E_RB:.3f}")
    print(f"  Particles:      {NUM_PARTICLES}")
    print(f"  Box size:       {BOX_SIZE:.2f}")
    print(f"  Equilibration:  {EQUILIBRATION_STEPS:,} steps")
    print(f"  Production:     {NUM_MD_STEPS:,} steps")
    print(f"  Sample interval:{SAMPLE_INTERVAL}")
    print(f"  Checkpoint int: {CHECKPOINT_INTERVAL:,} steps")
    print(f"  Temperature:    {TEMPERATURE}")
    print(f"  Realizations:   {NUM_REALIZATIONS}")
    print("=" * 60)

    # Create output directory and checkpoint subdirectory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint subfolder encodes all simulation parameters:
    #   checkpoints/N<particles>_alpha<deg>_zeta<val>/r<realization>/ckpt.npz
    ckpt_base = output_dir / 'checkpoints' / f'N{NUM_PARTICLES}_alpha{alpha_deg:.1f}_zeta{zeta:.3f}'

    # ---- Run all realizations ----
    all_pos_trajs = []
    all_ori_trajs = []
    all_e_trajs = []
    all_yields = []
    all_counts = []
    all_n_frames = []

    total_start = datetime.now()

    for r in range(NUM_REALIZATIONS):
        print(f"\n--- Realization {r + 1}/{NUM_REALIZATIONS} ---")
        key = random.PRNGKey(job_id * 1000 + r)
        ckpt_dir = ckpt_base / f'r{r}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        r_start = datetime.now()

        final_state, pos_traj, ori_traj, e_traj = run_many_particle_md(
            NUM_PARTICLES, BOX_SIZE, params, NUM_MD_STEPS,
            TEMPERATURE, key, DT_SIM, SAMPLE_INTERVAL,
            equilibration_steps=EQUILIBRATION_STEPS,
            checkpoint_dir=ckpt_dir,
            checkpoint_prefix='ckpt',
            checkpoint_interval=CHECKPOINT_INTERVAL,
        )

        r_elapsed = (datetime.now() - r_start).total_seconds()
        print(f"  Realization {r} completed in {r_elapsed:.1f}s")

        pos_traj_np = np.array(pos_traj)
        ori_traj_np = np.array(ori_traj)
        e_traj_np = np.array(e_traj)

        all_pos_trajs.append(pos_traj_np)
        all_ori_trajs.append(ori_traj_np)
        all_e_trajs.append(e_traj_np)

        # Compute yields for this realization
        print(f"  Computing yields...")
        avg_yields, avg_counts, n_frames = compute_realization_yields(
            pos_traj_np, ori_traj_np, params
        )
        all_yields.append(avg_yields)
        all_counts.append(avg_counts)
        all_n_frames.append(n_frames)

        print(f"  Y_tri={avg_yields['triangle']:.4f}  Y_sq={avg_yields['square']:.4f}")

        # Clean up checkpoint after successful realization
        ckpt_file = ckpt_dir / 'ckpt.npz'
        if ckpt_file.exists():
            ckpt_file.unlink()
            print(f"  Checkpoint cleaned up: {ckpt_file.relative_to(output_dir)}")

    total_elapsed = (datetime.now() - total_start).total_seconds()
    print(f"\nAll {NUM_REALIZATIONS} realizations completed in {total_elapsed:.1f}s")

    # ---- Save trajectory file ----
    traj_file = output_dir / f'job_{job_id:04d}_alpha{alpha_deg:.0f}_zeta{zeta:.2f}_trajectory.npz'
    traj_data = {
        'alpha_deg': alpha_deg,
        'selectivity': zeta,
        'box_size': BOX_SIZE,
        'num_realizations': NUM_REALIZATIONS,
        'sample_interval': SAMPLE_INTERVAL,
        'equilibration_steps': EQUILIBRATION_STEPS,
        'num_production_steps': NUM_MD_STEPS,
    }
    for r in range(NUM_REALIZATIONS):
        traj_data[f'r{r}_positions'] = all_pos_trajs[r]
        traj_data[f'r{r}_orientations'] = all_ori_trajs[r]
        traj_data[f'r{r}_energy'] = all_e_trajs[r]

    np.savez_compressed(traj_file, **traj_data)
    print(f"Saved trajectory to: {traj_file}")

    # ---- Save yields file ----
    yields_file = output_dir / f'job_{job_id:04d}_alpha{alpha_deg:.0f}_zeta{zeta:.2f}_yields.npz'
    yields_data = {
        'job_id': job_id,
        'alpha_deg': alpha_deg,
        'selectivity': zeta,
        'num_realizations': NUM_REALIZATIONS,
        'E_RR': params.E_RR,
        'E_BB': params.E_BB,
        'E_RB': params.E_RB,
    }
    for r in range(NUM_REALIZATIONS):
        for s in STRUCTURES:
            yields_data[f'r{r}_Y_{s}'] = all_yields[r][s]
            yields_data[f'r{r}_count_{s}'] = all_counts[r][s]
        yields_data[f'r{r}_n_frames'] = all_n_frames[r]

    # Realization-averaged yields
    for s in STRUCTURES:
        yields_data[f'avg_Y_{s}'] = np.mean([all_yields[r][s] for r in range(NUM_REALIZATIONS)])
        yields_data[f'avg_count_{s}'] = np.mean([all_counts[r][s] for r in range(NUM_REALIZATIONS)])

    np.savez(yields_file, **yields_data)
    print(f"Saved yields to: {yields_file}")

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("RESULTS (averaged over realizations)")
    print("=" * 60)
    print(f"  Y_tri  = {yields_data['avg_Y_triangle']:.4f}")
    print(f"  Y_sq   = {yields_data['avg_Y_square']:.4f}")
    print(f"  Y_diff = {yields_data['avg_Y_triangle'] - yields_data['avg_Y_square']:.4f}")
    print(f"\nStructure counts (average per frame, averaged over realizations):")
    for s in STRUCTURES:
        avg_c = yields_data[f'avg_count_{s}']
        if avg_c > 0.01:
            print(f"  {s:10s}: {avg_c:.2f}")
    print(f"\nTotal elapsed time: {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
