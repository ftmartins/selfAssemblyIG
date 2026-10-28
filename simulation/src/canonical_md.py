"""
Canonical-ensemble (NVT) molecular dynamics for 2-patch patchy particles.

Consolidates the three near-duplicate NVT Nose-Hoover drivers that used to be
scattered across the repo:

* ``modules/utility_functions.run_sim``          -> :func:`run_sim` / :func:`my_sim`
  (nested-scan rigid-body driver used by the design loop and the single-parameter
  yield runner)
* ``run_yield_simulation_cluster.run_many_particle_md`` -> :func:`run_many_particle_md`
  (chunked + checkpointed many-particle driver for SLURM array jobs)
* ``modules/relax_md.md_relax_system``           -> :func:`md_relax_system`
  (one-shot full-system relaxation)

All box sizes / step counts are explicit arguments — nothing reads global state.
The shared JAX-MD energy-function construction lives in
:func:`base.energy.build_pair_energy_fn`.
"""

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random, lax, vmap, remat
from jax_md import simulate, rigid_body

from base.config import (
    DT, CENTER_RADIUS, ALPHA, NUM_PATCHES, kT as KT_DEFAULT,
)
from base.geometry import thetas_to_shape, energy_matrix
from base.energy import build_pair_energy_fn

__all__ = [
    "YieldDefaults", "run_sim", "my_sim", "v_run_my_sim",
    "random_IC", "v_random_IC", "run_many_particle_md",
    "save_sim_checkpoint", "load_sim_checkpoint", "md_relax_system",
    "energy_fn_from_simple_params", "energy_fn_from_design_vector",
]


# ══════════════════════════════════════════════════════════════════════════
# Shared energy-function construction
# ══════════════════════════════════════════════════════════════════════════

def energy_fn_from_simple_params(params, box_size):
    """
    ``(energy_fn, shift_fn, shape)`` for a :class:`base.params.SimpleParams`.

    Opening angle from ``params.alpha``; Morse depths from
    ``params.E_RR / E_RB / E_BB``; soft-sphere prefactor from ``params.rep_A``.
    Morse decay/cutoff use the module ``ALPHA`` / ``MORSE_R_ONSET`` / ``MORSE_R_CUTOFF`` (the yield-sweep
    convention).
    """
    thetas = jnp.array([0.0, params.alpha])
    eng_mat = energy_matrix(jnp.array([params.E_RR, params.E_RB, params.E_BB]))
    energy_fn, shift = build_pair_energy_fn(
        thetas, eng_mat, box_size, rep_A=params.rep_A,
        alpha_morse=ALPHA, center_radius=CENTER_RADIUS,
    )
    return energy_fn, shift, thetas_to_shape(thetas, radius=CENTER_RADIUS)


def energy_fn_from_design_vector(thetas_and_energy, box_size,
                                center_radius=CENTER_RADIUS):
    """
    ``(energy_fn, shift_fn, shape, thetas)`` from the flat design vector
    ``[theta_0, theta_1, E_01, E_00, E_11, ...]`` (entry 0 pinned to angle 0,
    entries ``[:NUM_PATCHES]`` patch angles, rest the upper-triangular Morse
    depths).  Soft-sphere prefactor fixed at 1e4.
    """
    tae = thetas_and_energy.at[0].set(0.0)
    thetas = tae[:NUM_PATCHES]
    eng_mat = energy_matrix(tae[NUM_PATCHES:])
    energy_fn, shift = build_pair_energy_fn(
        thetas, eng_mat, box_size, rep_A=10_000.0, center_radius=center_radius,
    )
    return energy_fn, shift, thetas_to_shape(thetas, radius=center_radius), thetas


@dataclass(frozen=True)
class YieldDefaults:
    """Run-length defaults for the canonical yield sweep (were in
    ``config_patchy_particle.py``)."""
    num_particles: int = 3_000
    num_steps: int = 100_000
    equilibration_steps: int = 2_000_000
    sample_interval: int = 100
    checkpoint_interval: int = 10_000


DEFAULTS = YieldDefaults()


# ══════════════════════════════════════════════════════════════════════════
# Rigid-body nested-scan driver (design loop + single-parameter yield)
# ══════════════════════════════════════════════════════════════════════════

def run_sim(thetas_and_energy, x0, num_steps, center_radius, box_size, key,
            kT=KT_DEFAULT, print_sim=False, print_progress=False):
    """
    NVT Nose-Hoover MD via a nested ``lax.scan`` (``num_steps`` outer x
    ``num_steps`` inner = ``num_steps**2`` total steps), returning the outer-step
    trajectory of centers and orientations.

    ``thetas_and_energy`` is the flat design vector: entry 0 is pinned to 0
    (reference patch), entries ``[0:NUM_PATCHES]`` are patch angles, the rest are
    the upper-triangular Morse well depths.
    """
    energy_fn, shift, shape, _ = energy_fn_from_design_vector(
        thetas_and_energy, box_size, center_radius,
    )

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, DT, kT)
    step_fn = jit(step_fn)
    state = init_fn(key, x0, mass=shape.mass())
    do_step = jit(lambda st, t: (step_fn(st), 0.0))

    inner_steps = jnp.arange(num_steps)

    def do_loss_step(st, i):
        st, _ = lax.scan(do_step, st, inner_steps)
        if print_progress:
            jax.debug.print("outer step {i}/{n}", i=i + 1, n=num_steps)
        return st, (st.position.center, st.position.orientation)

    do_loss_step = jit(remat(do_loss_step))
    state, (positions, orientations) = lax.scan(do_loss_step, state, inner_steps)
    return state, positions, orientations


my_sim = jit(run_sim, static_argnums=(2, 8))
v_run_my_sim = jit(
    vmap(my_sim, in_axes=(None, 0, None, None, None, 0)),
    static_argnums=2,
)


# ── Uniform rigid-body initial condition ──────────────────────────────────

def random_IC(thetas_and_energy, key, num_particles, box_size):
    """RigidBody with uniform random centers in ``[0, box_size)`` and uniform
    random orientations."""
    key, pos_key, angle_key = random.split(key, 3)
    R = box_size * random.uniform(pos_key, (num_particles, 2), dtype=jnp.float64)
    angles = random.uniform(angle_key, (num_particles,), dtype=jnp.float64) * 2 * jnp.pi
    return rigid_body.RigidBody(R, angles)


v_random_IC = vmap(random_IC, in_axes=(None, 0, None, None))


# ══════════════════════════════════════════════════════════════════════════
# Many-particle chunked + checkpointed driver (SLURM array yield runner)
# ══════════════════════════════════════════════════════════════════════════

def save_sim_checkpoint(ckpt_path, state, phase, phase_step, **extra):
    np.savez(ckpt_path, full_md_state=state, phase=phase, phase_step=phase_step, **extra)
    print(f"    Checkpoint saved: {ckpt_path.name} (phase={phase}, step={phase_step:,})",
          flush=True)


def load_sim_checkpoint(ckpt_path):
    if not ckpt_path.exists():
        return None
    data = np.load(ckpt_path, allow_pickle=True)
    result = {
        'state': data['full_md_state'].item(),
        'phase': str(data['phase']),
        'phase_step': int(data['phase_step']),
    }
    for key in ['pos_traj', 'ori_traj', 'e_traj', 'sample_idx']:
        if key in data:
            result[key] = data[key] if key != 'sample_idx' else int(data[key])
    return result


def run_many_particle_md(num_particles, box_size, params, num_steps, kT, key, dt,
                         initial_positions, initial_orientations,
                         sample_interval=100, equilibration_steps=0,
                         checkpoint_dir=None, checkpoint_prefix='ckpt',
                         checkpoint_interval=200_000):
    """
    Many-particle NVT MD with chunked equilibration + production, disk
    checkpointing and trajectory recording.

    ``params`` is a :class:`base.params.SimpleParams`.  The caller supplies the
    initial ``(positions, orientations)`` (e.g. from
    :func:`base.initial_conditions.rsa_2d`).

    Returns ``(final_state, pos_traj, ori_traj, e_traj)``.
    """
    energy_fn, shift, shape = energy_fn_from_simple_params(params, box_size)

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)
    energy_fn_jit = jit(energy_fn)

    def _eq_step(carry, _):
        return step_fn(carry), None

    ckpt_path = (checkpoint_dir / f'{checkpoint_prefix}.npz') if checkpoint_dir else None
    ckpt = load_sim_checkpoint(ckpt_path) if ckpt_path else None

    if ckpt is not None:
        state = ckpt['state']
        phase = ckpt['phase']
        phase_step = ckpt['phase_step']
        print(f"  Resumed from checkpoint: phase={phase}, step={phase_step:,}", flush=True)
    else:
        x0 = rigid_body.RigidBody(jnp.asarray(initial_positions),
                                  jnp.asarray(initial_orientations))
        state = init_fn(key, x0, mass=shape.mass())
        phase = 'equilibration' if equilibration_steps > 0 else 'production'
        phase_step = 0

    if phase == 'equilibration':
        eq_done = phase_step
        print(f"  Equilibration ({equilibration_steps:,} steps, from {eq_done:,})...", flush=True)
        while eq_done < equilibration_steps:
            chunk = min(checkpoint_interval, equilibration_steps - eq_done)
            state, _ = lax.scan(_eq_step, state, jnp.arange(chunk))
            eq_done += chunk
            print(f"  Equilibration: {eq_done:,}/{equilibration_steps:,}", flush=True)
            if ckpt_path:
                save_sim_checkpoint(ckpt_path, state, 'equilibration', eq_done)
        phase, phase_step = 'production', 0

    max_samples = num_steps // sample_interval + 2

    if phase == 'production' and phase_step > 0 and ckpt is not None:
        positions_traj = jnp.array(ckpt['pos_traj'])
        orientations_traj = jnp.array(ckpt['ori_traj'])
        energy_traj = jnp.array(ckpt['e_traj'])
        sample_idx = jnp.array(ckpt['sample_idx'])
    else:
        positions_traj = jnp.zeros((max_samples, num_particles, 2))
        orientations_traj = jnp.zeros((max_samples, num_particles))
        energy_traj = jnp.zeros(max_samples)
        positions_traj = positions_traj.at[0].set(state.position.center)
        orientations_traj = orientations_traj.at[0].set(state.position.orientation)
        energy_traj = energy_traj.at[0].set(energy_fn_jit(state.position))
        sample_idx = jnp.array(0)

    def step_func(carry, step_idx):
        cur, s_idx, pos_t, ori_t, e_t = carry
        new = step_fn(cur)
        should = (step_idx + 1) % sample_interval == 0
        nxt = jnp.where(should & (s_idx < max_samples - 1), s_idx + 1, s_idx)
        pos_t = jnp.where(should, pos_t.at[nxt].set(new.position.center), pos_t)
        ori_t = jnp.where(should, ori_t.at[nxt].set(new.position.orientation), ori_t)
        e_t = jnp.where(should, e_t.at[nxt].set(energy_fn_jit(new.position)), e_t)
        return (new, nxt, pos_t, ori_t, e_t), None

    prod_done = phase_step
    carry = (state, sample_idx, positions_traj, orientations_traj, energy_traj)
    print(f"  Production ({num_steps:,} steps, from {prod_done:,})...", flush=True)
    while prod_done < num_steps:
        chunk = min(checkpoint_interval, num_steps - prod_done)
        carry, _ = lax.scan(step_func, carry, jnp.arange(prod_done, prod_done + chunk))
        prod_done += chunk
        print(f"  Production: {prod_done:,}/{num_steps:,}", flush=True)
        if ckpt_path:
            s_now, sidx, pt, ot, et = carry
            save_sim_checkpoint(ckpt_path, s_now, 'production', prod_done,
                                pos_traj=np.array(pt), ori_traj=np.array(ot),
                                e_traj=np.array(et), sample_idx=int(sidx))

    final_state, _, pos_traj, ori_traj, e_traj = carry
    return final_state, pos_traj, ori_traj, e_traj


# ══════════════════════════════════════════════════════════════════════════
# One-shot full-system relaxation
# ══════════════════════════════════════════════════════════════════════════

def md_relax_system(params, n_particles, num_steps, key, box_size,
                    kT=KT_DEFAULT, dt=None, zero_vel_flag=False):
    """
    Full-system NVT Nose-Hoover relaxation from a uniform random start.

    ``params`` needs ``.alpha``, ``.E_RR``, ``.E_RB``, ``.E_BB``, ``.rep_A``
    (energy built via :func:`energy_fn_from_simple_params`).

    Returns ``(q_relaxed (3n,), final_state, positions_traj, orientations_traj,
    energy_traj)``.
    """
    dt = DT if dt is None else dt
    energy_fn, shift, shape = energy_fn_from_simple_params(params, box_size)

    k1, k2 = random.split(key)
    positions = random.uniform(k1, (n_particles, 2), minval=0.0, maxval=box_size)
    orientations = random.uniform(k2, (n_particles,), minval=0.0, maxval=2 * jnp.pi)
    x0 = rigid_body.RigidBody(positions, orientations)

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt / 2.0, kT)
    step_fn = jit(step_fn)
    state = init_fn(key, x0, mass=shape.mass())
    if zero_vel_flag:
        from dataclasses import replace
        momentum = rigid_body.RigidBody(
            center=jnp.zeros_like(state.position.center),
            orientation=jnp.zeros_like(state.position.orientation),
        )
        state = replace(state, momentum=momentum)

    energy_fn_jit = jit(energy_fn)

    def scan_step(carry, _):
        st = step_fn(carry)
        return st, (st.position.center, st.position.orientation, energy_fn_jit(st.position))

    final_state, traj = lax.scan(scan_step, state, jnp.arange(num_steps))
    positions_traj = jnp.vstack([state.position.center[None, ...], traj[0]])
    orientations_traj = jnp.vstack([state.position.orientation[None, ...], traj[1]])
    energy_traj = jnp.array(traj[2])

    q_relaxed = jnp.zeros(n_particles * 3)
    q_relaxed = q_relaxed.at[0::3].set(final_state.position.center[:, 0])
    q_relaxed = q_relaxed.at[1::3].set(final_state.position.center[:, 1])
    q_relaxed = q_relaxed.at[2::3].set(final_state.position.orientation)
    return q_relaxed, final_state, positions_traj, orientations_traj, energy_traj
