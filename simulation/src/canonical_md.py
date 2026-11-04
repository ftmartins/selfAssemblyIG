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
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random, lax, vmap, remat
from jax_md import simulate, rigid_body

from base.config import (
    DT, CENTER_RADIUS, ALPHA, NUM_PATCHES, kT as KT_DEFAULT,
)
from base.geometry import thetas_to_shape, energy_matrix
from base.energy import build_pair_energy_fn, build_pair_energy_fn_neighbor_list
from base.trajectory import TrajectoryWriter

__all__ = [
    "YieldDefaults", "run_sim", "my_sim", "v_run_my_sim",
    "random_IC", "v_random_IC", "run_many_particle_md",
    "md_relax_system",
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


def neighbor_list_energy_fn_from_simple_params(params, box_size):
    """
    Neighbor-list twin of :func:`energy_fn_from_simple_params`:
    ``(neighbor_fns, energy_fn, shift_fn, shape)`` with ``energy_fn(body, neighbor)``.
    """
    thetas = jnp.array([0.0, params.alpha])
    eng_mat = energy_matrix(jnp.array([params.E_RR, params.E_RB, params.E_BB]))
    neighbor_fns, energy_fn, shift = build_pair_energy_fn_neighbor_list(
        thetas, eng_mat, box_size, rep_A=params.rep_A,
        alpha_morse=ALPHA, center_radius=CENTER_RADIUS,
    )
    return neighbor_fns, energy_fn, shift, thetas_to_shape(thetas, radius=CENTER_RADIUS)


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
    equilibration_sample_interval: int = 20_000   # sparse thinning of equilibration frames


DEFAULTS = YieldDefaults()

# Spare neighbor-list slots per point.  A dispersed start has few pairs, a
# clustered system many more, so the list is allocated generously up front.
NEIGHBOR_EXTRA_CAPACITY = 8


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

def _model_attrs(params, num_particles, box_size, num_steps, equilibration_steps,
                 kT, dt, sample_interval, equilibration_sample_interval, seed,
                 initial_condition, chain_length=5, chain_steps=2, sy_steps=3):
    E_AA, E_BB, E_AB = float(params.E_RR), float(params.E_BB), float(params.E_RB)
    return dict(
        ensemble='canonical', engine='nvt_nose_hoover', thermostat='nose_hoover_chain',
        seed=-1 if seed is None else int(seed), N=int(num_particles),
        box_size=float(box_size), kT=float(kT), dt=float(dt),
        opening_angle=float(params.alpha), E_AA=E_AA, E_BB=E_BB, E_AB=E_AB,
        bond_specificity=E_AB / (E_AA + E_BB), rep_A=float(params.rep_A),
        num_equilibration_steps=int(equilibration_steps),
        num_production_steps=int(num_steps),
        sample_interval=int(sample_interval),
        equilibration_sample_interval=int(equilibration_sample_interval),
        initial_condition=initial_condition, nhc_chain_length=int(chain_length),
        nhc_chain_steps=int(chain_steps), nhc_sy_steps=int(sy_steps),
    )


def run_many_particle_md(num_particles, box_size, params, num_steps, kT, key, dt,
                         initial_positions, initial_orientations,
                         sample_interval=100, equilibration_steps=0,
                         equilibration_sample_interval=0,
                         trajectory_path=None, resume=False, precision='float64',
                         checkpoint_interval=200_000, seed=None,
                         initial_condition='dispersed', neighbor_list=True,
                         chain_length=5, chain_steps=2, sy_steps=3):
    """
    Many-particle NVT MD, chunked, streaming every sampled frame to the HDF5
    trajectory store (:mod:`base.trajectory`) at ``trajectory_path``.

    ``params`` is a :class:`base.params.SimpleParams`.  The caller supplies the
    initial ``(positions, orientations)`` (e.g. from
    :func:`base.initial_conditions.rsa_2d`).  Production frames are written
    every ``sample_interval`` steps (including step 0); equilibration frames
    every ``equilibration_sample_interval`` steps (0 = none).  After each
    ``checkpoint_interval`` steps the MD state is written to the file's restart
    group; with ``resume=True`` an existing file is continued from its last
    restart, frames written after it are dropped and recomputed, so no frame is
    duplicated.  When the run completes the restart group holds the final state
    (phase ``'done'``).

    ``initial_condition`` (``'dispersed'`` / ``'preassembled'``) is recorded as a
    root attribute only; the caller builds the start.

    ``neighbor_list=True`` sums the energy over a Verlet list (O(N) per step);
    ``False`` uses the dense all-pairs energy (O(N^2), the reference).  The list
    is not persisted: it is rebuilt from the positions on resume, and a chunk
    that overflows its capacity is re-run from its start with a larger list (the
    dynamics are deterministic, so the re-run is identical).

    ``chain_length`` / ``chain_steps`` / ``sy_steps`` are the Nose-Hoover chain
    settings (chain links, chain sub-steps, Suzuki-Yoshida order).  The chain is a
    run of tiny sequential operations, so on a GPU it dominates the cost of a step;
    smaller values are cheaper.

    Returns the final MD state.
    """
    if trajectory_path is None:
        raise ValueError('trajectory_path is required')
    x0 = rigid_body.RigidBody(jnp.asarray(initial_positions),
                              jnp.asarray(initial_orientations))
    if neighbor_list:
        nbr_fns, energy_fn, shift, shape = neighbor_list_energy_fn_from_simple_params(
            params, box_size)
        nbrs = nbr_fns.allocate(x0, extra_capacity=NEIGHBOR_EXTRA_CAPACITY)
        init_fn, step_fn = simulate.nvt_nose_hoover(
            energy_fn, shift, dt, kT, chain_length=chain_length,
            chain_steps=chain_steps, sy_steps=sy_steps)
        state = init_fn(key, x0, mass=shape.mass(), neighbor=nbrs)
        step = lambda st, nb: step_fn(st, neighbor=nb)
        refresh = lambda st, nb: nb.update(st.position)
        potential = lambda pos, nb: energy_fn(pos, neighbor=nb)
    else:
        energy_fn, shift, shape = energy_fn_from_simple_params(params, box_size)
        init_fn, step_fn = simulate.nvt_nose_hoover(
            energy_fn, shift, dt, kT, chain_length=chain_length,
            chain_steps=chain_steps, sy_steps=sy_steps)
        state = init_fn(key, x0, mass=shape.mass())
        nbrs = None
        step = lambda st, nb: step_fn(st)
        refresh = lambda st, nb: nb
        potential = lambda pos, nb: energy_fn(pos)
    treedef = jax.tree_util.tree_structure(state)

    def observe(st, nb):
        return (st.position.center, st.position.orientation,
                potential(st.position, nb),
                rigid_body.kinetic_energy(st.position, st.momentum, st.mass))

    observe = jit(observe)

    @partial(jit, static_argnums=(2,))
    def advance(st, nb, n):
        def body(c, _):
            s_, nb_ = c
            s_ = step(s_, nb_)
            return (s_, refresh(s_, nb_)), None
        return lax.scan(body, (st, nb), None, length=n)[0]

    @partial(jit, static_argnums=(2, 3))
    def advance_sampled(st, nb, n_samples, interval):
        def body(c, _):
            c = advance(*c, interval)
            return c, observe(*c)
        return lax.scan(body, (st, nb), None, length=n_samples)

    def run_chunk(fn, st, *args):
        """Run ``fn(st, nbrs, *args)``; on neighbor-list overflow reallocate a larger
        list at the chunk's start state and run it again."""
        nonlocal nbrs
        extra = NEIGHBOR_EXTRA_CAPACITY
        while True:
            out = fn(st, nbrs, *args)
            new_nbrs = (out[0] if fn is advance_sampled else out)[1]
            if new_nbrs is None or not bool(new_nbrs.did_buffer_overflow):
                nbrs = new_nbrs
                return out
            # extra_capacity is per point (3N points); it also pads each cell
            extra *= 2
            if extra > 256:
                raise RuntimeError('neighbor list keeps overflowing; configuration '
                                   'has probably blown up')
            print(f"  Neighbor list overflow: reallocating (+{extra} pairs per point)",
                  flush=True)
            nbrs = nbr_fns.allocate(st.position, extra_capacity=extra)

    attrs = _model_attrs(params, num_particles, box_size, num_steps, equilibration_steps,
                         kT, dt, sample_interval, equilibration_sample_interval, seed,
                         initial_condition, chain_length, chain_steps, sy_steps)
    writer = TrajectoryWriter(trajectory_path, attrs, resume=resume, precision=precision)

    def record(st, phase, step):
        pos, ori, pe, ke = observe(st, nbrs)
        writer.append_frame(np.asarray(pos), np.asarray(ori), float(pe), step,
                            phase=phase, kinetic_energy=float(ke))

    def checkpoint(st, phase, step):
        writer.write_restart(phase, step, [np.asarray(l) for l in
                                           jax.tree_util.tree_leaves(st)])

    first = 'equilibration' if equilibration_steps > 0 else 'production'
    phase, done = first, 0
    if writer.restart is not None:
        phase, done, leaves, _ = writer.restart[:4]
        if phase == 'done' and done < num_steps:      # extend a finished run
            phase = 'production'
        state = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(l) for l in leaves])
        if neighbor_list:
            nbrs = nbr_fns.allocate(state.position,
                                    extra_capacity=NEIGHBOR_EXTRA_CAPACITY)
        print(f"  Resumed from checkpoint: phase={phase}, step={done:,}", flush=True)

    try:
        for name, total, interval in [('equilibration', equilibration_steps,
                                       equilibration_sample_interval),
                                      ('production', num_steps, sample_interval)]:
            if phase == 'done' or (phase == 'production' and name == 'equilibration'):
                continue
            if phase == 'equilibration' and name == 'production':
                phase, done = 'production', 0
            start = done if phase == name else 0
            if interval > 0 and start == 0 and writer.n_frames(name) == 0:
                record(state, name, 0)
            chunk = checkpoint_interval
            if interval > 0:
                chunk = max(interval, chunk // interval * interval)
            while start < total:
                c = min(chunk, total - start)
                n_s = c // interval if interval > 0 else 0
                if n_s:
                    (state, _), (pos, ori, pe, ke) = run_chunk(
                        advance_sampled, state, n_s, interval)
                    pos, ori, pe, ke = map(np.asarray, (pos, ori, pe, ke))
                    for k in range(n_s):
                        writer.append_frame(pos[k], ori[k], pe[k],
                                            start + interval * (k + 1), phase=name,
                                            kinetic_energy=ke[k])
                if c - n_s * interval:
                    state, _ = run_chunk(advance, state, c - n_s * interval)
                start += c
                print(f"  {name.capitalize()}: {start:,}/{total:,}", flush=True)
                checkpoint(state, name, start)
            phase, done = name, total
            if total == 0:
                checkpoint(state, name, 0)
        checkpoint(state, 'done', num_steps)
    finally:
        writer.close()
    return state


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
