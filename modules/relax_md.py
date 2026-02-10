import sys
from typing import Optional

import jax
import jax.numpy as jnp
from jax import random, jit, lax

try:
    # prefer project-local modules layout
    from modules.utility_functions import thetas_to_shape, energy_matrix
except Exception:
    from utility_functions import thetas_to_shape, energy_matrix

try:
    import config_patchy_particle as cfg
except Exception:
    cfg = None

try:
    from jax_md import space, energy as jmd_energy, simulate, rigid_body
except Exception:
    raise ImportError('jax-md is required for relax_md.md_relax_system; please install jax-md')


def _get_dt_and_box(dt: Optional[float], box_size: Optional[float]):
    if dt is None:
        if cfg is not None and hasattr(cfg, 'DT'):
            dt = cfg.DT
        else:
            raise ValueError('dt not provided and config_patchy_particle.DT not found; please supply dt')
    if box_size is None:
        if cfg is not None and hasattr(cfg, 'BOX_SIZE'):
            box_size = cfg.BOX_SIZE
        else:
            raise ValueError('box_size not provided and config_patchy_particle.BOX_SIZE not found; please supply box_size')
    return dt, box_size


def md_relax_system(params,
                    n_particles:int,
                    num_steps:int,
                    key:random.PRNGKey,
                    box_size:Optional[float]=None,
                    kT:Optional[float]=None,
                    dt:Optional[float]=None,
                    init_mode:str='random',
                    zero_vel_flag:bool=False):
    """
    Full-system MD relax using NVT Nose-Hoover with periodic boundaries.

    Parameters
    ----------
    params: object
        Parameter object expected to contain at least attributes `alpha`, `E_RR`, `E_BB`, `E_RB`, and optionally `rep_A`, `morse_a`, `rep_alpha`, `morse_rcut`, `r_patch`.
    n_particles: int
        Number of particles in the full system.
    num_steps: int
        Number of MD steps to run (trajectory length returned will be num_steps+1).
    key: jax.random.PRNGKey
        Random key for initialization.
    box_size: float, optional
        Periodic box size. If None, attempts to read `BOX_SIZE` from `config_patchy_particle`.
    kT: float, optional
        Temperature. If None, attempts to read `kT` from config.
    dt: float, optional
        Timestep. If None, attempts to read `DT` from config.
    init_mode: str
        'random' (uniform) or 'regular' (not implemented for full-system). Default 'random'.
    zero_vel_flag: bool
        If True, zero out velocities initially.

    Returns
    -------
    q_relaxed: jnp.ndarray
        Flat coordinates shape (3*n_particles,) [x0,y0,theta0,...].
    final_state: jax-md state
        Final NVT state object.
    positions_traj: jnp.ndarray
        Array shape (num_steps+1, n_particles, 2)
    orientations_traj: jnp.ndarray
        Array shape (num_steps+1, n_particles)
    energy_traj: jnp.ndarray
        Array shape (num_steps,)
    """
    dt, box_size = _get_dt_and_box(dt, box_size)

    if kT is None:
        if cfg is not None and hasattr(cfg, 'kT'):
            kT = cfg.kT
        else:
            kT = 1.0

    # Build thetas and energy vector expected by helpers
    # The original notebook used two patch angles and an energy vector [E_RR, E_RB, E_BB]
    thetas = jnp.array([0.0, params.alpha])
    eng = jnp.array([params.E_RR, params.E_RB, params.E_BB])
    thetas_and_energy = jnp.concatenate([thetas, eng])

    # Map to energy matrix and rigid-body shape
    thetas = thetas_and_energy[:2]
    eng = thetas_and_energy[2:]
    eng_mat = energy_matrix(eng)
    # thetas_to_shape expects radius argument in original code; try to read CENTER_RADIUS
    radius = getattr(cfg, 'CENTER_RADIUS', 1.0) if cfg is not None else 1.0
    shape = thetas_to_shape(thetas, radius=radius)

    # Periodic displacement and shift
    displacement, shift = space.periodic(box_size)

    # Build pair potentials (soft-sphere repulsion + morse attraction similar to notebook)
    morse_eps = jnp.pad(eng_mat, pad_width=(1, 0))

    soft_sphere_eps = jnp.zeros((len(thetas) + 1, len(thetas) + 1))
    soft_sphere_eps = soft_sphere_eps.at[0, 0].set(1.0)
    rep_A = getattr(params, 'rep_A', 100.0)
    soft_sphere_eps = rep_A * soft_sphere_eps

    ALPHA = getattr(params, 'morse_a', getattr(params, 'morse_a', 7))
    R_CUTOFF = getattr(params, 'morse_rcut', getattr(params, 'morse_rcut', 0.1))

    pair_energy_soft = jmd_energy.soft_sphere_pair(displacement, species=1 + len(thetas), sigma=radius * 2, epsilon=soft_sphere_eps)
    pair_energy_morse = jmd_energy.morse_pair(displacement, species=1 + len(thetas), sigma=0.0, epsilon=morse_eps, alpha=ALPHA, r_cutoff=R_CUTOFF)
    pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) + pair_energy_morse(R, **kwargs)
    energy_fn = rigid_body.point_energy(pair_energy_fn, shape)

    # Initialize positions and orientations
    subkey1, subkey2 = random.split(key)
    if init_mode == 'random':
        positions = random.uniform(subkey1, (n_particles, 2), minval=0.0, maxval=box_size)
        orientations = random.uniform(subkey2, (n_particles,), minval=0.0, maxval=2 * jnp.pi)
    else:
        # fallback to random
        positions = random.uniform(subkey1, (n_particles, 2), minval=0.0, maxval=box_size)
        orientations = random.uniform(subkey2, (n_particles,), minval=0.0, maxval=2 * jnp.pi)

    x0 = rigid_body.RigidBody(positions, orientations)

    # Create NVT integrator
    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt / 2.0, kT)
    step_fn = jit(step_fn)
    init_fn = jit(init_fn)

    state = init_fn(key, x0, mass=shape.mass())

    if zero_vel_flag:
        momentum = rigid_body.RigidBody(center=jnp.zeros_like(state.position.center), orientation=jnp.zeros_like(state.position.orientation))
        from dataclasses import replace
        state = replace(state, momentum=momentum)

    energy_fn_jit = jit(energy_fn)

    # Use lax.scan to step efficiently and collect trajectory
    xs = jnp.arange(num_steps)

    def scan_step(carry, _):
        st = carry
        st = step_fn(st)
        pos = st.position.center
        orient = st.position.orientation
        e = energy_fn_jit(st.position)
        return st, (pos, orient, e)

    # perform scan
    final_state, traj = lax.scan(scan_step, state, xs)

    positions_traj = jnp.vstack([state.position.center[None, ...], traj[0]])
    orientations_traj = jnp.vstack([state.position.orientation[None, ...], traj[1]])
    energy_traj = jnp.array(traj[2])

    # Build flat coordinates for final configuration
    q_relaxed = jnp.zeros(n_particles * 3)
    q_relaxed = q_relaxed.at[0::3].set(final_state.position.center[:, 0])
    q_relaxed = q_relaxed.at[1::3].set(final_state.position.center[:, 1])
    q_relaxed = q_relaxed.at[2::3].set(final_state.position.orientation)

    return q_relaxed, final_state, positions_traj, orientations_traj, energy_traj


__all__ = ['md_relax_system']
