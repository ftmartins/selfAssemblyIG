"""
Hybrid GCMC (Grand Canonical Monte Carlo + Langevin MD) for 2D patchy particles.

State representation: MCMCState dataclass carries both position (q) and
velocity (v) arrays, allowing velocity carry-over between MD displacement moves
(analogous to partial momentum refresh in Hybrid Monte Carlo).

Public API
----------
MCMCState           : dataclass (q, v)
make_gcmc_params    : build params dict from physical arguments
generate_ic         : random-sequential-addition initial condition
mc_sweep            : one hybrid GCMC sweep, returns (MCMCState, stats_dict)
run_gcmc            : high-level loop: equilibration + production
total_energy_pbc    : O(N^2) diagnostic energy (numpy)
plot_configuration  : matplotlib visualisation helper
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from jax import config as _jcfg

from config_patchy_particle import PATCH_SIZE, ALPHA

# Enable 64-bit JAX once at import time (idempotent)
_jcfg.update('jax_enable_x64', True)

PATCH_SIGMA = float(PATCH_SIZE)
ALPHA_VAL   = float(ALPHA)


# ── State representation ──────────────────────────────────────────────────────

@dataclass
class MCMCState:
    """
    Full kinematic state for hybrid GCMC.

    Attributes
    ----------
    q : np.ndarray, shape (N, 3)
        Particle DOFs [x, y, theta].  theta = direction of patch A.
    v : np.ndarray, shape (N, 3)
        Conjugate velocities [vx, vy, omega].  Carried between MD runs.
    """
    q: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        assert self.q.shape == self.v.shape, (
            f"q shape {self.q.shape} != v shape {self.v.shape}"
        )

    @property
    def N(self) -> int:
        return len(self.q)


# ── Parameter helpers ─────────────────────────────────────────────────────────

def make_gcmc_params(
    opening_angle: float,
    E_AB: float,
    E_AA: float = 1.0,
    E_BB: float = 1.0,
    r_patch: float = 1.0,
    rep_A: float = 10_000.0,
) -> dict:
    """
    Assemble the params dict used by all energy/MC functions.

    All values are cast to Python float so that frozenset(params.items())
    is correctly hashable for the LRU cache in _make_md_runner.

    Parameters
    ----------
    opening_angle : float
        Angle between patch A and patch B in radians.
    E_AB : float
        A-B cross-patch Morse well depth.
    E_AA, E_BB : float
        Like-patch Morse well depths (default 1.0).
    r_patch : float
        Patch arm length (= CENTER_RADIUS = 1.0 by default).
    rep_A : float
        Soft-sphere repulsion prefactor.
    """
    return dict(
        opening_angle = float(opening_angle),
        r_patch       = float(r_patch),
        E_AA          = float(E_AA),
        E_BB          = float(E_BB),
        E_AB          = float(E_AB),
        rep_A         = float(rep_A),
        alpha         = ALPHA_VAL,
        patch_sigma   = PATCH_SIGMA,
    )


# ── Numpy energy functions (PBC, O(N) per particle) ──────────────────────────

def minimum_image(dr: np.ndarray, L: float) -> np.ndarray:
    """Minimum-image displacement for any shape of dr."""
    return dr - L * np.round(dr / L)


def _smooth_morse_weight(r: np.ndarray, patch_sigma: float) -> np.ndarray:
    """
    Smooth multiplicative window matching jax-md's multiplicative_isotropic_cutoff
    with r_onset=0:  w(r) = (rc²-r²)²(rc²+2r²)/rc⁶
    w(0)=1, w(patch_sigma)=0, C¹ everywhere.
    """
    rc2 = patch_sigma ** 2
    r2  = r ** 2
    return np.where(
        r < patch_sigma,
        (rc2 - r2) ** 2 * (rc2 + 2.0 * r2) / rc2 ** 3,
        0.0,
    )


def get_patches(
    q: np.ndarray,
    r_patch: float,
    opening_angle: float,
) -> tuple:
    """
    Extract patch positions from configuration q.

    Parameters
    ----------
    q : (N, 3) array [x, y, theta]
    r_patch : patch arm length
    opening_angle : angle between patch A and patch B

    Returns
    -------
    pA, pB : each (N, 2) in lab frame.

    Canonical convention:
        pA_i = center_i + r_patch * [cos(theta_i), sin(theta_i)]
        pB_i = center_i + r_patch * [cos(theta_i+oa), sin(theta_i+oa)]
    """
    c   = q[:, :2]
    th  = q[:, 2]
    oa  = opening_angle
    pA  = c + r_patch * np.stack([np.cos(th),      np.sin(th)],      axis=-1)
    pB  = c + r_patch * np.stack([np.cos(th + oa), np.sin(th + oa)], axis=-1)
    return pA, pB


def particle_energy(i: int, q: np.ndarray, L: float, params: dict) -> float:
    """
    Energy of particle i interacting with all j != i.  O(N).
    Used for incremental updates in every MC move.
    """
    N = len(q)
    if N <= 1:
        return 0.0

    mask = np.ones(N, dtype=bool)
    mask[i] = False
    q_j = q[mask]

    # center-center soft-sphere
    ci       = q[i, :2]
    cj       = q_j[:, :2]
    dr_cc    = minimum_image(ci[None, :] - cj, L)
    r_cc     = np.sqrt(np.sum(dr_cc ** 2, axis=-1))
    sigma_cc = 2.0 * params['r_patch']
    v_rep    = np.where(
        r_cc < sigma_cc,
        params['rep_A'] * (1.0 - r_cc / sigma_cc) ** 2,
        0.0,
    )

    # patch positions
    thi = q[i, 2]
    thj = q_j[:, 2]
    oa  = params['opening_angle']
    rp  = params['r_patch']
    pAi = ci + rp * np.array([np.cos(thi),      np.sin(thi)])
    pBi = ci + rp * np.array([np.cos(thi + oa), np.sin(thi + oa)])
    pAj = cj + rp * np.stack([np.cos(thj),      np.sin(thj)],      axis=-1)
    pBj = cj + rp * np.stack([np.cos(thj + oa), np.sin(thj + oa)], axis=-1)

    def morse(pi, pj_arr, eps):
        dr = minimum_image(pi[None, :] - pj_arr, L)
        r  = np.sqrt(np.sum(dr ** 2, axis=-1))
        v  = eps * (1.0 - np.exp(-params['alpha'] * r)) ** 2 - eps
        return _smooth_morse_weight(r, params['patch_sigma']) * v

    return float(np.sum(
        v_rep
        + morse(pAi, pAj, params['E_AA'])
        + morse(pBi, pBj, params['E_BB'])
        + morse(pAi, pBj, params['E_AB'])
        + morse(pBi, pAj, params['E_AB'])
    ))


def total_energy_pbc(q: np.ndarray, L: float, params: dict) -> float:
    """O(N²) full system energy via pairwise sum. For diagnostics only."""
    if len(q) == 0:
        return 0.0
    return 0.5 * sum(particle_energy(i, q, L, params) for i in range(len(q)))


# ── JAX energy (differentiable, for forces in BAOAB) ─────────────────────────

def _total_energy_jax(
    q_flat: jnp.ndarray,
    L: float,
    params: dict,
) -> jnp.ndarray:
    """
    Pure-JAX PBC energy.  q_flat: shape (3N,).
    Called via jax.grad() for forces in BAOAB.
    Mirrors total_energy_pbc exactly.
    """
    N  = q_flat.shape[0] // 3
    q  = q_flat.reshape(N, 3)
    c  = q[:, :2]
    th = q[:, 2]
    oa = params['opening_angle']
    rp = params['r_patch']

    pA = c + rp * jnp.stack([jnp.cos(th),      jnp.sin(th)],      axis=-1)
    pB = c + rp * jnp.stack([jnp.cos(th + oa), jnp.sin(th + oa)], axis=-1)

    i_idx, j_idx = jnp.triu_indices(N, k=1)

    def mi(dr):
        return dr - L * jnp.round(dr / L)

    def smooth_weight(r):
        rc2 = params['patch_sigma'] ** 2
        r2  = r ** 2
        return jnp.where(
            r < params['patch_sigma'],
            (rc2 - r2) ** 2 * (rc2 + 2.0 * r2) / rc2 ** 3,
            0.0,
        )

    def morse(pi, pj, eps):
        r = jnp.sqrt(jnp.sum(mi(pi - pj) ** 2, axis=-1))
        v = eps * (1.0 - jnp.exp(-params['alpha'] * r)) ** 2 - eps
        return smooth_weight(r) * v

    dr_cc = mi(c[i_idx] - c[j_idx])
    r_cc  = jnp.sqrt(jnp.sum(dr_cc ** 2, axis=-1))
    sigma_cc = 2.0 * rp
    v_rep = jnp.where(
        r_cc < sigma_cc,
        params['rep_A'] * (1.0 - r_cc / sigma_cc) ** 2,
        0.0,
    )
    return jnp.sum(
        v_rep
        + morse(pA[i_idx], pA[j_idx], params['E_AA'])
        + morse(pB[i_idx], pB[j_idx], params['E_BB'])
        + morse(pA[i_idx], pB[j_idx], params['E_AB'])
        + morse(pB[i_idx], pA[j_idx], params['E_AB'])
    )


# ── BAOAB Langevin MD runner (JIT-compiled, cached by N and all params) ───────

@functools.lru_cache(maxsize=256)
def _make_md_runner(
    N: int,
    MD_STEPS: int,
    MD_DT: float,
    GAMMA: float,
    kT: float,
    L: float,
    params_frozen: frozenset,
):
    """
    Build and JIT-compile a BAOAB Langevin runner for N particles.

    Cache key = (N, MD_STEPS, MD_DT, GAMMA, kT, L, params_frozen).
    Recompiles only when any of these change.  N changes most often (±1
    per insertion/deletion event); each new N triggers ~2-5 s XLA compilation
    on first call.

    Returns
    -------
    run : callable
        Signature: run(q_init, v_init, key) -> (q_final, v_final)
        Both arrays have shape (N, 3), dtype float64.

    BAOAB sequence per step:
      B  half velocity kick    v += 0.5*dt*F
      A  half position drift   q += 0.5*dt*v
      O  Ornstein-Uhlenbeck    v  = decay*v + noise
      A  half position drift   q += 0.5*dt*v   (+ PBC wrap)
      B  half velocity kick    v += 0.5*dt*F   (recomputed forces)
    """
    params    = dict(params_frozen)
    decay     = float(np.exp(-GAMMA * MD_DT))
    noise_std = float(np.sqrt(kT * (1.0 - decay ** 2)))

    def _force(q):
        """All-particle forces, shape (N, 3)."""
        return -jax.grad(
            lambda qf: _total_energy_jax(qf, L, params)
        )(q.flatten()).reshape(N, 3)

    def baoab_body(_, carry):
        q, v, F, key = carry
        key, subkey = jax.random.split(key)
        # B: half velocity kick
        v = v + 0.5 * MD_DT * F
        # A: half position drift
        q = q + 0.5 * MD_DT * v
        # O: Ornstein-Uhlenbeck thermostat
        v = decay * v + noise_std * jax.random.normal(subkey, v.shape)
        # A: half position drift + PBC wrap on positions
        q = q + 0.5 * MD_DT * v
        q = q.at[:, :2].set(jnp.mod(q[:, :2] + L / 2, L) - L / 2)
        # B: recompute forces, half velocity kick
        F = _force(q)
        v = v + 0.5 * MD_DT * F
        return (q, v, F, key)

    @jax.jit
    def run(q_init, v_init, key):
        F0 = _force(q_init)
        q_f, v_f, _, _ = jax.lax.fori_loop(
            0, MD_STEPS, baoab_body,
            (q_init, v_init, F0, key),
        )
        return q_f, v_f   # carry-over: return both positions and velocities

    return run


def _get_runner(
    N: int,
    L: float,
    params: dict,
    MD_STEPS: int,
    MD_DT: float,
    GAMMA: float,
    kT: float,
):
    """Dispatch to _make_md_runner with a hashable params key."""
    params_frozen = frozenset(params.items())
    return _make_md_runner(N, MD_STEPS, MD_DT, GAMMA, kT, L, params_frozen)


@functools.lru_cache(maxsize=256)
def _make_md_runner_diagnostic(
    N: int,
    MD_STEPS: int,
    MD_DT: float,
    GAMMA: float,
    kT: float,
    L: float,
    params_frozen: frozenset,
    n_chunks: int,
):
    """
    Like _make_md_runner but also samples energy at n_chunks equally-spaced
    checkpoints along the MD run.  Uses jax.lax.scan over (n_chunks) chunks
    of (chunk_size = MD_STEPS // n_chunks) BAOAB steps each.

    Returns
    -------
    run : callable
        Signature: run(q_init, v_init, key) -> (q_final, v_final, E_traj)
        E_traj has shape (n_chunks,), dtype float64.

    Use this only when --dump_md_energies is active; the extra scan overhead
    is not needed in normal production runs.
    """
    params     = dict(params_frozen)
    chunk_size = max(1, MD_STEPS // n_chunks)
    decay      = float(np.exp(-GAMMA * MD_DT))
    noise_std  = float(np.sqrt(kT * (1.0 - decay ** 2)))

    def _force(q):
        return -jax.grad(
            lambda qf: _total_energy_jax(qf, L, params)
        )(q.flatten()).reshape(N, 3)

    def baoab_body(_, carry):
        q, v, F, key = carry
        key, subkey = jax.random.split(key)
        v = v + 0.5 * MD_DT * F
        q = q + 0.5 * MD_DT * v
        v = decay * v + noise_std * jax.random.normal(subkey, v.shape)
        q = q + 0.5 * MD_DT * v
        q = q.at[:, :2].set(jnp.mod(q[:, :2] + L / 2, L) - L / 2)
        F = _force(q)
        v = v + 0.5 * MD_DT * F
        return (q, v, F, key)

    def chunk_body(carry, _):
        q, v, F, key = carry
        q, v, F, key = jax.lax.fori_loop(0, chunk_size, baoab_body, (q, v, F, key))
        E = _total_energy_jax(q.flatten(), L, params)
        return (q, v, F, key), E

    @jax.jit
    def run(q_init, v_init, key):
        F0 = _force(q_init)
        (q_f, v_f, _, _), E_traj = jax.lax.scan(
            chunk_body, (q_init, v_init, F0, key), None, length=n_chunks,
        )
        return q_f, v_f, E_traj

    return run


# ── MC move functions ─────────────────────────────────────────────────────────

def md_displacement(
    state: MCMCState,
    L: float,
    kT: float,
    params: dict,
    rng: np.random.Generator,
    MD_STEPS: int = 5000,
    MD_DT: float = 1e-2,
    GAMMA: float = 1.0,
) -> MCMCState:
    """
    Short NVT Langevin MD displacement move with velocity carry-over.

    Unlike the notebook version, velocities from the previous MD run
    (state.v) are reused as initial momenta rather than being freshly
    resampled from Maxwell-Boltzmann.  Only the JAX PRNG key for the
    Ornstein-Uhlenbeck noise is resampled each call.

    Always accepted — Langevin thermostat already enforces the canonical
    distribution, so no Metropolis criterion is needed.

    On first call for a given N, JAX compiles the runner (~2-5 s).
    Subsequent calls reuse the cached XLA kernel.
    """
    N = state.N
    if N == 0:
        return state

    run  = _get_runner(N, L, params, MD_STEPS, MD_DT, GAMMA, kT)
    key  = jax.random.PRNGKey(int(rng.integers(0, 2 ** 31)))

    q_j  = jnp.array(state.q, dtype=jnp.float64)
    v_j  = jnp.array(state.v, dtype=jnp.float64)

    q_new_j, v_new_j = run(q_j, v_j, key)

    return MCMCState(
        q=np.array(q_new_j),
        v=np.array(v_new_j),
    )


def md_displacement_diagnostic(
    state: MCMCState,
    L: float,
    kT: float,
    params: dict,
    rng: np.random.Generator,
    MD_STEPS: int = 5000,
    MD_DT: float = 1e-2,
    GAMMA: float = 1.0,
    n_chunks: int = 10,
) -> tuple:
    """
    Like md_displacement but also returns the energy sampled at n_chunks
    equally-spaced checkpoints along the MD run.

    Use to diagnose whether MD_STEPS is long enough: if energy has not
    plateaued by the last checkpoint, increase MD_STEPS.

    Returns
    -------
    (MCMCState, E_traj)
        E_traj : np.ndarray, shape (n_chunks,), float32
    """
    N = state.N
    if N == 0:
        return state, np.zeros(n_chunks, dtype=np.float32)

    params_frozen = frozenset(params.items())
    run  = _make_md_runner_diagnostic(N, MD_STEPS, MD_DT, GAMMA, kT, L, params_frozen, n_chunks)
    key  = jax.random.PRNGKey(int(rng.integers(0, 2 ** 31)))

    q_j  = jnp.array(state.q, dtype=jnp.float64)
    v_j  = jnp.array(state.v, dtype=jnp.float64)

    q_new_j, v_new_j, E_traj_j = run(q_j, v_j, key)

    return (
        MCMCState(q=np.array(q_new_j), v=np.array(v_new_j)),
        np.array(E_traj_j, dtype=np.float32),
    )


def mc_insertion(
    state: MCMCState,
    L: float,
    kT: float,
    mu: float,
    params: dict,
    rng: np.random.Generator,
) -> tuple:
    """
    Grand-canonical insertion of a trial particle at a random position.

    New particle receives freshly sampled Maxwell-Boltzmann velocities
    (there is no prior velocity to carry for a newly created particle).

    Acceptance:  min(1, A/(N+1) * exp((mu - dE)/kT))
    """
    N  = state.N
    x  = rng.uniform(-L / 2, L / 2)
    y  = rng.uniform(-L / 2, L / 2)
    th = rng.uniform(0.0, 2.0 * np.pi)

    q_new = np.vstack([state.q, [[x, y, th]]])
    dE    = particle_energy(N, q_new, L, params)
    acc   = (L ** 2 / (N + 1)) * np.exp((mu - dE) / kT)

    if rng.random() < min(1.0, acc):
        v_new_particle = rng.normal(0.0, np.sqrt(kT), size=(1, 3))
        v_new = np.vstack([state.v, v_new_particle])
        return MCMCState(q=q_new, v=v_new), True

    return state, False


def mc_deletion(
    state: MCMCState,
    L: float,
    kT: float,
    mu: float,
    params: dict,
    rng: np.random.Generator,
) -> tuple:
    """
    Grand-canonical deletion of a random particle.

    The deleted particle's velocity row is removed from state.v.

    Acceptance:  min(1, N/A * exp((-mu + E_i)/kT))
    """
    N = state.N
    if N == 0:
        return state, False

    i   = rng.integers(0, N)
    E_i = particle_energy(i, state.q, L, params)
    acc = (N / L ** 2) * np.exp((-mu + E_i) / kT)

    if rng.random() < min(1.0, acc):
        q_new = np.delete(state.q, i, axis=0)
        v_new = np.delete(state.v, i, axis=0)
        return MCMCState(q=q_new, v=v_new), True

    return state, False


def mc_sweep(
    state: MCMCState,
    L: float,
    kT: float,
    mu: float,
    params: dict,
    rng: np.random.Generator,
    f_disp: float = 0.5,
    MD_STEPS: int = 5000,
    MD_DT: float = 1e-2,
    GAMMA: float = 1.0,
    dump_md_energies: bool = False,
    md_energy_chunks: int = 10,
) -> tuple:
    """
    One hybrid GCMC sweep.

    n_att = max(N, 1) move attempts, each independently:
      displacement   with probability f_disp       (always accepted)
      insertion      with probability (1-f_disp)/2
      deletion       with probability (1-f_disp)/2

    Parameters
    ----------
    dump_md_energies : bool
        If True, record energy at md_energy_chunks checkpoints along each
        MD displacement run.  Stored in stats['md_E_trajs'] as a list of
        (md_energy_chunks,) float32 arrays, one per displacement attempt.
    md_energy_chunks : int
        Number of energy checkpoints per MD run (only used when dump_md_energies).

    Returns
    -------
    (state_new, stats) where stats = {
        'nd': int,               displacement attempts
        'ni': int,               insertion attempts
        'nl': int,               deletion attempts
        'ad': int,               displacement accepted (always == nd)
        'ai': int,               insertions accepted
        'al': int,               deletions accepted
        'md_E_trajs': list       (only present when dump_md_energies=True)
            list of (md_energy_chunks,) float32 arrays
    }
    """
    n_att = max(state.N, 1)
    stats = {'nd': 0, 'ni': 0, 'nl': 0, 'ad': 0, 'ai': 0, 'al': 0}
    if dump_md_energies:
        stats['md_E_trajs'] = []
    f_ins = (1.0 - f_disp) / 2.0

    for _ in range(n_att):
        r = rng.random()
        if r < f_disp:
            if dump_md_energies:
                state, E_traj = md_displacement_diagnostic(
                    state, L, kT, params, rng,
                    MD_STEPS=MD_STEPS, MD_DT=MD_DT, GAMMA=GAMMA,
                    n_chunks=md_energy_chunks,
                )
                stats['md_E_trajs'].append(E_traj)
            else:
                state = md_displacement(state, L, kT, params, rng,
                                        MD_STEPS=MD_STEPS, MD_DT=MD_DT, GAMMA=GAMMA)
            stats['nd'] += 1
            stats['ad'] += 1   # always accepted
        elif r < f_disp + f_ins:
            state, acc = mc_insertion(state, L, kT, mu, params, rng)
            stats['ni'] += 1
            stats['ai'] += int(acc)
        else:
            state, acc = mc_deletion(state, L, kT, mu, params, rng)
            stats['nl'] += 1
            stats['al'] += int(acc)

    return state, stats


# ── Initial condition ─────────────────────────────────────────────────────────

def generate_ic(
    N: int,
    L: float,
    params: dict,
    kT: float = 1.0,
    seed: int = 42,
    max_att: int = 20_000,
) -> MCMCState:
    """
    Random Sequential Addition of N non-overlapping particles.

    Returns MCMCState with Maxwell-Boltzmann velocities at temperature kT.
    This is the only place velocities are freshly sampled from MB;
    thereafter they are carried between MD runs.
    """
    rng   = np.random.default_rng(seed)
    sigma = 2.0 * params['r_patch']
    pts   = []

    for n in range(N):
        for _ in range(max_att):
            x  = rng.uniform(-L / 2, L / 2)
            y  = rng.uniform(-L / 2, L / 2)
            th = rng.uniform(0.0, 2.0 * np.pi)
            ok = all(
                np.sqrt(np.sum(
                    minimum_image(np.array([x - p[0], y - p[1]]), L) ** 2
                )) >= sigma
                for p in pts
            )
            if ok:
                pts.append([x, y, th])
                break
        else:
            raise RuntimeError(f'RSA failed at particle {n + 1}/{N}')

    q = np.array(pts, dtype=np.float64)
    v = rng.normal(0.0, np.sqrt(kT), size=q.shape)
    return MCMCState(q=q, v=v)


# ── High-level simulation loop ────────────────────────────────────────────────

def run_gcmc(
    state: MCMCState,
    L: float,
    kT: float,
    mu: float,
    params: dict,
    rng: np.random.Generator,
    n_equil: int,
    n_prod: int,
    f_disp: float = 0.5,
    MD_STEPS: int = 5000,
    MD_DT: float = 1e-2,
    GAMMA: float = 1.0,
    snapshot_interval: int = 1,
    dump_md_energies: bool = False,
    md_energy_chunks: int = 10,
    verbose: bool = False,
) -> dict:
    """
    Full GCMC run: equilibration then production.

    Parameters
    ----------
    state             : initial MCMCState
    L                 : box side length
    kT                : thermal energy
    mu                : chemical potential
    params            : interaction parameters (from make_gcmc_params)
    rng               : numpy random generator
    n_equil           : number of equilibration sweeps (not recorded)
    n_prod            : number of production sweeps (recorded)
    f_disp            : fraction of moves that are MD displacement
    MD_STEPS          : Langevin steps per displacement move
    MD_DT             : Langevin time step
    GAMMA             : Langevin friction coefficient
    snapshot_interval : save position snapshot + energy every this many
                        production sweeps; N_traj is always recorded every sweep
    dump_md_energies  : if True, record energy at md_energy_chunks checkpoints
                        along every MD run in the production phase
    md_energy_chunks  : number of energy checkpoints per MD run
    verbose           : print progress every 10% of sweeps

    Returns
    -------
    dict with keys:
        'N_traj'         : (n_prod,) int32 — particle count every sweep
        'E_traj'         : (n_snapshots,) float32 — total energy at snapshot times
        'snapshot_sweeps': (n_snapshots,) int32 — production sweep index of each snapshot
        'snapshots'      : list of (N_i, 3) float64 arrays — positions+orientations
        'acc_md'         : float — mean displacement acceptance (always 1.0)
        'acc_ins'        : float — mean insertion acceptance fraction
        'acc_del'        : float — mean deletion acceptance fraction
        'final_state'    : MCMCState at end of production
        'q_final'        : (N_final, 3) float64
        'md_E_traj'      : (n_disp_moves, md_energy_chunks) float32
                           only present when dump_md_energies=True
    """
    sweep_kwargs = dict(
        f_disp=f_disp, MD_STEPS=MD_STEPS, MD_DT=MD_DT, GAMMA=GAMMA,
        dump_md_energies=False,   # disabled during equilibration
    )

    # ── Equilibration ──────────────────────────────────────────────────────────
    for s in range(n_equil):
        state, _ = mc_sweep(state, L, kT, mu, params, rng, **sweep_kwargs)
        if verbose and (s + 1) % max(1, n_equil // 10) == 0:
            print(f'  equil {s + 1:6d}/{n_equil}  N={state.N}')

    # ── Production ────────────────────────────────────────────────────────────
    prod_sweep_kwargs = dict(
        f_disp=f_disp, MD_STEPS=MD_STEPS, MD_DT=MD_DT, GAMMA=GAMMA,
        dump_md_energies=dump_md_energies,
        md_energy_chunks=md_energy_chunks,
    )

    N_traj       = np.empty(n_prod, dtype=np.int32)
    E_list       = []
    snap_sweeps  = []
    snapshots    = []   # list of (N_i, 3) arrays
    acc_md_list  = []
    acc_ins_list = []
    acc_del_list = []
    md_E_list    = []   # list of (n_chunks,) arrays, one per displacement move

    for s in range(n_prod):
        state, stats = mc_sweep(state, L, kT, mu, params, rng, **prod_sweep_kwargs)
        N_traj[s] = state.N

        if stats['nd'] > 0:
            acc_md_list.append(stats['ad'] / stats['nd'])
        if stats['ni'] > 0:
            acc_ins_list.append(stats['ai'] / stats['ni'])
        if stats['nl'] > 0:
            acc_del_list.append(stats['al'] / stats['nl'])

        if dump_md_energies and stats.get('md_E_trajs'):
            md_E_list.extend(stats['md_E_trajs'])

        if (s + 1) % snapshot_interval == 0:
            E_now = total_energy_pbc(state.q, L, params)
            E_list.append(np.float32(E_now))
            snap_sweeps.append(s + 1)
            snapshots.append(state.q.copy())
            if verbose and (s + 1) % max(1, n_prod // 10) == 0:
                print(f'  prod  {s + 1:6d}/{n_prod}  N={state.N}  '
                      f'E/N={E_now/max(state.N,1):.2f}')

    result = {
        'N_traj'          : N_traj,
        'E_traj'          : np.array(E_list, dtype=np.float32),
        'snapshot_sweeps' : np.array(snap_sweeps, dtype=np.int32),
        'snapshots'       : snapshots,
        'acc_md'          : float(np.mean(acc_md_list))  if acc_md_list  else 1.0,
        'acc_ins'         : float(np.mean(acc_ins_list)) if acc_ins_list else 0.0,
        'acc_del'         : float(np.mean(acc_del_list)) if acc_del_list else 0.0,
        'final_state'     : state,
        'q_final'         : state.q.copy(),
    }
    if dump_md_energies:
        result['md_E_traj'] = (
            np.array(md_E_list, dtype=np.float32) if md_E_list
            else np.zeros((0, md_energy_chunks), dtype=np.float32)
        )
    return result


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_configuration(
    q: np.ndarray,
    L: float,
    params: dict,
    ax,
    title: str = '',
    r_c: float = 1.0,
) -> None:
    """
    Draw patchy particle configuration on ax.

    Blue circles = particle cores.  Red dots = patch A.  Green dots = patch B.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    ax.set_xlim(-L / 2 - 0.5, L / 2 + 0.5)
    ax.set_ylim(-L / 2 - 0.5, L / 2 + 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=2)

    box = plt.Rectangle((-L / 2, -L / 2), L, L,
                        fill=False, edgecolor='dimgray', lw=1.2)
    ax.add_patch(box)

    if len(q) == 0:
        ax.text(0, 0, 'N=0', ha='center', va='center', fontsize=10, color='gray')
        return

    rp = params['r_patch']
    oa = params['opening_angle']

    for xi, yi, thi in q:
        ax.add_patch(Circle((xi, yi), r_c, color='steelblue', alpha=0.4, zorder=2))

    pA, pB = get_patches(q, rp, oa)
    ax.scatter(pA[:, 0], pA[:, 1], s=20, c='red',   zorder=3, linewidths=0)
    ax.scatter(pB[:, 0], pB[:, 1], s=20, c='green', zorder=3, linewidths=0)
    ax.set_title(f'{title}  N={len(q)}', fontsize=8, pad=2)
