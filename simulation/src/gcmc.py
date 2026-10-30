"""
Hybrid GCMC (Grand-Canonical Monte Carlo + BAOAB Langevin MD) for 2-D patchy
particles — the grand-canonical (µVT) ensemble engine.

State: :class:`MCMCState` carries positions ``q`` (N, 3) = [x, y, theta] and
conjugate velocities ``v`` (N, 3), so momenta persist between MD displacement
moves (partial-momentum-refresh Hybrid MC).

Public API
----------
MCMCState, make_gcmc_params, GcmcDefaults
generate_ic         random-sequential-addition initial condition (+ MB velocities)
mc_sweep            one hybrid GCMC sweep -> (MCMCState, stats)
run_gcmc            equilibration + production loop streaming to the HDF5 store

Energy primitives (periodic, params-dict) live in :mod:`base.energy`.
"""
from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from base.params import make_gcmc_params  # re-exported
from base.energy import (
    particle_energy,
    total_energy_pbc,
    total_energy_jax as _total_energy_jax,
)
from base.initial_conditions import rsa_2d

__all__ = [
    "MCMCState", "make_gcmc_params", "GcmcDefaults",
    "generate_ic", "mc_sweep", "run_gcmc",
    "md_displacement", "mc_insertion", "mc_deletion",
    "total_energy_pbc",
]


# ── Defaults (were config_gcmc.DEFAULT_*) ──────────────────────────────────

@dataclass(frozen=True)
class GcmcDefaults:
    box_area: float = 100.0
    target_phi: float = 0.2
    r_center: float = 1.0
    E_AA: float = 1.0
    E_BB: float = 1.0
    E_AB: float = 5.0
    rep_A: float = 10_000.0
    kT: float = 1.0
    md_steps: int = 5_000
    md_dt: float = 1e-2
    gamma: float = 1.0
    f_disp: float = 0.5
    n_equil: int = 10_000
    n_prod: int = 5_000
    n_equil_full: int = 20_000
    n_prod_full: int = 10_000
    save_interval: int = 100
    n_init: int = 16
    seed: int = 42
    alpha_equil: float = 0.05


DEFAULTS = GcmcDefaults()


# ── State ─────────────────────────────────────────────────────────────────

@dataclass
class MCMCState:
    q: np.ndarray   # (N, 3) [x, y, theta]
    v: np.ndarray   # (N, 3) [vx, vy, omega]

    def __post_init__(self):
        assert self.q.shape == self.v.shape, (
            f"q shape {self.q.shape} != v shape {self.v.shape}"
        )

    @property
    def N(self) -> int:
        return len(self.q)


# ── BAOAB Langevin MD runner (JIT-compiled, cached by N + all params) ─────

@functools.lru_cache(maxsize=256)
def _make_md_runner(N, MD_STEPS, MD_DT, GAMMA, kT, L, params_frozen):
    params = dict(params_frozen)
    decay = float(np.exp(-GAMMA * MD_DT))
    noise_std = float(np.sqrt(kT * (1.0 - decay ** 2)))

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

    @jax.jit
    def run(q_init, v_init, key):
        F0 = _force(q_init)
        q_f, v_f, _, _ = jax.lax.fori_loop(
            0, MD_STEPS, baoab_body, (q_init, v_init, F0, key),
        )
        return q_f, v_f

    return run


def _get_runner(N, L, params, MD_STEPS, MD_DT, GAMMA, kT):
    return _make_md_runner(N, MD_STEPS, MD_DT, GAMMA, kT, L, frozenset(params.items()))


@functools.lru_cache(maxsize=256)
def _make_md_runner_diagnostic(N, MD_STEPS, MD_DT, GAMMA, kT, L, params_frozen, n_chunks):
    params = dict(params_frozen)
    chunk_size = max(1, MD_STEPS // n_chunks)
    decay = float(np.exp(-GAMMA * MD_DT))
    noise_std = float(np.sqrt(kT * (1.0 - decay ** 2)))

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
        return (q, v, F, key), (E, q)

    @jax.jit
    def run(q_init, v_init, key):
        F0 = _force(q_init)
        (q_f, v_f, _, _), (E_traj, q_traj) = jax.lax.scan(
            chunk_body, (q_init, v_init, F0, key), None, length=n_chunks,
        )
        return q_f, v_f, E_traj, q_traj

    return run


# ── MC moves ─────────────────────────────────────────────────────────────

def md_displacement(state, L, kT, params, rng,
                    MD_STEPS=5000, MD_DT=1e-2, GAMMA=1.0):
    """Short NVT Langevin MD displacement with velocity carry-over (always
    accepted — the Langevin thermostat enforces the canonical distribution)."""
    N = state.N
    if N == 0:
        return state
    run = _get_runner(N, L, params, MD_STEPS, MD_DT, GAMMA, kT)
    key = jax.random.PRNGKey(int(rng.integers(0, 2 ** 31)))
    q_new, v_new = run(jnp.array(state.q, jnp.float64), jnp.array(state.v, jnp.float64), key)
    return MCMCState(q=np.array(q_new), v=np.array(v_new))


def md_displacement_diagnostic(state, L, kT, params, rng,
                               MD_STEPS=5000, MD_DT=1e-2, GAMMA=1.0, n_chunks=10):
    """Like :func:`md_displacement` but also returns energy + positions sampled
    at ``n_chunks`` checkpoints along the MD run."""
    N = state.N
    if N == 0:
        return (state,
                np.zeros(n_chunks, np.float32),
                np.zeros((n_chunks, 0, 3), np.float64))
    run = _make_md_runner_diagnostic(
        N, MD_STEPS, MD_DT, GAMMA, kT, L, frozenset(params.items()), n_chunks
    )
    key = jax.random.PRNGKey(int(rng.integers(0, 2 ** 31)))
    q_new, v_new, E_traj, q_traj = run(
        jnp.array(state.q, jnp.float64), jnp.array(state.v, jnp.float64), key
    )
    return (
        MCMCState(q=np.array(q_new), v=np.array(v_new)),
        np.array(E_traj, np.float32),
        np.array(q_traj, np.float64),
    )


def mc_insertion(state, L, kT, mu, params, rng):
    """GC insertion at a random position.  Acceptance
    ``min(1, A/(N+1) * exp((mu - dE)/kT))``."""
    N = state.N
    x = rng.uniform(-L / 2, L / 2)
    y = rng.uniform(-L / 2, L / 2)
    th = rng.uniform(0.0, 2.0 * np.pi)
    q_new = np.vstack([state.q, [[x, y, th]]])
    dE = particle_energy(N, q_new, L, params)
    acc = (L ** 2 / (N + 1)) * np.exp((mu - dE) / kT)
    if rng.random() < min(1.0, acc):
        v_new = np.vstack([state.v, rng.normal(0.0, np.sqrt(kT), size=(1, 3))])
        return MCMCState(q=q_new, v=v_new), True
    return state, False


def mc_deletion(state, L, kT, mu, params, rng):
    """GC deletion of a random particle.  Acceptance
    ``min(1, N/A * exp((-mu + E_i)/kT))``."""
    N = state.N
    if N == 0:
        return state, False
    i = rng.integers(0, N)
    E_i = particle_energy(i, state.q, L, params)
    acc = (N / L ** 2) * np.exp((-mu + E_i) / kT)
    if rng.random() < min(1.0, acc):
        return MCMCState(q=np.delete(state.q, i, axis=0),
                         v=np.delete(state.v, i, axis=0)), True
    return state, False


def mc_sweep(state, L, kT, mu, params, rng,
             f_disp=0.5, MD_STEPS=5000, MD_DT=1e-2, GAMMA=1.0,
             dump_md_energies=False, md_energy_chunks=10):
    """One hybrid GCMC sweep of ``max(N, 1)`` attempts (f_disp displacement,
    rest split insertion/deletion)."""
    n_att = max(state.N, 1)
    stats = {'nd': 0, 'ni': 0, 'nl': 0, 'ad': 0, 'ai': 0, 'al': 0}
    if dump_md_energies:
        stats['md_E_trajs'] = []
        stats['md_q_trajs'] = []
    f_ins = (1.0 - f_disp) / 2.0

    for _ in range(n_att):
        r = rng.random()
        if r < f_disp:
            if dump_md_energies:
                state, E_traj, q_traj = md_displacement_diagnostic(
                    state, L, kT, params, rng,
                    MD_STEPS=MD_STEPS, MD_DT=MD_DT, GAMMA=GAMMA,
                    n_chunks=md_energy_chunks,
                )
                stats['md_E_trajs'].append(E_traj)
                stats['md_q_trajs'].append(q_traj)
            else:
                state = md_displacement(state, L, kT, params, rng,
                                        MD_STEPS=MD_STEPS, MD_DT=MD_DT, GAMMA=GAMMA)
            stats['nd'] += 1
            stats['ad'] += 1
        elif r < f_disp + f_ins:
            state, acc = mc_insertion(state, L, kT, mu, params, rng)
            stats['ni'] += 1
            stats['ai'] += int(acc)
        else:
            state, acc = mc_deletion(state, L, kT, mu, params, rng)
            stats['nl'] += 1
            stats['al'] += int(acc)
    return state, stats


# ── Initial condition ────────────────────────────────────────────────────

def generate_ic(N, L, params, kT=1.0, seed=42, max_att=20_000) -> MCMCState:
    """RSA of ``N`` non-overlapping particles in a centred box, with
    Maxwell-Boltzmann velocities at temperature ``kT``."""
    rng = np.random.default_rng(seed)
    pos, ori = rsa_2d(N, L, min_distance=2.0 * params['r_patch'],
                      rng=rng, centered=True, max_attempts=max_att)
    q = np.column_stack([pos, ori]).astype(np.float64)
    v = rng.normal(0.0, np.sqrt(kT), size=q.shape)
    return MCMCState(q=q, v=v)


# ── High-level loop ──────────────────────────────────────────────────────

_COUNTERS = ('nd', 'ni', 'nl', 'ad', 'ai', 'al')


def _acceptance(c):
    """(displacement, insertion, deletion) acceptance rates from the counters."""
    def rate(a, n, default):
        return a / n if n else default
    return (rate(c['ad'], c['nd'], 1.0), rate(c['ai'], c['ni'], 0.0),
            rate(c['al'], c['nl'], 0.0))


def run_gcmc(state, L, kT, mu, params, rng, n_equil, n_prod,
             f_disp=0.5, MD_STEPS=5000, MD_DT=1e-2, GAMMA=1.0,
             snapshot_interval=1, dump_md_energies=False, md_energy_chunks=10,
             verbose=True, trajectory_path=None, attrs=None, resume=False,
             precision='float64', equilibration_sample_interval=0,
             checkpoint_interval=None) -> dict:
    """Full GCMC run: ``n_equil`` equilibration sweeps then ``n_prod``
    production sweeps.

    With ``trajectory_path`` every sampled frame is streamed to the HDF5
    trajectory store (:mod:`base.trajectory`, the same schema the canonical
    engine writes): production frames every ``snapshot_interval`` sweeps
    (including sweep 0), equilibration frames every
    ``equilibration_sample_interval`` sweeps (0 = none).  Frame ``step`` is the
    sweep count within the phase; N varies per frame via the store's offsets.
    Positions are stored wrapped into ``[0, L)``.  A restart (MD state, RNG
    state, acceptance counters) is written every ``checkpoint_interval``
    sweeps (default ``snapshot_interval``); ``resume=True`` continues from it
    without duplicating frames.  Production acceptance rates
    (``acc_insertion``, ``acc_deletion``, ``acc_displacement``) are stored as
    root attributes.

    Returns a dict with ``N_traj`` / ``E_traj`` (production frames after
    sweep 0), ``acc_md/ins/del``, ``final_state``, ``q_final`` (and
    ``md_E_traj`` / ``md_q_traj`` when ``dump_md_energies``; these are not
    resumable).
    """
    from base.trajectory import TrajectoryWriter
    ckpt_interval = checkpoint_interval or snapshot_interval
    sweep_kwargs = dict(f_disp=f_disp, MD_STEPS=MD_STEPS, MD_DT=MD_DT, GAMMA=GAMMA)
    counters = dict.fromkeys(_COUNTERS, 0)
    N_list, E_list, md_E_list, md_q_list = [], [], [], []

    writer = None
    if trajectory_path is not None:
        writer = TrajectoryWriter(trajectory_path, attrs, resume=resume,
                                  precision=precision)

    phase, done = 'equilibration', 0
    if writer is not None and writer.restart is not None:
        phase, done, leaves, extra = writer.restart[:4]
        state = MCMCState(q=leaves[0], v=leaves[1])
        rng.bit_generator.state = json.loads(extra['rng_state'])
        counters = json.loads(extra['counters'])
        if verbose:
            print(f'  [resume] phase={phase} sweep={done} N={state.N}')

    def record(st, name, step):
        E = float(total_energy_pbc(st.q, L, params))
        if writer is not None:
            pos = np.mod(st.q[:, :2] + L / 2.0, L)      # centred box -> [0, L)
            writer.append_frame(pos, st.q[:, 2], E, step, phase=name)
        return E

    def checkpoint(st, name, step):
        if writer is not None:
            writer.write_restart(
                name, step, [st.q, st.v],
                rng_state=json.dumps(rng.bit_generator.state),
                counters=json.dumps(counters))

    def sweep(st, **kw):
        st, stats = mc_sweep(st, L, kT, mu, params, rng, **sweep_kwargs, **kw)
        for k in _COUNTERS:
            counters[k] += stats[k]
        return st, stats

    try:
        if phase == 'equilibration':
            eq_int = equilibration_sample_interval if writer is not None else 0
            if eq_int > 0 and done == 0 and writer.n_frames('equilibration') == 0:
                record(state, 'equilibration', 0)
            for s in range(done, n_equil):
                state, _ = sweep(state)
                if eq_int > 0 and (s + 1) % eq_int == 0:
                    record(state, 'equilibration', s + 1)
                if verbose and (s + 1) % max(1, n_equil // 10) == 0:
                    print(f'  equil {s + 1:6d}/{n_equil}  N={state.N}')
                if (s + 1) % ckpt_interval == 0 and s + 1 < n_equil:
                    checkpoint(state, 'equilibration', s + 1)
            # equilibration moves are not part of the reported production rates
            counters = dict.fromkeys(_COUNTERS, 0)
            phase, done = 'production', 0
            if writer is not None:
                record(state, 'production', 0)
            checkpoint(state, 'production', 0)

        if phase == 'production':
            if writer is not None and done == 0 and writer.n_frames('production') == 0:
                record(state, 'production', 0)
            for s in tqdm(range(done, n_prod), disable=not verbose):
                state, stats = sweep(state, dump_md_energies=dump_md_energies,
                                     md_energy_chunks=md_energy_chunks)
                if dump_md_energies and stats.get('md_E_trajs'):
                    md_E_list.extend(stats['md_E_trajs'])
                    md_q_list.extend(stats['md_q_trajs'])
                if (s + 1) % snapshot_interval == 0:
                    E_now = record(state, 'production', s + 1)
                    N_list.append(state.N)
                    E_list.append(E_now)
                    if verbose and (s + 1) % max(1, n_prod // 10) == 0:
                        print(f'  prod  {s + 1:6d}/{n_prod}  N={state.N}  '
                              f'E/N={E_now / max(state.N, 1):.2f}')
                if (s + 1) % ckpt_interval == 0 and s + 1 < n_prod:
                    checkpoint(state, 'production', s + 1)

        acc_md, acc_ins, acc_del = _acceptance(counters)
        if writer is not None and phase != 'done':
            writer.set_attrs(acc_displacement=acc_md, acc_insertion=acc_ins,
                             acc_deletion=acc_del)
            checkpoint(state, 'done', n_prod)
            N_list = [int(n) for n in writer._f['production']['N'][1:]]
            E_list = list(writer._f['production']['potential_energy'][1:])
    finally:
        if writer is not None:
            writer.close()

    result = {
        'N_traj': np.array(N_list, np.int32),
        'E_traj': np.array(E_list, np.float32),
        'acc_md': float(acc_md), 'acc_ins': float(acc_ins), 'acc_del': float(acc_del),
        'final_state': state,
        'q_final': state.q.copy(),
    }
    if dump_md_energies:
        result['md_E_traj'] = (
            np.array(md_E_list, np.float32) if md_E_list
            else np.zeros((0, md_energy_chunks), np.float32)
        )
        result['md_q_traj'] = md_q_list
    return result
