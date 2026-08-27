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
run_gcmc            equilibration + production loop with checkpointing
save_checkpoint / load_checkpoint

Energy primitives (periodic, params-dict) live in :mod:`base.energy`.
"""
from __future__ import annotations

import functools
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
    "save_checkpoint", "load_checkpoint", "total_energy_pbc",
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


# ── Utility ──────────────────────────────────────────────────────────────

def _to_object_array(lst: list) -> np.ndarray:
    arr = np.empty(len(lst), dtype=object)
    for i, x in enumerate(lst):
        arr[i] = x
    return arr


# ── Checkpoint ───────────────────────────────────────────────────────────

def save_checkpoint(checkpoint_file, state, rng, phase, sweep,
                    N_traj_partial, E_list, snap_sweeps, snapshots,
                    acc_md_list, acc_ins_list, acc_del_list, md_E_list) -> None:
    data = dict(
        q=state.q.copy(), v=state.v.copy(),
        rng_state=rng.bit_generator.state,
        phase=phase, sweep=sweep,
        N_traj_partial=N_traj_partial.copy(),
        E_list=np.array(E_list, np.float32),
        snap_sweeps=np.array(snap_sweeps, np.int32),
        snapshots=_to_object_array(snapshots),
        acc_md_list=np.array(acc_md_list),
        acc_ins_list=np.array(acc_ins_list),
        acc_del_list=np.array(acc_del_list),
        md_E_list=np.array(md_E_list, dtype=object) if md_E_list else np.empty(0),
    )
    np.save(checkpoint_file, data, allow_pickle=True)
    print(f'  [checkpoint] saved  phase={phase}  sweep={sweep}  N={state.N}')


def load_checkpoint(checkpoint_file, rng):
    if not checkpoint_file or not os.path.exists(checkpoint_file):
        return None
    data = np.load(checkpoint_file, allow_pickle=True).item()
    rng.bit_generator.state = data['rng_state']
    print(f'  [checkpoint] resumed  phase={data["phase"]}  sweep={data["sweep"]}'
          f'  N={len(data["q"])}')
    return data


# ── High-level loop ──────────────────────────────────────────────────────

def run_gcmc(state, L, kT, mu, params, rng, n_equil, n_prod,
             f_disp=0.5, MD_STEPS=5000, MD_DT=1e-2, GAMMA=1.0,
             snapshot_interval=1, dump_md_energies=False, md_energy_chunks=10,
             verbose=True, checkpoint_file=None, checkpoint_interval=None) -> dict:
    """Full GCMC run: equilibration (unrecorded) then production (recorded).

    Returns a dict with ``N_traj``, ``E_traj``, ``snapshot_sweeps``,
    ``snapshots``, ``acc_md/ins/del``, ``final_state``, ``q_final`` (and
    ``md_E_traj`` / ``md_q_traj`` when ``dump_md_energies``).
    """
    ckpt_interval = checkpoint_interval if checkpoint_interval is not None else snapshot_interval

    sweep_kwargs = dict(f_disp=f_disp, MD_STEPS=MD_STEPS, MD_DT=MD_DT, GAMMA=GAMMA,
                        dump_md_energies=False)
    prod_sweep_kwargs = dict(f_disp=f_disp, MD_STEPS=MD_STEPS, MD_DT=MD_DT, GAMMA=GAMMA,
                             dump_md_energies=dump_md_energies,
                             md_energy_chunks=md_energy_chunks)

    ckpt = load_checkpoint(checkpoint_file, rng) if checkpoint_file else None
    if ckpt is not None:
        state = MCMCState(q=ckpt['q'], v=ckpt['v'])
        start_phase = str(ckpt['phase'])
        start_sweep = int(ckpt['sweep']) + 1
        N_traj = np.zeros(n_prod, np.int32)
        N_traj[:len(ckpt['N_traj_partial'])] = ckpt['N_traj_partial']
        E_list = list(ckpt['E_list'])
        snap_sweeps = list(ckpt['snap_sweeps'])
        snapshots = list(ckpt['snapshots'])
        acc_md_list = list(ckpt['acc_md_list'])
        acc_ins_list = list(ckpt['acc_ins_list'])
        acc_del_list = list(ckpt['acc_del_list'])
        md_E_list = list(ckpt['md_E_list']) if ckpt['md_E_list'].size > 0 else []
        md_q_list = []
    else:
        start_phase, start_sweep = 'eq', 0
        N_traj = np.empty(n_prod, np.int32)
        E_list, snap_sweeps, snapshots = [], [], []
        acc_md_list, acc_ins_list, acc_del_list = [], [], []
        md_E_list, md_q_list = [], []

    eq_start = start_sweep if start_phase == 'eq' else n_equil
    prod_start = start_sweep if start_phase == 'prod' else 0

    def _save(phase, s):
        if checkpoint_file is None:
            return
        save_checkpoint(
            checkpoint_file, state, rng, phase, s,
            N_traj[:s + 1] if phase == 'prod' else N_traj[:0],
            E_list, snap_sweeps, snapshots,
            acc_md_list, acc_ins_list, acc_del_list, md_E_list,
        )

    if eq_start < n_equil:
        for s in range(eq_start, n_equil):
            state, _ = mc_sweep(state, L, kT, mu, params, rng, **sweep_kwargs)
            if verbose and (s + 1) % max(1, n_equil // 10) == 0:
                print(f'  equil {s + 1:6d}/{n_equil}  N={state.N}')
            if checkpoint_file and (s + 1) % ckpt_interval == 0:
                _save('eq', s)

    for s in tqdm(range(prod_start, n_prod)):
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
            md_q_list.extend(stats['md_q_trajs'])
        if (s + 1) % snapshot_interval == 0:
            E_now = total_energy_pbc(state.q, L, params)
            E_list.append(np.float32(E_now))
            snap_sweeps.append(s + 1)
            snapshots.append(state.q.copy())
            if verbose and (s + 1) % max(1, n_prod // 10) == 0:
                print(f'  prod  {s + 1:6d}/{n_prod}  N={state.N}  '
                      f'E/N={E_now / max(state.N, 1):.2f}')
            if checkpoint_file and (s + 1) % ckpt_interval == 0:
                _save('prod', s)

    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print('[checkpoint] simulation complete, checkpoint removed.')

    result = {
        'N_traj': N_traj,
        'E_traj': np.array(E_list, np.float32),
        'snapshot_sweeps': np.array(snap_sweeps, np.int32),
        'snapshots': snapshots,
        'acc_md': float(np.mean(acc_md_list)) if acc_md_list else 1.0,
        'acc_ins': float(np.mean(acc_ins_list)) if acc_ins_list else 0.0,
        'acc_del': float(np.mean(acc_del_list)) if acc_del_list else 0.0,
        'final_state': state,
        'q_final': state.q.copy(),
    }
    if dump_md_energies:
        result['md_E_traj'] = (
            np.array(md_E_list, np.float32) if md_E_list
            else np.zeros((0, md_energy_chunks), np.float32)
        )
        result['md_q_traj'] = _to_object_array(md_q_list) if md_q_list else np.empty(0, dtype=object)
    return result
