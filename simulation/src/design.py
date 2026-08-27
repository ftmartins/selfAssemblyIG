"""
Patch-parameter design / optimization.

Gradient-based optimization of a 2-patch particle's opening angle and bond
energies so the system self-assembles into a target polygon (square / triangle).

Consolidates ``modules/optimizer.py`` + ``modules/evaluation_functions.py`` +
the design-only helpers from ``modules/utility_functions.py``.  All formerly
module-global knobs (``NUM_PARTICLES``, ``BOX_SIZE``, ``BATCH_SIZE``, ``CLIP``,
``OPTIMIZER``, ``JOBID`` …) now live on the explicit :class:`DesignConfig`, and
the JIT-compiled loss closures are built per run by :func:`make_loss_fns`.
"""

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
from jax import jit, random, value_and_grad, hessian, vmap
from jax.example_libraries import optimizers
from jax_md import space

from base.config import CENTER_RADIUS, NUM_PATCHES, get_box_size
from base.geometry import get_shape, body_to_plot
from base.structure_detection import (
    shape_match_loss, CLOSENESS_PENALTY, CLOSENESS_PENALTY_NEIGHBORS,
    PATCH_ALLOWANCE,
)
from simulation.src.canonical_md import v_run_my_sim

__all__ = [
    "DesignConfig", "make_loss_fns", "make_params", "generate_random_params",
    "random_search", "optimize", "get_clusters", "make_cluster_list",
]


# ── Configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DesignConfig:
    num_particles: int = 16
    density: float = 0.1
    batch_size: int = 64
    loop_batch: int = 3
    max_energy: float = 30.0
    num_patches: int = NUM_PATCHES
    clip: float = 10_000.0
    optimizer: str = "adam"                       # 'adam' | 'rms'
    learning_rates: tuple = (0.5, 0.1, 0.05, 1e-3)
    opt_stage_steps: tuple = (100, 100, 100, 100)
    num_steps: int = 40_000
    num_steps_to_opt: int = 1_000
    find_hessian: bool = False
    rand_eng_check: bool = True
    write_every: int = 10
    job_dir: str = "design_run"
    bug_print: bool = False

    @property
    def box_size(self) -> float:
        return get_box_size(self.density, self.num_particles, CENTER_RADIUS)

    @property
    def sqrt_num_steps_to_opt(self) -> int:
        return int(np.sqrt(self.num_steps_to_opt))

    @property
    def quick_steps(self) -> int:
        return int(np.sqrt(self.num_steps - self.num_steps_to_opt))


def _bprint(cfg: DesignConfig, msg):
    if cfg.bug_print:
        print(f"From Bug-Printer: {msg}", flush=True)


# ── Loss factory ─────────────────────────────────────────────────────────

def _desired_dists(ref_shape, box_size):
    displacement, _ = space.periodic(box_size)
    vdisp = space.map_product(displacement)
    return jnp.sort(space.distance(vdisp(ref_shape, ref_shape)))


def make_loss_fns(cfg: DesignConfig, ref_shape):
    """
    Build the JIT-compiled loss / gradient closures for one design run against a
    fixed ``ref_shape`` (an ``(n, 2)`` array from :func:`base.geometry.get_shape`).

    Returns a namespace with ``sys_loss``, ``avg_loss``, ``get_mean_loss``,
    ``g_mean_loss`` and (optionally) ``h_mean_loss``.
    """
    box_size = cfg.box_size
    num_particles = cfg.num_particles
    ref_shape = jnp.asarray(ref_shape)
    ref_size = len(ref_shape)
    ref_dists = _desired_dists(ref_shape, box_size)

    displacement, _ = space.periodic(box_size)
    vdisp = space.map_product(displacement)
    v_subtract = space.map_product(lambda a, b: a - b)

    @jit
    def sys_loss(R):
        ds = jnp.sort(space.distance(vdisp(R, R)))
        diffs = v_subtract(ds[:, :ref_size], ref_dists)
        nearest = jnp.min(jnp.linalg.norm(diffs, axis=-1), axis=0)
        if num_particles > ref_size:
            other = ds[:, ref_size:CLOSENESS_PENALTY_NEIGHBORS + ref_size]
            return jnp.sum(nearest) - CLOSENESS_PENALTY * jnp.mean(other)
        return jnp.sum(nearest)

    v_loss = vmap(sys_loss)

    @jit
    def avg_loss(R_batched):
        return jnp.mean(v_loss(R_batched))

    def get_mean_loss(params, initial_position, keys):
        states, _, _ = v_run_my_sim(
            params, initial_position, cfg.sqrt_num_steps_to_opt,
            CENTER_RADIUS, box_size, keys,
        )
        return avg_loss(states.position.center)

    ns = SimpleNamespace(
        sys_loss=sys_loss,
        avg_loss=avg_loss,
        get_mean_loss=get_mean_loss,
        g_mean_loss=jit(value_and_grad(get_mean_loss)),
    )
    if cfg.find_hessian:
        ns.h_mean_loss = jit(hessian(get_mean_loss))
    return ns


# ── Parameter helpers ────────────────────────────────────────────────────

def make_params(opt_params, num_patches=NUM_PATCHES, param_lim=30.0):
    """Prepend the pinned reference patch (angle 0), clamp the opening angle to
    ``[0, pi]`` and the bond energies to ``[.., param_lim]``."""
    params = np.zeros(len(opt_params) + 1)
    params[1:] = np.asarray(opt_params)
    if params[1] < 0:
        params[1] = np.abs(params[1])
    elif params[1] > np.pi:
        params[1] = 2 * np.pi - params[1]
    for m in range(num_patches, len(params)):
        params[m] = min(params[m], param_lim)
    return jnp.array(params)


def _find_like(params, num_patches=NUM_PATCHES, buffer=0.9):
    likes = np.zeros(num_patches)
    like_pos = np.zeros(num_patches)
    for k in range(1, num_patches):
        like_pos[k] = like_pos[k - 1] + num_patches - k + 1
    likes_v, mixed_v = [], []
    for i in range(len(params)):
        (likes_v if i in like_pos else mixed_v).append(params[i])
    return all(x < y * buffer for x in likes_v for y in mixed_v)


def generate_random_params(key, num_patches, max_energy, rand_eng_check=True):
    n_eng = int(num_patches * (num_patches + 1) / 2)
    thetas = random.uniform(key, (num_patches,), minval=0.0, maxval=2 * jnp.pi)
    eng_ok = False
    while not eng_ok:
        key, split = random.split(key)
        energies = random.uniform(split, (n_eng,), minval=0.0, maxval=max_energy)
        eng_ok = rand_eng_check or _find_like(energies)
    thetas = thetas.at[0].set(0.0)
    return jnp.array(list(thetas) + list(energies))


# ── Simulation warm-starts (design-only) ─────────────────────────────────

def run_partial_sim(params, key, cfg: DesignConfig):
    rand_key, run_key = random.split(key, 2)
    from simulation.src.canonical_md import v_random_IC
    init_positions = v_random_IC(params, random.split(rand_key, cfg.batch_size),
                                 cfg.num_particles, cfg.box_size)
    sim_keys = random.split(run_key, cfg.batch_size)
    states, _, _ = v_run_my_sim(
        params, init_positions, cfg.quick_steps, CENTER_RADIUS, cfg.box_size, sim_keys
    )
    return states.position


def run_sim_and_get_positions(param, state, keys, cfg: DesignConfig):
    xf, _, _ = v_run_my_sim(
        param, state, cfg.sqrt_num_steps_to_opt, CENTER_RADIUS, cfg.box_size, keys
    )
    return xf.position, xf.position.center, xf.position.orientation


def save_position(param, state, keys, stepk, file_dir, cfg: DesignConfig):
    _, center, orientation = run_sim_and_get_positions(param, state, keys, cfg)
    pos_dir = os.path.join(file_dir, "Position_log")
    os.makedirs(pos_dir, exist_ok=True)
    np.save(os.path.join(pos_dir, f"pos{stepk}.npy"), np.asarray(center))
    np.save(os.path.join(pos_dir, f"angle{stepk}.npy"), np.asarray(orientation))


# ── Cluster counting during optimization ────────────────────────────────

def get_clusters(state, params, cl_type, box_size, allowance=PATCH_ALLOWANCE,
                 cluster_check=0.5):
    """
    Number of validated ``cl_type`` clusters in a single rigid-body ``state``,
    using freud on the patch point clouds + :func:`shape_match_loss` validation.
    """
    import freud

    n_patch = NUM_PATCHES
    center_pos, patches_pos = body_to_plot(state, params[:n_patch])
    center_pos = np.asarray(center_pos)
    ref_cl_size = {'square': 4, 'triangle': 3}.get(cl_type.lower())
    if ref_cl_size is None:
        return 0, set(), []
    ref_shape = get_shape(cl_type, opening_angle_rad=float(params[1]))

    patch_dists = abs(2 * CENTER_RADIUS * np.sin(float(params[1]) / 2.0))
    max_dist = patch_dists + allowance
    box = freud.box.Box(box_size, box_size, is2D=True)

    patch_clusters = []
    for k in range(n_patch):
        pts = np.hstack((np.asarray(patches_pos[k]),
                         np.zeros((len(patches_pos[k]), 1))))
        cl = freud.cluster.Cluster()
        cl.compute((box, pts), neighbors={'r_max': max_dist, 'r_min': 0.001})
        keys = np.array(cl.cluster_keys, dtype=object)
        patch_clusters.append(tuple(c for c in keys if len(c) == ref_cl_size))

    set_a = set(map(tuple, patch_clusters[0]))
    set_b = set(map(tuple, patch_clusters[1]))
    final = set()
    for cluster in set_a | set_b:
        idx = np.array(cluster, dtype=int)
        if shape_match_loss(center_pos[idx], ref_shape) < cluster_check:
            final.add(tuple(sorted(cluster)))
    return len(final), final, [list(c) for c in final]


def make_cluster_list(state, params, batch_size, cl_type, box_size):
    counts = np.zeros(batch_size)
    for i in range(batch_size):
        counts[i], _, _ = get_clusters(state[i], params, cl_type, box_size)
    return counts


# ── Random search + optimization loop ──────────────────────────────────

def random_search(key, n_iterations, cfg: DesignConfig, ref_shape):
    """Draw ``n_iterations`` random parameter sets, briefly simulate each, keep
    the lowest-loss one.  Returns ``(min_loss, best_params_without_pinned_patch)``."""
    loss = make_loss_fns(cfg, ref_shape)
    min_params = generate_random_params(key, cfg.num_patches, cfg.max_energy,
                                        cfg.rand_eng_check)
    key, split = random.split(key)
    sim_keys = random.split(split, cfg.batch_size)
    init = run_partial_sim(min_params, split, cfg)
    min_loss = loss.get_mean_loss(min_params, init, sim_keys)
    for _ in range(n_iterations - 1):
        params = generate_random_params(key, cfg.num_patches, cfg.max_energy,
                                        cfg.rand_eng_check)
        key, split = random.split(key)
        sim_keys = random.split(split, cfg.batch_size)
        init = run_partial_sim(min_params, split, cfg)
        l = loss.get_mean_loss(params, init, sim_keys)
        if l < min_loss:
            min_loss, min_params = l, params
    return min_loss, min_params[1:]


def optimize(input_params, key, opt_steps, cfg: DesignConfig, tracker,
             learning_rate=0.5, cmd='w', myoptsteps=1, last_step=0,
             cl_type=None, return_history=False):
    """
    Adam/RMSProp optimization of ``input_params`` (opening angle + bond
    energies) against the ``cl_type`` target shape.  Writes ``loss_*.txt`` /
    ``params_*.txt`` / ``grad_*.txt`` under ``cfg.job_dir``.

    Returns ``(min_loss, min_loss_params, cl_loss, max_cl_params, final_params,
    myoptsteps[, history])``.
    """
    ref_shape = get_shape(cl_type) if cl_type is not None else None
    loss = make_loss_fns(cfg, ref_shape) if ref_shape is not None else None

    start = time.time()
    lr_sched = jnp.ones(opt_steps) * learning_rate
    ind = int(opt_steps / 3)
    lr_sched = lr_sched.at[ind:2 * ind].set(learning_rate * 0.5)
    lr_sched = lr_sched.at[2 * ind:].set(learning_rate * 0.1)
    lr_fn = lambda i: lr_sched[i]

    if cfg.optimizer == "rms":
        opt_init, opt_update, get_params = optimizers.rmsprop(step_size=lr_fn, gamma=0.9, eps=1e-8)
    else:
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr_fn)
    print(f"Optimizing using the {cfg.optimizer} algorithm.")

    loss_file = os.path.join(cfg.job_dir, f'loss_{tracker[-1]}.txt')
    param_file = os.path.join(cfg.job_dir, f'params_{tracker[-1]}.txt')
    grad_file = os.path.join(cfg.job_dir, f'grad_{tracker[-1]}.txt')
    hess_file = os.path.join(cfg.job_dir, f'hes_{tracker[-1]}.txt')

    def clip_gradient(g):
        return jnp.where(jnp.abs(g) > cfg.clip, jnp.sign(g) * cfg.clip, g)

    def step(stepk, opt_state, key, cmd):
        run_params = make_params(get_params(opt_state), cfg.num_patches, cfg.max_energy)
        key, split = random.split(key)
        sim_keys = random.split(split, cfg.batch_size)
        init_positions = run_partial_sim(run_params, split, cfg)

        gs, Hs, ls = [], [], 0.0
        for _ in range(cfg.loop_batch):
            l, g = loss.g_mean_loss(run_params, init_positions, sim_keys)
            if cfg.find_hessian:
                Hs.append(loss.h_mean_loss(run_params, init_positions, sim_keys))
            gs.append(clip_gradient(g))
            ls += l
        g = jnp.mean(jnp.array(gs), axis=0)
        loss_val = ls / cfg.loop_batch

        position, _, _ = run_sim_and_get_positions(run_params, init_positions, sim_keys, cfg)
        batch_shapes = make_cluster_list(position, run_params, cfg.batch_size,
                                         cl_type, cfg.box_size)
        step_shapes = float(np.sum(batch_shapes))

        if stepk % cfg.write_every == 0:
            save_position(run_params, init_positions, sim_keys, stepk + myoptsteps,
                          cfg.job_dir, cfg)
            with open(loss_file, cmd) as f:
                f.write(f"{loss_val}\n")
            with open(param_file, cmd) as f:
                f.write(" ".join(f"{v}" for v in np.array(run_params).tolist()) + "\n")
            with open(grad_file, cmd) as f:
                f.write(" ".join(f"{v}" for v in np.array(g).tolist()) + "\n")
            if cfg.find_hessian:
                Hes = jnp.mean(jnp.array(Hs), axis=0)
                with open(hess_file, cmd) as f:
                    f.write(" ".join(f"{v}" for v in np.array(Hes).tolist()) + "\n")

        return opt_update(stepk, g[1:], opt_state), loss_val, step_shapes

    print(f"Input params are {input_params}")
    opt_state = opt_init(input_params)
    min_loss_params = max_cl_params = input_params
    min_loss = cl_loss = 1e6
    max_cl = 0
    history = {'loss': [], 'params': [], 'grads': []} if return_history else None

    for i in range(last_step, opt_steps):
        key, split = random.split(key)
        new_opt_state, loss_val, step_cl = step(i, opt_state, split, cmd)
        if (i + 1) % cfg.write_every == 0:
            print(f"Step {i + 1}/{opt_steps} | Loss: {loss_val:.6f} | "
                  f"Elapsed: {int((time.time() - start) // 60)}min")
        if loss_val < min_loss:
            min_loss, min_loss_params = loss_val, get_params(opt_state)
        if step_cl > max_cl:
            max_cl, cl_loss, max_cl_params = step_cl, loss_val, get_params(opt_state)
        if return_history:
            history['loss'].append(float(loss_val))
            history['params'].append(np.array(make_params(get_params(opt_state),
                                                          cfg.num_patches, cfg.max_energy)))
        opt_state = new_opt_state
        cmd = 'a'

    myoptsteps += opt_steps
    print(f"This optimization took {int((time.time() - start) // 60)} minutes.")
    result = (min_loss, min_loss_params, cl_loss, max_cl_params,
              get_params(opt_state), myoptsteps)
    if return_history:
        return (*result, {k: np.array(v) for k, v in history.items()})
    return result
