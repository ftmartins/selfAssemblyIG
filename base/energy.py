"""
Energy functions for 2-patch patchy particles.

Three flavours live here:

1. :func:`total_energy` — a pure-JAX, **free-space** (no PBC) pairwise energy of
   ``n`` particles whose state is the flat vector ``q`` of shape ``(3n,)``.
   Differentiable in ``q``, ``E_AB`` and ``alpha_morse``.  Used by the ring
   minimisation.

2. Periodic, ``params``-dict energies used by the hybrid GCMC engine:
   :func:`minimum_image`, :func:`particle_energy` (O(N), numpy — for incremental
   MC moves), :func:`total_energy_pbc` (O(N^2) diagnostic), and
   :func:`total_energy_jax` (differentiable, for BAOAB forces).

3. :func:`build_pair_energy_fn` — the single JAX-MD ``rigid_body.point_energy``
   builder (soft-sphere core repulsion + Morse patch attraction) shared by every
   canonical-ensemble MD driver.
"""

import numpy as np
import jax.numpy as jnp
from jax_md import space, energy as jmd_energy, rigid_body

from base.config import PATCH_SIZE, ALPHA, R_CUTOFF, CENTER_RADIUS
from base.geometry import thetas_to_shape


# ══════════════════════════════════════════════════════════════════════════
# 1. Free-space JAX energy  (q: (3n,))
# ══════════════════════════════════════════════════════════════════════════

def total_energy(
    q,
    opening_angle,
    E_AB,
    alpha_morse,
    E_AA=1.0,
    E_BB=1.0,
    r_patch=1.0,
    rep_A=10.0,
    patch_sigma=None,
):
    """
    Total pairwise interaction energy of ``n`` 2-patch patchy particles in free
    space.  ``q`` is ``[x1, y1, theta1, x2, y2, theta2, ...]``.
    """
    if patch_sigma is None:
        patch_sigma = PATCH_SIZE

    n = q.shape[0] // 3
    q_r = q.reshape(n, 3)
    centers = q_r[:, :2]
    thetas = q_r[:, 2]

    pA = centers + r_patch * jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)
    pB = centers + r_patch * jnp.stack(
        [jnp.cos(thetas + opening_angle), jnp.sin(thetas + opening_angle)], axis=-1
    )

    i_idx, j_idx = jnp.triu_indices(n, k=1)

    sigma_cc = 2.0 * r_patch
    dr_cc = centers[i_idx] - centers[j_idx]
    r_cc = jnp.sqrt(jnp.sum(dr_cc ** 2, axis=-1))
    v_rep = rep_A * (1.0 - r_cc / sigma_cc) ** 2
    v_rep = jnp.where(r_cc < sigma_cc, v_rep, 0.0)

    def morse_pair(pos_i, pos_j, eps):
        dr = pos_i - pos_j
        r = jnp.sqrt(jnp.sum(dr ** 2, axis=-1))
        v = eps * (1.0 - jnp.exp(-alpha_morse * r)) ** 2 - eps
        return jnp.where(r < patch_sigma, v, 0.0)

    v_AA = morse_pair(pA[i_idx], pA[j_idx], E_AA)
    v_BB = morse_pair(pB[i_idx], pB[j_idx], E_BB)
    v_AB = morse_pair(pA[i_idx], pB[j_idx], E_AB)
    v_BA = morse_pair(pB[i_idx], pA[j_idx], E_AB)
    return jnp.sum(v_rep + v_AA + v_BB + v_AB + v_BA)


# ══════════════════════════════════════════════════════════════════════════
# 2. Periodic, params-dict energies  (q: (N, 3))
# ══════════════════════════════════════════════════════════════════════════

def minimum_image(dr, L):
    """Minimum-image displacement for any shape of ``dr`` (numpy or jax)."""
    xp = jnp if isinstance(dr, jnp.ndarray) else np
    return dr - L * xp.round(dr / L)


def _smooth_morse_weight(r, patch_sigma):
    """C1 multiplicative cutoff window matching jax-md's
    ``multiplicative_isotropic_cutoff`` with ``r_onset = 0``."""
    rc2 = patch_sigma ** 2
    r2 = r ** 2
    return np.where(
        r < patch_sigma,
        (rc2 - r2) ** 2 * (rc2 + 2.0 * r2) / rc2 ** 3,
        0.0,
    )


def particle_energy(i, q, L, params):
    """Energy of particle ``i`` with every other particle (numpy, O(N))."""
    N = len(q)
    if N <= 1:
        return 0.0

    mask = np.ones(N, dtype=bool)
    mask[i] = False
    q_j = q[mask]

    ci = q[i, :2]
    cj = q_j[:, :2]
    dr_cc = minimum_image(ci[None, :] - cj, L)
    r_cc = np.sqrt(np.sum(dr_cc ** 2, axis=-1))
    sigma_cc = 2.0 * params['r_patch']
    v_rep = np.where(
        r_cc < sigma_cc,
        params['rep_A'] * (1.0 - r_cc / sigma_cc) ** 2,
        0.0,
    )

    thi = q[i, 2]
    thj = q_j[:, 2]
    oa = params['opening_angle']
    rp = params['r_patch']
    pAi = ci + rp * np.array([np.cos(thi), np.sin(thi)])
    pBi = ci + rp * np.array([np.cos(thi + oa), np.sin(thi + oa)])
    pAj = cj + rp * np.stack([np.cos(thj), np.sin(thj)], axis=-1)
    pBj = cj + rp * np.stack([np.cos(thj + oa), np.sin(thj + oa)], axis=-1)

    def morse(pi, pj_arr, eps):
        dr = minimum_image(pi[None, :] - pj_arr, L)
        r = np.sqrt(np.sum(dr ** 2, axis=-1))
        v = eps * (1.0 - np.exp(-params['alpha'] * r)) ** 2 - eps
        return _smooth_morse_weight(r, params['patch_sigma']) * v

    return float(np.sum(
        v_rep
        + morse(pAi, pAj, params['E_AA'])
        + morse(pBi, pBj, params['E_BB'])
        + morse(pAi, pBj, params['E_AB'])
        + morse(pBi, pAj, params['E_AB'])
    ))


def total_energy_pbc(q, L, params):
    """O(N^2) full-system periodic energy (numpy).  Diagnostics only."""
    if len(q) == 0:
        return 0.0
    return 0.5 * sum(particle_energy(i, q, L, params) for i in range(len(q)))


def total_energy_jax(q_flat, L, params):
    """Pure-JAX periodic energy, ``q_flat`` shape ``(3N,)``.  Mirrors
    :func:`total_energy_pbc`; used via ``jax.grad`` for BAOAB forces."""
    N = q_flat.shape[0] // 3
    q = q_flat.reshape(N, 3)
    c = q[:, :2]
    th = q[:, 2]
    oa = params['opening_angle']
    rp = params['r_patch']

    pA = c + rp * jnp.stack([jnp.cos(th), jnp.sin(th)], axis=-1)
    pB = c + rp * jnp.stack([jnp.cos(th + oa), jnp.sin(th + oa)], axis=-1)

    i_idx, j_idx = jnp.triu_indices(N, k=1)

    def mi(dr):
        return dr - L * jnp.round(dr / L)

    def smooth_weight(r):
        rc2 = params['patch_sigma'] ** 2
        r2 = r ** 2
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
    r_cc = jnp.sqrt(jnp.sum(dr_cc ** 2, axis=-1))
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


# ══════════════════════════════════════════════════════════════════════════
# 3. JAX-MD rigid-body pair energy (canonical MD)
# ══════════════════════════════════════════════════════════════════════════

def build_pair_energy_fn(
    thetas,
    eng_mat,
    box_size,
    *,
    rep_A,
    alpha_morse=ALPHA,
    r_cutoff=R_CUTOFF,
    center_radius=CENTER_RADIUS,
):
    """
    Build the periodic JAX-MD rigid-body energy function: soft-sphere core
    repulsion (core-core only, prefactor ``rep_A``) plus Morse patch attraction
    with per-patch-pair well depths ``eng_mat``.

    Parameters
    ----------
    thetas : (n_patch,) array — body-frame patch angles ``[0, opening_angle]``
    eng_mat : (n_patch, n_patch) array — symmetric Morse well-depth matrix
    box_size : float — periodic box side length
    rep_A : float — soft-sphere repulsion prefactor
    alpha_morse, r_cutoff, center_radius : Morse / geometry parameters

    Returns
    -------
    (energy_fn, shift_fn)
        ``energy_fn(rigid_body_position) -> scalar`` and the periodic
        ``shift_fn`` for the integrator.
    """
    displacement, shift = space.periodic(box_size)
    n_species = 1 + len(thetas)

    morse_eps = jnp.pad(eng_mat, pad_width=(1, 0))
    soft_sphere_eps = jnp.zeros((n_species, n_species)).at[0, 0].set(rep_A)

    pair_soft = jmd_energy.soft_sphere_pair(
        displacement, species=n_species,
        sigma=center_radius * 2, epsilon=soft_sphere_eps,
    )
    pair_morse = jmd_energy.morse_pair(
        displacement, species=n_species,
        sigma=0.0, epsilon=morse_eps, alpha=alpha_morse, r_cutoff=r_cutoff,
    )
    pair_fn = lambda R, **kw: pair_soft(R, **kw) + pair_morse(R, **kw)

    shape = thetas_to_shape(thetas, radius=center_radius)
    return rigid_body.point_energy(pair_fn, shape), shift
