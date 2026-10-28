"""
Energy functions for 2-patch patchy particles.

Contents:

0. Numpy/JAX-agnostic scalar-field primitives reused by every energy below:
   :func:`minimum_image`, :func:`morse_well`, :func:`soft_sphere_repulsion`,
   :func:`smooth_cutoff_weight`.

1. :func:`total_energy` — a pure-JAX, **free-space** (no PBC) pairwise energy of
   ``n`` particles whose state is the flat vector ``q`` of shape ``(3n,)``.
   Differentiable in ``q``, ``E_AB`` and ``alpha_morse``.  Used by the ring
   minimisation.  (Same smooth Morse window as every other energy.)

2. Periodic, ``params``-dict energies used by the hybrid GCMC engine:
   :func:`particle_energy` (O(N), numpy — for incremental MC moves),
   :func:`total_energy_pbc` (O(N^2) diagnostic), and :func:`total_energy_jax`
   (differentiable, for BAOAB forces).  These share :func:`_patch_morse` and use
   the smooth C1 Morse window (``r_onset`` .. ``r_cutoff``).

3. :func:`build_pair_energy_fn` — the single JAX-MD ``rigid_body.point_energy``
   builder (soft-sphere core repulsion + Morse patch attraction) shared by every
   canonical-ensemble MD driver.
"""

import numpy as np
import jax.numpy as jnp
from jax_md import space, energy as jmd_energy, rigid_body

from base.config import ALPHA, MORSE_R_ONSET, MORSE_R_CUTOFF, CENTER_RADIUS
from base.geometry import thetas_to_shape, patch_positions


# ══════════════════════════════════════════════════════════════════════════
# 0. Shared scalar-field primitives  (numpy- or jax-array agnostic)
# ══════════════════════════════════════════════════════════════════════════

def _xp(a):
    return jnp if isinstance(a, jnp.ndarray) else np


def minimum_image(dr, L):
    """Minimum-image displacement for any shape of ``dr`` (numpy or jax)."""
    return dr - L * _xp(dr).round(dr / L)


def morse_well(r, eps, alpha):
    """Raw Morse potential ``eps * (1 - exp(-alpha r))**2 - eps``."""
    return eps * (1.0 - _xp(r).exp(-alpha * r)) ** 2 - eps


def soft_sphere_repulsion(r, sigma, prefactor):
    """Truncated soft-sphere core repulsion ``prefactor * (1 - r/sigma)**2`` for
    ``r < sigma`` (0 beyond)."""
    return _xp(r).where(r < sigma, prefactor * (1.0 - r / sigma) ** 2, 0.0)


def smooth_cutoff_weight(r, r_onset, r_cutoff):
    """C1 multiplicative window identical to jax-md's
    ``multiplicative_isotropic_cutoff``: 1 below ``r_onset``, 0 from
    ``r_cutoff`` on, smooth in between (ADR-0001)."""
    xp = _xp(r)
    r2, c2, o2 = r ** 2, r_cutoff ** 2, r_onset ** 2
    inner = xp.where(
        r < r_cutoff, (c2 - r2) ** 2 * (c2 + 2.0 * r2 - 3.0 * o2) / (c2 - o2) ** 3, 0.0
    )
    return xp.where(r < r_onset, 1.0, inner)


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
    r_onset=MORSE_R_ONSET,
    r_cutoff=MORSE_R_CUTOFF,
):
    """
    Total pairwise interaction energy of ``n`` 2-patch patchy particles in free
    space.  ``q`` is ``[x1, y1, theta1, x2, y2, theta2, ...]``.
    """
    n = q.shape[0] // 3
    q_r = q.reshape(n, 3)
    centers = q_r[:, :2]
    thetas = q_r[:, 2]

    pA, pB = patch_positions(centers, thetas, opening_angle, r_patch)

    i_idx, j_idx = jnp.triu_indices(n, k=1)

    r_cc = jnp.sqrt(jnp.sum((centers[i_idx] - centers[j_idx]) ** 2, axis=-1))
    v_rep = soft_sphere_repulsion(r_cc, 2.0 * r_patch, rep_A)

    def morse_pair(pos_i, pos_j, eps):
        r = jnp.sqrt(jnp.sum((pos_i - pos_j) ** 2, axis=-1))
        return smooth_cutoff_weight(r, r_onset, r_cutoff) * morse_well(r, eps, alpha_morse)

    v_AA = morse_pair(pA[i_idx], pA[j_idx], E_AA)
    v_BB = morse_pair(pB[i_idx], pB[j_idx], E_BB)
    v_AB = morse_pair(pA[i_idx], pB[j_idx], E_AB)
    v_BA = morse_pair(pB[i_idx], pA[j_idx], E_AB)
    return jnp.sum(v_rep + v_AA + v_BB + v_AB + v_BA)


# ══════════════════════════════════════════════════════════════════════════
# 2. Periodic, params-dict energies  (q: (N, 3))
# ══════════════════════════════════════════════════════════════════════════

def _patch_morse(pi, pj, L, eps, alpha, r_onset, r_cutoff):
    """Smooth-cutoff Morse energy between two aligned sets of patch positions,
    minimum-image applied."""
    r = _xp(pi).sqrt(_xp(pi).sum(minimum_image(pi - pj, L) ** 2, axis=-1))
    return smooth_cutoff_weight(r, r_onset, r_cutoff) * morse_well(r, eps, alpha)


def particle_energy(i, q, L, params):
    """Energy of particle ``i`` with every other particle (numpy, O(N))."""
    N = len(q)
    if N <= 1:
        return 0.0

    mask = np.ones(N, dtype=bool)
    mask[i] = False
    q_j = q[mask]
    oa, rp = params['opening_angle'], params['r_patch']

    r_cc = np.sqrt(np.sum(minimum_image(q[i, :2][None, :] - q_j[:, :2], L) ** 2, axis=-1))
    v_rep = soft_sphere_repulsion(r_cc, 2.0 * rp, params['rep_A'])

    (pAi, pBi) = patch_positions(q[i:i + 1, :2], q[i:i + 1, 2], oa, rp)
    pAj, pBj = patch_positions(q_j[:, :2], q_j[:, 2], oa, rp)
    pAi, pBi = pAi[0], pBi[0]

    def m(pi, pj, eps):
        return _patch_morse(pi, pj, L, eps, params['alpha'],
                            params['r_onset'], params['r_cutoff'])

    return float(np.sum(
        v_rep
        + m(pAi, pAj, params['E_AA'])
        + m(pBi, pBj, params['E_BB'])
        + m(pAi, pBj, params['E_AB'])
        + m(pBi, pAj, params['E_AB'])
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
    oa, rp = params['opening_angle'], params['r_patch']

    pA, pB = patch_positions(c, q[:, 2], oa, rp)
    i_idx, j_idx = jnp.triu_indices(N, k=1)

    def morse(pi, pj, eps):
        return _patch_morse(pi, pj, L, eps, params['alpha'],
                            params['r_onset'], params['r_cutoff'])

    r_cc = jnp.sqrt(jnp.sum(minimum_image(c[i_idx] - c[j_idx], L) ** 2, axis=-1))
    v_rep = soft_sphere_repulsion(r_cc, 2.0 * rp, params['rep_A'])
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
    r_onset=MORSE_R_ONSET,
    r_cutoff=MORSE_R_CUTOFF,
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
    alpha_morse, r_onset, r_cutoff, center_radius : Morse / geometry parameters

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
        sigma=0.0, epsilon=morse_eps, alpha=alpha_morse, r_onset=r_onset, r_cutoff=r_cutoff,
    )
    pair_fn = lambda R, **kw: pair_soft(R, **kw) + pair_morse(R, **kw)

    shape = thetas_to_shape(thetas, radius=center_radius)
    return rigid_body.point_energy(pair_fn, shape), shift
