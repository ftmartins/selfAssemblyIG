"""
Pure JAX energy function for n 2-patch patchy particles in free space.

Canonical patch convention (matches thetas_to_shape in modules/utility_functions.py):
    pA_i = center_i + r_patch * [cos(theta_i),              sin(theta_i)]
    pB_i = center_i + r_patch * [cos(theta_i + opening_angle), sin(theta_i + opening_angle)]

State vector q has shape (3*n,): [x1, y1, theta1, x2, y2, theta2, ..., xn, yn, thetan].
No periodic boundary conditions (free space).
"""

import jax.numpy as jnp
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_patchy_particle import PATCH_SIZE


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
    Total pairwise interaction energy of n 2-patch patchy particles.

    Parameters
    ----------
    q : array (3*n,)
        Flat DoF vector [x1, y1, theta1, x2, y2, theta2, ...].
    opening_angle : float
        Full angle between patch A and patch B (radians).
    E_AB : float
        Morse attraction strength for A-B patch pairs.
    alpha_morse : float
        Morse decay parameter (codebase default: ALPHA = 7).
    E_AA : float
        Morse attraction strength for A-A patch pairs (default 1.0).
    E_BB : float
        Morse attraction strength for B-B patch pairs (default 1.0).
    r_patch : float
        Patch arm length / CENTER_RADIUS (default 1.0).
    rep_A : float
        Soft-sphere repulsion prefactor, center-center only (default 10.0).
    patch_sigma : float or None
        Morse interaction cutoff radius. Defaults to PATCH_SIZE from config.

    Returns
    -------
    float
        Total potential energy.
    """
    if patch_sigma is None:
        patch_sigma = PATCH_SIZE

    n = q.shape[0] // 3
    q_r = q.reshape(n, 3)
    centers = q_r[:, :2]       # (n, 2)
    thetas = q_r[:, 2]          # (n,)

    # Patch positions in lab frame — canonical (theta, theta+alpha) convention
    pA = centers + r_patch * jnp.stack([jnp.cos(thetas),
                                         jnp.sin(thetas)], axis=-1)          # (n, 2)
    pB = centers + r_patch * jnp.stack([jnp.cos(thetas + opening_angle),
                                         jnp.sin(thetas + opening_angle)], axis=-1)  # (n, 2)

    # Pairwise indices i < j
    i_idx, j_idx = jnp.triu_indices(n, k=1)   # each array has length n*(n-1)/2

    # --- Center-center soft-sphere repulsion ---
    sigma_cc = 2.0 * r_patch   # = 2.0 when r_patch=1
    dr_cc = centers[i_idx] - centers[j_idx]           # (npairs, 2)
    r_cc = jnp.sqrt(jnp.sum(dr_cc ** 2, axis=-1))     # (npairs,)
    v_rep = rep_A * (1.0 - r_cc / sigma_cc) ** 2
    v_rep = jnp.where(r_cc < sigma_cc, v_rep, 0.0)

    # --- Patch-patch Morse attraction ---
    def morse_pair(pos_i, pos_j, eps):
        """Morse energy between two sets of patch positions, element-wise."""
        dr = pos_i - pos_j                                  # (npairs, 2)
        r = jnp.sqrt(jnp.sum(dr ** 2, axis=-1))            # (npairs,)
        v = eps * (1.0 - jnp.exp(-alpha_morse * r)) ** 2 - eps
        return jnp.where(r < patch_sigma, v, 0.0)

    v_AA = morse_pair(pA[i_idx], pA[j_idx], E_AA)
    v_BB = morse_pair(pB[i_idx], pB[j_idx], E_BB)
    v_AB = morse_pair(pA[i_idx], pB[j_idx], E_AB)
    v_BA = morse_pair(pB[i_idx], pA[j_idx], E_AB)

    return jnp.sum(v_rep + v_AA + v_BB + v_AB + v_BA)
