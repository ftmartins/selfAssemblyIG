"""
Patchy-particle geometry primitives.

Canonical patch convention (matches ``thetas_to_shape`` and the JAX-MD
rigid-body energy):

    pA_i = center_i + r_patch * [cos(theta_i),              sin(theta_i)]
    pB_i = center_i + r_patch * [cos(theta_i + opening_angle), sin(theta_i + opening_angle)]

i.e. patch A sits at body-frame angle 0 and patch B at body-frame angle
``opening_angle``.  The older symmetric ``theta ± opening_angle/2`` convention
is **not** used anywhere in this package.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax_md import rigid_body

from base.config import CENTER_RADIUS, CENTER_MASS, PATCH_MASS, NUM_PATCHES


# ── Rigid-body shape construction ───────────────────────────────────────────

@jit
def thetas_to_shape(thetas, radius=CENTER_RADIUS):
    """Point-union rigid-body shape: a central core plus one point per patch."""
    patch_positions = jnp.zeros((len(thetas), 2), dtype=jnp.float64)
    patch_positions = patch_positions.at[:, 0].set(radius * jnp.cos(thetas))
    patch_positions = patch_positions.at[:, 1].set(radius * jnp.sin(thetas))
    positions = jnp.concatenate((jnp.array([[0.0, 0.0]]), patch_positions), axis=0)
    species = jnp.arange(len(thetas) + 1, dtype=jnp.int32)
    patch_mass = PATCH_MASS * jnp.ones(len(thetas))
    mass = jnp.concatenate((jnp.array([CENTER_MASS]), patch_mass), axis=0)
    return rigid_body.point_union_shape(positions, mass).set(point_species=species)


@jit
def energy_matrix(eng):
    """Symmetric (NUM_PATCHES, NUM_PATCHES) matrix from the upper-triangular
    entries ``eng`` (ordered by ``jnp.triu_indices``)."""
    i, j = jnp.triu_indices(NUM_PATCHES)
    eng_m = jnp.zeros((NUM_PATCHES, NUM_PATCHES))
    eng_m = eng_m.at[i, j].set(eng)
    return eng_m.at[j, i].set(eng)


def body_to_plot(body, thetas, radius=CENTER_RADIUS):
    """Return ``(center_positions, [patch0_positions, patch1_positions, ...])``
    for a :class:`jax_md.rigid_body.RigidBody` with all particles the same shape."""
    shape = thetas_to_shape(thetas, radius)
    body_pos = vmap(rigid_body.transform, (0, None))(body, shape)
    bodypos = body_pos.reshape(-1, 2)

    species_list = jnp.array(list(jnp.arange(NUM_PATCHES + 1)) * len(body.center))
    inds_at_id = lambda i: jnp.squeeze(jnp.argwhere(species_list == i))

    center_particles = bodypos[inds_at_id(0)]
    patch_particles = [bodypos[inds_at_id(i)] for i in range(1, len(thetas) + 1)]
    return center_particles, patch_particles


# ── Patch positions in the lab frame ───────────────────────────────────────

def patch_positions(centers, thetas, opening_angle, r_patch=CENTER_RADIUS):
    """
    Lab-frame patch A and patch B positions for every particle.

    Works for both numpy and jax arrays (uses the caller's array module).

    Parameters
    ----------
    centers : (N, 2) array
    thetas : (N,) array — body orientation (= direction of patch A)
    opening_angle : float — angle between patch A and patch B
    r_patch : float — patch arm length

    Returns
    -------
    (pA, pB) : each (N, 2)
    """
    xp = jnp if isinstance(centers, jnp.ndarray) else np
    th = thetas
    pA = centers + r_patch * xp.stack([xp.cos(th), xp.sin(th)], axis=-1)
    pB = centers + r_patch * xp.stack(
        [xp.cos(th + opening_angle), xp.sin(th + opening_angle)], axis=-1
    )
    return pA, pB


def get_patches(q, r_patch, opening_angle):
    """``patch_positions`` for a state array ``q`` of shape (N, 3) = [x, y, theta]."""
    return patch_positions(q[:, :2], q[:, 2], opening_angle, r_patch)


# ── Reference n-gon geometry ───────────────────────────────────────────────

def regular_ngon_reference(n, R=1.0, opening_angle_rad=None):
    """
    Reference configuration for an n-gon of 2-patch particles.

    Returns
    -------
    (positions, orientations, patch_angles)
        positions     : (n, 2) vertices of a regular n-gon of circumradius R
        orientations   : (n,)  body orientation of each particle
        patch_angles   : (n, 2) body-frame patch angles, [0, opening_angle]
    """
    if n < 3:
        raise ValueError("Need at least a triangle (n >= 3).")
    if opening_angle_rad is None:
        opening_angle_rad = np.deg2rad(100.0)
    alpha = opening_angle_rad

    vertex_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.stack([R * np.cos(vertex_angles), R * np.sin(vertex_angles)], axis=1)

    orientations = np.zeros(n)
    patch_angles = np.zeros((n, 2))
    for i in range(n):
        # Evenly spaced orientations (works for square and triangle rings).
        orientations[i] = -np.pi - alpha / 2 + 2 * np.pi * i / n
        patch_angles[i, 0] = 0.0     # patch A in body frame
        patch_angles[i, 1] = alpha   # patch B in body frame
    return positions, orientations, patch_angles


SHAPE_CONFIGS = {
    'square': {
        'n_vertices': 4,
        'default_opening_angle': 100.0,   # degrees, reference only — not fixed
        'description': '2-patch square assembly',
    },
    'triangle': {
        'n_vertices': 3,
        'default_opening_angle': 60.0,    # matching angle pi(n-2)/n for n=3
        'description': '2-patch triangle assembly',
    },
}

# n-gon name -> vertex count, for polygon detection / reference shapes
NGON_SIZES = {
    'triangle': 3, 'square': 4, 'pentagon': 5,
    'hexagon': 6, 'heptagon': 7, 'octagon': 8,
}


def patch_separation(opening_angle, r=CENTER_RADIUS):
    """Lab-frame distance between a particle's two patches: ``2 r |sin(alpha/2)|``."""
    return 2.0 * r * abs(np.sin(opening_angle / 2.0))


def get_shape(shape_id, opening_angle_rad=None):
    """
    Reference-shape vertex positions ``(n, 2)`` for ``shape_id`` (a polygon name
    such as ``'square'`` or ``'triangle'``).  The circumradius is chosen so the
    center-to-center edge length equals ``2 * CENTER_RADIUS``.
    """
    key = shape_id.lower()
    n = NGON_SIZES.get(key)
    if n is None:
        raise ValueError(f"Unknown shape_id: {shape_id!r}")
    radius = 1.0 / np.sin(np.pi / n)
    positions, _, _ = regular_ngon_reference(n, R=radius, opening_angle_rad=opening_angle_rad)
    return positions
