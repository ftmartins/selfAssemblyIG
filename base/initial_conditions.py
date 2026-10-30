"""
Initial-condition generation for 2-D patchy-particle systems.

One Random-Sequential-Addition implementation (:func:`rsa_2d`) replaces the four
near-duplicates that used to live in ``modules/utility_functions.py``
(``random_IC`` / ``random_IC_nonoverlap``),
``run_yield_simulation_cluster.py`` (``generate_nonoverlap_ic``) and
``modules/gcmc_hybrid.py`` (``generate_ic``).

RSA here is plain numpy (it is inherently sequential and not JIT-able).  Callers
that need JAX arrays or a :class:`jax_md.rigid_body.RigidBody` wrap the output;
the GCMC engine adds Maxwell-Boltzmann velocities.
"""

import numpy as np

from base.config import CENTER_RADIUS, PATCH_SIZE


def _minimum_image(dr, L):
    return dr - L * np.round(dr / L)


def default_min_distance() -> float:
    """Center-to-center exclusion used by the RSA packing:
    ``2 * CENTER_RADIUS + 2 * PATCH_SIZE``."""
    return 2 * CENTER_RADIUS + 2 * PATCH_SIZE


def rsa_2d(
    n: int,
    box_size: float,
    min_distance: float = None,
    *,
    seed: int = None,
    rng: np.random.Generator = None,
    centered: bool = False,
    max_attempts: int = 20_000,
    verbose: bool = False,
):
    """
    Place ``n`` non-overlapping particles in a periodic square box by Random
    Sequential Addition.

    Parameters
    ----------
    n : int
    box_size : float — periodic box side length ``L``
    min_distance : float, optional — center-to-center exclusion (default
        :func:`default_min_distance`)
    seed / rng : reproducibility (``rng`` wins if both given)
    centered : bool — if True, coordinates in ``[-L/2, L/2)`` (GCMC convention);
        otherwise ``[0, L)`` (JAX-MD convention)
    max_attempts : int — rejections allowed per particle before raising

    Returns
    -------
    (positions, orientations)
        positions : (n, 2) float64
        orientations : (n,) float64, uniform in ``[0, 2 pi)``
    """
    if min_distance is None:
        min_distance = default_min_distance()
    if rng is None:
        rng = np.random.default_rng(seed)

    lo = -box_size / 2 if centered else 0.0
    hi = box_size / 2 if centered else box_size

    pts = np.empty((n, 2), dtype=np.float64)
    count = 0
    for i in range(n):
        for _ in range(max_attempts):
            cand = rng.uniform(lo, hi, size=2)
            if count == 0:
                break
            dr = _minimum_image(cand[None, :] - pts[:count], box_size)
            if np.all(np.sqrt(np.sum(dr ** 2, axis=-1)) >= min_distance):
                break
        else:
            raise RuntimeError(
                f"RSA failed to place particle {i + 1}/{n} after {max_attempts} "
                f"attempts (box too small / min_distance too large)."
            )
        pts[i] = cand
        count += 1
        if verbose and (i + 1) % 100 == 0:
            print(f"  RSA placed {i + 1}/{n}")

    orientations = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return pts, orientations


def random_uniform_ic(n: int, box_size: float, *, seed: int = None,
                      rng: np.random.Generator = None, centered: bool = False):
    """Uniform (possibly overlapping) positions + orientations — for high-T
    starts where RSA packing is unnecessary."""
    if rng is None:
        rng = np.random.default_rng(seed)
    lo = -box_size / 2 if centered else 0.0
    hi = box_size / 2 if centered else box_size
    positions = rng.uniform(lo, hi, size=(n, 2))
    orientations = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return positions, orientations
