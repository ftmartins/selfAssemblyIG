"""
Physical constants for the 2-patch patchy-particle model.

This module is a leaf: it imports only numpy and has **no import-time side
effects** (no prints, no JAX configuration, no derived global state such as
``NUM_PARTICLES`` / ``BOX_SIZE``).  Anything that depends on the size of a
particular system (particle count, box size, reference shape) is passed
explicitly as a function argument by the caller.

JAX 64-bit mode is configured separately and explicitly via
``base.jax_setup.configure_jax`` — importing this module does not touch JAX.
"""

import numpy as np

# ── Particle geometry ────────────────────────────────────────────────────────
NUM_PATCHES = 2
CENTER_RADIUS = 1.0
CENTER_MASS = 1.0
PATCH_MASS = 0.0

# ── Interaction parameters ───────────────────────────────────────────────────
ALPHA = 7.0                       # Morse potential decay parameter
DT = 1e-3                         # default MD time step
PATCH_SIZE = float(np.log(1.0 - np.sqrt(0.99)) / (-ALPHA) / 2.0)
R_CUTOFF = PATCH_SIZE             # Morse cutoff radius

# ── Thermodynamics ──────────────────────────────────────────────────────────
kT = 1.0

# ── Default system density (area fraction) ──────────────────────────────────
DENSITY = 0.1


def get_box_size(phi: float, n: int, r: float = CENTER_RADIUS) -> float:
    """
    Side length of a square box holding ``n`` disks of radius ``r`` at area
    fraction ``phi``:  ``L = sqrt(n * pi * r**2 / phi)``.
    """
    return float(np.sqrt(n * np.pi * r ** 2 / phi))
