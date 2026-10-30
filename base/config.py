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
# Patch radius a: half the patch-patch distance at which the Morse attraction
# falls to 1% of its depth (CONTEXT.md).  Also the bond distance (ADR-0002).
PATCH_RADIUS = float(np.log(1.0 - np.sqrt(0.99)) / (-ALPHA) / 2.0)
PATCH_SIZE = PATCH_RADIUS         # legacy alias of PATCH_RADIUS

# Morse bond range (ADR-0001): full Morse below the onset, C1 window to zero at
# the cutoff, nothing beyond.  Both are passed explicitly to every engine.
MORSE_R_ONSET = 1.5
MORSE_R_CUTOFF = 2.0

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
