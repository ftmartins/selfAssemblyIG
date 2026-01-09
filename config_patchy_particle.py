"""
Configuration file for patchy particle optimization and yield simulations.

All global constants and parameters are defined here.
"""

import numpy as np
import sys
import jax.numpy as jnp
from jax_md import space
from jax import random

# ============================================================================
# JAX CONFIGURATION
# ============================================================================
from jax import config
config.update('jax_enable_x64', True)
config.update("jax_debug_nans", True)

# ============================================================================
# CONTROL FLAGS
# ============================================================================
bug_print = True  # Enable debug printing
FIND_HESSIAN = False  # Compute Hessian during optimization
RAND_ENG_CHECK = True  # Enable energy randomization check

# ============================================================================
# PARTICLE PROPERTIES
# ============================================================================

# Particle geometry
NUM_PATCHES = 2
CENTER_RADIUS = 1.0
CENTER_MASS = 1.0
PATCH_MASS = 0.0

# Interaction parameters
ALPHA = 7  # Morse potential decay parameter
DT = 1e-3  # Time step
PATCH_SIZE = np.log(1 - np.sqrt(0.99)) / (-ALPHA) / 2
R_CUTOFF = PATCH_SIZE

# Temperature
kT = 1.0

# ============================================================================

# OPTIMIZATION PARAMETERS
# ============================================================================

# Number of steps for main simulation
NUM_STEPS = 40000  # Full simulation steps
# Number of steps for equilibration (run before main simulation)
EQUILIBRATION_STEPS = 2_000_000  # Set as needed for your system
NUM_STEPS_TO_OPT = 1000  # Steps during optimization
SQRT_NUM_STEPS_TO_OPT = int(np.sqrt(NUM_STEPS_TO_OPT))
QUICK_STEPS = int(np.sqrt((NUM_STEPS - NUM_STEPS_TO_OPT)))


# Particle count for optimization
NUM_PARTICLES = 16
NUM_PARTICLES_OPT = NUM_PARTICLES  # Alias for clarity

# Density and box size
DENSITY = 0.1
get_BOX_SIZE = lambda phi, N, rad: np.sqrt(N * np.pi * rad**2 / phi)
BOX_SIZE = get_BOX_SIZE(DENSITY, NUM_PARTICLES, CENTER_RADIUS)

# Optimization hyperparameters
BATCH_SIZE = 64
LEARNING_RATES = [0.5, 0.1, 0.05]  # Coarse, medium, fine
OPTIMIZER = 'adam'  # 'adam' or 'rms'
OPTIMIZER_TYPE = OPTIMIZER  # Alias
OPT_STEPS = 2
OPT_RUNS = 3

# Three-stage optimization schedule
OPT_STAGE_STEPS = [
    OPT_STEPS,   # Stage 1: Coarse (~17 steps)
    OPT_STEPS,  # Stage 2: Medium (~3 steps)
    OPT_STEPS,  # Stage 3: Fine (~3 steps)
]

# Energy constraints
MAX_ENERGY = 30.0
RAND_SEARCH_ITERATIONS = 1

# Gradient clipping
CLIP = 10000.0

# Logging
WRITE_EVERY = 10
LOG_STEPS = True

# ============================================================================
# YIELD SIMULATION PARAMETERS
# ============================================================================

# Particle count for yield measurement (scaled up from notebook's 20)
NUM_PARTICLES_YIELD = 1000

# Simulation length
NUM_STEPS_YIELD = 8000#0

# Box size for yield simulation
BOX_SIZE_YIELD = get_BOX_SIZE(DENSITY, NUM_PARTICLES_YIELD, CENTER_RADIUS)

# Polygon counting parameters
CLOSENESS_PENALTY = 0.01
CLOSENESS_PENALTY_NEIGHBORS = 1
CLUSTER_CHECK_TOLERANCE = 0.5
patch_allowance = PATCH_SIZE * 2 / 3.

# ============================================================================
# SHAPE CONFIGURATIONS
# ============================================================================

# Default shape for optimization (will be overridden by command-line args)
shape_ID = "Square"

# Shape-specific parameters
SHAPE_CONFIGS = {
    'square': {
        'n_vertices': 4,
        'ref_shape_size': 4,
        'default_opening_angle': 100.0,  # degrees (for reference only, NOT fixed!)
        'description': '2-patch square assembly'
    },
    'triangle': {
        'n_vertices': 3,
        'ref_shape_size': 3,
        'default_opening_angle': 120.0,  # degrees (for reference only, NOT fixed!)
        'description': '2-patch triangle assembly'
    }
}

# Helper function to get reference shape
def regular_ngon_reference(n, R=1.0, opening_angle_rad=None):
    """
    Construct a reference configuration for an n-gon of 2D patchy particles.

    Parameters
    ----------
    n : int
        Number of sides (must be >= 3)
    R : float
        Circumradius of the polygon
    opening_angle_rad : float, optional
        Opening angle in radians. If None, uses a default of ~1.75 rad (100°)

    Returns
    -------
    tuple : (positions, orientations, patch_angles)
        - positions: (n, 2) array of particle positions
        - orientations: (n,) array of particle orientations
        - patch_angles: (n, 2) array of patch angles relative to body frame
    """
    if n < 3:
        raise ValueError("Need at least a triangle (n >= 3).")

    if opening_angle_rad is None:
        opening_angle_rad = np.deg2rad(100.0)  # Default

    alpha = opening_angle_rad

    # Particle positions on a regular n-gon
    vertex_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.stack(
        [R * np.cos(vertex_angles), R * np.sin(vertex_angles)],
        axis=1
    )

    orientations = np.zeros(n)
    patch_angles = np.zeros((n, 2))

    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n

        v_prev = positions[i_prev] - positions[i]
        v_next = positions[i_next] - positions[i]

        beta_prev = np.arctan2(v_prev[1], v_prev[0])
        beta_next = np.arctan2(v_next[1], v_next[0])

        avg_vec = np.array([
            np.cos(beta_prev) + np.cos(beta_next),
            np.sin(beta_prev) + np.sin(beta_next),
        ])
        avg_beta = np.arctan2(avg_vec[1], avg_vec[0])

        theta_i = avg_beta
        orientations[i] = theta_i

        patch_to_prev = theta_i - alpha / 2.0
        patch_to_next = theta_i + alpha / 2.0

        patch_to_prev = (patch_to_prev + np.pi) % (2.0 * np.pi) - np.pi
        patch_to_next = (patch_to_next + np.pi) % (2.0 * np.pi) - np.pi

        patch_angles[i, 0] = patch_to_prev
        patch_angles[i, 1] = patch_to_next

    return positions, orientations, patch_angles

def get_shape(shape_id=None, opening_angle_rad=None):
    """
    Get reference shape geometry.

    Parameters
    ----------
    shape_id : str, optional
        Shape identifier ('square' or 'triangle'). If None, uses global shape_ID.
    opening_angle_rad : float, optional
        Opening angle in radians. If None, uses default value for the shape.

    Returns
    -------
    array : Reference shape positions (n, 2)
    """
    if shape_id is None:
        shape_id = shape_ID

    shape_id = shape_id.lower()

    if shape_id == 'triangle':
        n = 3
        # Radius such that edge length = 2*CENTER_RADIUS
        radius = 1.0 / np.sin(np.pi/3)
    elif shape_id == 'square':
        n = 4
        # Radius such that edge length = 2*CENTER_RADIUS
        radius = 1.0 / np.sin(np.pi/4)
    else:
        # Fallback: assume square
        n = 4
        radius = 1.0 / np.sin(np.pi/4)

    positions, _, _ = regular_ngon_reference(n, R=radius, opening_angle_rad=opening_angle_rad)

    return positions

# Initialize reference shape with default
REF_SHAPE = get_shape(shape_ID)
REF_SHAPE_SIZE = len(REF_SHAPE)

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_DIR_OPT = "optimal_params"
OUTPUT_DIR_YIELD = "yield_results"

# Job ID (will be set by scripts)
JOBID = "optimization_run"

# ============================================================================
# RANDOM SEEDS
# ============================================================================

KEY_PARAM = 0
KEY_PARAM_OPT = 0
KEY_PARAM_YIELD = 1273

# ============================================================================
# DERIVED PARAMETERS (computed from above)
# ============================================================================

# Create displacement and shift functions for periodic boundary conditions
displacement, shift = space.periodic(BOX_SIZE)

# Create random key
key = random.PRNGKey(KEY_PARAM)

# ============================================================================
# IMPORTANT NOTES
# ============================================================================

# NOTE 1: Opening angle optimization
# The opening angle between patches is OPTIMIZED, not fixed!
# - params[0] = theta_0 (fixed to 0)
# - params[1] = theta_1 (opening angle, OPTIMIZED via gradient descent)
# - The 'default_opening_angle' in SHAPE_CONFIGS is for reference geometry only

# NOTE 2: BOX_SIZE changes
# - BOX_SIZE is used during optimization (16 particles)
# - BOX_SIZE_YIELD is used during yield measurement (100+ particles)
# - Scripts must update BOX_SIZE dynamically when switching contexts

# NOTE 3: NUM_PARTICLES changes
# - NUM_PARTICLES = 16 for optimization
# - NUM_PARTICLES_YIELD = 100 for yield measurements
# - This affects BOX_SIZE calculation

print("Configuration loaded successfully!")
print(f"  Optimization: {NUM_PARTICLES} particles, box size {BOX_SIZE:.2f}")
print(f"  Yield measurement: {NUM_PARTICLES_YIELD} particles, box size {BOX_SIZE_YIELD:.2f}")

