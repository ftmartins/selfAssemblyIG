"""
Configuration file for patchy particle optimization and yield simulations.

All global constants and parameters are defined here.
"""

import numpy as np
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
bug_print = False  # Enable debug printing
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

# Particle count for optimization
NUM_PARTICLES = 16
NUM_PARTICLES_OPT = NUM_PARTICLES  # Alias for clarity

# Density and box size
DENSITY = 0.2
get_BOX_SIZE = lambda phi, N, rad: np.sqrt(N * np.pi * rad**2 / phi)
BOX_SIZE = get_BOX_SIZE(DENSITY, NUM_PARTICLES, CENTER_RADIUS)

# Simulation length
NUM_STEPS = 40000  # Full simulation steps
NUM_STEPS_TO_OPT = 1000  # Steps during optimization
SQRT_NUM_STEPS_TO_OPT = int(np.sqrt(NUM_STEPS_TO_OPT))
QUICK_STEPS = int(np.sqrt((NUM_STEPS - NUM_STEPS_TO_OPT)))

# Optimization hyperparameters
BATCH_SIZE = 64
LEARNING_RATES = [0.5, 0.1, 0.05]  # Coarse, medium, fine
OPTIMIZER = 'adam'  # 'adam' or 'rms'
OPTIMIZER_TYPE = OPTIMIZER  # Alias
OPT_STEPS = 100
OPT_RUNS = 3

# Three-stage optimization schedule
OPT_STAGE_STEPS = [
    OPT_STEPS // 6,   # Stage 1: Coarse (~17 steps)
    OPT_STEPS // 30,  # Stage 2: Medium (~3 steps)
    OPT_STEPS // 30,  # Stage 3: Fine (~3 steps)
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
NUM_PARTICLES_YIELD = 100

# Simulation length
NUM_STEPS_YIELD = 40_000

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
def get_shape(shape_id=None):
    """
    Get reference shape geometry.

    Parameters
    ----------
    shape_id : str, optional
        Shape identifier ('square' or 'triangle'). If None, uses global shape_ID.

    Returns
    -------
    array : Reference shape positions
    """
    if shape_id is None:
        shape_id = shape_ID

    shape_id = shape_id.lower()

    if shape_id == 'triangle':
        ref_shape = 2 * CENTER_RADIUS * jnp.array([
            [0., 0.],
            [1., 0.],
            [0.5, jnp.sqrt(3) / 2.]
        ])
    else:  # square
        ref_shape = np.array([
            [0., 0.],
            [0., 2 * CENTER_RADIUS],
            [2 * CENTER_RADIUS, 0.],
            [2 * CENTER_RADIUS, 2 * CENTER_RADIUS]
        ])

    return ref_shape

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
KEY_PARAM_YIELD = 123

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
