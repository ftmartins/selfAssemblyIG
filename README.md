# Patchy Particle Optimization System

A modular Python system for optimizing patchy particle parameters for target shape assembly and measuring polygon yields through large-scale simulations.

## Overview

This system provides:
- **Optimization**: Gradient-based optimization of 2-patch particle parameters for square or triangle assembly
- **Yield Measurement**: Large-scale simulations to count polygon formation and calculate yields
- **Modular Design**: Clean separation between configuration, core modules, and executable scripts

**Key Feature**: The opening angle between patches is **optimized via gradient descent**, not fixed!

## Folder Structure

```
selfAssemblyPatchyParticles/
├── config_patchy_particle.py       # All global constants and parameters
├── optimize_patchy_particles.py     # Main optimization script
├── run_yield_simulation.py          # Yield calculation script
├── requirements.txt                 # Python dependencies
├── optimal_params/                  # Optimization results (auto-created)
├── modules/                         # Core functionality
│   ├── __init__.py
│   ├── utility_functions.py        # Simulation infrastructure
│   ├── evaluation_functions.py     # Loss and cluster detection
│   └── optimizer.py                # Optimization algorithms
└── README.md                        # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for JAX acceleration)

### Setup

1. **Clone or navigate to this directory**:
   ```bash
   cd selfAssemblyPatchyParticles
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies:
   - JAX & JAX-MD: Differentiable molecular dynamics
   - freud-analysis: Cluster detection
   - NumPy, SciPy, Matplotlib: Numerical computing and visualization

## Usage

### 1. Optimization

Optimize patchy particle parameters for a target shape (square or triangle):

```bash
# Square optimization
python optimize_patchy_particles.py --shape square

# Triangle optimization
python optimize_patchy_particles.py --shape triangle
```

**Command-line options**:
- `--shape`: Target shape (`square` or `triangle`) **[required]**
- `--max_energy`: Maximum bond energy (default: 30.0)
- `--batch_size`: Batch size for optimization (default: 64)
- `--opt_steps`: Steps per optimization stage (default: 100)
- `--output_dir`: Output directory (default: `optimal_params/`)

**What it does**:
1. **Random Search**: Initializes parameters via random search
2. **Three-Stage Optimization**:
   - Stage 1: Coarse search (LR=0.5, ~17 steps)
   - Stage 2: Medium refinement (LR=0.1, ~3 steps)
   - Stage 3: Fine refinement (LR=0.05, ~3 steps)
3. **Saves Results**:
   - `{shape}_params_{timestamp}.npz`: Binary parameter file
   - `{shape}_summary_{timestamp}.txt`: Human-readable summary

**Example output**:
```
optimal_params/
├── square_params_20251205_143022.npz
└── square_summary_20251205_143022.txt
```

### 2. Yield Simulation

Run large-scale simulation to measure polygon yields from optimized parameters:

```bash
# Basic usage
python run_yield_simulation.py --params optimal_params/square_params_20251205_143022.npz

# Custom particle count
python run_yield_simulation.py --params optimal_params/square_params_*.npz --num_particles 200

# Shorter simulation for testing
python run_yield_simulation.py --params optimal_params/triangle_params_*.npz --num_steps 10000
```

**Command-line options**:
- `--params`: Path to optimized parameters NPZ file **[required]**
- `--num_particles`: Number of particles (default: 100)
- `--num_steps`: Simulation steps (default: 40,000)
- `--output_dir`: Output directory (default: `yield_results/`)

**What it does**:
1. Loads optimized parameters from NPZ file
2. Runs long simulation with specified particle count
3. Uses freud clustering to detect polygons (triangles, squares, pentagons, hexagons)
4. Calculates yields as fraction of particles in each polygon type
5. Saves comprehensive results

**Example output**:
```
Polygon counts:
  Triangle  : 5
  Square    : 18
  Pentagon  : 1
  Hexagon   : 0
  Monomers  : 6

Yields (as fraction of particles):
  Triangle  : 0.1500 (15.00%)
  Square    : 0.7200 (72.00%)
  Pentagon  : 0.0500 (5.00%)
  Hexagon   : 0.0000 (0.00%)
  Monomers  : 0.0600 (6.00%)
```

## Configuration

Edit `config_patchy_particle.py` to customize:

### Particle Properties
```python
NUM_PATCHES = 2          # Number of patches per particle
CENTER_RADIUS = 1.0      # Particle radius
ALPHA = 7                # Morse potential decay
kT = 1.0                 # Temperature
```

### Optimization Parameters
```python
NUM_PARTICLES_OPT = 16   # Particles during optimization
BATCH_SIZE = 64          # Batch size
LEARNING_RATES = [0.5, 0.1, 0.05]  # Three-stage LRs
MAX_ENERGY = 30.0        # Maximum bond energy
```

### Yield Simulation Parameters
```python
NUM_PARTICLES_YIELD = 100  # Particles for yield measurement
NUM_STEPS_YIELD = 40_000   # Simulation length
DENSITY = 0.2              # System density
```

## Opening Angle Optimization

**CRITICAL**: The opening angle between patches is **OPTIMIZED**, not fixed!

- **Parameter structure**: `params = [theta_1, E_RR, E_RB, E_BB]`
  - `theta_1`: Opening angle (0 to π) - **OPTIMIZED**
  - `E_RR, E_RB, E_BB`: Bond interaction energies - **OPTIMIZED**

- **How it works**:
  - `theta_0` is fixed to 0 (reference patch)
  - `theta_1` is initialized randomly via random search
  - Gradient descent adjusts `theta_1` to minimize loss
  - Loss function naturally guides optimization toward optimal opening angle

- **Results**: The optimized opening angle will differ for squares vs. triangles based on geometric requirements for stable assembly

## Output File Formats

### Optimization Output (`optimal_params/`)

**NPZ file** (`{shape}_params_{timestamp}.npz`):
```python
{
    'shape': str,                    # Target shape
    'final_params': array,           # Optimized parameters
    'min_params_stage1/2/3': array,  # Best loss params per stage
    'cluster_params_stage1/2/3': array,  # Best cluster params per stage
    'num_patches': int,
    'max_energy': float,
    'batch_size': int,
    'learning_rates': list,
    'optimizer': str
}
```

**Text summary** (`{shape}_summary_{timestamp}.txt`):
- Target shape and timestamp
- Final optimized parameters
- Parameter breakdown (angles in degrees)
- Opening angle explicitly highlighted

### Yield Output (`yield_results/`)

**NPZ file** (`{shape}_yields_{timestamp}.npz`):
```python
{
    'shape': str,
    'yields': dict,              # {poly_type: yield_fraction}
    'polygon_counts': dict,      # {poly_type: count}
    'num_particles': int,
    'num_steps': int,
    'box_size': float,
    'optimized_params': array,
    'final_centers': array,      # Particle positions
    'final_orientations': array  # Particle orientations
}
```

**Text summary** (`{shape}_yield_summary_{timestamp}.txt`):
- Simulation parameters
- Polygon counts
- Yields as percentages

## Advanced Usage

### Custom Optimization Schedule

Modify `OPT_STAGE_STEPS` in `config_patchy_particle.py`:
```python
OPT_STAGE_STEPS = [
    OPT_STEPS // 6,   # Stage 1 steps
    OPT_STEPS // 30,  # Stage 2 steps
    OPT_STEPS // 30,  # Stage 3 steps
]
```

### Accessing Results Programmatically

```python
import numpy as np

# Load optimization results
data = np.load('optimal_params/square_params_20251205_143022.npz')
optimized_params = data['final_params']
opening_angle_deg = np.rad2deg(optimized_params[0])
bond_energies = optimized_params[1:]

# Load yield results
yields_data = np.load('yield_results/square_yields_20251205_143530.npz',
                      allow_pickle=True)
polygon_counts = yields_data['polygon_counts'].item()
yields = yields_data['yields'].item()
```

## Troubleshooting

### Import Errors
If you get module import errors, ensure you're running scripts from the `selfAssemblyPatchyParticles` directory:
```bash
cd selfAssemblyPatchyParticles
python optimize_patchy_particles.py --shape square
```

### GPU Memory Issues
If JAX runs out of GPU memory:
1. Reduce `BATCH_SIZE` in `config_patchy_particle.py`
2. Reduce `NUM_PARTICLES_YIELD` for yield simulations
3. Force CPU usage: `export JAX_PLATFORM_NAME=cpu`

### Slow Optimization
First step takes longer due to JIT compilation. Subsequent steps are much faster.
Expected times (on GPU):
- First optimization step: 30-60 seconds
- Subsequent steps: 5-10 seconds each

## Technical Details

### Three-Stage Optimization

The optimization uses a coarse-to-fine strategy:

1. **Stage 1 (Coarse)**: Large learning rate (0.5), ~17 steps
   - Explores parameter space broadly
   - Identifies promising regions

2. **Stage 2 (Medium)**: Medium learning rate (0.1), ~3 steps
   - Refines parameters from Stage 1
   - Uses `cluster_params` (best cluster formation) as starting point

3. **Stage 3 (Fine)**: Small learning rate (0.05), ~3 steps
   - Final refinement
   - Converges to optimal parameters

Between stages, we use `cluster_params` (parameters with highest cluster count) rather than `min_loss_params` because cluster formation better indicates successful assembly.

### Polygon Detection

Polygon counting uses a multi-step process:

1. **Patch Position Calculation**: Compute world-frame positions of all patches
2. **Freud Clustering**: Identify spatial clusters of patches within distance threshold
3. **Cluster Validation**: For each cluster:
   - Check size matches target polygon (3, 4, 5, or 6 particles)
   - Compare pairwise distance matrix to reference polygon
   - Accept if shape loss < threshold (default: 0.5)
4. **Yield Calculation**: `yield = (polygon_count × n_vertices) / total_particles`

## Citation

If you use this code in your research, please cite:
- JAX-MD: https://github.com/google/jax-md
- freud: https://freud.readthedocs.io

## License

This software is provided as-is for research purposes.

## Contact

For questions or issues, please contact the research group or open an issue in the project repository.

---

**Last Updated**: December 2025
