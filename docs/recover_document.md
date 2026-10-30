# Recovery Document: Patchy Particle Ring Assembly Analysis

This document describes three self-contained implementations for studying the self-assembly
of 2-patch patchy particles into n-ring structures. Each section specifies inputs, outputs,
and figures in sufficient detail for an LLM to implement the piece from scratch.

The codebase context lives primarily in `run_yield_simulation_cluster.py`,
`config_patchy_particle.py`, `modules/params.py`, and `modules/utility_functions.py`.

---

## 1. Pairwise Energy Function for n-ring Structures in q-space

### Purpose

Implement a differentiable scalar energy function `total_energy(q, opening_angle, E_AB,
alpha_morse, ...)` that computes the total pairwise interaction energy of n 2-patch patchy
particles whose state is encoded as a flat vector `q`. This function is the core building
block for the L-BFGS minimization in Section 2.

### Particle model

Each particle has:
- A **center** at position `(x_i, y_i)` with a repulsive soft-sphere core of radius
  `CENTER_RADIUS = 1.0`.
- Two patches at distance `r_patch = CENTER_RADIUS = 1.0` from the center:
  - **Patch A** at body angle 'θ_i'
  - **Patch B** at body angle 'θ_i + opening_angle'
- `opening_angle` is the full angle between the two patches (in radians).

Patch positions in the lab frame:
```
pA_i = (x_i + r_patch * cos(θ_i),
         y_i + r_patch * sin(θ_i))

pB_i = (x_i + r_patch * cos(θ_i + opening_angle),
         y_i + r_patch * sin(θ_i + opening_angle))
```

### Inputs

| Variable          | Type/Shape       | Description                                                  |
|-------------------|------------------|--------------------------------------------------------------|
| `q`               | `(3*n,)` float   | Flat DoF vector: `[x1,y1,θ1, x2,y2,θ2, ..., xn,yn,θn]`     |
| `opening_angle`   | float (radians)  | Full angle between Patch A and Patch B                       |
| `E_AB`            | float            | Morse attraction strength for A–B patch pairs                |
| `alpha_morse`     | float            | Morse decay parameter (codebase default: `ALPHA = 7`)        |
| `E_AA`            | float (default 1.0) | Morse attraction strength for A–A patch pairs             |
| `E_BB`            | float (default 1.0) | Morse attraction strength for B–B patch pairs             |
| `r_patch`         | float (default 1.0) | Patch arm length (`CENTER_RADIUS` in codebase)            |
| `rep_A`           | float (default 10.0)| Soft-sphere repulsion prefactor (center–center only)      |
| `patch_sigma`     | float            | Morse interaction range; `PATCH_SIZE = log(1-sqrt(0.99))/(-ALPHA)/2` |

No periodic boundary conditions — this is a free-space (non-periodic) calculation.

### Energy terms

**1. Center–center soft-sphere repulsion** (for each pair `i < j`):
```
r_cc = ||center_i - center_j||
sigma_cc = 2 * CENTER_RADIUS   # = 2.0

V_rep(i,j) = rep_A * (1 - r_cc / sigma_cc)^2   if r_cc < sigma_cc
           = 0                                   otherwise
```

**2. Patch–patch Morse attraction** (for each pair `i < j`, for each patch combination):

The energy matrix (from `energy_matrix()` in `modules/utility_functions.py`) assigns:
- `pA_i`–`pA_j`: `E_AA`
- `pB_i`–`pB_j`: `E_BB`
- `pA_i`–`pB_j` and `pB_i`–`pA_j`: `E_AB`

For each patch pair with strength `eps` and distance `r`:
```
V_morse(r, eps) = eps * (1 - exp(-alpha_morse * r))^2 - eps   if r < r_cutoff
               = 0                                              otherwise
```
where `r_cutoff = patch_sigma` and the Morse equilibrium is at `r0 = 0` (sigma=0 in JAX-MD).

Note: the actual JAX-MD call in the cluster script uses `sigma=0.0` for Morse and a truncation
at `r_cutoff = PATCH_SIZE`. Replicate this truncation.

**Total energy:**
```
E_total(q) = sum_{i<j} [ V_rep(center_i, center_j)
                        + V_morse(pA_i, pA_j, E_AA)
                        + V_morse(pB_i, pB_j, E_BB)
                        + V_morse(pA_i, pB_j, E_AB)
                        + V_morse(pB_i, pA_j, E_AB) ]
```

### Outputs

| Name       | Type/Shape | Description                                       |
|------------|------------|---------------------------------------------------|
| `E_total`  | scalar     | Total potential energy of the n-particle system   |

### Implementation notes

- Use **JAX** (`jax.numpy`) so that `jax.grad` and `jax.hessian` work automatically.
- Implement as a pure function with no side effects.
- The function must be differentiable with respect to `q`, `E_AB`, and `alpha_morse`.
- Cross-reference: the JAX-MD equivalent in the cluster script builds `pair_energy_soft` and
  `pair_energy_morse` then wraps them via `rigid_body.point_energy`. Here we skip the rigid
  body layer and work directly with explicit patch positions derived from `q`.

---

## 2. L-BFGS Minimization Notebook for n-ring Structures

### Purpose

A Jupyter notebook that, for each n-ring type (n = 3, 4, 5, 6, 7, 8) and each point in a
2D parameter grid `(opening_angle, E_AB)`, minimizes the energy of the ring configuration
using L-BFGS, then visualizes stability and structural validity as phase diagram heatmaps.

### Parameter sweep

| Parameter       | Range                        | Notes                                      |
|-----------------|------------------------------|--------------------------------------------|
| `opening_angle` | 60° to 120°, ~13–25 values   | `np.linspace(60, 120, N)` converted to rad |
| `E_AB`          | 1 to 50, ~20–25 values       | `np.linspace(1, 50, M)` or log-spaced     |
| `E_AA`          | Fixed at 1.0                 |                                            |
| `E_BB`          | Fixed at 1.0                 |                                            |
| `alpha_morse`   | Fixed at 7 (codebase default)|                                            |
| n               | 3, 4, 5, 6, 7, 8             | Iterated over separately                   |

### Inputs

| Name                | Source                                     | Description                                  |
|---------------------|--------------------------------------------|----------------------------------------------|
| `total_energy`      | Section 1 implementation                  | Differentiable energy function               |
| `regular_ngon_reference(n, R, opening_angle_rad)` | `config_patchy_particle.py` | Initial ring geometry |
| `(opening_angle_grid, E_AB_grid)` | Defined within notebook   | 2D parameter arrays                          |

**Initial condition for minimization**: Use `regular_ngon_reference(n, R, opening_angle_rad)`
from `config_patchy_particle.py` to get `(positions, orientations, patch_angles)` for the
n-gon. Set the circumradius `R` such that the center-to-center edge length equals
approximately `2 * r_patch * sin(opening_angle / 2)` (so patches overlap at equilibrium).
Flatten to `q0 = [x1, y1, θ1, x2, y2, θ2, ...]`.

### Minimization procedure (per n, per grid point)

1. Build `q0` from `regular_ngon_reference(n, R, opening_angle_rad)`.
2. Define `f(q) = total_energy(q, opening_angle, E_AB, alpha_morse=7)`.
3. Run `scipy.optimize.minimize(f, q0, jac=jax.grad(f), method='L-BFGS-B',
   options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 10000})`.
4. Retrieve minimized state `q_min = result.x`.
5. Compute all diagnostics (see below).

### Minimization diagnostics

**Negative Hessian modes**:
```python
H = jax.hessian(total_energy)(q_min, ...)   # shape (3n, 3n)
eigenvalues = np.linalg.eigvalsh(H)
n_negative = np.sum(eigenvalues < -1e-6)    # exclude near-zero (rigid body modes)
```
Expected: a perfect ring has 3 zero modes (2 translations + 1 rotation); all others
should be positive for a stable minimum.

**Gradient magnitude**:
```python
g = jax.grad(total_energy)(q_min, ...)
grad_mag = np.linalg.norm(g)
```
Should be ≈ 0 at a true minimum.

**Validity** (structural check):
Going clockwise (i = 0, 1, ..., n−1, 0), for each consecutive particle pair (i, i+1 mod n):
```python
d_B_to_A = ||pB_i - pA_{i+1}||
valid_pair = d_B_to_A < 2 * r_patch
```
The ring is **valid** if all n such distances satisfy the condition. This checks that each
particle's B-patch points toward the next particle's A-patch, as required for a proper
A–B–A–B… ring topology.

### Outputs

For each n ∈ {3, 4, 5, 6, 7, 8}, save one `.npz` file, e.g. `minimized_n{n}_ring.npz`:

| Array name              | Shape                               | Description                                  |
|-------------------------|-------------------------------------|----------------------------------------------|
| `q_min_grid`            | `(N_angle, N_E_AB, 3*n)`            | Minimized coordinates for every grid point   |
| `n_negative_modes`      | `(N_angle, N_E_AB)`                 | Number of negative Hessian eigenvalues        |
| `grad_magnitude`        | `(N_angle, N_E_AB)`                 | ‖∇E‖ at minimum                              |
| `validity`              | `(N_angle, N_E_AB)`                 | 1 if ring is valid, 0 otherwise               |
| `opening_angles_deg`    | `(N_angle,)`                        | Opening angle sweep values (degrees)          |
| `E_AB_values`           | `(N_E_AB,)`                         | E_AB sweep values                             |
| `minimization_success`  | `(N_angle, N_E_AB)`                 | `result.success` from scipy                  |

### Figures

For each n-ring, produce three heatmaps with **opening angle (degrees)** on the x-axis and
**E_AB** on the y-axis:

1. **Negative Hessian modes heatmap**: color = `n_negative_modes`. Annotate with the count.
   Title: `"n={n} ring — Negative Hessian modes"`.

2. **Gradient magnitude heatmap**: color = `log10(grad_magnitude + 1e-14)` (log scale).
   Title: `"n={n} ring — ‖∇E‖ at minimum"`.

3. **Validity heatmap**: binary color (green = valid, red = invalid).
   Title: `"n={n} ring — Structural validity (B_i → A_{i+1} < 2·r_patch)"`.

Arrange these as a 3×1 column per n-ring (or a grid of subplots). Use `matplotlib` with
`imshow` or `seaborn.heatmap`. Annotate axes with actual parameter values, not grid indices.

---

## 3. Yield Phase Diagram Notebook

### Purpose

A Jupyter notebook that loads the minimized n-ring structures from Section 2 and plots
phase diagram heatmaps of the yield of each assembly type over the `(opening_angle, E_AB)`
parameter space. A relative yield diagram comparing triangles vs. squares is also produced.

### Inputs

| Name                       | Source                                | Description                                           |
|----------------------------|---------------------------------------|-------------------------------------------------------|
| `minimized_n{n}_ring.npz`  | Output of Section 2 notebook          | Minimized ring structures and their validity maps     |
| `yield_results_cluster/`   | Output of `run_yield_simulation_cluster.py` | MD simulation yield data per (alpha, E_AB) point |
| `(opening_angle_grid, E_AB_grid)` | Same grid as Section 2        | Must match the simulation parameter grid              |

Loading yield data: the cluster script saves results per job (indexed by SLURM task ID).
Each job corresponds to one `(alpha_deg, zeta)` pair from `YIELD_ALPHAS_DEG` ×
`YIELD_ZETAS` as defined in `config_patchy_particle.py`. The yield for each structure type
is the fraction of clusters of that size: `Y[structure] = count / total_clusters` (see
`compute_yields_from_frame()` in the cluster script). Load the per-realization yield arrays
and average over realizations and trajectory frames to get a single yield per grid point.

### Yield calculation

For each grid point `(opening_angle, E_AB)`, load or compute:
```
Y_triangle(opening_angle, E_AB)    # mean fraction of 3-clusters
Y_square(opening_angle, E_AB)      # mean fraction of 4-clusters
Y_pentagon(opening_angle, E_AB)    # mean fraction of 5-clusters
Y_hexagon(opening_angle, E_AB)     # mean fraction of 6-clusters
Y_heptagon(opening_angle, E_AB)    # mean fraction of 7-clusters
Y_octagon(opening_angle, E_AB)     # mean fraction of 8-clusters
```

**Relative yield** (triangle vs. square, ignoring other structures):
```
Y_tot = Y_triangle + Y_square          # normalization constant
Y_tri_norm = Y_triangle / Y_tot
Y_sq_norm  = Y_square  / Y_tot
relative_yield = Y_tri_norm - Y_sq_norm   # ranges in [-1, +1]
```
+1 means only triangles form; −1 means only squares form; 0 means equal.

### Outputs

No new data files required; figures are the main output.

### Figures

**Panel A — Per-structure yield heatmaps** (one per structure type):
For each n ∈ {3, 4, 5, 6, 7, 8}, a heatmap with:
- x-axis: opening angle (degrees), 60 to 120
- y-axis: E_AB, 1 to 50
- Color: mean yield Y for that structure (colormap: `viridis`, range [0, 1] or auto)
- Title: e.g. `"Yield — Triangle (n=3)"`
- Overlay the validity contour from Section 2 (dashed line where `validity == 1`).

Suggested layout: 2×3 grid of subplots (one per n-ring).

**Panel B — Relative yield heatmap** (triangle vs. square):
- x-axis: opening angle (degrees)
- y-axis: E_AB
- Color: `relative_yield = Y_tri_norm - Y_sq_norm` (diverging colormap, e.g. `RdBu`,
  centered at 0; red = triangle-dominated, blue = square-dominated)
- Title: `"Relative yield: (Y_tri − Y_sq) / (Y_tri + Y_sq)"`
- Annotate the transition line (relative_yield = 0) with a contour.
---

## Cross-section notes

- **Coordinate convention**: The canonical convention, as defined by `thetas_to_shape` in
  `modules/utility_functions.py` and used by the JAX-MD rigid-body energy, places
  **patch A at body angle `0`** and **patch B at body angle `α`** (the full opening angle).
  In the lab frame this gives:
  ```
  pA_i = center_i + r_patch * (cos(θ_i),          sin(θ_i))
  pB_i = center_i + r_patch * (cos(θ_i + α),      sin(θ_i + α))
  ```
  The ring topology is B→A directed (patch B of particle i bonds to patch A of particle i+1
  going clockwise).

  > **⚠ Convention inconsistency — must be fixed before implementing Sections 1–4.**
  > Several scripts still use the **old symmetric convention** `θ ± α/2` instead of `(θ, θ+α)`.
  > Every patch-position calculation must be audited and standardised to `(θ, θ+α)`.
  > Known locations of the old convention:
  >
  > | File | Lines | Code | Fix needed |
  > |------|-------|------|------------|
  > | `run_yield_simulation_cluster.py` | 134–135 | `angle_A = orientations - opening_angle/2` / `angle_B = orientations + opening_angle/2` in `detect_clusters` | Change to `angle_A = orientations`, `angle_B = orientations + opening_angle` |
  > | `run_yield_simulation.py` | 294–295 | `patch_to_prev = theta_i - alpha/2` / `patch_to_next = theta_i + alpha/2` in `regular_ngon_reference` copy | Change to `patch_A = theta_i`, `patch_B = theta_i + alpha` |
  > | `run_yield_simulation.py` | 518 | `patch_angles_local = np.array([-opening_angle/2, opening_angle/2])` | Change to `np.array([0.0, opening_angle])` |
  > | `config_patchy_particle.py` | 200–201 | `patch_to_prev = theta_i - alpha/2` / `patch_to_next = theta_i + alpha/2` in `regular_ngon_reference` | Same fix as above |
  >
  > Before implementing any section, search every `.py` file for `/ 2` combined with `opening_angle`,
  > `alpha`, `patch`, or `orient` to catch any remaining uses. The reference implementation is
  > `thetas_to_shape` in `modules/utility_functions.py` (lines 79–88).
- **Morse potential parameters** from `config_patchy_particle.py`:
  `ALPHA = 7`, `PATCH_SIZE = log(1 - sqrt(0.99)) / (-7) / 2`, `R_CUTOFF = PATCH_SIZE`.
- **Soft-sphere repulsion** acts only between centers (species 0–0 in JAX-MD), with
  `sigma = 2 * CENTER_RADIUS = 2.0` and `rep_A = 10.0` (default in `SimpleParams`).
- **Parameter grid alignment**: Sections 2 and 3 must use the same `(opening_angle, E_AB)`
  grid so that validity maps from Section 2 can be overlaid on yield maps from Section 3.
- **E_AA = E_BB = 1.0 fixed** throughout (as in the `generate_parameter_grid()` function in
  `modules/params.py`); only E_AB is swept.

---

## 4. Grand Canonical Ensemble Simulation Script

> **Performance requirement**: This script must be written for **maximum computational
> efficiency**. GCMC is inherently sequential in the accept/reject logic, so throughput
> depends critically on minimising Python overhead per move. The implementation should target
> GPU execution via JAX wherever possible:
>
> - All array operations (energy evaluation, patch-position computation, pairwise distances)
>   must be expressed as `jax.numpy` operations so they JIT-compile and dispatch to GPU.
> - The per-particle ΔU function must be `@jax.jit`-compiled and, where the active-particle
>   count is known at trace time, `jax.vmap`-vectorised over candidate particles.
> - The Python-level MC loop (accept/reject logic, mask updates) should call only JIT'd
>   functions; avoid any `.item()` / `np.asarray()` round-trips inside the hot loop.
> - Batch multiple independent MC chains (different random seeds / parameter points) in
>   parallel using `jax.vmap` over the chain axis, amortising JIT overhead across the batch.
> - Use `jax.lax.scan` or `jax.lax.while_loop` to push the inner accept/reject loop onto the
>   device entirely, eliminating Python-level step-by-step control flow at the cost of a
>   static step count per JIT'd chunk.
> - Pre-compute and cache the full `(N_max × N_max)` pairwise distance tables only when
>   needed; prefer incremental O(N) ΔU updates for single-particle moves.
> - Target NVIDIA A100 / H100 GPUs on the cluster; use `jax.devices()` to confirm GPU
>   availability at startup and raise an error if only CPU is found.

### Purpose

Create a new simulation script `run_yield_simulation_gcmc.py` based on
`run_yield_simulation_cluster.py` that runs in the **grand canonical (μVT) ensemble**:
temperature T, volume V (box size), and chemical potential μ are fixed while the particle
number N fluctuates via Monte Carlo insertion and deletion moves. This allows the simulation
to find the equilibrium N at each (μ, opening_angle, E_AB) point without the user having
to manually choose a particle count.

### Grand canonical ensemble — physics background

In the μVT ensemble:
- The acceptance probability for **inserting** a new particle at random position/orientation:
  ```
  P_acc(insert) = min(1,  V * z / (N + 1)  *  exp(-β ΔU_insert))
  ```
- The acceptance probability for **deleting** a randomly chosen particle i:
  ```
  P_acc(delete) = min(1,  N / (V * z)       *  exp(-β ΔU_delete))
  ```
  where `z = exp(β μ) / Λ²` is the activity (`Λ` is the thermal de Broglie wavelength;
  for 2D reduced-unit simulations set `Λ = 1`), `β = 1/kT`, and `ΔU` is the energy change
  caused by the insertion or deletion. Both follow from detailed balance with the canonical
  weight `exp(-β U) / Z_N`.

- **Displacement move** (translate + rotate a randomly chosen particle):
  ```
  P_acc(displace) = min(1, exp(-β ΔU_displace))
  ```
  Standard Metropolis criterion; no N-dependence.

- The new control parameter grid replaces fixed `NUM_PARTICLES` with a sweep over
  `μ` (or equivalently `log z`). Suggested range: `β μ ∈ [-10, 0]` (log-spaced in activity
  `z = exp(βμ)`).

### Strategy for variable N in JAX

JAX requires static array shapes. Use the **ghost-particle / buffer approach**:
1. Pre-allocate arrays for `N_max` particles (e.g. `N_max = 2000`).
2. Maintain a boolean mask `active: (N_max,)` — `True` for real particles, `False` for ghosts.
3. The energy function multiplies interactions by `active[i] * active[j]`, effectively zeroing
   out any pair involving a ghost particle.
4. **Insertion**: pick a random slot `k` where `active[k] == False`, assign random position
   and orientation, tentatively set `active[k] = True`, compute `ΔU`, accept/reject.
5. **Deletion**: pick a random index from the currently active particles, tentatively set
   `active[i] = False`, compute `ΔU`, accept/reject.
6. After each accepted insertion or deletion the energy function does not need to be
   recompiled — only the mask changes.
7. Track `N = active.sum()` as the observable.

### Script structure

The new script closely mirrors `run_yield_simulation_cluster.py`. The main differences are:

| Component | NVT cluster script | GCMC script |
|-----------|--------------------|-------------|
| Ensemble | NVT Nose-Hoover MD | GCMC (MC only, no MD integrator) |
| Fixed quantities | N, V, T | μ, V, T |
| Control parameter | `NUM_PARTICLES` | `mu` (chemical potential, in units of kT) |
| Move set | MD time steps | Displacement, insertion, deletion |
| Array shape | `(N, ...)` fixed | `(N_max, ...)` with `active` mask |
| JAX-MD usage | `simulate.nvt_nose_hoover`, `rigid_body.point_energy` | Energy function only (no integrator); MC loop in Python |
| Checkpointing | Checkpoint NVT state pytree | Checkpoint positions, orientations, active mask, step count |

**MC move frequencies** (suggested): 50% displacement, 25% insertion, 25% deletion.

**Step size tuning**: target 30–50% acceptance for displacement moves. Tune `max_dr` (max
translation) and `max_dtheta` (max rotation) during an initial burn-in.

### Inputs

| Variable          | Type           | Description                                                        |
|-------------------|----------------|--------------------------------------------------------------------|
| `job_id`          | int            | SLURM array index into `(YIELD_ALPHAS_DEG, MU_GRID)` parameter grid |
| `mu`              | float          | Chemical potential in units of kT (replaces `NUM_PARTICLES`)       |
| `opening_angle`   | float (rad)    | Same as NVT script                                                  |
| `E_AB`, `E_AA`, `E_BB` | float   | Same as NVT script                                                  |
| `alpha_morse`     | float          | Same as NVT script                                                  |
| `N_max`           | int            | Buffer size; set to 3× the expected ⟨N⟩ at highest μ             |
| `BOX_SIZE`        | float          | Same box as NVT script (or scaled to density of interest)           |
| `NUM_MC_STEPS`    | int            | Total MC sweeps (1 sweep = N attempted moves)                       |
| `EQUILIBRATION_SWEEPS` | int    | Sweeps before recording begins                                      |

Define a new config section in `config_patchy_particle.py`:
```python
MU_GRID = np.linspace(-10, 0, 20)   # β*μ values
NUM_MC_STEPS_GCMC = 500_000
EQUILIBRATION_SWEEPS_GCMC = 100_000
N_MAX_GCMC = 2000
```

### Energy function adaptation

The ghost-particle energy function wraps the existing pair energy:
```python
def masked_energy(positions, orientations, active, opening_angle, E_AB, ...):
    # positions: (N_max, 2), orientations: (N_max,), active: (N_max,) bool
    # Compute patch positions for all N_max particles (ghosts included)
    pA = positions + r_patch * jnp.stack([jnp.cos(orientations),
                                           jnp.sin(orientations)], axis=-1)
    pB = positions + r_patch * jnp.stack([jnp.cos(orientations + opening_angle),
                                           jnp.sin(orientations + opening_angle)], axis=-1)
    # Pairwise sum with mask
    # For each (i, j) pair: multiply by active[i] * active[j]
    # Use jnp.where or explicit masking
    ...
    return total_E
```
Note: patch positions use the canonical `(θ, θ+α)` convention (see cross-section note).

For efficiency, compute `ΔU` for a single particle (summing interactions with all other
active particles) rather than recomputing the full N²/2 sum. This makes insertion/deletion
O(N) instead of O(N²).

### Outputs

For each `(opening_angle, mu)` grid point, save one `.npz` file
`gcmc_alpha{alpha_deg:.1f}_mu{mu:.2f}.npz`:

| Array name          | Shape                             | Description                                      |
|---------------------|-----------------------------------|--------------------------------------------------|
| `N_traj`            | `(n_samples,)`                    | Number of active particles at each sample step    |
| `energy_traj`       | `(n_samples,)`                    | Total energy at each sample step                  |
| `yields_traj`       | `(n_samples, n_structures)`       | Yield of each structure at each sample step       |
| `accept_disp`       | scalar                            | Displacement acceptance rate                      |
| `accept_ins`        | scalar                            | Insertion acceptance rate                         |
| `accept_del`        | scalar                            | Deletion acceptance rate                          |
| `mu`                | scalar                            | Chemical potential used                           |
| `opening_angle_deg` | scalar                            | Opening angle in degrees                          |

Structure detection uses the same `detect_clusters` / `compute_yields_from_frame` logic as
the NVT script, updated to use the canonical `(θ, θ+α)` patch convention.

### Figures

**Panel A — ⟨N⟩ phase diagram**:
- x-axis: opening angle (degrees)
- y-axis: μ (in units of kT)
- Color: mean particle number ⟨N⟩ / V (number density)
- Title: `"Grand canonical — mean particle number density"`

**Panel B — Yield phase diagrams** (same layout as Section 3, Panel A but with μ on y-axis):
- One heatmap per structure type (triangle, square, …)
- x-axis: opening angle; y-axis: μ
- Color: mean yield of that structure

**Panel C — Relative yield** (same as Section 3, Panel B but with μ on y-axis):
- `relative_yield = (Y_tri_norm − Y_sq_norm)` with diverging colormap

**Panel D — N fluctuations**:
- x-axis: opening angle; y-axis: μ
- Color: standard deviation of N (σ_N), indicating proximity to a phase transition
