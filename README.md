# Patchy-particle self-assembly

Simulate the self-assembly of 2-patch patchy particles in the **grand-canonical
(µVT)** and **canonical (NVT)** ensembles, and analyze the resulting runs.

## Layout

Three code layers (see `docs/repository_organization.md` for the model and
`docs/migration_notes.md` for the old→new map):

```
base/                    shared primitives — importable with zero side effects
  config.py              physical constants (ALPHA, PATCH_RADIUS, MORSE_R_ONSET, MORSE_R_CUTOFF, CENTER_RADIUS, kT, …)
  jax_setup.py           configure_jax(x64=True, debug_nans=False)  — call explicitly
  geometry.py            thetas_to_shape, energy_matrix, patch_positions,
                         regular_ngon_reference, get_shape
  energy.py              free-space total_energy; periodic params-dict energies;
                         build_pair_energy_fn (JAX-MD soft-sphere + Morse)
  params.py              SimpleParams, generate_parameter_grid, make_gcmc_params
  grids.py               YIELD_* / MU_SCAN_* grids + job_id ↔ params mappings
  paths.py               repo-root-relative data/plot dirs, make_gcmc_run_dir
  initial_conditions.py  rsa_2d (one RSA), random_uniform_ic
  structure_detection.py bond-graph clustering + freud count_polygons
  io.py                  pickle-npy + rename_with_backup

simulation/
  src/
    gcmc.py              hybrid GCMC (µVT) engine: MCMCState, mc_sweep, run_gcmc
    canonical_md.py      NVT Nose-Hoover drivers: run_sim/my_sim (rigid body),
                         run_many_particle_md (chunked + checkpointed),
                         md_relax_system
    design.py            DesignConfig + make_loss_fns + optimize / random_search
  runners/               thin CLI entrypoints (each: sys.path + configure_jax)
    run_gcmc.py  scan_mu.py  run_yield.py  run_yield_cluster.py
    optimize_design.py  make_initial_condition.py  preview_grid.py
    slurm/              SLURM array wrappers

analysis/
  mu_scan.py             equilibration test, per-mu aggregation, mu* finder
  yields.py              (opening angle, selectivity) yield-grid assembly
  gcmc_runs.py           discover / parse / load run.npz directories
  plotting.py            all matplotlib figure functions
  aggregate_mu_scan.py   CLI wrapper
  notebooks/             one notebook per analysis theme

benchmarks/              timestep_comparison.py (MD dt convergence)
data/                    generated results (git-ignored)
plots/                   generated figures (git-ignored)
docs/                    design notes + migration map
```

`simulation/` and `analysis/` both depend only on `base/`, never on each other.

## Environment

```bash
conda env create -f environment.yml    # env: patchy_particles  (Python 3.11,
                                       # numpy/scipy/matplotlib, freud, jax, jax-md)
conda activate patchy_particles
```

`docs/selfAssembly.yml` is the pinned cluster lockfile (env `selfAssemblyIG`).

## Usage

Run everything from the repo root.

### Grand-canonical (µVT)

```bash
# find mu* for a target area fraction: a mu-scan then aggregate
sbatch simulation/runners/slurm/submit_mu_scan.sh
python analysis/aggregate_mu_scan.py --input_dir data/mu_scan --E_AB 5 --opening_angle 1.5708

# full run at a chosen mu
python simulation/runners/run_gcmc.py \
    --opening_angle 1.5708 --E_AB 5.0 --mu -4.0 --n_equil 20000 --n_prod 10000
# → data/gcmc_runs/oa90.00deg_EAB5.000_mu-4.0000_kT1.0_r000/run.npz

# phase-diagram sweep
MU=-4.0 sbatch simulation/runners/slurm/submit_gcmc.sh
```

### Canonical (NVT) yield

```bash
python simulation/runners/preview_grid.py            # job_id ↔ (alpha, zeta)
sbatch simulation/runners/slurm/submit_yield.sh      # SLURM array
```

### Patch-parameter design

```bash
python simulation/runners/optimize_design.py --shape square
# → data/design/square_params_YYYYMMDD_HHMMSS.npz
python simulation/runners/run_yield.py --params data/design/square_params_*.npz
```

### Analysis

Open the notebooks under `analysis/notebooks/` (run with the working directory
set to that folder — the first cell adds the repo root to `sys.path`).

## Notes

- `jax_debug_nans` is **off** by default now; pass
  `configure_jax(debug_nans=True)` when chasing a NaN.
- No global mutable state: particle count / box size are explicit arguments
  everywhere.  See `docs/migration_notes.md` for the full list of behavioural
  changes.

## Tests

```
python -m pytest            # whole suite
python -m pytest -m fast    # fast unit tests only
```

Requires `pytest` and `h5py` (listed in `environment.yml` / `requirements.txt`).
