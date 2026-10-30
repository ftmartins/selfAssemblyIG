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
  structure_detection.py snapshot-yield detectors (shape-filtered) + freud count_polygons
  io.py                  pickle-npy + rename_with_backup
  trajectory.py          HDF5 trajectory store (writer/reader) + legacy loader

simulation/
  src/
    gcmc.py              hybrid GCMC (µVT) engine: MCMCState, mc_sweep, run_gcmc
    run.py               unified entry run_simulation(ensemble, path, ...)
    canonical_md.py      NVT Nose-Hoover drivers: run_sim/my_sim (rigid body),
                         run_many_particle_md (chunked + checkpointed),
                         md_relax_system
    design.py            DesignConfig + make_loss_fns + optimize / random_search
  runners/               thin CLI entrypoints (each: sys.path + configure_jax)
    run.py (--ensemble canonical|grand_canonical)  run_gcmc.py  scan_mu.py  run_yield.py  run_yield_cluster.py
    optimize_design.py  make_initial_condition.py  preview_grid.py
    slurm/              SLURM array wrappers

analysis/
  report.py              one command: yields + checks + plots -> report directory
  equilibrium.py         stationarity, autocorrelation, agreement checks (+ _plots)
  structure_plots.py     snapshot / yield histograms / grid, shared colour map
  movie.py               trajectory GIF
  bond_graph.py          ADR-0002 bond-graph clusters, particle/cluster yields with
                         block errors, seed combination (API in module docstring)
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
# → data/gcmc_runs/oa90.00deg_EAB5.000_mu-4.0000_kT1.0_r000/run.h5
# (same HDF5 schema as NVT; N varies per frame, acceptance rates are root
#  attributes; rerunning resumes the file; --no_checkpoint starts afresh)

# either ensemble through one command
python simulation/runners/run.py --ensemble grand_canonical \
    --opening_angle 1.5708 --E_AB 5.0 --mu -4.0 --output data/run_gc.h5
python simulation/runners/run.py --ensemble canonical \
    --opening_angle 1.5708 --E_AB 5.0 --box_area 100 --N 40 --output data/run_nvt.h5

# phase-diagram sweep
MU=-4.0 sbatch simulation/runners/slurm/submit_gcmc.sh
```

### Canonical (NVT) yield

```bash
python simulation/runners/preview_grid.py            # job_id ↔ (alpha, zeta)
sbatch simulation/runners/slurm/submit_yield.sh      # SLURM array
```

### Trajectory file (HDF5)

Both ensembles write one `.h5` file per run with the same schema
(`base/trajectory.py`).  Read it with `open_trajectory(path).frames(phase,
start, stop, step)`; legacy `.npz` / pickled `.npy` files open through the same
call.  Rerunning with `--resume` (canonical runners and `run_gcmc.py` resume by
default) continues from the file without duplicating frames.

Root attributes (a test checks this list against files from both ensembles):

- Both ensembles: `schema_version`, `software_version`, `precision`, `ensemble`, `engine`, `thermostat`, `seed`, `box_size`, `kT`, `opening_angle`, `E_AA`, `E_BB`, `E_AB`, `bond_specificity`, `rep_A`, `alpha`, `patch_radius`, `morse_r_onset`, `morse_r_cutoff`, `center_radius`, `num_equilibration_steps`, `num_production_steps`, `sample_interval`, `equilibration_sample_interval`, `initial_condition`
- Canonical only: `N`, `dt`
- Grand canonical only: `mu`, `N_init`, `f_disp`, `md_steps`, `md_dt`, `gamma`, `acc_insertion`, `acc_deletion`, `acc_displacement`

`precision` is the dtype of coordinates; energies are always float64.  In the
canonical ensemble the step counts and `sample_interval` are MD steps, in the
grand canonical ensemble sweeps.

Groups `equilibration` and `production` hold per-frame datasets `step`, `N`,
`offset`, `potential_energy`, `kinetic_energy`, and flat `positions` (sum N, 2)
/ `orientations` (sum N), created with the first frame.  Frame i is rows
`offset[i]:offset[i]+N[i]`: no padding, so grand-canonical frames with varying
N work.  Group `restart` holds the engine state (canonical: MD state;
grand canonical: MD state plus RNG state and acceptance counters as attributes)
and, when finished, phase `done`.

### Running either ensemble

```bash
python simulation/runners/run.py --ensemble canonical --opening_angle 1.5708 \
    --E_AB 5.0 --box_area 100 --N 40 --n_equil 10000 --n_prod 100000 --output run.h5
python simulation/runners/run.py --ensemble grand_canonical --opening_angle 1.5708 \
    --E_AB 5.0 --mu -4.0 --n_equil 20000 --n_prod 10000 --output run.h5
```

Shared options: `--sample_interval`, `--equilibration_sample_interval` (0 = no
equilibration frames), `--checkpoint_interval`, `--precision float32|float64`,
`--seed`, `--kT`, `--E_AA`, `--E_BB`, `--rep_A`, `--resume`, `--quiet`, and
`--initial_condition dispersed|preassembled` (random monomers, or complete rings
of the size matching the opening angle: the forward/backward equilibrium check).
`simulation/runners/run_gcmc.py` and `run_yield*.py` are wrappers around the
same engines.

### Analysis, equilibrium checks and plots

Definitions follow `CONTEXT.md` and ADR-0002: a *bond* is an A-B patch pair
closer than the bond distance (default the patch radius), a *cluster* is a
connected component, a *ring* is a cluster in which every particle has both
patches bonded, anything else with two or more particles is an *open chain*.
*Particle yield* is the fraction of particles in a structure, *cluster yield*
the fraction of clusters.  Both are averaged over production frames with
block-averaged errors (seed-to-seed errors when combined across seeds).

```bash
# everything for one file: yields, equilibrium checks, plots, GIF
python -m analysis.report run.h5 [--outdir D] [--seeds s2.h5 s3.h5] \
    [--preassembled pre.h5] [--bond_distance d] [--no_gif]

# individual pieces
python -m analysis.structure_plots snapshot run.h5 --frame -1 --outdir D [--unwrap]
python -m analysis.structure_plots histogram run.h5 --outdir D
python -m analysis.structure_plots grid a.h5 b.h5 c.h5 --outdir D
python -m analysis.movie run.h5 out.gif --stride 5 --fps 10 [--unwrap] [--no_clusters]
```

The report directory holds `job_<stem>_yields.npz` (read by
`analysis.yields.load_yield_grid` for the phase diagram), `yields.json`,
`equilibrium.json` (stationarity, autocorrelation time, seed agreement,
forward/backward agreement, bond-distance sensitivity, each with pass/fail),
and PNG/GIF figures; see `analysis/report.py`.  In Python:
`analysis.bond_graph.yields`, `analysis.equilibrium.run_checks`,
`analysis.structure_plots`, `analysis.equilibrium_plots`.  Plotting needs
matplotlib; the GIF needs Pillow.  All run without a display.

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
