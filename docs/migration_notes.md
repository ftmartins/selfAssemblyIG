# Migration notes — flat layout → base / simulation / analysis

Date: 2026-08-27.  Branch: `refactor/three-layer-layout`.

The repo was reorganised into three layers modelled on `repository_organization.md`:

| Layer | Dir | Role | Depends on |
|-------|-----|------|-----------|
| Base | `base/` | geometry, energy, params, grids, paths, ICs, structure detection — no import-time side effects | nothing in-repo |
| Simulation | `simulation/src` + `simulation/runners` | GCMC (µVT) + canonical (NVT) engines + design/optimization + CLI entrypoints | `base/` |
| Analysis | `analysis/` (+ `analysis/notebooks/`) | load runs, aggregate, plot | `base/` |

## Module / path map

| Old | New |
|-----|-----|
| `config_patchy_particle.py` (constants) | `base/config.py` |
| `config_patchy_particle.py` (`regular_ngon_reference`, `get_shape`, `SHAPE_CONFIGS`) | `base/geometry.py` |
| `config_patchy_particle.py` (`YIELD_ALPHAS_DEG`, `YIELD_ZETAS`, `NUM_REALIZATIONS`, `KEY_PARAM_*`) | `base/grids.py` |
| `config_patchy_particle.py` (`config.update(...)`) | `base/jax_setup.py::configure_jax()` — **now explicit**, `jax_debug_nans` defaults **off** |
| `config_patchy_particle.py` (design knobs: `BATCH_SIZE`, `OPTIMIZER`, `CLIP`, `MAX_ENERGY`, `OPT_*`, `JOBID`, `shape_ID`) | `simulation/src/design.py::DesignConfig` |
| `config_gcmc.py` (`PROJECT_ROOT`, dirs, `make_run_dir`) | `base/paths.py` (repo-root-relative; `make_gcmc_run_dir`) |
| `config_gcmc.py` (`DEFAULT_*`) | `simulation/src/gcmc.py::GcmcDefaults` |
| `config_gcmc.py` (`MU_SCAN_VALUES`) | `base/grids.py` |
| `energy_functions.py` | `base/energy.py` (`total_energy`) |
| `modules/params.py` | `base/params.py` |
| `modules/gcmc_hybrid.py` | `simulation/src/gcmc.py` (energy prims → `base/energy.py`; `make_gcmc_params` → `base/params.py`; `plot_configuration` → `analysis/plotting.py`) |
| `modules/utility_functions.py` (`thetas_to_shape`, `energy_matrix`, `body_to_plot`) | `base/geometry.py` |
| `modules/utility_functions.py` (`random_IC_nonoverlap`, RSA) | `base/initial_conditions.py::rsa_2d` (numpy; also replaces `generate_nonoverlap_ic`, `gcmc_hybrid.generate_ic`) |
| `modules/utility_functions.py` (`run_sim`, `my_sim`, `v_run_my_sim`, `random_IC`) | `simulation/src/canonical_md.py` |
| `modules/utility_functions.py` (`make_params`, `run_partial_sim`, `run_sim_and_get_positions`, `save_position`) | `simulation/src/design.py` |
| `modules/utility_functions.py` (`rename_file`) | `base/io.py::rename_with_backup` |
| `modules/evaluation_functions.py` (`sys_loss`, `avg_loss`, `loc_loss`) | `simulation/src/design.py::make_loss_fns` (factory — closes over `box_size`/`num_particles`/`ref_shape`) |
| `modules/evaluation_functions.py` (`get_clusters`, `make_cluster_list`) | `simulation/src/design.py` |
| `modules/optimizer.py` | `simulation/src/design.py` |
| `modules/relax_md.py` | `simulation/src/canonical_md.py::md_relax_system` |
| `run_yield_simulation_cluster.py` (`detect_clusters`, `classify_cluster`, `compute_yields_from_frame`) | `base/structure_detection.py` |
| `run_yield_simulation_cluster.py` (`run_many_particle_md`, checkpoint helpers) | `simulation/src/canonical_md.py` |
| `run_yield_simulation.py` (`count_polygons`) | `base/structure_detection.py::count_polygons` (freud, lazy import) |
| `run_gcmc_hybrid.py` | `simulation/runners/run_gcmc.py` |
| `mu_scan_gcmc.py` | `simulation/runners/scan_mu.py` |
| `run_yield_simulation.py` | `simulation/runners/run_yield.py` |
| `run_yield_simulation_cluster.py` | `simulation/runners/run_yield_cluster.py` |
| `optimize_patchy_particles.py` | `simulation/runners/optimize_design.py` |
| `generate_initial_condition.py` | `simulation/runners/make_initial_condition.py` |
| `generate_simulation_tasks.py` | `simulation/runners/preview_grid.py` |
| `aggregate_mu_scan.py` | compute → `analysis/mu_scan.py`, plot → `analysis/plotting.py`, CLI → `analysis/aggregate_mu_scan.py` |
| `debug_timestep_comparison.py` | `benchmarks/timestep_comparison.py` |
| `submit_gcmc_hybrid.sh` / `submit_mu_scan.sh` / `submit_yield_simulation.sh` / `SubmitPolygonYield.sh` | `simulation/runners/slurm/submit_{gcmc,mu_scan,yield,yield_params}.sh` |
| `modify_yield_for_seed.py`, `SubmitPolygonYield_old.sh`, `text.txt` | deleted |
| `*.ipynb` | `analysis/notebooks/` (import cells rewritten; `analyze_gcmc_runs.ipynb` → `gcmc_runs_analysis.ipynb`) |

## Data

| Old | New |
|-----|-----|
| `data/gcmc_runs/` | unchanged (matches `base.paths.GCMC_RUNS_DIR`) |
| `data/mu_scan/` | unchanged (matches `base.paths.MU_SCAN_DIR`) |
| `data/mu_scan_old/` | `data/_archive/mu_scan_old/` |
| `data/results/{mu_scan,gcmc_runs,analysis}/` | `data/_archive/results/` |
| new: `data/yield/`, `data/design/`, `data/initial_conditions/` | canonical-yield / design / IC outputs |
| root `*.png`, `orientation_sweep.gif` | `plots/legacy/` |

`data/` and `plots/` are now git-ignored (`.gitignore`).

## Behavioural changes to be aware of

- **`jax_debug_nans` is now OFF by default.** Pass `configure_jax(debug_nans=True)`
  to restore the old behaviour when hunting a NaN.
- **No global mutable state.** `NUM_PARTICLES` / `BOX_SIZE` are no longer module
  globals; particle count and box size are explicit function arguments
  everywhere. `run_sim` gained a `box_size` parameter (inserted after
  `center_radius`); `my_sim`/`v_run_my_sim` `static_argnums` updated to match.
- **RSA is now numpy** (`base.initial_conditions.rsa_2d`), not JAX `random`.
  Initial conditions are still reproducible from a seed, but the exact particle
  coordinates for a given seed differ from the pre-refactor code.
- **One `regular_ngon_reference`** (canonical `(θ, θ+α)` patch convention, radians
  API). The `beta_prev`-based copy from `run_yield_simulation.py` is gone;
  polygon-detection only used its vertex positions, which are convention-independent.
- `run_yield_cluster.py` sizes the box for 3000 particles regardless of
  `--num-particles` (default 1000), reproducing the low effective density of the
  original cluster script.
- The pre-refactor `gcmc_patchy_particles*.ipynb` notebooks inline an old copy of
  the GCMC engine; only their import cells were updated. Prefer
  `simulation.src.gcmc` for new work.

## Verify

```bash
python -m compileall base simulation analysis benchmarks
python -c "import base.config, base.geometry, base.energy, base.params, base.grids, base.paths, base.initial_conditions, base.structure_detection, base.io"   # no stdout
python -c "import simulation.src.gcmc, simulation.src.canonical_md, simulation.src.design, analysis.mu_scan, analysis.yields, analysis.plotting, analysis.gcmc_runs"
python simulation/runners/preview_grid.py
python simulation/runners/run_gcmc.py --opening_angle 1.5708 --E_AB 5 --mu -3 --box_area 400 --n_equil 20 --n_prod 20 --no_checkpoint --output /tmp/g.npz
python analysis/aggregate_mu_scan.py --input_dir data/mu_scan --E_AB 5 --opening_angle 1.5708 --no_plot
python simulation/runners/optimize_design.py --shape square --opt_steps 2 --batch_size 4
python simulation/runners/run_yield_cluster.py 0 --output-dir /tmp/yc --equilibration-steps 50 --num-steps 200
grep -rn "config_patchy_particle\|config_gcmc\|from modules\|import \*" --include='*.py' base simulation analysis benchmarks   # empty
```
